#!/usr/bin/env python3
"""
LLM Serving Lab — unified benchmark runner.

Measures all engines under the same conditions, assuming each engine
(vLLM/SGLang/TGI/TensorRT-LLM) exposes an OpenAI-compatible
/v1/chat/completions streaming endpoint.

Collects: TTFT, end-to-end latency, output tokens/sec, throughput (req/s),
and DCGM GPU utilization/memory/power from Prometheus when available.

usage:
  uv run --project scripts/bench python scripts/bench/run_benchmark.py \
      --engine vllm --base-url http://localhost:8000 \
      --model Qwen/Qwen3-8B-FP8 --concurrency 1 4 8 16 \
      --input-tokens 512 --output-tokens 256 --num-prompts 200
"""
import argparse, asyncio, json, os, statistics, subprocess, sys, time
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen

import aiohttp
import requests

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


PROMPT_FILLER = "Explain the concept of distributed systems consistency models in detail. " \
                "Cover linearizability, sequential consistency, and eventual consistency. "

PROMPT_VARIANTS = {
    "decode_heavy": "Write a detailed operational checklist for debugging an LLM serving outage. ",
    "prefill_heavy": PROMPT_FILLER,
    "short_chat": "Answer in three concise sentences. Compare throughput and latency for an LLM server. ",
    "structured_json": "Return JSON with keys engine, bottleneck, and mitigation for LLM serving. ",
    "tool_call": "Decide whether a weather tool should be called for Seoul and explain why. ",
}


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def init_openlit_observability(service_name: str):
    if not env_flag("OPENLIT_ENABLED"):
        return {"enabled": False}

    endpoint = os.environ.get("OPENLIT_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    application_name = os.environ.get("OPENLIT_APPLICATION_NAME", "llm-serving-lab")
    environment = os.environ.get("LAB_PROFILE", "default")
    capture_content = env_flag("OPENLIT_CAPTURE_MESSAGE_CONTENT", default=False)

    try:
        import openlit

        openlit.init(
            application_name=application_name,
            service_name=service_name,
            environment=environment,
            otlp_endpoint=endpoint,
            capture_message_content=capture_content,
        )
        return {
            "enabled": True,
            "backend": "openlit",
            "otlp_endpoint": endpoint,
            "application_name": application_name,
            "service_name": service_name,
            "capture_message_content": capture_content,
        }
    except Exception as e:
        return {"enabled": False, "backend": "openlit", "error": str(e)[:200]}


def make_prompt(approx_tokens: int, tokenizer=None, workload="prefill_heavy", prompt_id=None, prompt_mode="repeated"):
    filler = PROMPT_VARIANTS.get(workload, PROMPT_FILLER)
    suffix = "" if prompt_mode == "repeated" else f" Unique request id: {prompt_id}."
    if tokenizer is not None:
        base_ids = tokenizer.encode(filler, add_special_tokens=False)
        if base_ids:
            repeats = (approx_tokens // len(base_ids)) + 2
            token_ids = (base_ids * repeats)[:approx_tokens]
            return tokenizer.decode(token_ids, skip_special_tokens=True) + suffix, len(token_ids), "tokenizer"

    # Approximate token count, roughly assuming 0.75 words/token.
    words_needed = int(approx_tokens * 0.75)
    base = filler.split()
    out = []
    while len(out) < words_needed:
        out.extend(base)
    prompt = " ".join(out[:words_needed]) + suffix
    return prompt, estimate_tokens(prompt), "word_estimate"


def load_tokenizer(model):
    if AutoTokenizer is None:
        return None, "transformers is not installed"
    try:
        return AutoTokenizer.from_pretrained(model, trust_remote_code=True), None
    except Exception as e:
        return None, str(e)[:200]


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text.split()) / 0.75))


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def prometheus_query_range(prometheus_url, query, start, end, step="5s"):
    params = urlencode({"query": query, "start": start, "end": end, "step": step})
    url = f"{prometheus_url.rstrip('/')}/api/v1/query_range?{params}"
    with urlopen(url, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if payload.get("status") != "success":
        return []
    values = []
    for series in payload.get("data", {}).get("result", []):
        for _, value in series.get("values", []):
            try:
                values.append(float(value))
            except (TypeError, ValueError):
                pass
    return values


def summarize_values(values):
    if not values:
        return None
    return {
        "avg": round(statistics.mean(values), 3),
        "max": round(max(values), 3),
        "samples": len(values),
    }


def collect_gpu_metrics(start, end, agg_tokens_per_s):
    prometheus_url = os.environ.get("PROMETHEUS_URL", "http://127.0.0.1:9090")
    queries = {
        "gpu_util_percent": "DCGM_FI_DEV_GPU_UTIL",
        "vram_used_mib": "DCGM_FI_DEV_FB_USED",
        "power_watts": "DCGM_FI_DEV_POWER_USAGE",
    }
    try:
        metrics = {
            name: summarize_values(prometheus_query_range(prometheus_url, query, start, end))
            for name, query in queries.items()
        }
    except Exception as e:
        return {"available": False, "source": prometheus_url, "error": str(e)[:160]}

    avg_power = (metrics.get("power_watts") or {}).get("avg")
    tokens_per_joule = round(agg_tokens_per_s / avg_power, 4) if avg_power else None
    return {
        "available": any(value for value in metrics.values()),
        "source": prometheus_url,
        **metrics,
        "tokens_per_joule": tokens_per_joule,
    }


def wait_for_server(base_url, timeout_s):
    url = base_url.rstrip("/")
    deadline = time.time() + timeout_s
    last_error = "not checked"
    while time.time() < deadline:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}"
        except Exception as e:
            last_error = str(e)[:160]
        time.sleep(5)
    raise SystemExit(f"[bench] server is not healthy after {timeout_s}s: {last_error}")


def resolve_model(base_url, requested_model):
    if requested_model:
        return requested_model, "argument_or_env"
    url = base_url.rstrip("/")
    try:
        response = requests.get(f"{url}/v1/models", timeout=30)
        response.raise_for_status()
        models = response.json().get("data") or []
        if models and models[0].get("id"):
            return models[0]["id"], "server_models"
    except Exception as e:
        raise SystemExit(f"[bench] MODEL_ID is not set and /v1/models could not be read: {str(e)[:160]}")
    raise SystemExit("[bench] MODEL_ID is not set and /v1/models returned no model ids")


def count_output_tokens(text, usage_completion_tokens, tokenizer):
    if usage_completion_tokens is not None:
        return usage_completion_tokens, "stream_usage"
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False)), "tokenizer"
    return estimate_tokens(text), "word_estimate"


async def one_request(session, base_url, model, prompt, max_tokens, tokenizer):
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft = None
    content_parts = []
    usage_completion_tokens = None
    parse_errors = 0
    try:
        for include_usage in (True, False):
            request_payload = dict(payload)
            if include_usage:
                request_payload["stream_options"] = {"include_usage": True}
            async with session.post(url, json=request_payload) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    if include_usage and "stream_options" in body:
                        continue
                    return {"ok": False, "error": f"HTTP {resp.status}: {body[:200]}"}
                async for raw in resp.content:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        usage = chunk.get("usage")
                        if usage and usage.get("completion_tokens") is not None:
                            usage_completion_tokens = usage["completion_tokens"]
                        choices = chunk.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        tool_calls = delta.get("tool_calls")
                        if content:
                            if ttft is None:
                                ttft = time.perf_counter() - t0
                            content_parts.append(content)
                        elif tool_calls and ttft is None:
                            ttft = time.perf_counter() - t0
                    except Exception:
                        parse_errors += 1
                break
    except Exception as e:
        return {"ok": False, "error": str(e)}
    e2e = time.perf_counter() - t0
    if parse_errors:
        return {"ok": False, "error": f"SSE parse errors: {parse_errors}"}
    output_text = "".join(content_parts)
    out_tokens, token_count_source = count_output_tokens(output_text, usage_completion_tokens, tokenizer)
    if not output_text and out_tokens == 0:
        return {"ok": False, "error": "empty streamed response"}
    return {
        "ok": True,
        "ttft": ttft if ttft is not None else e2e,
        "e2e": e2e,
        "out_tokens": out_tokens,
        "tps": out_tokens / e2e if e2e > 0 else 0.0,
        "token_count_source": token_count_source,
    }


async def run_level(base_url, model, concurrency, num_prompts, input_tokens, output_tokens, tokenizer, workload, prompt_mode):
    prompt, actual_input_tokens, input_token_count_source = make_prompt(input_tokens, tokenizer, workload)
    sem = asyncio.Semaphore(concurrency)
    results = []
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=600)
    wall0 = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def worker(prompt_id):
            async with sem:
                request_prompt = prompt
                if prompt_mode != "repeated":
                    request_prompt, _, _ = make_prompt(input_tokens, tokenizer, workload, prompt_id, prompt_mode)
                return await one_request(session, base_url, model, request_prompt, output_tokens, tokenizer)
        tasks = [asyncio.create_task(worker(i)) for i in range(num_prompts)]
        for fut in asyncio.as_completed(tasks):
            results.append(await fut)
    wall = time.perf_counter() - wall0

    ok = [r for r in results if r.get("ok")]
    fail = len(results) - len(ok)
    sample_errors = [r.get("error") for r in results if not r.get("ok") and r.get("error")]
    if not ok:
        return {
            "concurrency": concurrency,
            "failed": fail,
            "ok": 0,
            "sample_errors": sample_errors[:3],
            "actual_input_tokens": actual_input_tokens,
            "input_token_count_source": input_token_count_source,
        }

    ttfts = sorted(r["ttft"] for r in ok)
    e2es = sorted(r["e2e"] for r in ok)
    total_tokens = sum(r["out_tokens"] for r in ok)
    token_count_sources = sorted({r.get("token_count_source", "unknown") for r in ok})

    def pct(xs, p):
        if not xs:
            return 0.0
        i = min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))
        return xs[i]

    agg_tokens_per_s = round(total_tokens / wall, 1)
    return {
        "concurrency": concurrency,
        "ok": len(ok),
        "failed": fail,
        "sample_errors": sample_errors[:3],
        "actual_input_tokens": actual_input_tokens,
        "input_token_count_source": input_token_count_source,
        "ttft_ms_p50": round(pct(ttfts, 50) * 1000, 1),
        "ttft_ms_p95": round(pct(ttfts, 95) * 1000, 1),
        "e2e_s_p50": round(pct(e2es, 50), 3),
        "e2e_s_p95": round(pct(e2es, 95), 3),
        "mean_tps_per_req": round(statistics.mean(r["tps"] for r in ok), 1),
        "agg_tokens_per_s": agg_tokens_per_s,
        "throughput_req_s": round(len(ok) / wall, 2),
        "token_count_sources": token_count_sources,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default=os.environ.get("MODEL_ID"))
    ap.add_argument("--concurrency", nargs="+", type=int,
                    default=[int(x) for x in os.environ.get("BENCH_CONCURRENCY", "1 4 8 16").split()])
    ap.add_argument("--input-tokens", type=int, default=int(os.environ.get("BENCH_INPUT_TOKENS", 512)))
    ap.add_argument("--output-tokens", type=int, default=int(os.environ.get("BENCH_OUTPUT_TOKENS", 256)))
    ap.add_argument("--num-prompts", type=int, default=int(os.environ.get("BENCH_NUM_PROMPTS", 200)))
    ap.add_argument("--workload", choices=sorted(PROMPT_VARIANTS), default=os.environ.get("BENCH_WORKLOAD", "prefill_heavy"))
    ap.add_argument("--prompt-mode", choices=["repeated", "randomized"], default=os.environ.get("BENCH_PROMPT_MODE", "repeated"))
    ap.add_argument("--wait-timeout", type=int, default=int(os.environ.get("BENCH_WAIT_TIMEOUT", 900)))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    wait_for_server(args.base_url, args.wait_timeout)
    args.model, model_source = resolve_model(args.base_url, args.model)

    observability = init_openlit_observability("bench-runner")
    if observability.get("enabled"):
        print(f"[observability] openlit={observability['otlp_endpoint']}")
    elif observability.get("error"):
        print(f"[observability] openlit_error={observability['error']}")

    tokenizer, tokenizer_error = load_tokenizer(args.model)
    token_count_source = "tokenizer" if tokenizer is not None else "stream_usage_or_word_estimate"
    print(f"[bench] engine={args.engine} model={args.model} url={args.base_url}")
    print(f"[bench] model_source={model_source}")
    print(f"[bench] token_count_fallback={token_count_source}")
    if tokenizer_error:
        print(f"[bench] tokenizer_warning={tokenizer_error}")
    levels = []
    for c in args.concurrency:
        print(f"  -> concurrency={c} ...", flush=True)
        level_start = time.time()
        res = await run_level(args.base_url, args.model, c, args.num_prompts,
                              args.input_tokens, args.output_tokens, tokenizer,
                              args.workload, args.prompt_mode)
        level_end = time.time()
        if res.get("ok"):
            res["gpu_metrics"] = collect_gpu_metrics(level_start, level_end, res.get("agg_tokens_per_s", 0.0))
        print(f"     {json.dumps(res)}")
        levels.append(res)

    report = {
        "engine": args.engine,
        "model": args.model,
        "profile": os.environ.get("LAB_PROFILE", "unknown"),
        "gpu": os.environ.get("GPU_NAME", "unknown"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "command": sys.argv,
        "observability": observability,
        "images": {
            "vllm": os.environ.get("VLLM_IMAGE"),
            "sglang": os.environ.get("SGLANG_IMAGE"),
            "tgi": os.environ.get("TGI_IMAGE"),
            "tensorrt_llm": os.environ.get("TENSORRT_LLM_IMAGE"),
        },
        "config": {
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "num_prompts": args.num_prompts,
            "workload": args.workload,
            "prompt_mode": args.prompt_mode,
            "model_source": model_source,
            "wait_timeout_s": args.wait_timeout,
            "token_count_fallback": token_count_source,
            "tokenizer_error": tokenizer_error,
        },
        "levels": levels,
    }
    out = args.out or f"results/{args.engine}_{os.environ.get('LAB_PROFILE','x')}.json"
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[bench] saved -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
