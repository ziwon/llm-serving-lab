#!/usr/bin/env python3
"""
LLM Serving Lab — unified benchmark runner.

Measures all engines under the same conditions, assuming each engine
(vLLM/SGLang/TGI/TensorRT-LLM) exposes an OpenAI-compatible
/v1/chat/completions streaming endpoint.

Collects: TTFT, end-to-end latency, output tokens/sec, throughput (req/s).
GPU memory/util/power should be collected separately from the monitoring stack (DCGM).

usage:
  uv run --project scripts/bench python scripts/bench/run_benchmark.py \
      --engine vllm --base-url http://localhost:8000 \
      --model Qwen/Qwen3-8B-FP8 --concurrency 1 4 8 16 \
      --input-tokens 512 --output-tokens 256 --num-prompts 200
"""
import argparse, asyncio, json, os, statistics, time
from datetime import datetime, timezone

import aiohttp

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


PROMPT_FILLER = "Explain the concept of distributed systems consistency models in detail. " \
                "Cover linearizability, sequential consistency, and eventual consistency. "


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


def make_prompt(approx_tokens: int, tokenizer=None):
    if tokenizer is not None:
        base_ids = tokenizer.encode(PROMPT_FILLER, add_special_tokens=False)
        if base_ids:
            repeats = (approx_tokens // len(base_ids)) + 2
            token_ids = (base_ids * repeats)[:approx_tokens]
            return tokenizer.decode(token_ids, skip_special_tokens=True), len(token_ids), "tokenizer"

    # Approximate token count, roughly assuming 0.75 words/token.
    words_needed = int(approx_tokens * 0.75)
    base = PROMPT_FILLER.split()
    out = []
    while len(out) < words_needed:
        out.extend(base)
    prompt = " ".join(out[:words_needed])
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


async def run_level(base_url, model, concurrency, num_prompts, input_tokens, output_tokens, tokenizer):
    prompt, actual_input_tokens, input_token_count_source = make_prompt(input_tokens, tokenizer)
    sem = asyncio.Semaphore(concurrency)
    results = []
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    timeout = aiohttp.ClientTimeout(total=600)
    wall0 = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        async def worker():
            async with sem:
                return await one_request(session, base_url, model, prompt, output_tokens, tokenizer)
        tasks = [asyncio.create_task(worker()) for _ in range(num_prompts)]
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
        "agg_tokens_per_s": round(total_tokens / wall, 1),
        "throughput_req_s": round(len(ok) / wall, 2),
        "token_count_sources": token_count_sources,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default=os.environ.get("MODEL_ID", "Qwen/Qwen3-8B-FP8"))
    ap.add_argument("--concurrency", nargs="+", type=int,
                    default=[int(x) for x in os.environ.get("BENCH_CONCURRENCY", "1 4 8 16").split()])
    ap.add_argument("--input-tokens", type=int, default=int(os.environ.get("BENCH_INPUT_TOKENS", 512)))
    ap.add_argument("--output-tokens", type=int, default=int(os.environ.get("BENCH_OUTPUT_TOKENS", 256)))
    ap.add_argument("--num-prompts", type=int, default=int(os.environ.get("BENCH_NUM_PROMPTS", 200)))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    observability = init_openlit_observability("bench-runner")
    if observability.get("enabled"):
        print(f"[observability] openlit={observability['otlp_endpoint']}")
    elif observability.get("error"):
        print(f"[observability] openlit_error={observability['error']}")

    tokenizer, tokenizer_error = load_tokenizer(args.model)
    token_count_source = "tokenizer" if tokenizer is not None else "stream_usage_or_word_estimate"
    print(f"[bench] engine={args.engine} model={args.model} url={args.base_url}")
    print(f"[bench] token_count_fallback={token_count_source}")
    if tokenizer_error:
        print(f"[bench] tokenizer_warning={tokenizer_error}")
    levels = []
    for c in args.concurrency:
        print(f"  -> concurrency={c} ...", flush=True)
        res = await run_level(args.base_url, args.model, c, args.num_prompts,
                              args.input_tokens, args.output_tokens, tokenizer)
        print(f"     {json.dumps(res)}")
        levels.append(res)

    report = {
        "engine": args.engine,
        "model": args.model,
        "profile": os.environ.get("LAB_PROFILE", "unknown"),
        "gpu": os.environ.get("GPU_NAME", "unknown"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
