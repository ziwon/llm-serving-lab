#!/usr/bin/env python3
"""
Phase 2 — Advanced Features validation script.
Probes structured output / JSON mode / tool calling / streaming on each engine.
Support is version-dependent, so record the results directly in benchmark/<engine>.md.

usage:
  uv run --project scripts/bench python scripts/bench/validate_features.py \
      --base-url http://localhost:8000 --model Qwen/Qwen3-8B-FP8
"""
import argparse, concurrent.futures, json, os, time, requests


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def init_openlit_observability():
    if not env_flag("OPENLIT_ENABLED"):
        return
    try:
        import openlit

        openlit.init(
            application_name=os.environ.get("OPENLIT_APPLICATION_NAME", "llm-serving-lab"),
            service_name="feature-validator",
            environment=os.environ.get("LAB_PROFILE", "default"),
            otlp_endpoint=os.environ.get("OPENLIT_OTLP_ENDPOINT", "http://127.0.0.1:4318"),
            capture_message_content=env_flag("OPENLIT_CAPTURE_MESSAGE_CONTENT", default=False),
        )
        print(f"[observability] openlit={os.environ.get('OPENLIT_OTLP_ENDPOINT', 'http://127.0.0.1:4318')}")
    except Exception as e:
        print(f"[observability] openlit_error={str(e)[:200]}")


def wait_for_server(url: str, timeout_s: int) -> None:
    deadline = time.time() + timeout_s
    last_error = "not checked"
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                return
            last_error = f"HTTP {r.status_code}"
        except Exception as e:
            last_error = str(e)[:120]
        time.sleep(5)
    raise SystemExit(f"[features] server is not healthy after {timeout_s}s: {last_error}")


def check_streaming(url, model):
    try:
        r = requests.post(f"{url}/v1/chat/completions", json={
            "model": model, "messages": [{"role": "user", "content": "count to 3"}],
            "max_tokens": 32, "stream": True}, stream=True, timeout=120)
        return r.status_code == 200 and any(b for b in r.iter_lines())
    except Exception:
        return False


def check_openai_models(url):
    try:
        r = requests.get(f"{url}/v1/models", timeout=30)
        return r.status_code == 200, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)[:80]


def check_json_mode(url, model):
    try:
        r = requests.post(f"{url}/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Return a JSON object with key 'city' = 'Seoul'."}],
            "max_tokens": 64, "response_format": {"type": "json_object"}}, timeout=120)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        content = r.json()["choices"][0]["message"]["content"]
        json.loads(content)
        return True, content[:80]
    except Exception as e:
        return False, str(e)[:80]


def check_tool_calling(url, model):
    tools = [{"type": "function", "function": {
        "name": "get_weather",
        "description": "get weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]}}}]
    try:
        r = requests.post(f"{url}/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "What's the weather in Seoul?"}],
            "tools": tools, "tool_choice": "required", "max_tokens": 128}, timeout=120)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code}"
        msg = r.json()["choices"][0]["message"]
        return bool(msg.get("tool_calls")), json.dumps(msg.get("tool_calls"))[:80]
    except Exception as e:
        return False, str(e)[:80]


def one_short_completion(url, model, prompt):
    started = time.perf_counter()
    r = requests.post(f"{url}/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 32,
        "temperature": 0.0}, timeout=120)
    elapsed = time.perf_counter() - started
    return r.status_code, elapsed


def check_concurrent_requests(url, model):
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futs = [
                executor.submit(one_short_completion, url, model, f"reply with the number {i}")
                for i in range(4)
            ]
            results = [f.result() for f in futs]
        ok = all(status == 200 for status, _ in results)
        max_latency = max(elapsed for _, elapsed in results)
        return ok, f"max_latency_s={max_latency:.2f}"
    except Exception as e:
        return False, str(e)[:80]


def check_prefix_cache_smoke(url, model):
    prompt = "Repeat the word cache exactly once. " * 64
    try:
        first_status, first = one_short_completion(url, model, prompt)
        second_status, second = one_short_completion(url, model, prompt)
        ok = first_status == 200 and second_status == 200
        return ok, f"first_s={first:.2f}, second_s={second:.2f}"
    except Exception as e:
        return False, str(e)[:80]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--wait-timeout", type=int, default=600)
    args = ap.parse_args()
    url = args.base_url.rstrip("/")

    init_openlit_observability()
    wait_for_server(url, args.wait_timeout)

    print(f"[features] {args.model} @ {url}\n")
    print(f"  streaming      : {check_streaming(url, args.model)}")
    ok, detail = check_openai_models(url)
    print(f"  openai_models  : {ok}  ({detail})")
    ok, detail = check_concurrent_requests(url, args.model)
    print(f"  concurrency    : {ok}  ({detail})")
    ok, detail = check_prefix_cache_smoke(url, args.model)
    print(f"  prefix_cache   : {ok}  ({detail})")
    ok, detail = check_json_mode(url, args.model)
    print(f"  json_mode      : {ok}  ({detail})")
    ok, detail = check_tool_calling(url, args.model)
    print(f"  tool_calling   : {ok}  ({detail})")
    print("  speculative    : manual  (requires engine-specific startup config/log verification)")


if __name__ == "__main__":
    main()
