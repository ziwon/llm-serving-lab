# Benchmark Summary

Generated from `results/*.json`.

## Runs

| Result | Engine | Profile | Model | Timestamp | Image | Workload | Prompt Mode |
|---|---|---|---|---|---|---|---|
| `results/tensorrt-llm_qwen3-4b_homelab.json` | `tensorrt-llm` | `homelab` | `Qwen/Qwen3-4B` | `2026-06-03T06:01:21.032051+00:00` | `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc7` | `legacy` | `repeated` |
| `results/tensorrt-llm_engine_qwen3-4b_homelab_smoke.json` | `tensorrt-llm-engine` | `unknown` | `Qwen/Qwen3-4B` | `2026-06-03T07:02:41.274243+00:00` | `` | `legacy` | `repeated` |
| `results/vllm_homelab.json` | `vllm` | `homelab` | `Qwen/Qwen3-8B-FP8` | `2026-06-03T08:34:27.839301+00:00` | `vllm/vllm-openai:v0.21.0-cu129-ubuntu2404` | `prefill_heavy` | `repeated` |
| `results/sglang_homelab.json` | `sglang` | `homelab` | `Qwen/Qwen3-8B-FP8` | `2026-06-03T08:41:40.869907+00:00` | `lmsysorg/sglang:nightly-dev-cu13-20260522-c9153da5` | `prefill_heavy` | `repeated` |
| `results/tensorrt-llm_engine_qwen3-4b_homelab.json` | `tensorrt-llm-engine` | `homelab` | `Qwen/Qwen3-4B` | `2026-06-03T08:46:28.116731+00:00` | `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc7` | `prefill_heavy` | `repeated` |

## Best Throughput By Run

| Engine | Model | Profile | Concurrency | OK | Failed | Agg tok/s | TTFT p95 (ms) | E2E p95 (s) | GPU util avg (%) | VRAM max (GiB) | Power avg (W) | Tokens/J |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sglang` | `Qwen/Qwen3-8B-FP8` | `homelab` | 16 | 50 | 0 | 821.0 | 66.3 | 3.955 | 98.0 | 13.73 | 253.7 | 3.2363 |
| `tensorrt-llm` | `Qwen/Qwen3-4B` | `homelab` | 16 | 200 | 0 | 853.6 | 61.7 | 5.012 |  |  |  |  |
| `tensorrt-llm-engine` | `Qwen/Qwen3-4B` | `homelab` | 16 | 50 | 0 | 987.8 | 60.7 | 3.349 | 98.0 | 15.07 | 234.7 | 4.2090 |
| `tensorrt-llm-engine` | `Qwen/Qwen3-4B` | `unknown` | 1 | 3 | 0 | 77.8 | 58.8 | 0.230 |  |  |  |  |
| `vllm` | `Qwen/Qwen3-8B-FP8` | `homelab` | 16 | 200 | 0 | 1214.9 | 72.2 | 3.271 | 100.0 | 14.92 | 224.7 | 5.4059 |

## Per-Concurrency Results

### sglang / Qwen/Qwen3-8B-FP8 / homelab

Source: `results/sglang_homelab.json`

| Concurrency | OK | Failed | TTFT p50 (ms) | TTFT p95 (ms) | E2E p50 (s) | E2E p95 (s) | Mean tok/s/req | Agg tok/s | Req/s | GPU util avg (%) | VRAM max (GiB) | Power avg (W) | Tokens/J |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50 | 0 | 34.2 | 36.4 | 4.466 | 4.582 | 57.9 | 57.8 | 0.23 | 84.4 | 14.41 | 208.3 | 0.2774 |
| 4 | 50 | 0 | 31.1 | 32.0 | 3.619 | 3.627 | 70.7 | 271.8 | 1.06 | 100.0 | 13.76 | 242.5 | 1.1209 |
| 8 | 50 | 0 | 32.9 | 101.6 | 3.769 | 3.832 | 67.9 | 482.2 | 1.88 | 100.0 | 13.71 | 253.1 | 1.9053 |
| 16 | 50 | 0 | 64.7 | 66.3 | 3.949 | 3.955 | 65.0 | 821.0 | 3.21 | 98.0 | 13.73 | 253.7 | 3.2363 |

### tensorrt-llm / Qwen/Qwen3-4B / homelab

Source: `results/tensorrt-llm_qwen3-4b_homelab.json`

| Concurrency | OK | Failed | TTFT p50 (ms) | TTFT p95 (ms) | E2E p50 (s) | E2E p95 (s) | Mean tok/s/req | Agg tok/s | Req/s | GPU util avg (%) | VRAM max (GiB) | Power avg (W) | Tokens/J |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 0 | 42.9 | 45.3 | 2.651 | 4.468 | 86.1 | 82.2 | 0.32 |  |  |  |  |
| 4 | 200 | 0 | 43.5 | 49.0 | 2.851 | 4.995 | 77.8 | 296.2 | 1.16 |  |  |  |  |
| 8 | 200 | 0 | 48.2 | 62.3 | 2.921 | 4.989 | 74.7 | 568.4 | 2.22 |  |  |  |  |
| 16 | 200 | 0 | 53.3 | 61.7 | 4.995 | 5.012 | 56.3 | 853.6 | 3.33 |  |  |  |  |

### tensorrt-llm-engine / Qwen/Qwen3-4B / unknown

Source: `results/tensorrt-llm_engine_qwen3-4b_homelab_smoke.json`

| Concurrency | OK | Failed | TTFT p50 (ms) | TTFT p95 (ms) | E2E p50 (s) | E2E p95 (s) | Mean tok/s/req | Agg tok/s | Req/s | GPU util avg (%) | VRAM max (GiB) | Power avg (W) | Tokens/J |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 0 | 44.9 | 58.8 | 0.198 | 0.230 | 78.4 | 77.8 | 4.86 |  |  |  |  |

### tensorrt-llm-engine / Qwen/Qwen3-4B / homelab

Source: `results/tensorrt-llm_engine_qwen3-4b_homelab.json`

| Concurrency | OK | Failed | TTFT p50 (ms) | TTFT p95 (ms) | E2E p50 (s) | E2E p95 (s) | Mean tok/s/req | Agg tok/s | Req/s | GPU util avg (%) | VRAM max (GiB) | Power avg (W) | Tokens/J |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 50 | 0 | 43.4 | 44.9 | 2.942 | 3.044 | 86.6 | 86.6 | 0.34 | 85.1 | 15.08 | 235.2 | 0.3682 |
| 4 | 50 | 0 | 43.3 | 46.2 | 3.073 | 3.137 | 83.5 | 321.4 | 1.26 | 98.0 | 15.08 | 241.0 | 1.3335 |
| 8 | 50 | 0 | 44.0 | 51.0 | 3.119 | 3.164 | 81.9 | 587.0 | 2.29 | 98.0 | 15.07 | 228.6 | 2.5677 |
| 16 | 50 | 0 | 45.9 | 60.7 | 3.311 | 3.349 | 77.7 | 987.8 | 3.86 | 98.0 | 15.07 | 234.7 | 4.2090 |

### vllm / Qwen/Qwen3-8B-FP8 / homelab

Source: `results/vllm_homelab.json`

| Concurrency | OK | Failed | TTFT p50 (ms) | TTFT p95 (ms) | E2E p50 (s) | E2E p95 (s) | Mean tok/s/req | Agg tok/s | Req/s | GPU util avg (%) | VRAM max (GiB) | Power avg (W) | Tokens/J |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 0 | 24.6 | 27.5 | 4.386 | 4.417 | 58.3 | 57.3 | 0.22 | 97.1 | 14.04 | 189.3 | 0.3026 |
| 4 | 200 | 0 | 48.3 | 49.4 | 3.920 | 3.937 | 65.2 | 260.9 | 1.02 | 100.0 | 14.63 | 205.0 | 1.2729 |
| 8 | 200 | 0 | 51.0 | 59.8 | 4.059 | 4.099 | 66.2 | 523.5 | 2.04 | 99.8 | 14.69 | 205.7 | 2.5454 |
| 16 | 200 | 0 | 60.4 | 72.2 | 3.257 | 3.271 | 78.8 | 1214.9 | 4.75 | 100.0 | 14.92 | 224.7 | 5.4059 |
