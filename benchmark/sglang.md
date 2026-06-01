# sglang — Benchmark Results

Source: `results/sglang_homelab.json`

Last run: `2026-05-30T15:40:13.988662+00:00`

## Environment
- Profile: `homelab`
- GPU: `RTX 5080`
- Model: `Qwen/Qwen3-8B-FP8`
- Quantization: `fp8`
- Engine version/image tag: `lmsysorg/sglang:nightly-dev-cu13-20260522-c9153da5`
- Prompt / output tokens: `512` / `256`
- Prompts per level: `200`
- Token count source: tokenizer fallback, completion tokens from `stream_usage`
- Observability: none for this benchmark run

## Single GPU (Phase 1)

| Concurrency | OK | Failed | TTFT p50 (ms) | TTFT p95 (ms) | E2E p50 (s) | E2E p95 (s) | Mean tok/s/req | Agg tok/s | Throughput (req/s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 0 | 34.6 | 37.5 | 4.485 | 4.634 | 57.0 | 57.0 | 0.22 |
| 4 | 200 | 0 | 30.4 | 35.4 | 3.498 | 3.811 | 72.3 | 288.7 | 1.13 |
| 8 | 200 | 0 | 31.9 | 41.3 | 3.589 | 3.879 | 70.4 | 562.7 | 2.20 |
| 16 | 200 | 0 | 64.5 | 67.2 | 3.833 | 3.885 | 67.0 | 1032.0 | 4.03 |

## GPU Resources (DCGM)
Per-level DCGM windows were not captured in the benchmark JSON. Use Grafana/Prometheus over the run window to fill these values if needed.

| Concurrency | GPU Util (%) | VRAM (GB) | Power (W) |
|---:|---:|---:|---:|
| 1 | | | |
| 4 | | | |
| 8 | | | |
| 16 | | | |

## Multi-GPU Scaling (Phase 3)

| Model | TP Size | GPUs | Agg tok/s | Scaling Efficiency | E2E p95 (s) | Per-GPU VRAM (GB) | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
| | 1 | 1 | | baseline | | | |
| | 2 | 2 | | | | | |
| | 4 | 4 | | | | | |

- Interconnect:
- Communication overhead:
- Memory distribution:

## Phase 2 Feature Validation
Record the `validate_features.py` results.

| Feature | Supported | Notes |
|---|---|---|
| Continuous Batching | | |
| Prefix Cache | | |
| Speculative Decoding | | |
| Structured Output / JSON | | |
| Tool Calling | | |
| OpenAI API | | |
| Streaming | | |

## Observations / Notes

- Startup time: image pull dominated first setup; model load and CUDA graph capture completed after container start.
- OOM/stability: 0 failed requests across all tested concurrency levels.
- Operational ease: SGLang exposed OpenAI-compatible streaming responses and `stream_usage` token counts for all levels.
- Runtime note: `/metrics` returned 404 with the current Compose command, so Prometheus did not scrape SGLang-native server metrics in this run.
