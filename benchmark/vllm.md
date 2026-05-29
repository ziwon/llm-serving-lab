# vllm — Benchmark Results

Source: `results/vllm_homelab.json`

Last run: `2026-05-29T17:12:30.076793+00:00`

## Environment
- Profile: `homelab`
- GPU: `RTX 5080`
- Model: `Qwen/Qwen3-8B-FP8`
- Quantization: `fp8`
- Engine version/image tag: `vllm/vllm-openai:v0.21.0-cu129-ubuntu2404`
- Prompt / output tokens: `512` / `256`
- Prompts per level: `200`
- Token count source: tokenizer fallback, completion tokens from `stream_usage`
- Observability: none for this benchmark run

## Single GPU (Phase 1)

| Concurrency | OK | Failed | TTFT p50 (ms) | TTFT p95 (ms) | E2E p50 (s) | E2E p95 (s) | Mean tok/s/req | Agg tok/s | Throughput (req/s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 0 | 21.0 | 23.2 | 2.964 | 3.102 | 86.0 | 85.9 | 0.34 |
| 4 | 200 | 0 | 36.6 | 47.5 | 3.066 | 3.262 | 82.9 | 331.2 | 1.29 |
| 8 | 200 | 0 | 48.6 | 53.5 | 3.220 | 3.285 | 79.7 | 636.3 | 2.49 |
| 16 | 200 | 0 | 57.1 | 70.0 | 3.370 | 3.398 | 76.0 | 1169.7 | 4.57 |

## GPU Resources (DCGM)
Source: Grafana/Prometheus DCGM metrics over the benchmark completion windows. GPU utilization and power are averages; VRAM is peak `DCGM_FI_DEV_FB_USED`.

| Concurrency | GPU Util (%) | VRAM (GB) | Power (W) |
|---:|---:|---:|---:|
| 1 | 97.6 | 14.0 | 236.1 |
| 4 | 99.8 | 13.7 | 234.6 |
| 8 | 100.0 | 13.7 | 234.3 |
| 16 | 99.5 | 13.7 | 244.8 |

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

- Startup time: not captured in this benchmark JSON.
- OOM/stability: 0 failed requests across all tested concurrency levels.
- Operational ease: vLLM exposed OpenAI-compatible streaming responses and `stream_usage` token counts for all levels.
