# tgi — Benchmark Results

> After running `scripts/bench/run_benchmark.py`, fill in the tables from `results/tgi_<profile>.json`.

## Environment
- Profile: `homelab` / `datacenter`
- GPU:
- Model:
- Quantization:
- Engine version/image tag:
- Token count source:
- Observability: Prometheus/DCGM / OpenLIT / none

## Single GPU (Phase 1)

| Concurrency | TTFT p50 (ms) | TTFT p95 (ms) | E2E p50 (s) | Agg tok/s | Throughput (req/s) |
|---|---|---|---|---|---|
| 1 | | | | | |
| 4 | | | | | |
| 8 | | | | | |
| 16 | | | | | |
| 32 | | | | | |

## GPU Resources (DCGM)
| Concurrency | GPU Util (%) | VRAM (GB) | Power (W) |
|---|---|---|---|
| | | | |

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

- Startup time:
- OOM/stability:
- Operational ease:
