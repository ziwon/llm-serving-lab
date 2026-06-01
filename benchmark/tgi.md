# tgi — Benchmark Results

Source: no successful `results/tgi_homelab.json` was produced.

Last attempted: `2026-05-31T02:36:50+00:00`

## Environment
- Profile: `homelab`
- GPU: `RTX 5080`
- Model: `Qwen/Qwen3-8B-FP8`
- Quantization: `fp8`
- Engine version/image tag: `ghcr.io/huggingface/text-generation-inference:3.3.4`
- Prompt / output tokens: `512` / `256`
- Prompts per level: `200`
- Token count source: not captured; server did not become healthy
- Observability: none for this benchmark attempt

## Single GPU (Phase 1)

TGI did not reach a healthy serving state for the comparable homelab model, so no request-level benchmark was run.

| Concurrency | OK | Failed | TTFT p50 (ms) | TTFT p95 (ms) | E2E p50 (s) | E2E p95 (s) | Mean tok/s/req | Agg tok/s | Throughput (req/s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | | | | | | | | | |
| 4 | | | | | | | | | |
| 8 | | | | | | | | | |
| 16 | | | | | | | | | |

## GPU Resources (DCGM)
Per-level DCGM windows were not captured because the benchmark did not run.

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

- Startup time: image pull completed after pinning to `3.3.4`; `3.3.5` returned `manifest unknown` from GHCR in this environment.
- Port note: `sglang` was still bound to host port `8000`, so TGI was attempted on host port `8001`.
- OOM/stability: initial `Qwen/Qwen3-8B-FP8` startup hit CUDA OOM while `sglang` was still using GPU 0. After stopping `sglang`, the model loaded but TGI crashed during warmup.
- Compatibility blocker: `Qwen/Qwen3-8B-FP8` on TGI `3.3.4` failed during warmup with `RuntimeError: PassManager::run failed` inside Transformers fine-grained FP8 Triton matmul (`w8a8_block_fp8_matmul_triton`) on RTX 5080. Retrying with `--disable-custom-kernels` produced the same error.
- Runtime FP8 fallback: `Qwen/Qwen3-8B` with `--quantize fp8 --kv-cache-dtype fp8_e4m3fn` began downloading the BF16 checkpoint, but failed with `No space left on device` before reaching startup. The partial `Qwen/Qwen3-8B` cache created by this attempt was removed; the existing `Qwen/Qwen3-8B-FP8` cache was kept.
- Operational ease: TGI exposes the expected OpenAI-compatible endpoint, but this homelab RTX 5080 + Qwen3 FP8 path is currently blocked before serving.
