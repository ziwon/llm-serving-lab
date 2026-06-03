# tensorrt-llm — Benchmark Results

Sources:
- Comparable 8B run: no successful `results/tensorrt-llm_homelab.json` was produced.
- Qwen3-4B fallback run: `results/tensorrt-llm_qwen3-4b_homelab.json`

Last attempted: `2026-06-03T04:40:41+00:00`
Last successful fallback run: `2026-06-03T05:10:38.134517+00:00`

## Environment
- Profile: `homelab`
- GPU: `RTX 5080`
- Model: `Qwen/Qwen3-8B-FP8`
- Quantization: `fp8`
- Engine version/image tag: `nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc7`
- Prompt / output tokens: `512` / `256`
- Prompts per level: `200`
- Token count source: not captured; server did not become healthy
- Observability: none for this benchmark attempt

## Single GPU (Phase 1)

TensorRT-LLM did not reach a healthy serving state for the comparable homelab model, so no request-level benchmark was run.

| Concurrency | OK | Failed | TTFT p50 (ms) | TTFT p95 (ms) | E2E p50 (s) | E2E p95 (s) | Mean tok/s/req | Agg tok/s | Throughput (req/s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | | | | | | | | | |
| 4 | | | | | | | | | |
| 8 | | | | | | | | | |
| 16 | | | | | | | | | |

## Single GPU (Qwen3-4B Fallback)

This run used `Qwen/Qwen3-4B` with the same homelab request shape.

| Concurrency | OK | Failed | TTFT p50 (ms) | TTFT p95 (ms) | E2E p50 (s) | E2E p95 (s) | Mean tok/s/req | Agg tok/s | Throughput (req/s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 200 | 0 | 42.8 | 44.6 | 2.838 | 4.998 | 77.4 | 73.0 | 0.28 |
| 4 | 200 | 0 | 43.7 | 2031.5 | 4.995 | 5.009 | 59.0 | 224.6 | 0.88 |
| 8 | 200 | 0 | 42.1 | 53.4 | 4.999 | 5.010 | 53.0 | 419.4 | 1.64 |
| 16 | 200 | 0 | 53.2 | 65.8 | 5.000 | 5.023 | 51.2 | 787.7 | 3.08 |

## GPU Resources (DCGM)
Per-level DCGM windows were not captured in the benchmark JSON. For the Qwen3-4B fallback run, a live `nvidia-smi` sample showed about 14.6 GiB VRAM in use under load.

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

- Startup time: first attempt pulled the large NGC TensorRT-LLM image successfully, then launched `trtllm-serve` with the homelab profile. A second startup attempt with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` also failed before serving.
- Cache path: Compose now mounts Hugging Face data from `/data/LLM/models/hugging-face` into `/root/.cache/huggingface`.
- OOM/stability: startup failed before serving. TensorRT-LLM reported `Executor creation failed due to insufficient GPU memory` while loading `Qwen/Qwen3-8B-FP8`.
- Failure detail: the PyTorch backend used 8.79 GB for model weights and attempted FP8 post-load resmoothing (`resmooth_to_fp8_e8m0`). It then hit `torch.OutOfMemoryError` on a 384 MiB allocation. The allocator retry still failed with about 365.19 MiB free on a 15.44 GiB visible GPU.
- Free-memory note: TensorRT-LLM reported 12.78 GiB free before model component creation on the retry. Desktop/display processes were using roughly 2.1 GiB on the RTX 5080 during the attempt.
- Operational ease: `trtllm-serve` exposed no healthy OpenAI-compatible endpoint, so `run_benchmark.py` was not run.
- Qwen3-4B fallback: `Qwen/Qwen3-4B` reached healthy serving state and completed all Phase 1 concurrency levels with 0 failed requests.
- Qwen3-4B memory: TensorRT-LLM reported 8.22 GB for model weights, 7.74 GiB inside torch after loading, 6.04 GiB outside torch, and 3.51 GiB allocated for paged KV cache. A live `nvidia-smi` sample during the run showed about 14.6 GiB VRAM in use.
- Qwen3-4B runtime note: startup used the PyTorch backend with `max_seq_len=4097`, `max_num_tokens=8192`, and `max_batch_size=2048`. The 4-way run had a TTFT p95 outlier of 2031.5 ms despite 0 failures.
