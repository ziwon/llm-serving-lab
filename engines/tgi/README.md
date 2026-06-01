# HuggingFace TGI (Text Generation Inference)

Strong HF ecosystem integration and enterprise operability. It is the first-choice engine for the Enterprise HF Ecosystem scenario. Structured output support is limited.

## Run
```bash
source ../../configs/profiles/${LAB_PROFILE}.env
docker compose up -d
curl http://localhost:${SERVED_PORT:-8000}/health
```

## RTX 5080 Notes
- Confirm that the TGI image tag supports Blackwell. Builds without sm_120 support will fail to start.
- `3.3.5` was referenced by upstream docs but returned `manifest unknown` from GHCR during the homelab run; this lab pins `3.3.4`.
- `3.3.6` and `3.3.7` are available on GHCR, but `3.3.7` still reported the same relevant Python runtime stack in this environment: `torch=2.7.0+cu128`, `triton=3.3.0`, `transformers=4.51.0`.
- Port: container 80 -> host `SERVED_PORT`.
- `--max-input-tokens` must be smaller than `--max-total-tokens`.
- The OpenAI-compatible endpoint is available at `/v1/chat/completions`.
- Additional launcher flags can be passed with `TGI_EXTRA_ARGS`.

## Homelab RTX 5080 Failure Report

No successful `results/tgi_homelab.json` has been produced yet. These failures were reproduced on the homelab RTX 5080 (`sm_120`) with `HF_HOME=/data/LLM/models/hugging-face`.

### 1. Qwen3 FP8 checkpoint

`Qwen/Qwen3-8B-FP8` fails during TGI warmup before the HTTP server becomes healthy.

- Images tried: `ghcr.io/huggingface/text-generation-inference:3.3.4`, `latest`, and `latest` with `--disable-custom-kernels`.
- Failure: `RuntimeError: PassManager::run failed` in `transformers/integrations/finegrained_fp8.py`.
- Direct repro: calling `w8a8_block_fp8_matmul_triton` inside the TGI image fails for CUDA target `120` with `computeCapability not supported` in Triton `TritonGPUAccelerateMatmul`.
- Root cause: the Transformers fine-grained FP8 W8A8 block matmul path uses Triton JIT, and the TGI image's `triton=3.3.0` cannot compile that kernel for RTX 5080 / `sm_120`.
- `--disable-custom-kernels` does not help because this is a Transformers/Triton kernel path, not a TGI custom CUDA kernel.

### 2. Qwen3 AWQ fallback

`Qwen/Qwen3-8B-AWQ` is already cached locally, but it is not a usable TGI fallback with this image.

- Failure: `NotImplementedError: awq quantization is not supported for AutoModel`.
- Root cause: TGI routes Qwen3 through its generic `AutoModel` path, and AWQ is rejected there.

### 3. Runtime FP8 fallback

Using non-FP8 checkpoints with TGI runtime FP8 avoids the Transformers fine-grained FP8 path, but still does not reach a healthy server.

- `Qwen/Qwen2.5-7B-Instruct --quantize fp8 --kv-cache-dtype fp8_e4m3fn` selected `cutlass w8a8 kernels`, but TGI warned that available VRAM was below the model estimate and then crashed during warmup with `transport error`.
- `Qwen/Qwen2.5-3B-Instruct --quantize fp8 --kv-cache-dtype fp8_e4m3fn` also selected `cutlass w8a8 kernels`, then crashed during warmup with `transport error`.

### 4. Native non-FP8 fallback

Even the smaller cached `Qwen/Qwen2.5-3B-Instruct` without quantization failed during warmup.

- Reduced settings tried: `MAX_MODEL_LEN=1024`, `MAX_INPUT_TOKENS=768`, `--cuda-graphs 0`, `--max-batch-prefill-tokens 1024`, and `--disable-custom-kernels`.
- Failure: `CUDA Error: no kernel image is available for execution on the device /usr/src/flash-attention/csrc/layer_norm/ln_fwd_kernels.cuh 236`.
- Root cause: TGI's bundled FlashAttention-derived layernorm CUDA extension is not built with a compatible kernel image for `sm_120`.
- Installing packages on the host will not fix this because the failing extension lives inside the TGI container. A future fix likely requires an upstream TGI image with Blackwell-compatible extension builds or a custom experimental TGI image that rebuilds the CUDA extensions for `sm_120`.

### Current recommendation

Use SGLang or vLLM for RTX 5080 benchmarks until TGI ships an image with a newer Blackwell-compatible Torch/Triton/FlashAttention stack. Do not treat TGI results as missing benchmark data only; the current blocker is engine/runtime compatibility on `sm_120`.

## Benchmark
```bash
cd ../..
uv run --project scripts/bench python scripts/bench/run_benchmark.py --engine tgi --base-url http://localhost:${SERVED_PORT:-8000}
```
