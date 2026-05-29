# TensorRT-LLM

An engine for extracting maximum performance from NVIDIA hardware. However, **engine builds are GPU architecture (SM)-dependent**, and this has the heaviest setup.

## RTX 5080 (Blackwell, sm_120) Compatibility — Important

- Blackwell consumer GPUs require **CUDA 12.8+**. Older TRT-LLM containers do not include sm_120 kernels and will not run.
- Use a pinned `nvcr.io/nvidia/tensorrt-llm/release` tag that explicitly mentions **Blackwell support**. Floating tags do not always guarantee compatibility, so check the release notes.
- FP8 works well on Blackwell, but some plugins/kernels may not support sm_120, so always check build logs.
- Builds take a long time and consume significant VRAM/disk space. On 16GB, 8B FP8 is the realistic target.

## Two Paths

### A. trtllm-serve (simple) — compose default
Serve an HF checkpoint directly. It builds/caches the engine internally and is useful for a quick trial.

```bash
source ../../configs/profiles/${LAB_PROFILE}.env
docker compose up -d
```

### B. trtllm-build (standard) — maximum performance
Explicitly build a checkpoint into engine artifacts, then serve those artifacts. This gives precise control over quantization and plugin options.

```bash
# After entering the container
trtllm-build --checkpoint_dir <converted_ckpt> \
  --output_dir /engines/qwen3-8b-fp8 \
  --gemm_plugin auto --max_seq_len 8192
```

## Benchmark

```bash
cd ../..
uv run --project scripts/bench python scripts/bench/run_benchmark.py --engine tensorrt-llm --base-url http://localhost:8000
```

## Feature Notes
- Structured Output / Tool Calling support is version-dependent -> validate it directly in Phase 2.
