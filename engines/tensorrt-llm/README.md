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

From the repository root, the Qwen3-4B homelab profile can be started with the curated low-VRAM config:

```bash
source configs/profiles/homelab.env
just trtllm-qwen3-4b-up
```

Wait for the service to become healthy before probing `/v1/models`:

```bash
docker ps --filter name=trtllm
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

Run the Qwen3-4B benchmark:

```bash
just trtllm-qwen3-4b-bench
```

### B. convert_checkpoint.py + trtllm-build (standard) — maximum performance
Explicitly build a checkpoint into engine artifacts, then serve those artifacts. This gives precise control over quantization and plugin options.

Important: `trtllm-build` expects a TensorRT-LLM checkpoint/config directory, not a raw Hugging Face snapshot. In this TensorRT-LLM `1.2.0rc7` container, the Qwen3 quick-start path uses `trtllm-serve ... --config`; a raw `Qwen/Qwen3-4B` HF snapshot is not directly accepted by `trtllm-build`.

The repository-level build lab is documented at:

```bash
docs/labs/tensorrt-llm-build-lab.md
```

For the homelab Qwen3-4B path:

```bash
source configs/profiles/homelab.env
just trtllm-build-qwen3-4b-lab
just trtllm-qwen3-4b-engine-up
just trtllm-qwen3-4b-engine-bench
```

```bash
# After entering the container
trtllm-build --checkpoint_dir <converted_ckpt> \
  --output_dir /engines/qwen3-8b-fp8 \
  --gemm_plugin auto --max_seq_len 8192
```

From the repository root, if you already have a converted TensorRT-LLM checkpoint directory in the repo, build an engine with:

```bash
just trtllm-build path/to/converted_ckpt qwen3-4b-rtx5080
```

Serve that built engine:

```bash
just trtllm-engine-up qwen3-4b-rtx5080 Qwen/Qwen3-4B
```

## Benchmark

```bash
cd ../..
uv run --project scripts/bench python scripts/bench/run_benchmark.py --engine tensorrt-llm --base-url http://localhost:8000
```

## Feature Notes
- Structured Output / Tool Calling support is version-dependent -> validate it directly in Phase 2.
