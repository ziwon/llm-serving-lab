# TensorRT-LLM Conversion and Build Lab

This lab covers the real TensorRT-LLM path for LLMs:

```text
Hugging Face checkpoint -> TensorRT-LLM checkpoint -> TensorRT engine -> trtllm-serve
```

This is different from `docs/tensorrt-onnx-lab.md`, which uses a generic ONNX model and `trtexec`.

## Why This Exists

TensorRT-LLM engine artifacts are hardware-specific. Build on the target GPU architecture when possible, especially on Blackwell consumer GPUs such as RTX 5080 (`sm_120`).

The pinned TensorRT-LLM image used by this repo includes the Qwen conversion script at:

```bash
/app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py
```

The flow is based on NVIDIA's standard `convert_checkpoint.py` + `trtllm-build` workflow.

## Homelab Qwen3-4B Build

Start from the repository root:

```bash
source configs/profiles/homelab.env
just down tensorrt-llm
```

Convert the Hugging Face model into a TensorRT-LLM checkpoint:

```bash
just trtllm-convert-qwen Qwen/Qwen3-4B qwen3-4b-fp16-tp1 float16 1 1 --load_model_on_cpu
```

The recipe accepts either a container-visible local model directory or a Hugging Face repo id. For repo ids such as `Qwen/Qwen3-4B`, it first downloads a local snapshot under `engines/tensorrt-llm/hf-models/`, then passes that concrete directory to TensorRT-LLM's Qwen converter.

Build a TensorRT engine from that checkpoint:

```bash
just trtllm-build-checkpoint qwen3-4b-fp16-tp1 qwen3-4b-fp16-tp1-4096 4096 3584 4096 16 1
```

Or run both steps together:

```bash
just trtllm-build-qwen3-4b-lab
```

The combined recipe skips conversion when the target TensorRT-LLM checkpoint already exists.

Serve the built engine:

```bash
just trtllm-qwen3-4b-engine-up
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

The engine-serving recipe passes `--backend tensorrt` and runtime limits matching the built engine. Without `--backend tensorrt`, `trtllm-serve` treats the engine directory as a Hugging Face/PyTorch model path. If runtime `max_batch_size` is larger than the engine build limit, executor startup fails.

Benchmark it:

```bash
just trtllm-qwen3-4b-engine-bench
```

## Artifacts

Generated artifacts are ignored by git and written under:

```bash
/data/LLM/artifacts/llm-serving-lab/tensorrt-llm/checkpoints/
/data/LLM/artifacts/llm-serving-lab/tensorrt-llm/engines-out/
/data/LLM/artifacts/llm-serving-lab/tensorrt-llm/hf-models/
/data/LLM/artifacts/llm-serving-lab/tensorrt-llm/results/
```

Expected files for the default lab:

```bash
/data/LLM/artifacts/llm-serving-lab/tensorrt-llm/checkpoints/qwen3-4b-fp16-tp1/
/data/LLM/artifacts/llm-serving-lab/tensorrt-llm/engines-out/qwen3-4b-fp16-tp1-4096/
/data/LLM/artifacts/llm-serving-lab/tensorrt-llm/hf-models/Qwen_Qwen3-4B/
/data/LLM/artifacts/llm-serving-lab/tensorrt-llm/results/qwen3-4b-fp16-tp1.convert.log
/data/LLM/artifacts/llm-serving-lab/tensorrt-llm/results/qwen3-4b-fp16-tp1-4096.build.log
```

Override the artifact root with `TRTLLM_ARTIFACT_DIR` if needed.

## VRAM Notes

The default homelab build is intentionally conservative:

| Setting | Value |
|---|---:|
| Model | `Qwen/Qwen3-4B` |
| Checkpoint dtype | `float16` |
| Tensor parallel size | `1` |
| Max sequence length | `4096` |
| Max input length | `3584` |
| Max batched tokens | `4096` |
| Max batch size | `16` |
| Convert extra arg | `--load_model_on_cpu` |

If build or serve hits OOM, lower `max_batch_size`, then `max_num_tokens`, then `max_seq_len`.

If conversion completed but engine build failed, rerun only the build step:

```bash
just trtllm-build-checkpoint qwen3-4b-fp16-tp1 qwen3-4b-fp16-tp1-4096 4096 3584 4096 16 1
```

The build recipe removes the target engine directory before rebuilding, so a partial `rank0.engine` from a failed serialization is not reused.

## Custom Qwen Model

```bash
just trtllm-convert-qwen <hf-model-or-local-dir> <checkpoint-name> auto 1 1 --load_model_on_cpu
just trtllm-build-checkpoint <checkpoint-name> <engine-name> 4096 3584 4096 16 1
just trtllm-engine-up <engine-name> <hf-tokenizer-id>
```

For multi-GPU builds, set the profile and pass a larger tensor parallel size:

```bash
source configs/profiles/datacenter.env
just trtllm-convert-qwen Qwen/Qwen3-32B qwen3-32b-fp16-tp4 float16 4 4 --load_model_on_cpu
just trtllm-build-checkpoint qwen3-32b-fp16-tp4 qwen3-32b-fp16-tp4-32768 32768 31744 8192 128 4
```

## Troubleshooting

- `trtllm-build` requires a TensorRT-LLM checkpoint directory, not a raw Hugging Face snapshot.
- Engines should be rebuilt when the GPU architecture, TensorRT-LLM version, plugin settings, tensor parallel size, or important sequence/batch limits change.
- On RTX 5080, stop other GPU workloads first and check `nvidia-smi` before building.
- Keep Hugging Face caches on `/data/LLM/models/hugging-face` via `HF_HOME`.
