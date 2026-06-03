# TensorRT ONNX Lab — Benchmark Results

Source:
- Build log: `engines/tensorrt/results/resnet50-v2-7-fp16.plan.build.log`
- Benchmark log: `engines/tensorrt/results/resnet50-v2-7-fp16.plan.bench.log`

Last successful run: `2026-06-03`

## Environment

- Profile: `homelab`
- GPU: `NVIDIA GeForce RTX 5080`
- TensorRT image: `nvcr.io/nvidia/tensorrt:25.10-py3`
- TensorRT version: `10.13.3`
- Model: `resnet50-v2-7.onnx`
- Engine: `resnet50-v2-7-fp16.plan`
- Precision: `fp16`
- Input shape: `data:1x3x224x224`

## Artifacts

| Artifact | Path | Size |
|---|---|---:|
| ONNX model | `engines/tensorrt/models/resnet50-v2-7.onnx` | 98M |
| TensorRT engine | `engines/tensorrt/engines-out/resnet50-v2-7-fp16.plan` | 51M |
| Build log | `engines/tensorrt/results/resnet50-v2-7-fp16.plan.build.log` | 70K |
| Benchmark log | `engines/tensorrt/results/resnet50-v2-7-fp16.plan.bench.log` | 54K |

## Build Result

| Metric | Value |
|---|---:|
| ONNX parse | successful |
| Engine generation time | 25.7832 s |
| Engine size | 50.4261 MiB |
| Peak TensorRT GPU allocator memory | 70 MiB |
| Execution context device memory | 4.21094 MiB |

## Serialized Engine Benchmark

This benchmark loaded the serialized TensorRT engine and ran `trtexec` with the same static input shape.

| Metric | Value |
|---|---:|
| Status | `PASSED` |
| Throughput | 2411.25 qps |
| Host latency mean | 0.436992 ms |
| Host latency median | 0.412842 ms |
| Host latency p90 | 0.514648 ms |
| Host latency p95 | 0.637573 ms |
| Host latency p99 | 0.715820 ms |
| GPU compute mean | 0.413124 ms |
| GPU compute median | 0.389343 ms |
| GPU compute p95 | 0.614014 ms |
| GPU compute p99 | 0.692505 ms |
| Enqueue mean | 0.209124 ms |
| Total host walltime | 3.00093 s |
| Total GPU compute time | 2.98936 s |

## Notes

- This run validates the ONNX to TensorRT build path on the local GPU with very low VRAM pressure.
- The TensorRT container was run with `--rm`; no long-running serving container remains after the benchmark.
- `nvidia-smi` after the run reported about 2046 MiB used out of 16303 MiB total, mostly baseline desktop/system usage.
- `trtexec` warned that GPU compute time had high variance. Locking clocks or adding `--useSpinWait` may improve timing stability for more formal latency measurements.
