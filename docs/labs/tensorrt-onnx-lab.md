# TensorRT ONNX Lab

This lab covers the classic ONNX to TensorRT workflow. It is separate from TensorRT-LLM, which uses LLM checkpoints and `trtllm-serve` / `trtllm-build` instead of generic ONNX import.

## ResNet-50 Quick Run

Download the ONNX model:

```bash
just trt-onnx-download-resnet50
```

Open the model graph in Netron:

```bash
just trt-onnx-view-resnet50
```

Build a TensorRT FP16 engine:

```bash
just trt-onnx-build-resnet50
```

Benchmark the engine:

```bash
just trt-onnx-bench
```

Run download, build, and benchmark together:

```bash
just trt-onnx-lab-resnet50
```

The recorded homelab result is available at:

```bash
benchmark/tensorrt-onnx.md
```

Artifacts are written under:

```bash
engines/tensorrt/models/
engines/tensorrt/engines-out/
engines/tensorrt/results/
```

## Custom ONNX Model

```bash
just trt-onnx-build path/to/model.onnx model-fp16.plan "input:1x3x224x224" fp16
just trt-onnx-bench model-fp16.plan "input:1x3x224x224"
```

Set `TENSORRT_IMAGE` to override the container image:

```bash
export TENSORRT_IMAGE=nvcr.io/nvidia/tensorrt:25.10-py3
```
