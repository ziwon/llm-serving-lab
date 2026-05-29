# LLM Serving Lab

**2026 OSS LLM Serving Landscape** — A reference project for building, comparing, and benchmarking modern LLM serving frameworks directly, then defining production architecture criteria from an operator's perspective.

> The real value of this repository is not the benchmark numbers themselves. It is being able to explain, from an operator's perspective, **"why some companies choose vLLM, others choose TensorRT-LLM, and where Ray fits."**

## Comparison Targets

| Category | Framework |
|---|---|
| Core Serving Engine | vLLM, SGLang, HuggingFace TGI, TensorRT-LLM |
| Serving Platform | Ray Serve (Phase 4) |

## Hardware Profiles

Two environments are supported. Select one from `configs/profiles/`.

| Profile | Target GPU | Memory | Default Model | Notes |
|---|---|---|---|---|
| `homelab` | RTX 5080 | 16GB | Qwen3-8B-FP8 | Single GPU, conservative 4K context |
| `datacenter` | A100 / H100 | 40~80GB+ | Llama 3.1 8B → 70B | Multi-GPU TP benchmarks |

```bash
export LAB_PROFILE=homelab      # or datacenter
source configs/profiles/${LAB_PROFILE}.env
```

> **RTX 5080 (Blackwell, sm_120) warning**: TensorRT-LLM and some frameworks require recent CUDA 12.8+ builds with compatible kernels. See the compatibility notes in `engines/*/README.md`. A 70B model is not viable with 16GB VRAM, so the homelab profile is limited to 8B-class models plus quantization.

## Directory Layout

```
llm-serving-lab/
├── configs/profiles/        # homelab.env / datacenter.env
├── engines/                 # Docker Compose + README per engine
│   ├── vllm/
│   ├── sglang/
│   ├── tgi/
│   ├── tensorrt-llm/
│   └── ray/
├── scripts/bench/           # Benchmark runners (TTFT/throughput/...)
├── monitoring/              # Prometheus/Grafana/DCGM plus OpenLIT notes
├── benchmark/               # Result documents per engine
├── results/                 # Raw JSON results (partially gitignored)
└── docs/                    # Landscape map and production report notes
```

## Phase Roadmap

| Phase | Scope | Output |
|---|---|---|
| 0 | Environment Setup | Docker / NVIDIA Toolkit / monitoring stack |
| 1 | Single GPU Baseline | `benchmark/*.md` (4 engines) |
| 2 | Advanced Features | Feature Matrix validation |
| 3 | Multi GPU Benchmark | TP 1/2/4 scaling analysis |
| 4 | Production Serving (Ray) | Autoscaling / Routing / Canary |
| 5 | Kubernetes | Out of scope for this repository; Compose-focused |
| 6 | Production Report | `docs/README.md` |

> This scaffolding is **Docker Compose-focused**. Phase 5 (K8s/Helm/ArgoCD) is intentionally recommended as a separate repository.

## Quick Start

```bash
# 1. Select a profile
export LAB_PROFILE=homelab
source configs/profiles/${LAB_PROFILE}.env

# Optional: replace the pinned default image tags with your own tested tags or digests
# export VLLM_IMAGE=vllm/vllm-openai:<tag>

# 2. Start the monitoring stack
just mon-up

# 3. Start an engine, for example vLLM
just up vllm

# 4. Run the benchmark
just bench vllm
```

Benchmark tooling is managed with `uv` from `scripts/bench/pyproject.toml`. Run `just bench-sync` if you want to pre-create the local uv environment before the first benchmark.

For LLM/application tracing, OpenLIT can be used alongside the default GPU metrics stack:

```bash
just openlit-up
export OPENLIT_ENABLED=true
just bench vllm
```

To include native vLLM server spans as well as benchmark client spans:

```bash
just openlit-up
export VLLM_OTEL_ARGS="--otlp-traces-endpoint grpc://host.docker.internal:4317"
just down vllm
just up vllm
export OPENLIT_ENABLED=true
just bench vllm
```

## Collected Metrics

TTFT · End-to-end Latency · Tokens/sec · GPU Memory · GPU Utilization · Power Usage · Concurrent Requests

## Monitoring Choices

Use `just mon-up` for infrastructure metrics: Prometheus, Grafana, and DCGM Exporter cover engine `/metrics`, GPU utilization, VRAM, and power. Use `just openlit-up` when you also want OpenTelemetry-native traces from the benchmark client, including request spans, latency breakdowns, and optional prompt/response capture.

OpenLIT is complementary to DCGM, not a replacement for GPU measurement. See `monitoring/README.md` for the full workflow and tradeoffs.

## Reproducibility Notes

The profiles pin default container tags instead of using floating `latest` tags. For published results, record any tag or digest overrides together with the generated JSON output under `results/`.

## License

MIT
