# vLLM

The most widely used OSS serving engine. Its strengths are PagedAttention-based continuous batching and an OpenAI-compatible API. It is the first-choice engine for Fast MVP / OpenAI Compatible API scenarios.

## Run
```bash
source ../../configs/profiles/${LAB_PROFILE}.env
docker compose up -d
curl http://localhost:${SERVED_PORT:-8000}/v1/models
```

## RTX 5080 Notes
- Requires a vLLM build with Blackwell (sm_120) support plus CUDA 12.8+. If the pinned profile tag is not compatible with your driver, check nightly or hardware-specific tags.
- On 16GB, start with `Qwen/Qwen3-8B-FP8`, `--max-model-len 4096`, `--gpu-memory-utilization 0.82`, and `--max-num-seqs 32`. Desktop display usage can leave less than 14GB free even on a 16GB card.
- `--enable-prefix-caching` is enabled for Phase 2 prefix cache validation.

## Memory Startup Error

If vLLM exits with:

```text
Free memory on device cuda:0 ... is less than desired GPU memory utilization
```

the GPU already has less free memory than vLLM is allowed to reserve. Check other GPU processes:

```bash
nvidia-smi
```

Then either stop other GPU users or lower the profile settings before restarting:

```bash
export GPU_MEMORY_UTILIZATION=0.78
export MAX_MODEL_LEN=4096
export MAX_NUM_SEQS=16
just down vllm
just up vllm
```

## Benchmark
```bash
cd ../..
uv run --project scripts/bench python scripts/bench/run_benchmark.py --engine vllm --base-url http://localhost:${SERVED_PORT:-8000}
```

## OpenLIT Tracing

vLLM can export native OpenTelemetry server spans to OpenLIT. Start OpenLIT,
enable the vLLM OTLP endpoint flag, then recreate vLLM:

```bash
cd ../..
just openlit-up
export VLLM_OTEL_ARGS="--otlp-traces-endpoint grpc://host.docker.internal:4317"
just down vllm
just up vllm
```

For end-to-end visibility, also enable benchmark client instrumentation before
running traffic:

```bash
export OPENLIT_ENABLED=true
export OPENLIT_OTLP_ENDPOINT=http://127.0.0.1:4318
just bench vllm
```

The vLLM container reaches OpenLIT's OTLP gRPC receiver through
`host.docker.internal:4317`. The benchmark process uses OpenLIT's OTLP HTTP
receiver on `127.0.0.1:4318`.
