# Monitoring Options

This repository supports two complementary monitoring modes.

## Infrastructure Metrics

Use the default local monitoring stack for engine and GPU metrics:

```bash
just mon-up
```

This starts Prometheus, Grafana, and DCGM Exporter. Use this path for:

- GPU utilization, VRAM, and power
- `/metrics` endpoints exposed by serving engines
- long-running benchmark resource curves

Grafana is exposed on `http://localhost:3000` by default. Prometheus is exposed on `http://localhost:9090`.

Stop the stack with:

```bash
just mon-down
```

## LLM Observability with OpenLIT

OpenLIT is useful when you want OpenTelemetry-native traces and LLM/application-level observability from benchmark traffic. The recipe maps OpenLIT's UI to port `3001` by default to avoid colliding with Grafana on `3000`.

Start it with:

```bash
just openlit-up
just openlit-url
```

The OpenLIT recipe clones the upstream `openlit/openlit` repository into `.cache/openlit` and runs its Docker Compose stack from there. `.cache/` is gitignored, so the upstream stack is not vendored into this repository.

Then enable OpenLIT instrumentation for the benchmark tools:

```bash
export OPENLIT_ENABLED=true
export OPENLIT_OTLP_ENDPOINT=http://127.0.0.1:4318
just bench vllm
```

To also export native vLLM server traces to OpenLIT, start OpenLIT first, enable
vLLM's OTLP flag, and recreate vLLM:

```bash
just openlit-up
export VLLM_OTEL_ARGS="--otlp-traces-endpoint grpc://host.docker.internal:4317"
just down vllm
just up vllm
```

Use both layers when you want end-to-end request visibility:

```bash
export OPENLIT_ENABLED=true
just bench vllm
```

The benchmark process sends client spans to `http://127.0.0.1:4318`. The vLLM
container sends server spans to `grpc://host.docker.internal:4317`, which reaches
OpenLIT's OTLP gRPC receiver through Docker's host gateway.

OpenLIT's default UI is:

```text
http://127.0.0.1:3001
user@openlit.io / openlituser
```

Stop OpenLIT with:

```bash
just openlit-down
```

## OpenLIT Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `OPENLIT_ENABLED` | `false` | Enables OpenLIT instrumentation in benchmark scripts |
| `OPENLIT_PORT` | `3001` | Host port for the OpenLIT UI |
| `OPENLIT_OTLP_ENDPOINT` | `http://127.0.0.1:4318` | OTLP HTTP endpoint used by the benchmark scripts |
| `OPENLIT_APPLICATION_NAME` | `llm-serving-lab` | Application name shown in OpenLIT |
| `OPENLIT_CAPTURE_MESSAGE_CONTENT` | `false` | Captures prompt/response content when set to `true` |
| `VLLM_OTEL_ARGS` | empty | Optional vLLM server trace flag, for example `--otlp-traces-endpoint grpc://host.docker.internal:4317` |
| `VLLM_OTEL_SERVICE_NAME` | `vllm-server` | OpenTelemetry service name for native vLLM spans |
| `VLLM_OTEL_PROTOCOL` | `grpc` | OTLP trace exporter protocol for vLLM |
| `VLLM_OTEL_INSECURE` | `true` | Allows insecure local OTLP gRPC transport |

Keep `OPENLIT_CAPTURE_MESSAGE_CONTENT=false` for general benchmark runs. Enable it only when you explicitly need payload inspection because prompts and completions can contain sensitive data.

## What to Use When

OpenLIT is not a direct replacement for DCGM when the question is GPU efficiency. Treat it as the LLM trace and application observability layer, and keep Prometheus/DCGM for infrastructure measurements.

| Need | Recommended path |
|---|---|
| GPU power, VRAM, utilization | `just mon-up` |
| Engine `/metrics` scraping | `just mon-up` |
| Per-request client traces | `just openlit-up` + `OPENLIT_ENABLED=true` |
| Native vLLM server traces | `VLLM_OTEL_ARGS="--otlp-traces-endpoint grpc://host.docker.internal:4317"` + `just up vllm` |
| Prompt/response inspection | OpenLIT with `OPENLIT_CAPTURE_MESSAGE_CONTENT=true` |
| Published benchmark tables | Benchmark JSON + `benchmark/*.md` |

## Alternatives

| Tool | Best fit |
|---|---|
| OpenLIT | OpenTelemetry-native LLM traces, evaluations, guardrails, and AI engineering workflows |
| Arize Phoenix | LLM tracing, debugging, and evaluations with OpenTelemetry/OpenInference |
| Langfuse | Self-hostable LLM engineering platform with observability, prompts, evals, and experiments |
| Existing Prometheus/Grafana/DCGM | GPU and serving-runtime metrics |
