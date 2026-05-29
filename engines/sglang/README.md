# SGLang

Strong prefix caching based on RadixAttention. SGLang is well-suited for structured output, tool calling, and agent workflows. It is the first-choice engine for the Agent Workflow scenario.

## Run
```bash
source ../../configs/profiles/${LAB_PROFILE}.env
docker compose up -d
curl http://localhost:${SERVED_PORT:-8000}/v1/models
```

## RTX 5080 Notes
- Confirm that the image tag supports Blackwell (CUDA 12.8+).
- Port mapping: container port 30000 -> host `SERVED_PORT`.
- Use `--mem-fraction-static` to tune the KV cache share. If 16GB OOMs, try lowering it to 0.85.

## Benchmark
```bash
cd ../..
uv run --project scripts/bench python scripts/bench/run_benchmark.py --engine sglang --base-url http://localhost:${SERVED_PORT:-8000}
```
