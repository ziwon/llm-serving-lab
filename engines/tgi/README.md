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
- Port: container 80 -> host `SERVED_PORT`.
- `--max-input-tokens` must be smaller than `--max-total-tokens`.
- The OpenAI-compatible endpoint is available at `/v1/chat/completions`.
- Additional launcher flags can be passed with `TGI_EXTRA_ARGS`.
- `Qwen/Qwen3-8B-FP8` failed TGI warmup on RTX 5080 with a Transformers fine-grained FP8 Triton `PassManager::run failed` error. `TGI_EXTRA_ARGS="--disable-custom-kernels"` did not resolve it in the homelab attempt.

## Benchmark
```bash
cd ../..
uv run --project scripts/bench python scripts/bench/run_benchmark.py --engine tgi --base-url http://localhost:${SERVED_PORT:-8000}
```
