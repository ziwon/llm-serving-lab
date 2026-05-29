# Contributing / Workflow

1. Select a profile: `export LAB_PROFILE=homelab && source configs/profiles/$LAB_PROFILE.env`
2. Keep the profile-pinned image tags or replace them with tested digests for reproducible runs.
3. Start the network and infrastructure monitoring stack: `just mon-up`
4. Optionally start OpenLIT for client traces: `just openlit-up && export OPENLIT_ENABLED=true`
5. Start an engine: `just up vllm`
6. Run the benchmark: `just bench vllm`
7. Record the results in `benchmark/<engine>.md`
8. Repeat for all 4 engines → Phase 3 multi-GPU → Phase 4 Ray
9. Write the final `docs/README.md`

## Engine Addition Rules
- `engines/<name>/docker-compose.yml` must expose the OpenAI-compatible `/v1/chat/completions` endpoint on `SERVED_PORT`.
- This keeps the unified benchmark runner working without changes.
