# AGENTS.md

## Repository Guidance

- This repository benchmarks LLM serving engines. Keep generated benchmark outputs and engine runtime artifacts out of the root filesystem when possible.
- For Hugging Face downloads and caches, set `HF_HOME` to `/data/LLM/models/hugging-face`.
- When editing Docker Compose files, benchmark scripts, `.env.example`, profile env files, or documentation that references Hugging Face cache paths, prefer:

```bash
HF_HOME=/data/LLM/models/hugging-face
```

- Do not introduce new defaults that point Hugging Face model/cache data at `~/.cache/huggingface` unless the user explicitly asks for a local home-directory cache.
- Generated benchmark results should stay under `results/`, and temporary benchmark/runtime artifacts should be cleaned up after SGLang and TGI runs if they are not part of the intended recorded results.
