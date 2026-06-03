# Feature Matrix

Generated from `results/features_*.json`.

| Engine | Model | Profile | Streaming | OpenAI Models API | Concurrent Requests | Prefix Cache Smoke | JSON Mode | Tool Calling | Speculative Decoding |
|---|---|---|---|---|---|---|---|---|---|
| `sglang` | `Qwen/Qwen3-8B-FP8` | `homelab` | yes | yes | yes | yes | yes | no | manual |
| `tensorrt-llm-engine` | `Qwen/Qwen3-4B` | `homelab` | yes | yes | yes | yes | no | no | manual |
| `vllm` | `Qwen/Qwen3-8B-FP8` | `homelab` | yes | yes | yes | yes | yes | yes | manual |

## Details

### sglang / Qwen/Qwen3-8B-FP8 / homelab

Source: `results/features_sglang_homelab.json`

| Feature | Supported | Detail |
|---|---|---|
| Streaming | yes |  |
| OpenAI Models API | yes | HTTP 200 |
| Concurrent Requests | yes | max_latency_s=0.48 |
| Prefix Cache Smoke | yes | first_s=0.60, second_s=0.57 |
| JSON Mode | yes | {     "city": "Seoul" } |
| Tool Calling | no | null |
| Speculative Decoding | manual | manual; requires engine-specific startup config/log verification |

### tensorrt-llm-engine / Qwen/Qwen3-4B / homelab

Source: `results/features_tensorrt-llm-engine_homelab.json`

| Feature | Supported | Detail |
|---|---|---|
| Streaming | yes |  |
| OpenAI Models API | yes | HTTP 200 |
| Concurrent Requests | yes | max_latency_s=0.42 |
| Prefix Cache Smoke | yes | first_s=0.41, second_s=0.37 |
| JSON Mode | no | HTTP 400 |
| Tool Calling | no | HTTP 400 |
| Speculative Decoding | manual | manual; requires engine-specific startup config/log verification |

### vllm / Qwen/Qwen3-8B-FP8 / homelab

Source: `results/features_vllm_homelab.json`

| Feature | Supported | Detail |
|---|---|---|
| Streaming | yes |  |
| OpenAI Models API | yes | HTTP 200 |
| Concurrent Requests | yes | max_latency_s=0.38 |
| Prefix Cache Smoke | yes | first_s=0.39, second_s=0.36 |
| JSON Mode | yes | {     "city": "Seoul" } |
| Tool Calling | yes | [{"id": "chatcmpl-tool-827252f7f6a5b009", "type": "function", "function": {"name |
| Speculative Decoding | manual | manual; requires engine-specific startup config/log verification |
