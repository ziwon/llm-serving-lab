# LLM Serving Lab convenience tasks.
# Before use: export LAB_PROFILE=homelab && source configs/profiles/$LAB_PROFILE.env

set dotenv-load := true

# --- Common settings ---
profile := env_var_or_default("LAB_PROFILE", env_var_or_default("PROFILE", "homelab"))
engine := env_var_or_default("ENGINE", "vllm")
url := env_var_or_default("URL", "http://localhost:8000")

# --- Cache / image settings ---
hf_home := env_var_or_default("HF_HOME", "/data/LLM/models/hugging-face")
trtllm_image := env_var_or_default("TENSORRT_LLM_IMAGE", "nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc7")
trt_image := env_var_or_default("TENSORRT_IMAGE", "nvcr.io/nvidia/tensorrt:25.10-py3")
trtllm_dir := "engines/tensorrt-llm"
trtllm_artifact_dir := env_var_or_default("TRTLLM_ARTIFACT_DIR", "/data/LLM/artifacts/llm-serving-lab/tensorrt-llm")
trtllm_qwen_convert := "/app/tensorrt_llm/examples/models/core/qwen/convert_checkpoint.py"

# --- OpenLIT settings ---
openlit_dir := ".cache/openlit"
openlit_repo := "https://github.com/openlit/openlit.git"
openlit_port := env_var_or_default("OPENLIT_PORT", "3001")
openlit_collector_config := "monitoring/openlit/otel-collector-config.yaml"

# --- ONNX / TensorRT lab settings ---
onnx_viewer_port := env_var_or_default("ONNX_VIEWER_PORT", "8081")
trt_onnx_dir := "engines/tensorrt"
resnet50_onnx := "resnet50-v2-7.onnx"
resnet50_url := env_var_or_default("RESNET50_ONNX_URL", "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx")
resnet50_shapes := env_var_or_default("RESNET50_ONNX_SHAPES", "data:1x3x224x224")

default:
    @just --list

# --- General Docker / Compose ---
net:
    docker network inspect llm-serving-lab >/dev/null 2>&1 || docker network create llm-serving-lab

up engine=engine: net
    cd engines/{{engine}} && docker compose up -d

down engine=engine:
    cd engines/{{engine}} && docker compose down

# --- Monitoring / OpenLIT ---
mon-up: net
    docker compose -f monitoring/docker-compose.yml up -d

mon-down:
    docker compose -f monitoring/docker-compose.yml down

openlit-clone:
    mkdir -p .cache
    test -d {{openlit_dir}}/.git || git clone --depth 1 {{openlit_repo}} {{openlit_dir}}

openlit-up: openlit-clone
    cd {{openlit_dir}} && printf 'services:\n  openlit:\n    extra_hosts:\n      - "host.docker.internal:host-gateway"\n    ports: !override\n      - "%s:3000"\n      - "4317:4317"\n      - "4318:4318"\n' "{{openlit_port}}" > docker-compose.local.yml
    cp {{openlit_collector_config}} {{openlit_dir}}/assets/otel-collector-config.yaml
    cd {{openlit_dir}} && docker compose -f docker-compose.yml -f docker-compose.local.yml up -d --force-recreate openlit

openlit-down:
    test ! -d {{openlit_dir}} || cd {{openlit_dir}} && docker compose -f docker-compose.yml -f docker-compose.local.yml down

openlit-logs:
    test -d {{openlit_dir}} && cd {{openlit_dir}} && docker compose -f docker-compose.yml -f docker-compose.local.yml logs -f openlit

openlit-url:
    @echo "OpenLIT UI: http://127.0.0.1:{{openlit_port}}"
    @echo "Default login: user@openlit.io / openlituser"
    @echo "OTLP HTTP endpoint: http://127.0.0.1:4318"

# --- Unified benchmark scripts ---
bench engine=engine url=url model=env_var_or_default("MODEL_ID", ""):
    uv run --project scripts/bench python scripts/bench/run_benchmark.py --engine {{engine}} --base-url {{url}} $([ -n "{{model}}" ] && printf -- '--model %s' "{{model}}") --out results/{{engine}}_{{profile}}.json

features engine=engine model=env_var_or_default("MODEL_ID", "") url=url:
    test -n "{{model}}" || (echo "MODEL_ID is required. Source a profile or pass model=<id>." && exit 1)
    uv run --project scripts/bench python scripts/bench/validate_features.py --base-url {{url}} --model "{{model}}" --engine "{{engine}}"

bench-sync:
    uv sync --project scripts/bench

summarize:
    uv run --project scripts/bench python scripts/bench/summarize_results.py
    uv run --project scripts/bench python scripts/bench/render_charts.py

feature-matrix: summarize

verify:
    just --list >/dev/null
    uv run --project scripts/bench python -m py_compile scripts/bench/run_benchmark.py scripts/bench/validate_features.py scripts/bench/summarize_results.py scripts/bench/render_charts.py
    docker compose -f monitoring/docker-compose.yml config --quiet
    docker compose -f engines/vllm/docker-compose.yml config --quiet
    docker compose -f engines/sglang/docker-compose.yml config --quiet
    docker compose -f engines/tgi/docker-compose.yml config --quiet
    docker compose -f engines/tensorrt-llm/docker-compose.yml config --quiet
    test -d results || mkdir -p results

doctor:
    @echo "profile={{profile}}"
    @echo "HF_HOME={{hf_home}}"
    @test "{{hf_home}}" = "/data/LLM/models/hugging-face" || echo "warning: HF_HOME is not the repo-preferred cache path"
    @command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,power.draw,power.limit --format=csv || echo "nvidia-smi not available"
    @docker --version 2>/dev/null || echo "docker client not available"
    @docker info 2>/dev/null | awk '/Runtimes:|Default Runtime:/ {print}' || echo "docker daemon not reachable"
    @df -h /data /cache . 2>/dev/null || true
    @docker ps --format 'table {{ "{{.Names}}" }}\t{{ "{{.Image}}" }}\t{{ "{{.Ports}}" }}' 2>/dev/null || true
    @ss -ltnp 2>/dev/null | awk 'NR==1 || /:8000|:3000|:3001|:4317|:4318|:9090|:9400/' || true

clean-artifacts:
    rm -rf engines/tensorrt/models engines/tensorrt/engines-out
    rm -rf engines/tensorrt-llm/hf-models engines/tensorrt-llm/checkpoints engines/tensorrt-llm/engines-out engines/tensorrt-llm/results
    find results -type f -name '*.tmp' -delete 2>/dev/null || true

# --- TensorRT-LLM / Qwen3 serving ---
trtllm-dirs:
    mkdir -p {{trtllm_artifact_dir}}/hf-models {{trtllm_artifact_dir}}/checkpoints {{trtllm_artifact_dir}}/engines-out {{trtllm_artifact_dir}}/results

trtllm-qwen3-4b-up: net
    cd engines/tensorrt-llm && MODEL_ID=Qwen/Qwen3-4B MODEL_QUANTIZATION=none TENSORRT_LLM_EXTRA_ARGS="--config /configs/qwen3-4b-homelab.yaml" docker compose up -d

trtllm-qwen3-4b-bench url=url:
    uv run --project scripts/bench python scripts/bench/run_benchmark.py --engine tensorrt-llm --base-url {{url}} --model Qwen/Qwen3-4B --out results/tensorrt-llm_qwen3-4b_{{profile}}.json

trtllm-convert-qwen model_id="Qwen/Qwen3-4B" checkpoint_name="qwen3-4b-fp16-tp1" dtype="float16" tp_size="1" workers="1" extra_args="--load_model_on_cpu": trtllm-dirs
    docker run --rm --gpus all --ipc=host \
      -e HF_TOKEN="${HF_TOKEN:-}" \
      -e HF_HOME=/root/.cache/huggingface \
      -v {{hf_home}}:/root/.cache/huggingface \
      -v {{trtllm_artifact_dir}}:/workspace/tensorrt-llm-lab \
      {{trtllm_image}} bash -lc 'set -euo pipefail; model_dir="{{model_id}}"; if [ ! -e "$model_dir" ]; then safe_name="$(printf "%s" "$model_dir" | tr "/:" "__")"; model_dir="/workspace/tensorrt-llm-lab/hf-models/$safe_name"; if [ ! -f "$model_dir/config.json" ]; then mkdir -p "$model_dir"; huggingface-cli download "{{model_id}}" --local-dir "$model_dir"; fi; fi; rm -rf /workspace/tensorrt-llm-lab/checkpoints/{{checkpoint_name}}; python3 {{trtllm_qwen_convert}} --model_dir "$model_dir" --output_dir /workspace/tensorrt-llm-lab/checkpoints/{{checkpoint_name}} --dtype "{{dtype}}" --tp_size "{{tp_size}}" --workers "{{workers}}" {{extra_args}} | tee /workspace/tensorrt-llm-lab/results/{{checkpoint_name}}.convert.log'

trtllm-build-checkpoint checkpoint_name="qwen3-4b-fp16-tp1" engine_name="qwen3-4b-fp16-tp1-4096" max_seq_len="4096" max_input_len="3584" max_num_tokens="4096" max_batch_size="16" workers="1": trtllm-dirs
    test -d "{{trtllm_artifact_dir}}/checkpoints/{{checkpoint_name}}" || (echo "Missing TensorRT-LLM checkpoint: {{trtllm_artifact_dir}}/checkpoints/{{checkpoint_name}}" && exit 1)
    docker run --rm --gpus all --ipc=host \
      -e HF_TOKEN="${HF_TOKEN:-}" \
      -e HF_HOME=/root/.cache/huggingface \
      -v {{hf_home}}:/root/.cache/huggingface \
      -v {{trtllm_artifact_dir}}:/workspace/tensorrt-llm-lab \
      {{trtllm_image}} bash -lc 'set -euo pipefail; rm -rf /workspace/tensorrt-llm-lab/engines-out/{{engine_name}}; trtllm-build --checkpoint_dir /workspace/tensorrt-llm-lab/checkpoints/{{checkpoint_name}} --output_dir /workspace/tensorrt-llm-lab/engines-out/{{engine_name}} --max_seq_len "{{max_seq_len}}" --max_input_len "{{max_input_len}}" --max_num_tokens "{{max_num_tokens}}" --max_batch_size "{{max_batch_size}}" --workers "{{workers}}" --gpt_attention_plugin auto --gemm_plugin auto --kv_cache_type paged --monitor_memory --output_timing_cache /workspace/tensorrt-llm-lab/results/{{engine_name}}.timing.cache | tee /workspace/tensorrt-llm-lab/results/{{engine_name}}.build.log'

trtllm-build-qwen3-4b-lab:
    test -d "{{trtllm_artifact_dir}}/checkpoints/qwen3-4b-fp16-tp1" || just trtllm-convert-qwen Qwen/Qwen3-4B qwen3-4b-fp16-tp1 float16 1 1 --load_model_on_cpu
    just trtllm-build-checkpoint qwen3-4b-fp16-tp1 qwen3-4b-fp16-tp1-4096 4096 3584 4096 16 1

trtllm-qwen3-4b-engine-up: net
    just trtllm-engine-up qwen3-4b-fp16-tp1-4096 Qwen/Qwen3-4B

trtllm-qwen3-4b-engine-bench url=url:
    uv run --project scripts/bench python scripts/bench/run_benchmark.py --engine tensorrt-llm-engine --base-url {{url}} --model Qwen/Qwen3-4B --out results/tensorrt-llm_engine_qwen3-4b_{{profile}}.json

trtllm-build checkpoint_dir output_dir="qwen3-4b-rtx5080" max_seq_len="4096" max_num_tokens="4096" max_batch_size="16":
    test -d "{{checkpoint_dir}}" || (echo "checkpoint_dir must be a TensorRT-LLM checkpoint directory visible on the host, not a raw HF snapshot: {{checkpoint_dir}}" && exit 1)
    mkdir -p {{trtllm_artifact_dir}}/engines-out
    docker run --rm --gpus all --ipc=host \
      -v {{hf_home}}:/root/.cache/huggingface \
      -v "$PWD/{{checkpoint_dir}}:/checkpoint:ro" \
      -v {{trtllm_artifact_dir}}/engines-out:/engines \
      {{trtllm_image}} bash -lc 'rm -rf /engines/{{output_dir}}; trtllm-build --checkpoint_dir /checkpoint --output_dir /engines/{{output_dir}} --max_seq_len {{max_seq_len}} --max_num_tokens {{max_num_tokens}} --max_batch_size {{max_batch_size}} --gpt_attention_plugin auto --gemm_plugin auto --monitor_memory'

trtllm-engine-up engine_dir="qwen3-4b-rtx5080" tokenizer="Qwen/Qwen3-4B": net
    test -d "{{trtllm_artifact_dir}}/engines-out/{{engine_dir}}" || (echo "Missing built engine: {{trtllm_artifact_dir}}/engines-out/{{engine_dir}}" && exit 1)
    cd engines/tensorrt-llm && TRTLLM_ARTIFACT_DIR="{{trtllm_artifact_dir}}" MODEL_ID=/engines/{{engine_dir}} TENSORRT_LLM_EXTRA_ARGS="--tokenizer {{tokenizer}} --backend tensorrt --max_batch_size 16 --max_num_tokens 4096 --max_seq_len 4096" docker compose up -d

# --- ONNX viewer ---
onnx-view model="" port=onnx_viewer_port:
    test -n "{{model}}" || (echo "Usage: just onnx-view path/to/model.onnx [port]" && exit 1)
    test -f "{{model}}" || (echo "ONNX model not found: {{model}}" && exit 1)
    @echo "Netron ONNX viewer: http://127.0.0.1:{{port}}"
    uvx --from netron netron "{{model}}" --host 0.0.0.0 --port {{port}}

onnx-view-url port=onnx_viewer_port:
    @echo "Netron ONNX viewer: http://127.0.0.1:{{port}}"

# --- ONNX -> TensorRT lab ---
trt-onnx-dirs:
    mkdir -p {{trt_onnx_dir}}/models {{trt_onnx_dir}}/engines-out {{trt_onnx_dir}}/results

trt-onnx-download-resnet50: trt-onnx-dirs
    test -f {{trt_onnx_dir}}/models/{{resnet50_onnx}} || curl -L "{{resnet50_url}}" -o {{trt_onnx_dir}}/models/{{resnet50_onnx}}

trt-onnx-view-resnet50: trt-onnx-download-resnet50
    just onnx-view {{trt_onnx_dir}}/models/{{resnet50_onnx}}

trt-onnx-build model engine_name="model-fp16.plan" shapes="" precision="fp16": trt-onnx-dirs
    test -f "{{model}}" || (echo "ONNX model not found: {{model}}" && exit 1)
    docker run --rm --gpus all --ipc=host \
      -v "$PWD/{{trt_onnx_dir}}:/workspace" \
      -v "$PWD/{{model}}:/model.onnx:ro" \
      {{trt_image}} bash -lc 'trtexec --onnx=/model.onnx --saveEngine=/workspace/engines-out/{{engine_name}} $([ "{{precision}}" = "fp16" ] && echo --fp16) $([ -n "{{shapes}}" ] && echo --shapes={{shapes}}) --dumpProfile --separateProfileRun | tee /workspace/results/{{engine_name}}.build.log'

trt-onnx-build-resnet50: trt-onnx-download-resnet50
    just trt-onnx-build {{trt_onnx_dir}}/models/{{resnet50_onnx}} resnet50-v2-7-fp16.plan "{{resnet50_shapes}}" fp16

trt-onnx-bench engine_name="resnet50-v2-7-fp16.plan" shapes=resnet50_shapes:
    test -f "{{trt_onnx_dir}}/engines-out/{{engine_name}}" || (echo "TensorRT engine not found: {{trt_onnx_dir}}/engines-out/{{engine_name}}" && exit 1)
    docker run --rm --gpus all --ipc=host \
      -v "$PWD/{{trt_onnx_dir}}:/workspace" \
      {{trt_image}} bash -lc 'trtexec --loadEngine=/workspace/engines-out/{{engine_name}} $([ -n "{{shapes}}" ] && echo --shapes={{shapes}}) --iterations=1000 --warmUp=200 | tee /workspace/results/{{engine_name}}.bench.log'

trt-onnx-lab-resnet50: trt-onnx-build-resnet50
    just trt-onnx-bench resnet50-v2-7-fp16.plan "{{resnet50_shapes}}"

# --- Environment helpers ---
profile profile=profile:
    @echo "source configs/profiles/{{profile}}.env"
