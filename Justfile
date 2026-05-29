# LLM Serving Lab convenience tasks.
# Before use: export LAB_PROFILE=homelab && source configs/profiles/$LAB_PROFILE.env

set dotenv-load := true

profile := env_var_or_default("PROFILE", "homelab")
engine := env_var_or_default("ENGINE", "vllm")
url := env_var_or_default("URL", "http://localhost:8000")
openlit_dir := ".cache/openlit"
openlit_repo := "https://github.com/openlit/openlit.git"
openlit_port := env_var_or_default("OPENLIT_PORT", "3001")
openlit_collector_config := "monitoring/openlit/otel-collector-config.yaml"

default:
    @just --list

net:
    docker network inspect llm-serving-lab >/dev/null 2>&1 || docker network create llm-serving-lab

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

up engine=engine: net
    cd engines/{{engine}} && docker compose up -d

down engine=engine:
    cd engines/{{engine}} && docker compose down

bench engine=engine url=url:
    uv run --project scripts/bench python scripts/bench/run_benchmark.py --engine {{engine}} --base-url {{url}}

features model=env_var_or_default("MODEL_ID", "") url=url:
    test -n "{{model}}" || (echo "MODEL_ID is required. Source a profile or pass model=<id>." && exit 1)
    uv run --project scripts/bench python scripts/bench/validate_features.py --base-url {{url}} --model "{{model}}"

bench-sync:
    uv sync --project scripts/bench

profile profile=profile:
    @echo "source configs/profiles/{{profile}}.env"
