# Phase 4 — Production Serving with Ray Serve

This phase moves from benchmarking to operations. It covers autoscaling / request routing / failure recovery / canary / rolling update with Ray Serve.

## Topology
```
           Client
              |
        Load Balancer
              |
        Ray Head Node
              |
     +--------+--------+
     |                 |
 Ray Worker       Ray Worker
     |                 |
   vLLM              vLLM
```

## Run

Start a backend engine first, then run the Ray Serve proxy:

```bash
source ../../configs/profiles/${LAB_PROFILE}.env
cd ../vllm && docker compose up -d
cd ../ray && docker compose up -d
curl http://localhost:8001/v1/models
```

`RAY_BACKEND_URL` controls which engine Ray proxies to. The default is `http://vllm:8000` on the shared `llm-serving-lab` Docker network.

## Topics
- **Autoscaling**: based on `num_replicas="auto"` / target ongoing requests.
- **Request Routing**: power-of-two / least-requests.
- **Failure Recovery**: replica health checks + automatic restart.
- **Canary Deployment**: route N% of traffic to a new model version.
- **Rolling Update**: zero-downtime deployment.

## Implementation Notes
- This repo includes a minimal Ray head/worker Compose setup and an OpenAI-compatible proxy deployment in `serve_app.py`.
- vLLM has two paths: wrap it as a deployment on Ray Serve, or use vLLM's own distributed mode with the Ray backend.
- Full production deployment belongs in Phase 5 (K8s + KubeRay), preferably as a separate repository.
