# Ray Cluster에서 Triton Inference Server까지: 확장 가능한 AI Inference Serving Architecture

AI 모델을 실제 서비스에 배포할 때 가장 어려운 부분은 단순히 모델을 실행하는 것이 아니다.
진짜 어려운 문제는 **요청을 안정적으로 받고, GPU 자원을 효율적으로 배치하며, 모델 실행을 빠르게 최적화하고, 운영 가능한 형태로 확장하는 것**이다.

이 문서는 `Ray Cluster → Triton Inference Server` 구조를 기준으로, 대규모 AI inference platform이 어떤 계층으로 구성되는지 설명한다.

![Ray Cluster to Triton Inference Server architecture](../assets/images/ray.png)

전체 구조는 control plane과 data plane을 분리해서 이해해야 한다.

```text
Control plane
  Ray Head Node / GCS / Serve Controller / Autoscaler
      ↓ places and supervises
  Ray Serve replicas and worker actors

Data plane
  Client Request
      ↓
  Ray Serve HTTP/gRPC Proxy
      ↓
  Router / DeploymentHandle
      ↓
  Ray Serve Replica Actor
      ↓
  Triton Inference Server / Model Pipeline
      ↓
  Model Runtime
      ↓
  GPU Cluster
```

같은 내용을 Mermaid로 표현하면 다음과 같다.

```mermaid
flowchart TB
  subgraph CP["Control plane"]
    Head["Ray Head Node / GCS"]
    Controller["Serve Controller"]
    Autoscaler["Ray Autoscaler / KubeRay"]
    Repo["Model Repository"]
    Obs["Observability"]
  end

  subgraph DP["Data plane"]
    Client["Client Request"]
    Proxy["Ray Serve HTTP/gRPC Proxy"]
    Router["Router / Deployment Handle"]
    Replica["Ray Serve Replica Actor"]
    Triton["Triton Inference Server"]
    Runtime["TensorRT / ONNX / PyTorch / LLM Runtime"]
    GPU["GPU Cluster"]
  end

  Client --> Proxy --> Router --> Replica --> Triton --> Runtime --> GPU
  Head -.->|cluster state| Controller
  Controller -.->|routing config| Router
  Autoscaler -.->|replica and node scale| Replica
  Repo -.->|model versions| Triton
  Obs -.->|metrics and health| Proxy
  Obs -.->|metrics and health| Triton

  classDef data fill:#eff6ff,stroke:#2563eb,color:#0f172a;
  classDef control fill:#faf5ff,stroke:#7c3aed,color:#0f172a;
  class Client,Proxy,Router,Replica,Triton,Runtime,GPU data;
  class Head,Controller,Autoscaler,Repo,Obs control;
```

핵심 메시지는 단순하다.

> Ray handles placement, routing, and scale.
> Triton optimizes execution.
> Together they form the serving plane.

즉, **Ray는 모델 serving workload를 어디에, 얼마나, 어떤 방식으로 배치하고 라우팅할지 결정하고**, **Triton은 실제 모델 실행을 빠르게 처리하는 역할**을 담당한다.

---

## 1. 왜 Ray와 Triton을 함께 사용하는가?

AI inference system은 크게 두 가지 문제를 동시에 해결해야 한다.

첫 번째는 **serving orchestration** 문제다.

요청이 들어왔을 때 어떤 replica가 처리할 것인지, GPU가 부족하면 어떻게 확장할 것인지, 여러 모델 또는 여러 버전 사이에서 traffic을 어떻게 나눌 것인지 결정해야 한다.

두 번째는 **model execution optimization** 문제다.

GPU 위에서 모델을 어떻게 빠르게 실행할 것인지, batching을 어떻게 할 것인지, TensorRT/ONNX/PyTorch/vLLM 같은 runtime을 어떻게 사용할 것인지 결정해야 한다.

Ray와 Triton은 이 두 영역을 나누어 담당한다.

| Layer                   | Main Responsibility                                                                |
| ----------------------- | ---------------------------------------------------------------------------------- |
| Ray Serve               | HTTP/gRPC proxying, request routing, autoscaling, traffic splitting, replica management |
| Ray Core                | Actor/task placement, worker execution, cluster resource management                |
| Triton Inference Server | Model serving, dynamic batching, concurrent execution, model repository management |
| Model Runtime           | TensorRT, ONNX Runtime, PyTorch, LLM runtime integration                           |
| GPU Cluster             | Actual accelerated compute infrastructure                                          |

Ray만 사용하면 분산 serving과 autoscaling은 강력하지만, 모델 실행 최적화 측면에서는 별도의 runtime 전략이 필요하다.

반대로 Triton만 사용하면 모델 실행은 강력하지만, 복잡한 분산 orchestration, multi-service routing, autoscaling policy, actor 기반 pipeline 구성은 Ray가 더 자연스럽게 처리할 수 있다.

그래서 둘을 함께 사용하면 다음과 같은 구조가 된다.

```text
Ray = where, who receives traffic, and how many
Triton = how fast
```

---

## 2. Client Layer: Inference Request

가장 상단에는 client request가 있다.

```text
HTTP / gRPC endpoint for prompts, tokens, or embeddings payloads.
```

Client는 일반적으로 REST 또는 gRPC endpoint를 통해 inference request를 보낸다.

요청 payload는 workload 유형에 따라 달라진다.

예를 들어 일반적인 deep learning inference라면 image, tensor, feature vector가 들어올 수 있다.
LLM serving이라면 prompt, input tokens, chat messages, embedding request가 들어올 수 있다.

이 계층에서 중요한 것은 protocol 자체보다 **요청을 serving layer로 안정적으로 전달하는 것**이다.

대표적인 고려사항은 다음과 같다.

| Concern        | Description                         |
| -------------- | ----------------------------------- |
| Protocol       | REST, gRPC                          |
| Payload        | Prompt, tokens, embeddings, tensors |
| Authentication | API key, JWT, internal service auth |
| Rate limit     | Tenant별 request 제한                  |
| Timeout        | Long-running generation request 처리  |
| Streaming      | LLM token streaming 지원 여부           |

Client request는 이후 Ray Serve frontend로 전달된다.

---

## 3. Ray Serve Layer: Frontend Serving Plane

Ray Serve는 Ray 위에서 동작하는 model serving framework다.

그림에서는 다음과 같은 역할로 표현된다.

```text
Ray Serve (Frontend)
Autoscaling replicas. Traffic splitting. Load balancing across replicas.
```

Ray Serve는 외부 요청을 받아 내부 replica로 전달한다. 이때 Ray Serve는 단일 컴포넌트라기보다 proxy, controller, router, replica actor로 나뉜다.

주요 기능은 다음과 같다.

| Feature               | Description                         |
| --------------------- | ----------------------------------- |
| Autoscaling           | 요청량에 따라 replica 수를 자동 조절            |
| Traffic Splitting     | 모델 버전 또는 deployment 간 traffic 분배    |
| Load Balancing        | 여러 replica 사이에 요청 분산                |
| Deployment Graph      | 여러 deployment를 조합한 serving graph 구성 |
| Python-native Serving | Python 코드 기반으로 inference service 구성 |

운영 관점에서는 다음 컴포넌트 구분이 중요하다.

| Component | Plane | Responsibility |
|---|---|---|
| HTTP/gRPC Proxy | Data plane | 외부 요청을 받고 Serve application으로 진입시킴 |
| Serve Controller | Control plane | Deployment config, routing policy, replica lifecycle 관리 |
| Router / DeploymentHandle | Data plane | 요청을 적절한 deployment replica로 전달 |
| Replica Actor | Data plane | 사용자 코드 실행, preprocessing, Triton dispatch, postprocessing |
| Ray Autoscaler / KubeRay | Control plane | replica 배치에 필요한 node와 resource 확장 |

예를 들어 하나의 LLM service를 운영한다고 가정해보자.

```text
/llm/generate
/embedding
/rerank
/classify
```

이런 endpoint들을 Ray Serve deployment로 나누고, 각 deployment가 내부적으로 Triton endpoint를 호출하도록 구성할 수 있다.

Ray Serve의 장점은 단순히 모델을 실행하는 것이 아니라, **서비스 단위의 inference application을 구성하기 좋다**는 점이다.

예를 들어 다음과 같은 작업을 Ray Serve 계층에서 처리할 수 있다.

```text
Request validation
Prompt preprocessing
Tenant-based routing
A/B test routing
Fallback model selection
Model pipeline orchestration
Triton endpoint dispatch
Response post-processing
```

즉, Ray Serve는 inference system의 frontend이자 application-level serving plane 역할을 한다. 다만 실제 request data path는 `proxy → router/handle → replica actor`로 흐르고, Serve Controller와 Ray Head Node는 그 경로를 설정하고 감시하는 control plane에 가깝다.

---

## 4. Ray Core Layer: Head Node와 Worker Node

Ray Serve 아래에는 Ray Core가 있다.

Ray Core는 Ray cluster의 분산 실행 기반이다.

그림에서는 두 가지 구성 요소로 나뉜다.

```text
Head Node
Worker Node xN
```

---

### 4.1 Head Node

Head Node는 Ray cluster의 중심 control node다.

그림에서는 다음과 같이 설명된다.

```text
GCS, scheduler, dashboard.
Tracks cluster state and resource availability.
```

Head Node는 cluster 상태를 관리하고, scheduler를 통해 task와 actor를 적절한 worker node에 배치한다.

주요 역할은 다음과 같다.

| Component        | Role                                        |
| ---------------- | ------------------------------------------- |
| GCS              | Global Control Store. Cluster metadata 관리   |
| Scheduler        | Task/actor placement 결정                     |
| Dashboard        | Cluster 상태와 workload 모니터링                   |
| Resource Tracker | CPU, GPU, memory 등 resource availability 추적 |

Head Node는 실제 GPU computation이나 inference request forwarding을 담당한다기보다는, **cluster 전체의 control plane** 역할을 한다.

---

### 4.2 Worker Node

Worker Node는 실제 workload가 실행되는 node다.

그림에서는 다음과 같이 표현된다.

```text
Runs remote actors and tasks.
Each node hosts GPU resources assigned by the scheduler.
```

Ray에서 workload는 주로 task 또는 actor 형태로 실행된다.

| Concept      | Description                              |
| ------------ | ---------------------------------------- |
| Task         | Stateless remote function execution      |
| Actor        | Stateful worker process                  |
| GPU Resource | `num_gpus` 기반으로 Ray scheduler가 할당        |
| Placement    | 특정 node, 특정 resource constraint 기반 배치 가능 |

Inference serving에서는 일반적으로 model replica 또는 pipeline stage가 actor 형태로 동작할 수 있다.

예를 들어 Ray Serve replica가 worker node 위에서 실행되고, 해당 replica가 local Triton server 또는 remote Triton endpoint로 요청을 넘기는 구조를 만들 수 있다.

---

## 5. Ray에서 Triton으로: Model Server Dispatch

Ray Serve replica가 선택된 이후 다음 단계는 model server로 요청을 넘기는 것이다.

정확한 data path는 다음과 같이 보는 것이 안전하다.

```text
Client
  → Ray Serve Proxy
  → Router / DeploymentHandle
  → Ray Serve Replica Actor on Worker Node
  → Triton Inference Server or Model Pipeline
```

Head Node, GCS, Serve Controller는 replica placement, routing configuration, lifecycle management를 담당하지만, 일반적인 inference request payload를 Triton으로 직접 forward하지 않는다.

여기서 중요한 점은 Triton을 배치하는 방식이 하나로 고정되어 있지 않다는 것이다.

대표적으로 세 가지 방식이 가능하다.

```mermaid
flowchart LR
  subgraph Centralized["Centralized Triton Pool"]
    CReplica["Ray Serve Replica"]
    CPool["Triton Server Pool"]
    CGPU["GPU Pool"]
    CReplica -->|"HTTP/gRPC network call"| CPool --> CGPU
  end

  subgraph PerNode["Per-node Triton"]
    PReplica["Replica on GPU Worker"]
    PClient["Local Triton Client"]
    PTriton["Triton on Same Node"]
    PGPU["Local GPU"]
    PReplica --> PClient -->|"localhost / node-local"| PTriton --> PGPU
  end

  subgraph Embedded["Embedded Pipeline Style"]
    Actor["Ray Actor"]
    Pre["Preprocess / Route"]
    Call["Triton Call"]
    Post["Postprocess"]
    Actor --> Pre --> Call --> Post
  end

  classDef ray fill:#ecfdf5,stroke:#16a34a,color:#0f172a;
  classDef triton fill:#fef2f2,stroke:#dc2626,color:#0f172a;
  classDef gpu fill:#111827,stroke:#334155,color:#ffffff;
  class CReplica,PReplica,PClient,Actor,Pre,Post ray;
  class CPool,PTriton,Call triton;
  class CGPU,PGPU gpu;
```

---

### 5.1 Centralized Triton Server

하나 또는 여러 개의 Triton server를 독립적인 serving pool로 운영하고, Ray Serve replica가 network call을 통해 Triton에 요청을 보낸다.

```text
Ray Serve Replica
    ↓ HTTP/gRPC
Triton Inference Server
    ↓
GPU
```

이 방식은 구조가 명확하고 운영이 단순하다.

다만 Ray worker와 Triton server 사이의 network hop이 추가될 수 있다.

---

### 5.2 Per-node Triton Server

각 GPU worker node마다 Triton server를 배치하고, Ray actor가 같은 node의 Triton server로 요청을 보낸다.

```text
Ray Worker Node
  ├── Ray Serve Replica
  └── Triton Server
```

이 방식은 data locality를 활용하기 좋다.

Ray actor와 Triton server가 같은 node에 있다면 network overhead를 줄일 수 있고, GPU affinity를 명확하게 관리할 수 있다.

---

### 5.3 Embedded Pipeline Style

Ray actor가 preprocessing, routing, postprocessing을 담당하고, Triton은 순수한 model execution backend처럼 사용된다.

```text
Ray Actor
  ├── preprocessing
  ├── Triton inference call
  └── postprocessing
```

이 방식은 복잡한 business logic이나 multi-model pipeline을 구성하기 좋다.

---

## 6. Triton Inference Server Layer

Triton Inference Server는 NVIDIA에서 제공하는 고성능 inference server다.

그림에서는 다음과 같은 기능을 담당한다.

```text
Dynamic batching.
Concurrent model execution.
Model repository management.
Per-model configuration.
```

Triton의 핵심 역할은 모델 실행을 production-grade serving 형태로 제공하는 것이다.

주요 기능은 다음과 같다.

| Feature              | Description                          |
| -------------------- | ------------------------------------ |
| Dynamic Batching     | Stateless model request를 micro-batch로 처리 |
| Concurrent Execution | 여러 model instance 동시 실행              |
| Model Repository     | 모델 파일과 config 관리                     |
| Model Versioning     | 모델 버전 관리                             |
| Multiple Backends    | TensorRT, ONNX Runtime, PyTorch 등 지원 |
| Metrics              | Prometheus metrics 제공                |
| HTTP/gRPC API        | 표준 inference API 제공                  |

Triton의 강점은 **GPU utilization을 높이기 위한 inference execution optimization**에 있다.

특히 작은 stateless request가 자주 들어오는 vision, encoder, ranking, classical tensor workload에서는 dynamic batching이 중요하다.

예를 들어 request가 하나씩 들어오면 GPU utilization이 낮을 수 있다.
Triton은 짧은 시간 window 안에 들어온 request를 묶어서 batch inference로 실행할 수 있다.

```text
Request 1 ┐
Request 2 ├── Dynamic Batch ── GPU Execution
Request 3 ┘
```

이 방식은 latency와 throughput 사이의 trade-off를 조절하는 데 중요하다.

다만 LLM serving에서는 이 설명만으로는 부족하다. Autoregressive generation은 prefill과 decode 단계, KV cache, token streaming, request별 sequence state를 함께 다루기 때문에 일반적인 Triton dynamic batching보다 runtime scheduler의 continuous batching 또는 inflight batching이 더 핵심인 경우가 많다.

따라서 Triton 계층의 batching은 workload별로 나누어 봐야 한다.

| Workload | Primary scheduling concern |
|---|---|
| Stateless tensor inference | Triton dynamic batching, instance group placement |
| Stateful sequence inference | Triton sequence batching or runtime-managed state |
| LLM generation | Runtime-level continuous/inflight batching, KV cache scheduling, prefill/decode coordination |

```mermaid
flowchart TB
  subgraph Stateless["Stateless Tensor Inference"]
    S1["Request 1"]
    S2["Request 2"]
    S3["Request 3"]
    Batch["Triton Dynamic Batch"]
    Engine["TensorRT / ONNX / PyTorch Backend"]
    S1 --> Batch
    S2 --> Batch
    S3 --> Batch
    Batch --> Engine
  end

  subgraph LLM["LLM Generation"]
    P1["Prompt A"]
    P2["Prompt B"]
    Prefill["Prefill Scheduling"]
    Decode["Decode Loop"]
    KV["KV Cache Manager"]
    Stream["Token Streaming"]
    Runtime["vLLM / TensorRT-LLM / Custom Runtime"]
    P1 --> Prefill
    P2 --> Prefill
    Prefill --> Runtime
    Runtime --> Decode
    Decode --> KV
    KV --> Decode
    Decode --> Stream
  end

  classDef stateless fill:#eff6ff,stroke:#2563eb,color:#0f172a;
  classDef llm fill:#f5f3ff,stroke:#7c3aed,color:#0f172a;
  classDef runtime fill:#fef2f2,stroke:#dc2626,color:#0f172a;
  class S1,S2,S3,Batch,Engine stateless;
  class P1,P2,Prefill,Decode,KV,Stream llm;
  class Runtime runtime;
```

---

## 7. Model Pipeline Layer

Triton은 단일 모델 실행뿐 아니라 ensemble model pipeline도 지원한다.

그림에서는 다음과 같이 표현된다.

```text
Model Pipeline
Ensemble pipelines. Pre/post-processing.
Multi-model chaining without client round-trips.
```

일반적인 inference workload는 단일 모델 호출로 끝나지 않는다.

예를 들어 image inference pipeline은 다음처럼 구성될 수 있다.

```text
Image Decode
    ↓
Preprocess
    ↓
Object Detection Model
    ↓
Postprocess
    ↓
Response
```

LLM 기반 pipeline이라면 다음과 같은 구조도 가능하다.

```text
Prompt Validation
    ↓
Embedding
    ↓
Retriever
    ↓
Reranker
    ↓
LLM Generation
    ↓
Post-processing
```

Triton의 ensemble 기능을 사용하면 client가 여러 번 round-trip하지 않아도 server 내부에서 pipeline을 연결할 수 있다.

또한 BLS, 즉 Business Logic Scripting을 사용하면 Python 기반의 custom logic을 Triton model pipeline 안에 포함할 수 있다.

다만 복잡한 application-level orchestration은 Ray Serve가 더 자연스럽고, model execution 근처의 pipeline은 Triton ensemble이 더 적합할 수 있다.

따라서 실무에서는 다음처럼 역할을 나누는 것이 좋다.

| Layer               | Good For                                                     |
| ------------------- | ------------------------------------------------------------ |
| Ray Serve           | Application-level orchestration, routing, autoscaling        |
| Triton Ensemble/BLS | Model-level pipeline, pre/post-processing close to inference |
| Custom Service      | Business logic, external API integration                     |

---

## 8. Model Runtime Layer

Triton 아래에는 실제 모델을 실행하는 runtime/backend 계층이 있다.

그림에서는 세 가지 경로로 표현된다.

```text
TensorRT
ONNX / PyTorch
LLM Runtime Integration
```

---

### 8.1 TensorRT

TensorRT는 NVIDIA GPU에 최적화된 inference engine이다.

그림에서는 다음과 같이 표현된다.

```text
NVIDIA-optimized engine.
High throughput with FP8 / INT8 acceleration.
```

TensorRT는 모델 graph optimization, kernel fusion, precision calibration, quantization 등을 통해 GPU inference 성능을 높인다.

특히 고정된 input shape 또는 제한된 shape profile을 가진 workload에서 강력하다.

대표적인 장점은 다음과 같다.

| Strength                | Description                       |
| ----------------------- | --------------------------------- |
| High Throughput         | GPU inference 성능 최적화              |
| Low Latency             | Kernel fusion, graph optimization |
| Quantization            | FP16, INT8, FP8 등 활용 가능           |
| NVIDIA GPU Optimization | NVIDIA GPU에 특화된 실행 경로             |

하지만 TensorRT는 build 과정과 compatibility 관리가 필요하다.

특히 LLM에서는 model architecture, GPU architecture, TensorRT-LLM version, CUDA version, quantization format에 따라 build와 runtime 안정성이 크게 달라질 수 있다.

---

### 8.2 ONNX Runtime / PyTorch

ONNX Runtime과 PyTorch backend는 portability와 flexibility에 강점이 있다.

그림에서는 다음과 같이 표현된다.

```text
Portable execution path for general deep learning workloads.
```

ONNX는 여러 framework에서 export된 모델을 표준 graph 형태로 실행할 수 있게 해준다.

PyTorch backend는 PyTorch model을 비교적 자연스럽게 serving할 수 있게 해준다.

| Runtime        | Good For                                                |
| -------------- | ------------------------------------------------------- |
| ONNX Runtime   | Framework-neutral model serving                         |
| PyTorch        | Flexible model execution, research-to-production bridge |
| Torch-TensorRT | PyTorch와 TensorRT 사이의 최적화 경로                            |

이 경로는 TensorRT만큼 극단적인 최적화를 제공하지는 않을 수 있지만, 모델 호환성과 운영 편의성이 좋다.

특히 다양한 모델을 빠르게 production에 올려야 하는 환경에서는 ONNX/PyTorch runtime이 현실적인 선택이 될 수 있다.

---

### 8.3 LLM Runtime Integration

그림의 세 번째 runtime 경로는 `LLM Runtime Integration`이다.

이 영역을 특정 runtime 이름 하나로 표현하면 약간의 오해를 만들 수 있다.

Triton에는 vLLM backend가 있지만, runtime semantics는 TensorRT나 ONNX Runtime처럼 compiled graph를 실행하는 backend와 다르다. vLLM backend는 vLLM engine을 통해 LLM-specific scheduling과 memory management를 수행하는 통합 경로로 이해하는 것이 더 정확하다.

그래서 최종 다이어그램에서는 다음과 같이 표현한다.

```text
LLM Runtime Integration
vLLM Backend · TensorRT-LLM Backend · Custom Backend
PagedAttention, KV cache, and inflight batching are handled by the LLM runtime.
```

이 계층은 LLM serving에서 특히 중요하다.

LLM inference는 일반적인 CNN/Transformer encoder inference와 성격이 다르다.

| LLM-specific Concern | Description                               |
| -------------------- | ----------------------------------------- |
| KV Cache             | 이전 token의 key/value cache 재사용             |
| PagedAttention       | KV cache memory를 page 단위로 효율 관리           |
| Continuous Batching  | 생성 중인 sequence와 새 request를 함께 scheduling  |
| Token Streaming      | 생성 token을 client로 streaming               |
| Prefix Caching       | 동일 prefix 재사용                             |
| Speculative Decoding | 작은 draft model을 활용한 decoding acceleration |

vLLM과 TensorRT-LLM은 이런 LLM serving 특화 기능을 제공하는 runtime/backend 경로다.

> Note: Triton has a vLLM backend, and Triton also has a TensorRT-LLM backend. The important architectural point is that LLM scheduling semantics such as KV cache management, PagedAttention-style memory layout, continuous/inflight batching, and token streaming are primarily owned by the LLM runtime/backend, not by generic stateless dynamic batching.

---

## 9. Hardware Foundation: GPU Cluster

가장 아래 계층은 실제 GPU cluster다.

그림에서는 다음과 같이 표현된다.

```text
GPU Cluster (H100 / A100)
NVLink fabric · HBM2e/HBM3 memory · GPUDirect RDMA · NVMe / GDS data path
```

이 계층은 inference workload의 물리적 실행 기반이다.

중요한 구성 요소는 다음과 같다.

| Component      | Role                                 |
| -------------- | ------------------------------------ |
| GPU            | 실제 tensor computation 수행             |
| HBM            | GPU-local high bandwidth memory      |
| NVLink         | GPU 간 고속 연결                          |
| RDMA           | CPU 개입을 줄인 low-latency network path  |
| GPUDirect RDMA | NIC와 GPU memory 사이의 직접 data transfer |
| NVMe           | 빠른 local storage                     |
| GDS            | GPUDirect Storage 기반 data path       |

Inference workload에서는 training만큼 GPU 간 통신이 복잡하지 않을 수 있다.

하지만 대형 모델 serving, tensor parallelism, multi-GPU inference, embedding index serving, high-throughput batch inference에서는 GPU fabric과 storage path가 여전히 중요하다.

특히 LLM serving에서는 GPU memory capacity와 memory bandwidth가 성능을 크게 좌우한다.

---

## 10. System Properties

그림 하단에는 이 architecture가 달성하려는 네 가지 system property가 정리되어 있다.

```text
Performance First
Elastic by Design
Reliable Operations
Open Integration
```

---

### 10.1 Performance First

```text
Low latency and high throughput at scale.
```

Inference system의 성능은 단순히 GPU가 빠르다고 해결되지 않는다.

실제 성능은 여러 계층의 조합으로 결정된다.

```text
Request routing
Batching policy
Replica placement
GPU memory usage
Runtime backend
Network path
Model optimization
```

Ray는 request와 replica를 적절히 분산시키고, Triton은 model execution을 최적화한다.

이 두 계층이 함께 동작해야 end-to-end latency와 throughput을 모두 개선할 수 있다.

---

### 10.2 Elastic by Design

```text
Scale replicas, then scale GPU nodes through the cluster autoscaler.
```

AI serving workload는 traffic 변동이 크다.

특정 시간대에 요청이 몰릴 수도 있고, tenant별 사용량이 크게 다를 수도 있다.

Ray Serve는 autoscaling policy를 통해 deployment replica 수를 조절할 수 있다. 다만 GPU node 자체의 증감은 Ray Autoscaler, KubeRay, Kubernetes cluster autoscaler, 또는 cloud capacity manager가 담당한다.

예를 들어 다음과 같은 기준으로 scale-out할 수 있다.

```text
Ongoing requests per replica
Queue length
Latency
CPU/GPU utilization
Custom metrics
```

Triton은 replica 내부에서 batching, model instance placement, backend execution을 통해 GPU 사용률을 높인다.

즉, Ray Serve는 request-driven replica elasticity를 담당하고, Ray/KubeRay/cloud layer는 node-level capacity elasticity를 담당하며, Triton은 model-level execution efficiency를 담당한다.

---

### 10.3 Reliable Operations

```text
Health-aware routing, observability, and alerting.
```

Production inference system에서는 장애 대응이 중요하다.

고려해야 할 운영 요소는 다음과 같다.

| Area         | Example                                                     |
| ------------ | ----------------------------------------------------------- |
| Health Check | Ray Serve deployment health, Triton model readiness         |
| Metrics      | Request latency, queue time, GPU utilization, model latency |
| Logging      | Request log, model error, timeout log                       |
| Tracing      | End-to-end request path 추적                                  |
| Alerting     | Error rate, p95/p99 latency, GPU OOM, replica restart       |
| Rollback     | Model version rollback, traffic shift rollback              |

Ray와 Triton 모두 observability를 제공하지만, production에서는 Prometheus, Grafana, Loki, OpenTelemetry 같은 도구와 함께 구성하는 것이 일반적이다.

---

### 10.4 Open Integration

```text
Extensible interfaces, plugins, and standards.
```

AI inference platform은 하나의 runtime만으로 끝나지 않는다.

모델마다 최적의 runtime이 다를 수 있다.

예를 들어 다음과 같은 선택이 가능하다.

| Workload            | Possible Runtime                   |
| ------------------- | ---------------------------------- |
| Vision CNN          | TensorRT, ONNX Runtime             |
| Transformer Encoder | ONNX Runtime, TensorRT             |
| LLM                 | vLLM, TensorRT-LLM, custom backend |
| Classical ML        | Python backend, custom service     |
| Multi-step Pipeline | Ray Serve, Triton Ensemble, BLS    |

따라서 serving architecture는 특정 runtime에 종속되기보다, 여러 backend와 runtime을 통합할 수 있어야 한다.

---

## 11. End-to-End Request Flow

전체 request flow를 다시 정리하면 다음과 같다.

```text
1. Client sends inference request through REST or gRPC.
2. Ray Serve proxy receives the request at the serving endpoint.
3. Router / DeploymentHandle selects an available deployment replica.
4. The selected Ray Serve replica actor runs application logic on a worker node.
5. The replica dispatches the request to Triton or a model pipeline.
6. Triton applies model configuration, scheduling, and backend execution.
7. The selected runtime executes the model on GPU.
8. The response is returned through Triton and Ray Serve back to the client.
```

```mermaid
sequenceDiagram
  autonumber
  participant Client
  participant Proxy as Ray Serve Proxy
  participant Router as Router / Deployment Handle
  participant Replica as Replica Actor
  participant Triton
  participant Runtime as Model Runtime
  participant GPU
  participant Control as Serve Controller / GCS

  Control-->>Router: routing config and replica state
  Control-->>Replica: deployment lifecycle and placement
  Client->>Proxy: HTTP/gRPC inference request
  Proxy->>Router: route request
  Router->>Replica: select available replica
  Replica->>Replica: validate, preprocess, choose model path
  Replica->>Triton: inference request
  Triton->>Runtime: backend execution
  Runtime->>GPU: kernels / memory access
  GPU-->>Runtime: model outputs
  Runtime-->>Triton: backend response
  Triton-->>Replica: inference response
  Replica-->>Proxy: postprocessed response
  Proxy-->>Client: response or token stream
```

조금 더 단순화하면 다음과 같다.

```text
Client
  → Ray Serve Proxy
    → Router / DeploymentHandle
      → Ray Serve Replica Actor
        → Triton
          → TensorRT / ONNX / PyTorch / LLM Runtime
            → GPU Cluster
```

Ray Core는 이 흐름의 옆에서 replica actor와 worker resource를 배치하고 감시한다. Request payload 자체는 일반적으로 Head Node를 거쳐 Triton으로 전달되지 않는다.

---

## 12. Design Trade-offs

이 architecture를 설계할 때는 몇 가지 trade-off를 고려해야 한다.

---

### 12.1 Ray Serve vs Triton Ensemble

Ray Serve와 Triton Ensemble은 둘 다 pipeline을 구성할 수 있다.

하지만 적합한 영역이 다르다.

| Choice          | Better For                                                 |
| --------------- | ---------------------------------------------------------- |
| Ray Serve       | Application orchestration, business logic, dynamic routing |
| Triton Ensemble | Model-level pipeline, low-overhead pre/post-processing     |
| Triton BLS      | Model server 내부에서 Python logic 수행                          |

복잡한 business logic은 Ray Serve에 두고, model execution에 가까운 preprocessing/postprocessing은 Triton Ensemble로 두는 방식이 실용적이다.

---

### 12.2 Centralized Triton vs Per-node Triton

Triton server를 중앙에 둘지, 각 worker node마다 둘지도 중요한 설계 포인트다.

| Design                  | Pros                         | Cons                        |
| ----------------------- | ---------------------------- | --------------------------- |
| Centralized Triton Pool | 운영 단순, endpoint 관리 쉬움        | Network hop 증가, locality 약함 |
| Per-node Triton         | Locality 좋음, GPU affinity 명확 | 운영 복잡도 증가                   |
| Embedded Runtime        | 유연함                          | 표준화와 운영성 약화 가능              |

대규모 GPU cluster에서는 per-node Triton 또는 GPU pool 단위 Triton 배치가 더 자연스러울 수 있다.

---

### 12.3 TensorRT vs ONNX/PyTorch vs LLM Runtime

모든 모델을 TensorRT로 최적화하는 것이 항상 정답은 아니다.

| Runtime      | Use When                                              |
| ------------ | ----------------------------------------------------- |
| TensorRT     | 성능 최적화가 가장 중요하고 모델이 안정적일 때                            |
| ONNX Runtime | portability와 framework 중립성이 중요할 때                     |
| PyTorch      | 유연성과 빠른 배포가 중요할 때                                     |
| LLM Runtime  | KV cache, continuous batching, token streaming이 중요할 때 |

실무에서는 모델 유형별로 runtime을 분리하는 것이 일반적이다.

---

## 13. Recommended Production Architecture

Production 환경에서는 다음과 같은 구조를 추천할 수 있다.

```text
Ingress / API Gateway
    ↓
Ray Serve Proxy
    ↓
Router / DeploymentHandle
    ↓
Ray Serve Replica Actor
    ↓
Triton Server Pool
    ↓
Model Runtime Backends
    ↓
GPU Node Pool
```

운영 관점에서는 다음 요소가 함께 필요하다.

```text
Prometheus metrics
Grafana dashboard
Centralized logging
Distributed tracing
Model registry
Canary deployment
Autoscaling policy
GPU utilization monitoring
Model latency SLO
```

Control plane은 별도로 설계해야 한다.

```text
Ray Head Node / GCS
    ├── Serve Controller
    ├── Ray Autoscaler or KubeRay
    ├── Deployment configuration
    └── Replica lifecycle and placement
```

Kubernetes 환경이라면 다음 구성도 고려할 수 있다.

```text
Kubernetes
  ├── RayCluster / KubeRay
  ├── Ray Serve
  ├── Triton Inference Server Deployment
  ├── GPU Operator
  ├── Prometheus / Grafana
  ├── Model repository volume
  └── Ingress / Gateway API
```

---

## 14. Practical Notes

이 구조를 실제로 구현할 때는 다음 사항을 주의해야 한다.

### 14.1 Request Timeout

LLM generation은 일반 inference보다 시간이 오래 걸릴 수 있다.

Ray Serve timeout, Triton timeout, client timeout, ingress timeout을 함께 조정해야 한다.

---

### 14.2 GPU Memory

LLM serving에서는 GPU memory가 가장 중요한 병목이 될 수 있다.

KV cache, batch size, sequence length, model weight precision을 함께 고려해야 한다.

---

### 14.3 Dynamic Batching

Triton dynamic batching은 stateless tensor workload의 throughput을 높일 수 있지만 latency를 증가시킬 수 있다.

따라서 latency-sensitive workload와 throughput-oriented workload를 분리하는 것이 좋다.

LLM workload에서는 Triton dynamic batching만 보지 말고 runtime의 continuous/inflight batching, KV cache memory pressure, prefill/decode split, streaming backpressure를 함께 봐야 한다.

---

### 14.4 Model Versioning

Triton model repository는 model versioning을 지원한다.

Ray Serve의 traffic splitting과 함께 사용하면 canary deployment 또는 A/B test 구조를 만들 수 있다.

---

### 14.5 Observability

Ray와 Triton 각각의 metrics만 보는 것으로는 부족하다.

End-to-end request latency를 다음처럼 분해해서 봐야 한다.

```text
Client latency
Ingress latency
Ray Serve queue time
Ray actor execution time
Triton queue time
Model compute time
GPU utilization
Response serialization time
```

---

## 15. Summary

`Ray Cluster → Triton Inference Server` architecture는 AI inference platform을 두 개의 핵심 책임으로 분리한다.

Ray는 serving orchestration을 담당한다.

```text
Proxy ingress
Routing
Autoscaling
Replica management
Distributed scheduling
Application-level pipeline
```

Triton은 model execution optimization을 담당한다.

```text
Dynamic batching
Concurrent model execution
Model repository management
Backend runtime execution
GPU inference optimization
```

그리고 그 아래에서 TensorRT, ONNX Runtime, PyTorch, LLM runtime integration이 실제 모델 실행 경로를 구성한다.

마지막으로 GPU cluster는 HBM, NVLink, RDMA, NVMe/GDS 같은 accelerated infrastructure를 제공한다.

이 architecture의 핵심은 다음 문장으로 요약할 수 있다.

> Ray decides where and how many.
> Ray Serve decides which replica handles the request.
> Triton decides how fast.
> The serving system emerges from the handshake between orchestration and execution.

Production AI inference system은 단일 도구로 완성되지 않는다.

Ray, Triton, model runtime, GPU infrastructure가 서로 맞물릴 때 비로소 확장 가능하고 운영 가능한 inference platform이 된다.
