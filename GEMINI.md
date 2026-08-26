# DQN-LSTM Agent-driven Canary Release — Codebase Guide

> Tài liệu này được tự động nạp bởi AI coding agents mỗi khi bắt đầu hội thoại mới.
> Mục đích: cung cấp context nhanh, tránh phải khám phá lại codebase từ đầu.

---

## 1. Tổng quan Dự án

Dự án sử dụng **RL Agent (TransformerPPO)** để tự động điều phối **canary release** trên hạ tầng Kubernetes (K3s), thay thế phương pháp rule-based truyền thống. Agent quan sát metrics theo thời gian (latency, error rate, CPU, RAM, traffic weight) và đưa ra quyết định: **Hold (0)**, **Promote (1)**, hoặc **Rollback (2)**.

### Stack công nghệ
- **Cluster**: K3s (lightweight Kubernetes)
- **GitOps**: Argo CD + Argo Rollouts (canary strategy)
- **Service Mesh**: Istio (traffic splitting + sidecar metrics)
- **Metrics**: Prometheus (scrape Istio sidecar metrics)
- **RL Framework**: Stable-Baselines3 (PPO) + custom TransformerFeatureExtractor
- **Target service**: `checkoutservice` (gRPC microservice từ Google Online Boutique / Hipster Shop)
- **Load Testing**: Locust

---

## 2. RL Agent — Kiến trúc & Model

### Model được deploy: TransformerPPO — 5 Features

| Thuộc tính | Giá trị |
|-----------|---------|
| Algorithm | PPO (Stable-Baselines3) |
| Feature Extractor | `TransformerFeatureExtractor` (custom, xem `core/model.py`) |
| Observation shape | `(30, 5)` — seq_len=30 timesteps × 5 features |
| Features | `cpu_n`, `mem_n`, `l_ratio_n`, `e_ratio_n`, `weight_n` |
| Action space | `Discrete(3)` — Hold=0, Promote=1, Rollback=2 |
| d_model | 64 |
| n_heads / n_heads_feature | 4 / 1 |
| n_layers | 2 |
| Model file | `models/ppo_transformer_offline_best.zip` (đã verify: input shape [30, 5]) |
| Normalization | `models/vec_normalize.pkl` |

### Attention Mechanisms (Explainable AI)
- **Feature Attention**: Cross-attention xác định feature nào quan trọng nhất tại mỗi timestep
- **Temporal Attention**: Self-attention xác định timestep nào trong quá khứ ảnh hưởng quyết định

### Reward Logic (trong `core/env.py`)
- Hold: -0.5 (penalize lãng phí thời gian)
- Promote khi healthy: +2.0; Promote khi anomalous: -5.0 (terminal)
- Rollback khi anomalous: +5.0 (terminal); Rollback khi healthy: -10.0 (terminal)
- Promote tới 100% và healthy: +10.0 bonus
- Timeout (>50 steps): -5.0
- Anomaly detection: `e_ratio > 2.0` hoặc `l_ratio > 2.0` (smoothed window=4)

---

## 3. Cấu trúc Thư mục

```
├── core/                          # Core RL logic
│   ├── env.py                     # Simulated Canary environment (nhánh istio: 15 features cho 6G)
│   ├── model.py                   # TransformerFeatureExtractor (SB3 custom)
│   ├── feature_pipeline.py        # Raw metrics → normalized state vector
│   ├── online_env.py              # ⛔ IGNORE — deprecated online training env
│   └── env_ns3.py                 # ⛔ IGNORE — NS-3 simulation variant
│
├── training/                      # Training & evaluation scripts
│   ├── offline_training.py        # Main training script (PPO + Transformer)
│   ├── evaluate.py                # Evaluate model + generate attention heatmaps
│   ├── evaluate_ns3.py            # ⛔ IGNORE — NS-3 variant
│   └── online_training.py         # ⛔ IGNORE — deprecated (uses RecurrentPPO + online_env)
│
├── scripts/                       # Experiment orchestration
│   ├── run_experiment.sh / .ps1   # Orchestrate: reset → warmup → inject → cooldown → export
│   ├── run_all_v2.ps1             # Batch run all S1-S5 with RL agent → result_1/
│   ├── run_all_rule_based.ps1     # Batch run all S1-S5 with rule-based → result_2/
│   ├── inject_fault.py            # Inject chaos via Argo Rollouts env vars
│   ├── chaos_reset.py             # Reset all chaos env vars to defaults
│   ├── export_data.py             # Query Prometheus range API → CSV
│   ├── visualize.py               # ⚠️ CẦN REFACTOR — hiện giả lập rule-based bằng hardcode offset
│   └── simulate_safety.py         # ⚠️ CẦN REFACTOR — giả lập tương tự
│
├── scenarios/                     # Chaos experiment scenarios (YAML)
│   ├── S1_high_latency.yaml       # Progressive latency injection (10ms → 1500ms)
│   ├── S2_cpu_spike.yaml          # CPU stress (0% → 100%)
│   ├── S3_memory_leak.yaml        # Memory allocation leak (0 → 200MB)
│   ├── S4_error_burst.yaml        # gRPC error rate spikes (0.02 → 0.8, intermittent)
│   └── S5_cascading_failure.yaml  # Combined latency + error escalation
│
├── models/                        # Trained model artifacts
│   ├── ppo_transformer_offline_best.zip  # ✅ Main model (5 features, verified)
│   └── vec_normalize.pkl                 # VecNormalize stats
│
├── gitops/                        # Kubernetes & Argo manifests
│   ├── base/                      # Base Helm chart templates
│   ├── charts/                    # Helm charts
│   └── releases/                  # Per-service Helm values
│       └── checkoutservice-values.yaml  # checkoutservice config
│
├── result_1/                      # Experiment results — RL Agent (có dữ liệu)
│   ├── S1_high_latency-rl_agent-01/     # metrics.csv + timeline.json
│   ├── S2_cpu_spike-rl_agent-01/
│   ├── S3_memory_leak-rl_agent-01/
│   ├── S4_error_burst-rl_agent-01/
│   └── S5_cascading_failure-rl_agent-01/
│
├── result_2/                      # Experiment results — Rule-based
│   └── (⚠️ TẤT CẢ metrics.csv đều RỖNG — chỉ có header, chưa có dữ liệu thực)
│
├── results/                       # Visualization outputs
│   ├── S*_comparison.png          # ⚠️ Biểu đồ giả lập (cần regenerate với dữ liệu thực)
│   └── raw/                       # Additional raw exports
│
├── loadgenerator/                 # Locust load testing configs
├── tests/                         # Unit tests
│   └── test_network_noise.py      # Tests for 6G network noise simulation
│
├── mlflow/ & mlruns/              # ⛔ IGNORE — MLflow tracking (không active)
├── ns-3_scenario*/ & ns3_*/       # ⛔ IGNORE — NS-3 simulation data
└── services/                      # Microservice source code modifications
```

---

## 4. Nhánh Git — Quan trọng

| Nhánh | Mục đích | Features | Trạng thái |
|-------|---------|----------|-----------|
| **`istio`** (hiện tại) | Mở rộng cho nghiên cứu 6G | 15 features (thêm handover, SINR, RSRP, PRB, HARQ, NTN, ISAC, packet_loss, jitter, deploy_age) | Active development |
| **`linkerd`** | Phiên bản gốc cho 4G/5G | **5 features** (cpu, mem, latency_ratio, error_ratio, weight) | Reference/stable |
| `main` | Default | — | Sync point |
| `6g`, `ebpf`, `grafanabeyla` | Experimental branches | — | Reference only |

> **Lưu ý**: Model file `ppo_transformer_offline_best.zip` hiện tại **đã được train với 5 features** (xác nhận qua observation_space `[30, 5]`), tương thích với cả hai nhánh linkerd và istio (nếu dùng 5-feature observation trên istio).

---

## 5. Metrics Stack & Prometheus Queries

### Nguồn metrics: Istio Sidecar (Envoy Proxy)

Istio sidecar được inject vào mỗi pod, tự động expose metrics qua Prometheus.

### Queries quan trọng

```promql
# Error rate (gRPC) — dùng grpc_response_status cho chính xác
sum(rate(istio_requests_total{
  destination_service=~"checkoutservice.*",
  grpc_response_status!="0"
}[1m]))
/
sum(rate(istio_requests_total{
  destination_service=~"checkoutservice.*"
}[1m]))

# P95 Latency
histogram_quantile(0.95,
  sum(rate(istio_request_duration_milliseconds_bucket{
    destination_service=~"checkoutservice.*"
  }[1m])) by (le))

# Traffic weight canary
sum(rate(istio_requests_total{
  destination_service=~"checkoutservice-canary.*"
}[1m]))
/
sum(rate(istio_requests_total{
  destination_service=~"checkoutservice.*"
}[1m]))
```

### Lưu ý quan trọng
- `export_data.py` hiện dùng Istio queries nhưng escape regex có vấn đề
- `online_env.py` (deprecated) dùng OpenTelemetry metrics (`rpc_server_call_duration_seconds_count`) — **KHÔNG dùng** cho experiment

---

## 6. Experiment Workflow

### Pipeline hiện tại
```
chaos_reset.py → warmup (3min) → inject_fault.py (scenario YAML)
                                    → cooldown (2min) → export_data.py → CSV
```

### Vấn đề đã biết cần giải quyết
1. **Không có rule-based controller code** — cần implement `RuleBasedController`
2. **result_2/ (rule-based) hoàn toàn rỗng** — chưa có dữ liệu thực
3. **visualize.py giả lập** rule-based bằng hardcode offset trên dữ liệu RL
4. **Chỉ n=1 run/scenario** — cần ≥ 5 runs cho statistical significance
5. **Thiếu action logging** — không ghi lại quyết định của agent tại mỗi step

### Mục tiêu so sánh: RL Agent vs Rule-based
- Chuẩn bị cho khóa luận tốt nghiệp + bài báo khoa học
- Cần journal-grade rigor: ≥ 10 runs/combo, Mann-Whitney U test, effect size
- Đo 14 metrics: T_detect, T_react, AUC_error, AUC_latency, False Positive/Negative, v.v.
- 3 phương pháp đặt ngưỡng rule-based: static (SRE), ratio-based, burn rate

---

## 7. Quy tắc cho Agent

### PHẢI làm
- Dùng nhánh `istio` (hiện tại) cho development
- Model deploy dùng **5 features** (không phải 15)
- Dùng Istio metrics cho experiment (không OTel)
- Giữ `core/` cho RL logic, `scripts/` cho orchestration, `training/` cho training
- Test locally trước khi deploy

### KHÔNG được làm
- ❌ Sửa hoặc dùng `core/online_env.py` — deprecated
- ❌ Sửa hoặc dùng MLflow (`mlflow/`, `mlruns/`)
- ❌ Sửa NS-3 files (`env_ns3.py`, `evaluate_ns3.py`, `ns-3_*`)
- ❌ Mix ML code với Kubernetes manifests
- ❌ Modify live cluster state trực tiếp — dùng GitOps
