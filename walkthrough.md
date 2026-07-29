# Walkthrough: RL Agent Kubernetes Controller

Dựa trên yêu cầu của bạn, hệ thống đã được triển khai theo chuẩn **Kubernetes Operator Pattern**. Qua đó, thay vì chỉ tạo một Webhook server đơn giản, ta đã tạo ra một Custom Resource Definition (CRD) và một Controller chuyên biệt để quản lý nó.

Dưới đây là tổng hợp những gì đã được thực thi.

## 1. Khai báo CRD `RLAgent`
**File:** [gitops/base/agent-crd.yaml](file:///home/thuanthentran/RL/gitops/base/agent-crd.yaml)

Chúng ta đã định nghĩa một resource mới cho cụm K8s với tên gọi `RLAgent` thuộc group `rl.thuanthentran.io`. 
Nhờ CRD này, người dùng (hoặc pipeline CI/CD) có thể khai báo một tác nhân AI (AI Agent) dưới dạng file YAML thuần túy. Nó hỗ trợ cấu hình:
- Nguồn hình ảnh Docker (`image`)
- Đường dẫn tới Model (`modelPath`)
- URL của máy chủ Prometheus (`prometheusUrl`)

## 2. Xây dựng K8s Operator (Kopf Controller)
**File:** [services/controller/main.py](file:///home/thuanthentran/RL/services/controller/main.py)

Thay vì phải tự viết các vòng lặp watch API rườm rà của K8s, ta đã sử dụng **Kopf** (Kubernetes Operator Pythonic Framework). 
- Controller này liên tục theo dõi các sự kiện sinh ra (create) và cập nhật (update) của loại tài nguyên `RLAgent`.
- Mỗi khi có một `RLAgent` mới được khai báo, nó tự động biên dịch cấu hình đó và sinh ra một `Deployment` chứa Pod của FastAPI Webhook (Agent), đi kèm một `Service` để phục vụ mạng.

Đồng thời, [Dockerfile cho Operator](file:///home/thuanthentran/RL/services/controller/Dockerfile) cũng đã được thiết lập để đóng gói framework này.

## 3. Webhook Inference (Có sẵn)
**File:** [services/agent/main.py](file:///home/thuanthentran/RL/services/agent/main.py)

Rất tuyệt vời khi phần lõi AI của bạn đã hoàn thiện! Webhook FastAPI đã có sẵn:
- Nó cung cấp endpoint `/api/v1/decision`.
- Khi Argo Rollouts (thông qua AnalysisTemplate) gửi Webhook, Agent sẽ bóc tách `canary_hash` và `stable_hash`.
- Gọi hàm `_build_history_from_prometheus` tới `PROMETHEUS_URL` để cào metrics (từ Istio và cAdvisor).
- Nhồi qua mô hình Reinforcement Learning `RecurrentPPO.load(MODEL_PATH)` và trả về Quyết định (Action).

> [!TIP]
> Việc bạn đã dùng RecurrentPPO của `sb3_contrib` để xử lý chuỗi thời gian là cực kỳ chuẩn xác cho mô hình LSTM. Các trọng số cảnh báo (Safety Guards) cũng được hardcode rất cẩn thận, có thể chặn việc promote nếu lỗi vượt quá tỷ lệ cho phép mà chưa cần model quyết định.

## 4. Manifests Triển khai (GitOps)
- **Cấp quyền & Khởi động Operator:** [gitops/base/agent-operator.yaml](file:///home/thuanthentran/RL/gitops/base/agent-operator.yaml) (Bao gồm RBAC cho phép Operator có quyền đọc CRD và tạo/chỉnh sửa Deployment).
- **Tạo Instance:** [gitops/base/agent-instance.yaml](file:///home/thuanthentran/RL/gitops/base/agent-instance.yaml)

```yaml
# Ví dụ về bản mô tả Instance RLAgent
apiVersion: rl.thuanthentran.io/v1alpha1
kind: RLAgent
metadata:
  name: canary-eval-agent
spec:
  image: "thuanthentran/rl-canary-agent:latest"
  modelPath: "models/ppo_lstm_offline_best.zip"
  prometheusUrl: "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090"
  replicas: 1
  port: 8000
```

> [!IMPORTANT]
> **Bước tiếp theo dành cho bạn:**
> 1. Xây dựng Docker Image cho Agent và Operator (Chạy `docker build` và `docker push` vào registry).
> 2. Cài đặt các file yaml trong `gitops/base/` vào K8s (`kubectl apply -f agent-crd.yaml`, v.v...).
> 3. Cập nhật URL trong AnalysisTemplate của Argo Rollouts thành `http://canary-eval-agent-service.default.svc.cluster.local:8000/api/v1/decision`.
