# GitOps & Chaos Testing

Thư mục này chứa cấu hình Values (`*-values.yaml`) cho các đợt phát hành Microservices qua ArgoCD, cũng như ConfigMap mô phỏng kịch bản lỗi (Chaos Engineering).

## 1. Giới thiệu Kiến trúc Chaos Testing
Hiện tại, `samplemicroservice` đã được nâng cấp để hỗ trợ "bơm lỗi" trực tiếp vào ứng dụng nhằm kiểm thử khả năng phản ứng của AI Agent (RL Model PPO-LSTM) mà không cần phải viết thêm mã nguồn phức tạp hay build lại Docker Image. 

Cụ thể, mã nguồn logic bơm lỗi (`main.py`) được quản lý bằng tệp `service-b-configmap.yaml` và tự động ghi đè vào pod đang chạy bằng tính năng `extraVolumes` của Base Chart `universal-canary`.

### Các biến môi trường hỗ trợ Chaos:
- `CHAOS_ERROR_RATE`: Tỷ lệ request trả về lỗi 503 (ví dụ: `0.1` = 10%).
- `CHAOS_DELAY_MS`: Mô phỏng độ trễ Latency. Request sẽ bị block thêm số ms này (ví dụ: `2000` = 2s).
- `CHAOS_CPU_BURN_ITERS`: Mô phỏng High CPU. Ứng dụng chạy vòng lặp vô ích để ngốn CPU (ví dụ: `5000000`).
- `CHAOS_MEM_ALLOC_MB`: Mô phỏng High Memory/OOM. Ứng dụng tự động cấp phát số MB RAM này vào bộ nhớ (ví dụ: `256` = 256MB).

## 2. Cách Trigger Rollout Bản Lỗi Tạm Thời (Manual Patching)

Để kiểm thử Agent mà không làm bẩn mã nguồn (Git), bạn có thể tiến hành "Patch" cấu hình tạm thời trực tiếp trên cụm K8s. Argo Rollouts sẽ nhận diện có thay đổi (tạo ReplicaSet mới) và trigger tiến trình phân tích `AnalysisRun`.

**Bước 1: Trigger Lỗi bằng Kubectl Patch**
```bash
wsl -d k3s kubectl patch rollout service-b -n twin --type=json -p='[
  {"op": "replace", "path": "/spec/template/spec/containers/0/env/2/name", "value": "CHAOS_ERROR_RATE"}, 
  {"op": "replace", "path": "/spec/template/spec/containers/0/env/2/value", "value": "1.0"}
]'
```
*(Bạn có thể thay đổi `CHAOS_ERROR_RATE` bằng `CHAOS_DELAY_MS` hay bất kỳ biến nào phía trên).*

**Bước 2: Quan sát RL Agent phân tích**
- Load-tester `hey` sẽ đẩy tải vào bản Rollout mới.
- Linkerd & Prometheus thu thập thông tin và đẩy cho Agent.
- Theo dõi `AnalysisRun` để xem Agent đưa ra phán quyết (Promote, Wait, hay Rollback):
```bash
wsl -d k3s kubectl get analysisrun -n twin -w
```

## 3. Cách Phục hồi (Quay trở lại bản Healthy)

Do chúng ta thực hiện thao tác Patch thủ công (tạm thời) vượt quyền Git, trạng thái của cluster lúc này đã bị OutOfSync so với cấu hình trên Git.

**Để phục hồi:**
- Mở **ArgoCD UI**.
- Tìm ứng dụng `service-b-twin` hoặc `service-b-prod`.
- Nhấn nút **SYNC** để ArgoCD đồng bộ lại trạng thái của Rollout về đúng như những gì được thiết kế (không có các biến Chaos nguy hiểm). Bản phát hành lỗi (nếu đang chạy dở) sẽ bị ngừng, và trở lại trạng thái khoẻ mạnh ban đầu.
