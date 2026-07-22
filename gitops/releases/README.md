# Hướng Dẫn Kích Hoạt & Mô Phỏng Lỗi Canary Release (GitOps)

Thư mục này chứa các file cấu hình `values.yaml` phát hành cho 4 microservices:
- `frontend-values.yaml`
- `service-a-values.yaml`
- `service-b-values.yaml`
- `service-c-values.yaml`

---

## 1. Cách Trigger Đợt Canary Release Mới

Để khởi động một phiên bản release mới cho bất kỳ microservice nào, bạn chỉ cần chỉnh sửa file `values.yaml` tương ứng của dịch vụ đó, sau đó commit và push lên Git.

---

## 2. Kịch Bản 1: Triển Khai Phiên Bản Thành Công (PASS / PROMOTE)

Khi muốn phát hành phiên bản hoạt động bình thường, không bơm lỗi:

Sửa file (ví dụ `service-b-values.yaml`):

```yaml
serviceName: service-b
replicas: 1
image: ghcr.io/thuanthentran/samplemicroservice:latest
containerPort: 8000
portName: http
env:
  - name: SERVICE_NAME
    value: service-b
  - name: UPSTREAMS
    value: ""
  - name: CHAOS_ERROR_RATE
    value: "0.0"   # Không có lỗi
  - name: CHAOS_DELAY_MS
    value: "0"     # Không có độ trễ
```

**Luồng hoạt động:**
1. `twin` namespace chạy Canary Release (20% -> 50% -> 80%) + Stress Test tự động.
2. RL Agent đánh giá chỉ số Prometheus tốt -> Trả về `Promote`.
3. `twin` hoàn tất thành công (`Healthy`).
4. `sync-controller` CronJob phát hiện `twin` thành công -> Tự động Unpause cho `prod` tiến hành release trên Production!

---

## 3. Kịch Bản 2: Triển Khai Phiên Bản Lỗi / Giả Lập Chaos (FAIL / ABORT)

Để thử nghiệm khả năng tự động Rollback và bảo vệ môi trường Production khi phiên bản mới bị lỗi:

Sửa các biến `CHAOS_*` trong file (ví dụ `service-b-values.yaml`):

```yaml
serviceName: service-b
replicas: 1
image: ghcr.io/thuanthentran/samplemicroservice:latest
containerPort: 8000
portName: http
env:
  - name: SERVICE_NAME
    value: service-b
  - name: UPSTREAMS
    value: ""
  # === BƠM LỖI GIẢ LẬP ===
  - name: CHAOS_ERROR_RATE
    value: "0.5"   # Mô phỏng tỷ lệ lỗi 50%
  - name: CHAOS_DELAY_MS
    value: "2000"  # Mô phỏng trễ 2000ms (2 giây)
```

**Các biến Chaos khả dụng:**
- `CHAOS_ERROR_RATE`: Tỷ lệ ném lỗi HTTP 503 / gRPC UNAVAILABLE (từ `0.0` đến `1.0`).
- `CHAOS_DELAY_MS`: Thêm độ trễ tính theo mili-giây.
- `CHAOS_CPU_BURN_ITERS`: Vòng lặp chiếm dụng CPU để thử nghiệm nghẽn tính toán.

**Luồng hoạt động:**
1. `twin` namespace khởi tạo Canary Pod mới mang biến môi trường bị bơm lỗi.
2. Stress Test Job nã tải vào `frontend`, lưu lượng đi qua `service-b-canary` bị ném lỗi 50%.
3. Linkerd ghi nhận chỉ số lỗi/latency tăng vọt vào Prometheus.
4. RL Agent đánh giá thấy chất lượng kém -> Trả về `Abort`.
5. Rollout ở `twin` tự động Rollback (chuyển trạng thái `Degraded`).
6. `sync-controller` phát hiện `twin` bị `Degraded` -> Tự động chạy `kubectl argo rollouts abort` trên `prod`, giúp bảo vệ môi trường thật khỏi bị lỗi!

---

## 4. Lệnh Commit & Push Kích Hoạt

Sau khi chỉnh sửa file values mong muốn, chạy các lệnh sau trên Terminal:

```bash
git add gitops/releases/
git commit -m "feat(release): trigger canary rollout with chaos testing"
git push origin linkerd
```
