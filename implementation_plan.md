# Kích hoạt eBPF gRPC Observability cho Agent

## Phase 2: Tự động hóa Huấn luyện & Tiêm lỗi Cấp Ứng dụng

Dựa trên phản hồi và thiết kế tinh gọn, chúng ta **quyết định giữ lại phương pháp Tiêm lỗi vào Mã nguồn (Code-level Injection)** thông qua biến môi trường `FAULT_SCENARIO` thay vì dùng Chaos Mesh.

**Lý do chiến lược:**
1. **Mô phỏng chân thực bản chất ứng dụng**: Ứng dụng hoạt động không ổn định là lỗi phát sinh từ *bên trong* logic của ứng dụng (code lỗi, xử lý rườm rà), dẫn đến việc phải Rollback. Tiêm lỗi vào mã nguồn mô phỏng chính xác sự yếu kém này.
2. **Tiết kiệm tối đa tài nguyên Cluster**: Tránh việc triển khai thêm một hệ thống cồng kềnh (Chaos Mesh / Istio), vốn tiêu tốn CPU/RAM và tạo overhead cho K8s. Đặc biệt phù hợp với Sandbox K3s dùng để huấn luyện RL Agent.

### Các hạng mục thực hiện:
1. **Giữ nguyên mã nguồn `checkoutservice`**: Giữ logic `os.Getenv("FAULT_SCENARIO")` để Agent có thể gửi tín hiệu tiêm lỗi siêu nhẹ qua K8s API.
2. **Tiếp tục dùng Biến Môi trường trong `online_env.py`**: Khi reset môi trường, Agent sẽ gửi Patch cập nhật Rollout Template kèm theo `FAULT_SCENARIO`.
3. **Mở rộng Vòng lặp Huấn luyện**: Nâng số bước huấn luyện `TOTAL_TIMESTEPS` trong `online_training.py` lên 200 (tương đương 50 episodes) để Agent tự học và tự động lặp lại quy trình.

> [!NOTE]
> Mọi thay đổi liên quan đến Chaos Mesh đã được Revert sạch sẽ khỏi cluster và code để trả lại trạng thái nguyên bản.

## User Review Required
Bạn có muốn tôi bắt đầu chạy `python training/online_training.py` để Agent chính thức học liên tục 50 Episodes trên môi trường Sandbox này không?ruy vấn PromQL để đảm bảo Agent lấy đúng metric.
- Thêm debug log vào `online_env.py` để in ra các chỉ số Latency mới thu được.

## Kế hoạch Kiểm thử (Verification Plan)

### Automated Tests
1. Chạy `kubectl apply -f scratch/l7-visibility-checkoutservice.yaml`.
2. Tạo thử traffic bằng Locust (chạy nền).
3. Truy vấn trực tiếp API Prometheus (`http://192.168.142.165:30090`) để xác minh metric `hubble_http_responses_total` (hoặc requests) đã xuất hiện cho `destination_workload="checkoutservice"`.

### Manual Verification
Tiến hành chạy lại script huấn luyện siêu tốc `python training/online_training.py` (với `TOTAL_TIMESTEPS=4` vừa sửa) để xem Agent có nhận diện được sự tăng vọt của Latency (> 3000ms) ở `checkoutservice` và quyết định ABORT (0) thay vì PROMOTE (1) hay không.
