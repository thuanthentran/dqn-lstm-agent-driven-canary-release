# Sample Microservice (Target Apps)

Thư mục này chứa mã nguồn của `samplemicroservice` - một ứng dụng mẫu được thiết kế đặc biệt để chịu sự quản lý của luồng Canary và tiếp nhận các kịch bản tiêm lỗi (Chaos Engineering).

## 1. Kiến trúc Ứng dụng
Ứng dụng được viết bằng Python, sử dụng đồng thời 2 giao thức:
- **FastAPI (HTTP)**: Expose API REST (ví dụ GET `/`) để tiếp nhận traffic đầu vào từ Load Generator hoặc Ingress.
- **gRPC**: Expose service gRPC nội bộ để mô phỏng giao tiếp tốc độ cao giữa các vi dịch vụ.

Ứng dụng có cơ chế tự động phát hiện biến môi trường `UPSTREAMS`. Khi nhận một Request HTTP ở tầng frontend, nó sẽ lần lượt kích hoạt hàm `call_upstreams()` để gọi tiếp sang các dịch vụ backend khác qua HTTP hoặc gRPC tuỳ vào định nghĩa cấu hình.

## 2. Cơ chế Tiêm Lỗi (Chaos Engineering)
Điểm đặc biệt của microservice này là khả năng nhận cấu hình "tự phá hoại" thông qua các biến môi trường (hoặc cập nhật qua API POST `/chaos`).

Bất kỳ request nào đi qua ứng dụng đều phải gọi hàm `apply_chaos_async()`. Hàm này sẽ kiểm tra các biến số:
- `CHAOS_ERROR_RATE`: Cố ý văng Exception ngẫu nhiên dựa trên tỷ lệ phần trăm (Trả về 503).
- `CHAOS_DELAY_MS`: Cố ý `asyncio.sleep()` làm tăng vọt độ trễ.
- `CHAOS_CPU_BURN_ITERS`: Chạy vòng lặp vô nghĩa để làm nghẽn Event Loop và tăng vọt mức tiêu thụ CPU.
- `CHAOS_MEM_ALLOC_MB`: Tự động cấp phát các block RAM ảo vào bộ nhớ để mô phỏng OOM (Out Of Memory).

## 3. Vai trò trong Hệ thống
Ứng dụng này đóng vai trò như một "hình nhân thế mạng". Bằng cách khéo léo thay đổi cấu hình môi trường của Pod (Ví dụ: `service-b-configmap.yaml` ở thư mục `gitops/releases`), ta có thể giả lập một bản Canary bị lỗi ngầm, qua đó đánh giá xem AI Agent phát hiện lỗi và phản ứng nhanh hay chậm.