# Load Generator (Traffic Gen)

Thư mục này chứa mã nguồn của thành phần Load Generator - một máy tạo tải ảo được sử dụng để liên tục bắn traffic vào hệ thống, giúp kích hoạt luồng đo đạc metrics của Linkerd/Prometheus và làm đầu vào cho Agent phân tích.

## 1. Nó là gì?
Load Generator là một ứng dụng Python sử dụng framework **Locust** (`FastHttpUser`). Nó mô phỏng hành vi của người dùng thật truy cập vào các endpoint của hệ thống.

## 2. Kiến trúc và Mục tiêu
- Tập lệnh mô phỏng (`locustfile.py`) định nghĩa các hành vi như: truy cập trang chủ (`/`), đổi tiền tệ (`/setCurrency`), xem sản phẩm (`/product/X`), thêm vào giỏ hàng (`/cart`), và thanh toán (`/cart/checkout`).
- **Mục tiêu**: Load Generator không gọi trực tiếp vào IP của Pod mà gọi vào **Tên Service** của ứng dụng mục tiêu (được điều hướng bởi Linkerd).
- Việc đa dạng hóa các endpoint giúp tạo ra tải trọng (CPU/RAM) và độ trễ ngẫu nhiên giống hệt môi trường thực tế, từ đó hệ thống Canary bộc lộ các điểm yếu (nếu có lỗi tiêm vào) một cách rõ nét nhất.

## 3. Điều khiển Mật độ và Tốc độ
Sức ép sinh tải của Load Generator phụ thuộc vào 2 yếu tố chính:
- **Tham số `wait_time = between(1, 10)`**: Khai báo trong `locustfile.py`, khiến mỗi "người dùng" sẽ chờ ngẫu nhiên từ 1 đến 10 giây giữa các request. 
- **Số lượng Concurrent Users (Replicas)**: Mật độ tải thực tế được điều khiển bằng số lượng bản sao (Replicas) hoặc tham số dòng lệnh truyền cho locust khi chạy trong K8s. Càng tăng số lượng người dùng đồng thời, số lượng Request/second (RPS) bắn vào Service mục tiêu càng lớn.
