# Load Generator (Traffic Gen)

Thư mục này chứa manifest triển khai Load Generator - một máy tạo tải ảo được sử dụng để liên tục bắn traffic vào hệ thống, giúp kích hoạt luồng đo đạc metrics của Linkerd/Prometheus và làm đầu vào cho Agent phân tích.

## 1. Nó là gì?
Hiện tại, Load Generator sử dụng công cụ **hey** (một chương trình HTTP load generator gọn nhẹ viết bằng Go) thông qua Docker image `williamyeh/hey`. Nó được triển khai dưới dạng một Kubernetes Deployment nằm trong namespace `prod`.

## 2. Kiến trúc và Mục tiêu
- **Mục tiêu**: Load Generator bắn traffic liên tục vào endpoint của dịch vụ `frontend` (cụ thể là `http://frontend.prod.svc.cluster.local:8000/`).
- Bằng cách tạo ra tải HTTP, nó mô phỏng người dùng thật đang truy cập vào hệ thống. Việc có traffic thực tế chảy qua sẽ giúp Linkerd ghi nhận được các metrics như độ trễ (latency), tỷ lệ lỗi (error rate) và cung cấp dữ liệu cho Agent khi đánh giá bản Canary.

## 3. Điều khiển Mật độ và Tốc độ
Sức ép sinh tải của Load Generator được điều khiển qua các tham số dòng lệnh truyền cho `hey`:
- `-z 8760h`: Thời gian chạy liên tục (ở đây là 1 năm, coi như chạy vô hạn).
- `-c 2`: Số lượng worker/kết nối đồng thời (concurrent workers). Tăng con số này sẽ tạo ra nhiều traffic hơn.
- `-q 10`: Giới hạn tốc độ (Rate limit) cho mỗi worker là 10 queries/second.

Bạn có thể chỉnh sửa các tham số này trong file `loadgenerator.yaml` (nằm tại thư mục này) và apply lại (`kubectl apply -f loadgenerator.yaml`) để thay đổi cường độ giả lập người dùng.
