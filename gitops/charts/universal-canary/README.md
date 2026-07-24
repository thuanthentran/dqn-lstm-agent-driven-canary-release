# GitOps Universal Canary (Rollout Flow)

Chart Helm này chứa định nghĩa tiêu chuẩn để bọc (wrap) một Deployment thông thường thành một `Rollout` của Argo, có khả năng chia tách traffic (Linkerd-SMI) và kích hoạt phân tích thông minh qua Agent.

## Luồng Hoạt Động Của Canary Rollout (`templates/rollout.yaml`)

Khi có một phiên bản mới của ứng dụng được deploy (bản Canary), luồng `steps` trong `rollout.yaml` sẽ bắt đầu:

### Sự Tương Tác Giữa Các Thành Phần
1. **Argo Rollouts**: Là bộ não trung tâm quản lý luồng. Nó định hướng Linkerd-SMI bẻ một phần traffic (ví dụ 20%) sang bản Canary mới, trong khi phần còn lại giữ ở bản Stable cũ.
2. **Analysis Template**: Là các mẫu phân tích được Argo gọi ở mỗi bước (Step). Template này định nghĩa tham số đầu vào (như Tên Service, Hash của Pod bản Stable và bản Canary) và gọi HTTP POST đến Agent.
3. **Agent**: Nhận HTTP POST, truy vấn dữ liệu từ Prometheus, phân tích bằng mô hình AI và trả lời lại cho Argo Rollouts (`Promote`, `Abort`, hoặc `Wait`).

### Phân Tách Theo Môi Trường

**Ở môi trường `twin` (Sandbox):**
Luồng thử nghiệm diễn ra quyết liệt và có phần tải ảo (Stress Test):
- **Bước 1 (20% traffic)**: 
  - Kích hoạt `AnalysisRun` gửi thông tin sang Agent để đánh giá. 
  - ĐỒNG THỜI, kích hoạt một Job Load Test (bắn traffic ảo từ `hey` hoặc `locust`) để dồn tải vào bản Canary nhằm làm bộc lộ lỗi sớm nhất có thể.
- **Bước 2 (50% traffic)**: Chờ Agent đánh giá.
- **Bước 3 (80% traffic)**: Chờ Agent đánh giá bước cuối.

**Ở môi trường `prod` (Production):**
Luồng chạy ở đây thận trọng hơn:
- **Bước 1 (1% traffic)**: Bẻ thử 1% traffic và PAUSE (Dừng hẳn để kỹ sư quan sát thủ công hoặc chạy bài test đơn giản).
- **Bước 2 (20% traffic)**: Chuyển giao quyền quyết định cho Agent.
- **Bước 3 (50% traffic)** và **Bước 4 (80% traffic)**: Tương tự như môi trường `twin`, Argo phụ thuộc hoàn toàn vào phán quyết (JSON response) của Agent để đi tiếp hay huỷ bỏ phiên bản mới.
