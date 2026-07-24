# RL Agent Services (AI & UI)

Thư mục này chứa mã nguồn của bộ não trung tâm: AI Agent. Ứng dụng này được viết bằng Python (FastAPI) và đảm nhận hai vai trò chính: Đánh giá luồng Canary và Cung cấp giao diện Web trực quan.

## 1. Kiến trúc Thuật Toán Machine Learning
Thuật toán Reinforcement Learning (RL) đang được sử dụng là **TransformerPPO** (Proximal Policy Optimization kết hợp với kiến trúc Transformer). 
Thay vì sử dụng các mạng hồi quy như LSTM trước đây, kiến trúc Transformer giúp Agent nắm bắt được các chuỗi dữ liệu lịch sử (Metrics Sequence) tốt hơn nhờ cơ chế Self-Attention, từ đó phát hiện sớm các dị thường phức tạp về độ trễ, tài nguyên, hoặc tỷ lệ lỗi qua thời gian, và đưa ra quyết định chính xác hơn.

## 2. Luồng Đánh Giá Canary (Inference Flow)
Agent expose các REST API (ví dụ: HTTP POST từ Argo Rollouts `AnalysisTemplate`). Khi một yêu cầu phân tích cập bến:
1. **Lấy tham số**: Agent nhận Hash của bản Stable và Hash của bản Canary.
2. **Kéo Metrics**: Agent gửi PromQL query trực tiếp lên máy chủ Prometheus nội bộ (Linkerd-Viz) để trích xuất mảng dữ liệu (Latency, CPU, RAM, Error Rate) của cả 2 phiên bản trong khoảng thời gian vừa qua.
3. **Tiền xử lý & Suy luận**: Metrics được chuẩn hóa thành dạng chuỗi thời gian (Sequence) và nạp vào mô hình đã huấn luyện (pre-trained `TransformerPPO`). Mô hình trả về 1 trong 3 Actions:
   - `0 (Wait)`: Thiếu dữ liệu hoặc chần chừ, cần chờ thêm.
   - `1 (Promote)`: Bản Canary tốt, cho qua bước tiếp theo.
   - `2 (Abort)`: Phát hiện bất thường, yêu cầu Rollback lập tức để bảo vệ cụm.
4. **Phản hồi**: Action và lý do được bọc trong cấu trúc JSON trả về cho Argo Rollouts thực thi.

## 3. Web UI Dashboard (Quan Sát Tức Thời)
Để giúp các kỹ sư con người có thể theo dõi được "suy nghĩ" bên trong của Agent, hệ thống tích hợp sẵn một Dashboard chạy ngay trên port FastAPI.
- **HTMLResponse**: Root path `/` trả về một giao diện HTML gọn nhẹ mô phỏng đồ thị.
- **WebSockets**: Giao diện HTML này duy trì một kết nối WebSocket với backend (`main.py`). Bất cứ khi nào Agent nhận metrics mới và ra quyết định, nó đẩy (Push) một Broadcast Message qua WebSocket để biểu đồ trên Web UI nhảy dữ liệu ngay lập tức theo thời gian thực. Bằng cách này, người dùng không cần F5 trình duyệt vẫn thấy được sóng metrics và kết quả suy luận.
