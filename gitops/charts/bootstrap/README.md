# GitOps Bootstrap (Environments)

Thư mục này chịu trách nhiệm khởi tạo các môi trường (Namespaces) cơ bản nhất trước khi bất kỳ ứng dụng nào được triển khai lên cụm.

## Kiến trúc 2 Môi trường (Digital Twin)

Hệ thống được thiết kế với tư tưởng mô phỏng song sinh kỹ thuật số (Digital Twin), chia làm 2 namespace chính:

1. **`prod` (Production)**:
   - Môi trường thực tế, phục vụ người dùng cuối. Các chiến lược Rollout ở đây sẽ diễn ra một cách chuẩn mực và an toàn nhất.

2. **`twin` (Digital Twin / Sandbox)**:
   - Đây là một phiên bản mô phỏng y hệt của `prod`.
   - Mục đích của `twin` là tạo ra một môi trường an toàn tuyệt đối để AI Agent (TransformerPPO) được tự do "phá hoại", thử nghiệm và học hỏi (Training) thông qua các kịch bản tiêm lỗi (Chaos Engineering).
   - Agent có thể ra quyết định sai lầm, dẫn đến sập ứng dụng ở `twin` mà không gây ra bất kỳ gián đoạn nào đến khách hàng thật đang truy cập ở `prod`.
