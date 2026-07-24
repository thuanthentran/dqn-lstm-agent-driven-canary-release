# GitOps Base (CRD & Agent Integration)

Thư mục này chứa các manifest nền tảng để triển khai và tích hợp Tác tử Học tăng cường (RL Agent) vào bên trong cụm Kubernetes.

## 1. Kiến trúc Custom Resource Definition (CRD)

Để Agent không hoạt động như một ứng dụng độc lập, tách rời với K8s, hệ thống được thiết kế theo chuẩn Operator Pattern thông qua thư viện `kopf` của Python.
- **`agent-crd.yaml`**: Định nghĩa một Resource K8s mới có tên là `RLAgent` (`rl.thuanthentran.io`). Bằng cách này, ta có thể khai báo một Agent cho từng luồng Rollout chỉ bằng một file YAML đơn giản (xem `agent-instance.yaml`).

## 2. Agent Controller (Operator)

- **`agent-operator.yaml`**: Deploy một Controller lắng nghe các sự kiện tạo/sửa/xoá trên object `RLAgent`. Khi một `RLAgent` được tạo ra, Controller này sẽ tự động khởi tạo Pod Agent thực thi logic Machine Learning và theo dõi tiến trình Rollout tương ứng.

## 3. Quản lý Phân quyền (RBAC)

Agent cần đặc quyền để có thể can thiệp vào Rollout, lấy trạng thái và điều phối traffic. Việc này được quy định rõ trong `agent-rbac.yaml`:
- **ClusterRole `canary-eval-agent-role`**: Cấp quyền `get, patch, update, list` trên các resource `rollouts` và `rollouts/status` thuộc API Group `argoproj.io`.
- **ClusterRoleBinding**: Ràng buộc quyền này vào `ServiceAccount` mặc định (hoặc tài khoản cụ thể) mà Agent Pod sử dụng. Điều này đảm bảo tính bảo mật chặt chẽ (Principle of Least Privilege), giúp Agent có vừa đủ quyền để thực thi các quyết định Promote/Abort mà không làm tổn hại các tài nguyên hệ thống khác.
