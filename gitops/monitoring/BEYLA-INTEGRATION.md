# Hướng dẫn Tích hợp Grafana Beyla (Zero-Code Instrumentation)

Tài liệu này mô tả cách hệ thống **Grafana Beyla** thu thập số liệu (metrics) L7 của các microservices thông qua eBPF, cách Prometheus tự động quét các metrics này, và cách RL Agent truy xuất dữ liệu để phục vụ cho cơ chế ra quyết định Canary Rollouts.

---

## 1. Luồng Thu thập Metrics: Beyla -> Prometheus -> RL Agent

### Bước 1: Beyla & eBPF (Thu thập cấp độ Kernel)
- **Cơ chế**: Beyla được triển khai dưới dạng **DaemonSet**, chạy một Pod trên mỗi Node trong Cluster Kubernetes. Nó yêu cầu quyền `hostPID: true` và `privileged: true` để chạy dưới quyền cao nhất.
- **Hoạt động**: Thay vì sửa code ứng dụng, Beyla móc nối (hook) trực tiếp vào các hàm Kernel xử lý socket TCP/HTTP/gRPC của hệ điều hành.
- **Chỉ định mục tiêu**: Trong `beyla-config`, chúng ta thiết lập tham số `discovery` nhắm mục tiêu toàn bộ ứng dụng trong namespace `default`. Beyla tự động quét tất cả các process và gỡ băng (parse) các gói tin HTTP/gRPC bay qua chúng.

### Bước 2: Prometheus (Auto-Scraping bằng ServiceMonitor)
- Beyla tự xuất Metrics (Rate, Error, Duration) mà nó thu thập được ra endpoint `:9090/metrics`.
- Đi kèm với `beyla.yaml`, chúng ta có khai báo một tài nguyên `ServiceMonitor` mang nhãn `release: kube-prometheus-stack`.
- Trình quản lý Prometheus Operator liên tục theo dõi Kubernetes, thấy `ServiceMonitor` này và tự động thêm Beyla vào cấu hình Scraping của nó. Cứ mỗi 15 giây, Prometheus chủ động kéo (pull) metrics L7 từ toàn bộ các Pods Beyla đang chạy.

### Bước 3: RL Agent (Truy xuất qua API)
- Chương trình RL Agent (Python) của chúng ta thực hiện truy vấn HTTP đến API của Prometheus (`http://<prometheus-svc>:9090/api/v1/query`).
- Agent đánh giá sức khỏe của môi trường bằng cách tổng hợp dữ liệu từ các ReplicaSet tương ứng với `STABLE` và `CANARY` để quyết định xem có nên `Promote` hay `Abort`.

---

## 2. Cú pháp PromQL truy vấn Metrics từ Beyla

Khi Beyla bắt được traffic, nó tự động dán nhãn (label) cực kỳ chi tiết bao gồm `k8s_pod_name`, `rpc_method`, `rpc_system_name`...
Dưới đây là các câu lệnh PromQL cốt lõi mà Agent sử dụng để trích xuất số liệu.

### A. Lấy Traffic (Request Rate)
Đếm tổng số lượt request gRPC thành công hướng tới Endpoint `/hipstershop.CheckoutService/PlaceOrder`.

```promql
sum by (k8s_pod_name) (
  rate(
    rpc_server_call_duration_seconds_count{
      rpc_system_name="grpc", 
      rpc_method="/hipstershop.CheckoutService/PlaceOrder", 
      k8s_pod_name=~"checkoutservice.*"
    }[1m]
  )
)
```
- **Ý nghĩa**: Tính tốc độ (rate) số lượng request trong 1 phút gần nhất `[1m]`.
- Lọc những metric sinh ra bởi gRPC (`rpc_system_name="grpc"`).
- Gom nhóm theo tên Pod (`by (k8s_pod_name)`) để chúng ta có thể tách riêng Traffic của Pod Stable và Pod Canary.

### B. Lấy Tỷ lệ Lỗi (Error Rate)
Chỉ đếm những Request có mã trạng thái không hợp lệ. Beyla theo dõi trạng thái trả về của hệ thống và gắn vào nhãn `rpc_response_status_code`. 

```promql
sum by (k8s_pod_name) (
  rate(
    rpc_server_call_duration_seconds_count{
      rpc_system_name="grpc", 
      rpc_method="/hipstershop.CheckoutService/PlaceOrder", 
      rpc_response_status_code!="OK", 
      k8s_pod_name=~"checkoutservice.*"
    }[1m]
  )
)
```
- **Khác biệt**: Bộ lọc `rpc_response_status_code!="OK"` đảm bảo chỉ đếm các giao dịch thất bại (Ví dụ: `INTERNAL`, `UNKNOWN`...).

### C. Lấy Độ trễ P95 (Latency)
Tính độ trễ của 95% số lượng request (95th Percentile) để xem thời gian xử lý đa số của ứng dụng.

```promql
histogram_quantile(0.95, 
  sum(
    rate(
      rpc_server_call_duration_seconds_bucket{
        rpc_system_name="grpc", 
        rpc_method="/hipstershop.CheckoutService/PlaceOrder", 
        k8s_pod_name=~"checkoutservice.*"
      }[1m]
    )
  ) by (le, k8s_pod_name)
) * 1000
```
- Metric kết thúc bằng `_bucket` chứ không phải `_count`. Nó chia sẵn độ trễ thành các dải (buckets) như `<= 0.5s`, `<= 1.0s`, `<= 2.5s`...
- Hàm `histogram_quantile(0.95, ...)` dùng để ngoại suy độ trễ thực tế từ các bucket này.
- Bắt buộc phải gom nhóm bằng `by (le, k8s_pod_name)` vì `le` là dải ranh giới cần thiết cho hàm tính Quantile.
- Nhân thêm `1000` cuối cùng để đổi từ giây (Seconds) sang mili-giây (ms).

---

> **Lưu ý Quan trọng**: 
> Beyla gắn nhãn hệ thống gRPC là `rpc_system_name="grpc"` và trạng thái phản hồi là `rpc_response_status_code`. Hãy cẩn thận vì các công cụ khác (như Prometheus client library thông thường) có thể dùng nhãn `rpc_system` hay `rpc_grpc_status_code`. Bất cứ sự sai lệch nhỏ nào trong tên nhãn cũng sẽ khiến PromQL trả về rỗng (`[]`).
