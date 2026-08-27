# Chaos Middleware cho `checkoutservice`

Module Chaos mới được nhúng trực tiếp vào mã nguồn Go của `checkoutservice`, cho phép tiêm lỗi (Fault Injection) với **6 kịch bản động (patterns)** phức tạp mà không cần dựa vào service mesh hay phải sửa K8s manifest liên tục làm khởi động lại Pod.

## Nguyên lý hoạt động với Argo Rollouts

1. Ta khai báo 1 biến môi trường `CHAOS_CONFIG` chứa chuỗi JSON mô tả kịch bản lỗi trong K8s Manifest (`checkoutservice-values.yaml`).
2. Argo Rollouts phát hiện thay đổi và sinh ra một **Canary Pod** mới nhận biến môi trường này.
3. Pod **Stable** cũ vẫn hoạt động bình thường, không bị lỗi.
4. Pod **Canary** mới sẽ tự động đọc `CHAOS_CONFIG` và tiến hành tiêm lỗi. Vì các mẫu lỗi (như `linear`, `cyclic`) tự động nội suy mức độ lỗi theo thời gian (uptime) ngay bên trong code Go, ta **không cần thay đổi K8s Manifest nữa** trong suốt vòng đời của Canary Pod.

## Cấu trúc Cấu hình (`CHAOS_CONFIG`)

Chuỗi JSON cấu hình yêu cầu các trường sau:

- `enabled`: (bool) Kích hoạt chaos.
- `run_id`: (string) ID của lần chạy để ghi log metadata.
- `seed`: (int) Hạt giống ngẫu nhiên (để đảm bảo khả năng tái lập).
- `resource_safe_mode`: (bool) Nếu `true`, khi `cyclic` kết thúc chu kỳ High, bộ nhớ sẽ được GC và CPU sẽ được giải phóng hoàn toàn, không có dư lượng (residual leak).
- `signals`: (map) Chỉ định tín hiệu mục tiêu (như `latency`, `error`, `cpu`, `mem`) và pattern tương ứng.

Ví dụ:

```json
{
  "enabled": true,
  "run_id": "test-01",
  "seed": 42,
  "resource_safe_mode": true,
  "signals": {
    "error": {
      "pattern": "step_change",
      "params": {
        "before_value": 0.0,
        "after_value": 0.8,
        "trigger_type": "uptime_sec",
        "trigger_value": 120.0
      }
    }
  }
}
```

## Các Mẫu (Patterns) Hỗ Trợ

1. `static`: Cố định 1 giá trị. (Params: `value`)
2. `stochastic`: Lấy mẫu ngẫu nhiên (Uniform hoặc Log-normal). (Params: `min`, `max`, `dist`, `mean_shift`)
3. `linear`: Tăng dần tuyến tính. (Params: `start_value`, `growth_rate_per_sec`, `ceiling`)
4. `step_change`: Cắt ngang giá trị sau 1 thời gian. (Params: `before_value`, `after_value`, `trigger_type`, `trigger_value`)
5. `load_dependent`: Dựa vào lượng Request Per Second (RPS) thực tế. (Params: `rps_threshold`, `value_below`, `value_above`)
6. `cyclic`: Dao động hình chu kỳ. (Params: `value_low`, `value_high`, `period_sec`, `duty_cycle`)

Xem chi tiết trong thư mục `examples/`.

## Ground Truth Logging

Module tự động ghi nhận mọi sự kiện can thiệp (ví dụ mỗi khi delay hoặc trả về mã lỗi 503) vào file:
`/var/log/chaos/ground_truth.jsonl`

File này bao gồm timestamp đồng bộ NTP, ID của run, tham số lỗi, Git Commit và Node Name (qua Downward API). Bạn có thể trích xuất ra để phân tích sau mỗi lần thử nghiệm.
