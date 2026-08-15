# Báo cáo: Nâng cấp Môi trường RL Sandbox từ 5G lên 6G

Tài liệu này tổng hợp toàn bộ quá trình refactor, nâng cấp và gỡ lỗi để đưa RL Agent từ môi trường suy luận 5G (5-kênh) sang mô phỏng hạ tầng mạng 6G (12-kênh).

---

## 1. Bối cảnh và Mục tiêu
Trong môi trường mạng 6G thực tế, độ trễ và tỷ lệ lỗi không chỉ đến từ bản thân ứng dụng (Application Faults) mà còn bị ảnh hưởng nặng nề bởi các yếu tố hạ tầng vật lý (Network-layer Noise) như Handover Storm, NTN Gap (Vệ tinh), hay THz Blockage.

**Mục tiêu cốt lõi:**
RL Agent cần học được cách **phân biệt** giữa biến động metrics do lỗi ứng dụng (Canary) và biến động do hạ tầng mạng (ảnh hưởng đồng thời lên cả Canary và Stable). Nếu chỉ dựa vào các ngưỡng cứng (hard-threshold) hoặc suy luận 5-kênh cũ, Agent sẽ liên tục cảnh báo sai (False Positive) và Rollback nhầm các bản release hoàn toàn khỏe mạnh (Healthy).

---

## 2. Nâng cấp Không gian Quan sát (Observation Space)
Không gian trạng thái (State) của môi trường được mở rộng từ **5 kênh lên 12 kênh** để cung cấp cho mô hình Transformer đầy đủ ngữ cảnh về cả ứng dụng lẫn mạng:

- **Ứng dụng (5 kênh cũ):** `cpu_n`, `mem_n`, `l_ratio_n`, `e_ratio_n`, `weight_n`.
- **Hạ tầng 6G (7 kênh mới):**
  - `handover_n`: Số lượng sự kiện Handover.
  - `sinr_n`: Signal-to-Interference-plus-Noise Ratio.
  - `prb_n`: Mức sử dụng Physical Resource Block.
  - `harq_n`: Tỷ lệ lỗi truyền tải gốc (HARQ NACK).
  - `ntn_gap_n`: Cờ báo mất tín hiệu vệ tinh (Non-Terrestrial Network).
  - `isac_n`: Nhiễu do xung đột Cảm biến & Viễn thông (ISAC Contention).
  - `deploy_age_n`: Thời gian (số step) kể từ lúc bắt đầu rollout.

Mọi kênh đều được chuẩn hóa về khoảng `[0.0, 1.0]` trong `core/feature_pipeline.py`.

---

## 3. Mô phỏng Động lực học Mạng 6G (Network Noise)
Chúng tôi tách biệt hoàn toàn **App Scenario** (Lỗi app) và **Network Scenario** (Nhiễu mạng). 
Trong `core/env.py`, hàm `_network_noise_factor()` được thêm vào để mô phỏng 5 kịch bản mạng 6G:

1. **Stable:** Mạng ổn định (`burst_factor = 1.0`).
2. **HandoverStorm:** Nhiễu dao động liên tục hình sin (`burst_factor` lên tới 1.4).
3. **NTNGap:** Mất kết nối vệ tinh định kỳ (2/12 steps), đẩy `burst_factor = 2.5`.
4. **THzBlockage:** Vật cản sóng THz định kỳ (3/15 steps), đẩy `burst_factor = 3.0`.
5. **ISACContention:** Nhiễu xung đột radar hình sin (`burst_factor` lên tới 1.6).

> [!IMPORTANT]
> **Đặc tính Đối xứng (Symmetry):** `burst_factor` được nhân **đồng thời** vào cả `l_canary` và `l_stable`. Bằng cách này, tỷ lệ `l_ratio` vẫn xoay quanh mức 1.0 nếu app thực sự Healthy, buộc Agent phải nhìn vào kênh 6G để hiểu tại sao Latency tuyệt đối lại tăng vọt, thay vì mù quáng Rollback.

---

## 4. Xử lý Lỗi Tiền Tồn Tại (Pre-existing Bugs)
Quá trình nâng cấp đã phát hiện và xử lý dứt điểm 2 lỗi nghiêm trọng khiến Agent thỉnh thoảng bị phạt oan (False-trigger) ngay cả khi không có nhiễu 6G:

### A. Lỗi kẹp biên (Floor-clamp Asymmetry)
- **Triệu chứng:** Tỷ lệ lỗi (Error Rate) ở môi trường Stable vẫn thỉnh thoảng nhảy vọt lên `e_ratio > 2.0` thuần túy do nhiễu.
- **Nguyên nhân:** Nhiễu cộng (Additive Gaussian Noise `N(0, 0.01)`) quá lớn so với base value (`0.001`). Khi nhiễu âm, giá trị bị kẹp cứng ở `0.0005`. Khi nhiễu dương, giá trị bay tự do. Điều này tạo ra sự mất cân xứng.
- **Giải pháp:** Chuyển sang **Nhiễu nhân (Multiplicative Noise)** cho Error Rate với độ lệch chuẩn tương đối là `8%`. `e_canary = base * (1 + N(0, 0.08))`.

### B. Smoothing bằng Cửa sổ trượt (Rolling-window)
- **Triệu chứng:** Logic thưởng/phạt dựa trên `current_anomalous` chỉ nhìn vào đúng 1 step tức thời.
- **Giải pháp:** Thêm `self.ratio_window` (maxlen=4) trong `env.py`. Agent quyết định `current_anomalous` dựa trên trung bình cộng của 4 step gần nhất. Để tránh sai lệch ở các bước đầu (cold-start), buffer được pre-fill bằng giá trị khởi tạo ngay trong `reset()`.
- Lỗi thứ tự khởi tạo test (`env.scenario` gán sau `reset()`) cũng đã được sắp xếp lại.

---

## 5. Nâng cấp Kiến trúc Mô hình (Model Architecture)
Trong `training/offline_training.py`:
- Cập nhật số luồng `n_features` thành động (đọc trực tiếp từ `env.num_features`).
- Tăng `n_heads_feature = 4` cho TransformerFeatureExtractor để MultiheadAttention có thể xử lý luồng dữ liệu 12-kênh chéo (Cross-attention) tốt hơn.
- Chuyển hẳn định hướng từ Online Training (đã loại bỏ khỏi workflow) sang **Offline Training** sử dụng Stable-Baselines3 PPO.
- Tích hợp ghi log Tensorboard trực tiếp vào `logs/transformer_offline`.

---

## 6. Kết quả Cuối cùng
Sau khi refactor:
- Bộ Unit Test `test_network_noise.py` hoàn thành xuất sắc 29/29 tests.
- Quá trình Offline Training (150,000 timesteps) hội tụ hoàn hảo sau 23 phút.
- Validation Score đạt **12.30 / 20.00**, mô hình Pass bài kiểm tra phân biệt nhiễu mạng và lỗi Canary.

### Tồn đọng cần lưu ý (Known Blocker)
Tập lệnh `online_env.py` cũ hiện vẫn đang hardcode `num_features = 5`. Nếu dự án có kế hoạch sử dụng lại pipeline online inference trong tương lai, file này bắt buộc phải được refactor để đồng bộ với số lượng 12 kênh hiện tại.
