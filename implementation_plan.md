# Review: Task List — Agent bỏ qua CPU/RAM khi ra quyết định Rollback

## Tổng quan

Task list rất kỹ lưỡng và logic chung đúng hướng. Dưới đây là các vấn đề logic tôi tìm thấy sau khi đối chiếu với code hiện tại.

---

## 🔴 Vấn đề logic cần sửa trước khi bắt tay

### 1. Phase 0.3 — `model.py` KHÔNG có `n_features=5` cứng, và KHÔNG có `_raw_to_channels()`

Task list viết:
> Xác nhận `model.py`: `n_features=5`, thứ tự channel `[cpu_n, mem_n, lat_c, err_c, traffic_c]` trong `_raw_to_channels()`.

**Thực tế trong code:**
- [`model.py`](file:///c:/Users/ASUS/Desktop/rl/core/model.py) **không có** hàm `_raw_to_channels()`. Hàm này nằm ở [`env.py` L118-L124](file:///c:/Users/ASUS/Desktop/rl/core/env.py#L118-L124).
- `n_features` trong `model.py` là tham số truyền vào constructor (`n_features: int = 15` — default 15, nhưng thực tế được override bởi `TRANSFORMER_CONFIG` trong [`offline_training.py` L39](file:///c:/Users/ASUS/Desktop/rl/training/offline_training.py#L39) lấy từ `_tmp_env.num_features` = 5).

**Impact**: Phase 0.3 cần chỉnh lại mục tiêu verification — check `env.py._raw_to_channels()` + `env.num_features` + `TRANSFORMER_CONFIG["n_features"]`, không phải `model.py`.

---

### 2. Phase 1.3 — Đổi tên key sẽ **break** `env.py._raw_to_channels()` và `step()`

Task list đề xuất đổi `"cpu_n"` → `"cpu_ratio_n"`, `"mem_n"` → `"mem_ratio_n"`.

**Vấn đề**: [`_raw_to_channels()`](file:///c:/Users/ASUS/Desktop/rl/core/env.py#L118-L124) hiện đọc `norm.get("cpu_n", 0.0)` và `norm.get("mem_n", 0.0)`. Nếu đổi key mà không đồng thời sửa `_raw_to_channels()`, observation sẽ nhận toàn 0 cho CPU/RAM channel → agent mù hoàn toàn.

**Đề xuất**: Phase 1 cần có bước **1.X** đồng bộ key trong `_raw_to_channels()`: `norm.get("cpu_ratio_n", 0.0)`.

---

### 3. Phase 1.6 — Assertion sai khi `cpu_canary = cpu_stable`

Task list viết:
> `cpu_canary = cpu_stable` → `cpu_ratio_n ≈ 0.2`

**Thực tế**: `cpu_ratio = cpu_canary / max(cpu_stable, EPSILON)` → khi `cpu_canary = cpu_stable`, `cpu_ratio = 1.0`. Sau đó `cpu_ratio_n = clip(1.0 / MAX_RATIO, 0, 1) = clip(1.0/5.0, 0, 1) = 0.2`. ✅ Đúng 0.2.

Assertion này đúng, nhưng nên ghi rõ **lý do** (ratio=1.0, chia MAX_RATIO=5.0) để tránh nhầm khi đọc lại.

---

### 4. Phase 3.2 — "30-40% toàn bộ episode healthy" conflict tiềm ẩn với static weight 55-65%

Phase 3.1 nói mỗi **channel** sample `static` với xác suất 55-65%. Xác suất **tất cả 4 channel** đều `static` = (0.6)^4 ≈ **13%**, không phải 30-40%.

**Đề xuất giải quyết**: Có 2 cách:
- **(A)** Tăng per-channel static weight lên ~78% → (0.78)^4 ≈ 37% → đạt 30-40%, nhưng khi đó mỗi channel riêng lẻ hiếm khi faulty → agent ít thấy fault signal.
- **(B)** Sample theo 2 bước: đầu tiên tung đồng xu "episode này healthy hay mixed?" (30-40% healthy), nếu healthy → force 4 channel `static`; nếu mixed → mỗi channel sample ĐỘC LẬP với static ~40-50%.

**Khuyến nghị**: Dùng **(B)** — giữ cả 2 yêu cầu (đủ true-negative + đủ fault diversity) mà không conflict. Đây là vấn đề quan trọng nhất trong task list.

---

### 5. Phase 3.3 — `self.scenario` vẫn cần cho `_build_raw_metrics()` hiện tại (transition risk)

Task list nói xóa `if/elif self.scenario == ...` ở Phase 3.4, nhưng Phase 3.3 nói "KHÔNG còn dùng để chọn công thức". Nếu làm Phase 3.3 trước 3.4, sẽ break `_build_raw_metrics()` giữa chừng.

**Đề xuất**: Đánh dấu 3.3 và 3.4 phải thực hiện **CÙNG LÚC** (atomic commit), hoặc đổi thứ tự: 3.4 trước (thay code), 3.3 sau (cleanup).

---

### 6. Phase 4.1 — `current_anomalous` dùng raw ratio, nhưng thiếu nguồn `cpu_ratio`/`mem_ratio`

Task list viết:
```python
current_anomalous = (
    e_ratio > 2.0 or l_ratio > 2.0 or
    cpu_ratio > 2.0 or mem_ratio > 2.0
)
```

Hiện tại `step()` chỉ tính `e_ratio` và `l_ratio` từ `self.latest_raw`. Sau Phase 1, `self.latest_raw` sẽ có key `cpu_canary`, `cpu_stable`, `mem_canary_mb`, `mem_stable_mb` — cần thêm 2 dòng tính `cpu_ratio` và `mem_ratio` tương tự `e_ratio`/`l_ratio`:

```python
cpu_ratio = self.latest_raw["cpu_canary"] / max(self.latest_raw["cpu_stable"], EPSILON)
mem_ratio = self.latest_raw["mem_canary_mb"] / max(self.latest_raw["mem_stable_mb"], EPSILON)
```

Task list bỏ sót bước tính này — chỉ nói "thêm vào điều kiện" mà chưa nói tính ở đâu.

---

### 7. Phase 4 — Thiếu sửa auto-promote terminal check ở [L178-L183](file:///c:/Users/ASUS/Desktop/rl/core/env.py#L178-L183)

```python
if not self.done and self.weight >= 1.0:
    self.done = True
    if (norm["e_ratio_n"] <= 0.4) and (norm["l_ratio_n"] <= 0.4):
        reward += 10.0
    else:
        reward -= 10.0
```

Khi `weight >= 1.0`, điều kiện thành công chỉ check `e_ratio_n` và `l_ratio_n`, **thiếu** `cpu_ratio_n` và `mem_ratio_n`. Nghĩa là agent có thể "promote hết traffic" và nhận +10 reward trong khi CPU/RAM đang bất thường.

**Đề xuất**: Thêm task **4.4** — sửa terminal check ở L180 để include `cpu_ratio_n <= 0.4 and mem_ratio_n <= 0.4`.

---

### 8. Phase 5.1 — Earliness bonus cần cẩn thận với reward magnitude

Hiện tại rollback-sai penalty = -22.0, rollback-đúng = +5.0. Nếu thêm bonus:
```python
reward += 5.0 + bonus  # bonus = 2.0 * (50 - step) / 50
```
Ở step 1: bonus = 2.0 * 49/50 ≈ 1.96 → total = 6.96.
Ở step 50: bonus ≈ 0 → total = 5.0.

Gap giữa rollback-sai (-22.0) và rollback-đúng-sớm (+6.96) = 28.96, đủ lớn. ✅ Không có vấn đề incentive perverse.

Nhưng lưu ý: nếu `EARLY_BONUS_SCALE` được Optuna tune lên cao (vd 2.5), bonus max = 2.5 * 49/50 = 2.45, total max = 7.45. Vẫn an toàn. ✅

---

### 9. Phase 6.1 — Cần xác nhận `n_features` vẫn đúng, nhưng `STATE_KEYS` sẽ thay đổi

Sau Phase 1, `STATE_KEYS` sẽ có `cpu_ratio_n`, `mem_ratio_n` thay vì `cpu_n`, `mem_n`. Số lượng state keys = 8, nhưng `env._raw_to_channels()` chỉ dùng 5 trong số đó → `n_features=5` đúng. ✅

Tuy nhiên nếu tương lai muốn thêm `e_gap_n`/`l_gap_n`/`rps_n` vào observation, cần update. Hiện tại task list không đề cập → OK.

---

### 10. Phase 8 — `validate_model_locally()` dùng `VecNormalize` wrapper → `env.reset()` bị wrap

[`validate_model_locally()`](file:///c:/Users/ASUS/Desktop/rl/training/offline_training.py#L98-L185) gọi `eval_env.reset()` qua `VecNormalize(DummyVecEnv(...))`. Để truyền `episode_config`, cần:
- Hoặc unwrap để gọi `env_instance.reset(episode_config=...)` rồi wrap lại observation
- Hoặc hack qua `env_instance` trước reset

Task list chưa đề cập kỹ thuật unwrap này. Cần thêm sub-task **8.1a**: thiết kế cách truyền `episode_config` qua `VecNormalize`/`DummyVecEnv` wrapper (ví dụ: set `env_instance._next_episode_config` trước khi gọi `eval_env.reset()`, rồi trong `CanaryEnv.reset()` check `self._next_episode_config`).

---

### 11. Phase 8.2 — `evaluate_rule_based_baseline()` cũng cần sửa decision logic, không chỉ FPR/FNR

Hiện tại rule-based chỉ check `e_ratio` và `l_ratio` ([L113](file:///c:/Users/ASUS/Desktop/rl/training/sweep_optuna.py#L113)). Sau Phase 1, nên thêm `cpu_ratio_n` và `mem_ratio_n` vào rule-based decision để so sánh **công bằng** — rule-based cũng phải "nhìn thấy" CPU/RAM.

Task list chưa đề cập. Nên thêm **8.2a**: update rule-based logic để dùng 4 channels.

---

### 12. Phase 9.2 — "Chạy model đã train" — model nào?

Phase 9 nói chạy "model đã train" nhưng Phase 5.3 mới train 30k timesteps (test so sánh). Cần clarify:
- Nếu chạy Phase 5.3 xong thì dùng model đó? Hay cần train full trước Phase 9?
- **Đề xuất**: Thêm **Phase 8.5** — train full model với tất cả thay đổi từ Phase 1-8, SAU ĐÓ mới chạy Phase 9 acceptance test.

---

## 🟡 Cải thiện nhỏ / Ghi chú

### A. `_build_raw_metrics()` trả về `rps` — cần duy trì

Sau Phase 3.4, `_build_raw_metrics()` mới vẫn cần trả về `rps` cho:
- `feature_pipeline.py` tính `rps_n`
- `generate_channel_value()` pattern `load_dependent` dùng `rps`

Task list không đề cập rõ — nên ghi chú giữ `rps` trong raw metrics.

### B. `traffic_c` (weight_n) là channel thứ 5 — vẫn giữ nguyên

`_raw_to_channels()` trả về `[cpu, mem, lat, err, traffic]`. Sau khi đổi key, traffic vẫn lấy từ `norm["weight_n"]`. ✅ Nhưng nên confirm thứ tự channel không đổi.

### C. `RunningFeatureStats` trong `feature_pipeline.py` dùng `STATE_KEYS`

Class này track stats cho logging. Sau Phase 1.5 đổi `STATE_KEYS`, class này tự động cập nhật nếu caller truyền đúng. Không cần sửa class, nhưng cần sửa caller nếu có nơi nào hardcode old keys.

### D. `generate_channel_value()` — nên return (canary_value, stable_value) tuple

Phase 2.1 chỉ nói return 1 float. Nhưng Phase 1 cần **cả** `cpu_canary` **và** `cpu_stable`. Generator cần sinh cả baseline + anomalous value, hoặc tách thành 2 call (1 cho stable, 1 cho canary). Task list chưa rõ điểm này.

**Đề xuất**: `generate_channel_value()` chỉ sinh **1 giá trị** (dùng cho canary side). Stable/baseline value được lấy trực tiếp từ `episode_config["baseline"]` + noise nhỏ, không qua generator. Cần ghi rõ trong Phase 2.1.

---

## Tóm tắt các action item

| # | Mức độ | Mô tả |
|---|--------|-------|
| 1 | 🔴 | Phase 0.3: `_raw_to_channels()` nằm ở `env.py`, không phải `model.py` |
| 2 | 🔴 | Phase 1: Thiếu bước đồng bộ key mới trong `_raw_to_channels()` |
| 3 | 🔴 | Phase 3.1 + 3.2: Xác suất all-healthy = (0.6)^4 ≈ 13%, không đạt 30-40%. Cần 2-stage sampling |
| 4 | 🔴 | Phase 3.3 + 3.4: Phải atomic, không thể làm 3.3 trước 3.4 |
| 5 | 🔴 | Phase 4.1: Thiếu bước tính `cpu_ratio`/`mem_ratio` từ raw metrics trong `step()` |
| 6 | 🔴 | Phase 4: Thiếu sửa terminal check L178-183 cho CPU/RAM |
| 7 | 🔴 | Phase 8.1: Thiếu kỹ thuật truyền `episode_config` qua VecNormalize wrapper |
| 8 | 🟡 | Phase 8.2: Rule-based cần thêm CPU/RAM vào decision logic |
| 9 | 🟡 | Phase 9: Cần clarify "train full model" trước acceptance test |
| 10 | 🟡 | Phase 2.1: Clarify `generate_channel_value()` chỉ sinh canary value, stable lấy từ config |
| 11 | 🟡 | Phase 3: Duy trì `rps` trong raw metrics cho `load_dependent` pattern |
