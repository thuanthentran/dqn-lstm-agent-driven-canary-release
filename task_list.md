# Task List: Fix "Agent bỏ qua CPU/RAM khi quyết định Rollback"

> Bàn giao cho coding agent thực thi. Mỗi task nhỏ, có file/anchor cụ thể, và bước test riêng.
> **Làm đúng thứ tự phase. Không skip. Sau mỗi phase PHẢI chạy lệnh test ghi trong phase đó trước khi sang phase tiếp theo.**

## Nguyên tắc bắt buộc (áp dụng xuyên suốt mọi phase)

1. **EPSILON**: mọi phép chia chống chia-cho-0 dùng `EPSILON = 1e-6` (import từ `core/feature_pipeline.py`), **KHÔNG** dùng hằng số "baseline hợp lý" như `0.04` hay `12.0` làm floor cho mẫu số. Pattern chuẩn (đã có sẵn cho error/latency, phải copy y hệt cho cpu/mem):
   ```python
   e_ratio = self.latest_raw["e_canary"] / max(self.latest_raw["e_stable"], EPSILON)
   ```
2. **Domain randomization khi training**: không dùng scenario cố định. Scenario/kịch bản cố định **chỉ** dùng ở eval (Phase 8-9).
3. **Earliness bonus đối xứng**: cả Promote-đúng và Rollback-đúng đều phải có early bonus. Bắt buộc, không phải optional.
4. **`silent_degradation` ngoài scope** — không đụng tới, không thêm channel/scenario mới ngoài cpu/mem/latency/error/traffic.
5. Sau khi sửa xong 1 file, chạy nhanh `python -c "import core.env"` (hoặc file tương ứng) để bắt lỗi syntax trước khi chạy test đầy đủ.
6. **⚠️ Số dòng chỉ đúng cho Phase 1 (đối chiếu với file gốc chưa sửa gì).** Từ Phase 2 trở đi, file đã bị các phase trước chèn/xóa dòng nên số dòng ghi trong task KHÔNG còn chính xác — **luôn định vị code bằng cách search tên hàm hoặc đoạn text đặc trưng** (vd search `"if not self.done and self.weight >= 1.0:"` thay vì tin vào "dòng 178-183"), rồi mới sửa. Nếu số dòng ghi trong task không khớp với những gì đang thấy trong file thật, tin vào **tên hàm/text search**, không tin vào số dòng.

---

## Phase 0 — Xác minh baseline (KHÔNG sửa code, chỉ đọc + ghi chú)

- [ ] **0.1** Mở [core/env.py](core/env.py) dòng 21-38. Xác nhận: `num_features = 5`, thứ tự channel trong docstring là `[CPU, RAM, Latency, Error_Rate, Traffic_Pct]`. Ghi chú lại thứ tự này — Phase 1-4 phải giữ nguyên thứ tự, không đảo.
- [ ] **0.2** Mở `_raw_to_channels()` ở [core/env.py:118-124](core/env.py#L118-L124). Đây là hàm build 5-channel array, **không phải** ở `core/model.py`. `core/model.py` không có hàm `_raw_to_channels()` — `n_features` ở đó chỉ là tham số constructor, được set từ `TRANSFORMER_CONFIG["n_features"]` trong [training/offline_training.py:39](training/offline_training.py#L39) (lấy từ `_tmp_env.num_features`).
- [ ] **0.3** Mở [core/feature_pipeline.py:24-33](core/feature_pipeline.py#L24-L33). Ghi chú: `STATE_KEYS` có 8 keys (`weight_n, e_ratio_n, l_ratio_n, e_gap_n, l_gap_n, cpu_n, mem_n, rps_n`) nhưng `_raw_to_channels()` chỉ dùng 5 trong số đó (`cpu_n, mem_n, l_ratio_n, e_ratio_n, weight_n`). `e_gap_n/l_gap_n/rps_n` chỉ dùng cho `RunningFeatureStats` logging, không vào observation. Giữ nguyên, không cần sửa gì ở Phase này.
- [ ] **0.4** Ghi chú phát hiện quan trọng: ở [core/env.py:103](core/env.py#L103), `cpu = max(0.0001, 0.001 + (self.weight * 0.01) + noise())` — **không phụ thuộc `self.scenario`**. Tương tự `mem` ở dòng 104 chỉ cộng thêm `8.0` nếu `scenario==1`, các scenario khác không đụng tới mem. Đây là lý do gốc CPU/RAM chưa từng mang tín hiệu bất thường phân biệt được trong training data.

**Test Phase 0**: Không có, đây là bước đọc.

---

## Phase 1 — Đổi CPU/RAM từ "absolute value vs fixed threshold" sang "canary vs stable ratio"

Hiện tại CPU/RAM dùng 1 giá trị tuyệt đối, normalize bằng hằng số cố định (`CPU_REF=0.02`, `MEM_REF_MB=128.0`) — khác hoàn toàn cách error/latency đang làm (ratio canary/stable). Phase này đổi CPU/RAM sang đúng pattern ratio như error/latency.

> ℹ️ **Lưu ý**: công thức `cpu_stable`/`mem_stable`/`cpu_canary`/`mem_canary_mb` thêm ở 1.1/1.2 dưới đây là **scaffolding tạm thời** — mục đích là để test nhanh key/ratio math đúng trước (Phase 1.8), trước khi Phase 3.2 viết đè **toàn bộ** `_build_raw_metrics()` bằng cơ chế generator + baseline log-uniform. Đừng ngạc nhiên khi công thức ở 1.1/1.2 "biến mất" ở Phase 3 — đó là chủ đích, không phải lỗi.

- [ ] **1.1** Trong [core/env.py](core/env.py), hàm `_build_raw_metrics()` (dòng 79-116): thêm 2 biến stable riêng cho cpu và mem, theo đúng pattern `e_stable`/`l_stable` đã có ở đầu hàm (dòng 81-82):
   ```python
   cpu_stable = max(0.0001, 0.001 + noise())
   mem_stable = max(12.0, 24.0 + noise(20.0))
   ```
   Đặt 2 dòng này ngay sau dòng `l_stable = ...` (dòng 82).

- [ ] **1.2** Đổi dòng 103-104 (hiện là `cpu = ...` và `mem = ...`) thành `cpu_canary` và `mem_canary_mb`, giữ nguyên công thức tính hiện tại (không đổi logic tăng theo `weight`/`scenario` ở bước này — việc gắn logic bất thường theo channel độc lập sẽ làm ở Phase 2-3):
   ```python
   cpu_canary = max(0.0001, 0.001 + (self.weight * 0.01) + noise())
   mem_canary_mb = max(12.0, 24.0 + (self.weight * 20.0) + (8.0 if self.scenario == 1 else 0.0) + noise(20.0))
   ```

- [ ] **1.3** Đổi return dict của `_build_raw_metrics()` (dòng 107-116): thay `"cpu": float(cpu)` → `"cpu_canary": float(cpu_canary), "cpu_stable": float(cpu_stable)`; thay `"mem_mb": float(mem)` → `"mem_canary_mb": float(mem_canary_mb), "mem_stable_mb": float(mem_stable)`. **Giữ nguyên `"rps": float(rps)`** — không đụng, cần cho `load_dependent` pattern ở Phase 2 và cho `rps_n` trong feature_pipeline.

- [ ] **1.4** Trong [core/feature_pipeline.py](core/feature_pipeline.py):
   - Đổi `RAW_KEYS` (dòng 13-22): thay `"cpu"` → `"cpu_canary", "cpu_stable"`; thay `"mem_mb"` → `"mem_canary_mb", "mem_stable_mb"`. Giữ `"rps"`.
   - Xóa 2 hằng số `CPU_REF` và `MEM_REF_MB` (dòng 7-8) — không còn dùng, vì CPU/RAM giờ normalize bằng ratio + `MAX_RATIO` giống error/latency, không còn normalize bằng absolute threshold.

- [ ] **1.5** Trong `normalize_raw_metrics()` ([core/feature_pipeline.py:40-62](core/feature_pipeline.py#L40-L62)):
   - Thêm tính `cpu_ratio` và `mem_ratio` theo đúng pattern `e_ratio`/`l_ratio` (dòng 46-47), dùng **EPSILON thuần túy** làm floor mẫu số (theo Nguyên tắc #1 ở đầu file, KHÔNG dùng `0.04`/`12.0` làm floor chia):
     ```python
     cpu_ratio = float(raw["cpu_canary"]) / max(float(raw["cpu_stable"]), EPSILON)
     mem_ratio = float(raw["mem_canary_mb"]) / max(float(raw["mem_stable_mb"]), EPSILON)
     ```
   - Trong dict `state` trả về (dòng 51-60): xóa 2 dòng `"cpu_n": ...` và `"mem_n": ...`, thay bằng:
     ```python
     "cpu_ratio_n": _clip(cpu_ratio / MAX_RATIO, 0.0, 1.0),
     "mem_ratio_n": _clip(mem_ratio / MAX_RATIO, 0.0, 1.0),
     ```

- [ ] **1.6** Đổi `STATE_KEYS` ([core/feature_pipeline.py:24-33](core/feature_pipeline.py#L24-L33)): thay `"cpu_n"` → `"cpu_ratio_n"`, `"mem_n"` → `"mem_ratio_n"`.

- [ ] **1.7 (QUAN TRỌNG — bắt buộc làm cùng lúc với 1.5/1.6, không được tách)** Trong [core/env.py](core/env.py), hàm `_raw_to_channels()` (dòng 118-124): đổi
   ```python
   cpu_c = norm.get("cpu_n", 0.0)
   mem_c = norm.get("mem_n", 0.0)
   ```
   thành
   ```python
   cpu_c = norm.get("cpu_ratio_n", 0.0)
   mem_c = norm.get("mem_ratio_n", 0.0)
   ```
   Đây là bước bắt buộc nhất trong Phase 1 — vì `.get()` không raise lỗi khi key sai, nếu bỏ sót bước này thì observation sẽ nhận toàn 0 cho CPU/RAM channel một cách âm thầm (agent mù hoàn toàn mà không có lỗi nào báo).
   Thứ tự return array **giữ nguyên** `[cpu_c, mem_c, lat_c, err_c, traffic_c]` — không đảo (theo Nguyên tắc 0.1).

- [ ] **1.8** Thêm assertion test (ghi chú rõ lý do trong comment, vì `cpu_canary == cpu_stable` cho ratio=1.0 chứ không phải 0): tạo file tạm `scratch_test_phase1.py` ở repo root:
   ```python
   from core.feature_pipeline import normalize_raw_metrics

   raw = {
       "weight_pct": 50.0, "e_canary": 0.001, "e_stable": 0.001,
       "l_canary": 0.095, "l_stable": 0.095,
       "cpu_canary": 0.01, "cpu_stable": 0.01,
       "mem_canary_mb": 30.0, "mem_stable_mb": 30.0,
       "rps": 20.0,
   }
   norm = normalize_raw_metrics(raw)
   # cpu_canary == cpu_stable => ratio = 1.0 => ratio_n = clip(1.0 / MAX_RATIO(5.0), 0, 1) = 0.2
   assert abs(norm["cpu_ratio_n"] - 0.2) < 1e-6, norm["cpu_ratio_n"]
   assert abs(norm["mem_ratio_n"] - 0.2) < 1e-6, norm["mem_ratio_n"]
   print("Phase 1 normalize OK:", norm)
   ```

**Test Phase 1**:
```bash
python scratch_test_phase1.py
python -c "
from core.env import CanaryEnv
env = CanaryEnv()
obs, _ = env.reset()
print('obs shape:', obs.shape)  # kỳ vọng (30, 5)
print('latest_raw keys:', sorted(env.latest_raw.keys()))
assert 'cpu_canary' in env.latest_raw and 'cpu_stable' in env.latest_raw
assert 'mem_canary_mb' in env.latest_raw and 'mem_stable_mb' in env.latest_raw
obs2, r, done, trunc, info = env.step(1)
print('step OK, reward:', r)
print('ALL PHASE 1 CHECKS PASSED')
"
```
Xóa `scratch_test_phase1.py` sau khi pass.

---

## Phase 2 — Generator sinh giá trị canary theo channel (thay cho if/elif scenario)

- [ ] **2.1** Trong [core/env.py](core/env.py), thêm hàm mới `generate_channel_value(channel: str, pattern: str, weight: float, step_count: int, baseline: float, noise_fn) -> float`, trong đó `noise_fn` là callable **không tham số** (đã bind sẵn `baseline` từ closure — xem `relative_noise()` ở Phase 3.2) trả về 1 sample noise tương đối theo baseline. Hàm này **chỉ sinh 1 giá trị — giá trị canary side**. Giá trị stable/baseline **không** đi qua hàm này — lấy trực tiếp từ `baseline` (từ `episode_config`) cộng noise nhỏ độc lập, y hệt cách `e_stable`/`l_stable`/`cpu_stable`/`mem_stable` hiện đang được tính (một dòng `max(floor, baseline + noise_fn())`).

- [ ] **2.2** Định nghĩa tối thiểu 3 pattern trong `generate_channel_value()`, tái sử dụng công thức đã có sẵn trong `_build_raw_metrics()` cũ (scenario 1/2/3) làm gốc — chỉ đổi từ "chọn theo scenario" sang "chọn theo channel + pattern độc lập". ⚠️ **Vì baseline giờ log-uniform theo Phase 3.1b (dao động 0.2x-5x mỗi episode), mọi offset cộng thêm PHẢI tỉ lệ với `baseline` (nhân hệ số), KHÔNG được cộng hằng số tuyệt đối** — nếu không, pattern sẽ vô nghĩa (quá nhỏ so với baseline lớn, hoặc lấn át hoàn toàn baseline nhỏ):
   - `"static"`: trả về `baseline + noise_fn()` (không có anomaly) — dùng công thức tương tự nhánh `scenario==0` cũ.
   - `"leak"` (giống scenario 1 cũ — Resource Leak): tăng dần theo `weight` và `step_count`, **tỉ lệ với baseline**, ví dụ: `baseline * (1.0 + weight * 6.0 + step_count * 0.1) + noise_fn()`. Áp dụng tương tự cho error/cpu/mem với hệ số nhân riêng (không nhất thiết giống hệt latency, nhưng phải theo dạng `baseline * (1 + f(weight, step_count))`).
   - `"threshold_spike"` (giống scenario 2 "Ticking Bomb" cũ): chỉ tăng đột biến khi `weight > 0.25`, dạng `baseline * (1.0 + max(0, weight - 0.25) * 15.0) + noise_fn()` — công thức tỉ lệ hóa từ dòng 91-95 cũ.
   - `"load_dependent"`: hệ số nhân tỉ lệ thuận với `rps` (thay vì baseline cộng thẳng rps) — dùng cho CPU/mem khi muốn anomaly chỉ xuất hiện lúc traffic cao, dạng `baseline * (1.0 + (rps / RPS_REF) * hệ_số)`. Cần truyền `rps` vào hàm hoặc tính rps trước và truyền vào.

- [ ] **2.3** Không cần return tuple `(canary, stable)` — giữ đúng như 2.1 đã quyết định: 1 giá trị canary, stable tính riêng ở nơi gọi.

**Test Phase 2**:
```bash
python -c "
from core.env import CanaryEnv
env = CanaryEnv()
import inspect
assert hasattr(env, 'generate_channel_value'), 'thiếu hàm generate_channel_value'
sig = inspect.signature(env.generate_channel_value)
print('signature OK:', sig)
"
```

---

## Phase 3 — Domain randomization per-channel (thay scenario cố định)

### 3.1 — 2-stage sampling trong `reset()`

- [ ] **3.1a** Trong [core/env.py](core/env.py), hàm `reset()` (dòng 55-77): thay logic `self.scenario = random.randint(0, 4)` bằng 2-stage sampling:
   ```python
   HEALTHY_EPISODE_PROB = 0.35  # nằm trong khoảng 30-40% theo yêu cầu
   PER_CHANNEL_STATIC_PROB = 0.45  # trong episode "mixed", mỗi channel độc lập ~45% static

   is_healthy_episode = random.random() < HEALTHY_EPISODE_PROB
   channels = ["cpu", "mem", "latency", "error"]
   if is_healthy_episode:
       channel_anomalous = {c: False for c in channels}
   else:
       channel_anomalous = {c: (random.random() >= PER_CHANNEL_STATIC_PROB) for c in channels}
   ```
   Ghi chú lý do trong comment: xác suất all-healthy nếu mỗi channel sample độc lập ~60% static chỉ đạt `(0.6)^4 ≈ 13%`, không đạt 30-40% theo yêu cầu — nên dùng coin-flip 2 tầng: tầng 1 quyết định "episode này healthy hay mixed" (30-40%), tầng 2 chỉ áp dụng khi mixed, cho phép mỗi channel bất thường độc lập mà không kéo xác suất all-healthy xuống quá thấp.

- [ ] **3.1b** Với mỗi channel có `channel_anomalous[c] == True`, random chọn 1 pattern trong `["leak", "threshold_spike", "load_dependent"]` (từ Phase 2.2). Lưu vào `self.episode_config = {"channel_anomalous": channel_anomalous, "channel_pattern": {...}, "baseline": {...}}`. `baseline` lưu giá trị stable gốc cho từng channel — **không dùng hằng số cố định**, mà sample ngẫu nhiên mỗi episode theo log-uniform quanh center cũ, để agent thấy được cả service "nhẹ" lẫn "nặng" (domain randomization trên cả baseline, không chỉ trên anomaly injection).

   Thêm vào [core/env.py](core/env.py) (module-level, gần `SCENARIO_NAMES`/`ACTION_NAMES`):
   ```python
   BASELINE_CENTERS = {
       "error":   0.001,   # giữ đúng scale cũ (e_stable gốc)
       "latency": 0.095,   # giữ đúng scale cũ (l_stable gốc)
       "cpu":     0.001,   # giữ đúng scale cũ (cpu_stable gốc)
       "mem":     24.0,    # giữ đúng scale cũ (mem_stable gốc)
   }
   ```
   Thêm method `_sample_baseline(self, channel: str) -> float` trong `CanaryEnv`:
   ```python
   def _sample_baseline(self, channel: str) -> float:
       center = BASELINE_CENTERS[channel]
       # Log-uniform scale factor: x0.2 đến x5.0 quanh center.
       # Log-uniform (thay vì uniform) khiến đa số episode vẫn tập trung gần center,
       # nhưng vẫn có đủ số episode "siêu nhẹ" (0.2x) lẫn "siêu nặng" (5x) để agent
       # học được ratio-based detection thay vì học thuộc absolute value.
       scale = float(np.exp(np.random.uniform(np.log(0.2), np.log(5.0))))
       return center * scale
   ```
   Trong `reset()`, khi build `episode_config["baseline"]`, gọi `self._sample_baseline(ch)` cho từng channel trong `["error", "latency", "cpu", "mem"]` — mỗi channel sample **độc lập** (baseline của error không liên quan tới baseline của cpu).

- [ ] **3.1c** Giữ lại `self.scenario` nhưng đổi ý nghĩa thành nhãn coarse chỉ dùng cho logging/nhóm kết quả (KHÔNG dùng để chọn công thức nữa): `self.scenario = 0 if is_healthy_episode else 1`. Đây là điều chỉnh bắt buộc phát sinh từ Phase 3 — vì [training/offline_training.py:119,135](training/offline_training.py#L119) và [training/sweep_optuna.py:104,126](training/sweep_optuna.py#L104) đang dùng `env_instance.scenario` và `scenario in [0, 4]` để nhóm Healthy vs Anomalous khi validate. Nếu không giữ 1 nhãn coarse tương thích, Phase 8 sẽ crash hoặc nhóm sai. Sẽ update các nơi dùng `scenario in [0, 4]` → `scenario == 0` ở Phase 8.

- [ ] **3.1d (⚠️ RÀO CHẮN BẮT BUỘC)** `self.scenario` (nhãn coarse) chỉ được phép đọc ở 2 nơi: (a) code logging/reporting trong `validate_model_locally()` và `evaluate_rule_based_baseline()` (Phase 8, để group FPR/FNR khi in báo cáo), và (b) `reset()` để gán giá trị nó. **TUYỆT ĐỐI KHÔNG** được đọc `self.scenario` ở bất kỳ đâu ảnh hưởng tới training hoặc quyết định của agent — cụ thể là **không** được đọc trong `_build_raw_metrics()`, `step()` (reward, `current_healthy`, `current_anomalous`, terminal check), `generate_channel_value()`, hay bất kỳ input nào đưa vào observation/model. Lý do: nhãn coarse chỉ có 2 giá trị (0/1) và không mang thông tin channel nào đang anomalous — nếu lọt vào logic reward/observation, nó sẽ hoặc vô nghĩa hoặc (tệ hơn) tạo ra một kênh thông tin "rò rỉ" ngoài 5 channel quan sát chính thức, làm sai lệch bài toán RL.
   Verify bằng grep sau khi hoàn thành Phase 3 + Phase 8:
   ```bash
   grep -n "self.scenario\|env_instance.scenario\|\.scenario\b" core/env.py training/offline_training.py training/sweep_optuna.py
   ```
   Đối chiếu thủ công từng dòng kết quả: chỉ được xuất hiện trong `reset()` (gán) và trong đoạn group/log của 2 hàm eval — nếu thấy nó xuất hiện trong `step()` hay `_build_raw_metrics()` thì đó là lỗi, phải sửa ngay.

### 3.2 — Atomic rewrite `_build_raw_metrics()`

- [ ] **3.2** ⚠️ **Làm 3.2 và xóa if/elif cũ TRONG CÙNG MỘT LẦN SỬA, không tách làm 2 bước** — nếu chỉ xóa if/elif mà chưa nối generator mới vào, `_build_raw_metrics()` sẽ crash ngay (thiếu biến `e_canary`/`l_canary`).

  ⚠️ **Lưu ý bắt buộc về noise scale**: noise cũ (`noise = lambda s=1.0: np.random.normal(0, 0.01*s)`) là absolute std cố định, được tune thủ công cho từng scale riêng (`noise()` cho error/latency/cpu, `noise(20.0)` cho mem). Từ Phase 3.1b, `baseline` giờ dao động log-uniform 0.2x-5x mỗi episode — nếu giữ noise absolute cố định, ở baseline bị scale nhỏ (vd error baseline = 0.001*0.2 = 0.0002), noise std=0.01 sẽ lớn gấp ~50 lần baseline và lấn át hoàn toàn tín hiệu; ngược lại ở baseline scale lớn, noise trở nên không đáng kể. Phải đổi sang **noise tương đối theo baseline**:
  ```python
  def relative_noise(baseline: float, rel_std: float = 0.01) -> float:
      return float(np.random.normal(0, rel_std * baseline))
  ```

  Viết lại toàn bộ `_build_raw_metrics()` (dòng 79-116):
  ```python
  def _build_raw_metrics(self):
      cfg = self.episode_config
      rps = max(0.1, 40.0 * self.weight + np.random.normal(0, 2.0))

      values = {}
      for ch, key_canary, key_stable, baseline_key in [
          ("error", "e_canary", "e_stable", "error"),
          ("latency", "l_canary", "l_stable", "latency"),
          ("cpu", "cpu_canary", "cpu_stable", "cpu"),
          ("mem", "mem_canary_mb", "mem_stable_mb", "mem"),
      ]:
          baseline = cfg["baseline"][baseline_key]
          stable_val = max(baseline * 0.5, baseline + relative_noise(baseline))
          values[key_stable] = float(stable_val)
          if cfg["channel_anomalous"][ch]:
              pattern = cfg["channel_pattern"][ch]
              canary_val = self.generate_channel_value(
                  ch, pattern, self.weight, self.step_count, baseline,
                  lambda b=baseline: relative_noise(b),
              )
          else:
              canary_val = baseline + relative_noise(baseline)
          values[key_canary] = float(max(0.0, canary_val))

      return {
          "weight_pct": float(self.weight * 100.0),
          "e_canary": values["e_canary"], "e_stable": values["e_stable"],
          "l_canary": values["l_canary"], "l_stable": values["l_stable"],
          "cpu_canary": values["cpu_canary"], "cpu_stable": values["cpu_stable"],
          "mem_canary_mb": values["mem_canary_mb"], "mem_stable_mb": values["mem_stable_mb"],
          "rps": float(rps),
      }
  ```
  (Đây là khung sườn — coding agent cần điều chỉnh `generate_channel_value()` ở Phase 2 để nhận `noise_fn` không tham số thay vì `noise_fn(scale)`, do noise giờ tính theo baseline thay vì theo hệ số `s` cố định như bản cũ. Rà lại toàn bộ Phase 2.2 để đảm bảo các pattern `"leak"/"threshold_spike"/"load_dependent"` cộng thêm offset **tỉ lệ với `baseline`** — vd `baseline * 20.0 * weight` thay vì hằng số tuyệt đối `weight * 0.6` — nếu không, các pattern anomaly sẽ vô nghĩa khi baseline bị scale nhỏ/lớn.)

- [ ] **3.3** Xác nhận không còn đoạn `if getattr(self, "scenario", 0) == 0: ... elif self.scenario == 1: ...` nào sót lại trong `_build_raw_metrics()`. `self.scenario` giờ chỉ đọc ở `reset()`/logging, không đọc trong `_build_raw_metrics()`.

- [ ] **3.4** `episode_config` phải được set **trước** lần gọi `_build_raw_metrics()` đầu tiên trong `reset()` — kiểm tra thứ tự dòng: 3.1a/3.1b (set `self.episode_config`) phải nằm trước dòng gọi `raw = self._build_raw_metrics()` hiện tại (dòng 68).

**Test Phase 3**:
```bash
python -c "
from core.env import CanaryEnv
import random
random.seed(42)
env = CanaryEnv()
healthy_count = 0
N = 2000
for _ in range(N):
    env.reset()
    if not any(env.episode_config['channel_anomalous'].values()):
        healthy_count += 1
rate = healthy_count / N
print(f'all-healthy rate: {rate*100:.1f}%  (kỳ vọng 30-40%)')
assert 0.25 < rate < 0.45, f'Sai xác suất all-healthy: {rate}'
print('ALL PHASE 3 CHECKS PASSED')
"
```

---

## Phase 4 — Sửa decision logic trong `step()` để dùng CPU/RAM

- [ ] **4.1** Trong [core/env.py](core/env.py), hàm `step()` (dòng 139-190), sau dòng tính `l_ratio` (dòng 144), thêm:
   ```python
   cpu_ratio = self.latest_raw["cpu_canary"] / max(self.latest_raw["cpu_stable"], EPSILON)
   mem_ratio = self.latest_raw["mem_canary_mb"] / max(self.latest_raw["mem_stable_mb"], EPSILON)
   ```

- [ ] **4.2** Sửa dòng 146-147:
   ```python
   current_healthy = (
       (self.latest_norm["e_ratio_n"] <= 0.4) and (self.latest_norm["l_ratio_n"] <= 0.4)
       and (self.latest_norm["cpu_ratio_n"] <= 0.4) and (self.latest_norm["mem_ratio_n"] <= 0.4)
   )
   current_anomalous = (
       (e_ratio > 2.0) or (l_ratio > 2.0) or (cpu_ratio > 2.0) or (mem_ratio > 2.0)
   )
   ```

- [ ] **4.3 (⚠️ BUG NGHIÊM TRỌNG NHẤT — ưu tiên cao nhất trong toàn bộ task list)** Sửa terminal check ở dòng 178-183:
   ```python
   if not self.done and self.weight >= 1.0:
       self.done = True
       if (
           (norm["e_ratio_n"] <= 0.4) and (norm["l_ratio_n"] <= 0.4)
           and (norm["cpu_ratio_n"] <= 0.4) and (norm["mem_ratio_n"] <= 0.4)
       ):
           reward += 10.0
       else:
           reward -= 10.0
   ```
   Trước khi sửa, hành vi hiện tại cho phép agent promote hết traffic (weight→1.0) và vẫn nhận +10 reward dù CPU/RAM đang bất thường — đây là lỗ hổng chính khiến agent học được cách bỏ qua CPU/RAM.

**Test Phase 4**:
```bash
python -c "
from core.env import CanaryEnv
env = CanaryEnv()
env.reset()
# ép cpu bất thường thủ công để kiểm tra current_anomalous nhìn thấy cpu
env.latest_raw['cpu_canary'] = env.latest_raw['cpu_stable'] * 5.0
obs, reward, done, trunc, info = env.step(0)  # Hold, không trigger action branch nhưng vẫn set current_anomalous
print('step ran OK with anomalous cpu, reward:', reward)
print('ALL PHASE 4 CHECKS PASSED (manual review of current_anomalous/current_healthy required)')
"
```
Ngoài script trên, coding agent PHẢI đọc lại đoạn code đã sửa bằng mắt và xác nhận cả 4 điều kiện (`e/l/cpu/mem`) xuất hiện ở cả 3 chỗ: `current_healthy`, `current_anomalous`, và terminal check L178-183.

---

## Phase 5 — Earliness bonus đối xứng cho Rollback đúng

- [ ] **5.1** Trong [core/env.py](core/env.py), nhánh `action == 2` (dòng 163-174), sửa nhánh rollback-đúng (dòng 169-174):
   ```python
   else:
       # Rollback đúng: CÓ early_bonus, đối xứng với Promote đúng.
       reward += 5.0 + bonus
       self.weight = 0.0
       self.done = True
   ```
   (Xóa comment cũ "Rollback phải dựa vào bằng chứng, không phải tốc độ" — đây là quyết định đã đổi, rollback đúng giờ cũng được thưởng tốc độ giống promote.)

- [ ] **5.2** Kiểm tra magnitude an toàn: penalty rollback-sai vẫn là `-22.0` (không đổi), rollback-đúng tối đa (step đầu, `EARLY_BONUS_SCALE` mặc định 2.0) = `5.0 + 1.96 = 6.96`. Nếu Optuna tune `EARLY_BONUS_SCALE` lên tới 2.5 (giới hạn trong `sweep_optuna.py:38`), max = `5.0 + 2.45 = 7.45`. Gap với `-22.0` vẫn đủ lớn (~29.5) để không có perverse incentive — không cần sửa gì thêm, chỉ cần confirm bằng phép tính này, không cần code test riêng.

**Test Phase 5**:
```bash
python -c "
from core.env import CanaryEnv
import os
os.environ['EARLY_BONUS_SCALE'] = '2.0'
env = CanaryEnv()
env.reset()
env.step_count = 0  # step đầu tiên => bonus lớn nhất
# Force healthy state để rollback sẽ bị coi là sai (phạt -22), rồi force anomalous để test rollback đúng
print('manual check: rollback correct reward should be > 5.0 (has bonus), rollback wrong stays -22.0')
"
```
(Đây là smoke test tối thiểu — verification đầy đủ nằm ở Phase 9 khi so sánh reward curve.)

---

## Phase 6 — Xác nhận n_features/STATE_KEYS (ghi chú, không sửa code)

- [ ] **6.1** Xác nhận `core/env.py` `num_features = 5` vẫn đúng sau Phase 1-5 (không đổi số channel, chỉ đổi nội dung 2 channel cpu/mem). `STATE_KEYS` trong `feature_pipeline.py` giờ có 8 keys: `weight_n, e_ratio_n, l_ratio_n, e_gap_n, l_gap_n, cpu_ratio_n, mem_ratio_n, rps_n` — chỉ 5 trong số này (`weight_n, e_ratio_n, l_ratio_n, cpu_ratio_n, mem_ratio_n`) thực sự vào observation qua `_raw_to_channels()`. Không cần sửa gì thêm ở Phase này.

**Test Phase 6**: Không có, đây là bước ghi chú/review.

---

## Phase 7 — Rà soát các nơi hardcode key cũ

- [ ] **7.1** Grep toàn repo tìm key cũ còn sót lại:
   ```bash
   grep -rn "cpu_n\|mem_n\b\|\"cpu\"\|\"mem_mb\"" --include="*.py" .
   ```
   Với mỗi kết quả (ngoại trừ trong `task_list.md`/`implementation_plan.md`), sửa sang key mới (`cpu_ratio_n`, `mem_ratio_n`, `cpu_canary`/`cpu_stable`, `mem_canary_mb`/`mem_stable_mb`).

**Test Phase 7**:
```bash
grep -rn "cpu_n\b\|mem_n\b" --include="*.py" . | grep -v task_list.md
# kỳ vọng: không có output nào (rỗng)
```

---

## Phase 8 — Sửa eval harness (`offline_training.py`, `sweep_optuna.py`)

- [ ] **8.1** Trong [core/env.py](core/env.py), thêm hỗ trợ `episode_config` cố định vào `reset()` để phục vụ acceptance test (Phase 9). ⚠️ **Dòng `elif randomize_scenario:` bên dưới chỉ là placeholder mô tả — KHÔNG được copy-paste nguyên comment đó vào code.** Phải thay bằng đúng code thật đã viết ở Phase 3.1a/3.1b (tính `is_healthy_episode`, `channel_anomalous`, `channel_pattern`, gọi `self._sample_baseline()` cho từng channel, rồi gán vào `self.episode_config`). Nếu để nguyên comment placeholder, `self.episode_config` sẽ không được set trong nhánh này và toàn bộ `_build_raw_metrics()` sẽ crash (`KeyError`/`AttributeError`) ngay lần `reset()` không truyền `episode_config`.
   ```python
   def reset(self, seed=None, options=None, randomize_scenario=True, episode_config=None):
       super().reset(seed=seed)
       self.weight = 0.05
       self.step_count = 0
       if episode_config is not None:
           self.episode_config = episode_config
           self.scenario = 0 if not any(episode_config["channel_anomalous"].values()) else 1
       elif randomize_scenario:
           # <-- THAY DÒNG NÀY: dán code thật của Phase 3.1a + 3.1b vào đây
           #     (is_healthy_episode, channel_anomalous, channel_pattern, self._sample_baseline(),
           #      rồi gán self.episode_config và self.scenario) — không được để nguyên comment.
           pass
       else:
           # randomize_scenario=False và episode_config=None: giữ nguyên episode_config của lần
           # reset trước đó (nếu có). Hiện tại không nơi nào trong codebase gọi với tổ hợp này,
           # nhưng vẫn cần tránh crash nếu self.episode_config chưa tồn tại (lần reset đầu tiên).
           if not hasattr(self, "episode_config"):
               raise ValueError("reset() cần episode_config hoặc randomize_scenario=True ở lần gọi đầu tiên")
       # phần còn lại giữ nguyên
   ```

- [ ] **8.2a (làm TRƯỚC 8.2, bắt buộc)** Trước khi viết lại `validate_model_locally()`, xác định đúng shape `model.predict()` cần khi bỏ `DummyVecEnv`. SB3 `PPO.predict()` có auto-detect obs không có batch dim và tự thêm vào, nhưng cần verify thực tế trên model đã train (train bằng `DummyVecEnv`, tức policy quen nhận input đã có batch dim) — không được đoán, phải chạy thử cả 2 cách trên model thật rồi chọn cách cho ra `action` hợp lệ (scalar 0/1/2, không phải array lồng `[[...]]`):
   ```python
   from stable_baselines3 import PPO
   from core.env import CanaryEnv

   env = CanaryEnv()
   obs, _ = env.reset()
   print("obs shape:", obs.shape)  # (30, 5)

   model = PPO.load("models/ppo_transformer_offline_best.zip")  # dùng model hiện có để test shape, không quan trọng model cũ/mới ở bước này

   action1, _ = model.predict(obs, deterministic=True)               # Cách 1: raw obs, không thêm batch dim
   action2, _ = model.predict(obs[None, ...], deterministic=True)    # Cách 2: manual batch dim

   print("action1:", action1, type(action1), getattr(action1, "shape", None))
   print("action2:", action2, type(action2), getattr(action2, "shape", None))
   ```
   Chọn cách nào cho `action` ra đúng 1 số nguyên 0/1/2 (hoặc mảng 1 phần tử `[a]` extract được bằng `int(action[0])`) mà KHÔNG bị lồng thêm chiều thừa (`[[a]]`). Dùng đúng cách đó xuyên suốt code ở 8.2 — sửa lại dòng `model.predict(obs[None, ...], ...)` trong khung code dưới nếu kết quả test cho thấy Cách 1 (raw obs, không batch) mới là đúng.

- [ ] **8.2 (thay thế cách tiếp cận VecNormalize cũ)** Trong [training/offline_training.py](training/offline_training.py), hàm `validate_model_locally()` (dòng 98-185): **bỏ hẳn `DummyVecEnv`/`VecNormalize` khi eval** — vì `build_env()` set `norm_obs=False`, và `eval_env.norm_reward = False` được set ngay ở dòng 104, nên `VecNormalize` **không làm gì cả** ở eval time (no-op hoàn toàn). Đã xác nhận: bỏ là đúng, chỉ cần dùng đúng shape đã chốt ở 8.2a. Thay bằng dùng thẳng `CanaryEnv()`:
   ```python
   def validate_model_locally(model_path, norm_path, num_episodes=100, episode_configs=None):
       print(f"\n🔍 Đang chạy Validate nội bộ với {num_episodes} tập...")
       env_instance = CanaryEnv()
       model = PPO.load(model_path, device=DEVICE)

       metrics_by_scenario = {i: {"episodes": 0, "latency": [], "fp": 0, "fn": 0} for i in range(2)}
       rewards = []

       for ep in range(num_episodes):
           cfg = episode_configs[ep] if episode_configs else None
           obs, _ = env_instance.reset(episode_config=cfg)
           done = False
           total_rew = 0
           step_count = 0
           scenario = env_instance.scenario
           last_action = 0

           while not done:
               action, _ = model.predict(obs[None, ...], deterministic=True)
               obs, reward, done, trunc, _ = env_instance.step(int(action[0]))
               done = done or trunc
               total_rew += reward
               step_count += 1
               last_action = int(action[0])

           rewards.append(total_rew)
           metrics_by_scenario[scenario]["episodes"] += 1
           metrics_by_scenario[scenario]["latency"].append(step_count)
           is_healthy = scenario == 0
           if is_healthy:
               if last_action == 2:
                   metrics_by_scenario[scenario]["fp"] += 1
           else:
               if last_action != 2:
                   metrics_by_scenario[scenario]["fn"] += 1
       # phần còn lại của hàm (in báo cáo) giữ logic tương tự, chỉ đổi range(5) -> range(2)
       # và scenario_names -> {0: "Healthy", 1: "Mixed/Anomalous"}
   ```
   Lưu ý: bỏ tham số `norm_path` khỏi lời gọi thực tế không cần thiết nữa (có thể giữ tham số cho tương thích chữ ký hàm nhưng không dùng), và tất cả nơi gọi `validate_model_locally(...)` trong `offline_training.py` (dòng 257) và `sweep_optuna.py` (dòng 74) vẫn truyền `norm_path` bình thường — không cần sửa lời gọi.

- [ ] **8.3** Trong [training/sweep_optuna.py](training/sweep_optuna.py), hàm `evaluate_rule_based_baseline()` (dòng 86-139): thêm CPU/RAM vào decision logic để so sánh công bằng với RL agent (agent nhìn được 4 channel thì rule-based cũng phải nhìn được 4 channel):
   ```python
   norm = env.latest_norm
   e_ratio = norm.get("e_ratio_n", 0) * 5.0
   l_ratio = norm.get("l_ratio_n", 0) * 5.0
   cpu_ratio = norm.get("cpu_ratio_n", 0) * 5.0
   mem_ratio = norm.get("mem_ratio_n", 0) * 5.0

   if e_ratio > threshold or l_ratio > threshold or cpu_ratio > threshold or mem_ratio > threshold:
       action = 2  # Rollback
   else:
       action = 1  # Promote
   ```
   Sửa luôn `is_healthy = scenario in [0, 4]` (dòng 126) → `is_healthy = scenario == 0` (khớp với nhãn coarse mới từ Phase 3.1c).

- [ ] **8.4** Trong `validate_model_locally()`, sửa dòng dùng `scenario_names` (dòng 151) và mọi chỗ group theo `range(5)`/`s in [0,4]` sang scheme 2 nhãn (`0=Healthy, 1=Mixed`) — rà soát toàn hàm, không chỉ đoạn đã trích ở 8.2.

**Test Phase 8**:
```bash
python -c "
from training.offline_training import validate_model_locally
import os
# chỉ chạy nếu đã có model đã train sẵn từ trước (models/ppo_transformer_offline_best.zip)
if os.path.exists('models/ppo_transformer_offline_best.zip'):
    validate_model_locally('models/ppo_transformer_offline_best.zip', 'models/vec_normalize.pkl', num_episodes=5)
    print('ALL PHASE 8 CHECKS PASSED')
else:
    print('SKIP: chưa có model cũ để test — sẽ verify đầy đủ ở Phase 9 sau khi train lại')
"
```

---

## Phase 9 — Train full model + Acceptance test (làm SAU CÙNG, sau khi Phase 1-8 đã pass hết)

- [ ] **9.1** Train lại model từ đầu với toàn bộ thay đổi Phase 1-8:
   ```bash
   python -m training.offline_training
   ```
   Đây PHẢI là model dùng cho acceptance test — không dùng model cũ (`models/ppo_transformer_offline_best.zip` hiện tại là train trước khi sửa bug, phải train lại đè lên).

- [ ] **9.2** Viết acceptance test cố định episode_config theo từng trường hợp, dùng `episode_config` param từ Phase 8.1:
   - Case A: tất cả channel healthy → agent phải Promote tới weight=1.0, không Rollback oan.
   - Case B: chỉ `cpu` anomalous (error/latency/mem healthy) → agent phải Rollback. **Đây là test case then chốt chứng minh bug đã fix** — trước đây agent sẽ Promote vì không nhìn thấy CPU.
   - Case C: chỉ `mem` anomalous → agent phải Rollback.
   - Case D: chỉ `error`/`latency` anomalous (case cũ đã hoạt động đúng từ trước) → vẫn phải Rollback, để đảm bảo không regression.
   - Case E: nhiều channel cùng anomalous → agent phải Rollback.

- [ ] **9.3** So sánh RL agent (đã fix) với rule-based baseline (đã fix ở Phase 8.3) trên FPR/FNR/latency/reward, in ra bảng so sánh.

**Test Phase 9**: Chạy đủ 9.1 → 9.2 → 9.3, báo cáo kết quả FPR/FNR cho từng Case A-E, đặc biệt highlight Case B/C (CPU/RAM-only anomaly) — đây là tiêu chí nghiệm thu chính của toàn bộ task list.

---

## Checklist tổng hợp 12 điểm review đã merge

| # | Điểm | Phase áp dụng |
|---|------|----------------|
| 1 | `_raw_to_channels()` ở env.py, không phải model.py | 0.2 |
| 2 | Đồng bộ key mới trong `_raw_to_channels()` (silent failure risk) | 1.7 |
| 3 | Assertion `cpu_ratio_n≈0.2` đúng, ghi rõ lý do | 1.8 |
| 4 | 2-stage sampling cho xác suất all-healthy 30-40% | 3.1a |
| 5 | Phase xóa if/elif + thay generator phải atomic | 3.2 |
| 6a | Tính `cpu_ratio`/`mem_ratio` trong `step()` | 4.1 |
| 6b | Terminal check L178-183 thiếu CPU/RAM (bug nghiêm trọng nhất) | 4.3 |
| 7 | Bỏ VecNormalize khi eval (no-op), dùng thẳng CanaryEnv | 8.2 |
| 8 | Rule-based baseline thêm CPU/RAM để so sánh công bằng | 8.3 |
| 9 | Train full model trước acceptance test | 9.1 |
| 10 | `generate_channel_value()` chỉ sinh canary; stable lấy từ baseline | 2.1 |
| 11 | Duy trì `rps` trong raw metrics | 1.3 |
| — | Earliness bonus đối xứng cho Rollback đúng (nguyên tắc bắt buộc) | 5.1 |
| — | `self.scenario` (nhãn coarse) tuyệt đối không lọt vào reward/observation/training logic | 3.1d |
| — | Baseline per-episode log-uniform (0.2x-5x quanh center cũ), không hardcode | 3.1b |
| — | Noise phải tương đối theo baseline (không absolute cố định) khi baseline đã randomize | 3.2 |
| — | Verify shape `model.predict()` thực tế trước khi bỏ VecNormalize, không đoán | 8.2a |
| — | Số dòng chỉ đúng cho Phase 1 — từ Phase 2 trở đi định vị code bằng search, không tin số dòng | Nguyên tắc #6 |
| — | Placeholder comment ở Phase 8.1 phải thay bằng code thật, không copy nguyên văn | 8.1 |
