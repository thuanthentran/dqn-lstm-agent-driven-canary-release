import random
from collections import deque
from os import environ
from typing import Callable, Dict, Optional

import gymnasium as gym
import numpy as np

from core.feature_pipeline import EPSILON, normalize_raw_metrics

# Scenario / action names for logging
# self.scenario giờ chỉ là nhãn coarse 2 giá trị (0=Healthy, 1=Mixed/Anomalous),
# chỉ dùng để log/group kết quả — KHÔNG dùng trong _build_raw_metrics() hay reward logic.
SCENARIO_NAMES = {0: "Healthy", 1: "Mixed/Anomalous"}
ACTION_NAMES = {0: "Hold", 1: "Promote", 2: "Rollback"}

# Configurable sequence length for Conv1d input (channel-first)
SEQ_LEN = int(environ.get("SEQ_LEN", "30"))

# Episode length
MAX_STEPS_PER_EPISODE = 50

# RPS reference (dùng cho load_dependent pattern trong generate_channel_value)
RPS_REF = 40.0

# Baseline centers cho mỗi channel — giữ đúng scale cũ.
# _sample_baseline() sẽ sample log-uniform 0.2x–5x quanh center này mỗi episode.
BASELINE_CENTERS = {
    "error":   0.001,   # giữ đúng scale cũ (e_stable gốc)
    "latency": 0.095,   # giữ đúng scale cũ (l_stable gốc)
    "cpu":     0.001,   # giữ đúng scale cũ (cpu_stable gốc)
    "mem":     24.0,    # giữ đúng scale cũ (mem_stable gốc)
}


def relative_noise(baseline: float, rel_std: float = 0.01) -> float:
    """Noise tương đối theo baseline.

    Thay cho noise absolute cố định (lambda s: np.random.normal(0, 0.01*s)).
    Khi baseline dao động log-uniform 0.2x–5x, noise absolute cố định sẽ
    lấn át hoặc vô nghĩa; noise tương đối giữ SNR ổn định.
    """
    return float(np.random.normal(0, rel_std * baseline))


class CanaryEnv(gym.Env):
    """Gym-like Canary rollout environment that exposes a (T, C) matrix.

    Observation: shape (SEQ_LEN, 5) with channels [CPU, RAM, Latency, Error_Rate, Traffic_Pct]
    Action: Discrete(3): 0=Hold, 1=Promote, 2=Rollback
    """

    def __init__(self, seq_len: int = SEQ_LEN):
        super().__init__()
        self.seq_len = int(seq_len)
        self.num_features = 5
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.seq_len, self.num_features),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(3)

        self.latest_raw: Dict[str, float] = {}
        self.latest_norm: Dict[str, float] = {}
        self.history = deque(maxlen=self.seq_len)
        self.reset()

    def _generate_random_steps(self):
        """Generate a monotonic traffic schedule for one episode."""
        num_steps = random.randint(3, 8)
        steps_pct = set()
        while len(steps_pct) < num_steps - 1:
            steps_pct.add(random.choice(range(5, 100, 5)))
        sorted_pct = sorted(steps_pct)
        sorted_pct.append(100)
        return [float(x) / 100.0 for x in sorted_pct]

    def _sample_baseline(self, channel: str) -> float:
        """Sample baseline per-episode theo log-uniform 0.2x–5x quanh BASELINE_CENTERS[channel].

        Log-uniform (thay vì uniform) khiến đa số episode vẫn tập trung gần center,
        nhưng vẫn có đủ số episode "siêu nhẹ" (0.2x) lẫn "siêu nặng" (5x) để agent
        học được ratio-based detection thay vì học thuộc absolute value.
        """
        center = BASELINE_CENTERS[channel]
        scale = float(np.exp(np.random.uniform(np.log(0.2), np.log(5.0))))
        return center * scale

    def reset(
        self,
        seed=None,
        options=None,
        randomize_scenario: bool = True,
        episode_config: Optional[Dict] = None,
    ):
        super().reset(seed=seed)
        self.weight = 0.05
        self.step_count = 0

        if episode_config is not None:
            # Episode config cố định — dùng cho acceptance test (Phase 9)
            self.episode_config = episode_config
            self.scenario = 0 if not any(episode_config["channel_anomalous"].values()) else 1
        elif randomize_scenario:
            # 2-stage sampling:
            # Tầng 1: coin-flip quyết định "episode này healthy hay mixed" (30-40% healthy).
            # Nếu sample độc lập từng channel (mỗi channel ~60% static), xác suất all-healthy
            # chỉ đạt (0.6)^4 ≈ 13% — quá thấp. Dùng 2-tầng để đảm bảo 30-40%.
            # Tầng 2: chỉ áp dụng khi mixed, mỗi channel bất thường độc lập.
            HEALTHY_EPISODE_PROB = 0.35  # 30-40% all-healthy
            PER_CHANNEL_STATIC_PROB = 0.45  # mỗi channel ~45% static khi mixed

            is_healthy_episode = random.random() < HEALTHY_EPISODE_PROB
            channels = ["cpu", "mem", "latency", "error"]
            if is_healthy_episode:
                channel_anomalous = {c: False for c in channels}
            else:
                channel_anomalous = {
                    c: (random.random() >= PER_CHANNEL_STATIC_PROB) for c in channels
                }

            # Random chọn pattern cho các channel anomalous
            anomaly_patterns = ["leak", "threshold_spike", "load_dependent"]
            channel_pattern = {
                c: (random.choice(anomaly_patterns) if channel_anomalous[c] else "static")
                for c in channels
            }

            # Baseline per-episode: sample log-uniform độc lập cho từng channel
            baselines = {ch: self._sample_baseline(ch) for ch in channels}

            self.episode_config = {
                "channel_anomalous": channel_anomalous,
                "channel_pattern": channel_pattern,
                "baseline": baselines,
            }
            # Nhãn coarse: chỉ dùng để log/group kết quả (Phase 8), KHÔNG dùng trong reward/obs
            self.scenario = 0 if is_healthy_episode else 1
        else:
            # randomize_scenario=False và episode_config=None: giữ nguyên episode_config trước
            if not hasattr(self, "episode_config"):
                raise ValueError(
                    "reset() cần episode_config hoặc randomize_scenario=True ở lần gọi đầu tiên"
                )

        self.traffic_steps = self._generate_random_steps()
        self.current_step_idx = 0
        self.weight = self.traffic_steps[self.current_step_idx]
        self.done = False

        self.history.clear()
        raw = self._build_raw_metrics()
        norm = normalize_raw_metrics(raw)
        self.latest_raw = raw
        self.latest_norm = norm

        initial_channels = self._raw_to_channels(raw, norm)
        for _ in range(self.seq_len):
            self.history.append(initial_channels)

        return self._get_obs(), {}

    def generate_channel_value(
        self,
        channel: str,
        pattern: str,
        weight: float,
        step_count: int,
        baseline: float,
        noise_fn: Callable[[], float],
        rps: float = 0.0,
    ) -> float:
        """Sinh 1 giá trị canary cho channel theo pattern.

        Chỉ sinh giá trị canary side. Giá trị stable/baseline được tính riêng
        ở _build_raw_metrics() (1 dòng max(floor, baseline + noise_fn())).

        Mọi offset anomaly PHẢI tỉ lệ với baseline (nhân hệ số),
        KHÔNG cộng hằng số tuyệt đối — vì baseline dao động log-uniform 0.2x–5x.
        """
        if pattern == "static":
            # Không có anomaly — giống stable
            return baseline + noise_fn()

        elif pattern == "leak":
            # Resource Leak: tăng dần theo weight và step_count, tỉ lệ với baseline.
            # Hệ số khác nhau theo channel để phân biệt tín hiệu:
            #   - error/cpu: tăng chậm (hệ số nhỏ)
            #   - latency/mem: tăng nhanh hơn
            if channel in ("error", "cpu"):
                return baseline * (1.0 + weight * 6.0 + step_count * 0.1) + noise_fn()
            else:  # latency, mem
                return baseline * (1.0 + weight * 8.0 + step_count * 0.15) + noise_fn()

        elif pattern == "threshold_spike":
            # Ticking Bomb: chỉ tăng đột biến khi weight > 0.25.
            # Công thức tỉ lệ hóa từ scenario 2 cũ: baseline * (1 + f(weight)).
            spike = max(0.0, weight - 0.25) * 15.0
            return baseline * (1.0 + spike) + noise_fn()

        elif pattern == "load_dependent":
            # Anomaly chỉ xuất hiện khi traffic cao (rps cao).
            # Dùng cho CPU/mem khi muốn phân biệt với leak (không tăng theo thời gian, chỉ theo load).
            load_factor = (rps / RPS_REF) * 3.0  # hệ số tỉ lệ rps
            return baseline * (1.0 + load_factor * weight) + noise_fn()

        else:
            # Fallback: static
            return baseline + noise_fn()

    def _build_raw_metrics(self) -> Dict[str, float]:
        """Build raw metrics dict cho 1 timestep.

        Dùng episode_config (set bởi reset()) để sinh canary vs stable per-channel.
        Không còn if/elif self.scenario — scenario chỉ là nhãn coarse ở reset().
        """
        cfg = self.episode_config
        rps = max(0.1, 40.0 * self.weight + np.random.normal(0, 2.0))

        values: Dict[str, float] = {}
        for ch, key_canary, key_stable, baseline_key in [
            ("error",   "e_canary",      "e_stable",      "error"),
            ("latency", "l_canary",      "l_stable",      "latency"),
            ("cpu",     "cpu_canary",    "cpu_stable",    "cpu"),
            ("mem",     "mem_canary_mb", "mem_stable_mb", "mem"),
        ]:
            baseline = cfg["baseline"][baseline_key]
            # Stable: noise tương đối nhỏ quanh baseline (không âm, floor = 50% baseline)
            stable_val = max(baseline * 0.5, baseline + relative_noise(baseline))
            values[key_stable] = float(stable_val)

            if cfg["channel_anomalous"][ch]:
                pattern = cfg["channel_pattern"][ch]
                canary_val = self.generate_channel_value(
                    ch,
                    pattern,
                    self.weight,
                    self.step_count,
                    baseline,
                    lambda b=baseline: relative_noise(b),
                    rps=rps,
                )
            else:
                # Healthy channel: canary ≈ stable (không có anomaly)
                canary_val = baseline + relative_noise(baseline)

            values[key_canary] = float(max(0.0, canary_val))

        return {
            "weight_pct": float(self.weight * 100.0),
            "e_canary":      values["e_canary"],
            "e_stable":      values["e_stable"],
            "l_canary":      values["l_canary"],
            "l_stable":      values["l_stable"],
            "cpu_canary":    values["cpu_canary"],
            "cpu_stable":    values["cpu_stable"],
            "mem_canary_mb": values["mem_canary_mb"],
            "mem_stable_mb": values["mem_stable_mb"],
            "rps": float(rps),
        }

    def _raw_to_channels(self, raw: dict, norm: dict):
        """Map normalized dict → 5-channel array [CPU, RAM, Latency, Error, Traffic].

        Dùng cpu_ratio_n/mem_ratio_n (thay cho cpu_n/mem_n cũ).
        .get() không raise lỗi khi key sai — nếu bỏ sót bước đổi key,
        observation sẽ nhận toàn 0 cho CPU/RAM một cách âm thầm (silent failure).
        Thứ tự [cpu_c, mem_c, lat_c, err_c, traffic_c] giữ nguyên theo Phase 0.1.
        """
        cpu_c     = norm.get("cpu_ratio_n", 0.0)
        mem_c     = norm.get("mem_ratio_n", 0.0)
        lat_c     = norm.get("l_ratio_n", 0.0)
        err_c     = norm.get("e_ratio_n", 0.0)
        traffic_c = norm.get("weight_n", 0.0)
        return np.array([cpu_c, mem_c, lat_c, err_c, traffic_c], dtype=np.float32)

    def _get_obs(self):
        # Stack along axis=0: each row is a timestep with num_features features → (T, C)
        arr = np.stack(list(self.history), axis=0)
        return arr.astype(np.float32)

    def _update_state(self):
        raw = self._build_raw_metrics()
        norm = normalize_raw_metrics(raw)
        self.latest_raw = raw
        self.latest_norm = norm
        self.history.append(self._raw_to_channels(raw, norm))
        return raw, norm

    def step(self, action: int):
        self.step_count += 1
        reward = 0.0

        # Tính ratio từ raw metrics (dùng EPSILON theo Nguyên tắc #1)
        e_ratio   = self.latest_raw["e_canary"]    / max(self.latest_raw["e_stable"],      EPSILON)
        l_ratio   = self.latest_raw["l_canary"]    / max(self.latest_raw["l_stable"],      EPSILON)
        cpu_ratio = self.latest_raw["cpu_canary"]  / max(self.latest_raw["cpu_stable"],    EPSILON)
        mem_ratio = self.latest_raw["mem_canary_mb"] / max(self.latest_raw["mem_stable_mb"], EPSILON)

        # current_healthy: tất cả 4 channel ratio_n đều thấp
        current_healthy = (
            (self.latest_norm["e_ratio_n"]   <= 0.4)
            and (self.latest_norm["l_ratio_n"]   <= 0.4)
            and (self.latest_norm["cpu_ratio_n"] <= 0.4)
            and (self.latest_norm["mem_ratio_n"] <= 0.4)
        )
        # current_anomalous: bất kỳ channel nào bị spike > 2x stable
        current_anomalous = (
            (e_ratio > 2.0) or (l_ratio > 2.0)
            or (cpu_ratio > 2.0) or (mem_ratio > 2.0)
        )

        promote_step = 0.2

        current_bonus_scale = float(environ.get("EARLY_BONUS_SCALE", "2.0"))
        bonus = current_bonus_scale * (MAX_STEPS_PER_EPISODE - self.step_count) / MAX_STEPS_PER_EPISODE

        if action == 0:
            reward -= 0.5
        elif action == 1:
            if current_anomalous:
                reward -= 5.0
                self.done = True
            else:
                # Promote đúng: có early_bonus để khuyến khích promote nhanh khi môi trường ổn
                reward += 3.0 + (bonus * 0.5)
                self.weight = float(np.clip(self.weight + promote_step, 0.0, 1.0))
        elif action == 2:
            if current_healthy:
                # Rollback oan (False Positive): phạt nặng, không mercy
                reward -= 22.0
                self.weight = 0.0
                self.done = True
            else:
                # Rollback đúng: CÓ early_bonus, đối xứng với Promote đúng.
                # (Trước đây không có bonus — đã đổi theo Phase 5)
                reward += 5.0 + bonus
                self.weight = 0.0
                self.done = True

        raw, norm = self._update_state()

        # Terminal check: nếu weight đạt 1.0, kiểm tra tất cả 4 channel
        # (Đây là bug nghiêm trọng nhất trước khi fix: chỉ kiểm tra e/l, bỏ qua cpu/mem
        #  → agent học được cách bỏ qua CPU/RAM để vẫn nhận +10 khi promote xong)
        if not self.done and self.weight >= 1.0:
            self.done = True
            if (
                (norm["e_ratio_n"]   <= 0.4)
                and (norm["l_ratio_n"]   <= 0.4)
                and (norm["cpu_ratio_n"] <= 0.4)
                and (norm["mem_ratio_n"] <= 0.4)
            ):
                reward += 10.0
            else:
                reward -= 10.0

        if not self.done and self.step_count > MAX_STEPS_PER_EPISODE:
            self.done = True
            reward -= 5.0

        obs = self._get_obs()
        return obs, float(reward), bool(self.done), False, {}