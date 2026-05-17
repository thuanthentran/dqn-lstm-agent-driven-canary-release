import gymnasium as gym
import numpy as np
import random
from collections import deque

from core.feature_pipeline import EPSILON, normalize_raw_metrics

# Scenario / action names for logging
SCENARIO_NAMES = {0: "Healthy", 1: "Resource Leak", 2: "Ticking Bomb", 3: "Critical Crash", 4: "Stable Equiv"}
ACTION_NAMES = {0: "Hold", 1: "Promote", 2: "Rollback"}

# Configurable sequence length for Conv1d input (channel-first)
SEQ_LEN = int(__import__("os").environ.get("SEQ_LEN", "30"))

# Episode length
MAX_STEPS_PER_EPISODE = 50


class CanaryEnv(gym.Env):
    """Gym-like Canary rollout environment that exposes a (C, T) matrix.

    Observation: shape (5, SEQ_LEN) with channels [CPU, RAM, Latency, Error_Rate, Traffic_Pct]
    Action: Discrete(3): 0=Hold, 1=Promote, 2=Rollback
    """

    def __init__(self, seq_len: int = SEQ_LEN):
        super(CanaryEnv, self).__init__()
        self.seq_len = int(seq_len)
        self.num_features = 5
        # Values normalized in [0,1]
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.num_features, self.seq_len), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)

        self.latest_raw = {}
        self.latest_norm = {}
        self.history = deque(maxlen=self.seq_len)
        self.reset()
    def _generate_random_steps(self):
        """
        Sinh ra một cấu hình traffic steps ngẫu nhiên cho mỗi episode.
        Ví dụ: [0.05, 0.20, 0.50, 0.80, 1.0] hoặc [0.10, 0.30, 0.40, 1.0]
        """
        # 1. Quyết định độ dài của chuỗi release (từ 3 đến 8 mốc)
        num_steps = random.randint(3, 8)
        
        # 2. Random các mốc % ở giữa (chọn các bội số của 5 từ 5 đến 95)
        # Sử dụng set() để đảm bảo không bị trùng lặp mốc
        steps_pct = set()
        while len(steps_pct) < num_steps - 1:
            steps_pct.add(random.choice(range(5, 100, 5)))
            
        # 3. Sắp xếp tăng dần và chốt hạ mốc 100% ở cuối cùng
        sorted_pct = sorted(list(steps_pct))
        sorted_pct.append(100)
        
        # 4. Chuyển đổi sang float (0.0 -> 1.0)
        return [float(x) / 100.0 for x in sorted_pct]
    
    def reset(self, seed=None, options=None, randomize_scenario=True):
        super().reset(seed=seed)
        self.weight = 0.05
        self.step_count = 0
        if randomize_scenario:
            self.scenario = random.randint(0, 4)

        self.traffic_steps = self._generate_random_steps()
        self.current_step_idx = 0
        self.weight = self.traffic_steps[self.current_step_idx]
        self.done = False
        # initialize history with repeated initial measurement
        raw = self._build_raw_metrics()
        norm = normalize_raw_metrics(raw)
        self.latest_raw = raw
        self.latest_norm = norm

        ch = self._raw_to_channels(raw, norm)
        for _ in range(self.seq_len):
            self.history.append(ch)

        return self._get_obs(), {}

    def _build_raw_metrics(self):
        noise = lambda s=1.0: np.random.normal(0, 0.01 * s)
        e_stable = max(0.0005, 0.001 + noise())
        l_stable = max(0.04, 0.095 + noise())

        if getattr(self, "scenario", 0) == 0:  # Healthy
            e_canary = max(0.0005, 0.001 + noise())
            l_canary = max(0.04, 0.09 + noise())
        elif self.scenario == 1:  # Resource Leak
            e_canary = max(0.0005, 0.003 + (self.weight * 0.03) + noise())
            l_canary = max(0.05, 0.11 + (self.weight * 0.6) + (self.step_count * 0.01) + noise())
        elif self.scenario == 2:  # Ticking Bomb
            if self.weight > 0.25:
                e_canary = max(0.001, 0.02 + (self.weight - 0.25) * 1.5 + noise())
            else:
                e_canary = max(0.0005, 0.001 + noise())
            l_canary = max(0.05, 0.12 + (self.weight * 0.2) + noise())
        elif self.scenario == 3:  # Critical Crash
            e_canary = max(0.2, 0.45 + noise(2.0))
            l_canary = max(0.12, 0.18 + noise())
        else:  # Stable Equivalent
            e_canary = max(0.0005, e_stable + noise())
            l_canary = max(0.04, l_stable + noise())

        cpu = max(0.0001, 0.001 + (self.weight * 0.01) + noise())
        mem = max(12.0, 24.0 + (self.weight * 20.0) + (8.0 if self.scenario == 1 else 0.0) + noise(20.0))
        rps = max(0.1, 40.0 * self.weight + np.random.normal(0, 2.0))

        return {
            "weight_pct": float(self.weight * 100.0),
            "e_canary": float(e_canary),
            "e_stable": float(e_stable),
            "l_canary": float(l_canary),
            "l_stable": float(l_stable),
            "cpu": float(cpu),
            "mem_mb": float(mem),
            "rps": float(rps),
        }

    def _raw_to_channels(self, raw: dict, norm: dict):
        # channels: [CPU, RAM, Latency, Error_Rate, Traffic_Pct]
        cpu_c = norm.get("cpu_n", 0.0)
        mem_c = norm.get("mem_n", 0.0)
        lat_c = norm.get("l_ratio_n", 0.0)
        err_c = norm.get("e_ratio_n", 0.0)
        traffic_c = norm.get("weight_n", 0.0)
        return np.array([cpu_c, mem_c, lat_c, err_c, traffic_c], dtype=np.float32)

    def _get_obs(self):
        # return (C, T) numpy array
        arr = np.stack(list(self.history), axis=1)  # (C, T)
        return arr.astype(np.float32)

    def step(self, action: int):
        self.step_count += 1
        reward = 0.0

        # =================================================================
        # 1. PHÁN XÉT DỰA TRÊN TRẠNG THÁI HIỆN TẠI (TRƯỚC KHI HÀNH ĐỘNG)
        # =================================================================
        e_ratio = self.latest_raw["e_canary"] / max(self.latest_raw["e_stable"], EPSILON)
        l_ratio = self.latest_raw["l_canary"] / max(self.latest_raw["l_stable"], EPSILON)

        current_healthy = (self.latest_norm["e_ratio_n"] <= 0.4) and (self.latest_norm["l_ratio_n"] <= 0.4)
        current_anomalous = (e_ratio > 2.0) or (l_ratio > 2.0)

        PROMOTE_STEP = 0.2

        # =================================================================
        # 2. XỬ LÝ LOGIC VÀ PHẦN THƯỞNG (REWARD SHAPING)
        # =================================================================
        if action == 0:  # HOLD
            reward -= 0.5  # Phạt nhẹ vì câu giờ

        elif action == 1:  # PROMOTE
            if current_anomalous:
                # Promote mù quáng khi hệ thống đang lỗi -> Phạt nặng, kết thúc
                reward -= 5.0
                self.done = True
            else:
                # Promote hợp lý -> Thưởng
                reward += 2.0
                self.weight = float(np.clip(self.weight + PROMOTE_STEP, 0.0, 1.0))

        elif action == 2:  # ROLLBACK
            if current_healthy:
                # Hệ thống đang yên lành lại đi Rollback -> Phạt nặng, kết thúc
                reward -= 10.0
                self.weight = 0.0
                self.done = True
            else:
                # Nhận diện lỗi giỏi, Rollback cứu hệ thống -> Thưởng, KẾT THÚC
                reward += 5.0  # Tăng thưởng cứu net để khuyến khích cắt lỗ
                self.weight = 0.0
                self.done = True  # Sửa lỗi farm điểm vô hạn!

        # =================================================================
        # 3. CHẠY MÔ PHỎNG ĐỂ LẤY TRẠNG THÁI MỚI (NEW STATE)
        # =================================================================
        raw = self._build_raw_metrics()
        norm = normalize_raw_metrics(raw)
        self.latest_raw = raw
        self.latest_norm = norm
        ch = self._raw_to_channels(raw, norm)
        self.history.append(ch)

        # =================================================================
        # 4. KIỂM TRA CÁC ĐIỀU KIỆN KẾT THÚC KHÁC
        # =================================================================
        if not self.done and self.weight >= 1.0:
            self.done = True
            # Kiểm tra xem chạm 100% traffic thì metrics có ổn không
            if (norm["e_ratio_n"] <= 0.4) and (norm["l_ratio_n"] <= 0.4):
                reward += 10.0  # Đích đến cuối cùng thành công rực rỡ
            else:
                reward -= 10.0  # Chạm đích nhưng lọt bug

        if not self.done and self.step_count > MAX_STEPS_PER_EPISODE:
            self.done = True
            reward -= 5.0  # Phạt vì quá lề mề không quyết đoán

        obs = self._get_obs()
        return obs, float(reward), bool(self.done), False, {}