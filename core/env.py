import gymnasium as gym
import numpy as np
import random
 
from core.feature_pipeline import EPSILON, normalize_raw_metrics, to_state_vector
 
# Định nghĩa tên để Log cho đẹp
SCENARIO_NAMES = {0: "Healthy", 1: "Resource Leak", 2: "Ticking Bomb", 3: "Critical Crash", 4: "Stable Equiv"}
ACTION_NAMES = {0: "+10%", 1: "+5%", 2: "Stay", 3: "-5%", 4: "ROLLBACK"}
 
# Giới hạn số bước mỗi episode — export để train.py đồng bộ
MAX_STEPS_PER_EPISODE = 50
 
 
class CanaryEnv(gym.Env):
    def __init__(self):
        super(CanaryEnv, self).__init__()
        # State (normalized): [weight_n, e_ratio_n, l_ratio_n, e_gap_n, l_gap_n, cpu_n, mem_n, rps_n]
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)
        self.latest_raw = {}
        self.latest_norm = {}
        self.reset()
 
    def reset(self, seed=None, options=None, randomize_scenario=True):
        super().reset(seed=seed)
        self.weight = 0.05
        self.step_count = 0
        # Only randomize scenario if requested (allows fixed scenarios for testing)
        if randomize_scenario:
            self.scenario = random.randint(0, 4)
        self.done = False
        return self._get_obs(), {}
 
    def _build_raw_metrics(self):
        noise = lambda s=1.0: np.random.normal(0, 0.01 * s)
        e_stable = max(0.0005, 0.001 + noise())
        l_stable = max(0.04, 0.095 + noise())
 
        if self.scenario == 0:  # Healthy
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
 
    def _get_obs(self):
        self.latest_raw = self._build_raw_metrics()
        self.latest_norm = normalize_raw_metrics(self.latest_raw)
        return np.array(to_state_vector(self.latest_raw), dtype=np.float32)
 
    def step(self, action):
        self.step_count += 1
        prev_weight = self.weight
 
        if action == 0: self.weight += 0.1
        elif action == 1: self.weight += 0.05
        elif action == 2: pass
        elif action == 3: self.weight -= 0.05
        elif action == 4:
            self.weight = 0.0
            self.done = True
 
        self.weight = np.clip(self.weight, 0.0, 1.0)
        obs = self._get_obs()
        raw = self.latest_raw
        norm = self.latest_norm
        reward = 0.0
 
        e_canary = raw["e_canary"]
        e_stable = raw["e_stable"]
        l_canary = raw["l_canary"]
        l_stable = raw["l_stable"]
 
        e_ratio = e_canary / max(e_stable, EPSILON)
        l_ratio = l_canary / max(l_stable, EPSILON)
        e_gap = max(0.0, e_canary - e_stable)
        l_gap_ratio = max(0.0, (l_canary - l_stable) / max(l_stable, EPSILON))
 
        risk_intensity = (3.0 * e_gap) + (1.5 * l_gap_ratio)
        traffic_factor = 0.3 + self.weight
 
        # Main risk penalty: discourage promotion when metrics degrade (but keep minimal)
        reward -= risk_intensity * (1.0 + 2.0 * traffic_factor)
 
        delta_weight = self.weight - prev_weight
        safe_for_growth = norm["e_ratio_n"] <= 0.4 and norm["l_ratio_n"] <= 0.4
 
        # Reward/Penalty for weight changes
        if delta_weight > 0:  # Promotion
            growth_reward = delta_weight * 40.0
            if safe_for_growth:
                # Safe promotion: reward it
                reward += growth_reward
                # Add bonus for safe steps (incentivize continuing promotion safely)
                reward += 2.0
            else:
                # Unsafe promotion: penalize heavily
                reward -= growth_reward * (2.0 + 2.0 * traffic_factor)
        elif delta_weight < 0:  # Rollback/Decrease
            # Only reward rollback if there's actual risk
            if e_ratio > 2.0 or l_ratio > 2.0:
                reward += abs(delta_weight) * 25.0
            else:
                # Penalize unnecessary rollback
                reward -= abs(delta_weight) * 15.0
 
        # Step survival bonus: reward for staying in safe state
        if safe_for_growth and self.weight > 0.0:
            reward += 1.0  # Small bonus per safe step
 
        # Handle ROLLBACK action
        if action == 4:
            self.done = True
            if self.scenario in (1, 2, 3):  # Leak, Bomb, Crash
                # ROLLBACK on degraded scenarios is GOOD - reward it strongly
                reward += 40.0
            else:  # Healthy, Stable_Equiv
                # ROLLBACK on healthy scenarios is BAD - penalize heavily
                reward -= 40.0
 
        # Terminal reward for reaching 100% weight
        if not self.done and self.weight >= 1.0:
            self.done = True
            if self.scenario in (0, 4):  # Healthy, Stable_Equiv - promote fully
                reward += 200.0
            else:  # Leak, Bomb, Crash - should have rolled back by now
                reward -= 200.0
 
        # Step limit penalty — dùng hằng số được export thay vì hardcode
        if self.step_count > MAX_STEPS_PER_EPISODE:
            self.done = True
            reward -= 10.0
 
        return obs, reward, self.done, False, {}