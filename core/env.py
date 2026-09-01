import math
import random
from collections import deque
from os import environ

import gymnasium as gym
import numpy as np

from core.feature_pipeline import EPSILON, normalize_raw_metrics

# Scenario / action names for logging
SCENARIO_NAMES = {0: "Healthy", 1: "Resource Leak", 2: "Ticking Bomb", 3: "Critical Crash", 4: "Stable Equiv"}
NETWORK_SCENARIO_NAMES = {0: "Stable", 1: "HandoverStorm", 2: "NTNGap", 3: "THzBlockage", 4: "ISACContention"}
ACTION_NAMES = {0: "Hold", 1: "Promote", 2: "Rollback"}

# Configurable sequence length for Conv1d input (channel-first)
SEQ_LEN = int(environ.get("SEQ_LEN", "30"))

# Episode length
MAX_STEPS_PER_EPISODE = 50
EARLY_BONUS_SCALE = float(environ.get("EARLY_BONUS_SCALE", "0.1"))


class CanaryEnv(gym.Env):
    """Gym-like Canary rollout environment that exposes a (T, C) matrix.

    Observation: shape (SEQ_LEN, 15) with channels:
        [cpu_n, mem_n, l_ratio_n, e_ratio_n, weight_n,
         handover_n, sinr_n, rsrp_n, prb_n, harq_n, ntn_gap_n,
         isac_n, pkt_loss_n, jitter_n, deploy_age_n]
    Action: Discrete(3): 0=Hold, 1=Promote, 2=Rollback
    """

    def __init__(self, seq_len=30, num_features=15):
        self.seq_len = int(seq_len)
        self.num_features = num_features

        # --- Dynamic channel list (15 features, 3GPP TS 28.552 aligned) ---
        full_keys = [
            "cpu_n", "mem_n", "l_ratio_n", "e_ratio_n", "weight_n",
            "handover_n", "sinr_n", "rsrp_n", "prb_n", "harq_n", "ntn_gap_n",
            "isac_n", "pkt_loss_n", "jitter_n", "deploy_age_n",
        ]
        self.channel_keys = full_keys[:self.num_features]

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.seq_len, self.num_features),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(3)

        self.latest_raw = {}
        self.latest_norm = {}
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

    def reset(self, seed=None, options=None, randomize_scenario=True):
        super().reset(seed=seed)
        self.weight = 0.05
        self.step_count = 0
        
        if hasattr(self, 'force_scenario') and self.force_scenario is not None:
            self.scenario = self.force_scenario
        elif randomize_scenario:
            self.scenario = random.randint(0, 4)

        if hasattr(self, 'force_network_scenario') and self.force_network_scenario is not None:
            self.network_scenario = self.force_network_scenario
        elif randomize_scenario:
            self.network_scenario = random.randint(0, 4)

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

        # Rolling-window buffer for smoothed anomaly/healthy checks in step().
        # Pre-fill with initial ratios to avoid cold-start bias in first steps.
        # Fix for false-trigger from single-step noise spikes, discovered via
        # test_promote_not_penalized_by_pure_network_noise.
        init_e_ratio = raw["e_canary"] / max(raw["e_stable"], EPSILON)
        init_l_ratio = raw["l_canary"] / max(raw["l_stable"], EPSILON)
        self.ratio_window = deque(maxlen=4)
        for _ in range(4):
            self.ratio_window.append((init_e_ratio, init_l_ratio))

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # 6G Network noise — independent of app-layer scenario
    # ------------------------------------------------------------------
    def _network_noise_factor(self):
        """Compute network-layer noise based on self.network_scenario.

        Returns:
            burst_factor (float): multiplier for latency (applied symmetrically
                to both canary and stable).
            network_raw (dict): raw 6G telemetry signals.
        """
        sc = getattr(self, "network_scenario", 0)
        t = self.step_count

        # --- 3GPP-calibrated SINR baseline values ---
        # Ref: Koutlia et al., "Calibration of 5G-LENA", 3GPP TR 38.901
        # SINR range: [-10, +30] dB; RSRP range: [-140, -60] dBm
        if sc == 0:  # Stable
            burst = 1.0
            network_raw = {
                "handover_count": 0,
                "sinr_db": 20.0 + np.random.normal(0, 2.0),
                "rsrp_dbm": -75.0 + np.random.normal(0, 2.0),
                "prb_util": 0.4 + np.random.normal(0, 0.03),
                "harq_nack": 0.05 + abs(np.random.normal(0, 0.01)),
                "ntn_gap": 0,
                "isac_contention": 0.0,
                "packet_loss_rate": max(0.0, np.random.normal(0.001, 0.001)),
                "jitter_ms": max(0.0, 0.05 + np.random.normal(0, 0.02)),
            }

        elif sc == 1:  # HandoverStorm
            burst = 1.0 + 0.4 * math.sin(t) ** 2
            network_raw = {
                "handover_count": random.randint(2, 8),
                "sinr_db": 5.0 + np.random.normal(0, 4.0),
                "rsrp_dbm": -95.0 + np.random.normal(0, 5.0),
                "prb_util": 0.6 + np.random.normal(0, 0.05),
                "harq_nack": 0.08 + abs(np.random.normal(0, 0.02)),
                "ntn_gap": 0,
                "isac_contention": 0.0,
                "packet_loss_rate": max(0.0, np.random.normal(0.02, 0.01)),
                "jitter_ms": max(0.0, 0.5 + np.random.normal(0, 0.2)),
            }

        elif sc == 2:  # NTNGap
            in_gap = (t % 12) < 2
            if in_gap:
                burst = 2.5
                network_raw = {
                    "handover_count": 0,
                    "sinr_db": -5.0 + np.random.normal(0, 3.0),
                    "rsrp_dbm": -115.0 + np.random.normal(0, 3.0),
                    "prb_util": 0.3 + np.random.normal(0, 0.05),
                    "harq_nack": 0.15 + abs(np.random.normal(0, 0.03)),
                    "ntn_gap": 1,
                    "isac_contention": 0.0,
                    "packet_loss_rate": max(0.0, np.random.normal(0.1, 0.03)),
                    "jitter_ms": max(0.0, 5.0 + np.random.normal(0, 1.0)),
                }
            else:
                burst = 1.0
                network_raw = {
                    "handover_count": 0,
                    "sinr_db": 12.0 + np.random.normal(0, 2.0),
                    "rsrp_dbm": -90.0 + np.random.normal(0, 2.0),
                    "prb_util": 0.4 + np.random.normal(0, 0.03),
                    "harq_nack": 0.05 + abs(np.random.normal(0, 0.01)),
                    "ntn_gap": 0,
                    "isac_contention": 0.0,
                    "packet_loss_rate": max(0.0, np.random.normal(0.005, 0.002)),
                    "jitter_ms": max(0.0, 0.3 + np.random.normal(0, 0.1)),
                }

        elif sc == 3:  # THzBlockage
            blocked = (t % 15) < 3
            if blocked:
                burst = 3.0
                network_raw = {
                    "handover_count": 0,
                    "sinr_db": -2.0 + np.random.normal(0, 3.0),
                    "rsrp_dbm": -120.0 + np.random.normal(0, 3.0),
                    "prb_util": 0.2 + np.random.normal(0, 0.05),
                    "harq_nack": 0.20 + abs(np.random.normal(0, 0.03)),
                    "ntn_gap": 0,
                    "isac_contention": 0.0,
                    "packet_loss_rate": max(0.0, np.random.normal(0.15, 0.05)),
                    "jitter_ms": max(0.0, 3.0 + np.random.normal(0, 1.0)),
                }
            else:
                burst = 1.0
                network_raw = {
                    "handover_count": 0,
                    "sinr_db": 22.0 + np.random.normal(0, 2.0),
                    "rsrp_dbm": -72.0 + np.random.normal(0, 2.0),
                    "prb_util": 0.4 + np.random.normal(0, 0.03),
                    "harq_nack": 0.05 + abs(np.random.normal(0, 0.01)),
                    "ntn_gap": 0,
                    "isac_contention": 0.0,
                    "packet_loss_rate": max(0.0, np.random.normal(0.001, 0.001)),
                    "jitter_ms": max(0.0, 0.04 + np.random.normal(0, 0.01)),
                }

        elif sc == 4:  # ISACContention
            contention = 0.3 + 0.3 * math.sin(t * 0.5) ** 2
            burst = 1.0 + contention
            network_raw = {
                "handover_count": 0,
                "sinr_db": 15.0 + np.random.normal(0, 3.0),
                "rsrp_dbm": -80.0 + np.random.normal(0, 2.0),
                "prb_util": 0.5 + contention * 0.3 + np.random.normal(0, 0.03),
                "harq_nack": 0.05 + contention * 0.05 + abs(np.random.normal(0, 0.01)),
                "ntn_gap": 0,
                "isac_contention": contention,
                "packet_loss_rate": max(0.0, contention * 0.05 + np.random.normal(0, 0.01)),
                "jitter_ms": max(0.0, 0.2 + contention * 0.5 + np.random.normal(0, 0.1)),
            }

        else:  # fallback — Stable
            burst = 1.0
            network_raw = {
                "handover_count": 0,
                "sinr_db": 20.0,
                "rsrp_dbm": -75.0,
                "prb_util": 0.4,
                "harq_nack": 0.05,
                "ntn_gap": 0,
                "isac_contention": 0.0,
                "packet_loss_rate": 0.001,
                "jitter_ms": 0.05,
            }

        return burst, network_raw

    def _build_raw_metrics(self):
        # Additive noise for latency/cpu/mem (base values >> noise std, no clamp issue).
        noise = lambda s=1.0: np.random.normal(0, 0.01 * s)

        # Multiplicative noise for error rates — fix for false-trigger bug:
        # Old additive noise N(0, 0.01) on base~0.001 was 10x the signal,
        # combined with max(0.0005,...) floor clamp, created asymmetric ratio
        # spikes >2.0 in ~15-20% of steps even for Healthy app.
        # Multiplicative noise (8% relative std) keeps noise proportional
        # to signal magnitude. At 8%, worst-case single-step e_ratio is
        # ~1.94 (4σ: 1.32/0.68), and with rolling window=4 smoothing the
        # mean stays well under the 2.0 anomaly threshold.
        # 15% was too high: ratio reached 2.0 at ~4σ, guaranteed to hit
        # over 2000+ trial-steps in tests.
        # Discovered via test_promote_not_penalized_by_pure_network_noise.
        rel_noise = lambda scale=1.0: np.random.normal(0, 0.08 * scale)

        e_stable = max(EPSILON, 0.001 * (1.0 + rel_noise()))
        l_stable = max(0.04, 0.095 + noise())

        if getattr(self, "scenario", 0) == 0:
            e_canary = max(EPSILON, 0.001 * (1.0 + rel_noise()))
            l_canary = max(0.04, 0.09 + noise())
        elif self.scenario == 1:
            base_e = 0.003 + (self.weight * 0.03)
            e_canary = max(EPSILON, base_e * (1.0 + rel_noise()))
            l_canary = max(0.05, 0.11 + (self.weight * 0.6) + (self.step_count * 0.01) + noise())
        elif self.scenario == 2:
            if self.weight > 0.25:
                base_e = 0.02 + (self.weight - 0.25) * 1.5
                e_canary = max(EPSILON, base_e * (1.0 + rel_noise()))
            else:
                e_canary = max(EPSILON, 0.001 * (1.0 + rel_noise()))
            l_canary = max(0.05, 0.12 + (self.weight * 0.2) + noise())
        elif self.scenario == 3:
            e_canary = max(EPSILON, 0.45 * (1.0 + rel_noise(2.0)))
            l_canary = max(0.12, 0.18 + noise())
        else:
            e_canary = max(EPSILON, e_stable * (1.0 + rel_noise()))
            l_canary = max(0.04, l_stable + noise())

        # --- Apply network burst symmetrically to BOTH canary and stable ---
        # Both share the same physical RAN infrastructure, so network noise
        # must affect both equally. Only app-layer faults should cause
        # systematic divergence in l_ratio.
        burst_factor, network_raw = self._network_noise_factor()
        l_canary *= burst_factor
        l_stable *= burst_factor

        cpu_canary = max(0.0001, 0.001 + (self.weight * 0.05) + (0.05 if self.scenario == 2 else 0.0) + noise())
        cpu_stable = max(0.0001, 0.001 + ((1.0 - self.weight) * 0.05) + noise())
        
        mem_canary = max(12.0, 24.0 + (self.weight * 20.0) + (16.0 if self.scenario == 1 else 0.0) + noise(2.0))
        mem_stable = max(12.0, 24.0 + ((1.0 - self.weight) * 20.0) + noise(2.0))
        
        rps = max(0.1, 40.0 * self.weight + np.random.normal(0, 2.0))

        raw = {
            "weight_pct": float(self.weight * 100.0),
            "e_canary": float(e_canary),
            "e_stable": float(e_stable),
            "l_canary": float(l_canary),
            "l_stable": float(l_stable),
            "cpu_canary": float(cpu_canary),
            "cpu_stable": float(cpu_stable),
            "mem_canary_mb": float(mem_canary),
            "mem_stable_mb": float(mem_stable),
            "rps": float(rps),
            "time_since_deploy": self.step_count,
        }
        # Merge 6G network telemetry
        raw.update(network_raw)

        return raw

    def _raw_to_channels(self, raw: dict, norm: dict):
        return np.array([norm[k] for k in self.channel_keys], dtype=np.float32)

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

        e_ratio = self.latest_raw["e_canary"] / max(self.latest_raw["e_stable"], EPSILON)
        l_ratio = self.latest_raw["l_canary"] / max(self.latest_raw["l_stable"], EPSILON)

        # Rolling-window smoothed anomaly/healthy check — fix for false-trigger
        # from single-step noise spikes. Uses mean of last 4 raw ratios instead
        # of instantaneous value, matching the multi-step observation window
        # the model uses (seq_len=30). Window=4 is short enough to still detect
        # real app faults within a few steps (verified by
        # test_critical_crash_still_detected_after_smoothing).
        self.ratio_window.append((e_ratio, l_ratio))
        e_ratio_smoothed = float(np.mean([r[0] for r in self.ratio_window]))
        l_ratio_smoothed = float(np.mean([r[1] for r in self.ratio_window]))

        current_healthy = (e_ratio_smoothed <= 2.0) and (l_ratio_smoothed <= 2.0)
        current_anomalous = (e_ratio_smoothed > 2.0) or (l_ratio_smoothed > 2.0)
        promote_step = 0.2

        if action == 0:
            reward -= 0.5
        elif action == 1:
            if current_anomalous:
                reward -= 5.0
                self.done = True
            else:
                reward += 2.0
                self.weight = float(np.clip(self.weight + promote_step, 0.0, 1.0))
        elif action == 2:
            if current_healthy:
                reward -= 10.0
                self.weight = 0.0
                self.done = True
            else:
                reward += 5.0
                bonus = EARLY_BONUS_SCALE * max(0, MAX_STEPS_PER_EPISODE - self.step_count)
                reward += bonus
                self.weight = 0.0
                self.done = True

        raw, norm = self._update_state()

        if not self.done and self.weight >= 1.0:
            self.done = True
            if (norm["e_ratio_n"] <= 0.4) and (norm["l_ratio_n"] <= 0.4):
                reward += 10.0
                bonus = EARLY_BONUS_SCALE * max(0, MAX_STEPS_PER_EPISODE - self.step_count)
                reward += bonus
            else:
                reward -= 10.0

        if not self.done and self.step_count > MAX_STEPS_PER_EPISODE:
            self.done = True
            reward -= 5.0

        obs = self._get_obs()
        return obs, float(reward), bool(self.done), False, {}