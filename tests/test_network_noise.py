"""Unit tests for 6G network noise refactor.

Validates:
1. Observation shape (30, 15)
2. Network noise is symmetric (does NOT leak as app-fault signal)
3. App-layer scenarios still produce expected divergence
4. Training pipeline smoke test (shape/runtime)
5. Attention map shape with n_heads_feature=4
"""

import os
import sys
import math

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from core.env import CanaryEnv, NETWORK_SCENARIO_NAMES, SCENARIO_NAMES
from core.feature_pipeline import EPSILON, normalize_raw_metrics


# ── Test 1: Observation shape ──────────────────────────────────────────────

class TestObsShape:
    def test_reset_shape(self):
        env = CanaryEnv()
        obs, _ = env.reset()
        assert obs.shape == (30, 15), f"Expected (30, 15), got {obs.shape}"

    def test_step_shape(self):
        env = CanaryEnv()
        env.reset()
        obs, _, _, _, _ = env.step(0)
        assert obs.shape == (30, 15), f"Expected (30, 15), got {obs.shape}"

    def test_obs_range(self):
        """All normalized values should be in [0, 1]."""
        env = CanaryEnv()
        env.reset()
        for _ in range(10):
            obs, _, done, _, _ = env.step(0)
            assert obs.min() >= 0.0, f"Obs has negative value: {obs.min()}"
            assert obs.max() <= 1.0, f"Obs exceeds 1.0: {obs.max()}"
            if done:
                env.reset()


# ── Test 2: Network noise does NOT leak as app-fault signal ────────────────

class TestNetworkNoiseSymmetric:
    """With scenario=0 (Healthy), any network_scenario should NOT cause
    l_ratio or e_ratio to drift away from ~1.0 systematically."""

    @pytest.mark.parametrize("net_sc", range(5))
    def test_l_ratio_no_trend(self, net_sc):
        env = CanaryEnv()
        env.reset(randomize_scenario=False)
        env.scenario = 0          # Healthy app
        env.network_scenario = net_sc

        l_ratios = []
        for step in range(40):
            env.step_count = step
            raw = env._build_raw_metrics()
            l_ratio = raw["l_canary"] / max(raw["l_stable"], EPSILON)
            l_ratios.append(l_ratio)

        arr = np.array(l_ratios)
        mean_ratio = arr.mean()
        # For Healthy app, l_ratio should hover around ~0.95 (since
        # l_canary base=0.09, l_stable base=0.095). Allow wide margin for
        # noise, but the KEY check is no upward TREND.
        assert 0.5 < mean_ratio < 1.8, (
            f"net_sc={net_sc}: mean l_ratio={mean_ratio:.3f} out of expected range"
        )

        # Linear regression slope should be near 0 (no systematic drift)
        x = np.arange(len(l_ratios), dtype=float)
        slope = np.polyfit(x, l_ratios, 1)[0]
        assert abs(slope) < 0.05, (
            f"net_sc={net_sc}: l_ratio slope={slope:.4f} indicates systematic drift"
        )

    @pytest.mark.parametrize("net_sc", range(5))
    def test_e_ratio_no_trend(self, net_sc):
        """Network noise should NOT affect e_ratio (which has no burst factor).

        NOTE: e_ratio for Healthy is inherently heavy-tailed because base
        values (~0.001) are comparable to noise scale (0.01) and the
        max(0.0005,...) floor clamp amplifies variance. A single outlier
        ratio can exceed 50x. We use 200 samples and a generous slope
        tolerance to avoid false failures from noise.
        """
        env = CanaryEnv()
        env.reset(randomize_scenario=False)
        env.scenario = 0
        env.network_scenario = net_sc

        e_ratios = []
        for step in range(200):
            env.step_count = step
            raw = env._build_raw_metrics()
            e_ratio = raw["e_canary"] / max(raw["e_stable"], EPSILON)
            e_ratios.append(e_ratio)

        # Use median-based check: median should be near 1.0 for Healthy
        arr = np.array(e_ratios)
        median_ratio = np.median(arr)
        assert 0.3 < median_ratio < 5.0, (
            f"net_sc={net_sc}: median e_ratio={median_ratio:.3f} out of expected range"
        )

        # Slope check with generous tolerance for heavy-tailed noise
        x = np.arange(len(e_ratios), dtype=float)
        slope = np.polyfit(x, e_ratios, 1)[0]
        assert abs(slope) < 0.5, (
            f"net_sc={net_sc}: e_ratio slope={slope:.4f} indicates systematic drift"
        )


# ── Test 3: App scenarios still produce expected divergence ────────────────

class TestAppScenariosPreserved:
    """With network_scenario=0 (Stable), app scenarios should reproduce
    the same characteristic patterns as before the refactor."""

    def _run_episode(self, app_scenario, steps=30):
        env = CanaryEnv()
        env.reset(randomize_scenario=False)
        env.scenario = app_scenario
        env.network_scenario = 0  # Stable network

        l_ratios = []
        e_ratios = []
        for step in range(steps):
            env.step_count = step
            env.weight = min(0.05 + step * 0.03, 1.0)
            raw = env._build_raw_metrics()
            l_ratios.append(raw["l_canary"] / max(raw["l_stable"], EPSILON))
            e_ratios.append(raw["e_canary"] / max(raw["e_stable"], EPSILON))
        return np.array(l_ratios), np.array(e_ratios)

    def test_healthy_ratio_near_one(self):
        l_r, e_r = self._run_episode(0)
        assert 0.5 < l_r.mean() < 1.5, f"Healthy l_ratio mean={l_r.mean():.3f}"
        # e_ratio can be noisy (heavy-tailed) due to small base values;
        # check no systematic trend instead of tight bounds
        slope = np.polyfit(np.arange(len(e_r), dtype=float), e_r, 1)[0]
        assert abs(slope) < 0.5, f"Healthy e_ratio slope={slope:.4f}"

    def test_resource_leak_l_ratio_grows(self):
        """Scenario 1 (Resource Leak) should show l_ratio increasing over time."""
        l_r, _ = self._run_episode(1)
        slope = np.polyfit(np.arange(len(l_r), dtype=float), l_r, 1)[0]
        assert slope > 0.01, f"Resource Leak l_ratio slope={slope:.4f} — expected positive trend"

    def test_critical_crash_high_error(self):
        """Scenario 3 (Critical Crash) should have very high e_ratio."""
        _, e_r = self._run_episode(3)
        assert e_r.mean() > 50.0, f"Critical Crash e_ratio mean={e_r.mean():.3f} — expected >>1"

    def test_stable_equiv_near_one(self):
        """Scenario 4 (Stable Equiv) should be near 1.0."""
        l_r, e_r = self._run_episode(4)
        assert 0.5 < l_r.mean() < 1.5, f"Stable Equiv l_ratio mean={l_r.mean():.3f}"
        # e_ratio: same heavy-tail caveat as Healthy; check no trend
        slope = np.polyfit(np.arange(len(e_r), dtype=float), e_r, 1)[0]
        assert abs(slope) < 0.5, f"Stable Equiv e_ratio slope={slope:.4f}"


# ── Test 4: Training pipeline smoke test ───────────────────────────────────

class TestTrainingPipelineSmoke:
    def test_short_training_run(self):
        """Run a very short training to verify no shape/runtime errors."""
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from core.model import TransformerFeatureExtractor

        env = DummyVecEnv([lambda: Monitor(CanaryEnv(), None)])
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)

        _tmp = CanaryEnv()
        config = {
            "d_model": 64,
            "n_heads": 4,
            "n_heads_feature": 4,
            "n_layers": 2,
            "seq_len": 30,
            "n_features": _tmp.num_features,
            "dropout": 0.1,
        }
        del _tmp

        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            learning_rate=3e-4,
            n_steps=64,
            batch_size=32,
            policy_kwargs={
                "features_extractor_class": TransformerFeatureExtractor,
                "features_extractor_kwargs": config,
                "net_arch": dict(pi=[64, 64], vf=[64, 64]),
            },
            device="cpu",
        )

        # Train for a tiny number of steps — just verifying no crash
        model.learn(total_timesteps=256)

        # Verify attention map shapes
        extractor = model.policy.features_extractor
        attn_maps = extractor.get_attention_maps()
        fa = attn_maps["feature_attention"]
        assert fa is not None, "feature_attention is None after training"
        # fa shape: (B, n_heads_feature, T, n_features) = (B, 4, 30, 12)
        assert fa.shape[1] == 4, f"Expected n_heads_feature=4, got {fa.shape[1]}"
        assert fa.shape[2] == 30, f"Expected T=30, got {fa.shape[2]}"
        assert fa.shape[3] == 15, f"Expected n_features=15, got {fa.shape[3]}"

        ta = attn_maps["temporal_attention"]
        assert ta is not None, "temporal_attention is None after training"

        env.close()


# ── Test 5: network_noise_factor produces valid values ─────────────────────

class TestNetworkNoiseFactor:
    @pytest.mark.parametrize("net_sc", range(5))
    def test_burst_factor_positive(self, net_sc):
        env = CanaryEnv()
        env.reset(randomize_scenario=False)
        env.network_scenario = net_sc
        for step in range(20):
            env.step_count = step
            burst, raw = env._network_noise_factor()
            assert burst >= 1.0, f"burst_factor={burst} < 1.0 at step {step}, net_sc={net_sc}"
            assert "handover_count" in raw
            assert "sinr_db" in raw
            assert "prb_util" in raw
            assert "harq_nack" in raw
            assert "ntn_gap" in raw
            assert "isac_contention" in raw

    def test_ntn_gap_periodicity(self):
        """NTN gap should be active when step_count % 12 < 2."""
        env = CanaryEnv()
        env.reset(randomize_scenario=False)
        env.network_scenario = 2
        for step in range(24):
            env.step_count = step
            burst, raw = env._network_noise_factor()
            expected_gap = 1 if (step % 12) < 2 else 0
            assert raw["ntn_gap"] == expected_gap, (
                f"step={step}: ntn_gap={raw['ntn_gap']}, expected={expected_gap}"
            )

    def test_thz_blockage_periodicity(self):
        """THz blockage should be active when step_count % 15 < 3."""
        env = CanaryEnv()
        env.reset(randomize_scenario=False)
        env.network_scenario = 3
        for step in range(30):
            env.step_count = step
            burst, raw = env._network_noise_factor()
            if (step % 15) < 3:
                assert burst == 3.0, f"step={step}: burst={burst}, expected=3.0"
            else:
                assert burst == 1.0, f"step={step}: burst={burst}, expected=1.0"


# ── Test 6: Burst window local invariance ──────────────────────────────────

def _is_in_burst_window(network_scenario: int, step: int) -> bool:
    """Khớp đúng logic burst window đã cài trong _network_noise_factor()."""
    if network_scenario == 2:   # NTNGap
        return (step % 12) < 2
    if network_scenario == 3:   # THzBlockage
        return (step % 15) < 3
    return False  # HandoverStorm/ISACContention dao động liên tục, không có "window" rời rạc


def test_ntn_gap_burst_window_local_invariance():
    """Trong đúng cửa sổ NTN gap / THz blockage, l_ratio và e_ratio phải vẫn
    nằm trong khoảng hẹp — không leak thành tín hiệu giống lỗi app.

    l_ratio bound [0.5, 2.0]:
        burst_factor cancels in l_ratio (nhân cả tử lẫn mẫu). Với Healthy
        app, l_canary~0.09±0.01, l_stable~0.095±0.01. Worst case ±3σ:
        0.12/0.065 ≈ 1.85. Bound 2.0 covers >3σ comfortably.

    e_ratio bound [0.01, 50.0]:
        Network noise does NOT affect error rate at all (no burst on error).
        However, e_ratio for Healthy is inherently heavy-tailed: base~0.001
        with noise N(0,0.01) and floor clamp at 0.0005 means one side can
        be clamped while the other gets a positive noise spike, producing
        ratios of 10-50x. The wide bound [0.01, 50.0] is set to avoid false
        failures from this pre-existing distributional property while still
        catching catastrophic burst-leak bugs (which would push ratios to
        100x+ systematically). The KEY invariant for e_ratio is tested
        separately in test_e_ratio_no_trend (no slope/trend).
    """
    env = CanaryEnv()
    for network_scenario in range(5):
        env.scenario = 0  # Healthy — cố định app axis
        env.network_scenario = network_scenario
        env.reset(randomize_scenario=False)

        burst_ratios_l, burst_ratios_e = [], []
        for step in range(60):
            raw = env.latest_raw
            e_ratio = raw["e_canary"] / max(raw["e_stable"], EPSILON)
            l_ratio = raw["l_canary"] / max(raw["l_stable"], EPSILON)

            is_burst_step = _is_in_burst_window(network_scenario, step)
            if is_burst_step:
                burst_ratios_l.append(l_ratio)
                burst_ratios_e.append(e_ratio)

            obs, reward, done, truncated, info = env.step(0)  # Hold
            if done:
                break

        if burst_ratios_l:  # scenario Stable/HandoverStorm/ISAC sẽ không có burst step nào
            # l_ratio: burst cancels → ratio governed only by base values + Gaussian noise
            assert all(0.5 <= r <= 2.0 for r in burst_ratios_l), (
                f"network_scenario={network_scenario}: l_ratio vượt khoảng an toàn "
                f"trong burst window: {burst_ratios_l}"
            )
            # e_ratio: wide bounds due to inherent heavy-tail (see docstring)
            assert all(0.01 <= r <= 50.0 for r in burst_ratios_e), (
                f"network_scenario={network_scenario}: e_ratio vượt khoảng an toàn "
                f"trong burst window: {burst_ratios_e}"
            )


# ── Test 7: Reward không bị false-trigger bởi nhiễu mạng thuần túy ────────

def test_promote_not_penalized_by_pure_network_noise():
    """Với app Healthy, hành động Promote không được bị phạt (reward=-5.0,
    nhánh current_anomalous) chỉ vì nhiễu mạng thuần túy, ở bất kỳ
    network_scenario nào, tại bất kỳ step nào trong episode."""
    for network_scenario in range(5):
        for trial in range(10):  # nhiều seed để bắt được cả case xui nhất
            env = CanaryEnv()
            env.scenario = 0  # Healthy
            env.network_scenario = network_scenario
            env.reset(randomize_scenario=False)

            for step in range(40):
                obs, reward, done, truncated, info = env.step(1)  # Promote
                assert reward != -5.0, (
                    f"network_scenario={network_scenario}, trial={trial}, step={step}: "
                    f"Promote bị phạt -5.0 (current_anomalous=True) thuần do nhiễu mạng, "
                    f"không phải lỗi app (scenario=Healthy). "
                    f"latest_raw: e_canary={env.latest_raw['e_canary']:.6f}, "
                    f"e_stable={env.latest_raw['e_stable']:.6f}, "
                    f"l_canary={env.latest_raw['l_canary']:.6f}, "
                    f"l_stable={env.latest_raw['l_stable']:.6f}"
                )
                if done:
                    break


def test_rollback_not_rewarded_by_pure_network_noise():
    """Ngược lại: Rollback không nên được thưởng +5.0 (nhánh not current_healthy)
    chỉ vì nhiễu mạng làm current_healthy tạm thời False, khi app thực ra Healthy."""
    for network_scenario in range(5):
        for trial in range(10):
            env = CanaryEnv()
            env.scenario = 0  # Healthy
            env.network_scenario = network_scenario
            env.reset(randomize_scenario=False)

            # Chạy vài step Hold trước để vào giữa episode (dễ rơi vào burst window)
            for _ in range(3):
                obs, reward, done, truncated, info = env.step(0)
                if done:
                    break

            if not done:
                obs, reward, done, truncated, info = env.step(2)  # Rollback
                assert reward != 5.0, (
                    f"network_scenario={network_scenario}, trial={trial}: "
                    f"Rollback được thưởng +5.0 thuần do nhiễu mạng che current_healthy, "
                    f"trong khi app thực ra Healthy — đây là false positive nguy hiểm nhất "
                    f"vì nó rollback nhầm bản release TỐT. "
                    f"latest_raw: e_canary={env.latest_raw['e_canary']:.6f}, "
                    f"e_stable={env.latest_raw['e_stable']:.6f}, "
                    f"l_canary={env.latest_raw['l_canary']:.6f}, "
                    f"l_stable={env.latest_raw['l_stable']:.6f}"
                )

# ── Test 8: Smoothing doesn't hide real app faults ─────────────────────────

def test_critical_crash_still_detected_after_smoothing():
    """Sau khi thêm rolling-window smoothing, lỗi app nghiêm trọng
    (Critical Crash, scenario=3) vẫn phải bị phát hiện trong vài step,
    không bị làm trễ quá mức bởi smoothing window."""
    env = CanaryEnv()
    env.scenario = 3  # Critical Crash — e_canary luôn >= 0.45, rất cao
    env.network_scenario = 0  # Stable, cô lập biến
    env.reset(randomize_scenario=False)

    detected = False
    for step in range(6):  # buffer window=4, cho thêm buffer để ổn định
        obs, reward, done, truncated, info = env.step(1)  # thử Promote liên tục
        if reward == -5.0:  # bị chặn đúng vì current_anomalous
            detected = True
            break
        if done:
            break

    assert detected, (
        "Critical Crash (lỗi app nghiêm trọng, e_canary luôn >=0.45) không bị "
        "phát hiện trong 6 step đầu sau khi thêm rolling-window smoothing — "
        "smoothing đang che mất lỗi app thật, cần giảm maxlen của ratio_window"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
