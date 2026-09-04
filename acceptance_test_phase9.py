"""
Phase 9.2 + 9.3: Acceptance test và so sánh RL vs Rule-based baseline.

Chạy sau khi training Phase 9.1 hoàn thành.
"""
import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"

from stable_baselines3 import PPO
from core.env import CanaryEnv
from training.sweep_optuna import evaluate_rule_based_baseline

MODEL_PATH = "models/ppo_transformer_offline_best.zip"


def make_all_healthy_config():
    """Case A: tất cả channel healthy → agent phải Promote."""
    return {
        "channel_anomalous": {"cpu": False, "mem": False, "latency": False, "error": False},
        "channel_pattern":   {"cpu": "static", "mem": "static", "latency": "static", "error": "static"},
        "baseline": {"cpu": 0.001, "mem": 24.0, "latency": 0.095, "error": 0.001},
    }


def make_cpu_only_config():
    """Case B: chỉ CPU anomalous (error/latency/mem healthy) → agent phải Rollback."""
    return {
        "channel_anomalous": {"cpu": True, "mem": False, "latency": False, "error": False},
        "channel_pattern":   {"cpu": "leak", "mem": "static", "latency": "static", "error": "static"},
        "baseline": {"cpu": 0.001, "mem": 24.0, "latency": 0.095, "error": 0.001},
    }


def make_mem_only_config():
    """Case C: chỉ MEM anomalous → agent phải Rollback."""
    return {
        "channel_anomalous": {"cpu": False, "mem": True, "latency": False, "error": False},
        "channel_pattern":   {"cpu": "static", "mem": "threshold_spike", "latency": "static", "error": "static"},
        "baseline": {"cpu": 0.001, "mem": 24.0, "latency": 0.095, "error": 0.001},
    }


def make_error_latency_config():
    """Case D: chỉ error/latency anomalous (case cũ đã hoạt động đúng) → phải Rollback."""
    return {
        "channel_anomalous": {"cpu": False, "mem": False, "latency": True, "error": True},
        "channel_pattern":   {"cpu": "static", "mem": "static", "latency": "leak", "error": "leak"},
        "baseline": {"cpu": 0.001, "mem": 24.0, "latency": 0.095, "error": 0.001},
    }


def make_multi_anomalous_config():
    """Case E: nhiều channel cùng anomalous → agent phải Rollback."""
    return {
        "channel_anomalous": {"cpu": True, "mem": True, "latency": True, "error": True},
        "channel_pattern":   {"cpu": "leak", "mem": "leak", "latency": "leak", "error": "leak"},
        "baseline": {"cpu": 0.001, "mem": 24.0, "latency": 0.095, "error": 0.001},
    }


def run_case(model, env, case_name, config, n_trials=10):
    """Chạy 1 case nhiều lần để có kết quả ổn định hơn."""
    rollbacks = 0
    promotes = 0

    for _ in range(n_trials):
        obs, _ = env.reset(episode_config=config)
        done = False
        last_action = 0

        while not done:
            action, _ = model.predict(obs[None, ...], deterministic=True)
            obs, _, done, trunc, _ = env.step(int(action[0]))
            done = done or trunc
            last_action = int(action[0])

        if last_action == 2:
            rollbacks += 1
        else:
            promotes += 1

    rollback_rate = rollbacks / n_trials * 100
    print(f"  {case_name}: Rollback {rollbacks}/{n_trials} ({rollback_rate:.0f}%),  Promote/Hold {promotes}/{n_trials}")
    return rollbacks, promotes


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Vui long chay Phase 9.1 (python -m training.offline_training) truoc.")
        return

    print("=" * 60)
    print("PHASE 9.2 — ACCEPTANCE TEST (RL Agent)")
    print("=" * 60)

    model = PPO.load(MODEL_PATH)
    env = CanaryEnv()

    N = 20  # số trial mỗi case

    cases = [
        ("Case A (ALL HEALTHY → Phải PROMOTE)", make_all_healthy_config(), "promote"),
        ("Case B (CPU-ONLY ANOMALY → Phải ROLLBACK) [KEY TEST]", make_cpu_only_config(), "rollback"),
        ("Case C (MEM-ONLY ANOMALY → Phải ROLLBACK)", make_mem_only_config(), "rollback"),
        ("Case D (ERROR+LATENCY ANOMALY → Phải ROLLBACK)", make_error_latency_config(), "rollback"),
        ("Case E (ALL CHANNELS ANOMALY → Phải ROLLBACK)", make_multi_anomalous_config(), "rollback"),
    ]

    results = {}
    all_passed = True

    for case_name, config, expected in cases:
        rollbacks, promotes = run_case(model, env, case_name, config, n_trials=N)
        rollback_rate = rollbacks / N

        if expected == "rollback":
            passed = rollback_rate >= 0.5  # ít nhất 50% trial rollback đúng
            symbol = "✓ PASS" if passed else "✗ FAIL"
        else:  # expected == "promote"
            promote_rate = promotes / N
            passed = promote_rate >= 0.5
            symbol = "✓ PASS" if passed else "✗ FAIL"

        print(f"    → {symbol}")
        results[case_name] = {"rollbacks": rollbacks, "promotes": promotes, "passed": passed}
        if not passed:
            all_passed = False

    print("\n" + "=" * 60)
    print("PHASE 9.3 — SO SANH RL vs RULE-BASED BASELINE")
    print("=" * 60)

    print("\nRule-based Baseline (threshold=2.0, 4 channels):")
    rb_reward, rb_latency, rb_fpr, rb_fnr = evaluate_rule_based_baseline(threshold=2.0, num_episodes=100)
    print(f"  Reward: {rb_reward:.2f} | Latency: {rb_latency:.1f} | FPR: {rb_fpr*100:.1f}% | FNR: {rb_fnr*100:.1f}%")

    print("\nRL Agent (trained model):")
    from training.offline_training import validate_model_locally
    rl_reward, rl_latency, rl_fpr, rl_fnr = validate_model_locally(MODEL_PATH, None, num_episodes=100)
    print(f"  Reward: {rl_reward:.2f} | Latency: {rl_latency:.1f} | FPR: {rl_fpr*100:.1f}% | FNR: {rl_fnr*100:.1f}%")

    print("\n" + "=" * 60)
    print("BANG SO SANH RL vs RULE-BASED")
    print("=" * 60)
    print(f"{'Metric':<20} {'Rule-Based':>12} {'RL Agent':>12} {'Target':>10}")
    print("-" * 56)
    print(f"{'FPR (Healthy eps)':<20} {rb_fpr*100:>11.1f}% {rl_fpr*100:>11.1f}% {'< 10%':>10}")
    print(f"{'FNR (Mixed eps)':<20} {rb_fnr*100:>11.1f}% {rl_fnr*100:>11.1f}% {'< 5%':>10}")
    print(f"{'Avg Reward':<20} {rb_reward:>12.2f} {rl_reward:>12.2f} {'> 5.0':>10}")
    print(f"{'Avg Latency':<20} {rb_latency:>12.1f} {rl_latency:>12.1f} {'(lower better)':>10}")

    print("\n" + "=" * 60)
    print("KET QUA ACCEPTANCE TEST")
    print("=" * 60)
    if all_passed:
        print("✓ TẤT CẢ CASE A-E PASSED")
    else:
        print("✗ MỘT SỐ CASE FAILED — xem chi tiet tren")

    rl_targets = rl_fpr <= 0.1 and rl_fnr <= 0.05
    if rl_targets:
        print("✓ RL Agent dat FPR < 10% va FNR < 5% (PASSED)")
    else:
        print(f"✗ RL Agent chua dat target: FPR={rl_fpr*100:.1f}% FNR={rl_fnr*100:.1f}%")


if __name__ == "__main__":
    main()
