import os
import sys
import optuna
import numpy as np
import matplotlib.pyplot as plt

# FIX ĐỨNG LUỒNG CPU
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from core.env import CanaryEnv
from training.offline_training import validate_model_locally, TRANSFORMER_CONFIG, DEVICE

# Số lượng timestep ngắn để quét nhanh
SWEEP_TIMESTEPS = 30_000
NUM_EVAL_EPISODES = 50

def build_env():
    env = DummyVecEnv([lambda: CanaryEnv()])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)
    return env

def objective(trial):
    """
    Optuna objective: Quét các tham số gamma, ent_coef, và EARLY_BONUS_SCALE.
    Mục tiêu đa lượng: Maximize (reward), Minimize (Latency), Minimize (FPR + FNR)
    Ở đây sẽ gộp thành 1 điểm số (Scalar) để dễ tối ưu hoặc tối ưu đa mục tiêu (Multi-objective).
    Ta sẽ dùng Multi-objective để ra đường cong Pareto.
    """
    gamma = trial.suggest_float("gamma", 0.8, 0.999, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.1, log=True)
    early_bonus_scale = trial.suggest_float("early_bonus_scale", 0.5, 2.5)

    # Đặt biến môi trường để env.py đọc được
    os.environ["EARLY_BONUS_SCALE"] = str(early_bonus_scale)

    vec_env = build_env()
    
    from core.model import TransformerFeatureExtractor
    
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        learning_rate=3e-4,
        gamma=gamma,
        ent_coef=ent_coef,
        n_steps=512,
        batch_size=128,
        policy_kwargs={
            "features_extractor_class": TransformerFeatureExtractor,
            "features_extractor_kwargs": TRANSFORMER_CONFIG,
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        },
        device=DEVICE,
    )

    # Huấn luyện nhanh
    model.learn(total_timesteps=SWEEP_TIMESTEPS)

    # Lưu tạm
    temp_model_path = f"models/optuna_temp_model_{trial.number}.zip"
    temp_norm_path = f"models/optuna_temp_norm_{trial.number}.pkl"
    model.save(temp_model_path)
    vec_env.save(temp_norm_path)

    # Đánh giá
    mean_reward, mean_latency, fpr, fnr = validate_model_locally(temp_model_path, temp_norm_path, num_episodes=NUM_EVAL_EPISODES)

    # Xóa file tạm
    if os.path.exists(temp_model_path): os.remove(temp_model_path)
    if os.path.exists(temp_norm_path): os.remove(temp_norm_path)

    # Trả về 4 mục tiêu độc lập: Reward (Max), Latency (Min), FPR (Min), FNR (Min)
    # Tách FPR và FNR thành 2 objectives độc lập để Optuna buộc phải tối ưu cả hai đồng thời
    # thay vì đánh đổi FNR=0 bù đắp cho FPR=80%
    return -mean_reward, mean_latency, fpr, fnr


def evaluate_rule_based_baseline(threshold=2.0, num_episodes=100):
    """
    Giả lập một Rule-based baseline: Nếu e_ratio_n hoặc l_ratio_n > threshold thì Rollback.
    Ngược lại nếu weight < 1.0 thì Promote đều đặn.
    """
    env = CanaryEnv()
    rewards = []
    latencies = []
    fp = 0
    fn = 0
    total_healthy = 0
    total_anomalous = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_rew = 0
        step_count = 0
        scenario = env.scenario
        last_action = 0

        while not done:
            # Lấy raw metric trực tiếp từ env để rule-based ra quyết định
            norm = env.latest_norm
            e_ratio   = norm.get("e_ratio_n", 0) * 5.0  # Giải mã MAX_RATIO
            l_ratio   = norm.get("l_ratio_n", 0) * 5.0
            cpu_ratio = norm.get("cpu_ratio_n", 0) * 5.0
            mem_ratio = norm.get("mem_ratio_n", 0) * 5.0

            # Agent nhìn được 4 channel thì rule-based cũng phải nhìn được 4 channel
            if e_ratio > threshold or l_ratio > threshold or cpu_ratio > threshold or mem_ratio > threshold:
                action = 2  # Rollback
            else:
                action = 1  # Promote

            obs, reward, done, _, _ = env.step(action)
            total_rew += float(reward)
            step_count += 1
            last_action = action

        rewards.append(total_rew)
        latencies.append(step_count)
        
        is_healthy = scenario == 0  # nhãn coarse mới (0=Healthy, 1=Mixed)
        if is_healthy:
            total_healthy += 1
            if last_action == 2: fp += 1
        else:
            total_anomalous += 1
            if last_action != 2: fn += 1

    mean_reward = np.mean(rewards)
    mean_latency = np.mean(latencies)
    fpr = (fp / total_healthy) if total_healthy > 0 else 0.0
    fnr = (fn / total_anomalous) if total_anomalous > 0 else 0.0

    return mean_reward, mean_latency, fpr, fnr


def run_sweep(n_trials=10):
    # 1. Quét Rule-based Baseline để vẽ đường Pareto
    print("========================================")
    print("1. RUNNING RULE-BASED BASELINE SWEEP")
    print("========================================")
    thresholds = [1.5, 2.0, 2.5, 3.0]
    baseline_results = []
    for th in thresholds:
        rew, lat, fpr, fnr = evaluate_rule_based_baseline(th, num_episodes=200)
        baseline_results.append((th, rew, lat, fpr, fnr))
        print(f"Threshold: {th:.1f} | Reward: {rew:.1f} | Latency: {lat:.1f} | FPR: {fpr*100:.1f}% | FNR: {fnr*100:.1f}%")

    # 2. Quét RL Model bằng Optuna (Multi-objective)
    print("\n========================================")
    print("2. RUNNING RL HYPERPARAMETER SWEEP")
    print("========================================")
    # create_study cho multi-objective: [Reward(min vì đã nhân -1), Latency(min), FPR(min), FNR(min)]
    # Tách FPR và FNR thành 2 objectives độc lập để Pareto front chính xác hơn
    study = optuna.create_study(directions=["minimize", "minimize", "minimize", "minimize"])
    
    # Rút ngắn số trial xuống 10 để chạy demo. Trên thực tế có thể đặt n_trials=50
    study.optimize(objective, n_trials=n_trials)

    print("\n--- Paretos Front (Best Trials) ---")
    best_trials = study.best_trials
    for trial in best_trials:
        print(f"Trial {trial.number}:")
        print(f"  Params: {trial.params}")
        print(f"  Reward: {-trial.values[0]:.2f}")
        print(f"  Latency: {trial.values[1]:.2f}")
        print(f"  FPR: {trial.values[2]*100:.2f}%")
        print(f"  FNR: {trial.values[3]*100:.2f}%")

    # Chọn trial tối ưu nhất: ưu tiên FPR thấp (< 0.3), sau đó mới xét Reward cao nhất.
    # Thứ tự: FPR < 0.3 & reward cao > FPR < 0.5 & reward cao > tất cả & reward cao.
    def select_best(trials):
        # Thứ tự ưu tiên: FPR < 10% & FNR < 5% (cả 2 đều đạt target)
        # → FPR < 10% (ignore FNR) → FPR < 30% → tất cả
        # Trong mỗi tier, chọn trial có reward cao nhất (min của -reward).
        # Lý do thêm FNR filter: nếu chỉ filter FPR mà bỏ qua FNR,
        # Optuna có thể chọn trial FNR=8.8% dù có trial FNR=0% tồn tại.
        tier_best = [t for t in trials if t.values[2] < 0.1 and t.values[3] < 0.05]
        tier1 = [t for t in trials if t.values[2] < 0.1]
        tier2 = [t for t in trials if t.values[2] < 0.3]
        tier3 = [t for t in trials if t.values[2] < 0.5]
        candidates = (tier_best if tier_best
                      else tier1 if tier1
                      else tier2 if tier2
                      else tier3 if tier3
                      else trials)
        return min(candidates, key=lambda t: t.values[0])

    best_trial = select_best(best_trials)
    print(f"\n🏆 ĐÃ CHỌN BỘ THÔNG SỐ TỐI ƯU (FPR ưu tiên, sau đó Reward):")
    print(f"  Params: {best_trial.params}")
    return best_trial.params

if __name__ == "__main__":
    run_sweep(n_trials=10)
