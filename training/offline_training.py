import os
import sys
import pandas as pd
import numpy as np

# 1. FIX ĐỨNG LUỒNG CPU
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch

# 2. CHUẨN HÓA ĐƯỜNG DẪN THƯ MỤC LÀM VIỆC
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from core.env import CanaryEnv
from core.model import TransformerFeatureExtractor

LOG_DIR = os.path.join(BASE_DIR, "logs", "transformer_offline")
SAVE_PATH = os.path.join(BASE_DIR, "models", "ppo_transformer_offline_best")
NORM_SAVE_PATH = os.path.join(BASE_DIR, "models", "vec_normalize.pkl")
TOTAL_TIMESTEPS = 150_000

# --- CẤU HÌNH TRANSFORMER ---
_tmp_env = CanaryEnv()
TRANSFORMER_CONFIG = {
    "d_model": 64,
    "n_heads": 4,
    "n_heads_feature": 4,
    "n_layers": 2,
    "seq_len": 30,
    "n_features": _tmp_env.num_features,
    "dropout": 0.1,
}
del _tmp_env

# --- THIẾT BỊ ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


class ProgressCallback(BaseCallback):
    """In tiến trình training mỗi 10k steps."""
    def __init__(self, total_timesteps, verbose=0):
        super().__init__(verbose)
        self.total = total_timesteps

    def _on_step(self) -> bool:
        if self.num_timesteps % 10_000 == 0:
            pct = self.num_timesteps / self.total * 100
            print(f"   📈 [{pct:5.1f}%] {self.num_timesteps:,}/{self.total:,} timesteps")
        return True


def make_env(log_dir: str):
    def _init():
        return Monitor(CanaryEnv(), log_dir)
    return _init


def build_env(log_dir: str):
    env = DummyVecEnv([make_env(log_dir)])
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)
    return env


def build_model(vec_env, gamma=0.99, ent_coef=0.01):
    return PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        gamma=gamma,
        ent_coef=ent_coef,
        n_steps=2048,
        batch_size=256,
        policy_kwargs={
            "features_extractor_class": TransformerFeatureExtractor,
            "features_extractor_kwargs": TRANSFORMER_CONFIG,
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        },
        device=DEVICE,
    )


# HÀM VALIDATE NỘI BỘ VÀ TÍNH METRICS (FPR/FNR/Latency)
def validate_model_locally(model_path, norm_path, num_episodes=100, episode_configs=None):
    print(f"\n🔍 Đang chạy Validate nội bộ với {num_episodes} tập...")
    # Không dùng DummyVecEnv/VecNormalize khi eval vì build_env() set norm_obs=False
    # và norm_reward=False — VecNormalize là no-op hoàn toàn ở eval time.
    env_instance = CanaryEnv()
    model = PPO.load(model_path, device=DEVICE)

    # 2 nhãn: 0=Healthy, 1=Mixed/Anomalous
    metrics_by_scenario = {i: {"episodes": 0, "latency": [], "fp": 0, "fn": 0} for i in range(2)}
    rewards = []

    for ep in range(num_episodes):
        cfg = episode_configs[ep] if episode_configs else None
        obs, _ = env_instance.reset(episode_config=cfg)
        done = False
        total_rew = 0
        step_count = 0
        scenario = env_instance.scenario  # 0=Healthy, 1=Mixed
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
            if last_action == 2:  # Rollback khi Healthy -> False Positive
                metrics_by_scenario[scenario]["fp"] += 1
        else:
            # Promote tới cùng (action != 2 hoặc done tự nhiên) -> False Negative
            if last_action != 2:
                metrics_by_scenario[scenario]["fn"] += 1

    print(f"\n📊 KẾT QUẢ VALIDATE CHI TIẾT THEO KỊCH BẢN ({num_episodes} tập):")
    total_healthy = 0
    total_anomalous = 0
    total_fp = 0
    total_fn = 0
    all_latencies = []

    scenario_names = {0: "Healthy", 1: "Mixed/Anomalous"}

    for s in range(2):
        m = metrics_by_scenario[s]
        eps = m["episodes"]
        if eps == 0:
            continue

        lat_mean = np.mean(m["latency"])
        all_latencies.extend(m["latency"])

        is_healthy = s == 0
        if is_healthy:
            total_healthy += eps
            total_fp += m["fp"]
            rate = (m["fp"] / eps) * 100
            print(f"   - S{s} [{scenario_names[s]}]: {eps} eps | Latency: {lat_mean:.1f} | FPR: {rate:.1f}%")
        else:
            total_anomalous += eps
            total_fn += m["fn"]
            rate = (m["fn"] / eps) * 100
            print(f"   - S{s} [{scenario_names[s]}]: {eps} eps | Latency: {lat_mean:.1f} | FNR: {rate:.1f}%")

    mean_reward = np.mean(rewards)
    mean_latency = np.mean(all_latencies) if len(all_latencies) > 0 else 0.0
    fpr = (total_fp / total_healthy) if total_healthy > 0 else 0.0
    fnr = (total_fn / total_anomalous) if total_anomalous > 0 else 0.0

    print(f"\n📈 TỔNG KẾT ({num_episodes} tập):")
    print(f"   - Reward trung bình: {mean_reward:.2f}")
    print(f"   - Latency trung bình: {mean_latency:.1f} steps")
    print(f"   - Tỷ lệ FPR tổng (Healthy): {fpr*100:.1f}%")
    print(f"   - Tỷ lệ FNR tổng (Mixed/Anomalous): {fnr*100:.1f}%")

    return mean_reward, mean_latency, fpr, fnr


# HÀM VẼ BIỂU ĐỒ MƯỢT (Moving Average)
def plot_smoothed_curve(log_dir, save_path):
    monitor_path = os.path.join(log_dir, "monitor.csv")
    if not os.path.exists(monitor_path):
        print("⚠️ Không tìm thấy file monitor.csv để vẽ.")
        return False

    df = pd.read_csv(monitor_path, skiprows=1)
    df['r_smoothed'] = df['r'].rolling(window=100, min_periods=1).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['r'], alpha=0.2, color='tab:blue', label='Phần thưởng gốc (Nhiễu)')
    plt.plot(df.index, df['r_smoothed'], color='tab:red', linewidth=2.5, label='Trung bình động (100 Eps)')

    plt.xlabel('Số tập (Episodes)')
    plt.ylabel('Phần thưởng (Reward)')
    plt.title('TransformerPPO Canary Release: Đường cong học tập')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return True


def train() -> None:
    from training.sweep_optuna import run_sweep
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    print("🚀 BƯỚC 1: TỰ ĐỘNG CHẠY OPTUNA SWEEP ĐỂ TÌM THAM SỐ TỐI ƯU...")
    # Chạy 10 trials nhanh gọn (có thể tăng lên 50 ở production)
    best_params = run_sweep(n_trials=10)
    
    # Lấy thông số tối ưu
    best_gamma = best_params.get("gamma", 0.99)
    best_ent_coef = best_params.get("ent_coef", 0.01)
    best_early_bonus = best_params.get("early_bonus_scale", 2.0)
    
    # Thiết lập tham số môi trường
    os.environ["EARLY_BONUS_SCALE"] = str(best_early_bonus)
    print(f"\n🚀 BƯỚC 2: BẮT ĐẦU HUẤN LUYỆN OFFLINE VỚI BỘ THÔNG SỐ TỐI ƯU")
    print(f"   [Optuna] Gamma: {best_gamma:.4f} | Ent_coef: {best_ent_coef:.5f} | Early_Bonus: {best_early_bonus:.2f}")

    vec_env = build_env(LOG_DIR)
    model = build_model(vec_env, gamma=best_gamma, ent_coef=best_ent_coef)

    print(f"   Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"   Config: {TRANSFORMER_CONFIG}")

    progress_cb = ProgressCallback(TOTAL_TIMESTEPS)
    
    print("-" * 50)
    print(f"📊 Để xem đồ thị Tensorboard (Real-time), hãy mở Terminal/CMD mới và chạy:")
    print(f"   tensorboard --logdir \"{LOG_DIR}\"")
    print("-" * 50)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=progress_cb)

    # Lưu model cục bộ
    model.save(SAVE_PATH)
    vec_env.save(NORM_SAVE_PATH)
    print(f"💾 Model saved: {SAVE_PATH}.zip")
    print(f"💾 VecNormalize saved: {NORM_SAVE_PATH}")

    # TỰ ĐỘNG VALIDATE
    model_zip_path = f"{SAVE_PATH}.zip"
    mean_reward, mean_latency, fpr, fnr = validate_model_locally(model_zip_path, NORM_SAVE_PATH, num_episodes=100)

    if mean_reward >= 5.0 and fpr <= 0.1 and fnr <= 0.1:
        print("\n✅ Model đạt chuẩn xuất sắc (PASSED)!")
    else:
        print("\n⚠️ Model chưa đạt chuẩn (FAILED) hoặc cần điều chỉnh thêm.")

    # Vẽ Learning Curve
    print("📈 Đang vẽ Learning Curve mượt...")
    plot_path = os.path.join(LOG_DIR, "learning_curve_smoothed.png")
    if plot_smoothed_curve(LOG_DIR, plot_path):
        print(f"✅ Đã tạo đồ thị: {plot_path}")

    # GPU Memory Summary
    if DEVICE == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3
        print(f"🎮 GPU Peak Memory: {peak_mem:.2f} GB")

    print("🎉 QUÁ TRÌNH HUẤN LUYỆN VÀ VALIDATE KẾT THÚC!")

if __name__ == "__main__":
    train()