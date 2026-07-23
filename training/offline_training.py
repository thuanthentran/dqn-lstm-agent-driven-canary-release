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
TRANSFORMER_CONFIG = {
    "d_model": 64,
    "n_heads": 4,
    "n_heads_feature": 1,
    "n_layers": 2,
    "seq_len": 30,
    "n_features": 5,
    "dropout": 0.1,
}

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
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env


def build_model(vec_env):
    return PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        ent_coef=0.01,
        n_steps=512,
        batch_size=128,
        policy_kwargs={
            "features_extractor_class": TransformerFeatureExtractor,
            "features_extractor_kwargs": TRANSFORMER_CONFIG,
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        },
        device=DEVICE,
    )


# HÀM VALIDATE NỘI BỘ
def validate_model_locally(model_path, norm_path, num_episodes=10):
    print(f"\n🔍 Đang chạy Validate nội bộ với {num_episodes} tập...")

    eval_env = DummyVecEnv([lambda: CanaryEnv()])
    eval_env = VecNormalize.load(norm_path, eval_env)
    eval_env.training = False
    eval_env.norm_reward = False

    model = PPO.load(model_path, env=eval_env, device=DEVICE)

    rewards = []
    for ep in range(num_episodes):
        obs = eval_env.reset()
        done = False
        total_rew = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            total_rew += reward[0]
        rewards.append(total_rew)
        print(f"   - Bài test {ep + 1}/{num_episodes}: Điểm = {total_rew:.2f}")

    return np.mean(rewards)


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
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    vec_env = build_env(LOG_DIR)
    model = build_model(vec_env)

    print("🚀 Bắt đầu huấn luyện TransformerPPO...")
    print(f"   Timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"   Config: {TRANSFORMER_CONFIG}")

    progress_cb = ProgressCallback(TOTAL_TIMESTEPS)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=progress_cb)

    # Lưu model cục bộ
    model.save(SAVE_PATH)
    vec_env.save(NORM_SAVE_PATH)
    print(f"💾 Model saved: {SAVE_PATH}.zip")
    print(f"💾 VecNormalize saved: {NORM_SAVE_PATH}")

    # TỰ ĐỘNG VALIDATE
    model_zip_path = f"{SAVE_PATH}.zip"
    mean_reward = validate_model_locally(model_zip_path, NORM_SAVE_PATH)
    print(f"\n📊 KẾT QUẢ VALIDATE: Điểm trung bình = {mean_reward:.2f}")

    if mean_reward >= 5.0:
        print("✅ Model đạt chuẩn (PASSED)!")
    else:
        print("⚠️ Model chưa đạt chuẩn (FAILED)!")

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