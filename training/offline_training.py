import os
import sys
import pandas as pd # Dùng pandas để xử lý mượt biểu đồ
import numpy as np  # [MỚI] Dùng để tính toán điểm trung bình lúc Validate

# 1. FIX ĐỨNG LUỒNG CPU
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)

# 2. CHUẨN HÓA ĐƯỜNG DẪN THƯ MỤC LÀM VIỆC
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import matplotlib.pyplot as plt
import mlflow
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from core.env import CanaryEnv

# 3. CẤU HÌNH MLFLOW CLIENT-SERVER QUA HTTP PROXY
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Canary_RL_Final_Production"

LOG_DIR = os.path.join(BASE_DIR, "logs", "lstm_offline")
SAVE_PATH = os.path.join(BASE_DIR, "models", "ppo_lstm_offline_best")
NORM_SAVE_PATH = os.path.join(BASE_DIR, "models", "vec_normalize.pkl")
TOTAL_TIMESTEPS = 150000

class MLflowCallback(BaseCallback):
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            for key, value in self.logger.name_to_value.items():
                mlflow.log_metric(key, value, step=self.num_timesteps)
        return True

def configure_mlflow() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

def make_env(log_dir: str):
    def _init():
        return Monitor(CanaryEnv(), log_dir)
    return _init

def build_env(log_dir: str):
    env = DummyVecEnv([make_env(log_dir)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return env

def build_model(vec_env):
    return RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        ent_coef=0.01,
        n_steps=512,
        batch_size=128,
        policy_kwargs={
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
            "lstm_hidden_size": 64,
            "n_lstm_layers": 1,
        },
    )

# [MỚI] HÀM VALIDATE NỘI BỘ
def validate_model_locally(model_path, norm_path, num_episodes=10):
    print(f"\n🔍 Đang chạy Validate nội bộ với {num_episodes} tập...")
    
    # Khởi tạo môi trường thi (Đóng băng cập nhật hệ số chuẩn hóa)
    eval_env = DummyVecEnv([lambda: CanaryEnv()])
    eval_env = VecNormalize.load(norm_path, eval_env)
    eval_env.training = False 
    eval_env.norm_reward = False 
    
    model = RecurrentPPO.load(model_path, env=eval_env)
    
    rewards = []
    for ep in range(num_episodes):
        obs = eval_env.reset()
        done = False
        total_rew = 0
        while not done:
            # deterministic=True ép Agent dùng năng lực thực sự, không đoán mò
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = eval_env.step(action)
            total_rew += reward[0]
        rewards.append(total_rew)
        print(f"   - Bài test {ep + 1}/10: Điểm = {total_rew:.2f}")
    
    return np.mean(rewards)

# 4. HÀM MỚI: Vẽ biểu đồ mượt (Moving Average)
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
    plt.title('PPO+LSTM Canary Release: Đường cong học tập')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return True

def train() -> None:
    configure_mlflow()
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    with mlflow.start_run(run_name="Upload_HTTP_Proxy_With_Validation"):
        vec_env = build_env(LOG_DIR)
        model = build_model(vec_env)

        mlflow.log_param("total_timesteps", TOTAL_TIMESTEPS)
        mlflow.log_param("architecture", "PPO+LSTM")

        print("🚀 Bắt đầu huấn luyện PPO+LSTM...")
        mlflow_callback = MLflowCallback(log_freq=1000)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=mlflow_callback)

        # Lưu model cục bộ
        model.save(SAVE_PATH)
        vec_env.save(NORM_SAVE_PATH)
        
        # [MỚI] TỰ ĐỘNG VALIDATE TRƯỚC KHI ĐẨY FILE
        model_zip_path = f"{SAVE_PATH}.zip"
        mean_reward = validate_model_locally(model_zip_path, NORM_SAVE_PATH)
        mlflow.log_metric("val_mean_reward", mean_reward)
        print(f"\n📊 KẾT QUẢ VALIDATE: Điểm trung bình = {mean_reward:.2f}")
        
        # [MỚI] GÁN TAG TỰ ĐỘNG TRÊN MLFLOW
        if mean_reward >= 5.0: # Ngưỡng Pass (Bạn có thể tinh chỉnh số này)
            mlflow.set_tag("status", "PASSED")
            print("✅ Model đạt chuẩn, đã gắn tag PASSED!")
        else:
            mlflow.set_tag("status", "FAILED")
            print("⚠️ Model chưa đạt chuẩn, đã gắn tag FAILED!")
        
        # Log artifacts (Model) qua mạng lên WSL
        if os.path.exists(model_zip_path):
            mlflow.log_artifact(model_zip_path, "model_weights")
            print(f"✅ Đã log Model Zip lên máy chủ.")
            
        if os.path.exists(NORM_SAVE_PATH):
            mlflow.log_artifact(NORM_SAVE_PATH, "model_weights")
            print(f"✅ Đã log Model Normalize lên máy chủ.")

        print("📈 Đang vẽ Learning Curve mượt...")
        plot_path = os.path.join(LOG_DIR, "learning_curve_smoothed.png")
        if plot_smoothed_curve(LOG_DIR, plot_path):
            mlflow.log_artifact(plot_path, "plots")
            print("✅ Đã tạo và log ảnh đồ thị mượt thành công!")

        print("🎉 QUÁ TRÌNH HUẤN LUYỆN, VALIDATE VÀ ĐỒNG BỘ KẾT THÚC!")

if __name__ == "__main__":
    train()