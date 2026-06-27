import os
import sys
import mlflow
import torch
import numpy as np

# 1. FIX ĐỨNG LUỒNG CPU TỪ PYTORCH
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# 2. CHUẨN HÓA ĐƯỜNG DẪN DỰ ÁN
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# Import môi trường Online kết nối K3s thật
from core.online_env import OnlineCanaryEnv

# --- CẤU HÌNH HỆ THỐNG ---
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Canary_RL_Phase3_Online_Live"
LOG_DIR = os.path.join(BASE_DIR, "logs", "lstm_online")

# Trọng số Offline (Điểm xuất phát)
OFFLINE_MODEL_PATH = os.path.join(BASE_DIR, "models", "ppo_lstm_offline_best.zip")
OFFLINE_NORM_PATH = os.path.join(BASE_DIR, "models", "vec_normalize.pkl")

# Trọng số Online (Đích đến sau khi tinh chỉnh)
ONLINE_SAVE_PATH = os.path.join(BASE_DIR, "models", "ppo_lstm_online_best")
ONLINE_NORM_PATH = os.path.join(BASE_DIR, "models", "vec_normalize_online.pkl")

# Chạy trên K8s thật tốn nhiều thời gian (vài giây/step), nên số step sẽ ít lại
TOTAL_TIMESTEPS = 2000 
FINE_TUNE_LR = 1e-5 # Learning rate siêu nhỏ để học chậm từ từ

class MLflowCallback(BaseCallback):
    def __init__(self, log_freq=10, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        # Cập nhật metric lên MLflow thường xuyên hơn vì online step rất chậm
        if self.n_calls % self.log_freq == 0:
            for key, value in self.logger.name_to_value.items():
                mlflow.log_metric(f"online_{key}", value, step=self.num_timesteps)
        return True

def configure_mlflow() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

def make_online_env(log_dir: str):
    def _init():
        # Gọi OnlineCanaryEnv (Môi trường gọi lệnh kubectl và chọc Prometheus API thật)
        return Monitor(OnlineCanaryEnv(), log_dir)
    return _init

def train_online() -> None:
    configure_mlflow()
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(ONLINE_SAVE_PATH), exist_ok=True)

    print("🔄 Đang khởi tạo kết nối K3s Live Environment...")
    env = DummyVecEnv([make_online_env(LOG_DIR)])

    # 1. Tải bộ chuẩn hóa dữ liệu từ Phase Offline
    print(f"📥 Đang nạp kính râm (VecNormalize) từ: {OFFLINE_NORM_PATH}")
    env = VecNormalize.load(OFFLINE_NORM_PATH, env)
    
    # Cho phép hệ thống cập nhật nhẹ nhàng tham số mean/std theo data mới
    env.training = True 
    env.norm_reward = True

    # 2. Tải bộ não Offline và ép xung (Fine-tune)
    print(f"🧠 Đang nạp bộ não (PPO+LSTM) từ: {OFFLINE_MODEL_PATH}")
    custom_objects = {
        "learning_rate": FINE_TUNE_LR,
        "clip_range": 0.1 # Giảm biên độ update policy để tránh làm gãy não Agent vì độ trễ mạng
    }
    model = RecurrentPPO.load(
        OFFLINE_MODEL_PATH, 
        env=env, 
        custom_objects=custom_objects,
        tensorboard_log=LOG_DIR
    )

    with mlflow.start_run(run_name="Online_Fine_Tuning_K3s"):
        mlflow.log_param("total_timesteps", TOTAL_TIMESTEPS)
        mlflow.log_param("fine_tune_lr", FINE_TUNE_LR)

        print("🚀 Bắt đầu huấn luyện Trực Tuyến trên Digital Twin K3s...")
        mlflow_callback = MLflowCallback(log_freq=10)
        
        # Bắt đầu vòng lặp tương tác với K3s
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=mlflow_callback,
            reset_num_timesteps=True 
        )

        print("💾 Đang lưu mô hình Online...")
        model.save(ONLINE_SAVE_PATH)
        env.save(ONLINE_NORM_PATH)

        # Log artifacts qua HTTP Proxy của MLflow lên WSL
        if os.path.exists(f"{ONLINE_SAVE_PATH}.zip"):
            mlflow.log_artifact(f"{ONLINE_SAVE_PATH}.zip", "model_weights")
        if os.path.exists(ONLINE_NORM_PATH):
            mlflow.log_artifact(ONLINE_NORM_PATH, "model_weights")

        print("✅ QUÁ TRÌNH ONLINE TRAINING KẾT THÚC VÀ ĐÃ ĐỒNG BỘ!")

if __name__ == "__main__":
    train_online()