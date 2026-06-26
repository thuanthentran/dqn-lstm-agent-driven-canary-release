import os
import sys
import mlflow
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
# Chuẩn hóa đường dẫn
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from core.online_env import OnlineCanaryEnv

# Cấu hình MLflow & File
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "Canary_RL_Final_Production"

# Trọng số Offline (Điểm xuất phát)
OFFLINE_MODEL_PATH = os.path.join(BASE_DIR, "models", "ppo_lstm_offline_best.zip")
OFFLINE_NORM_PATH = os.path.join(BASE_DIR, "models", "vec_normalize.pkl")

# Trọng số Online (Đích đến)
ONLINE_SAVE_PATH = os.path.join(BASE_DIR, "models", "ppo_lstm_online_best")
ONLINE_NORM_PATH = os.path.join(BASE_DIR, "models", "vec_normalize_online.pkl")

class MLflowCallback(BaseCallback):
    def __init__(self, log_freq=50, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            for key, value in self.logger.name_to_value.items():
                mlflow.log_metric(f"online_{key}", value, step=self.num_timesteps)
        return True
    
def validate_model_locally(model_path, norm_path, num_episodes=10):
    print(f"\n🔍 Đang chạy Validate nội bộ với {num_episodes} tập...")
    
    # Khởi tạo môi trường thi (Đóng băng cập nhật hệ số chuẩn hóa)
    eval_env = DummyVecEnv([lambda: OnlineCanaryEnv()])
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

def train_online():
    # 1. KHỞI TẠO MÔI TRƯỜNG ONLINE (Kết nối K8s thật)
    print("🌍 Đang khởi tạo môi trường Online Canary Env...")
    # Chọn 'checkoutservice' làm vật tế thần để huấn luyện
    env = DummyVecEnv([lambda: OnlineCanaryEnv(service_name="checkoutservice")])
    
    # 2. LOAD LẠI BỘ CHUẨN HÓA OFFLINE
    print("🔄 Đang kế thừa bộ chuẩn hóa dữ liệu từ Phase 1...")
    env = VecNormalize.load(OFFLINE_NORM_PATH, env)
    env.training = True # Cho phép cập nhật nhẹ hệ số khi gặp dữ liệu mạng mới
    env.norm_reward = True

    # 3. KẾ THỪA BỘ NÃO VÀ TINH CHỈNH (FINE-TUNING)
    print("🧠 Đang bứng trọng số PPO+LSTM sang môi trường K8s...")
    # Ghi đè tham số: Giảm Learning Rate cực thấp để tránh mất trí nhớ (Catastrophic Forgetting)
    custom_objects = {
        "learning_rate": 3e-5, 
        "clip_range": 0.1
    }
    model = RecurrentPPO.load(OFFLINE_MODEL_PATH, env=env, custom_objects=custom_objects)

    # 4. GHI LOG LÊN MLFLOW
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="Online_FineTuning_Live_K8s"):
        mlflow.log_param("phase", "Online_Fine_Tuning")
        mlflow.log_param("learning_rate", 3e-5)
        
        print("🚀 BẮT ĐẦU HUẤN LUYỆN ONLINE (K8s API & Istio sẽ hoạt động)...")
        # Huấn luyện Online chậm hơn rất nhiều do phải chờ K8s thực thi lệnh
        # Chỉ chạy khoảng 2000 - 5000 steps là đủ để Agent tinh chỉnh
        TOTAL_TIMESTEPS = 2000 
        
        # reset_num_timesteps=False để MLflow nối tiếp biểu đồ của Phase 1
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=MLflowCallback(log_freq=10), reset_num_timesteps=False)
        # [MỚI] VALIDATE TRƯỚC KHI UP ARTIFACT
        print("\n🔍 Đang chạy Validate Online...")
        # Lưu model tạm để load vào hàm validate
        model.save(ONLINE_SAVE_PATH)
        mean_reward = validate_model_locally(f"{ONLINE_SAVE_PATH}.zip", OFFLINE_NORM_PATH) # Dùng norm offline hoặc online
        mlflow.log_metric("online_val_mean_reward", mean_reward)
        
        # [MỚI] GẮN TAG PASSED/FAILED
        if mean_reward >= 8.0: # Ngưỡng pass khắt khe hơn cho Phase Online
            mlflow.set_tag("status", "PASSED")
        else:
            mlflow.set_tag("status", "FAILED")
        # 5. LƯU BỘ NÃO HOÀN THIỆN
        print("\n💾 Đang lưu phiên bản Online Model...")
        model.save(ONLINE_SAVE_PATH)
        env.save(ONLINE_NORM_PATH)
        
        # Đẩy Artifacts lên kho
        mlflow.log_artifact(f"{ONLINE_SAVE_PATH}.zip", "online_model_weights")
        mlflow.log_artifact(ONLINE_NORM_PATH, "online_model_weights")
        mlflow.set_tag("status", "ONLINE_TRAINING_COMPLETED")

        print("🎉 HOÀN TẤT PHASE ONLINE!")

if __name__ == "__main__":
    train_online()