import os
import sys
import numpy as np
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_num_threads(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
from core.env import CanaryEnv

ONLINE_MODEL_PATH = os.path.join(BASE_DIR, "models", "ppo_lstm_online_best.zip")
ONLINE_NORM_PATH = os.path.join(BASE_DIR, "models", "vec_normalize_online.pkl")

def evaluate_online_model_offline():
    print("=" * 60)
    print("🧠 ĐÁNH GIÁ MÔ HÌNH HỌC ONLINE TRÊN MÔI TRƯỜNG GIẢ LẬP (OFFLINE)")
    print("=" * 60)

    if not os.path.exists(ONLINE_MODEL_PATH):
        print(f"❌ Không tìm thấy model tại {ONLINE_MODEL_PATH}")
        return
    if not os.path.exists(ONLINE_NORM_PATH):
        print(f"❌ Không tìm thấy VecNormalize tại {ONLINE_NORM_PATH}")
        return

    # Khởi tạo môi trường giả lập (Offline)
    env = DummyVecEnv([lambda: CanaryEnv()])
    env = VecNormalize.load(ONLINE_NORM_PATH, env)
    
    # Đóng băng quá trình chuẩn hóa (chỉ dùng thông số đã học được từ online)
    env.training = False 
    env.norm_reward = False 

    # Nạp mô hình
    print(f"✅ Đang nạp não bộ (PPO+LSTM) từ: {ONLINE_MODEL_PATH}")
    model = RecurrentPPO.load(ONLINE_MODEL_PATH, env=env)

    # Chạy thử 3 tập (episodes)
    num_episodes = 3
    for ep in range(num_episodes):
        print(f"\n" + "-"*50)
        print(f"🎬 BẮT ĐẦU TẬP KIỂM TRA SỐ {ep + 1}")
        print("-" * 50)
        
        obs = env.reset()
        done = False
        step = 0
        total_rew = 0
        
        while not done:
            step += 1
            # In ra chính xác những gì model nhìn thấy
            print(f"\n[Bước {step}]")
            print(f"   👁️ State vector truyền vào Model: {np.round(obs, 3)}")
            
            # Predict với deterministic=True để tắt cơ chế "thử nghiệm ngẫu nhiên" (exploration)
            # Qua đó bộc lộ đúng bản chất tư duy của Model
            action, lstm_states = model.predict(obs, deterministic=True)
            
            act_name = {0: "🟡 HOLD (0)", 1: "🟢 PROMOTE (1)", 2: "🔴 ABORT (2)"}.get(action[0], "UNKNOWN")
            print(f"   🤖 Model chốt hành động: {act_name}")
            
            obs, reward, done, info = env.step(action)
            
            print(f"   ⚖️ Kết quả: Reward = {reward[0]:.2f} | Hoàn thành? = {done[0]}")
            total_rew += reward[0]
            
        print(f"\n🏁 Kết thúc tập {ep + 1}. Tổng điểm đạt được: {total_rew:.2f}")

if __name__ == "__main__":
    evaluate_online_model_offline()
