import os
import sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import numpy as np
import time

class DualLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = DualLogger("training_2000_steps.log")
sys.stderr = sys.stdout

torch.set_num_threads(1)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from core.online_env import OnlineCanaryEnv, C # Import bảng màu từ online_env

LOG_DIR = os.path.join(BASE_DIR, "logs", "lstm_online")
OFFLINE_MODEL_PATH = os.path.join(BASE_DIR, "models", "ppo_lstm_offline_best.zip")
OFFLINE_NORM_PATH = os.path.join(BASE_DIR, "models", "vec_normalize.pkl")
ONLINE_SAVE_PATH = os.path.join(BASE_DIR, "models", "ppo_lstm_online_best")
ONLINE_NORM_PATH = os.path.join(BASE_DIR, "models", "vec_normalize_online.pkl")

TOTAL_TIMESTEPS = 2000 # ~500 episodes (4 steps each). Chạy qua đêm.
FINE_TUNE_LR = 1e-5 

import psutil

class DebugCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(DebugCallback, self).__init__(verbose)
        self.stuck_counter = 0
        self.last_step_time = time.time() if 'time' in sys.modules else 0

    def _on_step(self) -> bool:
        if self.num_timesteps % 5 == 0:
            print(f"{C.GREY}   [Tiến trình] PPO Agent đã học được {self.num_timesteps}/{TOTAL_TIMESTEPS} steps...{C.END}")
        
        # 1. Kiểm tra RAM hệ thống rò rỉ
        ram_percent = psutil.virtual_memory().percent
        if ram_percent > 98.0:
            print(f"{C.RED}🚨 BÁO ĐỘNG: Tràn RAM ({ram_percent}%). DỪNG HUẤN LUYỆN KHẨN CẤP!{C.END}")
            return False # Trả về False sẽ ép PPO.learn() dừng lại ngay lập tức
            
        return True

def make_online_env(log_dir: str):
    return lambda: Monitor(OnlineCanaryEnv(), log_dir)

def train_online() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(ONLINE_SAVE_PATH), exist_ok=True)

    print(f"\n{C.BOLD}{C.BLUE}======================================================{C.END}")
    print(f"{C.BOLD}{C.BLUE}   KHỞI TẠO HỆ THỐNG DIGITAL TWIN HỌC TRỰC TUYẾN{C.END}")
    print(f"{C.BOLD}{C.BLUE}======================================================{C.END}")
    env = DummyVecEnv([make_online_env(LOG_DIR)])

    print(f"{C.BLUE}📥 Nạp Kính râm (VecNormalize):{C.END} {OFFLINE_NORM_PATH}")
    env = VecNormalize.load(OFFLINE_NORM_PATH, env)
    env.training = True 
    env.norm_reward = True

    print(f"{C.BLUE}🧠 Nạp Bộ não (PPO+LSTM):{C.END} {OFFLINE_MODEL_PATH}")
    model = RecurrentPPO.load(
        OFFLINE_MODEL_PATH, 
        env=env, 
        custom_objects={"learning_rate": FINE_TUNE_LR, "clip_range": 0.1, "n_steps": 4, "batch_size": 4, "ent_coef": 0.0},
        tensorboard_log=LOG_DIR
    )

    print(f"\n{C.BOLD}{C.GREEN}🚀 BẮT ĐẦU HUẤN LUYỆN TRỰC TIẾP TRÊN K3S CỤM THẬT!{C.END}")
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[DebugCallback()], reset_num_timesteps=True)

    print(f"\n{C.BOLD}{C.YELLOW}💾 Đang lưu mô hình Online cục bộ...{C.END}")
    model.save(ONLINE_SAVE_PATH)
    env.save(ONLINE_NORM_PATH)
    print(f"{C.BOLD}{C.GREEN}✅ QUÁ TRÌNH ONLINE TRAINING KẾT THÚC THÀNH CÔNG!{C.END}\n")

if __name__ == "__main__":
    train_online()