import os

# Windows/Conda on this workspace can load duplicate OpenMP runtimes through
# numpy/torch/matplotlib. Set the compatibility flag before importing those libs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

# Thêm đường dẫn để import từ thư mục core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model import DRQN
from core.env import CanaryEnv, SCENARIO_NAMES, ACTION_NAMES

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LENGTH = 10
BATCH_SIZE = 32
GAMMA = 0.99
LR = 1e-4
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.05, float(os.getenv("EPS_DECAY", "0.999"))
TARGET_UPDATE = 10
MOVING_AVG_WINDOW = int(os.getenv("MOVING_AVG_WINDOW", "100"))
TRAIN_EPISODES = int(os.getenv("TRAIN_EPISODES", "12000"))

class Trainer:
    def __init__(self):
        self.env = CanaryEnv()
        self.policy_net = DRQN().to(DEVICE)
        self.target_net = DRQN().to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.memory = deque(maxlen=5000)
        self.epsilon = EPS_START
        self.episode_rewards = []
        self.episode_epsilons = []
        self.moving_avg_rewards = []
        self.best_moving_avg = float("-inf")
        # Create models directory with absolute path
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        os.makedirs(self.models_dir, exist_ok=True)

    def select_action(self, state_seq):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_seq).unsqueeze(0).to(DEVICE)
            q_values, _ = self.policy_net(state_tensor)
            return q_values.max(1)[1].item()

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE: return
        batch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)

        state_b = torch.FloatTensor(np.array(states)).to(DEVICE)
        next_state_b = torch.FloatTensor(np.array(next_states)).to(DEVICE)
        action_b = torch.LongTensor(actions).to(DEVICE)
        reward_b = torch.FloatTensor(rewards).to(DEVICE)
        done_b = torch.FloatTensor(dones).to(DEVICE)

        curr_q, _ = self.policy_net(state_b)
        curr_q = curr_q.gather(1, action_b.unsqueeze(1)).squeeze(1)
        next_q, _ = self.target_net(next_state_b)
        expected_q = reward_b + (1 - done_b) * GAMMA * next_q.max(1)[0]

        loss = nn.MSELoss()(curr_q, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def train(self, episodes=8000, plot_path="logs/training_metrics.png"):
        episodes = int(os.getenv("TRAIN_EPISODES", str(episodes)))
        os.makedirs("logs", exist_ok=True)
        
        print(f"--- Bắt đầu huấn luyện trên {DEVICE} ---")
        for ep in range(episodes):
            state, _ = self.env.reset()
            history = deque([state] * SEQ_LENGTH, maxlen=SEQ_LENGTH)
            total_reward = 0
            action_trace = []
            
            for t in range(100):
                state_seq = np.array(history)
                action = self.select_action(state_seq)
                action_trace.append(ACTION_NAMES.get(action, f"Act{action}"))
                
                next_state, reward, done, _, _ = self.env.step(action)
                next_history = list(history)[1:] + [next_state]
                self.memory.append((state_seq, action, reward, np.array(next_history), done))
                
                history.append(next_state)
                total_reward += reward
                self.optimize_model()
                if done: break
            
            self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)
            self.episode_rewards.append(total_reward)
            self.episode_epsilons.append(self.epsilon)

            window_start = max(0, len(self.episode_rewards) - MOVING_AVG_WINDOW)
            moving_avg_reward = float(np.mean(self.episode_rewards[window_start:]))
            self.moving_avg_rewards.append(moving_avg_reward)

            if moving_avg_reward > self.best_moving_avg:
                self.best_moving_avg = moving_avg_reward
                best_model_path = os.path.join(self.models_dir, "model_canary_drqn_best.pth")
                torch.save(self.policy_net.state_dict(), best_model_path)

            if ep % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            if (ep + 1) % 10 == 0:
                scenario_name = SCENARIO_NAMES.get(self.env.scenario, "Unknown")
                action_seq = " -> ".join(action_trace)
                print(f"[{ep + 1:5d}/{episodes}] Reward={total_reward:8.2f} | MA{MOVING_AVG_WINDOW}={moving_avg_reward:8.2f} | BestMA={self.best_moving_avg:8.2f} | Scenario={scenario_name:<15} | Steps={len(action_trace):2d} | Eps={self.epsilon:.3f}")
                print(f"  Actions: {action_seq}")

        torch.save(self.policy_net.state_dict(), os.path.join(self.models_dir, "model_canary_drqn.pth"))
        self.plot_training_metrics(plot_path)

    def plot_training_metrics(self, output_path):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.episode_rewards, alpha=0.5, label="Reward")
        plt.plot(self.moving_avg_rewards, color="orange", linewidth=2.0, label=f"Moving Avg ({MOVING_AVG_WINDOW})")
        plt.title("Training Reward Progress")
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(self.episode_epsilons, color='green', label="Epsilon")
        plt.title("Exploration Rate (Epsilon)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"[PLOT] Đã lưu biểu đồ tại: {output_path}")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train(episodes=TRAIN_EPISODES)