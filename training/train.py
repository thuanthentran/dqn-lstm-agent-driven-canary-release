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
from core.env import CanaryEnv, SCENARIO_NAMES, ACTION_NAMES, MAX_STEPS_PER_EPISODE
 
# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
SEQ_LENGTH   = 10          # Độ dài subsequence dùng cho BPTT
BATCH_SIZE   = 32          # Số episode (sequence) mỗi lần optimize
GAMMA        = 0.99
LR           = 1e-4
 
EPS_START = 1.0
EPS_END   = 0.05
EPS_DECAY = float(os.getenv("EPS_DECAY", "0.999"))
 
TARGET_UPDATE     = 50
MOVING_AVG_WINDOW = int(os.getenv("MOVING_AVG_WINDOW", "100"))
TRAIN_EPISODES    = int(os.getenv("TRAIN_EPISODES", "20000"))
 
# Buffer lưu tối đa 3 000 episode ≈ 60 000 transitions (avg 20 steps/ep)
BUFFER_MAX_EPISODES = 3000
 
 
# ---------------------------------------------------------------------------
# Episode-based Replay Buffer
# ---------------------------------------------------------------------------
class EpisodeReplayBuffer:
    """
    Lưu trữ các episode hoàn chỉnh thay vì transition riêng lẻ.
 
    Khi sample, trả về các *subsequence liên tiếp* độ dài SEQ_LENGTH từ
    mỗi episode ngẫu nhiên — cho phép LSTM được train qua BPTT trên đúng
    chuỗi thời gian, thay vì các transition độc lập.
 
    Mỗi transition trong buffer có dạng:
        (state, action, reward, next_state, done)
        np.ndarray(8,), int, float, np.ndarray(8,), float
    """
 
    def __init__(self, maxlen: int = BUFFER_MAX_EPISODES):
        self.buffer: deque[list] = deque(maxlen=maxlen)
 
    def add_episode(self, episode: list) -> None:
        """
        Thêm một episode vào buffer.
        Episode ngắn hơn SEQ_LENGTH bị bỏ qua vì không thể tạo subsequence.
        """
        if len(episode) >= SEQ_LENGTH:
            self.buffer.append(episode)
 
    def sample(self, batch_size: int) -> list[list]:
        """
        Sample batch_size subsequence liên tiếp độ dài SEQ_LENGTH.
        Mỗi subsequence lấy từ một episode khác nhau (có thể trùng episode
        nếu buffer nhỏ hơn batch_size).
        """
        k = min(batch_size, len(self.buffer))
        episodes = random.sample(self.buffer, k)
        sequences = []
        for ep in episodes:
            start = random.randint(0, len(ep) - SEQ_LENGTH)
            sequences.append(ep[start : start + SEQ_LENGTH])
        return sequences
 
    def __len__(self) -> int:
        return len(self.buffer)
 
 
# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(self):
        self.env        = CanaryEnv()
        self.policy_net = DRQN().to(DEVICE)
        self.target_net = DRQN().to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=LR)
 
        # FIX 3: Buffer lưu theo episode, đủ ~60k transitions
        self.memory  = EpisodeReplayBuffer(maxlen=BUFFER_MAX_EPISODES)
        self.epsilon = EPS_START
 
        self.episode_rewards     = []
        self.episode_epsilons    = []
        self.moving_avg_rewards  = []
        self.best_moving_avg     = float("-inf")
 
        # Create models directory with absolute path
        self.models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
        )
        os.makedirs(self.models_dir, exist_ok=True)
 
    # ------------------------------------------------------------------
    # FIX 1: select_action nhận state đơn lẻ + hidden, trả về hidden mới
    # ------------------------------------------------------------------
    def select_action(self, state: np.ndarray, hidden=None):
        """
        Chọn action cho một timestep, duy trì LSTM hidden state xuyên suốt episode.
 
        Args:
            state:  observation tại bước hiện tại, shape (8,)
            hidden: tuple (h_n, c_n) từ bước trước, hoặc None ở đầu episode
 
        Returns:
            action (int), new_hidden (tuple)
        """
        if random.random() < self.epsilon:
            return self.env.action_space.sample(), hidden
 
        with torch.no_grad():
            # (1, 1, 8) — batch=1, seq_len=1
            state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(DEVICE)
            # return_all=False (mặc định): trả về shape (1, n_actions)
            q_values, new_hidden = self.policy_net(state_tensor, hidden)
            action = q_values.squeeze(0).max(0)[1].item()
 
        return action, new_hidden
 
    # ------------------------------------------------------------------
    # FIX 2 + FIX 4: optimize_model dùng sequence + Double DQN
    # ------------------------------------------------------------------
    def optimize_model(self):
        """
        Train DRQN trên subsequence liên tiếp với Double DQN target.
 
        - Sequences liên tiếp: LSTM được train qua đúng thứ tự thời gian (BPTT).
        - Double DQN: policy_net chọn next action, target_net ước lượng giá trị
          → giảm overestimation bias so với vanilla DQN.
        - hidden=None ở đầu mỗi sequence: chuẩn cho "random subsequence" DRQN.
        """
        if len(self.memory) < BATCH_SIZE:
            return
 
        # Sample batch_size subsequences (mỗi cái dài SEQ_LENGTH)
        sequences = self.memory.sample(BATCH_SIZE)
 
        # Xây dựng tensor từ list of sequences
        # Mỗi sequence: list of SEQ_LENGTH tuple (s, a, r, s', done)
        states      = torch.FloatTensor(
            np.array([[t[0] for t in seq] for seq in sequences])
        ).to(DEVICE)       # (B, T, 8)
 
        actions     = torch.LongTensor(
            np.array([[t[1] for t in seq] for seq in sequences])
        ).to(DEVICE)       # (B, T)
 
        rewards     = torch.FloatTensor(
            np.array([[t[2] for t in seq] for seq in sequences])
        ).to(DEVICE)       # (B, T)
 
        next_states = torch.FloatTensor(
            np.array([[t[3] for t in seq] for seq in sequences])
        ).to(DEVICE)       # (B, T, 8)
 
        dones       = torch.FloatTensor(
            np.array([[t[4] for t in seq] for seq in sequences])
        ).to(DEVICE)       # (B, T)
 
        # --- Current Q-values (cần gradient) ---
        # hidden=None: reset hidden state ở đầu mỗi sequence (standard DRQN)
        q_all, _ = self.policy_net(states, return_all=True)             # (B, T, n_actions)
        curr_q = q_all.gather(2, actions.unsqueeze(2)).squeeze(2)       # (B, T)
 
        # --- Double DQN target (không cần gradient) ---
        with torch.no_grad():
            # policy_net chọn action tốt nhất tại next_state
            next_q_policy, _ = self.policy_net(next_states, return_all=True)  # (B, T, n_actions)
            # target_net ước lượng giá trị của action đó
            next_q_target, _ = self.target_net(next_states, return_all=True)  # (B, T, n_actions)
 
        next_actions = next_q_policy.max(2)[1].unsqueeze(2)             # (B, T, 1)
        next_q       = next_q_target.gather(2, next_actions).squeeze(2) # (B, T)
        expected_q   = rewards + (1.0 - dones) * GAMMA * next_q        # (B, T)
 
        loss = nn.MSELoss()(curr_q, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
 
    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def train(self, episodes=8000, plot_path="logs/training_metrics.png"):
        episodes = int(os.getenv("TRAIN_EPISODES", str(episodes)))
        os.makedirs("logs", exist_ok=True)
 
        print(f"--- Bắt đầu huấn luyện trên {DEVICE} ---")
 
        for ep in range(episodes):
            state, _ = self.env.reset()
 
            # FIX 1: Reset hidden state ở đầu mỗi episode
            hidden          = None
            current_episode = []   # FIX 2: thu thập transitions theo episode
            total_reward    = 0
            action_trace    = []
 
            # FIX 5: dùng MAX_STEPS_PER_EPISODE từ env thay vì hardcode 100
            for _ in range(MAX_STEPS_PER_EPISODE + 10):  # +10 safety margin
                action, hidden = self.select_action(state, hidden)
                action_trace.append(ACTION_NAMES.get(action, f"Act{action}"))
 
                next_state, reward, done, _, _ = self.env.step(action)
 
                # Lưu transition: done ép sang float để dùng trong tensor
                current_episode.append(
                    (state, action, reward, next_state, float(done))
                )
 
                state         = next_state
                total_reward += reward
 
                if done:
                    break
 
            # Nạp episode vào buffer (bị bỏ qua nếu quá ngắn)
            self.memory.add_episode(current_episode)
            self.optimize_model()
 
            # Cập nhật epsilon
            self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)
            self.episode_rewards.append(total_reward)
            self.episode_epsilons.append(self.epsilon)
 
            window_start       = max(0, len(self.episode_rewards) - MOVING_AVG_WINDOW)
            moving_avg_reward  = float(np.mean(self.episode_rewards[window_start:]))
            self.moving_avg_rewards.append(moving_avg_reward)
 
            if moving_avg_reward > self.best_moving_avg:
                self.best_moving_avg = moving_avg_reward
                best_model_path = os.path.join(self.models_dir, "model_canary_drqn_best.pth")
                torch.save(self.policy_net.state_dict(), best_model_path)
 
            if ep % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
 
            if (ep + 1) % 10 == 0:
                scenario_name = SCENARIO_NAMES.get(self.env.scenario, "Unknown")
                action_seq    = " -> ".join(action_trace)
                print(
                    f"[{ep + 1:5d}/{episodes}] "
                    f"Reward={total_reward:8.2f} | "
                    f"MA{MOVING_AVG_WINDOW}={moving_avg_reward:8.2f} | "
                    f"BestMA={self.best_moving_avg:8.2f} | "
                    f"Scenario={scenario_name:<15} | "
                    f"Steps={len(action_trace):2d} | "
                    f"Eps={self.epsilon:.3f} | "
                    f"Buf={len(self.memory)}"
                )
                print(f"  Actions: {action_seq}")
 
        torch.save(
            self.policy_net.state_dict(),
            os.path.join(self.models_dir, "model_canary_drqn.pth")
        )
        self.plot_training_metrics(plot_path)
 
    def plot_training_metrics(self, output_path):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(self.episode_rewards, alpha=0.5, label="Reward")
        plt.plot(
            self.moving_avg_rewards,
            color="orange", linewidth=2.0,
            label=f"Moving Avg ({MOVING_AVG_WINDOW})"
        )
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