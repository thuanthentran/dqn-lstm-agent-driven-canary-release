import os
 
# Windows/Conda on this workspace can load duplicate OpenMP runtimes through
# numpy/torch/matplotlib. Set the compatibility flag before importing those libs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
 
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
 
# Thêm đường dẫn để import từ thư mục core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model import PPOTCNPolicy
from core.env import CanaryEnv, SCENARIO_NAMES, ACTION_NAMES, MAX_STEPS_PER_EPISODE, SEQ_LEN
 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PPO hyperparameters
SEQ_LENGTH = SEQ_LEN
ROLLOUT_STEPS = int(os.environ.get("ROLLOUT_STEPS", "256"))
PPO_EPOCHS = int(os.environ.get("PPO_EPOCHS", "2"))
MINI_BATCH_SIZE = int(os.environ.get("MINI_BATCH_SIZE", "32"))
GAMMA = float(os.environ.get("GAMMA", "0.99"))
GAE_LAMBDA = float(os.environ.get("GAE_LAMBDA", "0.95"))
CLIP_RATIO = float(os.environ.get("CLIP_RATIO", "0.2"))
LR = float(os.environ.get("LR", "3e-4"))
VALUE_COEF = float(os.environ.get("VALUE_COEF", "0.5"))
ENTROPY_COEF = float(os.environ.get("ENTROPY_COEF", "0.01"))

MOVING_AVG_WINDOW = int(os.getenv("MOVING_AVG_WINDOW", "100"))
TRAIN_UPDATES = int(os.getenv("TRAIN_UPDATES", "1000"))
 
 
# ---------------------------------------------------------------------------
# Episode-based Replay Buffer
# ---------------------------------------------------------------------------
class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.states)
 
 
# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    def __init__(self):
        self.env = CanaryEnv()
        self.policy = PPOTCNPolicy(in_channels=5, seq_len=SEQ_LENGTH, action_dim=3).to(DEVICE)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)

        self.buffer = RolloutBuffer()

        self.episode_rewards = []
        self.moving_avg_rewards = []
        self.best_moving_avg = float("-inf")

        # models dir
        self.models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        os.makedirs(self.models_dir, exist_ok=True)
 
    def select_action(self, state):
        # state: np.ndarray (C, T)
        state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)  # (1, C, T)
        with torch.no_grad():
            logits, value = self.policy(state_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), log_prob.item(), value.item()

    def _compute_gae(self, rewards, values, dones, next_value, gamma=GAMMA, lam=GAE_LAMBDA):
        values = np.append(values, next_value)
        gae = 0.0
        advantages = np.zeros_like(rewards, dtype=np.float32)
        for step in reversed(range(len(rewards))):
            mask = 1.0 - float(dones[step])
            delta = rewards[step] + gamma * values[step + 1] * mask - values[step]
            gae = delta + gamma * lam * mask * gae
            advantages[step] = gae
        returns = advantages + values[:-1]
        return advantages, returns
 
    def ppo_update(self, advantages, returns):
        # Convert storage to tensors
        states = torch.stack([torch.FloatTensor(s) for s in self.buffer.states]).to(DEVICE)  # (N, C, T)
        actions = torch.LongTensor(self.buffer.actions).to(DEVICE)
        old_log_probs = torch.FloatTensor(self.buffer.log_probs).to(DEVICE)
        advantages = torch.FloatTensor(advantages).to(DEVICE)
        returns = torch.FloatTensor(returns).to(DEVICE)

        N = states.size(0)
        inds = np.arange(N)
        for _ in range(PPO_EPOCHS):
            np.random.shuffle(inds)
            for start in range(0, N, MINI_BATCH_SIZE):
                mb_inds = inds[start : start + MINI_BATCH_SIZE]
                mb_states = states[mb_inds]
                mb_actions = actions[mb_inds]
                mb_old_logp = old_log_probs[mb_inds]
                mb_adv = advantages[mb_inds]
                mb_ret = returns[mb_inds]

                logits, values = self.policy(mb_states)
                dist = torch.distributions.Categorical(logits=logits)
                entropy = dist.entropy().mean()
                new_logp = dist.log_prob(mb_actions)

                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - CLIP_RATIO, 1.0 + CLIP_RATIO) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values, mb_ret)

                loss = actor_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()

    def train(self):
        os.makedirs("logs", exist_ok=True)
        print(f"--- Starting PPO training on {DEVICE} ---")

        update = 0
        while update < TRAIN_UPDATES:
            # Collect rollouts
            self.buffer.clear()
            timesteps_collected = 0
            ep_rewards = []
            ep_reward = 0.0
            ep_action_traces = []
            ep_scenarios = []
            current_episode_actions = []

            state, _ = self.env.reset()
            while timesteps_collected < ROLLOUT_STEPS:
                action, logp, value = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)

                # record action for current episode trace
                current_episode_actions.append(int(action))

                self.buffer.add(state, action, logp, reward, value, float(done))

                state = next_state
                ep_reward += reward
                timesteps_collected += 1

                if done:
                    # capture scenario before reset
                    ep_rewards.append(ep_reward)
                    ep_action_traces.append(list(current_episode_actions))
                    ep_scenarios.append(int(self.env.scenario))
                    current_episode_actions = []
                    ep_reward = 0.0
                    state, _ = self.env.reset()

            # compute last value for bootstrapping
            with torch.no_grad():
                last_state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                _, last_value = self.policy(last_state_t)
                next_value = last_value.item()

            # Gather arrays
            rewards = np.array(self.buffer.rewards, dtype=np.float32)
            values = np.array(self.buffer.values, dtype=np.float32)
            dones = np.array(self.buffer.dones, dtype=np.float32)

            # --- Diagnostics: action distribution, average logprob/value ---
            if len(self.buffer.actions) > 0:
                action_counts = np.bincount(np.array(self.buffer.actions, dtype=np.int64), minlength=self.policy.action_dim)
                avg_logp = float(np.mean(self.buffer.log_probs))
                avg_value = float(np.mean(self.buffer.values))
            else:
                action_counts = np.zeros(self.policy.action_dim, dtype=int)
                avg_logp = 0.0
                avg_value = 0.0

            # Save a small sample of episode traces to a file for inspection and print to console
            os.makedirs("logs", exist_ok=True)
            sample_path = os.path.join("logs", f"actions_update_{update+1}.txt")
            output_lines = []
            output_lines.append(f"=== Update {update+1} action traces ===")
            output_lines.append(f"Collected steps: {len(self.buffer.actions)}")
            output_lines.append(f"Action counts: {action_counts.tolist()}")
            output_lines.append(f"Avg log_prob: {avg_logp:.4f}, Avg value: {avg_value:.4f}\n")
            for i, (rew, trace, scen) in enumerate(zip(ep_rewards, ep_action_traces, ep_scenarios)):
                if i >= 50:
                    break
                output_lines.append(f"EP{i+1}: Scenario={SCENARIO_NAMES.get(scen,'?')}, Reward={rew:.3f}, Actions={trace}")

            # Print to console
            print("\n".join(output_lines))

            # Also persist to file for later inspection
            try:
                with open(sample_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(output_lines))
            except Exception:
                pass

            advantages, returns = self._compute_gae(rewards, values, dones, next_value)
            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # PPO update
            self.ppo_update(advantages, returns)

            # Logging
            if len(ep_rewards) > 0:
                total_reward = float(np.mean(ep_rewards))
            else:
                total_reward = 0.0
            self.episode_rewards.append(total_reward)
            window_start = max(0, len(self.episode_rewards) - MOVING_AVG_WINDOW)
            moving_avg_reward = float(np.mean(self.episode_rewards[window_start:]))
            self.moving_avg_rewards.append(moving_avg_reward)

            if moving_avg_reward > self.best_moving_avg:
                self.best_moving_avg = moving_avg_reward
                best_model_path = os.path.join(self.models_dir, "model_canary_ppo_tcn_best.pth")
                torch.save(self.policy.state_dict(), best_model_path)

            update += 1
            if update % 1 == 0:
                scenario_name = SCENARIO_NAMES.get(self.env.scenario, "Unknown")
                print(
                    f"[Update {update}/{TRAIN_UPDATES}] MeanEpReward={total_reward:.3f} MA{MOVING_AVG_WINDOW}={moving_avg_reward:.3f} "
                    f"Scenario={scenario_name} | Actions={action_counts.tolist()} | AvgLogP={avg_logp:.4f} | AvgV={avg_value:.4f} | TracesSaved={sample_path}"
                )

        # final save
        torch.save(self.policy.state_dict(), os.path.join(self.models_dir, "model_canary_ppo_tcn_final.pth"))
        self.plot_training_metrics("logs/training_metrics.png")
 
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
        plt.plot(self.moving_avg_rewards, color='green', label="MovingAvg")
        plt.title("Moving Average Reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"[PLOT] Đã lưu biểu đồ tại: {output_path}")
 
 
if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    trainer.train()