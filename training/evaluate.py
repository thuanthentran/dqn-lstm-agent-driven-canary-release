"""Evaluate TransformerPPO model with attention heatmap visualization.

Runs deterministic episodes, extracts attention maps from the Transformer
feature extractor, and generates heatmaps showing:
  - Feature Attention: which features the agent focuses on per timestep
  - Temporal Attention: which past timesteps the agent attends to
"""

import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving plots

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from core.env import CanaryEnv, ACTION_NAMES, SCENARIO_NAMES

MODEL_PATH = os.path.join(BASE_DIR, "models", "ppo_transformer_offline_best.zip")
NORM_PATH = os.path.join(BASE_DIR, "models", "vec_normalize.pkl")
HEATMAP_DIR = os.path.join(BASE_DIR, "logs", "attention_heatmaps")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_NAMES = ["CPU", "RAM", "Latency", "Error_Rate", "Traffic"]
NUM_EVAL_EPISODES = 10


def plot_feature_attention_heatmap(feature_attn, episode_idx, step_idx, save_dir):
    """Plot feature attention weights as a heatmap.

    Args:
        feature_attn: np.ndarray (n_heads, T, n_features) — single sample
        episode_idx: episode number
        step_idx: step number within episode
        save_dir: directory to save plot
    """
    # Average over heads: (T, n_features)
    attn_avg = feature_attn.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_avg, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xlabel("Feature")
    ax.set_ylabel("Timestep")
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right")
    ax.set_title(f"Feature Attention — Episode {episode_idx + 1}, Step {step_idx + 1}")
    plt.colorbar(im, ax=ax, label="Attention Weight")

    plt.tight_layout()
    path = os.path.join(save_dir, f"feat_attn_ep{episode_idx + 1}_step{step_idx + 1}.png")
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def plot_temporal_attention_heatmap(temp_attn, episode_idx, step_idx, save_dir):
    """Plot temporal self-attention weights as a heatmap.

    Args:
        temp_attn: np.ndarray (n_heads, T, T) — single sample
        episode_idx: episode number
        step_idx: step number within episode
        save_dir: directory to save plot
    """
    # Average over heads: (T, T)
    attn_avg = temp_attn.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(attn_avg, aspect="auto", cmap="Blues", interpolation="nearest")

    ax.set_xlabel("Key Timestep (attended to)")
    ax.set_ylabel("Query Timestep (attending from)")
    ax.set_title(f"Temporal Attention — Episode {episode_idx + 1}, Step {step_idx + 1}")
    plt.colorbar(im, ax=ax, label="Attention Weight")

    plt.tight_layout()
    path = os.path.join(save_dir, f"temp_attn_ep{episode_idx + 1}_step{step_idx + 1}.png")
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def analyze_attention(feature_attn, step_idx):
    """Print human-readable attention analysis.

    Args:
        feature_attn: np.ndarray (n_heads, T, n_features)
    """
    # Average over heads, take the last timestep: (n_features,)
    last_step_attn = feature_attn.mean(axis=0)[-1]  # attention at final timestep
    top_indices = np.argsort(last_step_attn)[::-1]

    parts = []
    for idx in top_indices[:3]:
        parts.append(f"{FEATURE_NAMES[idx]}={last_step_attn[idx]:.3f}")
    top_str = ", ".join(parts)

    print(f"   🧠 [Attention] Top features tại timestep cuối: {top_str}")

    # Check for degenerate attention (uniform distribution)
    entropy = -np.sum(last_step_attn * np.log(last_step_attn + 1e-8))
    max_entropy = np.log(len(FEATURE_NAMES))
    if entropy > 0.95 * max_entropy:
        print(f"   ⚠️  Attention gần uniform (entropy={entropy:.3f}/{max_entropy:.3f})")


def evaluate():
    print("=" * 60)
    print("🧠 ĐÁNH GIÁ TRANSFORMERPPO + TRÍCH XUẤT ATTENTION MAPS")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy model tại {MODEL_PATH}")
        return
    if not os.path.exists(NORM_PATH):
        print(f"❌ Không tìm thấy VecNormalize tại {NORM_PATH}")
        return

    os.makedirs(HEATMAP_DIR, exist_ok=True)

    # Khởi tạo môi trường
    env = DummyVecEnv([lambda: CanaryEnv()])
    env = VecNormalize.load(NORM_PATH, env)
    env.training = False
    env.norm_reward = False

    # Nạp mô hình
    print(f"✅ Đang nạp TransformerPPO từ: {MODEL_PATH}")
    print(f"   Device: {DEVICE}")
    model = PPO.load(MODEL_PATH, env=env, device=DEVICE)

    # Trích xuất feature extractor
    extractor = model.policy.features_extractor

    all_rewards = []

    for ep in range(NUM_EVAL_EPISODES):
        # Force specific scenario for first 5 episodes
        inner_env = env.venv.envs[0]
        if ep < 5:
            inner_env.scenario = ep
            scenario_name = SCENARIO_NAMES.get(ep, "Unknown")
        else:
            inner_env.scenario = np.random.randint(0, 5)
            scenario_name = SCENARIO_NAMES.get(inner_env.scenario, "Unknown")

        print(f"\n{'─' * 50}")
        print(f"🎬 TẬP {ep + 1}/{NUM_EVAL_EPISODES} — Scenario: {scenario_name}")
        print("─" * 50)

        obs = env.reset()
        done = False
        step = 0
        total_rew = 0

        while not done:
            step += 1
            action, _ = model.predict(obs, deterministic=True)
            act_name = ACTION_NAMES.get(int(action[0]), "UNKNOWN")

            # Get attention maps from the forward pass that just happened
            attn_maps = extractor.get_attention_maps()

            print(f"\n[Bước {step}]")
            print(f"   🤖 Hành động: {act_name}")

            if attn_maps["feature_attention"] is not None:
                fa = attn_maps["feature_attention"][0]  # First (only) sample
                analyze_attention(fa, step)

                # Save heatmaps for first 2 steps of first 3 episodes
                if ep < 3 and step <= 2:
                    plot_feature_attention_heatmap(fa, ep, step - 1, HEATMAP_DIR)
                    if attn_maps["temporal_attention"] is not None:
                        ta = attn_maps["temporal_attention"][0]
                        plot_temporal_attention_heatmap(ta, ep, step - 1, HEATMAP_DIR)

            obs, reward, done, info = env.step(action)
            print(f"   ⚖️ Reward = {reward[0]:.2f} | Done = {done[0]}")
            total_rew += reward[0]

        all_rewards.append(total_rew)
        print(f"\n🏁 Kết thúc tập {ep + 1}. Tổng điểm: {total_rew:.2f}")

    # Summary
    mean_rew = np.mean(all_rewards)
    std_rew = np.std(all_rewards)
    print(f"\n{'=' * 60}")
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ ({NUM_EVAL_EPISODES} tập)")
    print(f"   Mean Reward: {mean_rew:.2f} ± {std_rew:.2f}")
    print(f"   Min: {np.min(all_rewards):.2f} | Max: {np.max(all_rewards):.2f}")
    print(f"   Heatmaps saved to: {HEATMAP_DIR}")

    if mean_rew >= 5.0:
        print(f"   ✅ PASSED (≥ 5.0)")
    else:
        print(f"   ⚠️ FAILED (< 5.0)")

    print("=" * 60)


if __name__ == "__main__":
    evaluate()
