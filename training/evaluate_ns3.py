"""Evaluate TransformerPPO model with attention heatmap visualization on ns-3 traces.

Runs deterministic episodes, extracts attention maps from the Transformer
feature extractor, and generates heatmaps showing:
  - Feature Attention: which features the agent focuses on per timestep
  - Temporal Attention: which past timesteps the agent attends to
"""

import os
import sys

# Fix Unicode output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

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
from core.env_ns3 import CanaryEnvNs3
from core.env import ACTION_NAMES, SCENARIO_NAMES, NETWORK_SCENARIO_NAMES

MODEL_PATH = os.path.join(BASE_DIR, "models", "ppo_transformer_offline_best.zip")
NORM_PATH = os.path.join(BASE_DIR, "models", "vec_normalize.pkl")
HEATMAP_DIR = os.path.join(BASE_DIR, "logs", "attention_heatmaps_ns3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_NAMES = [
    "CPU", "RAM", "Latency", "Error_Rate", "Traffic",
    "Handover", "SINR", "RSRP", "PRB_Util", "HARQ_NACK", "NTN_Gap",
    "ISAC", "Pkt_Loss", "Jitter", "Deploy_Age",
]
NUM_APP_SCENARIOS = 5
NUM_NET_SCENARIOS = 5
NUM_EVAL_EPISODES = NUM_APP_SCENARIOS * NUM_NET_SCENARIOS

def plot_feature_attention_heatmap(feature_attn, episode_idx, step_idx, save_dir):
    """Plot feature attention weights as a heatmap."""
    attn_avg = feature_attn.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attn_avg, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_xlabel("Feature")
    ax.set_ylabel("Timestep")
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=45, ha="right")
    ax.set_title(f"Feature Attention — Episode {episode_idx + 1}, Step {step_idx + 1} (ns-3 Trace)")
    plt.colorbar(im, ax=ax, label="Attention Weight")

    plt.tight_layout()
    path = os.path.join(save_dir, f"feat_attn_ep{episode_idx + 1}_step{step_idx + 1}.png")
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def plot_temporal_attention_heatmap(temp_attn, episode_idx, step_idx, save_dir):
    """Plot temporal self-attention weights as a heatmap."""
    attn_avg = temp_attn.mean(axis=0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(attn_avg, aspect="auto", cmap="Blues", interpolation="nearest")

    ax.set_xlabel("Key Timestep (attended to)")
    ax.set_ylabel("Query Timestep (attending from)")
    ax.set_title(f"Temporal Attention — Episode {episode_idx + 1}, Step {step_idx + 1} (ns-3 Trace)")
    plt.colorbar(im, ax=ax, label="Attention Weight")

    plt.tight_layout()
    path = os.path.join(save_dir, f"temp_attn_ep{episode_idx + 1}_step{step_idx + 1}.png")
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def analyze_attention(feature_attn, step_idx):
    """Print human-readable attention analysis."""
    last_step_attn = feature_attn.mean(axis=0)[-1]
    top_indices = np.argsort(last_step_attn)[::-1]

    parts = []
    for idx in top_indices[:3]:
        parts.append(f"{FEATURE_NAMES[idx]}={last_step_attn[idx]:.3f}")
    top_str = ", ".join(parts)

    print(f"   🧠 [Attention] Top features tại timestep cuối: {top_str}")

    entropy = -np.sum(last_step_attn * np.log(last_step_attn + 1e-8))
    max_entropy = np.log(len(FEATURE_NAMES))
    if entropy > 0.95 * max_entropy:
        print(f"   ⚠️  Attention gần uniform (entropy={entropy:.3f}/{max_entropy:.3f})")


def evaluate():
    print("=" * 60)
    print("🧠 ĐÁNH GIÁ TRANSFORMERPPO + NS-3 TRACES")
    print("=" * 60)

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Không tìm thấy model tại {MODEL_PATH}")
        return
    if not os.path.exists(NORM_PATH):
        print(f"❌ Không tìm thấy VecNormalize tại {NORM_PATH}")
        return

    os.makedirs(HEATMAP_DIR, exist_ok=True)

    # Use the ns-3 environment
    env = DummyVecEnv([lambda: CanaryEnvNs3()])
    env = VecNormalize.load(NORM_PATH, env)
    env.training = False
    env.norm_reward = False

    print(f"✅ Đang nạp TransformerPPO từ: {MODEL_PATH}")
    print(f"   Device: {DEVICE}")
    model = PPO.load(MODEL_PATH, env=env, device=DEVICE)
    extractor = model.policy.features_extractor

    all_rewards = []

    ep = 0
    for app_id in range(NUM_APP_SCENARIOS):
        for net_id in range(NUM_NET_SCENARIOS):
            ep += 1
            inner_env = env.venv.envs[0]
            inner_env.force_scenario = app_id
            inner_env.force_network_scenario = net_id
            
            app_name = SCENARIO_NAMES.get(app_id, "Unknown")
            net_name = NETWORK_SCENARIO_NAMES.get(net_id, "Unknown")

            print(f"\n{'─' * 50}")
            print(f"🎬 TẬP {ep}/{NUM_EVAL_EPISODES} — App: {app_name} | Net: {net_name}")
            print("─" * 50)

            obs = env.reset()
            done = False
            step = 0
            total_rew = 0
            
            last_fa = None
            last_ta = None

            while not done:
                step += 1
                action, _ = model.predict(obs, deterministic=True)
                act_name = ACTION_NAMES.get(int(action[0]), "UNKNOWN")

                attn_maps = extractor.get_attention_maps()

                print(f"\n[Bước {step}]")
                print(f"   🤖 Hành động: {act_name}")

                if attn_maps["feature_attention"] is not None:
                    fa = attn_maps["feature_attention"][0]
                    analyze_attention(fa, step)
                    last_fa = fa
                    if attn_maps["temporal_attention"] is not None:
                        last_ta = attn_maps["temporal_attention"][0]

                obs, reward, done, info = env.step(action)
                print(f"   ⚖️ Reward = {reward[0]:.2f} | Done = {done[0]}")
                total_rew += reward[0]

            # Save heatmaps for the final step of this episode
            if last_fa is not None:
                plot_feature_attention_heatmap(last_fa, ep - 1, step - 1, HEATMAP_DIR)
            if last_ta is not None:
                plot_temporal_attention_heatmap(last_ta, ep - 1, step - 1, HEATMAP_DIR)

            all_rewards.append(total_rew)
            print(f"\n🏁 Kết thúc tập {ep}. Tổng điểm: {total_rew:.2f}")

    mean_rew = np.mean(all_rewards)
    std_rew = np.std(all_rewards)
    print(f"\n{'=' * 60}")
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ NS-3 ({NUM_EVAL_EPISODES} tập)")
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
