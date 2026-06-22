import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import mlflow
from sb3_contrib import RecurrentPPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from core.env import CanaryEnv

TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "Canary_RL_Offline_Phase"
LOG_DIR = "logs/lstm_offline"
SAVE_PATH = "models/ppo_lstm_offline_best"
TOTAL_TIMESTEPS = 150000


def configure_mlflow() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)


def make_env(log_dir: str):
    def _init():
        return Monitor(CanaryEnv(), log_dir)

    return _init


def build_env(log_dir: str):
    return DummyVecEnv([make_env(log_dir)])


def build_model(vec_env):
    return RecurrentPPO(
        "MlpLstmPolicy",
        vec_env,
        verbose=1,
        policy_kwargs={
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
            "lstm_hidden_size": 64,
            "n_lstm_layers": 1,
        },
    )


def train() -> None:
    configure_mlflow()
    os.makedirs(LOG_DIR, exist_ok=True)

    vec_env = build_env(LOG_DIR)
    model = build_model(vec_env)

    print("🚀 Bắt đầu huấn luyện PPO+LSTM...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS)
    model.save(SAVE_PATH)

    print("📈 Đang vẽ Learning Curve...")
    results_plotter.plot_results(
        [LOG_DIR],
        TOTAL_TIMESTEPS,
        results_plotter.X_TIMESTEPS,
        "PPO+LSTM Canary Release",
    )

    plot_path = os.path.join(LOG_DIR, "learning_curve.png")
    plt.savefig(plot_path)
    plt.close()

    with mlflow.start_run(run_name="Offline_LSTM_Training_V2"):
        mlflow.log_artifact(f"{SAVE_PATH}.zip", artifact_path="model_weights")
        mlflow.log_artifact(plot_path, artifact_path="plots")
        mlflow.log_param("total_timesteps", TOTAL_TIMESTEPS)
        mlflow.log_param("architecture", "PPO+LSTM")

        print("✅ Đã đẩy model và biểu đồ lên MLFlow thành công!")


if __name__ == "__main__":
    train()