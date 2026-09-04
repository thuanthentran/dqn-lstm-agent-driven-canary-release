# K8s RL Canary Agent - Antigravity Agent Guidelines

This `GEMINI.md` file serves as the main rule and context file for the AI agent working on the `dqn-lstm-agent-driven-canary-release` project.

## 1. Project Context
- **Objective**: Develop and maintain a Reinforcement Learning (RL) agent (using PPO/DQN+LSTM) to manage K8s Canary releases.
- **Key Technologies**: Kubernetes, Argo Rollouts, Cilium Gateway API, Grafana Beyla (eBPF), Prometheus, Stable Baselines 3, PyTorch.
- **Core Architecture**: The agent relies on zero-instrumentation metrics from eBPF and manipulates traffic via Gateway API based on RL policies.

## 2. Code Organization & Guidelines
- **`core/`**: Contains the core RL environment (`env.py`), feature pipelines, and model-facing logic. Maintain a strict separation between K8s interaction and pure RL logic.
- **`training/`**: Contains scripts for training the model (`online_training.py`, `offline_training.py`, `sweep_optuna.py`).
- **`models/`**: Destination for trained model artifacts (`.pt` or `.zip`).
- **`logs/` & `mlruns/`**: Use MLflow for all experiment tracking, saving metrics, rewards, and parameters. Do not commit large database files (`mlflow.db`).

## 3. RL Training & Validation Rules
- **Validation**: Any trained model must be validated across predefined scenarios (S0 to S4). Key metrics to track:
  - **FPR (False Positive Rate)**: Rejecting a healthy deployment. Target < 10%.
  - **FNR (False Negative Rate)**: Promoting a broken deployment. Target < 5%.
  - **Latency**: Number of steps taken before deciding. Lower is better, but accuracy is paramount.
- **Experiment Tracking**: Always log Hyperparameters, Reward curves, and final FPR/FNR metrics to MLflow.

## 4. Development Style
- Use standard Python typing (Type Hints) for all functions.
- Keep the `walkthrough.md` updated with the latest progress.
- Adhere to PEP 8 standards. Include docstrings for any new core environment functions or metrics extraction logic.
- Ensure any modifications to Kubernetes manifests (`gitops/`) match the Argo Rollouts and Cilium Gateway API specifications.
