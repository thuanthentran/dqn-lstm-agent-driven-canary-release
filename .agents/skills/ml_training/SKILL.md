---
name: ml_training
description: Guidelines for training, tracking, and managing the DQN LSTM RL model.
---

# ML Training Guidelines

1. **Experiment Tracking**: Always use MLflow to track parameters, metrics, and models. The local tracking database is `mlflow.db` and artifacts are in `mlruns/`.
2. **Model Artifacts**: Store trained model artifacts (e.g., `.pt` or `.pth` files) in the `models/` directory.
3. **Training Scripts**: Keep all training-related scripts in the `training/` directory.
4. **Core Environment**: The RL environment, feature pipeline, and model-facing logic should reside in the `core/` directory. Maintain clean separation between the online environment and training logic.
5. **Testing**: Validate the RL agent's behavior locally before deploying as a canary.
