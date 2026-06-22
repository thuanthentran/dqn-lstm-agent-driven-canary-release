# DQN_LSTM_Agent_for_Canary_Release

This repository is organized as a small monorepo for canary release experiments on Kubernetes, with separate areas for ML logic, deployment manifests, microservices examples, and training assets.

## Layout

- `core/` contains the environment, feature pipeline, and model-facing logic.
- `deploy/` contains runtime entrypoints and container build material.
- `gitops/` contains Argo CD applications and Kubernetes manifests.
- `microservices-demo/` contains the supporting demo services and its Istio/Kubernetes manifests.
- `models/` stores trained model artifacts.
- `training/` contains training scripts.

## Working Order

1. Keep deployment and GitOps changes isolated from model logic.
2. Treat `gitops/app/manifests/` as the rollout source of truth.
3. Update the root README when new top-level folders are added.
