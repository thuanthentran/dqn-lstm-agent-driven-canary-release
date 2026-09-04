# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A Reinforcement Learning agent that decides whether to **Promote / Hold / Rollback** a Kubernetes canary release. It has two halves that must stay in sync:

- **Offline plane** — a synthetic Gym environment ([core/env.py](core/env.py)) + PPO training ([training/](training/)). No cluster required; runs on Windows/laptop.
- **Serving plane** — a FastAPI webhook ([services/agent/main.py](services/agent/main.py)) that Argo Rollouts calls mid-rollout via an `AnalysisTemplate`, scraping Prometheus for the same features the simulator produced.

Docs, comments and console output are predominantly **Vietnamese**; match that when editing existing files.

## Commands

Run everything from the repo root (scripts do `sys.path.append(BASE_DIR)` and import `core.*` as a package).

```bash
# Full offline pipeline: 10-trial Optuna sweep -> 150k-step PPO train -> validate -> learning curve
python -m training.offline_training

# Hyperparameter sweep alone (multi-objective: reward, latency, FPR, FNR) + rule-based baseline
python -m training.sweep_optuna

# Acceptance test: 5 fixed episode configs (A-E) + RL vs rule-based comparison table
python acceptance_test_phase9.py

# Training curves
tensorboard --logdir logs/transformer_offline

# Serving webhook locally (needs a reachable PROMETHEUS_URL)
uvicorn services.agent.main:app --host 0.0.0.0 --port 8000

# Agent image — build context is the REPO ROOT, not services/agent (Dockerfile copies core/ and models/)
docker build -f services/agent/Dockerfile -t agent .
```

Cluster-dependent (needs kubeconfig, Argo Rollouts, Locust, a live Prometheus):
`python -m training.online_training` (fine-tunes the offline model against [core/online_env.py](core/online_env.py)), `python training/evaluate.py`, `python training/debug_prometheus.py`.

**There is no pytest suite in this repo.** `tests/` and `scripts/` contain only stale `__pycache__`. Verification is `acceptance_test_phase9.py` plus `validate_model_locally()` in [training/offline_training.py](training/offline_training.py). After editing an env/pipeline file, the repo convention is a fast syntax check first: `python -c "import core.env"`.

Note `training.offline_training.train()` always runs the Optuna sweep before training (~10 trials × 30k steps). For a plain retrain, call `build_env` / `build_model` / `model.learn` directly instead of `train()`.

## The feature contract (most important thing to know)

[core/feature_pipeline.py](core/feature_pipeline.py) `normalize_raw_metrics()` is the **only** code shared between simulation and production. Both planes build the same raw dict (`e_canary`, `e_stable`, `l_canary`, `l_stable`, `cpu_canary`, `cpu_stable`, `mem_canary_mb`, `mem_stable_mb`, `weight_pct`, `rps`), normalize it, then pick **5 channels in this fixed order**: `[cpu_ratio_n, mem_ratio_n, l_ratio_n, e_ratio_n, weight_n]` over a `SEQ_LEN=30` window.

Changing a normalized key or the channel order silently breaks inference — `_raw_to_channels()` uses `norm.get(key, 0.0)`, so a renamed key yields an all-zero channel with no error. Any such change must land in all four places at once:

1. [core/feature_pipeline.py:58-69](core/feature_pipeline.py#L58-L69) — the `state` dict
2. [core/env.py:264-277](core/env.py#L264-L277) — `_raw_to_channels()` (simulation)
3. [services/agent/main.py:163-172](services/agent/main.py#L163-L172) — the production channel build
4. `TRANSFORMER_CONFIG["n_features"]` in [training/offline_training.py:33-41](training/offline_training.py#L33-L41), which reads `CanaryEnv.num_features`

**Ratio, not absolute.** All four health channels are `canary / max(stable, EPSILON)` clipped by `MAX_RATIO=5.0` — so "canary equals stable" normalizes to `0.2`, and the healthy/anomalous thresholds in `step()` (`<= 0.4` healthy, raw ratio `> 2.0` anomalous) are calibrated against that. Never floor a denominator with a "reasonable baseline" constant (e.g. `0.04`, `12.0`); use `EPSILON` only. CPU/RAM were migrated off fixed reference constants precisely because absolute thresholds made the agent blind to CPU/memory regressions — `CPU_REF`/`MEM_REF_MB` are gone from the pipeline for that reason.

**Tensor layout differs between planes.** Offline `CanaryEnv` emits `(SEQ_LEN, 5)` = time-major, which is what `TransformerFeatureExtractor` expects `(B, T, C)`. `OnlineCanaryEnv` and the serving webhook build channel-major `(5, SEQ_LEN)`. Check which layout a model was trained with before wiring a new one in.

## Simulation design ([core/env.py](core/env.py))

Episodes are **domain-randomized per channel**, not chosen from a fixed scenario list:

- Two-stage sampling: ~35% of episodes are all-healthy; otherwise each of `cpu / mem / latency / error` independently becomes anomalous (~55%) and gets a pattern from `leak | threshold_spike | load_dependent`.
- Per-episode baselines are log-uniform 0.2×–5× around `BASELINE_CENTERS`, so **every anomaly offset must be multiplicative on the baseline**, never an additive constant — and noise is relative (`relative_noise`) to keep SNR stable across baseline scales.
- `self.scenario` (0=Healthy, 1=Mixed) is a **coarse logging label only**. It must not feed observations or reward. Fixed scenarios are for eval only (`reset(episode_config=...)`, see [acceptance_test_phase9.py](acceptance_test_phase9.py)).

Reward shape: correct Promote `+3` and correct Rollback `+5` both carry a symmetric earliness bonus scaled by `EARLY_BONUS_SCALE` (env var, swept by Optuna); a false-positive Rollback on a healthy canary costs `-22`. The terminal check at `weight >= 1.0` gates on all four channels — gating on only error/latency was the bug that let the agent ignore CPU/RAM.

Success targets used throughout: **FPR < 10%** (rolling back a healthy release), **FNR < 5%** (promoting a broken one), mean reward > 5.

## Serving plane ([services/agent/main.py](services/agent/main.py))

Argo Rollouts posts `{service, stable_hash, canary_hash, target_weight, namespace}`; the response is `{"action": N}` where the `AnalysisTemplate` maps **1 = Promote (success), 2 = Rollback (fail), 0 = Hold/Running (inconclusive)** — see [gitops/base/universal-analysis-template.yaml](gitops/base/universal-analysis-template.yaml). Adding a payload field requires updating the `WebhookPayload` model, the template's `args` + `jsonBody`, and every `analysis` step in [gitops/charts/universal-canary/templates/rollout.yaml](gitops/charts/universal-canary/templates/rollout.yaml) (weights 20/50/80).

Two guards sit in front of and behind the model: a data-completeness check (canary weight > 0 but zero canary RPS ⇒ immediate Rollback; incomplete series ⇒ Hold), and `_evaluate_safety_guard()`, which can force Rollback on severe error/latency breaches or downgrade a model Promote to Hold. All guard thresholds are env vars (`SAFETY_*`).

PromQL currently queries **`istio_*` series** plus cAdvisor, while the README and rollout manifests describe a Cilium Gateway API + Grafana Beyla (eBPF) stack. Confirm which metric source the target cluster actually exposes before trusting or editing these queries.

The default `MODEL_PATH` is `models/ppo_lstm_offline_best.zip` loaded via `sb3_contrib.RecurrentPPO`, but the current offline pipeline trains a Transformer-PPO to `models/ppo_transformer_offline_best.zip` loaded via `stable_baselines3.PPO`. The two are not interchangeable — a `MODEL_PATH` change needs the matching loader class.

## Deployment layout

- [gitops/charts/universal-canary/](gitops/charts/universal-canary/) — one Helm chart templating Rollout + Services + Gateway + HTTPRoute for every microservice; per-service overrides live in [gitops/releases/](gitops/releases/) (`<service>-values.yaml`). Add a service by adding a values file, not by forking the chart.
- The agent itself is deployed through a **CRD + Kopf operator**: `RLAgent` ([gitops/base/agent-crd.yaml](gitops/base/agent-crd.yaml), instance in [gitops/base/agent-instance.yaml](gitops/base/agent-instance.yaml)) is reconciled by [services/controller/main.py](services/controller/main.py) into a Deployment + Service.
- [services/src/](services/src/) is the vendored Online Boutique microservice demo (the workload under test), not project code. `FAULT_SCENARIO` env vars in the release values inject faults into it.
- [.github/workflows/build.yaml](.github/workflows/build.yaml) builds changed services to GHCR — it triggers on pushes to the **`grafanabeyla`** branch under `services/**`, not `main`.

## Conventions from the repo's agent rules

From [GEMINI.md](GEMINI.md) and [.agents/skills/](.agents/skills/):

- Keep Kubernetes/GitOps changes isolated from ML code; treat manifests as the source of truth and never mutate live cluster state directly — `kubectl` is for read-only inspection.
- Type-hint new functions and docstring anything touching the environment or metric extraction.
- Model artifacts go in `models/`, training scripts in `training/`, RL logic in `core/`.
- The rules ask for MLflow experiment tracking (`mlflow.db`, `mlruns/`), but no Python code currently imports mlflow — training logs to TensorBoard and `logs/*/monitor.csv` instead. Don't assume MLflow is wired up.
- Keep [walkthrough.md](walkthrough.md) updated with notable progress.

`task_list.md` / `implementation_plan.md` are phase-by-phase working docs for the CPU/RAM channel migration and explain the reasoning behind several constants above; line numbers cited inside them are stale — locate code by function name or distinctive text.
