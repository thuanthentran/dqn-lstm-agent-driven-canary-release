"""RL Agent Inference Runner — Live cluster deployment.

Runs the trained TransformerPPO agent in inference mode on the live K3s cluster:
  observe (Prometheus) → predict (model) → execute (Argo Rollouts)

Produces an action_log.json with the same schema as RuleBasedController,
enabling direct apples-to-apples comparison in analyze_comparison.py.

Usage:
    python scripts/run_rl_agent_inference.py \
        --model models/ppo_transformer_offline_best.zip \
        --norm  models/vec_normalize.pkl \
        --out   result_1/S1_high_latency-rl_agent-01/action_log.json \
        [--config configs/rule_based_thresholds.yaml]
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import numpy as np
import requests
import subprocess
import torch
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Reuse Prometheus + Argo clients from rule_based_controller
from core.rule_based_controller import PrometheusClient, ArgoRolloutsClient, ACTION_NAMES

# RL agent constants (must match training)
SEQ_LEN = 30
NUM_FEATURES = 5  # cpu_n, mem_n, l_ratio_n, e_ratio_n, weight_n

FEATURE_NAMES = ["cpu_n", "mem_n", "l_ratio_n", "e_ratio_n", "weight_n"]

# Normalization constants from core/feature_pipeline.py (linkerd branch)
CPU_REF = 0.02
MEM_REF_MB = 128.0
RPS_REF = 50.0
MAX_RATIO = 5.0
EPSILON = 1e-6


def _normalize_metrics(raw: dict) -> np.ndarray:
    """Convert raw Prometheus metrics to normalized 5-feature state vector.

    Mirrors the normalization in core/feature_pipeline.py (linkerd branch, 5 features).
    Features: [cpu_n, mem_n, l_ratio_n, e_ratio_n, weight_n]
    """
    def clip(v, lo=0.0, hi=1.0):
        return float(max(lo, min(hi, v)))

    e_canary = max(0.0, raw.get("canary_error_rate", 0.0))
    e_stable = max(0.0, raw.get("stable_error_rate", 0.0))
    l_canary = max(0.0, raw.get("canary_p95_latency_ms", 0.0))
    l_stable = max(0.0, raw.get("stable_p95_latency_ms", 40.0))  # floor 40ms
    cpu = max(0.0, raw.get("cpu_cores", 0.0))
    mem = max(0.0, raw.get("mem_mb", 0.0))
    weight_pct = max(0.0, raw.get("traffic_weight_canary", 0.0)) * 100.0

    e_ratio = e_canary / max(e_stable, 0.001)
    l_ratio = l_canary / max(l_stable, 40.0)

    # cpu_n: cpu_ratio normalized — use cpu directly vs CPU_REF
    cpu_n = clip(cpu / (CPU_REF * MAX_RATIO))
    # mem_n: mem_ratio normalized — use mem directly vs MEM_REF_MB
    mem_n = clip(mem / (MEM_REF_MB * MAX_RATIO))
    l_ratio_n = clip(l_ratio / MAX_RATIO)
    e_ratio_n = clip(e_ratio / MAX_RATIO)
    weight_n = clip(weight_pct / 100.0)

    return np.array([cpu_n, mem_n, l_ratio_n, e_ratio_n, weight_n], dtype=np.float32)


@dataclass
class ActionRecord:
    """Single decision record — matches RuleBasedController schema."""
    timestamp: str
    step: int
    action: int
    action_name: str
    method: str  # always "rl_agent"
    rationale: str
    metrics: dict
    consecutive_failures: int = 0
    consecutive_healthy: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class RLAgentRunner:
    """Runs the trained RL agent inference loop on the live cluster.

    Args:
        model_path:     Path to .zip model file.
        norm_path:      Path to vec_normalize.pkl.
        prometheus:     PrometheusClient instance.
        argo:           ArgoRolloutsClient instance.
        config:         Parsed YAML config (uses target + prometheus sections).
        verbose:        Print step info.
    """

    def __init__(
        self,
        model_path: str,
        norm_path: str,
        prometheus: PrometheusClient,
        argo: ArgoRolloutsClient,
        config: dict,
        verbose: bool = True,
    ):
        self.prom = prometheus
        self.argo = argo
        self.verbose = verbose
        self.config = config
        self.action_log: list[ActionRecord] = []
        self.done = False

        # Observation history buffer (rolling window of SEQ_LEN steps)
        self.history: deque = deque(maxlen=SEQ_LEN)

        # Additional Prometheus queries for CPU + RAM
        svc = config["target"]["service_name"]
        ns = config["target"]["namespace"]
        win = config["prometheus"]["scrape_window"]
        self._q_cpu = (
            f'sum(rate(container_cpu_usage_seconds_total{{'
            f'namespace="{ns}", pod=~"{svc}.*", container!="POD", container!=""}}[{win}]))'
        )
        self._q_mem = (
            f'sum(container_memory_working_set_bytes{{'
            f'namespace="{ns}", pod=~"{svc}.*", container!="POD", container!=""}})'
        )

        svc_w = config["target"]["service_name"]
        win_w = config["prometheus"]["scrape_window"]
        self._q_canary_err = (
            f'sum(rate(istio_requests_total{{reporter="destination", destination_service=~"{svc_w}-canary.*",'
            f'grpc_response_status!="0"}}[{win_w}])) / '
            f'sum(rate(istio_requests_total{{reporter="destination", destination_service=~"{svc_w}-canary.*"}}[{win_w}]))'
        )
        self._q_stable_err = (
            f'sum(rate(istio_requests_total{{reporter="destination", destination_service=~"{svc_w}-stable.*",'
            f'grpc_response_status!="0"}}[{win_w}])) / '
            f'sum(rate(istio_requests_total{{reporter="destination", destination_service=~"{svc_w}-stable.*"}}[{win_w}]))'
        )
        self._q_canary_lat = (
            f'histogram_quantile(0.95, sum(rate('
            f'istio_request_duration_milliseconds_bucket{{'
            f'reporter="destination", destination_service=~"{svc_w}-canary.*"}}[{win_w}])) by (le))'
        )
        self._q_stable_lat = (
            f'histogram_quantile(0.95, sum(rate('
            f'istio_request_duration_milliseconds_bucket{{'
            f'reporter="destination", destination_service=~"{svc_w}-stable.*"}}[{win_w}])) by (le))'
        )
        self._q_weight = (
            f'sum(rate(istio_requests_total{{reporter="destination", destination_service=~"{svc_w}-canary.*"}}[{win_w}])) / '
            f'sum(rate(istio_requests_total{{reporter="destination", destination_service=~"{svc_w}-(canary|stable).*"}}[{win_w}]))'
        )

        # Load model
        print(f"  [RL] Loading model from: {model_path}")
        self.model = PPO.load(model_path, device="cpu")

        # Pre-fill history with zeros (cold start)
        zero_obs = np.zeros(NUM_FEATURES, dtype=np.float32)
        for _ in range(SEQ_LEN):
            self.history.append(zero_obs)

        print(f"  [RL] Model loaded. Observation: ({SEQ_LEN}, {NUM_FEATURES})")

    def _get_obs(self) -> np.ndarray:
        """Stack history into (SEQ_LEN, NUM_FEATURES) observation."""
        return np.stack(list(self.history), axis=0).astype(np.float32)

    def observe(self) -> dict:
        """Query Prometheus for all needed metrics."""
        canary_err = self.prom.query(self._q_canary_err) or 0.0
        stable_err = self.prom.query(self._q_stable_err) or 0.0
        canary_lat = self.prom.query(self._q_canary_lat) or 0.0
        stable_lat = self.prom.query(self._q_stable_lat) or 0.0
        weight = self.prom.query(self._q_weight) or 0.0
        cpu = self.prom.query(self._q_cpu) or 0.0
        mem_bytes = self.prom.query(self._q_mem) or 0.0
        mem_mb = mem_bytes / (1024 * 1024)

        return {
            "canary_error_rate": canary_err,
            "stable_error_rate": stable_err,
            "canary_p95_latency_ms": canary_lat,
            "stable_p95_latency_ms": stable_lat,
            "traffic_weight_canary": weight,
            "cpu_cores": cpu,
            "mem_mb": mem_mb,
        }

    def run_loop(self, check_interval: float = 15.0, max_steps: int = 50) -> list[dict]:
        """RL agent inference loop — observe → predict → execute.

        Returns:
            action_log as list of dicts.
        """
        print(f"\n{'='*60}")
        print(f"  RL Agent Inference Loop starting")
        print(f"  max_steps={max_steps}, interval={check_interval}s")
        print(f"{'='*60}")

        for step in range(1, max_steps + 1):
            t0 = time.time()

            raw = self.observe()
            feature_vec = _normalize_metrics(raw)
            self.history.append(feature_vec)
            obs = self._get_obs()  # (SEQ_LEN, NUM_FEATURES)

            # Model expects batch dim: (1, SEQ_LEN, NUM_FEATURES)
            obs_tensor = obs[np.newaxis, ...]
            action_arr, _ = self.model.predict(obs_tensor, deterministic=True)
            action = int(action_arr[0])

            rationale = (
                f"RL prediction — features: "
                f"cpu_n={feature_vec[0]:.3f}, mem_n={feature_vec[1]:.3f}, "
                f"l_ratio_n={feature_vec[2]:.3f}, e_ratio_n={feature_vec[3]:.3f}, "
                f"weight_n={feature_vec[4]:.3f}"
            )

            if self.verbose:
                action_colors = {0: "\033[93m", 1: "\033[92m", 2: "\033[91m"}
                color = action_colors.get(action, "")
                reset = "\033[0m"
                print(
                    f"\n[Step {step:02d}] {color}{ACTION_NAMES[action]}{reset}\n"
                    f"  err: canary={raw['canary_error_rate']:.4f} stable={raw['stable_error_rate']:.4f}\n"
                    f"  lat: canary={raw['canary_p95_latency_ms']:.1f}ms stable={raw['stable_p95_latency_ms']:.1f}ms\n"
                    f"  weight={raw['traffic_weight_canary']:.3f} cpu={raw['cpu_cores']:.4f} mem={raw['mem_mb']:.1f}MB"
                )

            # Execute
            if action == 1:  # Promote
                self.argo.promote()
            elif action == 2:  # Rollback
                self.argo.abort()
                self.done = True

            record = ActionRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                step=step,
                action=action,
                action_name=ACTION_NAMES[action],
                method="rl_agent",
                rationale=rationale,
                metrics={**raw, "features": feature_vec.tolist()},
            )
            self.action_log.append(record)

            if self.done:
                print(f"\n  [RL] Rollback executed → episode complete.")
                break

            if raw.get("traffic_weight_canary", 0.0) >= 0.99:
                print(f"\n  [RL] Canary reached 100% → episode complete.")
                self.done = True
                break

            elapsed = time.time() - t0
            sleep_time = max(0.0, check_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if not self.done:
            print(f"\n  [RL] Max steps ({max_steps}) reached → timeout.")

        return [r.to_dict() for r in self.action_log]

    def export_action_log(self, path: str) -> None:
        """Save action log to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        log = [r.to_dict() for r in self.action_log]
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"  [RL] Action log saved → {path} ({len(log)} records)")

    @classmethod
    def from_config(
        cls,
        model_path: str,
        norm_path: str,
        config_path: str = "configs/rule_based_thresholds.yaml",
        verbose: bool = True,
    ) -> "RLAgentRunner":
        """Instantiate from model files + shared YAML config."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        prom_cfg = config["prometheus"]
        tgt_cfg = config["target"]

        prometheus = PrometheusClient(
            url=prom_cfg["url"],
            scrape_window=prom_cfg["scrape_window"],
            timeout=prom_cfg["timeout_s"],
        )
        argo = ArgoRolloutsClient(
            service_name=tgt_cfg["service_name"],
            namespace=tgt_cfg["namespace"],
        )
        return cls(
            model_path=model_path,
            norm_path=norm_path,
            prometheus=prometheus,
            argo=argo,
            config=config,
            verbose=verbose,
        )


# ─── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL agent inference on live cluster")
    parser.add_argument("--model", default="models/ppo_transformer_offline_best.zip")
    parser.add_argument("--norm", default="models/vec_normalize.pkl")
    parser.add_argument("--config", default="configs/rule_based_thresholds.yaml")
    parser.add_argument("--out", required=True, help="Path to save action_log.json")
    parser.add_argument("--interval", type=float, default=15.0, help="Check interval in seconds")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    runner = RLAgentRunner.from_config(
        model_path=args.model,
        norm_path=args.norm,
        config_path=args.config,
        verbose=not args.quiet,
    )
    runner.run_loop(check_interval=args.interval, max_steps=args.max_steps)
    runner.export_action_log(args.out)
