"""Rule-based Canary Controller — Baseline for comparison with RL Agent.

Implements two threshold strategies:
  - 'static': Absolute thresholds (Argo Rollouts / SRE best practice).
              Rollback if error_rate > 1% or p95_latency > 500ms
              for N consecutive checks.
  - 'ratio':  Canary-vs-stable ratio thresholds (matches RL env anomaly
              criterion: e_ratio > 2.0 or l_ratio > 2.0 for N consecutive).

Both methods share the same control loop and action interface as the RL agent:
    observe() → decide() → execute()

Action space (mirrors RL env):
    0 = Hold    — do nothing this step
    1 = Promote — increase canary weight by promote_step
    2 = Rollback — abort canary deployment

Usage:
    from core.rule_based_controller import RuleBasedController
    ctrl = RuleBasedController.from_config('configs/rule_based_thresholds.yaml', method='static')
    ctrl.run_loop()
    ctrl.export_action_log('result_2/S1_high_latency-rule_based_static-01/action_log.json')
"""

import json
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

import requests
import yaml

# ─── Action constants (mirrors core/env.py) ──────────────────────────────────
HOLD = 0
PROMOTE = 1
ROLLBACK = 2
ACTION_NAMES = {HOLD: "Hold", PROMOTE: "Promote", ROLLBACK: "Rollback"}


@dataclass
class MetricsSnapshot:
    """Raw metrics queried from Prometheus at a single timestep."""
    timestamp: str
    step: int
    canary_error_rate: float
    stable_error_rate: float
    canary_p95_latency_ms: float
    stable_p95_latency_ms: float
    traffic_weight_canary: float
    # Derived ratios (computed from above)
    error_ratio: float = 0.0
    latency_ratio: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ActionRecord:
    """Single decision record written to action_log.json."""
    timestamp: str
    step: int
    action: int
    action_name: str
    method: str
    rationale: str
    metrics: dict
    consecutive_failures: int
    consecutive_healthy: int


class PrometheusClient:
    """Thin wrapper for Prometheus instant query API."""

    def __init__(self, url: str, scrape_window: str = "1m", timeout: float = 5.0):
        self.url = url.rstrip("/")
        self.window = scrape_window
        self.timeout = timeout

    def query(self, promql: str) -> float | None:
        """Execute instant query, return scalar float or None on failure."""
        try:
            resp = requests.get(
                f"{self.url}/api/v1/query",
                params={"query": promql},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "success":
                return None
            results = data.get("data", {}).get("result", [])
            if not results:
                return None
            value = results[0]["value"][1]
            return float(value) if value != "NaN" else None
        except Exception as e:
            print(f"  [Prometheus] Query failed: {e}")
            return None


class ArgoRolloutsClient:
    """Thin wrapper around kubectl / argo-rollouts CLI for canary actions."""

    def __init__(self, service_name: str, namespace: str):
        self.service = service_name
        self.ns = namespace

    def promote(self) -> bool:
        """Step-promote canary (increase weight by one step, as configured in Rollout)."""
        result = subprocess.run(
            ["kubectl", "argo", "rollouts", "promote", self.service, "-n", self.ns],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            # Fall back to direct argo-rollouts binary
            result = subprocess.run(
                ["argo-rollouts", "promote", self.service, "-n", self.ns],
                capture_output=True, text=True,
            )
        ok = result.returncode == 0
        if not ok:
            print(f"  [Argo] Promote failed: {result.stderr.strip()}")
        return ok

    def abort(self) -> bool:
        """Abort (rollback) the canary deployment."""
        result = subprocess.run(
            ["kubectl", "argo", "rollouts", "abort", self.service, "-n", self.ns],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            result = subprocess.run(
                ["argo-rollouts", "abort", self.service, "-n", self.ns],
                capture_output=True, text=True,
            )
        ok = result.returncode == 0
        if not ok:
            print(f"  [Argo] Abort failed: {result.stderr.strip()}")
        return ok

    def get_canary_weight(self) -> float:
        """Query current canary traffic weight from Rollout status."""
        try:
            result = subprocess.run(
                ["kubectl", "get", "rollout", self.service, "-n", self.ns,
                 "-o", "jsonpath={.status.canary.weights.canary.weight}"],
                capture_output=True, text=True, check=True,
            )
            return float(result.stdout.strip() or "0")
        except Exception:
            return 0.0


class RuleBasedController:
    """Rule-based canary controller — drop-in baseline for RL agent comparison.

    Args:
        method:      'static' or 'ratio' (threshold strategy).
        config:      Parsed dict from rule_based_thresholds.yaml.
        prometheus:  PrometheusClient instance.
        argo:        ArgoRolloutsClient instance.
        verbose:     Print decision info each step.
    """

    def __init__(
        self,
        method: str,
        config: dict,
        prometheus: PrometheusClient,
        argo: ArgoRolloutsClient,
        verbose: bool = True,
    ):
        assert method in ("static", "ratio"), f"Unknown method: {method}"
        self.method = method
        self.cfg = config[method]
        self.prom = prometheus
        self.argo = argo
        self.verbose = verbose

        self.step: int = 0
        self.consecutive_failures: int = 0
        self.consecutive_healthy: int = 0
        self.action_log: list[ActionRecord] = []
        self.done: bool = False

        svc = config["target"]["service_name"]
        win = config["prometheus"]["scrape_window"]

        # PromQL templates — use grpc_response_status for gRPC error accuracy
        self._q_canary_err = (
            f'sum(rate(istio_requests_total{{destination_service=~"{svc}-canary.*",'
            f'grpc_response_status!="0"}}[{win}])) / '
            f'sum(rate(istio_requests_total{{destination_service=~"{svc}-canary.*"}}[{win}]))'
        )
        self._q_stable_err = (
            f'sum(rate(istio_requests_total{{destination_service=~"{svc}\\\\..*",'
            f'grpc_response_status!="0"}}[{win}])) / '
            f'sum(rate(istio_requests_total{{destination_service=~"{svc}\\\\..*"}}[{win}]))'
        )
        self._q_canary_lat = (
            f'histogram_quantile(0.95, sum(rate('
            f'istio_request_duration_milliseconds_bucket{{'
            f'destination_service=~"{svc}-canary.*"}}[{win}])) by (le))'
        )
        self._q_stable_lat = (
            f'histogram_quantile(0.95, sum(rate('
            f'istio_request_duration_milliseconds_bucket{{'
            f'destination_service=~"{svc}\\\\..*"}}[{win}])) by (le))'
        )
        self._q_weight = (
            f'sum(rate(istio_requests_total{{destination_service=~"{svc}-canary.*"}}[{win}])) / '
            f'sum(rate(istio_requests_total{{destination_service=~"{svc}.*"}}[{win}]))'
        )

    # ── Observe ───────────────────────────────────────────────────────────────

    def observe(self) -> MetricsSnapshot:
        """Query Prometheus, return current metrics snapshot."""
        eps = self.cfg.get("epsilon", 1e-6)
        stable_err_floor = self.cfg.get("stable_error_floor", eps)
        stable_lat_floor = self.cfg.get("stable_latency_floor_ms", 40.0)

        canary_err = self.prom.query(self._q_canary_err) or 0.0
        stable_err = self.prom.query(self._q_stable_err) or 0.0
        canary_lat = self.prom.query(self._q_canary_lat) or 0.0
        stable_lat = self.prom.query(self._q_stable_lat) or 0.0
        weight = self.prom.query(self._q_weight) or 0.0

        error_ratio = canary_err / max(stable_err, stable_err_floor)
        latency_ratio = canary_lat / max(stable_lat, stable_lat_floor)

        return MetricsSnapshot(
            timestamp=datetime.now(timezone.utc).isoformat(),
            step=self.step,
            canary_error_rate=canary_err,
            stable_error_rate=stable_err,
            canary_p95_latency_ms=canary_lat,
            stable_p95_latency_ms=stable_lat,
            traffic_weight_canary=weight,
            error_ratio=error_ratio,
            latency_ratio=latency_ratio,
        )

    # ── Decide ────────────────────────────────────────────────────────────────

    def decide(self, metrics: MetricsSnapshot) -> tuple[int, str]:
        """Apply threshold logic, return (action, rationale) pair."""
        if self.method == "static":
            return self._decide_static(metrics)
        return self._decide_ratio(metrics)

    def _decide_static(self, m: MetricsSnapshot) -> tuple[int, str]:
        err_thr = self.cfg["error_rate_threshold"]
        lat_thr = self.cfg["latency_p95_threshold_ms"]
        fail_n = self.cfg["consecutive_failures"]
        ok_n = self.cfg["consecutive_healthy"]

        # Check anomaly
        anomalous = (m.canary_error_rate > err_thr) or (m.canary_p95_latency_ms > lat_thr)

        if anomalous:
            self.consecutive_failures += 1
            self.consecutive_healthy = 0
            if self.consecutive_failures >= fail_n:
                return ROLLBACK, (
                    f"Static threshold breached for {self.consecutive_failures} consecutive steps: "
                    f"err={m.canary_error_rate:.4f} (>{err_thr}), "
                    f"lat={m.canary_p95_latency_ms:.1f}ms (>{lat_thr}ms)"
                )
            return HOLD, (
                f"Threshold breached but only {self.consecutive_failures}/{fail_n} steps: "
                f"err={m.canary_error_rate:.4f}, lat={m.canary_p95_latency_ms:.1f}ms"
            )
        else:
            self.consecutive_failures = 0
            self.consecutive_healthy += 1
            if self.consecutive_healthy >= ok_n:
                self.consecutive_healthy = 0  # reset after promote
                return PROMOTE, (
                    f"Healthy for {ok_n} consecutive steps: "
                    f"err={m.canary_error_rate:.4f}, lat={m.canary_p95_latency_ms:.1f}ms"
                )
            return HOLD, (
                f"Healthy but only {self.consecutive_healthy}/{ok_n} steps: "
                f"err={m.canary_error_rate:.4f}, lat={m.canary_p95_latency_ms:.1f}ms"
            )

    def _decide_ratio(self, m: MetricsSnapshot) -> tuple[int, str]:
        err_thr = self.cfg["error_ratio_threshold"]
        lat_thr = self.cfg["latency_ratio_threshold"]
        fail_n = self.cfg["consecutive_failures"]
        ok_n = self.cfg["consecutive_healthy"]

        anomalous = (m.error_ratio > err_thr) or (m.latency_ratio > lat_thr)

        if anomalous:
            self.consecutive_failures += 1
            self.consecutive_healthy = 0
            if self.consecutive_failures >= fail_n:
                return ROLLBACK, (
                    f"Ratio threshold breached for {self.consecutive_failures} consecutive steps: "
                    f"err_ratio={m.error_ratio:.2f} (>{err_thr}), "
                    f"lat_ratio={m.latency_ratio:.2f} (>{lat_thr})"
                )
            return HOLD, (
                f"Ratio breached but only {self.consecutive_failures}/{fail_n} steps: "
                f"err_ratio={m.error_ratio:.2f}, lat_ratio={m.latency_ratio:.2f}"
            )
        else:
            self.consecutive_failures = 0
            self.consecutive_healthy += 1
            if self.consecutive_healthy >= ok_n:
                self.consecutive_healthy = 0
                return PROMOTE, (
                    f"Healthy ratio for {ok_n} consecutive steps: "
                    f"err_ratio={m.error_ratio:.2f}, lat_ratio={m.latency_ratio:.2f}"
                )
            return HOLD, (
                f"Healthy ratio but only {self.consecutive_healthy}/{ok_n} steps: "
                f"err_ratio={m.error_ratio:.2f}, lat_ratio={m.latency_ratio:.2f}"
            )

    # ── Execute ───────────────────────────────────────────────────────────────

    def execute(self, action: int) -> bool:
        """Execute action via Argo Rollouts. Returns True on success."""
        if action == PROMOTE:
            return self.argo.promote()
        if action == ROLLBACK:
            ok = self.argo.abort()
            if ok:
                self.done = True
            return ok
        return True  # HOLD — no-op

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run_loop(self) -> list[dict]:
        """Observe → decide → execute loop until rollback / full promote / timeout.

        Returns:
            action_log as list of dicts (same data as self.action_log).
        """
        max_steps = self.cfg["max_steps"]
        interval = self.cfg["check_interval_s"]

        print(f"\n{'='*60}")
        print(f"  RuleBasedController [{self.method.upper()}] starting")
        print(f"  max_steps={max_steps}, interval={interval}s")
        print(f"{'='*60}")

        while self.step < max_steps and not self.done:
            t0 = time.time()
            self.step += 1

            metrics = self.observe()
            action, rationale = self.decide(metrics)

            if self.verbose:
                self._print_step(metrics, action, rationale)

            self.execute(action)

            record = ActionRecord(
                timestamp=metrics.timestamp,
                step=self.step,
                action=action,
                action_name=ACTION_NAMES[action],
                method=self.method,
                rationale=rationale,
                metrics=metrics.to_dict(),
                consecutive_failures=self.consecutive_failures,
                consecutive_healthy=self.consecutive_healthy,
            )
            self.action_log.append(record)

            # Check full promote
            if metrics.traffic_weight_canary >= 0.99:
                print(f"\n  [Controller] Canary reached 100% traffic → episode complete.")
                self.done = True
                break

            # Sleep for remainder of interval
            elapsed = time.time() - t0
            sleep_time = max(0.0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if self.step >= max_steps:
            print(f"\n  [Controller] Max steps ({max_steps}) reached → timeout.")

        return [asdict(r) for r in self.action_log]

    # ── Export ────────────────────────────────────────────────────────────────

    def export_action_log(self, path: str) -> None:
        """Save action log to JSON for post-hoc analysis."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        log = [asdict(r) for r in self.action_log]
        with open(path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"  [Controller] Action log saved → {path} ({len(log)} records)")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _print_step(self, m: MetricsSnapshot, action: int, rationale: str) -> None:
        action_colors = {HOLD: "\033[93m", PROMOTE: "\033[92m", ROLLBACK: "\033[91m"}
        color = action_colors.get(action, "")
        reset = "\033[0m"
        print(
            f"\n[Step {self.step:02d}] {color}{ACTION_NAMES[action]}{reset}\n"
            f"  err: canary={m.canary_error_rate:.4f} stable={m.stable_error_rate:.4f} "
            f"ratio={m.error_ratio:.2f}\n"
            f"  lat: canary={m.canary_p95_latency_ms:.1f}ms "
            f"stable={m.stable_p95_latency_ms:.1f}ms ratio={m.latency_ratio:.2f}\n"
            f"  weight={m.traffic_weight_canary:.3f} | {rationale}"
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config_path: str,
        method: str,
        verbose: bool = True,
    ) -> "RuleBasedController":
        """Instantiate from YAML config file."""
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
        return cls(method=method, config=config, prometheus=prometheus, argo=argo, verbose=verbose)


# ─── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run rule-based canary controller")
    parser.add_argument("--method", choices=["static", "ratio"], required=True)
    parser.add_argument("--config", default="configs/rule_based_thresholds.yaml")
    parser.add_argument("--out", required=True, help="Path to save action_log.json")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    controller = RuleBasedController.from_config(
        config_path=args.config,
        method=args.method,
        verbose=not args.quiet,
    )
    controller.run_loop()
    controller.export_action_log(args.out)
