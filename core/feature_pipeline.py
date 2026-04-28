import math
from typing import Dict, List

EPSILON = 1e-6

# Runtime-aware reference scales for normalization.
CPU_REF = 0.02
MEM_REF_MB = 128.0
RPS_REF = 50.0
MAX_RATIO = 5.0
MAX_ERR = 1.0

RAW_KEYS = [
    "weight_pct",
    "e_canary",
    "e_stable",
    "l_canary",
    "l_stable",
    "cpu",
    "mem_mb",
    "rps",
]

STATE_KEYS = [
    "weight_n",
    "e_ratio_n",
    "l_ratio_n",
    "e_gap_n",
    "l_gap_n",
    "cpu_n",
    "mem_n",
    "rps_n",
]


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def normalize_raw_metrics(raw: Dict[str, float]) -> Dict[str, float]:
    e_canary = max(0.0, float(raw["e_canary"]))
    e_stable = max(0.0, float(raw["e_stable"]))
    l_canary = max(0.0, float(raw["l_canary"]))
    l_stable = max(0.0, float(raw["l_stable"]))

    e_ratio = e_canary / max(e_stable, EPSILON)
    l_ratio = l_canary / max(l_stable, EPSILON)
    e_gap = max(0.0, e_canary - e_stable)
    l_gap_ratio = max(0.0, (l_canary - l_stable) / max(l_stable, EPSILON))

    state = {
        "weight_n": _clip(float(raw["weight_pct"]) / 100.0, 0.0, 1.0),
        "e_ratio_n": _clip(e_ratio / MAX_RATIO, 0.0, 1.0),
        "l_ratio_n": _clip(l_ratio / MAX_RATIO, 0.0, 1.0),
        "e_gap_n": _clip(e_gap / MAX_ERR, 0.0, 1.0),
        "l_gap_n": _clip(l_gap_ratio / MAX_RATIO, 0.0, 1.0),
        "cpu_n": _clip(float(raw["cpu"]) / CPU_REF, 0.0, 1.0),
        "mem_n": _clip(float(raw["mem_mb"]) / MEM_REF_MB, 0.0, 1.0),
        "rps_n": _clip(float(raw["rps"]) / RPS_REF, 0.0, 1.0),
    }

    return state


def to_state_vector(raw: Dict[str, float]) -> List[float]:
    normalized = normalize_raw_metrics(raw)
    return [normalized[key] for key in STATE_KEYS]


class RunningFeatureStats:
    def __init__(self, keys: List[str]):
        self.keys = list(keys)
        self.count = 0
        self.mean = {k: 0.0 for k in self.keys}
        self.m2 = {k: 0.0 for k in self.keys}
        self.min = {k: math.inf for k in self.keys}
        self.max = {k: -math.inf for k in self.keys}

    def update(self, values: Dict[str, float]) -> None:
        self.count += 1
        for key in self.keys:
            value = float(values.get(key, 0.0))
            self.min[key] = min(self.min[key], value)
            self.max[key] = max(self.max[key], value)

            delta = value - self.mean[key]
            self.mean[key] += delta / self.count
            delta2 = value - self.mean[key]
            self.m2[key] += delta * delta2

    def summary(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        for key in self.keys:
            variance = 0.0
            if self.count > 1:
                variance = self.m2[key] / (self.count - 1)
            summary[key] = {
                "min": self.min[key] if self.min[key] != math.inf else 0.0,
                "max": self.max[key] if self.max[key] != -math.inf else 0.0,
                "mean": self.mean[key],
                "std": math.sqrt(max(variance, 0.0)),
            }
        return summary
