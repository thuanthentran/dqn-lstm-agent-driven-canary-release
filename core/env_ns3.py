import os
import csv
import math
import random
import numpy as np
from collections import defaultdict
from core.env import CanaryEnv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE_DIR = os.path.join(BASE_DIR, "ns-3_scenario_trace")

TRACE_FILES = {
    0: "kpi_trace_scenario0_stable.csv",
    1: "kpi_trace_scenario1_handoverstorm.csv",
    2: "kpi_trace_scenario2_ntngap.csv",
    3: "kpi_trace_scenario3_thzblockage.csv",
    4: "kpi_trace_scenario4_isaccontention.csv",
}

class CanaryEnvNs3(CanaryEnv):
    def __init__(self, seq_len=30):
        self.traces = {}
        self.trace_idx = 0
        self._load_traces()
        super().__init__(seq_len=seq_len)

    def reset(self, seed=None, options=None, randomize_scenario=True):
        self.trace_idx = 0
        return super().reset(seed=seed, options=options, randomize_scenario=randomize_scenario)

    def _load_traces(self):
        for scenario_id, filename in TRACE_FILES.items():
            filepath = os.path.join(TRACE_DIR, filename)
            trace_data = []
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Convert values to float
                        trace_data.append({
                            "time": float(row.get("time", 0)),
                            "throughput_mbps": float(row.get("throughput_mbps", 0)),
                            "delay_ms": float(row.get("delay_ms", 0)),
                            "jitter_ms": float(row.get("jitter_ms", 0)),
                            "lost_packets": float(row.get("lost_packets", 0)),
                        })
            if trace_data:
                self.traces[scenario_id] = trace_data
            else:
                # Fallback if trace is missing
                self.traces[scenario_id] = [{"time": 0, "throughput_mbps": 100, "delay_ms": 0.5, "jitter_ms": 0.05, "lost_packets": 0}]
                print(f"Warning: Trace file {filename} not found. Using fallback data.")

    def _network_noise_factor(self):
        sc = getattr(self, "network_scenario", 0)
        trace_data = self.traces.get(sc, self.traces[0])
        
        # Step through the trace sequentially to preserve temporal coherence of network events
        row = trace_data[self.trace_idx % len(trace_data)]
        self.trace_idx += 1
        
        delay_ms = row["delay_ms"]
        lost_packets = row["lost_packets"]
        throughput = row["throughput_mbps"]
        
        # Baseline terrestrial delay in the traces is around 0.2 - 0.9ms. We set baseline to 1.0ms.
        # So burst_factor scales linearly with delay_ms
        burst_factor = max(1.0, delay_ms / 1.0)
        
        # Additional burst factor if there are lost packets
        if lost_packets > 0:
            burst_factor += 1.0 + (lost_packets * 0.1)
        
        # Reverse engineer 6G telemetry from traces
        network_raw = {
            "handover_count": 0,
            "sinr_db": -85.0 + np.random.normal(0, 1.0),
            "prb_util": 0.4,
            "harq_nack": 0.02,
            "ntn_gap": 0,
            "isac_contention": 0.0,
        }

        # Override based on scenarios to feed the correct expected signals to the agent
        if sc == 1:  # HandoverStorm
            if delay_ms > 2.0 or throughput < 80:  # Heuristic for handover event in trace
                network_raw["handover_count"] = random.randint(2, 5)
                network_raw["sinr_db"] = -95.0 + np.random.normal(0, 5.0)
                network_raw["prb_util"] = 0.6
                network_raw["harq_nack"] = 0.06
        elif sc == 2:  # NTNGap
            if delay_ms > 10.0:  # Satellite link delay
                network_raw["ntn_gap"] = 1
                network_raw["sinr_db"] = -110.0 + np.random.normal(0, 2.0)
                network_raw["harq_nack"] = 0.15
        elif sc == 3:  # THzBlockage
            if throughput < 50.0 or lost_packets > 0:
                network_raw["sinr_db"] = -120.0 + np.random.normal(0, 2.0)
                network_raw["harq_nack"] = 0.2
        elif sc == 4:  # ISACContention
            if throughput < 90.0:
                contention = 0.3 + 0.3 * np.random.uniform()
                network_raw["isac_contention"] = contention

        return burst_factor, network_raw
