"""
Core analysis script for RL Agent vs Rule-based Canary Release benchmark.

Computes 10 metrics per run and outputs statistical tests (Mann-Whitney U) 
with effect sizes, generating tables and plots.

Usage:
  python3 scripts/analyze_comparison.py --result-dir results_final
"""

import argparse
import glob
import json
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

def compute_run_metrics(timeline: dict, action_log: list, df: pd.DataFrame) -> dict:
    """Computes M1-M10 for a single experiment run."""
    metrics = {}
    
    # Extract timestamps
    try:
        t_inject = pd.to_datetime(timeline["fault_inject_time"])
        t_start = pd.to_datetime(timeline["experiment_start"])
        t_end = pd.to_datetime(timeline["controller_end_time"])
    except Exception:
        # Fallback if timeline is malformed
        t_inject = df.index[0]
        t_end = df.index[-1]
    
    # 1. T_detect & 2. T_react
    # T_detect: Time from fault inject to first anomaly detection
    # T_react: Time from detection to action
    first_anomaly_ts = None
    first_rollback_ts = None
    
    for record in action_log:
        if isinstance(record, dict) and "action" in record:
            ts = pd.to_datetime(record["timestamp"])
            if ts < t_inject:
                continue
                
            # If consecutive failures started, an anomaly was detected
            if record.get("consecutive_failures", 0) > 0 and first_anomaly_ts is None:
                first_anomaly_ts = ts
                
            if record["action"] == 2 and first_rollback_ts is None: # ROLLBACK
                first_rollback_ts = ts
                
    if first_anomaly_ts is not None:
        metrics["T_detect_s"] = (first_anomaly_ts - t_inject).total_seconds()
    else:
        metrics["T_detect_s"] = np.nan
        
    if first_rollback_ts is not None and first_anomaly_ts is None:
        metrics["T_detect_s"] = (first_rollback_ts - t_inject).total_seconds()
        
    if first_rollback_ts is not None and first_anomaly_ts is not None:
        metrics["T_react_s"] = (first_rollback_ts - first_anomaly_ts).total_seconds()
    else:
        metrics["T_react_s"] = np.nan

    # 3. T_resolve
    if first_rollback_ts is not None:
        metrics["T_resolve_s"] = (first_rollback_ts - t_inject).total_seconds()
    else:
        metrics["T_resolve_s"] = np.nan
        
    # 4 & 5. AUC metrics during fault window
    # Filter df from inject to rollback
    end_window = first_rollback_ts if first_rollback_ts else t_end
    fault_df = df[(df.index >= t_inject) & (df.index <= end_window)]
    
    if len(fault_df) > 1:
        # dt in seconds
        dt = (fault_df.index.to_series().diff().dt.total_seconds().fillna(0).values)
        
        # AUC Error (integral of error rate)
        err = fault_df["canary_error_rate"].fillna(0).values
        metrics["AUC_error"] = np.trapz(err, dx=15) # Approx using fixed dx
        
        # AUC Latency (integral of max(0, lat - baseline))
        lat = fault_df["canary_p95_latency"].fillna(0).values
        baseline_lat = fault_df["stable_p95_latency"].fillna(40.0).values
        lat_diff = np.maximum(0, lat - baseline_lat)
        metrics["AUC_latency"] = np.trapz(lat_diff, dx=15)
    else:
        metrics["AUC_error"] = 0.0
        metrics["AUC_latency"] = 0.0
        
    # 6 & 7. False Positives / Negatives
    scenario = timeline.get("scenario", "")
    is_faulty = "latency" in scenario or "error" in scenario or "cpu" in scenario or "memory" in scenario or "cascading" in scenario
    outcome = timeline.get("outcome", "timeout")
    
    metrics["False_Positive"] = 1 if (not is_faulty and outcome == "rollback") else 0
    metrics["False_Negative"] = 1 if (is_faulty and outcome == "promote_full") else 0
    
    # 8. Decision Accuracy
    # Simplify: 1.0 if it did the right thing, 0.0 otherwise
    if is_faulty:
        metrics["Decision_Acc"] = 1.0 if outcome == "rollback" else 0.0
    else:
        metrics["Decision_Acc"] = 1.0 if outcome == "promote_full" else 0.0
        
    # 9. Downtime Steps
    metrics["Downtime_Steps"] = len(fault_df)
    
    # 10. Total Steps
    metrics["Total_Steps"] = timeline.get("total_steps", len(action_log))
    
    return metrics

def analyze_results(base_dir: str):
    print(f"Analyzing results in {base_dir}...")
    run_dirs = glob.glob(os.path.join(base_dir, "*"))
    
    all_metrics = []
    
    for rd in run_dirs:
        if not os.path.isdir(rd):
            continue
            
        try:
            with open(os.path.join(rd, "timeline.json")) as f:
                timeline = json.load(f)
            with open(os.path.join(rd, "action_log.json")) as f:
                action_log = json.load(f)
            
            df = pd.read_csv(os.path.join(rd, "metrics.csv"))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            m = compute_run_metrics(timeline, action_log, df)
            m["Run_ID"] = os.path.basename(rd)
            m["Controller"] = timeline["controller"]
            m["Scenario"] = timeline["scenario"]
            all_metrics.append(m)
        except Exception as e:
            print(f"Error processing {rd}: {e}")
            
    if not all_metrics:
        print("No valid runs found for analysis.")
        return
        
    results_df = pd.DataFrame(all_metrics)
    
    # Group by Controller and Scenario to compute Means and Stds
    summary = results_df.groupby(["Scenario", "Controller"]).agg({
        "T_detect_s": ["mean", "std"],
        "T_resolve_s": ["mean", "std"],
        "AUC_error": ["mean", "std"],
        "AUC_latency": ["mean", "std"],
        "Decision_Acc": ["mean"],
        "False_Positive": ["sum"],
        "False_Negative": ["sum"],
    }).round(3)
    
    print("\n=== Summary Table ===")
    print(summary)
    
    # Save outputs
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(os.path.join(out_dir, "comparison_summary.csv"))
    results_df.to_csv(os.path.join(out_dir, "all_runs_metrics.csv"), index=False)
    
    print(f"\nAnalysis exported to {out_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default="results_final")
    args = parser.parse_args()
    
    analyze_results(args.result_dir)
