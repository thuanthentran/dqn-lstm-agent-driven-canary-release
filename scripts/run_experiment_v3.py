"""
Unified Experiment Runner for RL Agent vs Rule-based Canary Release.

Usage:
  python3 scripts/run_experiment_v3.py \
    --scenario scenarios/S1_high_latency.yaml \
    --controller rl_agent \
    --run-num 01 \
    --out-dir result_1 \
    [--dry-run]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from core.rule_based_controller import RuleBasedController
try:
    from scripts.run_rl_agent_inference import RLAgentRunner
except ImportError:
    RLAgentRunner = None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--controller", choices=["rl_agent", "rule_based_static", "rule_based_ratio"], required=True)
    parser.add_argument("--run-num", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--config", default="configs/rule_based_thresholds.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-locust", action="store_true")
    parser.add_argument("--warmup-sec", type=int, default=180)
    parser.add_argument("--cooldown-sec", type=int, default=120)
    args = parser.parse_args()

    # Setup directories
    scenario_name = os.path.basename(args.scenario).replace(".yaml", "")
    run_id = f"{scenario_name}-{args.controller}-{args.run_num:02d}"
    out_path = os.path.join(BASE_DIR, args.out_dir, run_id)
    os.makedirs(out_path, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"=== Starting Experiment: {run_id} ===")
    print(f"{'='*60}")
    
    # 1. Reset state
    print("[1/6] Resetting cluster state...")
    if not args.dry_run:
        subprocess.run(["python3", "scripts/chaos_reset.py"], check=False)
    
    experiment_start = datetime.now(timezone.utc).isoformat()
    
    # 2. Locust
    locust_process = None
    if not args.no_locust:
        print("[2/6] Starting Locust load generator...")
        if not args.dry_run:
            locust_process = subprocess.Popen(
                ["locust", "-f", "loadgenerator/locustfile.py", "--headless", "-u", "100", "-r", "10", "--host", "http://localhost"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
    else:
        print("[2/6] Skipping Locust (no-locust flag set)")
            
    # 3. Warmup
    print(f"[3/6] Warmup sleep ({args.warmup_sec if not args.dry_run else 0.1}s)...")
    if not args.dry_run:
        time.sleep(args.warmup_sec)
        
    # 4. Inject fault (in background)
    print(f"[4/6] Injecting fault: {args.scenario}...")
    fault_inject_time = datetime.now(timezone.utc).isoformat()
    fault_process = None
    if not args.dry_run:
        fault_process = subprocess.Popen(["python3", "scripts/inject_fault.py", "--scenario", args.scenario, "--run-id", run_id])
        
    # 5. Controller loop
    print(f"[5/6] Starting controller: {args.controller}...")
    controller_start_time = datetime.now(timezone.utc).isoformat()
    
    action_log = []
    if not args.dry_run:
        if args.controller == "rl_agent":
            if RLAgentRunner is None:
                print("Error: Could not import RLAgentRunner. Missing dependencies (e.g. stable_baselines3)?")
                sys.exit(1)
            runner = RLAgentRunner.from_config(
                model_path="models/ppo_transformer_offline_best.zip",
                norm_path="models/vec_normalize.pkl",
                config_path=args.config
            )
            action_log = runner.run_loop()
        elif args.controller.startswith("rule_based_"):
            method = args.controller.split("_")[-1] # "static" or "ratio"
            ctrl = RuleBasedController.from_config(
                config_path=args.config,
                method=method
            )
            action_log = ctrl.run_loop()
            
    controller_end_time = datetime.now(timezone.utc).isoformat()
    
    # Analyze outcome from action_log
    outcome = "timeout"
    first_rollback = None
    first_promote = None
    if action_log:
        for rec in action_log:
            act = rec["action"]
            if act == 2 and first_rollback is None:
                first_rollback = rec["step"]
                outcome = "rollback"
            if act == 1 and first_promote is None:
                first_promote = rec["step"]
        
        # Check if full promote
        last_rec = action_log[-1]
        if last_rec.get("metrics", {}).get("traffic_weight_canary", 0) >= 0.99:
            outcome = "promote_full"

    # Save action log
    action_log_path = os.path.join(out_path, "action_log.json")
    if args.dry_run:
        action_log = [{"dry_run": True}]
    with open(action_log_path, "w") as f:
        json.dump(action_log, f, indent=2)

    # 6. Cooldown & Export
    if locust_process:
        print("[6/6] Stopping Locust...")
        locust_process.terminate()
        
    if fault_process:
        print("[6/6] Stopping Fault Injection (if still running)...")
        fault_process.terminate()
        
    print(f"[6/6] Cooldown sleep ({args.cooldown_sec if not args.dry_run else 0.1}s)...")
    if not args.dry_run:
        time.sleep(args.cooldown_sec)
        
    print(f"[6/6] Exporting metrics to {out_path}/metrics.csv...")
    if not args.dry_run:
        subprocess.run([
            "python3", "scripts/export_data.py", 
            "--out", os.path.join(out_path, "metrics.csv"),
            "--start", experiment_start,
            "--end", datetime.now(timezone.utc).isoformat()
        ], check=False)
        
    print("[6/6] Final cleanup...")
    if not args.dry_run:
        subprocess.run(["python3", "scripts/chaos_reset.py"], check=False)
        
    # Save timeline
    timeline = {
        "scenario": scenario_name,
        "controller": args.controller,
        "run_num": args.run_num,
        "experiment_start": experiment_start,
        "fault_inject_time": fault_inject_time,
        "controller_start_time": controller_start_time,
        "controller_end_time": controller_end_time,
        "outcome": outcome,
        "total_steps": len(action_log),
        "first_rollback_step": first_rollback,
        "first_promote_step": first_promote
    }
    
    with open(os.path.join(out_path, "timeline.json"), "w") as f:
        json.dump(timeline, f, indent=2)
        
    print(f"\n=== Experiment {run_id} Complete ===")
    print(f"Outcome: {outcome}")
    print(f"Total steps: {timeline['total_steps']}")

if __name__ == "__main__":
    main()
