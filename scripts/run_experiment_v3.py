"""
Unified Experiment Runner for RL Agent vs Rule-based Canary Release.

Usage:
  python3 scripts/run_experiment_v3.py \
    --scenario services/src/checkoutservice/chaos/examples/linear.json \
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

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
    except:
        return "unknown"

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
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--max-steps", type=int, default=50)
    args = parser.parse_args()

    # Setup directories
    scenario_name = os.path.basename(args.scenario).replace(".yaml", "").replace(".json", "")
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
                ["locust", "-f", "services/src/loadgenerator/locustfile.py", "--headless", "-u", "100", "-r", "10", "--host", "http://localhost"],
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
    tail_process = None
    if not args.dry_run:
        fault_process = subprocess.Popen(["python3", "scripts/inject_fault.py", "--scenario", args.scenario, "--run-id", run_id])
        
        # Start streaming ground truth log immediately to avoid losing it if canary pod is deleted
        try:
            time.sleep(2) # Give the new canary pod a moment to start
            cmd = "kubectl get pods -n msdemo -l app=checkoutservice -o json"
            res = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
            pods = json.loads(res.stdout).get("items", [])
            
            canary_pod = None
            for pod in pods:
                for c in pod.get("spec", {}).get("containers", []):
                    for env in c.get("env", []):
                        if env["name"] == "CHAOS_CONFIG" and run_id in env["value"]:
                            canary_pod = pod
                            break
            
            if canary_pod:
                pod_name = canary_pod["metadata"]["name"]
                gt_path = os.path.join(out_path, "ground_truth.jsonl")
                print(f"Streaming ground truth log from {pod_name} to {gt_path}...")
                tail_cmd = f"kubectl exec -n msdemo {pod_name} -c server -- tail -f /var/log/chaos/ground_truth.jsonl > {gt_path}"
                tail_process = subprocess.Popen(tail_cmd, shell=True)
                
                # Save metadata now
                metadata = {
                    "run_id": run_id,
                    "node_name": canary_pod.get("spec", {}).get("nodeName", "unknown"),
                    "git_commit": get_git_commit(),
                    "controller": args.controller,
                    "scenario": scenario_name
                }
                with open(os.path.join(out_path, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to start log stream: {e}")

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
            action_log = runner.run_loop(check_interval=args.interval, max_steps=args.max_steps)
        elif args.controller.startswith("rule_based_"):
            method = args.controller.split("_")[-1]
            ctrl = RuleBasedController.from_config(
                config_path=args.config,
                method=method
            )
            action_log = ctrl.run_loop(check_interval=args.interval, max_steps=args.max_steps)
            
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
        
        last_rec = action_log[-1]
        if last_rec.get("metrics", {}).get("traffic_weight_canary", 0) >= 0.99:
            outcome = "promote_full"

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
        
    if tail_process:
        print("[6/6] Stopping Ground Truth log stream...")
        tail_process.terminate()
        
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
        
    print("[6/6] End of data extraction.")

    print("[6/6] Final cleanup...")
    if not args.dry_run:
        subprocess.run(["python3", "scripts/chaos_reset.py"], check=False)
        
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
