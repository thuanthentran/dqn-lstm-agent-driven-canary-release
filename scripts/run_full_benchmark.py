"""
Orchestrates the full benchmark experiment (100+ runs) with checkpointing.

Usage:
  python3 scripts/run_full_benchmark.py \
    --out-dir results_final \
    --n-runs 10 \
    --controllers rl_agent rule_based_static rule_based_ratio \
    --seed 42
"""

import argparse
import json
import os
import random
import subprocess
import time
from datetime import datetime

def generate_schedule(scenarios, controllers, n_runs, seed=42):
    random.seed(seed)
    schedule = []
    
    # We will run blocks where each block contains one run of every combination
    for run_num in range(1, n_runs + 1):
        block = []
        for scenario in scenarios:
            for controller in controllers:
                run_id = f"{scenario.replace('.yaml', '')}-{controller}-{run_num:02d}"
                block.append({
                    "id": run_id,
                    "scenario": f"scenarios/{scenario}",
                    "controller": controller,
                    "run_num": run_num
                })
        # Shuffle the block to avoid ordering biases
        random.shuffle(block)
        schedule.extend(block)
        
    return schedule

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--controllers", nargs="+", default=["rl_agent", "rule_based_static", "rule_based_ratio"])
    parser.add_argument("--scenarios", nargs="+", default=[
        "S1_high_latency.yaml", 
        "S2_cpu_spike.yaml", 
        "S3_memory_leak.yaml",
        "S4_error_burst.yaml", 
        "S5_cascading_failure.yaml"
    ])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    checkpoint_file = os.path.join(args.out_dir, "benchmark_progress.json")
    
    completed_runs = set()
    started_at = datetime.now().isoformat()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            data = json.load(f)
            completed_runs = set(data.get("completed", []))
            started_at = data.get("started_at", started_at)

    schedule = generate_schedule(args.scenarios, args.controllers, args.n_runs, args.seed)
    total_runs = len(schedule)
    
    print(f"Total runs scheduled: {total_runs}")
    print(f"Completed runs found: {len(completed_runs)}")
    print(f"Runs remaining: {total_runs - len(completed_runs)}")
    
    for i, run in enumerate(schedule, 1):
        if run["id"] in completed_runs:
            print(f"[{i}/{total_runs}] SKIP (done): {run['id']}")
            continue
            
        print(f"\n[{i}/{total_runs}] STARTING: {run['id']}")
        
        cmd = [
            "python3", "scripts/run_experiment_v3.py",
            "--scenario", run["scenario"],
            "--controller", run["controller"],
            "--run-num", str(run["run_num"]),
            "--out-dir", args.out_dir,
            "--no-locust",
            "--warmup-sec", "60",
            "--cooldown-sec", "30"
        ]
        if args.dry_run:
            cmd.append("--dry-run")
            
        start_time = time.time()
        
        # Execute the run
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"ERROR running {run['id']}: {e}")
            print("Aborting benchmark.")
            break
            
        elapsed = time.time() - start_time
        print(f"[{i}/{total_runs}] COMPLETED: {run['id']} in {elapsed:.1f}s")
        
        completed_runs.add(run["id"])
        
        # Update checkpoint
        with open(checkpoint_file, "w") as f:
            json.dump({
                "started_at": started_at,
                "completed": list(completed_runs)
            }, f, indent=2)

    print("\nBenchmark sequence finished!")

if __name__ == "__main__":
    main()
