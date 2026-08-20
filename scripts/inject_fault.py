import argparse
import yaml
import time
import subprocess
import os
import json
from datetime import datetime

def run_cmd(cmd):
    print(f"[{datetime.utcnow().isoformat()}Z] Executing: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e.stderr.decode()}")

def inject_fault(scenario_file, run_id):
    with open(scenario_file, 'r') as f:
        scenario = yaml.safe_load(f)

    print(f"=== Starting Scenario: {scenario['name']} ===")
    target = scenario.get('target_rollout', 'checkoutservice')
    namespace = scenario.get('namespace', 'default')
    
    # Store timeline for audit
    timeline = {
        "run_id": run_id,
        "scenario": scenario['name'],
        "target": target,
        "events": []
    }
    
    start_time = time.time()
    
    for step in scenario['steps']:
        t_offset = step['t_offset_seconds']
        chaos_env = step['chaos']
        
        # Calculate time to sleep
        elapsed = time.time() - start_time
        sleep_time = t_offset - elapsed
        if sleep_time > 0:
            print(f"Waiting for {sleep_time:.1f}s until t={t_offset}s...")
            time.sleep(sleep_time)
            
        # Build kubectl set env command
        env_args = " ".join([f"{k}='{v}'" for k, v in chaos_env.items()])
        cmd = f"kubectl set env rollout/{target} -n {namespace} {env_args}"
        run_cmd(cmd)
        
        # Log event
        timeline['events'].append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "t_offset": t_offset,
            "chaos_params": chaos_env
        })
        
    print("=== Scenario Steps Completed ===")
    
    # Save timeline
    os.makedirs('results/raw', exist_ok=True)
    out_file = f"results/raw/{run_id}_timeline.json"
    with open(out_file, 'w') as f:
        json.dump(timeline, f, indent=2)
    print(f"Saved chaos timeline to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject faults according to a scenario YAML")
    parser.add_argument("--scenario", required=True, help="Path to scenario YAML file")
    parser.add_argument("--run-id", required=True, help="Unique Run ID (e.g. S1-RB-01)")
    args = parser.parse_args()
    
    inject_fault(args.scenario, args.run_id)
