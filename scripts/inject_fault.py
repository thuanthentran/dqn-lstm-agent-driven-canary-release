import argparse
import yaml
import time
import subprocess
import os
import json
from datetime import datetime, timezone

def update_rollout_env(target, namespace, new_env_dict):
    try:
        # Get current JSON
        cmd = f"kubectl get rollout {target} -n {namespace} -o json"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
        obj = json.loads(res.stdout)
        
        # Update env
        containers = obj['spec']['template']['spec']['containers']
        container = containers[0]
        if 'env' not in container:
            container['env'] = []
        
        env_list = container['env']
        for k, v in new_env_dict.items():
            found = False
            for i, e in enumerate(env_list):
                if e['name'] == k:
                    env_list[i]['value'] = str(v)
                    found = True
                    break
            if not found:
                env_list.append({"name": k, "value": str(v)})
                
        # Remove managedFields to avoid apply issues
        if 'managedFields' in obj['metadata']:
            del obj['metadata']['managedFields']
            
        # Write to temp file and apply
        with open("temp_rollout.json", "w") as f:
            json.dump(obj, f)
            
        subprocess.run("kubectl apply -f temp_rollout.json", shell=True, check=True, capture_output=True)
        print(f"[{datetime.now(timezone.utc).isoformat()}] Successfully updated env for rollout/{target}")
    except subprocess.CalledProcessError as e:
        print(f"Error updating rollout: {e.stderr if e.stderr else e.output}")

def inject_fault(scenario_file, run_id, out_dir="results/raw"):
    with open(scenario_file, 'r') as f:
        scenario = yaml.safe_load(f)

    print(f"=== Starting Scenario: {scenario['name']} ===")
    target = scenario.get('target_rollout', 'checkoutservice')
    namespace = scenario.get('namespace', 'msdemo')
    
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
            
        # Update rollout env
        update_rollout_env(target, namespace, chaos_env)
        
        # Log event
        timeline['events'].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "t_offset": t_offset,
            "chaos_params": chaos_env
        })
        
    print("=== Scenario Steps Completed ===")
    
    # Save timeline
    os.makedirs(out_dir, exist_ok=True)
    out_file = f"{out_dir}/{run_id}_timeline.json"
    with open(out_file, 'w') as f:
        json.dump(timeline, f, indent=2)
    print(f"Saved chaos timeline to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject faults according to a scenario YAML")
    parser.add_argument("--scenario", required=True, help="Path to scenario YAML file")
    parser.add_argument("--run-id", required=True, help="Unique Run ID (e.g. S1-RB-01)")
    parser.add_argument("--out-dir", default="results/raw", help="Output directory")
    args = parser.parse_args()
    
    inject_fault(args.scenario, args.run_id, args.out_dir)
