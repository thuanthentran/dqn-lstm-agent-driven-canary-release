import argparse
import json
import subprocess
from datetime import datetime, timezone

def update_rollout_env(target, namespace, config_dict):
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
        config_str = json.dumps(config_dict)
        
        found = False
        for i, e in enumerate(env_list):
            if e['name'] == 'CHAOS_CONFIG':
                env_list[i]['value'] = config_str
                found = True
                break
        if not found:
            env_list.append({"name": "CHAOS_CONFIG", "value": config_str})
                
        # Remove managedFields to avoid apply issues
        if 'managedFields' in obj['metadata']:
            del obj['metadata']['managedFields']
            
        # Write to temp file and apply
        with open("temp_rollout.json", "w") as f:
            json.dump(obj, f)
            
        subprocess.run("kubectl apply -f temp_rollout.json", shell=True, check=True, capture_output=True)
        print(f"[{datetime.now(timezone.utc).isoformat()}] Successfully injected CHAOS_CONFIG to rollout/{target}")
    except subprocess.CalledProcessError as e:
        print(f"Error updating rollout: {e.stderr if e.stderr else e.output}")

def inject_fault(scenario_file, run_id, out_dir="results/raw"):
    with open(scenario_file, 'r') as f:
        if scenario_file.endswith('.json'):
            config = json.load(f)
        else:
            # Fallback wrapper if accidentally passing old YAML (we will assume it's static latency)
            print("Warning: Passed .yaml file. Converting to static pattern.")
            import yaml
            y = yaml.safe_load(f)
            config = {
                "enabled": True,
                "pattern": "static",
                "signals": {
                    "latency": {
                        "pattern": "static",
                        "params": {"value": 500.0}
                    }
                }
            }

    config["run_id"] = run_id
    
    target = "checkoutservice"
    namespace = "msdemo"
    
    print(f"=== Starting Injection for Run: {run_id} ===")
    update_rollout_env(target, namespace, config)
    print("=== Injection Completed (Module will evolve fault internally) ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inject faults via CHAOS_CONFIG")
    parser.add_argument("--scenario", required=True, help="Path to scenario JSON file")
    parser.add_argument("--run-id", required=True, help="Unique Run ID")
    parser.add_argument("--out-dir", default="results/raw", help="Output directory")
    args = parser.parse_args()
    
    inject_fault(args.scenario, args.run_id, args.out_dir)
