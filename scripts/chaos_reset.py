import subprocess
import json
import sys

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
        print(f"Successfully updated env for rollout/{target}")
    except subprocess.CalledProcessError as e:
        print(f"Error updating rollout: {e.stderr if e.stderr else e.output}")

def reset_chaos(target="checkoutservice", namespace="msdemo"):
    print("=== Resetting Chaos State ===")
    
    # Set all chaos-related env vars back to defaults/disabled
    env_dict = {
        "CHAOS_ENABLED": "false",
        "CHAOS_LATENCY_MS_MIN": "0",
        "CHAOS_LATENCY_MS_MAX": "0",
        "CHAOS_ERROR_RATE": "0.0",
        "CHAOS_CPU_PERCENT": "0",
        "CHAOS_MEM_ALLOC_MB": "0"
    }
    
    update_rollout_env(target, namespace, env_dict)
    print("Chaos reset completed.")

if __name__ == "__main__":
    reset_chaos()
