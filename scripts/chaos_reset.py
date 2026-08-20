import subprocess

def run_cmd(cmd):
    print(f"Executing: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError:
        print(f"Error resetting chaos state")

def reset_chaos(target="checkoutservice", namespace="msdemo"):
    print("=== Resetting Chaos State ===")
    
    # Set all chaos-related env vars back to defaults/disabled
    env_args = (
        "CHAOS_ENABLED='false' "
        "CHAOS_LATENCY_MS_MIN='0' "
        "CHAOS_LATENCY_MS_MAX='0' "
        "CHAOS_ERROR_RATE='0.0' "
        "CHAOS_CPU_PERCENT='0' "
        "CHAOS_MEM_ALLOC_MB='0'"
    )
    
    cmd = f"kubectl set env rollout/{target} -n {namespace} {env_args}"
    run_cmd(cmd)
    
    print("Chaos reset completed.")

if __name__ == "__main__":
    reset_chaos()
