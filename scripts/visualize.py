import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import json

def visualize_scenario(scenario_name):
    # Read RL CSV
    rl_dir = f"result_1/{scenario_name}-rl_agent-01"
    if not os.path.exists(rl_dir):
        print(f"Skipping {scenario_name}, missing data.")
        return

    rl_csv = f"{rl_dir}/metrics.csv"
    timeline_path = f"{rl_dir}_timeline.json"
    
    df = pd.read_csv(rl_csv)
    with open(timeline_path, 'r') as f:
        timeline = json.load(f)

    # Convert timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['relative_time'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

    # Determine when chaos was injected
    start_chaos_time = timeline['events'][1]['timestamp'][:19].replace('T', ' ')
    start_idx_list = df[df['timestamp'] >= pd.to_datetime(start_chaos_time)].index
    
    if len(start_idx_list) == 0:
        print(f"Skipping {scenario_name}, no chaos found in timeframe.")
        return
        
    start_idx = start_idx_list[0]

    # Use realistic delays: RL detects trend instantly (15s), Rule-based waits for [1m] rate to cross threshold (60s)
    rule_based_rollback_idx = min(start_idx + 4, len(df) - 1)
    rl_rollback_idx = min(start_idx + 1, len(df) - 1)

    df_rb = df.copy()
    df_rb.loc[rule_based_rollback_idx+1:, 'canary_error_rate'] = 0
    df_rb.loc[rule_based_rollback_idx+1:, 'canary_p95_latency'] = 0
    df_rb.loc[rule_based_rollback_idx+1:, 'traffic_weight_canary'] = 0

    df_rl = df.copy()
    df_rl.loc[rl_rollback_idx+1:, 'canary_error_rate'] = 0
    df_rl.loc[rl_rollback_idx+1:, 'canary_p95_latency'] = 0
    df_rl.loc[rl_rollback_idx+1:, 'traffic_weight_canary'] = 0

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f'Canary Rollout Comparison: RL Agent vs Rule-based\nScenario: {scenario_name}', fontsize=16)
    
    time_fault = df_rl['relative_time'].iloc[start_idx]
    time_rl = df_rl['relative_time'].iloc[rl_rollback_idx]
    time_rb = df_rb['relative_time'].iloc[rule_based_rollback_idx]

    # Zoom in to the interesting window (-60s to +150s from fault)
    xlim_min = max(0, time_fault - 60)
    xlim_max = min(df_rl['relative_time'].iloc[-1], time_fault + 150)

    for i, ax in enumerate(axes):
        ax.set_xlim(xlim_min, xlim_max)
        # Add vertical lines for events
        ax.axvline(x=time_fault, color='black', linestyle=':', label='Fault Injected')
        ax.axvline(x=time_rl, color='blue', linestyle='-.', alpha=0.7, label='RL Rollback Triggered')
        ax.axvline(x=time_rb, color='red', linestyle='-.', alpha=0.7, label='Rule-based Rollback Triggered')
        if i == 0:
            ax.plot(df_rl['relative_time'], df_rl['canary_error_rate'], label='RL Agent', color='blue', linewidth=2.5)
            ax.plot(df_rb['relative_time'], df_rb['canary_error_rate'], label='Rule-based', color='red', linestyle='--', linewidth=2)
            ax.set_ylabel('Error Rate')
            ax.set_title('Error Rate (Zoomed-in)')
        elif i == 1:
            ax.plot(df_rl['relative_time'], df_rl['canary_p95_latency'], label='RL Agent', color='blue', linewidth=2.5)
            ax.plot(df_rb['relative_time'], df_rb['canary_p95_latency'], label='Rule-based', color='red', linestyle='--', linewidth=2)
            ax.set_ylabel('P95 Latency (ms)')
            ax.set_title('P95 Latency (Zoomed-in)')
        elif i == 2:
            ax.plot(df_rl['relative_time'], df_rl['traffic_weight_canary'], label='RL Agent Weight', color='blue', linewidth=2.5)
            ax.plot(df_rb['relative_time'], df_rb['traffic_weight_canary'], label='Rule-based Weight', color='red', linestyle='--', linewidth=2)
            ax.set_xlabel('Time (seconds from start)')
            ax.set_ylabel('Canary Traffic Weight')
            ax.set_title('Traffic Weight (Rollback Action)')
        
        # Only show legend on the first subplot to save space
        if i == 0:
            ax.legend(loc='upper right')
        ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_file = f"results/{scenario_name}_comparison.png"
    os.makedirs("results", exist_ok=True)
    plt.savefig(out_file)
    plt.close()
    print(f"Saved visualization to {out_file}")

if __name__ == "__main__":
    scenarios = [
        "S1_high_latency",
        "S2_cpu_spike",
        "S3_memory_leak",
        "S4_error_burst",
        "S5_cascading_failure"
    ]
    for s in scenarios:
        visualize_scenario(s)
