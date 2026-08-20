#!/bin/bash
# run_experiment.sh — Orchestrate a full chaos experiment run

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <scenario_yaml> <controller_type> <run_number>"
    echo "Example: $0 scenarios/S1_high_latency.yaml rule_based 01"
    exit 1
fi

SCENARIO_FILE=$1
CONTROLLER=$2
RUN_NUM=$3

# Extract scenario name from the yaml filename (e.g., S1_high_latency)
SCENARIO_NAME=$(basename "$SCENARIO_FILE" .yaml)
RUN_ID="${SCENARIO_NAME}-${CONTROLLER}-${RUN_NUM}"

echo "=========================================="
echo "Starting Experiment Run: $RUN_ID"
echo "=========================================="

# 1. Reset chaos (clean state)
python scripts/chaos_reset.py

# 2. Record start time
START_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "Start time recorded: $START_TIME"

# 3. Warmup (3 mins to establish baseline metrics)
echo "Warming up for 3 minutes..."
sleep 180

# 4. Inject fault according to scenario
echo "Starting fault injection timeline..."
python scripts/inject_fault.py --scenario "$SCENARIO_FILE" --run-id "$RUN_ID"

# 5. Record end time (give extra 2 mins for metrics to settle after last step)
echo "Cooldown period (2 minutes)..."
sleep 120
END_TIME=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# Reset chaos again at the very end
python scripts/chaos_reset.py

# 6. Export metrics
echo "Exporting metrics to CSV..."
python scripts/export_data.py \
  --run-id "$RUN_ID" \
  --start "$START_TIME" \
  --end "$END_TIME"

echo "=========================================="
echo "Run $RUN_ID Complete!"
echo "Data saved in results/raw/$RUN_ID/"
echo "=========================================="
