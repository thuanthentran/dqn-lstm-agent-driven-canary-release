import argparse
import requests
import pandas as pd
from datetime import datetime, timezone
import os
import json

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")

# Define the queries we want to export
QUERIES = {
    "canary_error_rate": 'sum(rate(istio_requests_total{destination_version="canary",response_code=~"5.."}[1m])) / sum(rate(istio_requests_total{destination_version="canary"}[1m]))',
    "stable_error_rate": 'sum(rate(istio_requests_total{destination_version="stable",response_code=~"5.."}[1m])) / sum(rate(istio_requests_total{destination_version="stable"}[1m]))',
    "canary_p95_latency": 'histogram_quantile(0.95, sum(rate(istio_request_duration_milliseconds_bucket{destination_version="canary"}[1m])) by (le))',
    "stable_p95_latency": 'histogram_quantile(0.95, sum(rate(istio_request_duration_milliseconds_bucket{destination_version="stable"}[1m])) by (le))',
    "traffic_weight_canary": 'sum(rate(istio_requests_total{destination_version="canary"}[1m])) / sum(rate(istio_requests_total[1m]))'
}

def query_prometheus(query, start_time, end_time, step="15s"):
    params = {
        'query': query,
        'start': start_time,
        'end': end_time,
        'step': step
    }
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query_range", params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] != 'success' or not data['data']['result']:
            return pd.Series(dtype=float)
            
        # Parse the first result (assuming single scalar timeseries per query due to sum())
        values = data['data']['result'][0]['values']
        
        # Convert to pandas Series: index is timestamp, value is float
        timestamps = [pd.to_datetime(v[0], unit='s') for v in values]
        metrics = [float(v[1]) for v in values]
        return pd.Series(metrics, index=timestamps)
        
    except Exception as e:
        print(f"Error querying {query}: {e}")
        return pd.Series(dtype=float)

def export_run_data(run_id, start_time_iso, end_time_iso):
    # Convert ISO strings to unix timestamps for Prometheus
    start_ts = datetime.fromisoformat(start_time_iso.replace('Z', '+00:00')).timestamp()
    end_ts = datetime.fromisoformat(end_time_iso.replace('Z', '+00:00')).timestamp()
    
    print(f"=== Exporting data for Run: {run_id} ===")
    print(f"Time range: {start_time_iso} to {end_time_iso}")
    
    df_dict = {}
    for name, query in QUERIES.items():
        print(f"Querying {name}...")
        series = query_prometheus(query, start_ts, end_ts)
        df_dict[name] = series
        
    # Combine into a single DataFrame
    df = pd.DataFrame(df_dict)
    
    # Save to CSV
    os.makedirs(f"results/raw/{run_id}", exist_ok=True)
    csv_path = f"results/raw/{run_id}/metrics.csv"
    df.to_csv(csv_path, index_label="timestamp")
    print(f"Data saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Prometheus metrics for an experiment run")
    parser.add_argument("--run-id", required=True, help="Unique Run ID (e.g. S1-RB-01)")
    parser.add_argument("--start", required=True, help="Start time in ISO format (e.g. 2026-08-20T10:00:00Z)")
    parser.add_argument("--end", required=True, help="End time in ISO format")
    args = parser.parse_args()
    
    export_run_data(args.run_id, args.start, args.end)
