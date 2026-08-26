import argparse
import requests
import pandas as pd
from datetime import datetime, timezone
import os
import json

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:30090")

# Define the queries we want to export
QUERIES = {
    # Fix 1 & 2: gRPC error rate (using grpc_response_status) and proper escape for stable service
    "canary_error_rate": 'sum(rate(istio_requests_total{reporter="destination", destination_service=~"checkoutservice-canary.*", grpc_response_status!="0"}[1m])) / sum(rate(istio_requests_total{reporter="destination", destination_service=~"checkoutservice-canary.*"}[1m]))',
    "stable_error_rate": 'sum(rate(istio_requests_total{reporter="destination", destination_service=~"checkoutservice-stable.*", grpc_response_status!="0"}[1m])) / sum(rate(istio_requests_total{reporter="destination", destination_service=~"checkoutservice-stable.*"}[1m]))',
    
    # Latencies (P95)
    "canary_p95_latency": 'histogram_quantile(0.95, sum(rate(istio_request_duration_milliseconds_bucket{reporter="destination", destination_service=~"checkoutservice-canary.*"}[1m])) by (le))',
    "stable_p95_latency": 'histogram_quantile(0.95, sum(rate(istio_request_duration_milliseconds_bucket{reporter="destination", destination_service=~"checkoutservice-stable.*"}[1m])) by (le))',
    
    # Traffic distribution
    "traffic_weight_canary": 'sum(rate(istio_requests_total{reporter="destination", destination_service=~"checkoutservice-canary.*"}[1m])) / sum(rate(istio_requests_total{reporter="destination", destination_service=~"checkoutservice-(canary|stable).*"}[1m]))',
    
    # Fix 3: Resource Efficiency Metrics
    "cpu_canary_cores": 'sum(rate(container_cpu_usage_seconds_total{namespace="msdemo", pod=~"checkoutservice-canary.*", container!="POD", container!=""}[1m]))',
    "mem_canary_mb": 'sum(container_memory_working_set_bytes{namespace="msdemo", pod=~"checkoutservice-canary.*", container!="POD", container!=""}) / (1024 * 1024)',
    "cpu_stable_cores": 'sum(rate(container_cpu_usage_seconds_total{namespace="msdemo", pod=~"checkoutservice-[a-z0-9]+-[a-z0-9]+", pod!~"checkoutservice-canary.*", container!="POD", container!=""}[1m]))',
    "mem_stable_mb": 'sum(container_memory_working_set_bytes{namespace="msdemo", pod=~"checkoutservice-[a-z0-9]+-[a-z0-9]+", pod!~"checkoutservice-canary.*", container!="POD", container!=""}) / (1024 * 1024)'
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
        
        if data['status'] != 'success' or not data.get('data', {}).get('result'):
            return pd.Series(dtype=float)
            
        # Parse the first result (assuming single scalar timeseries per query due to sum())
        values = data['data']['result'][0]['values']
        
        # Convert to pandas Series: index is timestamp, value is float
        timestamps = [pd.to_datetime(v[0], unit='s') for v in values]
        metrics = [float(v[1]) if str(v[1]).lower() != "nan" else 0.0 for v in values]
        return pd.Series(metrics, index=timestamps)
        
    except Exception as e:
        print(f"Error querying {query}: {e}")
        return pd.Series(dtype=float)

def export_run_data(start_time_iso, end_time_iso, out_file):
    # Convert ISO strings to unix timestamps for Prometheus
    start_ts = datetime.fromisoformat(start_time_iso.replace('Z', '+00:00')).timestamp()
    end_ts = datetime.fromisoformat(end_time_iso.replace('Z', '+00:00')).timestamp()
    
    print(f"=== Exporting data ===")
    print(f"Time range: {start_time_iso} to {end_time_iso}")
    
    df_dict = {}
    for name, query in QUERIES.items():
        print(f"Querying {name}...")
        series = query_prometheus(query, start_ts, end_ts)
        df_dict[name] = series
        
    # Combine into a single DataFrame
    df = pd.DataFrame(df_dict)
    
    # Save to CSV
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df.to_csv(out_file, index_label="timestamp")
    print(f"Data saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Prometheus metrics for an experiment run")
    parser.add_argument("--start", required=True, help="Start time in ISO format (e.g. 2026-08-20T10:00:00Z)")
    parser.add_argument("--end", required=True, help="End time in ISO format")
    parser.add_argument("--out", required=True, help="Output CSV file path")
    args = parser.parse_args()
    
    export_run_data(args.start, args.end, args.out)
