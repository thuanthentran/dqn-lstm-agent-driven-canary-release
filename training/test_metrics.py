import time
import requests
import numpy as np
import subprocess
import re
import os

# --- CẤU HÌNH ---
PROMETHEUS_URL = "http://172.26.52.132:30090" 
TARGET_SERVICE = "checkoutservice"
EPSILON = 1e-6
CPU_REF = 0.02
MEM_REF_MB = 128.0
MAX_RATIO = 5.0

# --- BỘ MÀU SẮC THẨM MỸ ---
class C:
    RED, GREEN, YELLOW, BLUE, CYAN, GREY, BOLD, END = '\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[96m', '\033[90m', '\033[1m', '\033[0m'

def query_prometheus(query):
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=3)
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'success':
                return result['data']['result']
    except Exception:
        pass
    return []

def get_live_weight_pct(service_name):
    try:
        cmd = ["argo-rollouts", "get", "rollout", service_name, "-n", "default"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        match = re.search(r'ActualWeight:\s+(\d+)', res.stdout)
        if match: return float(match.group(1))
    except: pass
    return 0.0

def fetch_and_process_metrics():
    # ĐỒNG BỘ 100% VỚI ONLINE_ENV: Sử dụng [30s]
    queries = {
        "traffic": 'sum by (destination_workload) (rate(istio_requests_total{reporter="destination"}[30s]))',
        "http_errors": 'sum by (destination_workload) (rate(istio_requests_total{reporter="destination", response_code=~"5.*"}[30s]))',
        "grpc_errors": 'sum by (destination_workload) (rate(istio_requests_total{reporter="destination", grpc_response_status!="0", grpc_response_status!=""}[30s]))',
        "latency": 'histogram_quantile(0.95, sum(rate(istio_request_duration_milliseconds_bucket{reporter="destination"}[30s])) by (le, destination_workload))',
        "cpu": f'sum(rate(container_cpu_usage_seconds_total{{namespace="default", pod=~"{TARGET_SERVICE}.*", container!="POD", container!=""}}[30s]))',
        "ram": f'sum(container_memory_working_set_bytes{{namespace="default", pod=~"{TARGET_SERVICE}.*", container!="POD", container!=""}})'
    }

    raw_traffic = query_prometheus(queries["traffic"])
    raw_http = query_prometheus(queries["http_errors"])
    raw_grpc = query_prometheus(queries["grpc_errors"])
    raw_latency = query_prometheus(queries["latency"])
    raw_cpu = query_prometheus(queries["cpu"])
    raw_ram = query_prometheus(queries["ram"])

    traffic_dict = {item['metric'].get('destination_workload', 'unknown'): float(item['value'][1]) for item in raw_traffic}
    http_dict = {item['metric'].get('destination_workload', 'unknown'): float(item['value'][1]) for item in raw_http}
    grpc_dict = {item['metric'].get('destination_workload', 'unknown'): float(item['value'][1]) for item in raw_grpc}
    latency_dict = {item['metric'].get('destination_workload', 'unknown'): float(item['value'][1]) for item in raw_latency if str(item['value'][1]) != 'NaN'}

    cpu_cores = float(raw_cpu[0]['value'][1]) if raw_cpu else 0.0
    ram_mb = (float(raw_ram[0]['value'][1]) if raw_ram else 0.0) / (1024 * 1024)

    error_rates = {}
    for svc, total_req in traffic_dict.items():
        err_req = http_dict.get(svc, 0.0) + grpc_dict.get(svc, 0.0)
        error_rates[svc] = err_req / total_req if total_req > 0 else 0.0

    canary_err = error_rates.get(TARGET_SERVICE, 0.0)
    canary_lat = latency_dict.get(TARGET_SERVICE, 0.0)
    stable_services_err = [rate for svc, rate in error_rates.items() if svc != TARGET_SERVICE]
    stable_err = np.mean(stable_services_err) if stable_services_err else 0.0
    weight_pct = get_live_weight_pct(TARGET_SERVICE)

    e_ratio = canary_err / max(stable_err, EPSILON)
    e_ratio_clipped = min(e_ratio, MAX_RATIO)
    
    obs_array = np.array([
        min(cpu_cores / CPU_REF, 1.0),
        min(ram_mb / MEM_REF_MB, 1.0),
        min(canary_lat / 1000.0, 5.0),
        e_ratio_clipped / MAX_RATIO,
        weight_pct / 100.0
    ], dtype=np.float32)

    os.system('cls' if os.name == 'nt' else 'clear') # Xóa màn hình cho gọn
    print(f"\n{C.CYAN}╭─────────────────────────────────────────────────────────╮{C.END}")
    print(f"{C.CYAN}│{C.END} {C.BOLD}📡 LIVE METRICS TOOL | Mục tiêu: {TARGET_SERVICE.upper()} {C.END}".ljust(69) + f"{C.CYAN}│{C.END}")
    print(f"{C.CYAN}├─────────────────────────────────────────────────────────┤{C.END}")
    print(f"{C.CYAN}│{C.END} {C.BLUE}📊 Thực tế:{C.END} Err_Canary: {canary_err*100:5.2f}% | Latency: {canary_lat:7.1f}ms  ".ljust(69) + f"{C.CYAN}│{C.END}")
    print(f"{C.CYAN}│{C.END}             Err_Stable: {stable_err*100:5.2f}% | Traffic: {weight_pct:5.1f}%  ".ljust(69) + f"{C.CYAN}│{C.END}")
    print(f"{C.CYAN}│{C.END}             CPU_Cores : {cpu_cores:5.4f} | RAM_Use: {ram_mb:5.1f}MB  ".ljust(69) + f"{C.CYAN}│{C.END}")
    print(f"{C.CYAN}│{C.END} {C.YELLOW}🧠 Agent  :{C.END} CPU:{obs_array[0]:.2f} RAM:{obs_array[1]:.2f} Lat:{obs_array[2]:.2f} Err:{obs_array[3]:.2f}".ljust(69) + f"{C.CYAN}│{C.END}")
    print(f"{C.CYAN}╰─────────────────────────────────────────────────────────╯{C.END}")

if __name__ == "__main__":
    print(f"{C.GREEN}🚀 BẮT ĐẦU VÒNG LẶP KIỂM TRA METRICS (Bấm Ctrl+C để dừng)...{C.END}")
    time.sleep(2)
    while True:
        fetch_and_process_metrics()
        time.sleep(10)