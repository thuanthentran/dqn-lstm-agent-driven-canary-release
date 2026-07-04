import time
import requests
import numpy as np
import subprocess
import re
import os

# --- CẤU HÌNH ---
# Địa chỉ Prometheus đang chạy qua NodePort trong cụm của bạn
PROMETHEUS_URL = "http://192.168.142.165:30090" 
TARGET_SERVICE = "checkoutservice"
EPSILON = 1e-6
CPU_REF = 0.02
MEM_REF_MB = 128.0
MAX_RATIO = 5.0

# --- BỘ MÀU SẮC THẨM MỸ ---
class C:
    RED, GREEN, YELLOW, BLUE, CYAN, GREY, BOLD, END = '\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[96m', '\033[90m', '\033[1m', '\033[0m'

def query_prometheus(query):
    """Gửi query PromQL tới Prometheus server"""
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=3)
        if response.status_code == 200:
            result = response.json()
            if result.get('status') == 'success':
                return result['data']['result']
    except Exception as e:
        print(f"{C.RED}Lỗi kết nối Prometheus: {e}{C.END}")
    return []

def get_live_weight_pct(service_name):
    """Lấy trọng số canary hiện tại từ Argo Rollouts[cite: 1]"""
    cmd = f"argo-rollouts get rollout {service_name} -n default"
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if res.returncode != 0:
            print(f"{C.RED}❌ LỖI LỆNH: {cmd}{C.END}\n{C.RED}Chi tiết: {res.stderr}{C.END}")
            sys.exit(1)
        match = re.search(r'ActualWeight:\s+(\d+)', res.stdout)
        if match: return float(match.group(1))
    except Exception as e:
        print(f"{C.RED}❌ KHÔNG THỂ CHẠY LỆNH: {cmd}{C.END}\n{C.RED}Lỗi hệ thống: {e}{C.END}")
        sys.exit(1)
    return 0.0

def get_rollout_hashes(service_name):
    """Lấy tên ReplicaSet (bao gồm hash) của bản stable và canary từ Kubernetes"""
    cmd = f"kubectl get rollout {service_name} -n default -o jsonpath={{.status.stableRS}},{{.status.currentPodHash}}"
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if res.returncode != 0:
            print(f"{C.RED}❌ LỖI LỆNH: {cmd}{C.END}\n{C.RED}Chi tiết: {res.stderr}{C.END}")
            sys.exit(1)
        if res.stdout:
            parts = res.stdout.strip().split(',')
            if len(parts) == 2:
                stable_hash, current_hash = parts
                return f"{service_name}-{stable_hash}", f"{service_name}-{current_hash}"
    except Exception as e:
        print(f"{C.RED}❌ KHÔNG THỂ CHẠY LỆNH: {cmd}{C.END}\n{C.RED}Lỗi hệ thống: {e}{C.END}")
        sys.exit(1)
    return f"{service_name}-stable", f"{service_name}-canary"

def fetch_and_process_metrics():
    # --- ĐỒNG BỘ VỚI HẠ TẦNG eBPF (CILIUM/HUBBLE) ---
    # Thay thế istio_requests_total bằng hubble_http_responses_total
    # Lưu ý: Hubble đo latency bằng giây (seconds), nên cần nhân 1000 để ra ms.
    queries = {
        "traffic": 'sum by (destination_workload) (rate(hubble_http_responses_total{}[30s]))',
        "http_errors": 'sum by (destination_workload) (rate(hubble_http_responses_total{status=~"5.*"}[30s]))',
        # Thay thế metric latency của Istio bằng metric của Hubble
        "latency": 'histogram_quantile(0.95, sum(rate(hubble_http_request_duration_seconds_bucket{}[30s])) by (le, destination_workload)) * 1000',
        
        # Resource metrics giữ nguyên vì dùng standard cAdvisor của Kubernetes[cite: 1]
        "cpu": f'sum(rate(container_cpu_usage_seconds_total{{namespace="default", pod=~"{TARGET_SERVICE}.*", container!="POD", container!=""}}[30s]))',
        "ram": f'sum(container_memory_working_set_bytes{{namespace="default", pod=~"{TARGET_SERVICE}.*", container!="POD", container!=""}})'
    }

    raw_traffic = query_prometheus(queries["traffic"])
    raw_http = query_prometheus(queries["http_errors"])
    raw_latency = query_prometheus(queries["latency"])
    raw_cpu = query_prometheus(queries["cpu"])
    raw_ram = query_prometheus(queries["ram"])

    # Xử lý dữ liệu thô
    traffic_dict = {item['metric'].get('destination_workload', 'unknown'): float(item['value'][1]) for item in raw_traffic}
    http_dict = {item['metric'].get('destination_workload', 'unknown'): float(item['value'][1]) for item in raw_http}
    latency_dict = {item['metric'].get('destination_workload', 'unknown'): float(item['value'][1]) for item in raw_latency if str(item['value'][1]) != 'NaN'}

    cpu_cores = float(raw_cpu[0]['value'][1]) if raw_cpu else 0.0
    ram_mb = (float(raw_ram[0]['value'][1]) if raw_ram else 0.0) / (1024 * 1024)

    # Tính toán Error Rate
    error_rates = {}
    for svc, total_req in traffic_dict.items():
        err_req = http_dict.get(svc, 0.0)
        error_rates[svc] = err_req / total_req if total_req > 0 else 0.0

    # Lấy ReplicaSet Hash từ Argo Rollouts
    stable_rs_name, canary_rs_name = get_rollout_hashes(TARGET_SERVICE)

    # Lấy thông số cho RL Agent[cite: 1]
    canary_err = error_rates.get(canary_rs_name, 0.0)
    canary_lat = latency_dict.get(canary_rs_name, 0.0)
    
    if stable_rs_name == canary_rs_name:
        # Nếu chưa có canary (100% stable), lấy thông số của stable làm chuẩn
        canary_err = error_rates.get(stable_rs_name, 0.0)
        canary_lat = latency_dict.get(stable_rs_name, 0.0)
    
    # Lấy Error Rate của bản Stable làm baseline thay vì trung bình toàn cụm
    stable_err = error_rates.get(stable_rs_name, 0.0)
    
    weight_pct = get_live_weight_pct(TARGET_SERVICE)

    # Tính toán State cho mảng Obs[cite: 1]
    e_ratio = canary_err / max(stable_err, EPSILON)
    e_ratio_clipped = min(e_ratio, MAX_RATIO)
    
    obs_array = np.array([
        min(cpu_cores / CPU_REF, 1.0),
        min(ram_mb / MEM_REF_MB, 1.0),
        min(canary_lat / 1000.0, 5.0), # Ràng buộc Latency (max 5s)
        e_ratio_clipped / MAX_RATIO,
        weight_pct / 100.0
    ], dtype=np.float32)

    # Giao diện Console UI[cite: 1]
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"\n{C.CYAN}╭─────────────────────────────────────────────────────────╮{C.END}")
    print(f"{C.CYAN}│{C.END} {C.BOLD}📡 CILIUM eBPF METRICS | Mục tiêu: {TARGET_SERVICE.upper()} {C.END}".ljust(69) + f"{C.CYAN}│{C.END}")
    print(f"{C.CYAN}├─────────────────────────────────────────────────────────┤{C.END}")
    print(f"{C.CYAN}│{C.END} {C.BLUE}📊 Thực tế:{C.END} Err_Canary: {canary_err*100:5.2f}% | Latency: {canary_lat:7.1f}ms  ".ljust(69) + f"{C.CYAN}│{C.END}")
    print(f"{C.CYAN}│{C.END}             Err_Stable: {stable_err*100:5.2f}% | Traffic: {weight_pct:5.1f}%  ".ljust(69) + f"{C.CYAN}│{C.END}")
    print(f"{C.CYAN}│{C.END}             CPU_Cores : {cpu_cores:5.4f} | RAM_Use: {ram_mb:5.1f}MB  ".ljust(69) + f"{C.CYAN}│{C.END}")
    print(f"{C.CYAN}│{C.END} {C.YELLOW}🧠 Agent  :{C.END} CPU:{obs_array[0]:.2f} RAM:{obs_array[1]:.2f} Lat:{obs_array[2]:.2f} Err:{obs_array[3]:.2f}".ljust(69) + f"{C.CYAN}│{C.END}")
    print(f"{C.CYAN}╰─────────────────────────────────────────────────────────╯{C.END}")

if __name__ == "__main__":
    print(f"{C.GREEN}🚀 BẮT ĐẦU VÒNG LẶP KIỂM TRA METRICS TỪ CILIUM (Ctrl+C để dừng)...{C.END}")
    time.sleep(2)
    while True:
        fetch_and_process_metrics()
        time.sleep(5)