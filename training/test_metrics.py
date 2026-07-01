import time
import requests
import numpy as np
import subprocess
import re

# --- CẤU HÌNH ---
# Điền chính xác URL Prometheus mà K3s của bạn đang expose
PROMETHEUS_URL = "http://172.26.52.132:30090" 
TARGET_SERVICE = "checkoutservice"
EPSILON = 1e-6

def query_prometheus(query):
    """Gửi query đến Prometheus và trả về JSON data"""
    try:
        response = requests.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query}, timeout=5)
        if response.status_code == 200:
            result = response.json()
            if result['status'] == 'success':
                return result['data']['result']
    except Exception as e:
        print(f"  [!] Lỗi kết nối Prometheus: {e}")
    return []

def get_live_weight_pct(service_name):
    """Lấy % traffic hiện tại từ Argo Rollouts CLI"""
    try:
        cmd = ["argo-rollouts", "get", "rollout", service_name, "-n", "default"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        match = re.search(r'ActualWeight:\s+(\d+)', res.stdout)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return 0.0

def fetch_and_process_metrics():
    print(f"\n{'-'*50}")
    print(f"📡 ĐANG THU THẬP METRICS CHO: {TARGET_SERVICE.upper()}")
    
    # Hằng số chuẩn hóa (Khớp với core/online_env.py)
    CPU_REF = 0.02
    MEM_REF_MB = 128.0
    
    # 1. Định nghĩa các câu Query bao gồm cả CPU và RAM
    queries = {
        "traffic": 'sum by (destination_workload) (rate(istio_requests_total{reporter="destination"}[1m]))',
        "http_errors": 'sum by (destination_workload) (rate(istio_requests_total{reporter="destination", response_code=~"5.*"}[1m]))',
        "grpc_errors": 'sum by (destination_workload) (rate(istio_requests_total{reporter="destination", grpc_response_status!="0", grpc_response_status!=""}[1m]))',
        "latency": 'histogram_quantile(0.95, sum(rate(istio_request_duration_milliseconds_bucket{reporter="destination"}[1m])) by (le, destination_workload))',
        # CPU: Tổng lượng CPU usage của các pod thuộc checkoutservice (đơn vị: core)
        "cpu": f'sum(rate(container_cpu_usage_seconds_total{{namespace="default", pod=~"{TARGET_SERVICE}.*", container!="POD", container!=""}}[1m]))',
        # RAM: Tổng lượng Working Set Memory (đơn vị: byte)
        "ram": f'sum(container_memory_working_set_bytes{{namespace="default", pod=~"{TARGET_SERVICE}.*", container!="POD", container!=""}})'
    }

    # 2. Thực thi Query
    raw_traffic = query_prometheus(queries["traffic"])
    raw_http = query_prometheus(queries["http_errors"])
    raw_grpc = query_prometheus(queries["grpc_errors"])
    raw_latency = query_prometheus(queries["latency"])
    raw_cpu = query_prometheus(queries["cpu"])
    raw_ram = query_prometheus(queries["ram"])

    # 3. Chuyển đổi dữ liệu
    traffic_dict = {item['metric'].get('destination_workload', 'unknown'): float(item['value'][1]) for item in raw_traffic}
    http_dict = {item['metric'].get('destination_workload', 'unknown'): float(item['value'][1]) for item in raw_http}
    grpc_dict = {item['metric'].get('destination_workload', 'unknown'): float(item['value'][1]) for item in raw_grpc}
    latency_dict = {item['metric'].get('destination_workload', 'unknown'): float(item['value'][1]) for item in raw_latency if str(item['value'][1]) != 'NaN'}

    # Xử lý CPU và RAM
    cpu_cores = float(raw_cpu[0]['value'][1]) if raw_cpu else 0.0
    ram_bytes = float(raw_ram[0]['value'][1]) if raw_ram else 0.0
    ram_mb = ram_bytes / (1024 * 1024) # Đổi sang Megabytes

    # 4. Tính toán Error Rate
    error_rates = {}
    for svc, total_req in traffic_dict.items():
        err_req = http_dict.get(svc, 0.0) + grpc_dict.get(svc, 0.0)
        error_rates[svc] = err_req / total_req if total_req > 0 else 0.0

    # 5. Phân tích Canary vs Stable
    canary_err = error_rates.get(TARGET_SERVICE, 0.0)
    canary_lat = latency_dict.get(TARGET_SERVICE, 0.0)
    
    stable_services_err = [rate for svc, rate in error_rates.items() if svc != TARGET_SERVICE]
    stable_err = np.mean(stable_services_err) if stable_services_err else 0.0

    weight_pct = get_live_weight_pct(TARGET_SERVICE)

    print(f"📊 DỮ LIỆU THÔ:")
    print(f"   - Error Rate: {canary_err:.4f} | Stable Err: {stable_err:.4f}")
    print(f"   - Latency (p95): {canary_lat:.2f} ms")
    print(f"   - Resource: CPU = {cpu_cores:.4f} cores | RAM = {ram_mb:.2f} MB")
    print(f"   - Argo Weight: {weight_pct}%")

    # 6. Chuẩn hóa
    e_ratio = canary_err / max(stable_err, EPSILON)
    e_ratio_clipped = min(e_ratio, 5.0)
    
    cpu_norm = min(cpu_cores / CPU_REF, 1.0)
    ram_norm = min(ram_mb / MEM_REF_MB, 1.0)

    obs_array = np.array([
        cpu_norm,
        ram_norm,
        min(canary_lat / 1000.0, 5.0), # Tránh vỡ form nếu Latency vọt lên 10000ms
        e_ratio_clipped / 5.0,
        weight_pct / 100.0
    ], dtype=np.float32)

    print(f"\n🧠 DỮ LIỆU ĐÃ CHUẨN HÓA CHO AGENT:")
    print(f"   [ CPU, RAM, Latency(s), E_Ratio(norm), Traffic(norm) ]")
    print(f"   {obs_array}")

if __name__ == "__main__":
    print("🚀 BẮT ĐẦU VÒNG LẶP KIỂM TRA METRICS (Bấm Ctrl+C để dừng)...")
    while True:
        fetch_and_process_metrics()
        time.sleep(10) # Lấy mẫu mỗi 10 giây