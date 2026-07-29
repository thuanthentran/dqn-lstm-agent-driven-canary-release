import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import subprocess
import time
import random
from collections import deque
import atexit
import sys

# --- IMPORT KUBERNETES NATIVE API ---
from kubernetes import client, config, watch
from core.feature_pipeline import normalize_raw_metrics

EPSILON = 1e-6
CPU_REF = 0.02 * 5
MEM_REF_MB = 128.0 * 5
MAX_RATIO = 5.0
LAPLACE_BUFFER = 2.0
FRONTEND_URL = "http://192.168.142.165:30080"

class C:
    RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, GREY, BOLD, END = '\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[95m', '\033[96m', '\033[97m', '\033[90m', '\033[1m', '\033[0m'

class OnlineCanaryEnv(gym.Env):
    def __init__(self, service_list=None, prometheus_url="http://192.168.142.165:30090", seq_len=30):
        super().__init__()
        self.prometheus_url = prometheus_url
        self.seq_len = seq_len
        self.num_features = 5
        self.service_list = service_list or ['checkoutservice']
        self.current_service = None
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.seq_len, self.num_features), dtype=np.float32)
        self.history = deque(maxlen=self.seq_len)
        self.current_step = 0
        self.max_steps = 20  
        self.locust_process = None
        
        atexit.register(self._stop_locust)
        
        print(f"{C.CYAN}🔌 Đang kết nối trực tiếp đến K3s API Server...{C.END}")
        try: config.load_kube_config()
        except: sys.exit(1)
            
        self.custom_api = client.CustomObjectsApi()
        self.core_api = client.CoreV1Api() # Tích hợp thêm Core API để nội soi sâu
        self.argo_group, self.argo_version, self.argo_plural = "argoproj.io", "v1alpha1", "rollouts"

    def _run_cmd(self, cmd, ignore_error=False):
        try:
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0 and not ignore_error:
                print(f"{C.RED}❌ LỖI LỆNH: {' '.join(cmd)}{C.END}")
                print(f"{C.RED}Chi tiết: {res.stderr}{C.END}")
                sys.exit(1)
            return res
        except FileNotFoundError as e:
            if not ignore_error:
                print(f"{C.RED}❌ KHÔNG TÌM THẤY LỆNH: {cmd[0]}{C.END}")
                print(f"{C.RED}Lỗi hệ thống: {e}{C.END}")
                print(f"{C.RED}Vui lòng cài đặt công cụ này hoặc thêm vào biến môi trường PATH!{C.END}")
                sys.exit(1)
            return None
        except Exception as e:
            if not ignore_error:
                print(f"{C.RED}❌ LỖI HỆ THỐNG KHI CHẠY: {' '.join(cmd)}{C.END}\n{C.RED}Chi tiết: {e}{C.END}")
                sys.exit(1)
            return None

    # ================= K8S DEEP DEBUGGER =================
    def _debug_cluster_health(self):
        print(f"{C.GREY}   🔍 [K8s Debug] Quét sâu tình trạng sinh tử của các Pods...{C.END}")
        try:
            pods = self.core_api.list_namespaced_pod("default", label_selector=f"app={self.current_service}")
            for pod in pods.items:
                phase = pod.status.phase
                ready = "False"
                restarts = 0
                if pod.status.container_statuses:
                    restarts = pod.status.container_statuses[0].restart_count
                    ready = str(pod.status.container_statuses[0].ready)
                
                color = C.GREEN if ready == "True" else C.RED
                print(f"       │ Pod: {pod.metadata.name[-14:]} | Phase: {phase} | Ready: {color}{ready}{C.END} | Restarts: {restarts}")
        except Exception as e:
            print(f"{C.RED}   ❌ [K8s Debug] Lỗi khi lấy thông tin Pod: {e}{C.END}")
    # =====================================================

    def _start_locust(self):
        import sys
        if self.locust_process is None:
            print(f"🔥 {C.BOLD}{C.RED}[Hỏa lực] Đang khai hỏa Locust bắn thẳng vào K3s ({FRONTEND_URL})...{C.END}")
            cmd = [sys.executable, "-m", "locust", "-f", "loadgenerator/locustfile.py", "--headless", "-u", "50", "-r", "10", "-H", FRONTEND_URL]
            self.locust_log_file = open("locust_debug.log", "w", encoding="utf-8")
            try:
                self.locust_process = subprocess.Popen(cmd, stdout=self.locust_log_file, stderr=subprocess.STDOUT)
                time.sleep(2)
                if self.locust_process.poll() is not None:
                    print(f"{C.BOLD}{C.RED}   ❌ [LỖI HỎA LỰC] Tiến trình Locust đã CHẾT NGAY LẬP TỨC!{C.END}")
                    sys.exit(1)
            except FileNotFoundError as e:
                print(f"{C.BOLD}{C.RED}❌ KHÔNG TÌM THẤY LỆNH: locust{C.END}")
                print(f"{C.RED}Chi tiết: {e}{C.END}")
                print(f"{C.RED}Vui lòng cài đặt locust (pip install locust) hoặc thêm thư mục chứa pip vào PATH!{C.END}")
                sys.exit(1)
            except Exception as e:
                print(f"{C.BOLD}{C.RED}❌ LỖI HỆ THỐNG KHI KHỞI ĐỘNG LOCUST: {e}{C.END}")
                sys.exit(1)

    def _stop_locust(self):
        if self.locust_process is not None:
            print(f"{C.GREY}   🚿 [Hỏa lực] Ngừng bắn Locust. Đóng van traffic!{C.END}")
            self.locust_process.terminate()
            self.locust_process.wait()
            self.locust_process = None
            if hasattr(self, 'locust_log_file') and self.locust_log_file:
                self.locust_log_file.close()

    def _get_live_weight_pct_api(self):
        try:
            ro = self.custom_api.get_namespaced_custom_object(self.argo_group, self.argo_version, "default", self.argo_plural, self.current_service)
            return float(ro.get('status', {}).get('canary', {}).get('weights', {}).get('canary', {}).get('weight', 0.0))
        except: return 0.0

    def _get_rollout_hashes(self, svc_name):
        try:
            ro = self.custom_api.get_namespaced_custom_object(self.argo_group, self.argo_version, "default", self.argo_plural, svc_name)
            stable_hash = ro.get("status", {}).get("stableRS", "")
            current_hash = ro.get("status", {}).get("currentPodHash", "")
            if stable_hash and current_hash:
                return f"{svc_name}-{stable_hash}", f"{svc_name}-{current_hash}"
        except: pass
        return f"{svc_name}-stable", f"{svc_name}-canary"

    def _wait_for_weight_change_api(self, old_weight):
        print(f"{C.GREY}   🔄 [K8s API] Lắng nghe Event, đợi traffic rẽ (cũ: {old_weight}%)...{C.END}")
        w = watch.Watch()
        try:
            for event in w.stream(self.custom_api.list_namespaced_custom_object, self.argo_group, self.argo_version, "default", self.argo_plural, timeout_seconds=45):
                ro = event.get('object', {})
                if ro.get('metadata', {}).get('name') != self.current_service: continue
                try: curr_weight = float(ro.get('status', {}).get('canary', {}).get('weights', {}).get('canary', {}).get('weight', 0.0))
                except: curr_weight = 0.0
                if curr_weight != old_weight:
                    print(f"{C.GREEN}   ✅ [K8s API] Ghi nhận Traffic chạm mốc {curr_weight}%!{C.END}")
                    return curr_weight
        finally: w.stop()
        return old_weight

    def _apply_fault_to_rollout_api(self, svc_name, fault_value):
        ro = self.custom_api.get_namespaced_custom_object(self.argo_group, self.argo_version, "default", self.argo_plural, svc_name)
        containers = ro.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        for c in containers:
            if c.get("name") in [svc_name, "server"]: 
                env_list = c.get("env", [])
                found = False
                for e in env_list:
                    if e.get("name") == "FAULT_SCENARIO":
                        e["value"] = fault_value; found = True; break
                if not found: env_list.append({"name": "FAULT_SCENARIO", "value": fault_value})
                c["env"] = env_list
                break
        patch_body = {"spec": {"template": {"metadata": {"annotations": {"rl-train-tick": str(int(time.time()))}}, "spec": {"containers": containers}}}}
        self.custom_api.patch_namespaced_custom_object(self.argo_group, self.argo_version, "default", self.argo_plural, svc_name, patch_body)

    def _wait_for_prometheus(self, seconds=35, phase_name="WARM-UP"):
        print(f"{C.GREY}   ⏳ [Prometheus] Đang đo đạc metrics liên tục trong {seconds}s...{C.END}")
        for i in range(1, seconds + 1):
            start_time = time.time()
            raw = self._get_metrics_from_prometheus()
            lat_color = C.RED if raw['canary_lat'] > 1000.0 else (C.YELLOW if raw['canary_lat'] > 200.0 else C.GREEN)
            err_color = C.RED if raw['canary_err'] > 0.05 else (C.YELLOW if raw['canary_err'] > 0.00 else C.GREEN)
            print(f"       {C.CYAN}│ {phase_name} [{i:02d}/{seconds:02d}s]{C.END} "
                  f"Traffic: {raw['weight_pct']:5.1f}% | Err: {err_color}{raw['canary_err']*100:6.2f}%{C.END} | "
                  f"Lat: {lat_color}{raw['canary_lat']:7.1f}ms{C.END} | CPU: {raw['cpu_cores']:6.4f} | RAM: {raw['ram_mb']:6.1f}MB")
            elapsed = time.time() - start_time
            if elapsed < 1.0: time.sleep(1.0 - elapsed)

    def _get_metrics_from_prometheus(self):
        queries = {
            "traffic": f'sum by (k8s_pod_name) (rate(rpc_server_call_duration_seconds_count{{rpc_system_name="grpc", rpc_method="/hipstershop.CheckoutService/PlaceOrder", k8s_pod_name=~"{self.current_service}.*"}}[1m]))',
            "http_errors": f'sum by (k8s_pod_name) (rate(rpc_server_call_duration_seconds_count{{rpc_system_name="grpc", rpc_method="/hipstershop.CheckoutService/PlaceOrder", rpc_response_status_code!="OK", k8s_pod_name=~"{self.current_service}.*"}}[1m]))',
            "latency": f'histogram_quantile(0.95, sum(rate(rpc_server_call_duration_seconds_bucket{{rpc_system_name="grpc", rpc_method="/hipstershop.CheckoutService/PlaceOrder", k8s_pod_name=~"{self.current_service}.*"}}[1m])) by (le, k8s_pod_name)) * 1000',
            "cpu": f'sum(rate(container_cpu_usage_seconds_total{{namespace="default", pod=~"{self.current_service}.*", container!="POD", container!=""}}[30s]))',
            "ram": f'sum(container_memory_working_set_bytes{{namespace="default", pod=~"{self.current_service}.*", container!="POD", container!=""}})'
        }
        def safe_query(query):
            try:
                res = requests.get(f"{self.prometheus_url}/api/v1/query", params={"query": query}, timeout=3).json()
                if res.get('status') == 'success' and res.get('data', {}).get('result'): return res['data']['result']
            except: pass
            return []

        raw_traffic, raw_http, raw_latency, raw_cpu, raw_ram = map(safe_query, [queries["traffic"], queries["http_errors"], queries["latency"], queries["cpu"], queries["ram"]])

        traffic_dict = {item['metric'].get('k8s_pod_name', 'unknown'): float(item['value'][1]) for item in raw_traffic}
        http_dict = {item['metric'].get('k8s_pod_name', 'unknown'): float(item['value'][1]) for item in raw_http}
        latency_dict = {item['metric'].get('k8s_pod_name', 'unknown'): float(item['value'][1]) for item in raw_latency if str(item['value'][1]) != 'NaN'}

        def aggregate_by_rs(data_dict):
            rs_data = {}
            for pod, val in data_dict.items():
                # Pod names format: checkoutservice-5bb66dcdd9-2lhn6 -> RS is checkoutservice-5bb66dcdd9
                rs_name = "-".join(pod.split("-")[:-1]) if "-" in pod else pod
                rs_data[rs_name] = rs_data.get(rs_name, 0.0) + val
            return rs_data

        def max_latency_by_rs(data_dict):
            rs_data = {}
            for pod, val in data_dict.items():
                rs_name = "-".join(pod.split("-")[:-1]) if "-" in pod else pod
                rs_data[rs_name] = max(rs_data.get(rs_name, 0.0), val)
            return rs_data

        traffic_rs = aggregate_by_rs(traffic_dict)
        http_rs = aggregate_by_rs(http_dict)
        latency_rs = max_latency_by_rs(latency_dict)

        error_rates = {}
        for svc, total_req in traffic_rs.items():
            err_req = http_rs.get(svc, 0.0)
            error_rates[svc] = err_req / (total_req + LAPLACE_BUFFER) if total_req > 0 else 0.0

        stable_rs_name, canary_rs_name = self._get_rollout_hashes(self.current_service)
        
        canary_err = error_rates.get(canary_rs_name, 0.0)
        canary_lat = latency_rs.get(canary_rs_name, 0.0)
        
        if stable_rs_name == canary_rs_name:
            canary_err = error_rates.get(stable_rs_name, 0.0)
            canary_lat = latency_rs.get(stable_rs_name, 0.0)

        stable_err = error_rates.get(stable_rs_name, 0.0)
        stable_lat = latency_rs.get(stable_rs_name, 0.0)
        total_rps = sum(traffic_rs.values()) / 60.0 # Approximate RPS
        
        return {
            "e_canary": canary_err, 
            "e_stable": stable_err, 
            "l_canary": canary_lat,
            "l_stable": stable_lat,
            "canary_err": canary_err, # Keep for dashboard print
            "stable_err": stable_err, # Keep for dashboard print
            "canary_lat": canary_lat, # Keep for dashboard print
            "cpu": float(raw_cpu[0]['value'][1]) if raw_cpu else 0.0,
            "cpu_cores": float(raw_cpu[0]['value'][1]) if raw_cpu else 0.0, # Keep for dashboard print
            "mem_mb": (float(raw_ram[0]['value'][1]) if raw_ram else 0.0) / (1024 * 1024), 
            "ram_mb": (float(raw_ram[0]['value'][1]) if raw_ram else 0.0) / (1024 * 1024), # Keep for dashboard print
            "rps": total_rps,
            "weight_pct": self._get_live_weight_pct_api()
        }

    def _process_and_normalize_features(self, raw):
        norm = normalize_raw_metrics(raw)
        return np.array([
            norm["cpu_n"], 
            norm["mem_n"], 
            norm["l_ratio_n"], 
            norm["e_ratio_n"], 
            norm["weight_n"]
        ], dtype=np.float32)

    def _get_obs(self): return np.stack(list(self.history), axis=0).astype(np.float32)

    def _print_dashboard(self, title, action_str, raw, norm):
        print(f"\n{C.CYAN}╭─────────────────────────────────────────────────────────────╮{C.END}")
        print(f"{C.CYAN}│{C.END} {C.BOLD}🤖 {title}{C.END} | Hành động: {action_str}  ")
        print(f"{C.CYAN}├─────────────────────────────────────────────────────────────┤{C.END}")
        print(f"{C.CYAN}│{C.END} {C.BLUE}📊 Thực tế:{C.END} Err_Canary: {raw['canary_err']*100:6.2f}% | Latency: {raw['canary_lat']:7.1f}ms   ")
        print(f"{C.CYAN}│{C.END}             Err_Stable: {raw['stable_err']*100:6.2f}% | Traffic: {raw['weight_pct']:7.1f}%   ")
        print(f"{C.CYAN}│{C.END}             CPU_Cores : {raw['cpu_cores']:6.4f}  | RAM_Use: {raw['ram_mb']:7.1f}MB   ")
        print(f"{C.CYAN}│{C.END} {C.YELLOW}🧠 Agent  :{C.END} CPU:{norm[0]:.2f} RAM:{norm[1]:.2f} Lat:{norm[2]:.2f} Err:{norm[3]:.2f} ")
        print(f"{C.CYAN}╰─────────────────────────────────────────────────────────────╯{C.END}")

    def step(self, action):
        # 1. Đánh giá trạng thái môi trường TRƯỚC KHI thực hiện action (để tránh lỗi mất metrics khi ABORT)
        raw_before = self._get_metrics_from_prometheus()
        e_ratio_before = raw_before["canary_err"] / max(raw_before["stable_err"], EPSILON)
        l_ratio_before = raw_before["canary_lat"] / max(50.0, EPSILON) 
        is_high_error_before = (e_ratio_before > 2.0) and (raw_before["canary_err"] > 0.05)
        is_high_latency_before = raw_before["canary_lat"] > 1000.0
        was_anomalous = is_high_error_before or is_high_latency_before

        old_weight = self._get_live_weight_pct_api()
        
        act_str = f"{C.GREEN}🟢 PROMOTE [ID: 1]{C.END}" if action == 1 else (f"{C.RED}🔴 ABORT [ID: 2]{C.END}" if action == 2 else f"{C.YELLOW}🟡 HOLD [ID: 0]{C.END}")
        print(f"\n{C.BOLD}{C.MAGENTA}⚡ [Agent Quyết Định] Dựa vào metrics hiện tại, Agent chọn: {act_str}{C.END} (Bắt đầu thực thi...)")
        
        if action == 1: 
            self._run_cmd(["argo-rollouts", "promote", self.current_service, "-n", "default"])
            self._wait_for_weight_change_api(old_weight)
        elif action == 2: 
            self._run_cmd(["argo-rollouts", "abort", self.current_service, "-n", "default"])
            self._wait_for_weight_change_api(old_weight)
        
        self._debug_cluster_health() # Nội soi trạng thái Pods trước khi quét metrics
        
        if action in [1, 2]:
            self._wait_for_prometheus(25, f"STEP {self.current_step+1:02d}") 
            
        raw = self._get_metrics_from_prometheus()
        norm_channels = self._process_and_normalize_features(raw)
        
        act_str = f"{C.GREEN}🟢 PROMOTE [ID: 1]{C.END}" if action == 1 else (f"{C.RED}🔴 ABORT [ID: 2]{C.END}" if action == 2 else f"{C.YELLOW}🟡 HOLD [ID: 0]{C.END}")
        self._print_dashboard(f"STEP {self.current_step+1:02d}", act_str, raw, norm_channels)
        print(f"   👁️ [Góc nhìn Agent] Dữ liệu truyền vào não bộ (Normalized State): {norm_channels}")
        
        self.history.append(norm_channels)
        obs = self._get_obs()

        # SỬA LỖI LOGIC: Kiểm tra Rollout Phase thay vì Weight_pct
        try:
            ro = self.custom_api.get_namespaced_custom_object(self.argo_group, self.argo_version, "default", self.argo_plural, self.current_service)
            phase = ro.get('status', {}).get('phase', '')
        except:
            phase = 'Unknown'
            
        reward = 0.0
        terminated = False

        if was_anomalous:
            print(f"{C.BOLD}{C.RED}   🚨 BÁO ĐỘNG LỖI (Tại thời điểm ra quyết định)! E_Ratio: {e_ratio_before:.2f} | L_Ratio: {l_ratio_before:.2f}{C.END}")
            if action == 1: reward -= 15.0; terminated = True 
            elif action == 2: reward += 10.0; terminated = True
            elif action == 0: reward -= 0.5 
        else: 
            if action == 1: 
                reward += 1.0
                if phase == "Healthy": # Kích hoạt khi Rollout chạy xong 100%
                    print(f"{C.BOLD}{C.GREEN}   🏁 Rollout đạt 100% an toàn (Phase: Healthy)!{C.END}")
                    reward += 20.0; terminated = True 
            elif action == 2: reward -= 10.0; terminated = True
            elif action == 0: reward -= 0.1 

        self.current_step += 1
        if self.current_step >= self.max_steps and not terminated:
            terminated = True; reward -= 5.0

        if terminated:
            rew_color = C.GREEN if reward > 0 else C.RED
            print(f"\n{C.BOLD}🏁 TẬP HUẤN LUYỆN KẾT THÚC | TỔNG ĐIỂM: {rew_color}{reward:+.2f}{C.END}")

        return obs, float(reward), bool(terminated), False, {}

    def _graceful_reset_baseline(self, svc_name):
        self._stop_locust() 
        print(f"{C.YELLOW}   🧹 Trả môi trường về STABLE (none)...{C.END}")
        self._run_cmd(["argo-rollouts", "abort", svc_name, "-n", "default"])
        self._apply_fault_to_rollout_api(svc_name, "none")
        time.sleep(3)
        
        print(f"{C.YELLOW}   ⏩ Ép Promote Full 100% để dọn dẹp Canary cũ...{C.END}")
        self._run_cmd(["argo-rollouts", "promote", svc_name, "--full", "-n", "default"])
        
        print(f"{C.GREY}   ⏳ [K8s API] Lắng nghe Event cho đến khi Rollout đạt trạng thái Healthy...{C.END}")
        w = watch.Watch()
        try:
            for event in w.stream(self.custom_api.list_namespaced_custom_object, self.argo_group, self.argo_version, "default", self.argo_plural, timeout_seconds=90):
                ro = event.get('object', {})
                if ro.get('metadata', {}).get('name') != svc_name: continue
                if ro.get('status', {}).get('phase', '') == 'Healthy':
                    print(f"{C.GREEN}   ✅ [K8s API] Cụm 5 Pods STABLE đã sẵn sàng 100%!{C.END}")
                    break
        finally: w.stop()

    def _wait_for_canary_pods_ready(self, svc_name):
        print(f"{C.GREY}   ⏳ [K8s API] Chờ Canary Pods khởi động và Ready...{C.END}")
        
        # 1. Đợi cho tới khi K8s sinh ra Canary Hash mới (khác Stable Hash)
        start_wait_hash = time.time()
        stable_hash, canary_hash = self._get_rollout_hashes(svc_name)
        while stable_hash == canary_hash and (time.time() - start_wait_hash < 30):
            time.sleep(1)
            stable_hash, canary_hash = self._get_rollout_hashes(svc_name)
            
        if stable_hash == canary_hash:
            print(f"{C.RED}   ⚠️ [K8s API] Không thấy Canary mới xuất hiện sau 30s! Bỏ qua...{C.END}")
            return
            
        # 2. Chờ cho các Pods thuộc Canary Hash đó Ready
        start_time = time.time()
        while time.time() - start_time < 90:
            try:
                pods = self.core_api.list_namespaced_pod("default", label_selector=f"app={svc_name}").items
                canary_ready_count = 0
                canary_total_count = 0
                for pod in pods:
                    owner = pod.metadata.owner_references[0].name if pod.metadata.owner_references else ""
                    if owner == canary_hash:
                        canary_total_count += 1
                        if pod.status.phase == "Running" and pod.status.conditions:
                            for cond in pod.status.conditions:
                                if cond.type == "Ready" and cond.status == "True":
                                    canary_ready_count += 1
                if canary_total_count > 0 and canary_ready_count == canary_total_count:
                    print(f"{C.GREEN}   ✅ [K8s API] Toàn bộ Canary Pods đã Ready ({canary_ready_count}/{canary_total_count})! Bắt đầu nạp Traffic...{C.END}")
                    return
            except Exception as e: pass
            time.sleep(2)
        print(f"{C.RED}   ⚠️ [K8s API] Hết thời gian chờ Canary Pods (Vẫn tiếp tục)!{C.END}")

    def reset(self, seed=None, options=None):   
        super().reset(seed=seed)
        self.current_step = 0
        self.service_list = ['checkoutservice']
        self.current_service = random.choice(self.service_list)

        print(f"\n{C.BOLD}{C.MAGENTA}═════════════════════════════════════════════════════════{C.END}")
        print(f"{C.BOLD}{C.MAGENTA} 🔄 BẮT ĐẦU EPISODE MỚI CHO: {self.current_service.upper()}{C.END}")
        print(f"{C.BOLD}{C.MAGENTA}═════════════════════════════════════════════════════════{C.END}")
        
        self._graceful_reset_baseline(self.current_service)
        
        faults = ["none", "high_latency", "high_error", "combined"]
        chosen_fault = random.choice(faults)
        
        fault_color = C.GREEN if chosen_fault == "none" else C.RED
        print(f"\n{C.BOLD}🚀 BƠM KỊCH BẢN CANARY: {fault_color}[{chosen_fault.upper()}]{C.END}")
        
        self._apply_fault_to_rollout_api(self.current_service, chosen_fault)
        self._wait_for_weight_change_api(100.0)
        self._wait_for_canary_pods_ready(self.current_service)
        
        self._start_locust()
        self._debug_cluster_health() # Nội soi trạng thái mồi

        self._wait_for_prometheus(35, "STEP 00") 

        raw = self._get_metrics_from_prometheus()
        initial_channels = self._process_and_normalize_features(raw)
        
        self._print_dashboard("STEP 00", "Bắt đầu trạng thái mồi (Sandbox K3s)", raw, initial_channels)

        self.history = deque([initial_channels]*self.seq_len, maxlen=self.seq_len)

        return self._get_obs(), {}