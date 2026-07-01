import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import subprocess
import time
import random
from collections import deque
import re
import os
import sys

# --- IMPORT KUBERNETES NATIVE API ---
from kubernetes import client, config, watch

EPSILON = 1e-6
CPU_REF = 0.02
MEM_REF_MB = 128.0
MAX_RATIO = 5.0

class C:
    RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, GREY, BOLD, END = '\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[95m', '\033[96m', '\033[97m', '\033[90m', '\033[1m', '\033[0m'

class OnlineCanaryEnv(gym.Env):
    def __init__(self, service_list=None, prometheus_url="http://172.26.52.132:30090", seq_len=30):
        super().__init__()
        self.prometheus_url = prometheus_url
        self.seq_len = seq_len
        self.num_features = 5
        self.service_list = service_list or ['checkoutservice']
        self.current_service = None
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.num_features, self.seq_len), dtype=np.float32)
        self.history = deque(maxlen=self.seq_len)
        self.current_step = 0
        self.max_steps = 20  
        
        # --- KHỞI TẠO KUBERNETES API CLIENT ---
        print(f"{C.CYAN}🔌 Đang kết nối trực tiếp đến K3s API Server...{C.END}")
        try:
            config.load_kube_config()
        except:
            print(f"{C.RED}❌ Lỗi: Không tìm thấy file Kubeconfig.{C.END}")
            sys.exit(1)
            
        self.custom_api = client.CustomObjectsApi()
        self.apps_api = client.AppsV1Api()
        self.argo_group = "argoproj.io"
        self.argo_version = "v1alpha1"
        self.argo_plural = "rollouts"

    def _run_cmd(self, cmd, ignore_error=False):
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 and not ignore_error:
            print(f"\n{C.BOLD}{C.RED}❌ [FATAL ERROR] Lệnh hệ thống thất bại!{C.END}")
            print(f"   {C.RED}Lệnh: {' '.join(cmd)}{C.END}")
            print(f"   {C.RED}STDERR: {res.stderr.strip()}{C.END}")
            sys.exit(1)
        return res

    def _get_live_weight_pct_api(self):
        """Đọc thẳng JSON từ K3s API Memory, bỏ qua CLI, độ trễ 0ms"""
        try:
            ro = self.custom_api.get_namespaced_custom_object(self.argo_group, self.argo_version, "default", self.current_service)
            return float(ro.get('status', {}).get('canary', {}).get('weights', {}).get('canary', {}).get('weight', 0.0))
        except: return 0.0

    def _wait_for_weight_change_api(self, old_weight):
        """[HƯỚNG SỰ KIỆN] Mở luồng WebSocket nghe ngóng sự thay đổi của Rollout"""
        print(f"{C.GREY}   🔄 [K8s API] Mở luồng Event Stream, đợi traffic rẽ (cũ: {old_weight}%)...{C.END}")
        w = watch.Watch()
        try:
            for event in w.stream(self.custom_api.list_namespaced_custom_object,
                                  self.argo_group, self.argo_version, "default", self.argo_plural,
                                  timeout_seconds=45):
                ro = event.get('object', {})
                if ro.get('metadata', {}).get('name') != self.current_service:
                    continue
                
                try:
                    curr_weight = float(ro.get('status', {}).get('canary', {}).get('weights', {}).get('canary', {}).get('weight', 0.0))
                except: curr_weight = 0.0
                
                if curr_weight != old_weight:
                    print(f"{C.GREEN}   ✅ [K8s API] K3s PING: Traffic đã chạm mốc {curr_weight}%!{C.END}")
                    return curr_weight
        finally:
            w.stop()
        return old_weight

    def _apply_fault_to_rollout_api(self, svc_name, fault_value):
        """Bơm lỗi thẳng vào cấu trúc dữ liệu của Kubernetes (Không dùng temp file)"""
        ro = self.custom_api.get_namespaced_custom_object(self.argo_group, self.argo_version, "default", svc_name)
        containers = ro.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        
        for c in containers:
            if c.get("name") in [svc_name, "server"]: 
                env_list = c.get("env", [])
                found = False
                for e in env_list:
                    if e.get("name") == "FAULT_SCENARIO":
                        e["value"] = fault_value
                        found = True
                        break
                if not found: env_list.append({"name": "FAULT_SCENARIO", "value": fault_value})
                c["env"] = env_list
                break

        patch_body = {
            "spec": {"template": {"metadata": {"annotations": {"rl-train-tick": str(int(time.time()))}}, "spec": {"containers": containers}}}
        }
        self.custom_api.patch_namespaced_custom_object(self.argo_group, self.argo_version, "default", self.argo_plural, svc_name, patch_body)

    def _restart_loadgenerator_api(self):
        """Khởi động lại LoadGen bằng K8s API để xả hàng đợi kẹt xe ngay lập tức"""
        patch_body = {"spec": {"template": {"metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": str(time.time())}}}}}
        try:
            self.apps_api.patch_namespaced_deployment("loadgenerator", "default", patch_body)
        except: pass

    def _wait_for_prometheus(self, seconds=35):
        """Hàm duy nhất cần dùng Sleep vì toán học Prometheus cần thu thập đủ Time-series"""
        print(f"{C.GREY}   ⏳ [Prometheus] Chờ {seconds}s để cỗ máy Time-Series tính toán mảng [30s]...{C.END}")
        for i in range(seconds, 0, -5):
            print(f"{C.GREY}       ... còn {i}s{C.END}")
            time.sleep(5)

    def _get_metrics_from_prometheus(self):
        queries = {
            "traffic": 'sum by (destination_workload) (rate(istio_requests_total{reporter="destination"}[30s]))',
            "http_errors": 'sum by (destination_workload) (rate(istio_requests_total{reporter="destination", response_code=~"5.*"}[30s]))',
            "grpc_errors": 'sum by (destination_workload) (rate(istio_requests_total{reporter="destination", grpc_response_status!="0", grpc_response_status!=""}[30s]))',
            "latency": 'histogram_quantile(0.95, sum(rate(istio_request_duration_milliseconds_bucket{reporter="destination"}[30s])) by (le, destination_workload))',
            "cpu": f'sum(rate(container_cpu_usage_seconds_total{{namespace="default", pod=~"{self.current_service}.*", container!="POD", container!=""}}[30s]))',
            "ram": f'sum(container_memory_working_set_bytes{{namespace="default", pod=~"{self.current_service}.*", container!="POD", container!=""}})'
        }
        def safe_query(query):
            try:
                res = requests.get(f"{self.prometheus_url}/api/v1/query", params={"query": query}, timeout=3).json()
                if res.get('status') == 'success' and res.get('data', {}).get('result'): return res['data']['result']
            except: pass
            return []

        raw_traffic, raw_http, raw_grpc, raw_latency, raw_cpu, raw_ram = map(safe_query, [queries["traffic"], queries["http_errors"], queries["grpc_errors"], queries["latency"], queries["cpu"], queries["ram"]])

        traffic_dict = {item['metric'].get('destination_workload', 'unknown'): float(item['value'][1]) for item in raw_traffic}
        http_dict = {item['metric'].get('destination_workload', 'unknown'): float(item['value'][1]) for item in raw_http}
        grpc_dict = {item['metric'].get('destination_workload', 'unknown'): float(item['value'][1]) for item in raw_grpc}
        latency_dict = {item['metric'].get('destination_workload', 'unknown'): float(item['value'][1]) for item in raw_latency if str(item['value'][1]) != 'NaN'}

        error_rates = {}
        for svc, total_req in traffic_dict.items():
            err_req = http_dict.get(svc, 0.0) + grpc_dict.get(svc, 0.0)
            error_rates[svc] = err_req / total_req if total_req > 0 else 0.0

        stable_services_err = [rate for svc, rate in error_rates.items() if svc != self.current_service]
        
        return {
            "canary_err": error_rates.get(self.current_service, 0.0), 
            "stable_err": np.mean(stable_services_err) if stable_services_err else 0.0, 
            "canary_lat": latency_dict.get(self.current_service, 0.0),
            "cpu_cores": float(raw_cpu[0]['value'][1]) if raw_cpu else 0.0, 
            "ram_mb": (float(raw_ram[0]['value'][1]) if raw_ram else 0.0) / (1024 * 1024), 
            "weight_pct": self._get_live_weight_pct_api()
        }

    def _process_and_normalize_features(self, raw):
        return np.array([
            min(raw["cpu_cores"] / CPU_REF, 1.0), min(raw["ram_mb"] / MEM_REF_MB, 1.0),
            min(raw["canary_lat"] / 1000.0, 5.0), min((raw["canary_err"] / max(raw["stable_err"], EPSILON)), MAX_RATIO) / MAX_RATIO,
            raw["weight_pct"] / 100.0
        ], dtype=np.float32)

    def _get_obs(self): return np.stack(list(self.history), axis=1).astype(np.float32)

    def step(self, action):
        old_weight = self._get_live_weight_pct_api()
        
        # Chỉ ủy quyền hành động rẽ cho CLI, còn lại Python làm chủ
        if action == 1: 
            self._run_cmd(["argo-rollouts", "promote", self.current_service, "-n", "default"])
            self._wait_for_weight_change_api(old_weight)
        elif action == 2: 
            self._run_cmd(["argo-rollouts", "abort", self.current_service, "-n", "default"])
            self._wait_for_weight_change_api(old_weight)
        
        if action in [1, 2]:
            self._wait_for_prometheus(25) 
            
        raw = self._get_metrics_from_prometheus()
        norm_channels = self._process_and_normalize_features(raw)
        
        act_str = f"{C.GREEN}🟢 PROMOTE (1){C.END}" if action == 1 else (f"{C.RED}🔴 ABORT (2){C.END}" if action == 2 else f"{C.YELLOW}🟡 HOLD (0){C.END}")
        print(f"\n{C.CYAN}╭───────────────────────────────────────────────────╮{C.END}")
        print(f"{C.CYAN}│{C.END} {C.BOLD}🤖 STEP {self.current_step+1:02d} | Quyết định: {act_str}".ljust(60) + f"{C.CYAN}│{C.END}")
        print(f"{C.CYAN}├───────────────────────────────────────────────────┤{C.END}")
        print(f"{C.CYAN}│{C.END} {C.BLUE}📊 Thực tế:{C.END} Err_Canary: {raw['canary_err']*100:5.2f}% | Latency: {raw['canary_lat']:7.1f}ms".ljust(60) + f"{C.CYAN}│{C.END}")
        print(f"{C.CYAN}│{C.END}             Err_Stable: {raw['stable_err']*100:5.2f}% | Traffic: {raw['weight_pct']:5.1f}%".ljust(60) + f"{C.CYAN}│{C.END}")
        print(f"{C.CYAN}│{C.END} {C.YELLOW}🧠 Agent  :{C.END} CPU:{norm_channels[0]:.2f} RAM:{norm_channels[1]:.2f} Lat:{norm_channels[2]:.2f} Err:{norm_channels[3]:.2f}".ljust(60) + f"{C.CYAN}│{C.END}")
        print(f"{C.CYAN}╰───────────────────────────────────────────────────╯{C.END}")
        
        self.history.append(norm_channels)
        obs = self._get_obs()

        e_ratio = raw["canary_err"] / max(raw["stable_err"], EPSILON)
        l_ratio = raw["canary_lat"] / max(50.0, EPSILON) 
        
        is_high_error = (e_ratio > 2.0) and (raw["canary_err"] > 0.05)
        is_high_latency = raw["canary_lat"] > 1000.0
        current_anomalous = is_high_error or is_high_latency
        
        reward = 0.0
        terminated = False

        if current_anomalous:
            print(f"{C.BOLD}{C.RED}   🚨 BÁO ĐỘNG LỖI! E_Ratio: {e_ratio:.2f} | L_Ratio: {l_ratio:.2f}{C.END}")
            if action == 1: reward -= 15.0; terminated = True 
            elif action == 2: reward += 10.0; terminated = True
            elif action == 0: reward -= 0.5 
        else: 
            if action == 1: 
                reward += 1.0
                if raw["weight_pct"] >= 99.0:
                    print(f"{C.BOLD}{C.GREEN}   🏁 Rollout đạt 100% an toàn!{C.END}")
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
        print(f"{C.YELLOW}   🧹 Trả môi trường về STABLE (none)...{C.END}")
        self._run_cmd(["argo-rollouts", "abort", svc_name, "-n", "default"], ignore_error=True)
        self._apply_fault_to_rollout_api(svc_name, "none")
        time.sleep(3) # Cho phép K3s chốt database
        
        print(f"{C.YELLOW}   ⏩ Ép Promote Full 100% để dọn dẹp Canary cũ...{C.END}")
        self._run_cmd(["argo-rollouts", "promote", svc_name, "--full", "-n", "default"], ignore_error=True)
        
        print(f"{C.YELLOW}   🚿 Xả hàng đợi kẹt xe của LoadGenerator bằng K8s API...{C.END}")
        self._restart_loadgenerator_api()
        
        print(f"{C.GREY}   ⏳ [K8s API] Lắng nghe Event cho đến khi Rollout đạt trạng thái Healthy...{C.END}")
        w = watch.Watch()
        try:
            for event in w.stream(self.custom_api.list_namespaced_custom_object,
                                  self.argo_group, self.argo_version, "default", self.argo_plural,
                                  timeout_seconds=90):
                ro = event.get('object', {})
                if ro.get('metadata', {}).get('name') != svc_name: continue
                if ro.get('status', {}).get('phase', '') == 'Healthy':
                    print(f"{C.GREEN}   ✅ [K8s API] Hệ thống STABLE đã sẵn sàng 100%!{C.END}")
                    break
        finally:
            w.stop()

    def reset(self, seed=None, options=None):   
        super().reset(seed=seed)
        self.current_step = 0
        self.service_list = ['checkoutservice']
        self.current_service = random.choice(self.service_list)

        print(f"\n{C.BOLD}{C.MAGENTA}═════════════════════════════════════════════════════════{C.END}")
        print(f"{C.BOLD}{C.MAGENTA} 🔄 BẮT ĐẦU EPISODE MỚI CHO: {self.current_service.upper()}{C.END}")
        print(f"{C.BOLD}{C.MAGENTA}═════════════════════════════════════════════════════════{C.END}")
        
        self._graceful_reset_baseline(self.current_service)
        
        faults = ["none", "none", "none", "high_error", "high_latency", "combined"]
        chosen_fault = random.choice(faults)
        
        fault_color = C.GREEN if chosen_fault == "none" else C.RED
        print(f"\n{C.BOLD}🚀 BƠM KỊCH BẢN CANARY: {fault_color}[{chosen_fault.upper()}]{C.END}")
        
        self._apply_fault_to_rollout_api(self.current_service, chosen_fault)
        self._wait_for_weight_change_api(100.0) # Khóa luồng cho đến khi rẽ nhánh xong
        
        self._wait_for_prometheus(35) 

        raw = self._get_metrics_from_prometheus()
        initial_channels = self._process_and_normalize_features(raw)
        self.history = deque([initial_channels]*self.seq_len, maxlen=self.seq_len)

        return self._get_obs(), {}