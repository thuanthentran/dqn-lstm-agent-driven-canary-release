import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import subprocess
import time
import random
from collections import deque
import re
import json
import sys
import tempfile
import os

EPSILON = 1e-6
CPU_REF = 0.02
MEM_REF_MB = 128.0
MAX_RATIO = 5.0

# --- BỘ MÀU SẮC THẨM MỸ (ANSI COLORS) ---
class C:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GREY = '\033[90m'
    BOLD = '\033[1m'
    END = '\033[0m'

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

    def _run_cmd(self, cmd, ignore_error=False):
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 and not ignore_error:
            print(f"\n{C.BOLD}{C.RED}❌ [FATAL ERROR] Lệnh hệ thống thất bại!{C.END}")
            print(f"   {C.RED}Lệnh: {' '.join(cmd)}{C.END}")
            print(f"   {C.RED}STDERR: {res.stderr.strip()}{C.END}")
            sys.exit(1)
        return res

    def _wait_for_rollout_step(self, old_weight):
        print(f"{C.GREY}   🔄 Đang chờ Argo rẽ traffic (Đang ở: {old_weight}%)...{C.END}")
        for _ in range(40): 
            current_weight = self._get_live_weight_pct()
            if current_weight != old_weight:
                print(f"{C.GREEN}   ✅ Đã rẽ traffic thành công lên: {current_weight}%{C.END}")
                return
            time.sleep(1)

    def _apply_fault_to_rollout(self, svc_name, fault_value):
        res = self._run_cmd(["kubectl", "get", "rollout", svc_name, "-n", "default", "-o", "json"])
        rollout_data = json.loads(res.stdout)
        
        containers = rollout_data.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        for c in containers:
            if c.get("name") in [svc_name, "server"]: 
                env_list = c.get("env", [])
                found = False
                for e in env_list:
                    if e.get("name") == "FAULT_SCENARIO":
                        e["value"] = fault_value
                        found = True
                        break
                if not found:
                    env_list.append({"name": "FAULT_SCENARIO", "value": fault_value})
                c["env"] = env_list
                break

        patch_obj = { "spec": { "template": { "metadata": { "annotations": { "rl-train-tick": str(int(time.time())) } }, "spec": { "containers": containers } } } }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(patch_obj, f)
            temp_path = f.name
            
        self._run_cmd(["kubectl", "patch", f"rollout/{svc_name}", "-n", "default", "--type=merge", "--patch-file", temp_path])
        os.remove(temp_path)

    def _get_live_weight_pct(self):
        try:
            res = self._run_cmd(["argo-rollouts", "get", "rollout", self.current_service, "-n", "default"])
            match = re.search(r'ActualWeight:\s+(\d+)', res.stdout)
            if match: return float(match.group(1))
            return 0.0
        except SystemExit:
            raise
        except Exception:
            return 0.0

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
                if res.get('status') == 'success' and res.get('data', {}).get('result'):
                    return res['data']['result']
            except: pass
            return []

        raw_traffic, raw_http, raw_grpc, raw_latency, raw_cpu, raw_ram = map(safe_query, [queries["traffic"], queries["http_errors"], queries["grpc_errors"], queries["latency"], queries["cpu"], queries["ram"]])

        traffic_dict = {item['metric'].get('destination_workload', 'unknown'): float(item['value'][1]) for item in raw_traffic}
        http_dict = {item['metric'].get('destination_workload', 'unknown'): float(item['value'][1]) for item in raw_http}
        grpc_dict = {item['metric'].get('destination_workload', 'unknown'): float(item['value'][1]) for item in raw_grpc}
        latency_dict = {item['metric'].get('destination_workload', 'unknown'): float(item['value'][1]) for item in raw_latency if str(item['value'][1]) != 'NaN'}

        cpu_cores = float(raw_cpu[0]['value'][1]) if raw_cpu else 0.0
        ram_bytes = float(raw_ram[0]['value'][1]) if raw_ram else 0.0

        error_rates = {}
        for svc, total_req in traffic_dict.items():
            err_req = http_dict.get(svc, 0.0) + grpc_dict.get(svc, 0.0)
            error_rates[svc] = err_req / total_req if total_req > 0 else 0.0

        stable_services_err = [rate for svc, rate in error_rates.items() if svc != self.current_service]
        
        return {
            "canary_err": error_rates.get(self.current_service, 0.0), 
            "stable_err": np.mean(stable_services_err) if stable_services_err else 0.0, 
            "canary_lat": latency_dict.get(self.current_service, 0.0),
            "cpu_cores": cpu_cores, 
            "ram_mb": ram_bytes / (1024 * 1024), 
            "weight_pct": self._get_live_weight_pct()
        }

    def _process_and_normalize_features(self, raw):
        return np.array([
            min(raw["cpu_cores"] / CPU_REF, 1.0), min(raw["ram_mb"] / MEM_REF_MB, 1.0),
            min(raw["canary_lat"] / 1000.0, 5.0), min((raw["canary_err"] / max(raw["stable_err"], EPSILON)), MAX_RATIO) / MAX_RATIO,
            raw["weight_pct"] / 100.0
        ], dtype=np.float32)

    def _get_obs(self): return np.stack(list(self.history), axis=1).astype(np.float32)

    def step(self, action):
        old_weight = self._get_live_weight_pct()
        
        if action == 1: 
            self._run_cmd(["argo-rollouts", "promote", self.current_service, "-n", "default"])
            self._wait_for_rollout_step(old_weight)
        elif action == 2: 
            self._run_cmd(["argo-rollouts", "abort", self.current_service, "-n", "default"])
            self._wait_for_rollout_step(old_weight)
        
        if action in [1, 2]:
            print(f"{C.GREY}   ⏳ Đợi 25s cho Prometheus chốt sổ metrics mới...{C.END}")
            time.sleep(25) 
            
        raw = self._get_metrics_from_prometheus()
        norm_channels = self._process_and_normalize_features(raw)
        
        # --- UI DASHBOARD CHO STEP ---
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
        print(f"{C.YELLOW}   🧹 Hủy bỏ Canary dở dang...{C.END}")
        self._run_cmd(["argo-rollouts", "abort", svc_name, "-n", "default"], ignore_error=True)
        
        print(f"{C.YELLOW}   🧹 Trả môi trường về STABLE (none)...{C.END}")
        self._apply_fault_to_rollout(svc_name, "none")
        
        # --- PATCH CHỐNG RACE CONDITION ---
        print(f"{C.GREY}   ⏳ Đợi 8s cho Argo Controller khởi tạo Revision mới...{C.END}")
        time.sleep(8) 
        # ----------------------------------
        
        print(f"{C.YELLOW}   ⏩ Ép Promote Full 100%...{C.END}")
        self._run_cmd(["argo-rollouts", "promote", svc_name, "--full", "-n", "default"], ignore_error=True)
        
        print(f"{C.GREY}   ⏳ Đang đợi toàn bộ Pod cũ shutdown êm ái...{C.END}")
        # Tôi khuyên bạn nên nới lỏng timeout lên 180s cho chắc chắn đối với máy cá nhân
        self._run_cmd(["argo-rollouts", "status", svc_name, "--timeout=180s", "-n", "default"])

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
        self._apply_fault_to_rollout(self.current_service, chosen_fault)

        self._wait_for_rollout_step(0.0) 
        print(f"{C.GREY}   ⏳ Đợi 35s cho Prometheus thu thập data mồi...{C.END}")
        time.sleep(35) 

        raw = self._get_metrics_from_prometheus()
        initial_channels = self._process_and_normalize_features(raw)
        self.history = deque([initial_channels]*self.seq_len, maxlen=self.seq_len)

        return self._get_obs(), {}