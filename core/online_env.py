import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import subprocess
import time
import random
import os

class OnlineCanaryEnv(gym.Env):
    def __init__(self, service_list=None, prometheus_url="http://172.26.52.132:30090"):
        self.prometheus_url = prometheus_url
        
        # Sửa lại cú pháp mảng chuẩn của Python
        self.service_list = service_list or [
            'adservice', 'cartservice', 'checkoutservice', 'currencyservice', 
            'emailservice', 'frontend', 'paymentservice', 'productcatalogservice', 
            'recommendationservice', 'shippingservice'
        ]
        self.current_service = None # Sẽ được gán ngẫu nhiên trong hàm reset()
        
        # Không gian hành động: 0 (Rollback/Abort), 1 (Promote)
        self.action_space = spaces.Discrete(2)
        
        # Không gian quan sát: [cpu_usage (%), error_rate (0-1), latency (ms)]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]), 
            high=np.array([100.0, 1.0, 5000.0]), 
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = 10 

    def _query_prometheus(self, query):
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/query", params={"query": query}, timeout=5)
            response.raise_for_status()
            result = response.json()
            
            if result['data']['result']:
                val = float(result['data']['result'][0]['value'][1])
                return val if not np.isnan(val) else 0.0
            return 0.0
        except Exception as e:
            print(f"[Cảnh báo] Lỗi truy vấn Prometheus: {e}")
            return 0.0

    def _get_metrics_from_prometheus(self):
        # Chú ý: Đã đổi self.service_name thành self.current_service
        err_query = f'sum(rate(istio_requests_total{{destination_service=~"{self.current_service}.*", response_code=~"5.."}}[1m])) / sum(rate(istio_requests_total{{destination_service=~"{self.current_service}.*"}}[1m]))'
        error_rate = self._query_prometheus(err_query)
        
        lat_query = f'histogram_quantile(0.95, sum(rate(istio_request_duration_milliseconds_bucket{{destination_service=~"{self.current_service}.*"}}[1m])) by (le))'
        latency = self._query_prometheus(lat_query)
        
        cpu_query = f'sum(rate(container_cpu_usage_seconds_total{{pod=~"{self.current_service}.*"}}[1m])) * 100'
        cpu_usage = self._query_prometheus(cpu_query)

        return np.array([cpu_usage, error_rate, latency], dtype=np.float32)

    def _execute_argo_command(self, action):
        if action == 1: 
            print(f"  -> Thực thi K8s: PROMOTE {self.current_service}")
            cmd = f"kubectl argo rollouts promote {self.current_service} -n default"
        else: 
            print(f"  -> Thực thi K8s: ABORT/ROLLBACK {self.current_service}")
            cmd = f"kubectl argo rollouts abort {self.current_service} -n default"
            
        subprocess.run(cmd.split(), capture_output=True)
        time.sleep(15) 

    def step(self, action):
        self._execute_argo_command(action)
        obs = self._get_metrics_from_prometheus()
        reward = self._calculate_reward(obs, action)
        
        self.current_step += 1
        terminated = bool(self.current_step >= self.max_steps or action == 0)
        truncated = False
        
        return obs, reward, terminated, truncated, {}

    def _calculate_reward(self, obs, action):
        cpu, error, latency = obs
        if (error > 0.05 or latency > 2000.0) and action == 1:
            return -15.0
        elif action == 1:
            return 1.0
        return 0.0

    def _inject_random_fault(self):
        """Logic tung xúc xắc tiêm lỗi thẳng vào VirtualService gốc"""
        fault_types = ["no_fault", "high_latency", "http_error"]
        chosen_fault = random.choice(fault_types)
        print(f"🌪 Kịch bản lỗi cho episode này: [{chosen_fault}] trên [{self.current_service}]")

        # Tên VirtualService phải khớp với hậu tố -vsvc trong Helm Chart của bạn
        vsvc_name = f"{self.current_service}-vsvc"

        # 1. Dọn dẹp (Remove): Xóa block fault cũ nếu có (bằng cách patch 1 chuỗi json rỗng)
        # Chú ý: Dùng shell=True vì lệnh patch chứa nhiều dấu nháy đơn/kép phức tạp
        remove_fault_cmd = f"""kubectl patch virtualservice {vsvc_name} -n default --type='json' -p='[{{"op": "remove", "path": "/spec/http/0/fault"}}]'"""
        subprocess.run(remove_fault_cmd, shell=True, capture_output=True)

        # 2. Tiêm lỗi mới (Inject)
        if chosen_fault == "high_latency":
            # Chèn lỗi delay 3 giây cho 50% traffic
            patch_delay = f"""kubectl patch virtualservice {vsvc_name} -n default --type='merge' -p '{{"spec":{{"http":[{{"fault":{{"delay":{{"percentage":{{"value":50.0}},"fixedDelay":"3s"}} }}}} ]}} }}'"""
            subprocess.run(patch_delay, shell=True, capture_output=True)
            
        elif chosen_fault == "http_error":
            # Chèn lỗi HTTP 503 cho 30% traffic
            patch_abort = f"""kubectl patch virtualservice {vsvc_name} -n default --type='merge' -p '{{"spec":{{"http":[{{"fault":{{"abort":{{"percentage":{{"value":30.0}},"httpStatus":503}} }}}} ]}} }}'"""
            subprocess.run(patch_abort, shell=True, capture_output=True)
            
        elif chosen_fault == "no_fault":
            # Không làm gì thêm, hệ thống ở trạng thái khỏe mạnh
            pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # 1. Chọn ngẫu nhiên 1 service cho đợt train này
        self.current_service = random.choice(self.service_list)
        print(f"\n🔄 --- BẮT ĐẦU EPISODE MỚI: {self.current_service} ---")
        
        # 2. Restart Rollout
        cmd = f"kubectl argo rollouts restart {self.current_service} -n default"
        subprocess.run(cmd.split(), capture_output=True)
        
        # 3. Chờ Pod mới khởi động
        time.sleep(15) 
        
        # 4. Tiêm lỗi Istio (Chaos Engineering)
        self._inject_random_fault()
        
        # Chờ traffic Istio kịp ghi nhận lỗi trước khi lấy obs đầu tiên
        time.sleep(15)
        
        obs = self._get_metrics_from_prometheus()
        return obs, {}