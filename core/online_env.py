import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import subprocess
import time

class OnlineCanaryEnv(gym.Env):
    def __init__(self, service_name="checkoutservice", metrics_url="http://localhost:8000"):
        super(OnlineCanaryEnv, self).__init__()
        
        self.service_name = service_name
        self.metrics_url = metrics_url
        
        # Không gian hành động: 0 (Rollback), 1 (Promote)
        self.action_space = spaces.Discrete(2)
        
        # Không gian quan sát (Giống hệt offline): [cpu_usage, error_rate, latency]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0]), 
            high=np.array([100.0, 1.0, 5000.0]), 
            dtype=np.float32
        )
        
        self.current_step = 0
        self.max_steps = 10 # Số bước tối đa cho 1 đợt rollout

    def _get_metrics_from_injector(self):
        """Gọi HTTP sang Trụ cột 1 (FastAPI) để lấy số liệu mạng"""
        try:
            response = requests.get(f"{self.metrics_url}/metrics/canary")
            data = response.json()["data"]
            return np.array([
                data["cpu_usage_percent"],
                data["error_rate"],
                data["latency_ms"]
            ], dtype=np.float32)
        except Exception as e:
            print(f"Lỗi kết nối tới Metrics Injector: {e}")
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    def _execute_argo_command(self, action):
        """Dịch quyết định của PPO thành lệnh thật trên Kubernetes"""
        if action == 1: # Promote
            cmd = f"kubectl argo rollouts promote {self.service_name} -n default"
        else: # Rollback (Abort)
            cmd = f"kubectl argo rollouts abort {self.service_name} -n default"
            
        # Thực thi lệnh qua bash
        subprocess.run(cmd.split(), capture_output=True)
        # Bắt buộc: Sleep một chút để mô phỏng thời gian K8s bẻ Istio traffic
        time.sleep(2) 

    def step(self, action):
        # 1. Thực thi lệnh lên cụm K8s
        self._execute_argo_command(action)
        
        # 2. Lấy số liệu phản hồi từ cụm
        obs = self._get_metrics_from_injector()
        
        # 3. Tính toán phần thưởng (Reward)
        reward = self._calculate_reward(obs, action)
        
        self.current_step += 1
        terminated = bool(self.current_step >= self.max_steps or action == 0)
        truncated = False
        
        return obs, reward, terminated, truncated, {}

    def _calculate_reward(self, obs, action):
        # Logic tính điểm (Giữ nguyên hoặc tinh chỉnh từ phase Offline)
        # Ví dụ: Nếu Error rate quá cao mà vẫn Promote -> Phạt nặng
        cpu, error, latency = obs
        if error > 0.05 and action == 1:
            return -10.0
        elif action == 1:
            return 1.0
        return 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Đưa K8s Rollout về trạng thái ban đầu (nếu cần)
        # subprocess.run(...) 
        
        obs = self._get_metrics_from_injector()
        return obs, {}