import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import subprocess
import time
import random
from collections import deque
import re

# Định nghĩa các hằng số cấu hình lấy chính xác từ feature_pipeline.py
EPSILON = 1e-6
CPU_REF = 0.02
MEM_REF_MB = 128.0
MAX_RATIO = 5.0

class OnlineCanaryEnv(gym.Env):
    """
    Môi trường Live Canary trên cụm K3s thật sử dụng Istio & Prometheus.
    Đáp ứng chuẩn ma trận (5, 30) và các kênh: [CPU, RAM, Latency, Error_Rate, Traffic_Pct]
    """
    def __init__(self, service_list=None, prometheus_url="http://172.26.52.132:30090", seq_len=30):
        super().__init__()
        self.prometheus_url = prometheus_url
        self.seq_len = seq_len
        self.num_features = 5
        
        # Danh sách microservices ngẫu nhiên cho mỗi episode
        self.service_list = service_list or ['frontend']
        self.current_service = None
        
        # 1. PATCH ĐỦ 3 HÀNH ĐỘNG: 0 = Hold, 1 = Promote, 2 = Rollback (Abort)
        self.action_space = spaces.Discrete(3)
        
        # 2. PATCH ĐỦ MATRIX KÍCH THƯỚC (5, 30) chuẩn hóa trong khoảng [0, 1]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_features, self.seq_len),
            dtype=np.float32
        )
        
        # Hàng đợi lưu giữ lịch sử trượt 30 timesteps giống offline
        self.history = deque(maxlen=self.seq_len)
        self.current_step = 0
        self.max_steps = 20  # Giới hạn số step tối đa cho một đợt rollout live

    def _query_prometheus(self, query):
        try:
            response = requests.get(f"{self.prometheus_url}/api/v1/query", params={"query": query}, timeout=5)
            result = response.json()
            if result['status'] == 'success' and result['data']['result']:
                val = float(result['data']['result'][0]['value'][1])
                return val
            # --- THÊM DÒNG NÀY ĐỂ DEBUG ---
            else:
                print(f"DEBUG: Query rỗng/lỗi: {query}") 
                return 0.0
        except Exception as e:
            return 0.0

    def _get_live_weight_pct(self):
        """Lấy phần trăm traffic bằng CLI Argo để tránh lỗi JSONPath trên Windows"""
        try:
            cmd = ["argo-rollouts", "get", "rollout", self.current_service, "-n", "default"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            
            # Dùng Regex quét tìm dòng "ActualWeight:  20"
            match = re.search(r'ActualWeight:\s+(\d+)', res.stdout)
            if match:
                return float(match.group(1))
            return 0.0
        except Exception as e:
            print(f"Lỗi lấy weight: {e}")
            return 0.0

    def _debug_labels(self):
        query = 'count(istio_requests_total) by (destination_service)'
        # In ra 5 service đầu tiên tìm thấy
        response = requests.get(f"{self.prometheus_url}/api/v1/query", params={"query": query}).json()
        print("\n🔍 DEBUG: Danh sách các Service Istio đang ghi nhận metrics:")
        if 'data' in response and 'result' in response['data']:
            for item in response['data']['result'][:5]:
                print(f"   -> {item['metric'].get('destination_service')}")

    def _get_metrics_from_prometheus(self):
        svc = self.current_service
        canary_svc = f"{svc}-canary"
        stable_svc = f"{svc}-stable"
        ns = "default"

        def get_val(query):
            try:
                res = requests.get(f"{self.prometheus_url}/api/v1/query", params={"query": query}, timeout=5).json()
                if res.get('status') == 'success' and res['data']['result']:
                    return float(res['data']['result'][0]['value'][1])
            except Exception:
                pass
            return 0.0

        # Dùng linh hoạt destination_service (cho Host routing) và destination_workload (cho Subset routing)
        # 1. TỔNG REQUEST
        tot_canary_q = f'sum(rate(istio_requests_total{{namespace="{ns}", destination_service=~"^{canary_svc}.*", reporter="destination"}}[2m])) or sum(rate(istio_requests_total{{namespace="{ns}", destination_workload=~"^{svc}-.*", destination_canonical_revision="canary", reporter="destination"}}[2m]))'
        tot_stable_q = f'sum(rate(istio_requests_total{{namespace="{ns}", destination_service=~"^{stable_svc}.*", reporter="destination"}}[2m])) or sum(rate(istio_requests_total{{namespace="{ns}", destination_workload=~"^{svc}-.*", destination_canonical_revision="latest", reporter="destination"}}[2m]))'

        # 2. LỖI HTTP (Mã 5xx)
        http_err_can_q = f'sum(rate(istio_requests_total{{namespace="{ns}", destination_service=~"^{canary_svc}.*", reporter="destination", response_code=~"5.*"}}[2m])) or sum(rate(istio_requests_total{{namespace="{ns}", destination_workload=~"^{svc}-.*", destination_canonical_revision="canary", reporter="destination", response_code=~"5.*"}}[2m]))'
        http_err_sta_q = f'sum(rate(istio_requests_total{{namespace="{ns}", destination_service=~"^{stable_svc}.*", reporter="destination", response_code=~"5.*"}}[2m])) or sum(rate(istio_requests_total{{namespace="{ns}", destination_workload=~"^{svc}-.*", destination_canonical_revision="latest", reporter="destination", response_code=~"5.*"}}[2m]))'

        # 3. LỖI gRPC (Mã trạng thái khác 0)
        grpc_err_can_q = f'sum(rate(istio_requests_total{{namespace="{ns}", destination_service=~"^{canary_svc}.*", reporter="destination", grpc_response_status=~"^[1-9].*"}}[2m])) or sum(rate(istio_requests_total{{namespace="{ns}", destination_workload=~"^{svc}-.*", destination_canonical_revision="canary", reporter="destination", grpc_response_status=~"^[1-9].*"}}[2m]))'
        grpc_err_sta_q = f'sum(rate(istio_requests_total{{namespace="{ns}", destination_service=~"^{stable_svc}.*", reporter="destination", grpc_response_status=~"^[1-9].*"}}[2m])) or sum(rate(istio_requests_total{{namespace="{ns}", destination_workload=~"^{svc}-.*", destination_canonical_revision="latest", reporter="destination", grpc_response_status=~"^[1-9].*"}}[2m]))'

        # 4. LATENCY (P95)
        metric_lat = "istio_request_duration_milliseconds_bucket" 
        l_can_q = f'histogram_quantile(0.95, sum by (le) (rate({metric_lat}{{namespace="{ns}", destination_service=~"^{canary_svc}.*", reporter="destination"}}[2m]))) or histogram_quantile(0.95, sum by (le) (rate({metric_lat}{{namespace="{ns}", destination_workload=~"^{svc}-.*", destination_canonical_revision="canary", reporter="destination"}}[2m])))'
        l_sta_q = f'histogram_quantile(0.95, sum by (le) (rate({metric_lat}{{namespace="{ns}", destination_service=~"^{stable_svc}.*", reporter="destination"}}[2m]))) or histogram_quantile(0.95, sum by (le) (rate({metric_lat}{{namespace="{ns}", destination_workload=~"^{svc}-.*", destination_canonical_revision="latest", reporter="destination"}}[2m])))'

        # --- THỰC THI VÀ TÍNH TOÁN ---
        tot_can = max(get_val(tot_canary_q), 0.0001)
        tot_sta = max(get_val(tot_stable_q), 0.0001)

        err_can = get_val(http_err_can_q) + get_val(grpc_err_can_q)
        err_sta = get_val(http_err_sta_q) + get_val(grpc_err_sta_q)

        e_canary_rate = err_can / tot_can
        e_stable_rate = err_sta / tot_sta

        # 5. CPU/RAM
        cpu_q = f'sum(rate(container_cpu_usage_seconds_total{{container="server", pod=~"{svc}-.*"}}[2m]))'
        mem_q = f'sum(container_memory_working_set_bytes{{container="server", pod=~"{svc}-.*"}}) / 1024 / 1024'

        return {
            "weight_pct": self._get_live_weight_pct(),
            "e_canary": e_canary_rate,
            "e_stable": e_stable_rate,
            "l_canary": get_val(l_can_q),
            "l_stable": get_val(l_sta_q),
            "cpu": get_val(cpu_q),
            "mem_mb": get_val(mem_q)
        }

    def _clip(self, value, low, high):
        return max(low, min(high, float(value)))

    def _process_and_normalize_features(self, raw):
        # 1. Bảo vệ phép chia: Đặt epsilon lớn hơn một chút để tránh division by zero
        EPS = 1e-3 
        
        # 2. Tính tỉ lệ an toàn
        e_ratio = float(raw["e_canary"]) / max(float(raw["e_stable"]) + EPS, EPS)
        l_ratio = float(raw["l_canary"]) / max(float(raw["l_stable"]) + EPS, EPS)
        
        # 3. Ép kiểu và Clipping (Rất quan trọng để PPO không bị "ngáo")
        # Giới hạn ratio trong khoảng [0, 5] để tránh số quá lớn
        e_ratio_n = np.clip(e_ratio, 0.0, 5.0) / 5.0
        l_ratio_n = np.clip(l_ratio, 0.0, 5.0) / 5.0
        
        # 4. Chuẩn hóa CPU/RAM (dùng tham số cũ của bạn)
        cpu_n = np.clip(float(raw["cpu"]) / CPU_REF, 0.0, 1.0)
        mem_n = np.clip(float(raw["mem_mb"]) / MEM_REF_MB, 0.0, 1.0)
        traffic_n = np.clip(float(raw["weight_pct"]) / 100.0, 0.0, 1.0)

        # Trả về 5 features chuẩn
        return np.array([cpu_n, mem_n, l_ratio_n, e_ratio_n, traffic_n], dtype=np.float32)

    def _get_obs(self):
        """Gộp bộ nhớ history thành ma trận không gian (5, 30)"""
        arr = np.stack(list(self.history), axis=1)
        return arr.astype(np.float32)

    def _execute_argo_command(self, action):    
        cmd = []
        if action == 1: # Promote
            # Gọi trực tiếp argo-rollouts, không dùng kubectl
            cmd = ["argo-rollouts", "promote", self.current_service, "-n", "default"]
        elif action == 2: # Abort
            cmd = ["argo-rollouts", "abort", self.current_service, "-n", "default"]
        
        if cmd:
            # DÒNG DEBUG: In ra lệnh thực tế để xem nó gọi cái gì
            print(f"DEBUG: Đang chạy lệnh: {' '.join(cmd)}")
            
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                print(f"  ✅ [SUCCESS] Lệnh Argo thành công: {res.stdout.strip()}")
            else:
                print(f"  ❌ [ERROR] Lệnh Argo thất bại: {res.stderr.strip()}")
            time.sleep(15)

    def step(self, action):

        # 1. Thực thi hành động
        self._execute_argo_command(action)
        
        # 2. Thu thập dữ liệu
        raw = self._get_metrics_from_prometheus()
        norm_channels = self._process_and_normalize_features(raw)
        
        # DEBUG: In ra dữ liệu Agent nhìn thấy
        print(f"\n--- DEBUG STEP ---")
        print(f"Action chọn: {action} (0=Hold, 1=Promote, 2=Rollback)")
        print(f"Raw Metrics: Err_Canary={raw['e_canary']:.4f}, Err_Stable={raw['e_stable']:.4f}, Weight={raw['weight_pct']}%")
        print(f"Normalized: CPU={norm_channels[0]:.2f}, RAM={norm_channels[1]:.2f}, Latency={norm_channels[2]:.2f}, Error={norm_channels[3]:.2f}, Traffic={norm_channels[4]:.2f}")
        
        self.history.append(norm_channels)
        obs = self._get_obs()

        e_ratio = raw["e_canary"] / max(raw["e_stable"], EPSILON)
        l_ratio = raw["l_canary"] / max(raw["l_stable"], EPSILON)
        
        # BỘ LỌC NHIỄU: Chỉ báo động khi tỷ lệ lỗi thực > 5% (0.05) VÀ gấp đôi bản stable
        is_high_error = (e_ratio > 2.0) and (raw["e_canary"] > 0.05)
        
        # BỘ LỌC ĐỘ TRỄ: Chỉ báo động khi trễ gấp đôi VÀ lớn hơn 50ms
        is_high_latency = (l_ratio > 2.0) and (raw["l_canary"] > 50.0)
        
        current_anomalous = is_high_error or is_high_latency
        
        reward = 0.0
        terminated = False

        if action == 1: # Promote
            if current_anomalous:
                print(f"🚨 PHÁT HIỆN BẤT THƯỜNG! E_Ratio: {e_ratio:.2f}, L_Ratio: {l_ratio:.2f}")
                reward -= 10.0 # Phạt nặng nếu Promote khi đang lỗi
                terminated = True # Ép dừng và Reset
            else:
                reward += 1.0 # Thưởng nhẹ cho hành động Promote an toàn
                # KHÔNG set terminated = True ở đây để nó tiếp tục bước sau
        
        elif action == 2: # Rollback/Abort
            reward += 5.0 # Thưởng nếu Abort khi thấy lỗi
            terminated = True # Kết thúc tập vì đã xử lý xong (dù tốt hay xấu)
            
        elif action == 0: # Hold
            reward -= 0.1 # Phạt nhẹ để tránh tình trạng Agent không làm gì cả
            # Không terminate, chờ xem bước sau thế nào

        # Điều kiện dừng: Rollout hoàn tất 100%
        if raw["weight_pct"] >= 99.0:
            print("🏁 Rollout đạt 100%, kết thúc tập.")
            reward += 20.0 # Thưởng lớn vì hoàn thành Release
            terminated = True 

        # Điều kiện dừng: Quá số bước tối đa
        self.current_step += 1
        if self.current_step >= self.max_steps:
            terminated = True
            reward -= 5.0
        # Thêm vào cuối hàm step(), ngay trước lệnh return
        if terminated:
            print(f"🏁 TẬP HUẤN LUYỆN KẾT THÚC. Reward: {reward:.2f}. Đang reset môi trường...")

        return obs, float(reward), bool(terminated), False, {}

    def _inject_random_fault(self):
        fault_types = ["http_error"]
        chosen_fault = random.choice(fault_types)
        print(f"🌪  [Chaos Mesh] Kịch bản kích hoạt: [{chosen_fault}] trên Pods [{self.current_service}]")

        # Xóa Chaos cũ
        subprocess.run(["kubectl", "delete", "networkchaos,podchaos,httpchaos", "--all", "-n", "default"], capture_output=True)

        if chosen_fault == "high_latency":
            # Tiêm độ trễ 3s vào tất cả các pod của service hiện tại
            yaml_manifest = f"""
apiVersion: chaos-mesh.org/v1alpha1
kind: NetworkChaos
metadata:
  name: latency-{self.current_service}
  namespace: default
spec:
  action: delay
  mode: all
  selector:
    labelSelectors:
      app: {self.current_service}
  delay:
    latency: '3s'
    correlation: '100'
    jitter: '0ms'
"""
            subprocess.run(["kubectl", "apply", "-f", "-"], input=yaml_manifest, text=True, capture_output=True)
            
        elif chosen_fault == "http_error":
            # Dùng PodChaos thay vì NetworkChaos để vượt qua rào cản CNI của K3s
            yaml_manifest = f"""
apiVersion: chaos-mesh.org/v1alpha1
kind: PodChaos
metadata:
  name: fail-{self.current_service}
  namespace: default
spec:
  action: pod-failure
  mode: all
  duration: '45s'
  selector:
    labelSelectors:
      app: {self.current_service}
"""
            subprocess.run(["kubectl", "apply", "-f", "-"], input=yaml_manifest, text=True, capture_output=True)

    def reset(self, seed=None, options=None):   
        super().reset(seed=seed)
        
        print("\n🧹 [CLEANUP] Đang thực hiện RESET toàn bộ hệ thống (Hard Flush)...")
        
        # 1. Dọn dẹp triệt để TOÀN BỘ lỗi từ Chaos Mesh (không chỉ ở episode trước)
        subprocess.run(["kubectl", "delete", "networkchaos,podchaos,httpchaos", "--all", "-n", "default"], capture_output=True)
        
        self.service_list = ['frontend']

        # 2. Hủy bỏ (abort) mọi tiến trình Rollout đang dở dang của TẤT CẢ các service
        # (Phòng trường hợp episode trước bị ngắt giữa chừng)
        for svc in self.service_list:
            subprocess.run(["argo-rollouts", "abort", svc, "-n", "default"], capture_output=True)
        
        # 3. Ép khởi động lại toàn bộ (Redeploy) để xả rác bộ nhớ và connection
        print("🔄 [CLEANUP] Đang khởi tạo lại (Restart) toàn bộ Microservices...")
        for svc in self.service_list:
            subprocess.run(["kubectl", "rollout", "restart", f"rollout/{svc}", "-n", "default"], capture_output=True)
        
        # Đợi hệ thống ổn định (5s là không đủ, cần ít nhất 30-40s cho 10 services)
        print("⏳ Đợi 40 giây để K3s cấp phát xong tài nguyên mới...")
        time.sleep(2) 
        
        # --- KẾT THÚC CLEANUP ---

        self.current_service = random.choice(self.service_list)
        print(f"\n🚀 KHỞI ĐỘNG EPISODE: Tiến trình Rollout cho [{self.current_service}]")

        # Ép tạo Revision mới cho Rollout bằng cách đổi annotation
        tick = str(int(time.time()))
        patch_json = f'{{"spec":{{"template":{{"metadata":{{"annotations":{{"rl-train-tick":"{tick}"}}}}}}}}}}'
        cmd = ["kubectl", "patch", f"rollout/{self.current_service}", "-n", "default", "--type=merge", "-p", patch_json]
        subprocess.run(cmd, capture_output=True)

        time.sleep(10) 
        
        # Tiêm lỗi vật lý bằng Chaos Mesh (bản patch mới nhất của bạn)
        self._inject_random_fault()
        
        # Chờ metric ngấm vào Prometheus
        time.sleep(15) 

        # Đọc dữ liệu ban đầu
        raw = self._get_metrics_from_prometheus()
        initial_channels = self._process_and_normalize_features(raw)
        self.history = deque([initial_channels]*self.seq_len, maxlen=self.seq_len)

        return self._get_obs(), {}