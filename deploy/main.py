import os
import time
from typing import List
import logging

import httpx
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- 1. KIẾN TRÚC MÔ HÌNH (DRQN) ---
class DRQN(nn.Module):
    def __init__(self, n_obs=8, n_actions=5):
        super(DRQN, self).__init__()
        self.fc1 = nn.Linear(n_obs, 64) 
        self.lstm = nn.LSTM(64, 128, batch_first=True) 
        self.fc2 = nn.Linear(128, n_actions) 

    def forward(self, x, hidden=None):
        x = torch.relu(self.fc1(x))
        x, hidden = self.lstm(x, hidden)
        # Lấy timestep cuối cùng của chuỗi (Sequence length = 10) [cite: 35, 81]
        x = self.fc2(x[:, -1, :]) 
        return x, hidden

# --- 2. CẤU HÌNH HỆ THỐNG ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/model_canary_drqn.pth")
SEQ_LENGTH = 10
DEVICE = torch.device("cpu")
PROMETHEUS_URL = os.getenv(
    "PROMETHEUS_URL",
    "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090",
)
PROM_QUERY_STEP = os.getenv("PROM_QUERY_STEP", "30s")
PROM_QUERY_WINDOW_SECONDS = int(os.getenv("PROM_QUERY_WINDOW_SECONDS", "300"))
PROM_QUERY_TIMEOUT_SECONDS = float(os.getenv("PROM_QUERY_TIMEOUT_SECONDS", "5"))

app = FastAPI(title="Canary AI Agent Service")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("canary-ai-agent")

# Load model đã huấn luyện
model = DRQN(n_obs=8, n_actions=5).to(DEVICE)
MODEL_READY = False
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    MODEL_READY = True
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

# --- 3. ĐỊNH NGHĨA DỮ LIỆU ---
class AppInfo(BaseModel):
    name: str
    weight: float
    namespace: str = "default"
    canary_service: str = "my-app-canary"
    stable_service: str = "my-app-stable"

class InferenceRequest(BaseModel):
    app_info: AppInfo


def _prom_query_range(query: str, start_ts: int, end_ts: int, step: str) -> List[float]:
    params = {
        "query": query,
        "start": start_ts,
        "end": end_ts,
        "step": step,
    }
    try:
        with httpx.Client(timeout=PROM_QUERY_TIMEOUT_SECONDS) as client:
            response = client.get(f"{PROMETHEUS_URL}/api/v1/query_range", params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError:
        # Trả về rỗng để agent vẫn phản hồi /predict, tránh làm AnalysisRun lỗi sớm.
        return []

    if payload.get("status") != "success":
        return []

    result = payload.get("data", {}).get("result", [])
    if not result:
        return []

    # Query mong đợi trả về đúng 1 series tổng hợp.
    points = result[0].get("values", [])
    series = []
    for _, value in points:
        try:
            series.append(float(value))
        except (TypeError, ValueError):
            series.append(0.0)
    return series


def _normalize_series(values: List[float], length: int, default: float = 0.0) -> List[float]:
    if not values:
        return [default] * length
    if len(values) >= length:
        return values[-length:]
    pad = [values[0]] * (length - len(values))
    return pad + values


def _build_history_from_prometheus(app_info: AppInfo) -> List[List[float]]:
    end_ts = int(time.time())
    start_ts = end_ts - PROM_QUERY_WINDOW_SECONDS

    ns = app_info.namespace
    canary_svc = app_info.canary_service
    stable_svc = app_info.stable_service

    e_canary = _prom_query_range(
        (
            f"sum(rate(http_requests_total{{namespace=\"{ns}\",service=\"{canary_svc}\",status=~\"5..\"}}[1m]))"
            f" / clamp_min(sum(rate(http_requests_total{{namespace=\"{ns}\",service=\"{canary_svc}\"}}[1m])), 0.001)"
        ),
        start_ts,
        end_ts,
        PROM_QUERY_STEP,
    )
    e_stable = _prom_query_range(
        (
            f"sum(rate(http_requests_total{{namespace=\"{ns}\",service=\"{stable_svc}\",status=~\"5..\"}}[1m]))"
            f" / clamp_min(sum(rate(http_requests_total{{namespace=\"{ns}\",service=\"{stable_svc}\"}}[1m])), 0.001)"
        ),
        start_ts,
        end_ts,
        PROM_QUERY_STEP,
    )
    l_canary = _prom_query_range(
        (
            "histogram_quantile(0.95, "
            f"sum by (le) (rate(http_request_duration_seconds_bucket{{namespace=\"{ns}\",service=\"{canary_svc}\"}}[1m]))"
            ")"
        ),
        start_ts,
        end_ts,
        PROM_QUERY_STEP,
    )
    l_stable = _prom_query_range(
        (
            "histogram_quantile(0.95, "
            f"sum by (le) (rate(http_request_duration_seconds_bucket{{namespace=\"{ns}\",service=\"{stable_svc}\"}}[1m]))"
            ")"
        ),
        start_ts,
        end_ts,
        PROM_QUERY_STEP,
    )
    cpu = _prom_query_range(
        (
            "avg(rate(container_cpu_usage_seconds_total{"
            f"namespace=\"{ns}\",pod=~\"my-app-release-.*\",container!=\"\",container!=\"POD\""
            "}[1m]))"
        ),
        start_ts,
        end_ts,
        PROM_QUERY_STEP,
    )
    mem = _prom_query_range(
        (
            "avg(container_memory_working_set_bytes{"
            f"namespace=\"{ns}\",pod=~\"my-app-release-.*\",container!=\"\",container!=\"POD\""
            "}) / 1048576"
        ),
        start_ts,
        end_ts,
        PROM_QUERY_STEP,
    )
    rps = _prom_query_range(
        (
            f"sum(rate(http_requests_total{{namespace=\"{ns}\",service=~\"{canary_svc}|{stable_svc}\"}}[1m]))"
        ),
        start_ts,
        end_ts,
        PROM_QUERY_STEP,
    )

    e_canary = _normalize_series(e_canary, SEQ_LENGTH)
    e_stable = _normalize_series(e_stable, SEQ_LENGTH)
    l_canary = _normalize_series(l_canary, SEQ_LENGTH)
    l_stable = _normalize_series(l_stable, SEQ_LENGTH)
    cpu = _normalize_series(cpu, SEQ_LENGTH)
    mem = _normalize_series(mem, SEQ_LENGTH)
    rps = _normalize_series(rps, SEQ_LENGTH)

    history = []
    for i in range(SEQ_LENGTH):
        history.append(
            [
                float(app_info.weight),
                e_canary[i],
                e_stable[i],
                l_canary[i],
                l_stable[i],
                cpu[i],
                mem[i],
                rps[i] / 1000.0,
            ]
        )
    return history


def _action_to_traffic_signal(action: int, current_weight: float):
    # Heuristic signal for logs only; Argo still controls actual setWeight steps.
    if action == 0:
        return "increase-fast", min(100.0, current_weight + 10.0)
    if action == 1:
        return "increase-slow", min(100.0, current_weight + 5.0)
    if action == 2:
        return "hold", current_weight
    if action == 3:
        return "decrease", max(0.0, current_weight - 5.0)
    if action == 4:
        return "rollback", 0.0
    return "hold", current_weight

# --- 4. ENDPOINT DỰ ĐOÁN (ĐÃ TINH CHỈNH MAPPING) ---
@app.post("/predict")
async def predict(request: InferenceRequest):
    started_at = time.perf_counter()

    if not MODEL_READY:
        logger.warning("predict model_not_ready")
        raise HTTPException(status_code=503, detail="Model is not ready")

    try:
        data = _build_history_from_prometheus(request.app_info)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Cannot query Prometheus: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build history: {exc}")
    
    input_tensor = torch.FloatTensor([data]).to(DEVICE)

    with torch.no_grad():
        q_values, _ = model(input_tensor)
        action = torch.argmax(q_values).item()

    # --- ĐỒNG BỘ HÓA VỚI ARGO ROLLOUTS TẠI ĐÂY ---
    # 0, 1: AI muốn tiến lên -> Argo trả về "Successful" để nhảy step tiếp theo [cite: 56]
    # 2, 3: AI muốn giữ nguyên hoặc lùi lại -> Argo trả về "Running" để chờ [cite: 58]
    # 4: AI muốn hủy bỏ ngay lập tức -> Argo trả về "Rollback" 
    action_mapping = {
        0: "Successful", # Fast Forward (+10%)
        1: "Successful", # Step Forward (+5%)
        2: "Running",    # Stay
        3: "Running",    # Step Back
        4: "Rollback"    # EMERGENCY ROLLBACK
    }
    
    decision = action_mapping.get(action, "Running")
    confidence = float(torch.softmax(q_values, dim=1).max())

    current_weight = float(request.app_info.weight)
    traffic_signal, suggested_weight = _action_to_traffic_signal(action, current_weight)
    latency_ms = (time.perf_counter() - started_at) * 1000.0

    logger.info(
        (
            "predict app=%s ns=%s current_weight=%.2f action=%d "
            "signal=%s suggested_weight=%.2f decision=%s confidence=%.4f latency_ms=%.1f"
        ),
        request.app_info.name,
        request.app_info.namespace,
        current_weight,
        action,
        traffic_signal,
        suggested_weight,
        decision,
        confidence,
        latency_ms,
    )
    
    return {
        "action_id": action,
        "decision": decision,
        "confidence": confidence,
        "traffic_signal": traffic_signal,
        "suggested_weight": suggested_weight,
        "latency_ms": latency_ms,
    }

@app.get("/health")
def health():
    return {"status": "alive"}