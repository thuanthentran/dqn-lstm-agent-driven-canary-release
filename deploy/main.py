import os
import time
from functools import lru_cache
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
K8S_API_HOST = os.getenv("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
K8S_API_PORT = os.getenv("KUBERNETES_SERVICE_PORT_HTTPS", "443")
K8S_API_URL = os.getenv("K8S_API_URL", f"https://{K8S_API_HOST}:{K8S_API_PORT}")
K8S_API_TIMEOUT_SECONDS = float(os.getenv("K8S_API_TIMEOUT_SECONDS", "5"))
K8S_SERVICEACCOUNT_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
K8S_SERVICEACCOUNT_CA_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
MIN_PROM_SERIES_POINTS = int(os.getenv("MIN_PROM_SERIES_POINTS", "1"))

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
    rollout_name: str = "my-app-release"
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


def _normalize_weight(weight: float) -> float:
    if weight > 1.0:
        weight = weight / 100.0
    return max(0.0, min(1.0, weight))


@lru_cache(maxsize=1)
def _k8s_verify_setting():
    return K8S_SERVICEACCOUNT_CA_PATH if os.path.exists(K8S_SERVICEACCOUNT_CA_PATH) else False


@lru_cache(maxsize=1)
def _k8s_headers():
    headers = {}
    if os.path.exists(K8S_SERVICEACCOUNT_TOKEN_PATH):
        with open(K8S_SERVICEACCOUNT_TOKEN_PATH, "r", encoding="utf-8") as token_file:
            token = token_file.read().strip()
        if token:
            headers["Authorization"] = f"Bearer {token}"
    return headers


def _query_rollout_current_weight(app_info: AppInfo) -> float:
    rollout_name = app_info.rollout_name or app_info.name
    url = (
        f"{K8S_API_URL}/apis/argoproj.io/v1alpha1/"
        f"namespaces/{app_info.namespace}/rollouts/{rollout_name}"
    )
    try:
        with httpx.Client(
            timeout=K8S_API_TIMEOUT_SECONDS,
            verify=_k8s_verify_setting(),
            headers=_k8s_headers(),
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Cannot query Rollout status: {exc}")

    status = payload.get("status", {}) if isinstance(payload, dict) else {}
    canary_status = status.get("canary", {}) if isinstance(status, dict) else {}

    for candidate in [canary_status.get("currentWeight"), status.get("currentWeight")]:
        if candidate is None:
            continue
        try:
            return _normalize_weight(float(candidate))
        except (TypeError, ValueError):
            continue

    raise HTTPException(
        status_code=503,
        detail=f"Rollout status for {rollout_name} does not expose current weight",
    )


def _collect_metric_snapshot(values: List[float]) -> dict:
    normalized = _normalize_series(values, SEQ_LENGTH)
    return {
        "count": len(values),
        "series": normalized,
    }


def _build_history_from_prometheus(app_info: AppInfo, current_weight: float):
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

    e_canary_snapshot = _collect_metric_snapshot(e_canary)
    e_stable_snapshot = _collect_metric_snapshot(e_stable)
    l_canary_snapshot = _collect_metric_snapshot(l_canary)
    l_stable_snapshot = _collect_metric_snapshot(l_stable)
    cpu_snapshot = _collect_metric_snapshot(cpu)
    mem_snapshot = _collect_metric_snapshot(mem)
    rps_snapshot = _collect_metric_snapshot(rps)

    weight = _normalize_weight(current_weight)

    history = []
    for i in range(SEQ_LENGTH):
        history.append(
            [
                weight,
                e_canary_snapshot["series"][i],
                e_stable_snapshot["series"][i],
                l_canary_snapshot["series"][i],
                l_stable_snapshot["series"][i],
                cpu_snapshot["series"][i],
                mem_snapshot["series"][i] / 1024.0,
                rps_snapshot["series"][i] / 1000.0,
            ]
        )
    return history, {
        "e_canary": e_canary_snapshot,
        "e_stable": e_stable_snapshot,
        "l_canary": l_canary_snapshot,
        "l_stable": l_stable_snapshot,
        "cpu": cpu_snapshot,
        "mem": mem_snapshot,
        "rps": rps_snapshot,
    }


def _has_enough_prometheus_data(metric_snapshot: dict) -> bool:
    metric_names = ["e_canary", "e_stable", "l_canary", "l_stable", "cpu", "mem", "rps"]
    if any(metric_snapshot[name]["count"] < MIN_PROM_SERIES_POINTS for name in metric_names):
        return False

    observed_values = []
    for name in metric_names:
        observed_values.extend(metric_snapshot[name]["series"])

    return any(abs(value) > 1e-9 for value in observed_values)


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
        current_weight = _query_rollout_current_weight(request.app_info)
    except HTTPException as exc:
        logger.warning(
            "predict rollout=%s app=%s ns=%s rollout_weight_unavailable detail=%s",
            request.app_info.rollout_name,
            request.app_info.name,
            request.app_info.namespace,
            exc.detail,
        )
        return {
            "action_id": -1,
            "decision": "Running",
            "confidence": 0.0,
            "traffic_signal": "hold",
            "suggested_weight": _normalize_weight(request.app_info.weight),
            "latency_ms": (time.perf_counter() - started_at) * 1000.0,
            "reason": "rollout_weight_unavailable",
        }

    try:
        data, metric_snapshot = _build_history_from_prometheus(request.app_info, current_weight)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Cannot query Prometheus: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build history: {exc}")

    if not _has_enough_prometheus_data(metric_snapshot):
        logger.warning(
            (
                "predict rollout=%s app=%s ns=%s current_weight=%.2f insufficient_prometheus_data "
                "counts=%s last_features=%s"
            ),
            request.app_info.rollout_name,
            request.app_info.name,
            request.app_info.namespace,
            current_weight,
            {
                "e_canary": metric_snapshot["e_canary"]["count"],
                "e_stable": metric_snapshot["e_stable"]["count"],
                "l_canary": metric_snapshot["l_canary"]["count"],
                "l_stable": metric_snapshot["l_stable"]["count"],
                "cpu": metric_snapshot["cpu"]["count"],
                "mem": metric_snapshot["mem"]["count"],
                "rps": metric_snapshot["rps"]["count"],
            },
            data[-1],
        )
        return {
            "action_id": -1,
            "decision": "Running",
            "confidence": 0.0,
            "traffic_signal": "hold",
            "suggested_weight": current_weight,
            "latency_ms": (time.perf_counter() - started_at) * 1000.0,
            "reason": "insufficient_prometheus_data",
        }
    
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

    traffic_signal, suggested_weight = _action_to_traffic_signal(action, current_weight)
    latency_ms = (time.perf_counter() - started_at) * 1000.0

    logger.info(
        (
            "predict rollout=%s app=%s ns=%s current_weight=%.2f action=%d signal=%s "
            "suggested_weight=%.2f decision=%s confidence=%.4f latency_ms=%.1f counts=%s last_features=%s"
        ),
        request.app_info.rollout_name,
        request.app_info.name,
        request.app_info.namespace,
        current_weight,
        action,
        traffic_signal,
        suggested_weight,
        decision,
        confidence,
        latency_ms,
        {
            "e_canary": metric_snapshot["e_canary"]["count"],
            "e_stable": metric_snapshot["e_stable"]["count"],
            "l_canary": metric_snapshot["l_canary"]["count"],
            "l_stable": metric_snapshot["l_stable"]["count"],
            "cpu": metric_snapshot["cpu"]["count"],
            "mem": metric_snapshot["mem"]["count"],
            "rps": metric_snapshot["rps"]["count"],
        },
        data[-1],
    )
    
    return {
        "action_id": action,
        "decision": decision,
        "confidence": confidence,
        "traffic_signal": traffic_signal,
        "suggested_weight": suggested_weight,
        "latency_ms": latency_ms,
        "rollout_weight": current_weight,
    }

@app.get("/health")
def health():
    return {"status": "alive"}