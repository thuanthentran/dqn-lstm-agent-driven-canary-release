import os
import time
import asyncio
from typing import Dict, List, Tuple
import logging

import httpx
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.feature_pipeline import RAW_KEYS, STATE_KEYS, RunningFeatureStats, normalize_raw_metrics, to_state_vector

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
        x = self.fc2(x[:, -1, :]) 
        return x, hidden

# --- 2. CẤU HÌNH HỆ THỐNG ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/model_canary_drqn_best.pth")
SEQ_LENGTH = 10
DEVICE = torch.device("cpu")
PROMETHEUS_URL = os.getenv(
    "PROMETHEUS_URL",
    "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090",
)
PROM_QUERY_STEP = os.getenv("PROM_QUERY_STEP", "30s")
PROM_QUERY_WINDOW_SECONDS = int(os.getenv("PROM_QUERY_WINDOW_SECONDS", "300"))
PROM_QUERY_TIMEOUT_SECONDS = float(os.getenv("PROM_QUERY_TIMEOUT_SECONDS", "5"))
FEATURE_STATS_LOG_EVERY = int(os.getenv("FEATURE_STATS_LOG_EVERY", "20"))
SAFETY_GUARD_ENABLED = os.getenv("SAFETY_GUARD_ENABLED", "true").lower() == "true"
SAFETY_MIN_RPS = float(os.getenv("SAFETY_MIN_RPS", "3.0"))
SAFETY_RUNNING_ERROR_RATIO = float(os.getenv("SAFETY_RUNNING_ERROR_RATIO", "1.8"))
SAFETY_RUNNING_LAT_RATIO = float(os.getenv("SAFETY_RUNNING_LAT_RATIO", "1.8"))
SAFETY_ROLLBACK_ERROR_RATIO = float(os.getenv("SAFETY_ROLLBACK_ERROR_RATIO", "3.0"))
SAFETY_ROLLBACK_ERROR_GAP = float(os.getenv("SAFETY_ROLLBACK_ERROR_GAP", "0.15"))
SAFETY_ROLLBACK_LAT_RATIO = float(os.getenv("SAFETY_ROLLBACK_LAT_RATIO", "2.5"))
SAFETY_ROLLBACK_LAT_GAP_SEC = float(os.getenv("SAFETY_ROLLBACK_LAT_GAP_SEC", "0.12"))
SAFETY_ROLLBACK_MIN_WEIGHT = float(os.getenv("SAFETY_ROLLBACK_MIN_WEIGHT", "5.0"))

app = FastAPI(title="Canary AI Agent Service")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("canary-ai-agent")
raw_feature_stats = RunningFeatureStats(RAW_KEYS)
state_feature_stats = RunningFeatureStats(STATE_KEYS)
feature_stats_updates = 0

# Load model đã huấn luyện
model = DRQN(n_obs=8, n_actions=5).to(DEVICE)
MODEL_READY = False
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    MODEL_READY = True
    logger.info("model_load_success path=%s", MODEL_PATH)
except Exception as e:
    logger.exception("model_load_failed path=%s error=%s", MODEL_PATH, e)

# --- 3. ĐỊNH NGHĨA DỮ LIỆU ---
class AppInfo(BaseModel):
    name: str
    weight: float = 0.0
    namespace: str = "default"
    canary_service: str = "my-app-canary"
    stable_service: str = "my-app-stable"

class InferenceRequest(BaseModel):
    app_info: AppInfo

# Chuyển thành async function và dùng httpx.AsyncClient
async def _prom_query_range(
    query: str,
    start_ts: int,
    end_ts: int,
    step: str,
    empty_as_zero: bool = False,
) -> List[float]:
    params = {
        "query": query,
        "start": start_ts,
        "end": end_ts,
        "step": step,
    }
    query_started_at = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=PROM_QUERY_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{PROMETHEUS_URL}/api/v1/query_range", params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        logger.warning("prom_query_http_error query=%s error=%s", query, exc)
        return []

    if payload.get("status") != "success":
        logger.warning("prom_query_bad_status query=%s status=%s", query, payload.get("status"))
        return []

    result = payload.get("data", {}).get("result", [])
    if not result:
        if empty_as_zero:
            return [0.0]
        logger.warning("prom_query_empty_result query=%s", query)
        return []

    points = result[0].get("values", [])
    series = []
    for _, value in points:
        try:
            series.append(float(value))
        except (TypeError, ValueError):
            series.append(0.0)

    logger.info(
        "prom_query_success points=%d duration_ms=%.1f query=%s",
        len(series),
        (time.perf_counter() - query_started_at) * 1000.0,
        query,
    )
    return series

def _normalize_series(values: List[float], length: int, default: float = 0.0) -> List[float]:
    if not values:
        return [default] * length
    if len(values) >= length:
        return values[-length:]
    pad = [values[0]] * (length - len(values))
    return pad + values

def _latest_value(values: List[float], default: float = 0.0) -> float:
    if not values:
        return default
    return float(values[-1])

# Chuyển thành async function
async def _build_history_from_prometheus(app_info: AppInfo) -> Tuple[List[List[float]], bool, float, Dict[str, float], Dict[str, float], float]:
    build_started_at = time.perf_counter()
    end_ts = int(time.time())
    start_ts = end_ts - PROM_QUERY_WINDOW_SECONDS

    ns = app_info.namespace
    canary_svc = app_info.canary_service
    stable_svc = app_info.stable_service

    # Tạo danh sách các task chạy song song (Concurrency)
    tasks = [
        _prom_query_range(
            f"sum(rate(http_requests_total{{namespace=\"{ns}\",service=\"{canary_svc}\",status=~\"5..\"}}[1m])) / clamp_min(sum(rate(http_requests_total{{namespace=\"{ns}\",service=\"{canary_svc}\"}}[1m])), 0.001)",
            start_ts, end_ts, PROM_QUERY_STEP, empty_as_zero=True
        ),
        _prom_query_range(
            f"sum(rate(http_requests_total{{namespace=\"{ns}\",service=\"{stable_svc}\",status=~\"5..\"}}[1m])) / clamp_min(sum(rate(http_requests_total{{namespace=\"{ns}\",service=\"{stable_svc}\"}}[1m])), 0.001)",
            start_ts, end_ts, PROM_QUERY_STEP, empty_as_zero=True
        ),
        _prom_query_range(
            f"histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{{namespace=\"{ns}\",service=\"{canary_svc}\"}}[1m])))",
            start_ts, end_ts, PROM_QUERY_STEP
        ),
        _prom_query_range(
            f"histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{{namespace=\"{ns}\",service=\"{stable_svc}\"}}[1m])))",
            start_ts, end_ts, PROM_QUERY_STEP
        ),
        _prom_query_range(
            f"sum(rate(http_requests_total{{namespace=\"{ns}\",service=\"{canary_svc}\"}}[1m]))",
            start_ts, end_ts, PROM_QUERY_STEP
        ),
        _prom_query_range(
            f"sum(rate(http_requests_total{{namespace=\"{ns}\",service=\"{stable_svc}\"}}[1m]))",
            start_ts, end_ts, PROM_QUERY_STEP
        ),
        _prom_query_range(
            f"avg(rate(container_cpu_usage_seconds_total{{namespace=\"{ns}\",pod=~\"my-app-release-.*\",container!=\"\",container!=\"POD\"}}[1m]))",
            start_ts, end_ts, PROM_QUERY_STEP
        ),
        _prom_query_range(
            f"avg(container_memory_working_set_bytes{{namespace=\"{ns}\",pod=~\"my-app-release-.*\",container!=\"\",container!=\"POD\"}}) / 1048576",
            start_ts, end_ts, PROM_QUERY_STEP
        ),
        _prom_query_range(
            f"sum(rate(http_requests_total{{namespace=\"{ns}\",service=~\"{canary_svc}|{stable_svc}\"}}[1m]))",
            start_ts, end_ts, PROM_QUERY_STEP
        )
    ]

    # Thực thi toàn bộ HTTP requests cùng một lúc
    results = await asyncio.gather(*tasks)
    (e_canary, e_stable, l_canary, l_stable, canary_rps, stable_rps, cpu, mem, rps) = results

    data_complete = all(series for series in (l_canary, l_stable, canary_rps, stable_rps, cpu, mem, rps))

    latest_canary_rps = _latest_value(canary_rps) # Lấy giá trị để check Dead Pod

    e_canary = _normalize_series(e_canary, SEQ_LENGTH)
    e_stable = _normalize_series(e_stable, SEQ_LENGTH)
    l_canary = _normalize_series(l_canary, SEQ_LENGTH)
    l_stable = _normalize_series(l_stable, SEQ_LENGTH)
    cpu = _normalize_series(cpu, SEQ_LENGTH)
    mem = _normalize_series(mem, SEQ_LENGTH)
    rps = _normalize_series(rps, SEQ_LENGTH)

    observed_weight = float(app_info.weight)

    history = []
    latest_raw: Dict[str, float] = {}
    latest_state: Dict[str, float] = {}
    for i in range(SEQ_LENGTH):
        raw = {
            "weight_pct": observed_weight,
            "e_canary": e_canary[i],
            "e_stable": e_stable[i],
            "l_canary": l_canary[i],
            "l_stable": l_stable[i],
            "cpu": cpu[i],
            "mem_mb": mem[i],
            "rps": rps[i],
        }
        history.append(to_state_vector(raw))
        if i == SEQ_LENGTH - 1:
            latest_raw = raw
            latest_state = normalize_raw_metrics(raw)

    logger.info("history_build_success duration_ms=%.1f data_complete=%s", (time.perf_counter() - build_started_at) * 1000.0, data_complete)
    return history, data_complete, observed_weight, latest_raw, latest_state, latest_canary_rps


def _action_to_traffic_signal(action: int, current_weight: float):
    if action == 0: return "increase-fast", min(100.0, current_weight + 10.0)
    if action == 1: return "increase-slow", min(100.0, current_weight + 5.0)
    if action == 2: return "hold", current_weight
    if action == 3: return "decrease", max(0.0, current_weight - 5.0)
    if action == 4: return "rollback", 0.0
    return "hold", current_weight


def _evaluate_safety_guard(latest_raw: Dict[str, float], observed_weight: float) -> Tuple[str, str]:
    if not SAFETY_GUARD_ENABLED: return "", "disabled"
    
    rps = float(latest_raw.get("rps", 0.0))
    if rps < SAFETY_MIN_RPS: return "", "insufficient-rps"

    e_ratio = float(latest_raw.get("e_canary", 0.0)) / max(float(latest_raw.get("e_stable", 0.0)), 1e-6)
    l_ratio = float(latest_raw.get("l_canary", 0.0)) / max(float(latest_raw.get("l_stable", 0.0)), 1e-6)
    e_gap = max(0.0, float(latest_raw.get("e_canary", 0.0)) - float(latest_raw.get("e_stable", 0.0)))
    l_gap_sec = max(0.0, float(latest_raw.get("l_canary", 0.0)) - float(latest_raw.get("l_stable", 0.0)))

    if observed_weight >= SAFETY_ROLLBACK_MIN_WEIGHT:
        if (e_ratio >= SAFETY_ROLLBACK_ERROR_RATIO and e_gap >= SAFETY_ROLLBACK_ERROR_GAP) or \
           (l_ratio >= SAFETY_ROLLBACK_LAT_RATIO and l_gap_sec >= SAFETY_ROLLBACK_LAT_GAP_SEC):
            return "Rollback", "severe-threshold-breach"

    if e_ratio >= SAFETY_RUNNING_ERROR_RATIO or l_ratio >= SAFETY_RUNNING_LAT_RATIO:
        return "Running", "elevated-metrics"

    return "", "pass"

# --- 4. ENDPOINT DỰ ĐOÁN ---
@app.post("/predict")
async def predict(request: InferenceRequest):
    global feature_stats_updates
    started_at = time.perf_counter()
    incoming_weight = float(request.app_info.weight)

    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model is not ready")

    try:
        # Await luồng xử lý bất đồng bộ
        data, data_complete, observed_weight, latest_raw, latest_state, latest_canary_rps = await _build_history_from_prometheus(request.app_info)
    except Exception as exc:
        logger.exception("predict_build_history_failed app=%s error=%s", request.app_info.name, exc)
        raise HTTPException(status_code=500, detail=f"Failed to build history: {exc}")

    # --- CHỐT CHẶN DEAD POD ---
    # Nếu weight > 0 (đã có traffic) nhưng canary_rps = 0 sau 1 phút chờ, Pod đã crash ngầm
    if observed_weight > 0 and latest_canary_rps == 0.0:
        logger.error("canary_dead: Traffic routed but no metrics found. Triggering fallback Rollback.")
        return {
            "action_id": 4,
            "decision": "Rollback",
            "confidence": 1.0,
            "traffic_signal": "rollback",
            "suggested_weight": 0.0,
            "latency_ms": (time.perf_counter() - started_at) * 1000.0,
        }

    # --- CHỐT CHẶN THIẾU DATA ---
    if not data_complete:
        return {
            "action_id": -1,
            "decision": "Running",
            "confidence": 0.0,
            "traffic_signal": "hold",
            "suggested_weight": observed_weight,
            "latency_ms": (time.perf_counter() - started_at) * 1000.0,
        }

    raw_feature_stats.update(latest_raw)
    state_feature_stats.update(latest_state)

    input_tensor = torch.FloatTensor([data]).to(DEVICE)

    with torch.no_grad():
        q_values, _ = model(input_tensor)
        action = torch.argmax(q_values).item()

    action_mapping = {0: "Successful", 1: "Successful", 2: "Running", 3: "Running", 4: "Rollback"}
    decision = action_mapping.get(action, "Running")
    confidence = float(torch.softmax(q_values, dim=1).max())

    guard_decision, guard_reason = _evaluate_safety_guard(latest_raw, observed_weight)
    if guard_decision:
        decision = guard_decision

    traffic_signal, suggested_weight = _action_to_traffic_signal(action, observed_weight)

    logger.info("predict finished action=%d decision=%s latency_ms=%.1f", action, decision, (time.perf_counter() - started_at) * 1000.0)
    
    return {
        "action_id": action,
        "decision": decision,
        "confidence": confidence,
        "traffic_signal": traffic_signal,
        "suggested_weight": suggested_weight,
        "latency_ms": (time.perf_counter() - started_at) * 1000.0,
    }

@app.get("/health")
def health():
    return {"status": "alive"}