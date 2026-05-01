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
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ai-agent")
raw_feature_stats = RunningFeatureStats(RAW_KEYS)
state_feature_stats = RunningFeatureStats(STATE_KEYS)
feature_stats_updates = 0

logger.info(
    "service_boot mode=init model_path=%s prom_url=%s step=%s window_sec=%d seq_length=%d",
    MODEL_PATH, PROMETHEUS_URL, PROM_QUERY_STEP, PROM_QUERY_WINDOW_SECONDS, SEQ_LENGTH
)

# Load model đã huấn luyện
model = DRQN(n_obs=8, n_actions=5).to(DEVICE)
MODEL_READY = False
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    MODEL_READY = True
    logger.info("model_load status=success path=%s device=%s", MODEL_PATH, DEVICE)
except Exception as e:
    logger.exception("model_load status=failed path=%s error=%s", MODEL_PATH, e)

# --- 3. ĐỊNH NGHĨA DỮ LIỆU ---
class AppInfo(BaseModel):
    name: str
    weight: float = 0.0
    namespace: str = "default"
    canary_service: str = "my-app-canary"
    stable_service: str = "my-app-stable"

class InferenceRequest(BaseModel):
    app_info: AppInfo

async def _prom_query_range(
    query: str,
    start_ts: int,
    end_ts: int,
    step: str,
    empty_as_zero: bool = False,
) -> List[float]:
    params = {"query": query, "start": start_ts, "end": end_ts, "step": step}
    query_started_at = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=PROM_QUERY_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{PROMETHEUS_URL}/api/v1/query_range", params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        logger.warning("prom_query status=http_error query='%s' error='%s'", query, exc)
        return []

    if payload.get("status") != "success":
        logger.warning("prom_query status=api_error query='%s' response_status=%s", query, payload.get("status"))
        return []

    result = payload.get("data", {}).get("result", [])
    if not result:
        if empty_as_zero:
            return [0.0]
        logger.debug("prom_query status=empty_result query='%s'", query)
        return []

    points = result[0].get("values", [])
    series = []
    for _, value in points:
        try:
            series.append(float(value))
        except (TypeError, ValueError):
            series.append(0.0)

    logger.debug(
        "prom_query status=success points=%d duration_ms=%.1f query='%s'",
        len(series), (time.perf_counter() - query_started_at) * 1000.0, query
    )
    return series

def _normalize_series(values: List[float], length: int, default: float = 0.0) -> List[float]:
    if not values: return [default] * length
    if len(values) >= length: return values[-length:]
    return [values[0]] * (length - len(values)) + values

def _latest_value(values: List[float], default: float = 0.0) -> float:
    return float(values[-1]) if values else default

async def _build_history_from_prometheus(app_info: AppInfo) -> Tuple[List[List[float]], bool, float, Dict[str, float], Dict[str, float], float]:
    build_started_at = time.perf_counter()
    end_ts = int(time.time())
    start_ts = end_ts - PROM_QUERY_WINDOW_SECONDS
    ns, canary_svc, stable_svc = app_info.namespace, app_info.canary_service, app_info.stable_service

    tasks = [
        _prom_query_range(f"sum(rate(http_requests_total{{namespace=\"{ns}\",service=\"{canary_svc}\",status=~\"5..\"}}[1m])) / clamp_min(sum(rate(http_requests_total{{namespace=\"{ns}\",service=\"{canary_svc}\"}}[1m])), 0.001)", start_ts, end_ts, PROM_QUERY_STEP, empty_as_zero=True),
        _prom_query_range(f"sum(rate(http_requests_total{{namespace=\"{ns}\",service=\"{stable_svc}\",status=~\"5..\"}}[1m])) / clamp_min(sum(rate(http_requests_total{{namespace=\"{ns}\",service=\"{stable_svc}\"}}[1m])), 0.001)", start_ts, end_ts, PROM_QUERY_STEP, empty_as_zero=True),
        _prom_query_range(f"histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{{namespace=\"{ns}\",service=\"{canary_svc}\"}}[1m])))", start_ts, end_ts, PROM_QUERY_STEP),
        _prom_query_range(f"histogram_quantile(0.95, sum by (le) (rate(http_request_duration_seconds_bucket{{namespace=\"{ns}\",service=\"{stable_svc}\"}}[1m])))", start_ts, end_ts, PROM_QUERY_STEP),
        _prom_query_range(f"sum(rate(http_requests_total{{namespace=\"{ns}\",service=\"{canary_svc}\"}}[1m]))", start_ts, end_ts, PROM_QUERY_STEP),
        _prom_query_range(f"sum(rate(http_requests_total{{namespace=\"{ns}\",service=\"{stable_svc}\"}}[1m]))", start_ts, end_ts, PROM_QUERY_STEP),
        _prom_query_range(f"avg(rate(container_cpu_usage_seconds_total{{namespace=\"{ns}\",pod=~\"my-app-release-.*\",container!=\"\",container!=\"POD\"}}[1m]))", start_ts, end_ts, PROM_QUERY_STEP),
        _prom_query_range(f"avg(container_memory_working_set_bytes{{namespace=\"{ns}\",pod=~\"my-app-release-.*\",container!=\"\",container!=\"POD\"}}) / 1048576", start_ts, end_ts, PROM_QUERY_STEP),
        _prom_query_range(f"sum(rate(http_requests_total{{namespace=\"{ns}\",service=~\"{canary_svc}|{stable_svc}\"}}[1m]))", start_ts, end_ts, PROM_QUERY_STEP)
    ]

    results = await asyncio.gather(*tasks)
    (e_canary, e_stable, l_canary, l_stable, canary_rps, stable_rps, cpu, mem, rps) = results

    data_complete = all(series for series in (l_canary, l_stable, canary_rps, stable_rps, cpu, mem, rps))
    latest_canary_rps = _latest_value(canary_rps)

    e_canary = _normalize_series(e_canary, SEQ_LENGTH)
    e_stable = _normalize_series(e_stable, SEQ_LENGTH)
    l_canary = _normalize_series(l_canary, SEQ_LENGTH)
    l_stable = _normalize_series(l_stable, SEQ_LENGTH)
    cpu = _normalize_series(cpu, SEQ_LENGTH)
    mem = _normalize_series(mem, SEQ_LENGTH)
    rps = _normalize_series(rps, SEQ_LENGTH)

    observed_weight = float(app_info.weight)
    history, latest_raw, latest_state = [], {}, {}

    for i in range(SEQ_LENGTH):
        raw = {
            "weight_pct": observed_weight, "e_canary": e_canary[i], "e_stable": e_stable[i],
            "l_canary": l_canary[i], "l_stable": l_stable[i], "cpu": cpu[i], "mem_mb": mem[i], "rps": rps[i],
        }
        history.append(to_state_vector(raw))
        if i == SEQ_LENGTH - 1:
            latest_raw, latest_state = raw, normalize_raw_metrics(raw)

    logger.info(
        "history_build app=%s target_weight=%.1f data_complete=%s duration_ms=%.1f metrics_raw=%s metrics_norm=%s",
        app_info.name, observed_weight, data_complete, (time.perf_counter() - build_started_at) * 1000.0, latest_raw, latest_state
    )
    return history, data_complete, observed_weight, latest_raw, latest_state, latest_canary_rps

def _action_to_traffic_signal(action: int, current_weight: float):
    if action == 0: return "increase-fast", min(100.0, current_weight + 10.0)
    if action == 1: return "increase-slow", min(100.0, current_weight + 5.0)
    if action == 2: return "hold", current_weight
    if action == 3: return "rollback", max(0.0, current_weight - 5.0)
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
        if (e_ratio >= SAFETY_ROLLBACK_ERROR_RATIO and e_gap >= SAFETY_ROLLBACK_ERROR_GAP):
            return "Rollback", f"severe-error-breach (e_ratio={e_ratio:.2f}, e_gap={e_gap:.2f})"
        if (l_ratio >= SAFETY_ROLLBACK_LAT_RATIO and l_gap_sec >= SAFETY_ROLLBACK_LAT_GAP_SEC):
             return "Rollback", f"severe-latency-breach (l_ratio={l_ratio:.2f}, l_gap={l_gap_sec:.2f}s)"

    if e_ratio >= SAFETY_RUNNING_ERROR_RATIO: return "Running", f"elevated-errors (e_ratio={e_ratio:.2f})"
    if l_ratio >= SAFETY_RUNNING_LAT_RATIO: return "Running", f"elevated-latency (l_ratio={l_ratio:.2f})"

    return "", "pass"

# --- 4. ENDPOINT DỰ ĐOÁN ---
@app.post("/predict")
async def predict(request: InferenceRequest):
    started_at = time.perf_counter()
    app_name, weight = request.app_info.name, float(request.app_info.weight)

    logger.info("predict_start app=%s target_weight=%.1f", app_name, weight)

    if not MODEL_READY:
        logger.error("predict_abort app=%s reason=model_not_ready", app_name)
        raise HTTPException(status_code=503, detail="Model is not ready")

    try:
        data, data_complete, observed_weight, latest_raw, latest_state, latest_canary_rps = await _build_history_from_prometheus(request.app_info)
    except Exception as exc:
        logger.exception("predict_abort app=%s reason=history_build_failed error='%s'", app_name, exc)
        raise HTTPException(status_code=500, detail=f"Failed to build history: {exc}")

    # --- CHỐT CHẶN DEAD POD ---
    if observed_weight > 0 and latest_canary_rps == 0.0:
        logger.warning("predict_decision app=%s target_weight=%.1f decision=Rollback reason='canary_dead_no_metrics'", app_name, weight)
        return {
            "action_id": 4, "decision": "Rollback", "confidence": 1.0, "traffic_signal": "rollback",
            "suggested_weight": 0.0, "latency_ms": (time.perf_counter() - started_at) * 1000.0,
        }

    # --- CHỐT CHẶN THIẾU DATA ---
    if not data_complete:
        logger.info("predict_decision app=%s target_weight=%.1f decision=Running reason='insufficient_data'", app_name, weight)
        return {
            "action_id": -1, "decision": "Running", "confidence": 0.0, "traffic_signal": "hold",
            "suggested_weight": observed_weight, "latency_ms": (time.perf_counter() - started_at) * 1000.0,
        }

    # --- AI INFERENCE ---
    input_tensor = torch.FloatTensor([data]).to(DEVICE)
    with torch.no_grad():
        q_values, _ = model(input_tensor)
        action = torch.argmax(q_values).item()
    
    q_values_list = [round(float(v), 2) for v in q_values.squeeze(0).tolist()]
    confidence = float(torch.softmax(q_values, dim=1).max())
    
    action_mapping = {0: "Successful", 1: "Successful", 2: "Running", 3: "Running", 4: "Rollback"}
    model_decision = action_mapping.get(action, "Running")
    
    logger.info("model_inference app=%s q_values=%s chosen_action=%d model_decision=%s confidence=%.3f", app_name, q_values_list, action, model_decision, confidence)

    # --- TÍNH TOÁN QUYẾT ĐỊNH CUỐI CÙNG VỚI SAFETY GUARD ---
    final_decision = model_decision
    reason = "model_approved"
    
    guard_decision, guard_reason = _evaluate_safety_guard(latest_raw, observed_weight)
    
    if guard_decision:
        final_decision = guard_decision
        reason = f"safety_guard_override ({guard_reason})"
        logger.warning("safety_override app=%s target_weight=%.1f model_wanted=%s guard_forced=%s reason='%s'", app_name, weight, model_decision, final_decision, guard_reason)

    traffic_signal, suggested_weight = _action_to_traffic_signal(action, observed_weight)

    logger.info(
        "predict_finish app=%s target_weight=%.1f final_decision=%s action_id=%d signal=%s latency_ms=%.1f reason='%s'",
        app_name, weight, final_decision, action, traffic_signal, (time.perf_counter() - started_at) * 1000.0, reason
    )
    
    return {
        "action_id": action,
        "decision": final_decision,
        "confidence": confidence,
        "traffic_signal": traffic_signal,
        "suggested_weight": suggested_weight,
        "latency_ms": (time.perf_counter() - started_at) * 1000.0,
    }

@app.get("/health")
def health():
    return {"status": "alive"}