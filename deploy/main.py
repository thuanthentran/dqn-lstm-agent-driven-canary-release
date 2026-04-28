import os
import time
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
        # Lấy timestep cuối cùng của chuỗi (Sequence length = 10) [cite: 35, 81]
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

logger.info(
    "service_boot config model_path=%s prom_url=%s step=%s window_seconds=%d timeout_seconds=%.1f seq_length=%d device=%s",
    MODEL_PATH,
    PROMETHEUS_URL,
    PROM_QUERY_STEP,
    PROM_QUERY_WINDOW_SECONDS,
    PROM_QUERY_TIMEOUT_SECONDS,
    SEQ_LENGTH,
    DEVICE,
)

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


def _prom_query_range(
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
    logger.info(
        "prom_query_start start_ts=%d end_ts=%d step=%s query=%s",
        start_ts,
        end_ts,
        step,
        query,
    )
    query_started_at = time.perf_counter()
    try:
        with httpx.Client(timeout=PROM_QUERY_TIMEOUT_SECONDS) as client:
            response = client.get(f"{PROMETHEUS_URL}/api/v1/query_range", params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        # Trả về rỗng để agent vẫn phản hồi /predict, tránh làm AnalysisRun lỗi sớm.
        logger.warning(
            "prom_query_http_error query=%s error=%s",
            query,
            exc,
        )
        return []

    if payload.get("status") != "success":
        logger.warning(
            "prom_query_bad_status query=%s status=%s",
            query,
            payload.get("status"),
        )
        return []

    result = payload.get("data", {}).get("result", [])
    if not result:
        if empty_as_zero:
            logger.info("prom_query_empty_result_as_zero query=%s", query)
            return [0.0]
        logger.warning("prom_query_empty_result query=%s", query)
        return []

    # Query mong đợi trả về đúng 1 series tổng hợp.
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
        logger.warning(
            "normalize_series_empty default_applied length=%d default=%.3f",
            length,
            default,
        )
        return [default] * length
    if len(values) >= length:
        logger.info(
            "normalize_series_trim original_len=%d target_len=%d",
            len(values),
            length,
        )
        return values[-length:]
    pad = [values[0]] * (length - len(values))
    logger.info(
        "normalize_series_pad original_len=%d target_len=%d pad_count=%d",
        len(values),
        length,
        len(pad),
    )
    return pad + values


def _latest_value(values: List[float], default: float = 0.0) -> float:
    if not values:
        return default
    return float(values[-1])


def _build_history_from_prometheus(app_info: AppInfo) -> Tuple[List[List[float]], bool, float, Dict[str, float], Dict[str, float]]:
    build_started_at = time.perf_counter()
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
        empty_as_zero=True,
    )
    e_stable = _prom_query_range(
        (
            f"sum(rate(http_requests_total{{namespace=\"{ns}\",service=\"{stable_svc}\",status=~\"5..\"}}[1m]))"
            f" / clamp_min(sum(rate(http_requests_total{{namespace=\"{ns}\",service=\"{stable_svc}\"}}[1m])), 0.001)"
        ),
        start_ts,
        end_ts,
        PROM_QUERY_STEP,
        empty_as_zero=True,
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
    canary_rps = _prom_query_range(
        f"sum(rate(http_requests_total{{namespace=\"{ns}\",service=\"{canary_svc}\"}}[1m]))",
        start_ts,
        end_ts,
        PROM_QUERY_STEP,
    )
    stable_rps = _prom_query_range(
        f"sum(rate(http_requests_total{{namespace=\"{ns}\",service=\"{stable_svc}\"}}[1m]))",
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

    # Required live signals for safe inference. Error-rate queries can legitimately be empty when there are no 5xx.
    data_complete = all(
        series
        for series in (
            l_canary,
            l_stable,
            canary_rps,
            stable_rps,
            cpu,
            mem,
            rps,
        )
    )

    e_canary = _normalize_series(e_canary, SEQ_LENGTH)
    e_stable = _normalize_series(e_stable, SEQ_LENGTH)
    l_canary = _normalize_series(l_canary, SEQ_LENGTH)
    l_stable = _normalize_series(l_stable, SEQ_LENGTH)
    canary_rps = _normalize_series(canary_rps, SEQ_LENGTH)
    stable_rps = _normalize_series(stable_rps, SEQ_LENGTH)
    cpu = _normalize_series(cpu, SEQ_LENGTH)
    mem = _normalize_series(mem, SEQ_LENGTH)
    rps = _normalize_series(rps, SEQ_LENGTH)

    latest_canary_rps = _latest_value(canary_rps)
    latest_stable_rps = _latest_value(stable_rps)
    latest_total_rps = latest_canary_rps + latest_stable_rps
    observed_weight = float(app_info.weight)
    if latest_total_rps > 0.0:
        observed_weight = (latest_canary_rps / latest_total_rps) * 100.0

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

    logger.info(
        "history_build_success app=%s ns=%s samples=%d duration_ms=%.1f data_complete=%s latest_raw={reported_weight:%.2f,observed_weight:%.2f,e_canary:%.4f,e_stable:%.4f,l_canary:%.4f,l_stable:%.4f,cpu:%.4f,mem:%.2f,rps:%.4f} latest_state={weight_n:%.4f,e_ratio_n:%.4f,l_ratio_n:%.4f,e_gap_n:%.4f,l_gap_n:%.4f,cpu_n:%.4f,mem_n:%.4f,rps_n:%.4f}",
        app_info.name,
        app_info.namespace,
        len(history),
        (time.perf_counter() - build_started_at) * 1000.0,
        data_complete,
        float(app_info.weight),
        latest_raw["weight_pct"],
        latest_raw["e_canary"],
        latest_raw["e_stable"],
        latest_raw["l_canary"],
        latest_raw["l_stable"],
        latest_raw["cpu"],
        latest_raw["mem_mb"],
        latest_raw["rps"],
        latest_state["weight_n"],
        latest_state["e_ratio_n"],
        latest_state["l_ratio_n"],
        latest_state["e_gap_n"],
        latest_state["l_gap_n"],
        latest_state["cpu_n"],
        latest_state["mem_n"],
        latest_state["rps_n"],
    )
    return history, data_complete, observed_weight, latest_raw, latest_state


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


def _evaluate_safety_guard(latest_raw: Dict[str, float], observed_weight: float) -> Tuple[str, str]:
    if not SAFETY_GUARD_ENABLED:
        return "", "disabled"

    rps = float(latest_raw.get("rps", 0.0))
    if rps < SAFETY_MIN_RPS:
        return "", "insufficient-rps"

    e_canary = float(latest_raw.get("e_canary", 0.0))
    e_stable = float(latest_raw.get("e_stable", 0.0))
    l_canary = float(latest_raw.get("l_canary", 0.0))
    l_stable = float(latest_raw.get("l_stable", 0.0))

    e_ratio = e_canary / max(e_stable, 1e-6)
    l_ratio = l_canary / max(l_stable, 1e-6)
    e_gap = max(0.0, e_canary - e_stable)
    l_gap_sec = max(0.0, l_canary - l_stable)

    severe_error = e_ratio >= SAFETY_ROLLBACK_ERROR_RATIO and e_gap >= SAFETY_ROLLBACK_ERROR_GAP
    severe_latency = l_ratio >= SAFETY_ROLLBACK_LAT_RATIO and l_gap_sec >= SAFETY_ROLLBACK_LAT_GAP_SEC
    if observed_weight >= SAFETY_ROLLBACK_MIN_WEIGHT and (severe_error or severe_latency):
        reason = (
            f"rollback:e_ratio={e_ratio:.2f},e_gap={e_gap:.4f},"
            f"l_ratio={l_ratio:.2f},l_gap_sec={l_gap_sec:.3f},rps={rps:.2f},weight={observed_weight:.2f}"
        )
        return "Rollback", reason

    elevated_error = e_ratio >= SAFETY_RUNNING_ERROR_RATIO
    elevated_latency = l_ratio >= SAFETY_RUNNING_LAT_RATIO
    if elevated_error or elevated_latency:
        reason = (
            f"running:e_ratio={e_ratio:.2f},l_ratio={l_ratio:.2f},"
            f"rps={rps:.2f},weight={observed_weight:.2f}"
        )
        return "Running", reason

    return "", "pass"

# --- 4. ENDPOINT DỰ ĐOÁN (ĐÃ TINH CHỈNH MAPPING) ---
@app.post("/predict")
async def predict(request: InferenceRequest):
    global feature_stats_updates
    started_at = time.perf_counter()
    incoming_weight = float(request.app_info.weight)
    logger.info(
        "predict_request_received app=%s ns=%s incoming_weight=%.2f canary_service=%s stable_service=%s",
        request.app_info.name,
        request.app_info.namespace,
        incoming_weight,
        request.app_info.canary_service,
        request.app_info.stable_service,
    )

    if not MODEL_READY:
        logger.warning("predict model_not_ready")
        raise HTTPException(status_code=503, detail="Model is not ready")

    try:
        data, data_complete, observed_weight, latest_raw, latest_state = _build_history_from_prometheus(request.app_info)
    except httpx.HTTPError as exc:
        logger.exception("predict_prometheus_query_failed app=%s error=%s", request.app_info.name, exc)
        raise HTTPException(status_code=502, detail=f"Cannot query Prometheus: {exc}")
    except Exception as exc:
        logger.exception("predict_build_history_failed app=%s error=%s", request.app_info.name, exc)
        raise HTTPException(status_code=500, detail=f"Failed to build history: {exc}")

    raw_feature_stats.update(latest_raw)
    state_feature_stats.update(latest_state)
    feature_stats_updates += 1
    if feature_stats_updates % FEATURE_STATS_LOG_EVERY == 0:
        logger.info(
            "feature_distribution_snapshot samples=%d raw=%s state=%s",
            feature_stats_updates,
            raw_feature_stats.summary(),
            state_feature_stats.summary(),
        )

    if not data_complete:
        latency_ms = (time.perf_counter() - started_at) * 1000.0
        logger.warning(
            "predict_insufficient_metrics app=%s ns=%s incoming_weight=%.2f observed_weight=%.2f decision=Running latency_ms=%.1f",
            request.app_info.name,
            request.app_info.namespace,
            incoming_weight,
            observed_weight,
            latency_ms,
        )
        return {
            "action_id": -1,
            "decision": "Running",
            "confidence": 0.0,
            "traffic_signal": "hold",
            "suggested_weight": observed_weight,
            "latency_ms": latency_ms,
        }
    
    input_tensor = torch.FloatTensor([data]).to(DEVICE)
    logger.info("predict_input_tensor_ready shape=%s device=%s", tuple(input_tensor.shape), DEVICE)

    with torch.no_grad():
        q_values, _ = model(input_tensor)
        action = torch.argmax(q_values).item()

    q_values_list = [float(v) for v in q_values.squeeze(0).tolist()]
    logger.info("predict_model_output q_values=%s chosen_action=%d", q_values_list, action)

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

    guard_decision, guard_reason = _evaluate_safety_guard(latest_raw, observed_weight)
    if guard_decision:
        logger.warning(
            "safety_guard_override app=%s ns=%s model_action=%d model_decision=%s override_decision=%s reason=%s",
            request.app_info.name,
            request.app_info.namespace,
            action,
            decision,
            guard_decision,
            guard_reason,
        )
        decision = guard_decision

    traffic_signal, suggested_weight = _action_to_traffic_signal(action, observed_weight)
    latency_ms = (time.perf_counter() - started_at) * 1000.0

    logger.info(
        (
            "predict app=%s ns=%s incoming_weight=%.2f observed_weight=%.2f action=%d "
            "signal=%s suggested_weight=%.2f decision=%s confidence=%.4f latency_ms=%.1f"
        ),
        request.app_info.name,
        request.app_info.namespace,
        incoming_weight,
        observed_weight,
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
    logger.info("health_check status=alive")
    return {"status": "alive"}