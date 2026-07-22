import os
import time
import asyncio
from typing import Dict, List, Tuple
import logging

import httpx
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from core.feature_pipeline import normalize_raw_metrics
# Đã đổi sang sb3_contrib để dùng PPO+LSTM thay vì TCN tự viết
from sb3_contrib import RecurrentPPO

# --- 1. CẤU HÌNH HỆ THỐNG ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/ppo_lstm_offline_best.zip")
SEQ_LENGTH = int(os.getenv("SEQ_LEN", "30")) 
DEVICE = torch.device("cpu")
PROMETHEUS_URL = os.getenv(
    "PROMETHEUS_URL",
    "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090",
)
PROM_QUERY_STEP = os.getenv("PROM_QUERY_STEP", "15s") 
PROM_QUERY_WINDOW_SECONDS = int(os.getenv("PROM_QUERY_WINDOW_SECONDS", str(SEQ_LENGTH * 15))) 
PROM_QUERY_TIMEOUT_SECONDS = float(os.getenv("PROM_QUERY_TIMEOUT_SECONDS", "5"))

# --- SAFETY GUARDS ---
SAFETY_GUARD_ENABLED = os.getenv("SAFETY_GUARD_ENABLED", "true").lower() == "true"
SAFETY_MIN_RPS = float(os.getenv("SAFETY_MIN_RPS", "3.0"))
SAFETY_RUNNING_ERROR_RATIO = float(os.getenv("SAFETY_RUNNING_ERROR_RATIO", "1.8"))
SAFETY_RUNNING_LAT_RATIO = float(os.getenv("SAFETY_RUNNING_LAT_RATIO", "1.8"))
SAFETY_ROLLBACK_ERROR_RATIO = float(os.getenv("SAFETY_ROLLBACK_ERROR_RATIO", "3.0"))
SAFETY_ROLLBACK_ERROR_GAP = float(os.getenv("SAFETY_ROLLBACK_ERROR_GAP", "0.15"))
SAFETY_ROLLBACK_LAT_RATIO = float(os.getenv("SAFETY_ROLLBACK_LAT_RATIO", "2.5"))
SAFETY_ROLLBACK_LAT_GAP_SEC = float(os.getenv("SAFETY_ROLLBACK_LAT_GAP_SEC", "0.12"))
SAFETY_ROLLBACK_MIN_WEIGHT = float(os.getenv("SAFETY_ROLLBACK_MIN_WEIGHT", "0.05")) # 5%

app = FastAPI(title="Canary AI Agent Service")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ai-agent")

# --- 2. LOAD MÔ HÌNH PPO + LSTM ---
MODEL_READY = False
try:
    # Dùng RecurrentPPO.load thay vì torch.load
    model = RecurrentPPO.load(MODEL_PATH, device=DEVICE)
    MODEL_READY = True
    logger.info("model_load status=success path=%s device=%s", MODEL_PATH, DEVICE)
except Exception as e:
    logger.exception("model_load status=failed path=%s error=%s", MODEL_PATH, e)

# --- 3. ĐỊNH NGHĨA DỮ LIỆU TỪ ARGO ROLLOUTS ---
class WebhookPayload(BaseModel):
    service: str          # vd: paymentservice
    stable_hash: str      # vd: 6f8b5d...
    canary_hash: str      # vd: 9a2c1f...
    target_weight: float  # Phải truyền từ Rollout sang (vd: 20.0)
    namespace: str = "default"

async def _prom_query_range(query: str, start_ts: int, end_ts: int, step: str, empty_as_zero: bool = False) -> List[float]:
    params = {"query": query, "start": start_ts, "end": end_ts, "step": step}
    try:
        async with httpx.AsyncClient(timeout=PROM_QUERY_TIMEOUT_SECONDS) as client:
            response = await client.get(f"{PROMETHEUS_URL}/api/v1/query_range", params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        logger.warning("prom_query status=http_error query='%s' error='%s'", query, exc)
        return []

    if payload.get("status") != "success": return []
    result = payload.get("data", {}).get("result", [])
    if not result: return [0.0] if empty_as_zero else []

    points = result[0].get("values", [])
    series = []
    for _, value in points:
        try:
            series.append(float(value))
        except (TypeError, ValueError):
            series.append(0.0)
    return series

def _normalize_series(values: List[float], length: int, default: float = 0.0) -> List[float]:
    if not values: return [default] * length
    if len(values) >= length: return values[-length:]
    return [values[0]] * (length - len(values)) + values

async def _build_history_from_prometheus(payload: WebhookPayload) -> Tuple[List[List[float]], bool, float, Dict[str, float], Dict[str, float], float]:
    end_ts = int(time.time())
    start_ts = end_ts - PROM_QUERY_WINDOW_SECONDS
    ns = payload.namespace
    svc = payload.service
    c_hash = payload.canary_hash
    s_hash = payload.stable_hash

    # Cập nhật query chuẩn Grafana Beyla
    tasks = [
        # e_canary: Error Ratio cho Canary
        # Lỗi có thể là classification="failure"
        _prom_query_range(f"(sum(rate(response_total{{namespace=\"{ns}\",pod=~\".*{c_hash}.*\",direction=\"inbound\",classification=\"failure\"}}[1m])) or vector(0)) / clamp_min(sum(rate(response_total{{namespace=\"{ns}\",pod=~\".*{c_hash}.*\",direction=\"inbound\"}}[1m])), 0.001)", start_ts, end_ts, PROM_QUERY_STEP, empty_as_zero=True),
        _prom_query_range(f"(sum(rate(response_total{{namespace=\"{ns}\",pod=~\".*{s_hash}.*\",direction=\"inbound\",classification=\"failure\"}}[1m])) or vector(0)) / clamp_min(sum(rate(response_total{{namespace=\"{ns}\",pod=~\".*{s_hash}.*\",direction=\"inbound\"}}[1m])), 0.001)", start_ts, end_ts, PROM_QUERY_STEP, empty_as_zero=True),
        # l_canary: P95 Latency (Linkerd reports in ms, divide by 1000 for seconds)
        _prom_query_range(f"histogram_quantile(0.95, sum by (le) (rate(response_latency_ms_bucket{{namespace=\"{ns}\",pod=~\".*{c_hash}.*\",direction=\"inbound\"}}[1m]))) / 1000", start_ts, end_ts, PROM_QUERY_STEP),
        _prom_query_range(f"histogram_quantile(0.95, sum by (le) (rate(response_latency_ms_bucket{{namespace=\"{ns}\",pod=~\".*{s_hash}.*\",direction=\"inbound\"}}[1m]))) / 1000", start_ts, end_ts, PROM_QUERY_STEP),
        _prom_query_range(f"sum(rate(response_total{{namespace=\"{ns}\",pod=~\".*{c_hash}.*\",direction=\"inbound\"}}[1m]))", start_ts, end_ts, PROM_QUERY_STEP),
        _prom_query_range(f"sum(rate(response_total{{namespace=\"{ns}\",pod=~\".*{s_hash}.*\",direction=\"inbound\"}}[1m]))", start_ts, end_ts, PROM_QUERY_STEP),
        # CPU/Mem vẫn phải lấy từ cAdvisor của K8s dựa vào pod hash
        _prom_query_range(f"avg(rate(container_cpu_usage_seconds_total{{namespace=\"{ns}\",pod=~\".*{c_hash}.*\",container!=\"\",container!=\"POD\"}}[1m]))", start_ts, end_ts, PROM_QUERY_STEP),
        _prom_query_range(f"avg(container_memory_working_set_bytes{{namespace=\"{ns}\",pod=~\".*{c_hash}.*\",container!=\"\",container!=\"POD\"}}) / 1048576", start_ts, end_ts, PROM_QUERY_STEP),
        _prom_query_range(f"avg(rate(container_cpu_usage_seconds_total{{namespace=\"{ns}\",pod=~\".*{s_hash}.*\",container!=\"\",container!=\"POD\"}}[1m]))", start_ts, end_ts, PROM_QUERY_STEP),
        _prom_query_range(f"avg(container_memory_working_set_bytes{{namespace=\"{ns}\",pod=~\".*{s_hash}.*\",container!=\"\",container!=\"POD\"}}) / 1048576", start_ts, end_ts, PROM_QUERY_STEP),
        _prom_query_range(f"sum(rate(response_total{{namespace=\"{ns}\",pod=~\"{svc}.*\",direction=\"inbound\"}}[1m]))", start_ts, end_ts, PROM_QUERY_STEP)
    ]

    results = await asyncio.gather(*tasks)
    (e_canary, e_stable, l_canary, l_stable, canary_rps, stable_rps, cpu_canary, mem_canary, cpu_stable, mem_stable, rps) = results

    data_complete = all(series for series in (l_canary, l_stable, canary_rps, stable_rps, cpu_canary, mem_canary, cpu_stable, mem_stable, rps))
    latest_canary_rps = float(canary_rps[-1]) if canary_rps else 0.0

    e_canary = _normalize_series(e_canary, SEQ_LENGTH)
    e_stable = _normalize_series(e_stable, SEQ_LENGTH)
    l_canary = _normalize_series(l_canary, SEQ_LENGTH)
    l_stable = _normalize_series(l_stable, SEQ_LENGTH)
    cpu_canary = _normalize_series(cpu_canary, SEQ_LENGTH)
    mem_canary = _normalize_series(mem_canary, SEQ_LENGTH)
    cpu_stable = _normalize_series(cpu_stable, SEQ_LENGTH)
    mem_stable = _normalize_series(mem_stable, SEQ_LENGTH)
    rps = _normalize_series(rps, SEQ_LENGTH)

    # Đưa weight về thang 0.0 -> 1.0
    observed_weight = float(payload.target_weight) / 100.0
    
    ch_cpu, ch_mem, ch_lat, ch_err, ch_traffic = [], [], [], [], []
    latest_raw, latest_state = {}, {}

    for i in range(SEQ_LENGTH):
        safe_e_canary = max(e_canary[i], 0.001)
        safe_e_stable = max(e_stable[i], 0.001)
        raw = {
            "weight_pct": observed_weight * 100.0, "e_canary": safe_e_canary, "e_stable": safe_e_stable,
            "l_canary": l_canary[i], "l_stable": l_stable[i], 
            "cpu_canary": cpu_canary[i], "cpu_stable": cpu_stable[i], 
            "mem_canary_mb": mem_canary[i], "mem_stable_mb": mem_stable[i], 
            "rps": rps[i],
        }
        norm = normalize_raw_metrics(raw)
        
        ch_cpu.append(norm.get("cpu_n", 0.0))
        ch_mem.append(norm.get("mem_n", 0.0))
        ch_lat.append(norm.get("l_ratio_n", 0.0))
        ch_err.append(norm.get("e_ratio_n", 0.0))
        ch_traffic.append(norm.get("weight_n", 0.0))
        
        if i == SEQ_LENGTH - 1:
            latest_raw, latest_state = raw, norm

    data = [ch_cpu, ch_mem, ch_lat, ch_err, ch_traffic]
    return data, data_complete, observed_weight, latest_raw, latest_state, latest_canary_rps

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
            return "Rollback", f"severe-error-breach"
        if (l_ratio >= SAFETY_ROLLBACK_LAT_RATIO and l_gap_sec >= SAFETY_ROLLBACK_LAT_GAP_SEC):
             return "Rollback", f"severe-latency-breach"

    if e_ratio >= SAFETY_RUNNING_ERROR_RATIO: return "Running", f"elevated-errors"
    if l_ratio >= SAFETY_RUNNING_LAT_RATIO: return "Running", f"elevated-latency"

    return "", "pass"

# --- 4. ENDPOINT ---
@app.post("/api/v1/decision")
async def get_decision(payload: WebhookPayload):
    if not MODEL_READY:
        raise HTTPException(status_code=503, detail="Model is not ready")

    try:
        data, data_complete, observed_weight, latest_raw, latest_state, latest_canary_rps = await _build_history_from_prometheus(payload)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to build history: {exc}")

    # Nếu dữ liệu chưa thu thập đủ (Prometheus chưa kịp scrape), trả về trạng thái chờ
    if not data_complete:
        return {"action": 0, "decision": "Running"}

    # Chốt chặn Data: Nếu đã có dữ liệu đầy đủ nhưng RPS canary = 0, Rollback ngay
    if observed_weight > 0 and latest_canary_rps == 0.0:
        print("Guard triggered: RPS canary is 0")
        return {"action": 0, "decision": "Stay (Waiting for traffic)"}

    try:
        ch_cpu, ch_mem, ch_lat, ch_err, ch_traffic = data
        obs_channels = [
            ch_cpu,
            ch_mem,
            ch_lat,
            ch_err,
            ch_traffic
        ]
        obs = np.array(obs_channels, dtype=np.float32)
        obs = np.expand_dims(obs, axis=0)

        # Áp dụng VecNormalize để chuẩn hóa dữ liệu theo phân phối lúc train
        try:
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            import gymnasium as gym
            
            # Khởi tạo DummyEnv chỉ để load VecNormalize
            class DummyEnv(gym.Env):
                def __init__(self):
                    super().__init__()
                    self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5, SEQ_LENGTH), dtype=np.float32)
                    self.action_space = gym.spaces.Discrete(3)
                def reset(self, seed=None, options=None):
                    return np.zeros((5, SEQ_LENGTH), dtype=np.float32), {}
                def step(self, action):
                    return np.zeros((5, SEQ_LENGTH), dtype=np.float32), 0.0, False, False, {}
                    
            dummy_venv = DummyVecEnv([lambda: DummyEnv()])
            vec_normalize = VecNormalize.load("models/vec_normalize.pkl", dummy_venv)
            vec_normalize.training = False
            obs = vec_normalize.normalize_obs(obs)
        except Exception as e:
            logger.error(f"Failed to normalize obs: {e}")

        # Inference bằng RecurrentPPO (SB3)
        action_val, _states = model.predict(obs, state=latest_state.get('lstm_states'))
        action_val = int(action_val)
        
        # Đánh giá Guard
        guard_decision, _ = _evaluate_safety_guard(latest_raw, observed_weight)
        
        if guard_decision == "Rollback":
            action_val = 2
        elif guard_decision == "Running" and action_val == 0:
            action_val = 1 # Ép dừng lại (Stay) không cho Promote nếu hơi nguy hiểm
            
        decision = "Stay"
        api_action = 0
        
        if action_val == 2:
            decision = "Rollback"
            api_action = 2
        elif action_val == 1:
            decision = "Promote"
            api_action = 1
        elif action_val == 0:
            decision = "Stay"
            api_action = 0

        logger.info(f"Decision Debug: action_val={action_val}, guard_decision={guard_decision}, api_action={api_action}, canary_rps={latest_canary_rps}")
        logger.info(f"Raw metrics: {latest_raw}")
        logger.info(f"State metrics: {latest_state}")
        return {"action": api_action, "decision": decision}
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

@app.get("/health")
def health():
    return {"status": "alive"}


#force rebuild