import os
import time
import asyncio
import json
from typing import Dict, List, Optional, Tuple
import logging

import httpx
import torch
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from core.feature_pipeline import normalize_raw_metrics
from stable_baselines3 import PPO
##trigger build

# --- 1. CẤU HÌNH HỆ THỐNG ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/ppo_transformer_offline_best.zip")
SEQ_LENGTH = int(os.getenv("SEQ_LEN", "30"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
SAFETY_ROLLBACK_MIN_WEIGHT = float(os.getenv("SAFETY_ROLLBACK_MIN_WEIGHT", "0.05"))  # 5%

FEATURE_NAMES = ["CPU", "RAM", "Latency", "Error_Rate", "Traffic"]

app = FastAPI(title="Canary AI Agent Service — TransformerPPO")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ai-agent")

# --- 2. LOAD MÔ HÌNH TRANSFORMERPPO ---
MODEL_READY = False
model: Optional[PPO] = None
try:
    model = PPO.load(MODEL_PATH, device=DEVICE)
    MODEL_READY = True
    logger.info("model_load status=success path=%s device=%s arch=TransformerPPO", MODEL_PATH, DEVICE)
except Exception as e:
    logger.exception("model_load status=failed path=%s error=%s", MODEL_PATH, e)

# --- 3. XAI WEBSOCKET INFRASTRUCTURE ---
xai_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
# Track connected WebSocket clients for broadcast
xai_clients: List[WebSocket] = []


# --- 4. ĐỊNH NGHĨA DỮ LIỆU TỪ ARGO ROLLOUTS ---
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
        _prom_query_range(f"(sum(rate(response_total{{namespace=\"{ns}\",pod=~\".*{c_hash}.*\",direction=\"inbound\",classification=\"failure\"}}[1m])) or vector(0)) / clamp_min(sum(rate(response_total{{namespace=\"{ns}\",pod=~\".*{c_hash}.*\",direction=\"inbound\"}}[1m])), 0.001)", start_ts, end_ts, PROM_QUERY_STEP, empty_as_zero=True),
        _prom_query_range(f"(sum(rate(response_total{{namespace=\"{ns}\",pod=~\".*{s_hash}.*\",direction=\"inbound\",classification=\"failure\"}}[1m])) or vector(0)) / clamp_min(sum(rate(response_total{{namespace=\"{ns}\",pod=~\".*{s_hash}.*\",direction=\"inbound\"}}[1m])), 0.001)", start_ts, end_ts, PROM_QUERY_STEP, empty_as_zero=True),
        # l_canary: P95 Latency
        _prom_query_range(f"histogram_quantile(0.95, sum by (le) (rate(response_latency_ms_bucket{{namespace=\"{ns}\",pod=~\".*{c_hash}.*\",direction=\"inbound\"}}[1m]))) / 1000", start_ts, end_ts, PROM_QUERY_STEP),
        _prom_query_range(f"histogram_quantile(0.95, sum by (le) (rate(response_latency_ms_bucket{{namespace=\"{ns}\",pod=~\".*{s_hash}.*\",direction=\"inbound\"}}[1m]))) / 1000", start_ts, end_ts, PROM_QUERY_STEP),
        _prom_query_range(f"sum(rate(response_total{{namespace=\"{ns}\",pod=~\".*{c_hash}.*\",direction=\"inbound\"}}[1m]))", start_ts, end_ts, PROM_QUERY_STEP),
        _prom_query_range(f"sum(rate(response_total{{namespace=\"{ns}\",pod=~\".*{s_hash}.*\",direction=\"inbound\"}}[1m]))", start_ts, end_ts, PROM_QUERY_STEP),
        # CPU/Mem từ cAdvisor
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


# --- 5. XAI BROADCAST ---
async def _broadcast_xai(payload: dict):
    """Broadcast XAI data to all connected WebSocket clients."""
    dead_clients = []
    for ws in xai_clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead_clients.append(ws)
    for ws in dead_clients:
        xai_clients.remove(ws)


# --- 6. ENDPOINTS ---
@app.post("/api/v1/decision")
async def get_decision(payload: WebhookPayload):
    if not MODEL_READY or model is None:
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

    # Chốt chặn Data: Nếu đã có dữ liệu đầy đủ nhưng RPS canary = 0, trả về chờ
    if observed_weight > 0 and latest_canary_rps == 0.0:
        logger.info("Guard triggered: RPS canary is 0")
        return {"action": 0, "decision": "Stay (Waiting for traffic)"}

    try:
        ch_cpu, ch_mem, ch_lat, ch_err, ch_traffic = data

        # Build observation: (1, SEQ_LENGTH, 5) — sequence-first for Transformer
        obs_timesteps = []
        for i in range(SEQ_LENGTH):
            obs_timesteps.append([ch_cpu[i], ch_mem[i], ch_lat[i], ch_err[i], ch_traffic[i]])
        obs = np.array(obs_timesteps, dtype=np.float32)  # (30, 5)
        obs = np.expand_dims(obs, axis=0)  # (1, 30, 5)

        # Áp dụng VecNormalize để chuẩn hóa dữ liệu theo phân phối lúc train
        try:
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            import gymnasium as gym

            # Khởi tạo DummyEnv chỉ để load VecNormalize
            class DummyEnv(gym.Env):
                def __init__(self):
                    super().__init__()
                    self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(SEQ_LENGTH, 5), dtype=np.float32)
                    self.action_space = gym.spaces.Discrete(3)
                def reset(self, seed=None, options=None):
                    return np.zeros((SEQ_LENGTH, 5), dtype=np.float32), {}
                def step(self, action):
                    return np.zeros((SEQ_LENGTH, 5), dtype=np.float32), 0.0, False, False, {}

            dummy_venv = DummyVecEnv([lambda: DummyEnv()])
            vec_normalize = VecNormalize.load("models/vec_normalize.pkl", dummy_venv)
            vec_normalize.training = False
            obs = vec_normalize.normalize_obs(obs)
        except Exception as e:
            logger.error(f"Failed to normalize obs: {e}")

        # Inference bằng TransformerPPO (SB3)
        action_val, _states = model.predict(obs)
        action_val = int(action_val)

        # --- Trích xuất XAI Attention Maps ---
        xai_payload = None
        try:
            extractor = model.policy.features_extractor
            attn_maps = extractor.get_attention_maps()
            if attn_maps["feature_attention"] is not None:
                xai_payload = {
                    "timestamp": time.time(),
                    "service": payload.service,
                    "action": action_val,
                    "decision": "",  # will be set below
                    "feature_attention": attn_maps["feature_attention"][0].tolist(),  # (n_heads, T, 5)
                    "temporal_attention": attn_maps["temporal_attention"][0].tolist() if attn_maps["temporal_attention"] is not None else None,  # (n_heads, T, T)
                    "feature_names": FEATURE_NAMES,
                    "raw_metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in latest_raw.items()},
                }
        except Exception as e:
            logger.warning(f"XAI extraction failed: {e}")

        # Đánh giá Guard
        guard_decision, guard_reason = _evaluate_safety_guard(latest_raw, observed_weight)

        if guard_decision == "Rollback":
            action_val = 2
        elif guard_decision == "Running" and action_val == 0:
            action_val = 1  # Ép dừng lại (Stay) không cho Promote nếu hơi nguy hiểm

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

        # --- Broadcast XAI data qua WebSocket ---
        if xai_payload is not None:
            xai_payload["decision"] = decision
            xai_payload["action"] = api_action
            xai_payload["guard_decision"] = guard_decision
            xai_payload["guard_reason"] = guard_reason
            asyncio.create_task(_broadcast_xai(xai_payload))

        logger.info(f"Decision Debug: action_val={action_val}, guard_decision={guard_decision}, api_action={api_action}, canary_rps={latest_canary_rps}")
        logger.info(f"Raw metrics: {latest_raw}")
        logger.info(f"State metrics: {latest_state}")
        return {"action": api_action, "decision": decision}
    except Exception as exc:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")


# --- 7. WEBSOCKET ENDPOINT ---
@app.websocket("/ws/xai")
async def xai_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time XAI attention data streaming."""
    await websocket.accept()
    xai_clients.append(websocket)
    logger.info("XAI WebSocket client connected. Total clients: %d", len(xai_clients))
    try:
        while True:
            # Keep connection alive; wait for client messages (ping/pong)
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in xai_clients:
            xai_clients.remove(websocket)
        logger.info("XAI WebSocket client disconnected. Total clients: %d", len(xai_clients))


# --- 8. XAI DASHBOARD (SERVED BY FASTAPI) ---
@app.get("/dashboard", response_class=HTMLResponse)
async def xai_dashboard():
    """Serve the real-time XAI dashboard as a live web page."""
    return DASHBOARD_HTML


@app.get("/health")
def health():
    return {"status": "alive", "model_ready": MODEL_READY, "architecture": "TransformerPPO"}


# --- DASHBOARD HTML ---
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Canary AI — XAI Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }

  :root {
    --bg-primary: #0a0e1a;
    --bg-card: #111827;
    --bg-card-hover: #1a2235;
    --border: #1e293b;
    --text-primary: #e2e8f0;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-yellow: #f59e0b;
    --accent-purple: #8b5cf6;
    --glow-blue: rgba(59, 130, 246, 0.15);
    --glow-green: rgba(16, 185, 129, 0.15);
    --glow-red: rgba(239, 68, 68, 0.15);
  }

  body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Animated background gradient */
  body::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse at 20% 20%, rgba(59, 130, 246, 0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(139, 92, 246, 0.04) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
  }

  .container { max-width: 1400px; margin: 0 auto; padding: 20px; position: relative; z-index: 1; }

  /* Header */
  .header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 24px; margin-bottom: 24px;
    background: linear-gradient(135deg, var(--bg-card) 0%, #0f172a 100%);
    border: 1px solid var(--border); border-radius: 16px;
  }
  .header h1 { font-size: 20px; font-weight: 700; letter-spacing: -0.5px; }
  .header h1 span { background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .status-badge {
    display: flex; align-items: center; gap: 8px;
    padding: 6px 14px; border-radius: 20px; font-size: 12px; font-weight: 500;
  }
  .status-badge.connected { background: var(--glow-green); color: var(--accent-green); border: 1px solid rgba(16,185,129,0.3); }
  .status-badge.disconnected { background: var(--glow-red); color: var(--accent-red); border: 1px solid rgba(239,68,68,0.3); }
  .status-dot { width: 8px; height: 8px; border-radius: 50%; }
  .status-badge.connected .status-dot { background: var(--accent-green); animation: pulse 2s infinite; }
  .status-badge.disconnected .status-dot { background: var(--accent-red); }

  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }

  /* Grid layout */
  .grid { display: grid; gap: 20px; }
  .grid-2 { grid-template-columns: 1fr 1fr; }
  .grid-3 { grid-template-columns: 1fr 1fr 1fr; }

  /* Cards */
  .card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 14px; padding: 20px; transition: all 0.3s ease;
  }
  .card:hover { border-color: rgba(59,130,246,0.3); background: var(--bg-card-hover); }
  .card-title {
    font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;
    color: var(--text-muted); margin-bottom: 16px; display: flex; align-items: center; gap: 8px;
  }
  .card-title .icon { font-size: 16px; }

  /* Decision display */
  .decision-row {
    display: flex; align-items: center; gap: 16px;
    padding: 14px 18px; border-radius: 12px; margin-bottom: 12px;
    background: rgba(255,255,255,0.02); border: 1px solid var(--border);
    transition: all 0.3s ease; animation: slideIn 0.4s ease;
  }
  @keyframes slideIn { from { opacity: 0; transform: translateY(-10px); } to { opacity: 1; transform: translateY(0); } }
  .decision-action {
    padding: 4px 12px; border-radius: 8px; font-size: 12px; font-weight: 600;
    min-width: 80px; text-align: center;
  }
  .action-stay { background: rgba(245,158,11,0.15); color: var(--accent-yellow); border: 1px solid rgba(245,158,11,0.3); }
  .action-promote { background: var(--glow-green); color: var(--accent-green); border: 1px solid rgba(16,185,129,0.3); }
  .action-rollback { background: var(--glow-red); color: var(--accent-red); border: 1px solid rgba(239,68,68,0.3); }
  .decision-service { font-size: 13px; color: var(--text-secondary); flex: 1; }
  .decision-time { font-size: 11px; color: var(--text-muted); font-family: monospace; }

  /* Heatmap canvas */
  .heatmap-container { position: relative; }
  canvas { width: 100%; border-radius: 8px; image-rendering: pixelated; }

  /* Top features */
  .top-features { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }
  .feature-chip {
    display: flex; align-items: center; gap: 6px;
    padding: 6px 12px; border-radius: 8px; font-size: 12px; font-weight: 500;
    background: var(--glow-blue); color: var(--accent-blue); border: 1px solid rgba(59,130,246,0.2);
    transition: all 0.3s ease;
  }
  .feature-chip.highlight {
    background: rgba(245,158,11,0.15); color: var(--accent-yellow);
    border-color: rgba(245,158,11,0.4); transform: scale(1.05);
  }
  .feature-bar { width: 40px; height: 4px; border-radius: 2px; background: rgba(255,255,255,0.1); overflow: hidden; }
  .feature-bar-fill { height: 100%; border-radius: 2px; transition: width 0.5s ease; }

  /* Metrics */
  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 10px; }
  .metric-item {
    text-align: center; padding: 12px; border-radius: 10px;
    background: rgba(255,255,255,0.02); border: 1px solid var(--border);
  }
  .metric-value { font-size: 18px; font-weight: 700; font-family: 'Inter', monospace; }
  .metric-label { font-size: 11px; color: var(--text-muted); margin-top: 4px; }

  /* Empty state */
  .empty-state {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    min-height: 200px; color: var(--text-muted); gap: 12px;
  }
  .empty-state .icon { font-size: 40px; opacity: 0.3; }
  .empty-state p { font-size: 13px; }

  /* Scrollable log */
  .decision-log { max-height: 360px; overflow-y: auto; scrollbar-width: thin;
    scrollbar-color: var(--border) transparent; }
  .decision-log::-webkit-scrollbar { width: 4px; }
  .decision-log::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  @media (max-width: 900px) { .grid-2, .grid-3 { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="container">
  <!-- Header -->
  <div class="header">
    <h1>🧠 <span>Canary AI</span> — Explainable Dashboard</h1>
    <div id="wsStatus" class="status-badge disconnected">
      <div class="status-dot"></div>
      <span>Disconnected</span>
    </div>
  </div>

  <!-- Top Features + Latest Decision -->
  <div class="grid grid-3" style="margin-bottom: 20px;">
    <div class="card" style="grid-column: span 2;">
      <div class="card-title"><span class="icon">🎯</span> Feature Importance (Latest Timestep)</div>
      <div id="topFeatures" class="top-features">
        <div class="empty-state" style="min-height:60px"><p>Waiting for data...</p></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title"><span class="icon">📊</span> Live Metrics</div>
      <div id="liveMetrics" class="metrics-grid">
        <div class="empty-state" style="min-height:60px"><p>—</p></div>
      </div>
    </div>
  </div>

  <!-- Heatmaps -->
  <div class="grid grid-2" style="margin-bottom: 20px;">
    <div class="card">
      <div class="card-title"><span class="icon">🔬</span> Feature Attention Heatmap</div>
      <div class="heatmap-container">
        <canvas id="featureCanvas" width="500" height="300"></canvas>
      </div>
    </div>
    <div class="card">
      <div class="card-title"><span class="icon">⏱️</span> Temporal Attention Heatmap</div>
      <div class="heatmap-container">
        <canvas id="temporalCanvas" width="500" height="300"></canvas>
      </div>
    </div>
  </div>

  <!-- Decision Log -->
  <div class="card">
    <div class="card-title"><span class="icon">📋</span> Decision Log</div>
    <div id="decisionLog" class="decision-log">
      <div class="empty-state"><span class="icon">⏳</span><p>Waiting for agent decisions...</p></div>
    </div>
  </div>
</div>

<script>
const FEATURE_NAMES = ['CPU', 'RAM', 'Latency', 'Error_Rate', 'Traffic'];
const FEATURE_COLORS = ['#3b82f6', '#8b5cf6', '#f59e0b', '#ef4444', '#10b981'];

let ws = null;
let reconnectTimer = null;
let decisionCount = 0;

function connectWebSocket() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${proto}//${location.host}/ws/xai`;
  ws = new WebSocket(url);

  ws.onopen = () => {
    document.getElementById('wsStatus').className = 'status-badge connected';
    document.getElementById('wsStatus').innerHTML = '<div class="status-dot"></div><span>Connected</span>';
    if (reconnectTimer) { clearInterval(reconnectTimer); reconnectTimer = null; }
    // Send keepalive ping every 30s
    setInterval(() => { if (ws && ws.readyState === 1) ws.send('ping'); }, 30000);
  };

  ws.onclose = () => {
    document.getElementById('wsStatus').className = 'status-badge disconnected';
    document.getElementById('wsStatus').innerHTML = '<div class="status-dot"></div><span>Reconnecting...</span>';
    if (!reconnectTimer) reconnectTimer = setTimeout(connectWebSocket, 3000);
  };

  ws.onerror = () => ws.close();

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      updateDashboard(data);
    } catch(e) { console.error('Parse error:', e); }
  };
}

function updateDashboard(data) {
  updateFeatureImportance(data);
  updateFeatureHeatmap(data);
  updateTemporalHeatmap(data);
  updateDecisionLog(data);
  updateLiveMetrics(data);
}

function updateFeatureImportance(data) {
  if (!data.feature_attention) return;
  const fa = data.feature_attention; // (n_heads, T, 5)
  // Average over heads, take last timestep
  const nHeads = fa.length, T = fa[0].length, C = fa[0][0].length;
  const lastStep = new Array(C).fill(0);
  for (let h = 0; h < nHeads; h++)
    for (let c = 0; c < C; c++)
      lastStep[c] += fa[h][T-1][c] / nHeads;

  const maxVal = Math.max(...lastStep);
  const indices = lastStep.map((v,i) => ({v,i})).sort((a,b) => b.v - a.v);

  let html = '';
  for (const {v, i} of indices) {
    const pct = maxVal > 0 ? (v / maxVal * 100) : 0;
    const isTop = indices[0].i === i;
    html += `<div class="feature-chip ${isTop ? 'highlight' : ''}">
      <span>${FEATURE_NAMES[i]}</span>
      <span style="opacity:0.7">${v.toFixed(3)}</span>
      <div class="feature-bar"><div class="feature-bar-fill" style="width:${pct}%;background:${FEATURE_COLORS[i]}"></div></div>
    </div>`;
  }
  document.getElementById('topFeatures').innerHTML = html;
}

function drawHeatmap(canvasId, matrix, xLabels, yLabel, colormap) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext('2d');
  const rows = matrix.length, cols = matrix[0].length;

  const margin = { top: 10, right: 10, bottom: 30, left: 40 };
  canvas.width = 500; canvas.height = 300;
  const w = canvas.width - margin.left - margin.right;
  const h = canvas.height - margin.top - margin.bottom;
  const cellW = w / cols, cellH = h / rows;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Find min/max
  let minV = Infinity, maxV = -Infinity;
  for (const row of matrix) for (const v of row) { minV = Math.min(minV, v); maxV = Math.max(maxV, v); }
  const range = maxV - minV || 1;

  // Draw cells
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const t = (matrix[r][c] - minV) / range;
      ctx.fillStyle = colormap(t);
      ctx.fillRect(margin.left + c * cellW, margin.top + r * cellH, cellW + 0.5, cellH + 0.5);
    }
  }

  // X labels
  ctx.fillStyle = '#64748b'; ctx.font = '10px Inter, sans-serif'; ctx.textAlign = 'center';
  if (xLabels && xLabels.length <= 10) {
    for (let c = 0; c < cols; c++)
      ctx.fillText(xLabels[c] || c, margin.left + (c + 0.5) * cellW, canvas.height - 5);
  } else {
    for (let c = 0; c < cols; c += 5)
      ctx.fillText(c, margin.left + (c + 0.5) * cellW, canvas.height - 5);
  }

  // Y label
  ctx.textAlign = 'right';
  for (let r = 0; r < rows; r += 5)
    ctx.fillText(r, margin.left - 5, margin.top + (r + 0.5) * cellH + 3);

  ctx.fillStyle = '#94a3b8'; ctx.font = '11px Inter'; ctx.textAlign = 'center';
  ctx.fillText(yLabel || '', margin.left + w/2, canvas.height - 16);
}

function cmapWarm(t) {
  // Yellow-Orange-Red colormap
  const r = Math.round(255);
  const g = Math.round(255 * (1 - t * 0.8));
  const b = Math.round(80 * (1 - t));
  return `rgb(${r},${g},${b})`;
}

function cmapCool(t) {
  // Blue gradient
  const r = Math.round(10 + 50 * t);
  const g = Math.round(20 + 100 * t);
  const b = Math.round(80 + 175 * t);
  return `rgb(${r},${g},${b})`;
}

function updateFeatureHeatmap(data) {
  if (!data.feature_attention) return;
  const fa = data.feature_attention;
  const nHeads = fa.length, T = fa[0].length, C = fa[0][0].length;
  // Average over heads: (T, C)
  const avg = [];
  for (let t = 0; t < T; t++) {
    avg[t] = new Array(C).fill(0);
    for (let h = 0; h < nHeads; h++)
      for (let c = 0; c < C; c++)
        avg[t][c] += fa[h][t][c] / nHeads;
  }
  drawHeatmap('featureCanvas', avg, FEATURE_NAMES, 'Features', cmapWarm);
}

function updateTemporalHeatmap(data) {
  if (!data.temporal_attention) return;
  const ta = data.temporal_attention;
  const nHeads = ta.length, T = ta[0].length;
  const avg = [];
  for (let i = 0; i < T; i++) {
    avg[i] = new Array(T).fill(0);
    for (let h = 0; h < nHeads; h++)
      for (let j = 0; j < T; j++)
        avg[i][j] += ta[h][i][j] / nHeads;
  }
  drawHeatmap('temporalCanvas', avg, null, 'Key Timestep', cmapCool);
}

function updateDecisionLog(data) {
  const log = document.getElementById('decisionLog');
  if (decisionCount === 0) log.innerHTML = '';
  decisionCount++;

  const actionMap = {0: ['Stay', 'action-stay'], 1: ['Promote', 'action-promote'], 2: ['Rollback', 'action-rollback']};
  const [label, cls] = actionMap[data.action] || ['Unknown', 'action-stay'];
  const ts = new Date(data.timestamp * 1000).toLocaleTimeString();

  const row = document.createElement('div');
  row.className = 'decision-row';
  row.innerHTML = `
    <div class="decision-action ${cls}">${label}</div>
    <div class="decision-service">${data.service || '—'}${data.guard_decision ? ' <span style="color:var(--accent-yellow);font-size:11px">[Guard: '+data.guard_reason+']</span>' : ''}</div>
    <div class="decision-time">${ts}</div>`;
  log.insertBefore(row, log.firstChild);

  // Limit to 50 entries
  while (log.children.length > 50) log.removeChild(log.lastChild);
}

function updateLiveMetrics(data) {
  if (!data.raw_metrics) return;
  const m = data.raw_metrics;
  const container = document.getElementById('liveMetrics');
  container.innerHTML = `
    <div class="metric-item"><div class="metric-value" style="color:var(--accent-red)">${((m.e_canary||0)*100).toFixed(2)}%</div><div class="metric-label">Error Rate</div></div>
    <div class="metric-item"><div class="metric-value" style="color:var(--accent-yellow)">${(m.l_canary||0).toFixed(0)}ms</div><div class="metric-label">Latency P95</div></div>
    <div class="metric-item"><div class="metric-value" style="color:var(--accent-cyan)">${(m.weight_pct||0).toFixed(0)}%</div><div class="metric-label">Traffic</div></div>
    <div class="metric-item"><div class="metric-value" style="color:var(--accent-blue)">${(m.rps||0).toFixed(1)}</div><div class="metric-label">RPS</div></div>`;
}

// Auto-connect on load
connectWebSocket();
</script>
</body>
</html>
"""