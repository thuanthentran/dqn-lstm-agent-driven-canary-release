import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import torch.nn as nn

# --- 1. RE-DEFINE MODEL ARCHITECTURE (Phải khớp với file train) ---
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

# --- 2. CONFIGURATION ---
MODEL_PATH = "model_canary_drqn.pth"
SEQ_LENGTH = 10
DEVICE = torch.device("cpu") # Chạy inference dùng CPU cho nhẹ

app = FastAPI(title="Canary AI Agent Service")

# Load model
model = DRQN(n_obs=8, n_actions=5).to(DEVICE)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print(f"Successfully loaded model from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")

# --- 3. DATA SCHEMA ---
class MetricPoint(BaseModel):
    # Các metrics phải khớp với 8 chỉ số trong Env
    weight: float
    e_canary: float
    e_stable: float
    l_canary: float
    l_stable: float
    cpu: float
    mem: float
    rps: float

class InferenceRequest(BaseModel):
    history: List[MetricPoint] # Nhận vào một list 10 điểm dữ liệu gần nhất

# --- 4. PREDICT ENDPOINT ---
@app.post("/predict")
async def predict(request: InferenceRequest):
    if len(request.history) < SEQ_LENGTH:
        # Nếu chưa đủ 10 điểm, ta "pad" thêm bằng các điểm cũ nhất
        needed = SEQ_LENGTH - len(request.history)
        history_list = [request.history] * needed + request.history
    else:
        history_list = request.history[-SEQ_LENGTH:]

    # Chuyển dữ liệu sang Tensor (8 features) và CHUẨN HÓA
    data = []
    for p in history_list:
        # Chuẩn hóa RPS bằng cách chia cho 1000 (khớp với kịch bản train)
        normalized_rps = p.rps / 1000.0
        
        data.append([
            p.weight, 
            p.e_canary, 
            p.e_stable, 
            p.l_canary, 
            p.l_stable, 
            p.cpu, 
            p.mem, 
            normalized_rps
        ])
    
    input_tensor = torch.FloatTensor([data]).to(DEVICE) # Shape: (1, 10, 8)

    with torch.no_grad():
        q_values, _ = model(input_tensor)
        action = torch.argmax(q_values).item()

    # Ánh xạ action chi tiết sang cấu hình Argo Rollouts
    action_mapping = {
        0: "Advance_10", # Fast Forward (+10% weight)
        1: "Advance_5",  # Step Forward (+5% weight)
        2: "Pause",      # Stay (Giữ nguyên để quan sát)
        3: "StepBack",   # Step Back (-5% weight)
        4: "Rollback"    # EMERGENCY ROLLBACK (Hủy rollout)
    }
    
    decision = action_mapping.get(action, "Unknown")
    
    return {
        "action_id": action,
        "decision": decision,
        "confidence": float(torch.softmax(q_values, dim=1).max())
    }

@app.get("/health")
def health():
    return {"status": "alive"}