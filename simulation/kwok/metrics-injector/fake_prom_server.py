from fastapi import FastAPI
import random
import time
import uvicorn

app = FastAPI()

# Biến lưu trữ trọng số traffic hiện tại (Rollout sẽ cập nhật biến này)
current_canary_weight = 0.0

@app.get("/metrics/canary")
def get_canary_metrics():
    """
    Trả về số liệu giả lập dựa trên % traffic đang được đẩy vào bản Canary.
    Càng nhiều traffic, tỷ lệ xuất hiện lỗi hoặc trễ càng thay đổi theo kịch bản.
    """
    # Mô phỏng độ trễ mạng (Network Jitter)
    time.sleep(random.uniform(0.01, 0.05))
    
    # 1. Kịch bản: Tỷ lệ lỗi tăng nhẹ khi traffic tăng
    base_error_rate = 0.01
    error_rate = base_error_rate + (current_canary_weight / 100.0) * random.uniform(0.02, 0.1)
    
    # 2. Kịch bản: Độ trễ (Latency - ms)
    base_latency = 120.0
    latency = base_latency + (current_canary_weight / 100.0) * random.uniform(10.0, 50.0)
    
    # 3. Kịch bản: CPU Usage (%)
    cpu_usage = 10.0 + (current_canary_weight / 100.0) * random.uniform(20.0, 40.0)

    return {
        "status": "success",
        "data": {
            "error_rate": round(error_rate, 4),
            "latency_ms": round(latency, 2),
            "cpu_usage_percent": round(cpu_usage, 2)
        }
    }

@app.post("/set_weight/{weight}")
def set_traffic_weight(weight: float):
    """API để môi trường Env (Trụ cột 2) báo cáo mức traffic hiện tại cho Injector"""
    global current_canary_weight
    current_canary_weight = weight
    return {"message": f"Updated canary weight to {weight}%"}

if __name__ == "__main__":
    print("🚀 Khởi động Metrics Injector (Fake Prometheus) tại cổng 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)