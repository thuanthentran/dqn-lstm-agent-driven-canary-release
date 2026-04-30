import os
import time
import math
import asyncio
import random
from fastapi import FastAPI, Response
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")

SCENARIO = os.getenv("APP_SCENARIO", "healthy")
VERSION = os.getenv("APP_VERSION", "v1.0.0")

START_TIME = time.time()
memory_leak_list = []
active_requests = 0

def cpu_intensive_task(duration_sec: float):
    """Giả lập việc tính toán ngốn CPU bằng một vòng lặp chặt."""
    end_time = time.time() + duration_sec
    # Vòng lặp rỗng liên tục sẽ ăn 100% của 1 core CPU trong thời gian duration_sec
    while time.time() < end_time:
        pass 

@app.get("/")
async def root():
    global active_requests
    active_requests += 1
    uptime = time.time() - START_TIME
    
    # NHIỄU (JITTER) TIÊU CHUẨN CỦA HỆ THỐNG: 10ms - 50ms
    base_jitter = random.uniform(0.01, 0.05)
    
    try:
        # --- CÁC KỊCH BẢN TÍCH CỰC (AGENT NÊN CHỌN ACTION 0 HOẶC 1) ---
        
        # 1. Kịch bản Healthy Tiêu chuẩn
        if SCENARIO == "healthy":
            await asyncio.sleep(base_jitter)
            
        # 2. Kịch bản Refactored (Ngang bằng bản cũ, cấu trúc code mới)
        elif SCENARIO == "refactored_healthy":
            # Thi thoảng tốn thêm 2ms CPU (không đáng kể, nằm trong vùng an toàn)
            # để xem Agent có bị nhạy cảm quá mức với biến động nhỏ của CPU không.
            if random.random() < 0.1:
                await asyncio.to_thread(cpu_intensive_task, 0.002) 
            await asyncio.sleep(base_jitter)

        # 3. Kịch bản Optimized (Nhanh và mượt hơn hẳn bản Stable)
        elif SCENARIO == "optimized_fast":
            # Latency cực thấp (2ms - 10ms) -> Agent phải nhận ra và tự tin Promote nhanh
            fast_jitter = random.uniform(0.002, 0.01) 
            await asyncio.sleep(fast_jitter)


        # --- CÁC KỊCH BẢN TIÊU CỰC (AGENT NÊN CHỌN ACTION 4 HOẶC 3) ---

        # 4. Kịch bản: Latency Leak
        elif SCENARIO == "latency_leak":
            memory_leak_list.append(" " * 50 * 1024) 
            delay = min(0.5 + (len(memory_leak_list) * 0.001), 3.0)
            noise = random.uniform(-0.1, 0.1)
            await asyncio.sleep(max(0, delay + base_jitter + noise))

        # 5. Kịch bản: Critical Crash
        elif SCENARIO == "critical_crash":
            await asyncio.sleep(base_jitter)
            if random.random() < 0.5:
                error_codes = [500, 502, 503, 504]
                return Response(content="System Failure", status_code=random.choice(error_codes))

        # 6. Kịch bản: CPU Spike
        elif SCENARIO == "cpu_spike":
            if random.random() < 0.3:
                await asyncio.to_thread(cpu_intensive_task, 0.2) 
            await asyncio.sleep(base_jitter)

        # 7. Kịch bản: Cascading Failure
        elif SCENARIO == "cascading_failure":
            degradation_factor = min(uptime / 120.0, 1.0) 
            
            if random.random() < degradation_factor:
                await asyncio.to_thread(cpu_intensive_task, 0.05)
            
            if random.random() < degradation_factor:
                memory_leak_list.append(" " * 100 * 1024)
            
            latency_spike = degradation_factor * random.uniform(0.5, 2.5)
            await asyncio.sleep(base_jitter + latency_spike)
            
            if random.random() < (degradation_factor * 0.4): 
                return Response(content="Cascading Error", status_code=random.choice([500, 503, 504]))

        return {
            "version": VERSION,
            "scenario": SCENARIO,
            "status": "online",
            "uptime_sec": round(uptime, 2)
        }
    finally:
        active_requests -= 1

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}