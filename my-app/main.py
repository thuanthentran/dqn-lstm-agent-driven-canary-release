import os
import asyncio
import random
from fastapi import FastAPI, Response
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI()
Instrumentator().instrument(app).expose(app, include_in_schema=False, endpoint="/metrics")

SCENARIO = os.getenv("APP_SCENARIO", "healthy")
VERSION = os.getenv("APP_VERSION", "v1.0.0")

memory_leak_list = []
active_requests = 0 # Đếm số lượng request đang xử lý đồng thời

@app.get("/")
async def root():
    global active_requests
    active_requests += 1
    
    try:
        # 1. Kịch bản: Latency Leak & Memory Leak
        if SCENARIO == "latency_leak":
            # Leak nhẹ nhàng 50KB mỗi request để app "sống" đủ lâu cho Agent theo dõi
            memory_leak_list.append(" " * 50 * 1024) 
            
            # Tính toán độ trễ dựa trên lượng RAM đã leak (tối đa delay 3 giây)
            delay = min(0.5 + (len(memory_leak_list) * 0.001), 3.0)
            
            # SỬ DỤNG ASYNC SLEEP: Chỉ cho request này ngủ, không chặn web server
            await asyncio.sleep(delay)

        # 2. Kịch bản: Critical Crash (Tạch 50%)
        if SCENARIO == "critical_crash":
            if random.random() < 0.5:
                return Response(content="Internal Server Error", status_code=500)

        # 3. Kịch bản: Error Bomb (Đúng nghĩa: Chỉ nổ khi bị ép tải)
        if SCENARIO == "error_bomb":
            # Nếu có nhiều hơn 5 request gọi cùng 1 tích tắc -> Sập hầm
            if active_requests > 5: 
                # Cố tình delay một nhịp để mô phỏng nghẽn tài nguyên rồi mới văng lỗi
                await asyncio.sleep(0.1) 
                return Response(content="Service Unavailable Overload", status_code=503)

        return {
            "version": VERSION,
            "scenario": SCENARIO,
            "status": "online",
            "latency": "fast" if SCENARIO != "latency_leak" else f"delayed_{delay}s"
        }
    finally:
        active_requests -= 1

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}