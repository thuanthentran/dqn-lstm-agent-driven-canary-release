# Python slim image cho deployment
FROM python:3.11-slim

WORKDIR /app

# 1. Copy requirements và cài đặt dependencies
COPY services/agent/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 2. Copy core module (TransformerFeatureExtractor, CanaryEnv, feature_pipeline)
COPY core/ ./core/

# 3. Copy trained TransformerPPO model artifacts
COPY models/ ./models/

# 4. Copy FastAPI agent service (includes XAI WebSocket + Dashboard)
COPY services/agent/main.py .

ENV PYTHONPATH=/app

EXPOSE 8000

# Chạy uvicorn — Dashboard tại /dashboard, XAI WebSocket tại /ws/xai
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

