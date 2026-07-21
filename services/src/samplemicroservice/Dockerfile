# --- Stage 1: Build & Compile Proto ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Cài đặt công cụ biên dịch gRPC
RUN pip install --no-cache-dir grpcio-tools

COPY hollow.proto .
# Biên dịch file .proto ra hollow_pb2.py và hollow_pb2_grpc.py
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. hollow.proto

# --- Stage 2: Runtime Image ---
FROM python:3.11-slim

WORKDIR /app

# Cài đặt dependencies nhẹ
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy file đã biên dịch từ Stage 1 và source code
COPY --from=builder /app/hollow_pb2*.py ./
COPY main.py .

# Expose cổng HTTP (8000) và gRPC (50051)
EXPOSE 8000 50051

CMD ["python", "main.py"]