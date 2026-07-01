#!/bin/bash

# Đường dẫn đến môi trường ảo
VENV_PATH="$HOME/venv"

echo "Đang kích hoạt môi trường ảo..."
source $VENV_PATH/bin/activate

echo "Đang khởi động MLFlow server (Chế độ HTTP Proxy)..."
# Chạy ở background, ép cấu hình mlflow-artifacts:/ để hứng file từ Windows
nohup mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root mlflow-artifacts:/ \
    --artifacts-destination ./mlruns \
    --host 0.0.0.0 \
    --port 5000 > $HOME/mlflow.log 2>&1 &

echo "✅ MLFlow đã chạy ngầm thành công với cấu hình Proxy!"
echo "📂 Xem log tại: $HOME/mlflow.log"