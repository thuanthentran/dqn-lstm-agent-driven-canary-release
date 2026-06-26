#!/bin/bash

# Dừng script ngay lập tức nếu có lỗi xảy ra
set -e

CLUSTER_NAME="rl-canary-cluster"

echo "🚀 [1/3] Bắt đầu khởi tạo cụm KWOK K8s ảo ($CLUSTER_NAME)..."
# Kiểm tra xem cụm đã tồn tại chưa để tránh lỗi
if kwokctl get clusters | grep -q "$CLUSTER_NAME"; then
    echo "   -> Cụm đã tồn tại. Đang khởi động lại..."
    kwokctl start cluster --name $CLUSTER_NAME
else
    kwokctl create cluster --name $CLUSTER_NAME
fi

echo "🔌 [2/3] Trỏ kubectl context vào cụm..."
kubectl config use-context kwok-$CLUSTER_NAME

echo "🗺️ [3/3] Cấp phát Node ảo (Fake Nodes)..."
# Lấy đường dẫn tuyệt đối của thư mục chứa script này
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
NODE_FILE="$SCRIPT_DIR/../nodes/fake-nodes.yaml"

if [ -f "$NODE_FILE" ]; then
    kubectl apply -f "$NODE_FILE"
else
    echo "❌ Lỗi: Không tìm thấy file $NODE_FILE"
    exit 1
fi

echo "========================================="
echo "✅ HOÀN TẤT! CỤM K8S ẢO ĐÃ SẴN SÀNG."
echo "Danh sách Node hiện tại:"
kubectl get nodes