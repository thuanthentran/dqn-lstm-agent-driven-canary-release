#!/usr/bin/env bash
# =============================================================================
# setup-cluster.sh
# DQN-LSTM Canary Release — K8s Cluster Setup
# Stack: kubeadm + Flannel + Istio + Prometheus (minimal) + Argo Rollouts + ArgoCD
#
# Chạy: bash setup-cluster.sh
# Dừng: sau khi cài ArgoCD xong (bạn tự sync app qua UI)
# =============================================================================

set -euo pipefail

# ─── Màu cho log ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')] $*${RESET}"; }
ok()   { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✔  $*${RESET}"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠  $*${RESET}"; }
die()  { echo -e "${RED}[$(date '+%H:%M:%S')] ✘  $*${RESET}"; exit 1; }

# ─── Phiên bản có thể chỉnh ──────────────────────────────────────────────────
K8S_VERSION="1.30"
ISTIO_VERSION="1.23.0"
ARGOCD_NODEPORT=30443
PROMETHEUS_NODEPORT=30090
ARGOCD_NAMESPACE="argocd"

# ─── Kiểm tra đang chạy với quyền root ───────────────────────────────────────
[[ $EUID -ne 0 ]] && die "Script cần chạy với sudo: sudo bash setup-cluster.sh"

# =============================================================================
# BƯỚC 0 — Chuẩn bị hệ thống
# =============================================================================
log "${BOLD}[0/7] Chuẩn bị hệ thống...${RESET}"

# Tắt swap (bắt buộc với K8s)
swapoff -a
sed -i '/ swap / s/^\(.*\)$/#\1/' /etc/fstab
ok "Swap đã tắt"

# Tải kernel modules cần thiết
cat > /etc/modules-load.d/k8s.conf <<EOF
overlay
br_netfilter
EOF
modprobe overlay
modprobe br_netfilter

# Cài đặt sysctl cho pod networking
cat > /etc/sysctl.d/k8s.conf <<EOF
net.bridge.bridge-nf-call-iptables  = 1
net.bridge.bridge-nf-call-ip6tables = 1
net.ipv4.ip_forward                 = 1
EOF
sysctl --system > /dev/null
ok "Kernel modules và sysctl đã cài"

# Cài containerd
apt-get update -qq
apt-get install -y -qq containerd curl apt-transport-https ca-certificates gpg

mkdir -p /etc/containerd
containerd config default > /etc/containerd/config.toml
# Bật SystemdCgroup (bắt buộc với K8s 1.22+)
sed -i 's/SystemdCgroup = false/SystemdCgroup = true/' /etc/containerd/config.toml
systemctl restart containerd
systemctl enable containerd
ok "containerd đã cài và cấu hình"

# =============================================================================
# BƯỚC 1 — Cài kubeadm, kubelet, kubectl
# =============================================================================
log "${BOLD}[1/7] Cài kubeadm, kubelet, kubectl (K8s ${K8S_VERSION})...${RESET}"

curl -fsSL "https://pkgs.k8s.io/core:/stable:/v${K8S_VERSION}/deb/Release.key" \
  | gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg 2>/dev/null

echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] \
https://pkgs.k8s.io/core:/stable:/v${K8S_VERSION}/deb/ /" \
  > /etc/apt/sources.list.d/kubernetes.list

apt-get update -qq
apt-get install -y -qq kubelet kubeadm kubectl
apt-mark hold kubelet kubeadm kubectl
systemctl enable --now kubelet
ok "kubeadm $(kubeadm version --output short) đã cài"

# =============================================================================
# BƯỚC 2 — Khởi tạo cụm K8s
# =============================================================================
log "${BOLD}[2/7] Khởi tạo cụm K8s...${RESET}"

# Nếu cụm đã được khởi tạo rồi thì bỏ qua
if kubectl cluster-info &>/dev/null; then
  warn "Cụm đã tồn tại, bỏ qua bước kubeadm init"
else
  # --pod-network-cidr=10.244.0.0/16 là CIDR mặc định Flannel yêu cầu
  kubeadm init --pod-network-cidr=10.244.0.0/16 2>&1 | tee /tmp/kubeadm-init.log
  ok "kubeadm init thành công"
fi

# Cấu hình kubectl cho user hiện tại (cả root và user gốc nếu có SUDO_USER)
REAL_HOME="/root"
if [[ -n "${SUDO_USER:-}" ]]; then
  REAL_HOME=$(getent passwd "$SUDO_USER" | cut -d: -f6)
fi
mkdir -p "$REAL_HOME/.kube"
cp /etc/kubernetes/admin.conf "$REAL_HOME/.kube/config"
chown -R "${SUDO_USER:-root}:${SUDO_USER:-root}" "$REAL_HOME/.kube" 2>/dev/null || true

# Export để dùng trong script này
export KUBECONFIG=/etc/kubernetes/admin.conf
ok "kubectl config đã thiết lập tại $REAL_HOME/.kube/config"

# Single-node: bỏ taint để schedule pods lên control-plane
kubectl taint nodes --all node-role.kubernetes.io/control-plane- 2>/dev/null || true
ok "Đã bỏ control-plane taint (single-node mode)"

# =============================================================================
# BƯỚC 3 — Cài CNI: Flannel
# =============================================================================
log "${BOLD}[3/7] Cài Flannel CNI...${RESET}"

kubectl apply -f \
  https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml

log "Chờ Flannel pods sẵn sàng..."
kubectl rollout status daemonset/kube-flannel-ds -n kube-flannel --timeout=120s
ok "Flannel đã sẵn sàng"

# Chờ control-plane node Ready
log "Chờ node Ready..."
kubectl wait node --all --for=condition=Ready --timeout=120s
ok "Node đã Ready: $(kubectl get nodes --no-headers | awk '{print $1, $2}')"

# =============================================================================
# BƯỚC 4 — Cài Istio
# =============================================================================
log "${BOLD}[4/7] Cài Istio ${ISTIO_VERSION}...${RESET}"

ISTIO_DIR="/opt/istio-${ISTIO_VERSION}"

if [[ ! -d "$ISTIO_DIR" ]]; then
  curl -L https://istio.io/downloadIstio | ISTIO_VERSION=${ISTIO_VERSION} TARGET_ARCH=x86_64 sh -
  mv "istio-${ISTIO_VERSION}" "$ISTIO_DIR"
fi

ISTIOCTL="$ISTIO_DIR/bin/istioctl"
ln -sf "$ISTIOCTL" /usr/local/bin/istioctl

# Pre-check (bỏ qua lỗi không nghiêm trọng)
istioctl x precheck 2>&1 || warn "Pre-check có cảnh báo, tiếp tục..."

# Cài profile default (istiod + ingress gateway)
istioctl install --set profile=default -y
ok "Istio đã cài xong"

# Chờ istiod sẵn sàng
kubectl rollout status deployment/istiod -n istio-system --timeout=180s
ok "istiod sẵn sàng"

# =============================================================================
# BƯỚC 5 — Cài Prometheus (tối giản)
# =============================================================================
log "${BOLD}[5/7] Cài Prometheus (minimal)...${RESET}"

# Cài Helm nếu chưa có
if ! command -v helm &>/dev/null; then
  curl -fsSL https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
  ok "Helm đã cài"
fi

helm repo add prometheus-community \
  https://prometheus-community.github.io/helm-charts 2>/dev/null || true
helm repo update -q

kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f -

# Cài kube-prometheus-stack tối giản
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --set prometheus.prometheusSpec.retention=6h \
  --set prometheus.prometheusSpec.retentionSize="3GB" \
  --set prometheus.prometheusSpec.resources.requests.memory=256Mi \
  --set prometheus.prometheusSpec.resources.limits.memory=512Mi \
  --set alertmanager.enabled=false \
  --set grafana.enabled=false \
  --set prometheusOperator.resources.requests.memory=64Mi \
  --set prometheusOperator.resources.limits.memory=128Mi \
  --wait --timeout=300s

# Expose qua NodePort
kubectl patch svc prometheus-kube-prometheus-prometheus -n monitoring \
  -p "{\"spec\":{\"type\":\"NodePort\",\"ports\":[{\"port\":9090,\"targetPort\":9090,\"nodePort\":${PROMETHEUS_NODEPORT}}]}}" \
  2>/dev/null || true

ok "Prometheus sẵn sàng tại NodePort :${PROMETHEUS_NODEPORT}"

# =============================================================================
# BƯỚC 6 — Cài Argo Rollouts
# =============================================================================
log "${BOLD}[6/7] Cài Argo Rollouts...${RESET}"

kubectl create namespace argo-rollouts --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n argo-rollouts \
  -f https://github.com/argoproj/argo-rollouts/releases/latest/download/install.yaml

# Cài kubectl plugin
ARCH=$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')
ROLLOUTS_BIN="/usr/local/bin/kubectl-argo-rollouts"
curl -sL \
  "https://github.com/argoproj/argo-rollouts/releases/latest/download/kubectl-argo-rollouts-linux-${ARCH}" \
  -o "$ROLLOUTS_BIN"
chmod +x "$ROLLOUTS_BIN"

kubectl rollout status deployment/argo-rollouts -n argo-rollouts --timeout=120s
ok "Argo Rollouts sẵn sàng"

# =============================================================================
# BƯỚC 7 — Cài ArgoCD
# =============================================================================
log "${BOLD}[7/7] Cài ArgoCD...${RESET}"

kubectl create namespace "$ARGOCD_NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n "$ARGOCD_NAMESPACE" \
  -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

log "Chờ ArgoCD pods sẵn sàng..."
kubectl wait pod --all -n "$ARGOCD_NAMESPACE" \
  --for=condition=Ready --timeout=300s

# Expose qua NodePort
kubectl patch svc argocd-server -n "$ARGOCD_NAMESPACE" -p "{
  \"spec\": {
    \"type\": \"NodePort\",
    \"ports\": [
      {\"name\": \"http\",  \"port\": 80,  \"targetPort\": 8080, \"nodePort\": 30080},
      {\"name\": \"https\", \"port\": 443, \"targetPort\": 8080, \"nodePort\": ${ARGOCD_NODEPORT}}
    ]
  }
}"

# Lấy mật khẩu admin
ARGOCD_PASSWORD=$(kubectl -n "$ARGOCD_NAMESPACE" get secret argocd-initial-admin-secret \
  -o jsonpath="{.data.password}" | base64 -d)

# ─── Tóm tắt kết quả ─────────────────────────────────────────────────────────
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')

echo ""
echo -e "${GREEN}${BOLD}============================================================${RESET}"
echo -e "${GREEN}${BOLD}  ✅  SETUP HOÀN TẤT${RESET}"
echo -e "${GREEN}${BOLD}============================================================${RESET}"
echo ""
echo -e "  ${BOLD}Node IP:${RESET}        ${NODE_IP}"
echo ""
echo -e "  ${BOLD}ArgoCD UI:${RESET}      https://${NODE_IP}:${ARGOCD_NODEPORT}"
echo -e "  ${BOLD}Username:${RESET}       admin"
echo -e "  ${BOLD}Password:${RESET}       ${ARGOCD_PASSWORD}"
echo ""
echo -e "  ${BOLD}Prometheus:${RESET}     http://${NODE_IP}:${PROMETHEUS_NODEPORT}"
echo ""
echo -e "  ${CYAN}Bước tiếp theo:${RESET}"
echo -e "  1. Mở ArgoCD UI tại https://${NODE_IP}:${ARGOCD_NODEPORT}"
echo -e "  2. Đăng nhập admin / ${ARGOCD_PASSWORD}"
echo -e "  3. Apply root-app:"
echo -e "     ${YELLOW}kubectl apply -f root-app.yaml${RESET}"
echo -e "  4. Sync apps theo thứ tự: cluster-base → bootstrap-prod → bootstrap-twin"
echo -e "     (hoặc bấm Sync trực tiếp trên UI)"
echo ""
echo -e "${GREEN}${BOLD}============================================================${RESET}"
