# DQN_LSTM_Agent_for_Canary_Release

This repository is organized as a small monorepo for canary release experiments on Kubernetes, with separate areas for ML logic, deployment manifests, microservices examples, and training assets.

## Architectural Choices & Sandbox Setup (eBPF Branch)

This branch (`ebpf`) contains a specialized Digital Twin sandbox environment designed to train a Reinforcement Learning (RL) agent. The architecture uses **Cilium eBPF** for observability and networking, **Gateway API (Envoy)** for L7 traffic interception, and **Argo Rollouts** for Canary deployments.

Due to the nature of building a sandbox for RL agents that requires precise metrics and traffic control, several specific configurations and "workarounds" were implemented to glue these tools together seamlessly.

### 1. Cilium & Hubble (eBPF Observability)
- **Role:** Replaces Istio for network observability without sidecars. Provides high-performance metrics (L3/L4/L7) via eBPF.
- **Specific Configurations:** 
  - Hubble metrics are enabled and exposed to Prometheus using specific annotations (`hubble_http_responses_total`, `hubble_http_request_duration_seconds_bucket`).
  - L7 visibility policies (`CiliumNetworkPolicy`) were removed/relaxed to allow the Gateway API to take full control of L7 parsing without conflicts.

### 2. Cilium Gateway API (Envoy)
- **Role:** Acts as the ingress and L7 router. By default, K8s Services use DNS round-robin or iptables which bypasses L7 observability for East-West gRPC traffic. The Gateway API intercepts traffic to emit L7 metrics and perform Canary weight splitting.
- **Workaround (East-West Traffic Routing):** 
  - Standard Gateway API is designed for North-South (Ingress) traffic. To force internal microservices to be intercepted (e.g., `frontend` calling `checkoutservice`), we bypassed the native K8s DNS resolution. 
  - **Patch:** We modified the environment variables of all microservices (in `gitops/releases/*-values.yaml`) to point directly to the Gateway's internal address: `cilium-gateway-<svc>-gateway:80` (e.g., `CHECKOUT_SERVICE_ADDR="cilium-gateway-checkoutservice-gateway:80"`). This forces all internal traffic through the Envoy proxy, ensuring 100% L7 metrics capture and accurate traffic splitting.

### 3. Argo Rollouts (Canary Controller)
- **Role:** Orchestrates the Canary release process (creating ReplicaSets and splitting traffic).
- **Workaround (Gateway API Plugin):** 
  - Argo Rollouts does *not* support Gateway API natively in its core reconciler. It requires an external plugin (`argoproj-labs/gatewayAPI`).
  - **Patch 1 (ConfigMap):** Created a specific ConfigMap (`argo-rollouts-config`) in the `argo-rollouts` namespace to instruct the controller to download the plugin binary from GitHub (`gatewayapi-plugin-linux-amd64`).
  - **Patch 2 (RBAC):** Granted the `argo-rollouts` ServiceAccount additional cluster permissions to `get, list, watch, update, patch` the `httproutes` resource in the `gateway.networking.k8s.io` API group.
  - **Patch 3 (Syntax):** Changed the rollout definition (`gitops/charts/universal-canary/templates/rollout.yaml`) from the built-in syntax to the plugin syntax:
    ```yaml
    trafficRouting:
      plugins:
        argoproj-labs/gatewayAPI:
          httpRoute: <route-name>
          namespace: default
    ```

### 4. RL Agent (Python Training Environment)
- **Role:** The `online_training.py` script acts as the "Brain", listening to K8s API events, querying Prometheus, and injecting faults.
- **Workaround (Locust Load Generator):**
  - To generate realistic traffic, the environment triggers `locust` via a Python `subprocess`. 
  - **Patch:** To avoid `PATH` resolution issues in virtual environments (like Conda), the script is explicitly patched to invoke Locust using `sys.executable -m locust`.

## Layout

- `core/` contains the environment, feature pipeline, and model-facing logic (`online_env.py`).
- `deploy/` contains runtime entrypoints and container build material.
- `gitops/` contains Argo CD applications, helm charts (`universal-canary`), and values files.
- `loadgenerator/` contains Locust scripts for simulating traffic.
- `models/` stores trained model artifacts.
- `training/` contains the primary training script (`online_training.py`).

## Working Order

1. Keep deployment and GitOps changes isolated from model logic.
2. Treat `gitops/releases/` as the rollout source of truth.
3. Update this README when architectural changes or new workarounds are introduced.
