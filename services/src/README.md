# Microservices Demo

This folder contains a Kubernetes-friendly layout for the microservices demo, now wired for Istio.

## Layout

- `k8s/` contains the namespace, workloads, and Istio resources.
- Each service directory contains the application source and Dockerfile for that service.

## Build Images

Build and tag each service image locally using the same names used by the manifests:

- `microservices-demo/adservice:latest`
- `microservices-demo/cartservice:latest`
- `microservices-demo/checkoutservice:latest`
- `microservices-demo/currencyservice:latest`
- `microservices-demo/emailservice:latest`
- `microservices-demo/frontend:latest`
- `microservices-demo/loadgenerator:latest`
- `microservices-demo/paymentservice:latest`
- `microservices-demo/productcatalogservice:latest`
- `microservices-demo/recommendationservice:latest`
- `microservices-demo/shippingservice:latest`
- `microservices-demo/shoppingassistantservice:latest`

## Deploy

1. Install Istio and confirm the ingress gateway is running.
2. Apply the manifests with `kubectl apply -k microservices-demo/k8s`.
3. Expose the app through the Istio gateway and reach the frontend service.

## Notes

- The namespace is labeled for automatic Istio sidecar injection.
- `shoppingassistantservice` needs AlloyDB and Secret Manager-related environment values before it can run against real infrastructure.
- The load generator runs as a headless workload and targets `frontend:8080` inside the mesh.