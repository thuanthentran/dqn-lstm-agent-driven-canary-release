---
name: kubernetes_gitops
description: Guidelines and best practices for managing Kubernetes manifests, GitOps, and Canary Releases.
---

# Kubernetes & GitOps Guidelines

1. **Isolation**: Keep deployment and GitOps changes isolated from model logic. Do not mix ML code with Kubernetes manifests.
2. **Source of Truth**: Treat `gitops/app/manifests/` as the absolute rollout source of truth. Never modify live cluster state directly; always update the manifests.
3. **Canary Release**: The project uses Istio for traffic routing in canary releases. Ensure Istio VirtualServices and DestinationRules are properly configured when modifying routing.
4. **Tooling**: Use `kubectl` for read-only cluster inspection, and rely on GitOps (e.g., Argo CD) to apply state changes.
