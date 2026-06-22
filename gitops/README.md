# GitOps Layout

This folder keeps the Kubernetes manifests used by Argo CD.

## Structure

- `app/` contains the application rollout, service, and analysis template.
- `monitoring/` contains the Argo CD application for `kube-prometheus-stack`.

## Notes

- Keep manifest names and namespaces stable unless the deployment target changes.
- Prefer one manifest per concern and keep Argo CD `Application` resources declarative.
- Store operational comments in this README instead of inline where possible.

## Apply Order

1. Install or sync monitoring first.
2. Sync the app manifests after monitoring is available.
3. Keep canary-related changes isolated to the `gitops/app/manifests/` directory.