import kopf
import kubernetes.client
from kubernetes.client.rest import ApiException
import logging

logger = logging.getLogger(__name__)

def create_agent_deployment_dict(name, namespace, spec):
    image = spec.get("image", "ghcr.io/thuanthentran/agent:latest")
    replicas = spec.get("replicas", 1)
    model_path = spec.get("modelPath", "models/ppo_lstm_offline_best.zip")
    prometheus_url = spec.get("prometheusUrl")
    port = spec.get("port", 8000)

    return {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": f"{name}-deployment",
            "namespace": namespace,
            "labels": {
                "app": name,
                "component": "rl-agent"
            }
        },
        "spec": {
            "replicas": replicas,
            "selector": {
                "matchLabels": {
                    "app": name,
                    "component": "rl-agent"
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": name,
                        "component": "rl-agent"
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "name": "agent",
                            "image": image,
                            "ports": [{"containerPort": port}],
                            "env": [
                                {"name": "MODEL_PATH", "value": model_path},
                                {"name": "PROMETHEUS_URL", "value": prometheus_url}
                            ]
                        }
                    ]
                }
            }
        }
    }

def create_agent_service_dict(name, namespace, spec):
    port = spec.get("port", 8000)

    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {
            "name": f"{name}-service",
            "namespace": namespace,
            "labels": {
                "app": name,
                "component": "rl-agent"
            }
        },
        "spec": {
            "selector": {
                "app": name,
                "component": "rl-agent"
            },
            "ports": [
                {
                    "port": port,
                    "targetPort": port,
                    "protocol": "TCP",
                    "name": "http"
                }
            ]
        }
    }

@kopf.on.create('rl.thuanthentran.io', 'v1alpha1', 'rlagents')
def create_fn(name, namespace, spec, logger, **kwargs):
    logger.info(f"Creating RLAgent: {name} in {namespace}")
    api = kubernetes.client.AppsV1Api()
    core_api = kubernetes.client.CoreV1Api()
    
    # Define Deployment
    deployment_obj = create_agent_deployment_dict(name, namespace, spec)
    kopf.adopt(deployment_obj)
    
    # Define Service
    service_obj = create_agent_service_dict(name, namespace, spec)
    kopf.adopt(service_obj)
    
    # Create Resources
    try:
        api.create_namespaced_deployment(namespace=namespace, body=deployment_obj)
        logger.info(f"Deployment {deployment_obj['metadata']['name']} created")
    except ApiException as e:
        logger.error(f"Failed to create Deployment: {e}")
        raise kopf.PermanentError(f"Failed to create Deployment: {e}")

    try:
        core_api.create_namespaced_service(namespace=namespace, body=service_obj)
        logger.info(f"Service {service_obj['metadata']['name']} created")
    except ApiException as e:
        logger.error(f"Failed to create Service: {e}")
        raise kopf.PermanentError(f"Failed to create Service: {e}")
    
    return {'message': 'Created successfully', 'deployment': deployment_obj['metadata']['name'], 'service': service_obj['metadata']['name']}

@kopf.on.update('rl.thuanthentran.io', 'v1alpha1', 'rlagents')
def update_fn(name, namespace, spec, logger, **kwargs):
    logger.info(f"Updating RLAgent: {name} in {namespace}")
    api = kubernetes.client.AppsV1Api()
    core_api = kubernetes.client.CoreV1Api()
    
    deployment_obj = create_agent_deployment_dict(name, namespace, spec)
    kopf.adopt(deployment_obj)
    
    service_obj = create_agent_service_dict(name, namespace, spec)
    kopf.adopt(service_obj)
    
    try:
        api.patch_namespaced_deployment(
            name=deployment_obj['metadata']['name'],
            namespace=namespace,
            body=deployment_obj
        )
        logger.info(f"Deployment {deployment_obj['metadata']['name']} updated")
    except ApiException as e:
        logger.error(f"Failed to update Deployment: {e}")
        
    try:
        core_api.patch_namespaced_service(
            name=service_obj['metadata']['name'],
            namespace=namespace,
            body=service_obj
        )
        logger.info(f"Service {service_obj['metadata']['name']} updated")
    except ApiException as e:
        logger.error(f"Failed to update Service: {e}")
