import asyncio
import httpx
import os
import random
import time
from collections import Counter

# CẤU HÌNH
# Bơm vào ingress controller để Argo Rollouts có thể chia traffic stable/canary.
TARGET_URL = os.getenv(
    "TARGET_URL",
    "http://ingress-nginx-controller.ingress-nginx.svc.cluster.local/",
)
CONCURRENT_REQUESTS = int(os.getenv("CONCURRENT_REQUESTS", "10"))
REQUEST_INTERVAL_SECONDS = float(os.getenv("REQUEST_INTERVAL_SECONDS", "0.5"))
REPORT_EVERY_BATCHES = int(os.getenv("REPORT_EVERY_BATCHES", "5"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "5"))
HOST_HEADER = os.getenv("HOST_HEADER", "")

stats = Counter()


def _build_headers():
    headers = {}
    # Dùng Host header khi ingress rule match theo host.
    if HOST_HEADER:
        headers["Host"] = HOST_HEADER
    return headers

async def send_request(client):
    try:
        # Giả lập hành vi người dùng: random một chút thời gian chờ
        await asyncio.sleep(random.uniform(0.01, 0.1))
        response = await client.get(TARGET_URL, headers=_build_headers())
        stats[f"status_{response.status_code}"] += 1

        try:
            payload = response.json()
            version = payload.get("version", "unknown")
        except ValueError:
            version = "non-json"

        stats[f"version_{version}"] += 1
        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"Status: {response.status_code} | Version: {version}"
        )
    except Exception as e:
        stats["errors"] += 1
        print(f"Error: {e}")


def print_summary(batch_id):
    summary = [f"batch={batch_id}"]
    status_parts = sorted(
        (k.replace("status_", ""), v) for k, v in stats.items() if k.startswith("status_")
    )
    version_parts = sorted(
        (k.replace("version_", ""), v) for k, v in stats.items() if k.startswith("version_")
    )

    if status_parts:
        summary.append(
            "status{" + ", ".join(f"{code}:{count}" for code, count in status_parts) + "}"
        )
    if version_parts:
        summary.append(
            "version{" + ", ".join(f"{name}:{count}" for name, count in version_parts) + "}"
        )
    if stats.get("errors", 0):
        summary.append(f"errors:{stats['errors']}")

    print("[SUMMARY] " + " | ".join(summary))


async def main():
    print(f"--- Bắt đầu bơm traffic vào {TARGET_URL} ---")
    if HOST_HEADER:
        print(f"--- Dùng Host header: {HOST_HEADER} ---")

    timeout = httpx.Timeout(REQUEST_TIMEOUT_SECONDS)
    async with httpx.AsyncClient(timeout=timeout) as client:
        batch_id = 0
        while True:
            batch_id += 1
            tasks = [send_request(client) for _ in range(CONCURRENT_REQUESTS)]
            await asyncio.gather(*tasks)
            if batch_id % REPORT_EVERY_BATCHES == 0:
                print_summary(batch_id)
            # Nghỉ một chút để không làm sập cluster của bạn nếu cấu hình thấp
            await asyncio.sleep(REQUEST_INTERVAL_SECONDS)

if __name__ == "__main__":
    asyncio.run(main())