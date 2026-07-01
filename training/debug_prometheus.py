import requests
import sys

# Dùng IP bạn đã cung cấp
PROMETHEUS_URL = "http://172.26.52.132:30090/api/v1/query"

def query_prometheus(query):
    try:
        # Tăng timeout vì kết nối qua WSL có thể trễ hơn một chút
        response = requests.get(PROMETHEUS_URL, params={'query': query}, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        results = data.get('data', {}).get('result', [])
        
        if results:
            return float(results[0]['value'][1])
        return 0.0
    except Exception as e:
        print(f"❌ Kết nối thất bại tới {PROMETHEUS_URL}: {e}")
        return 0.0

def test_metrics(svc_name):
    print(f"\n🚀 Đang truy vấn từ Windows -> Prometheus tại {PROMETHEUS_URL}")
    # 
    
    canary = f"{svc_name}-canary"
    
    # Query mẫu để test xem có nhận được dữ liệu không
    q = f'sum(rate(istio_requests_total{{destination_service=~"{canary}.*"}}[1m]))'
    val = query_prometheus(q)
    
    print(f"Kết quả cho {canary}: {val:.4f} req/s")
    if val == 0:
        print("⚠️ Không tìm thấy dữ liệu. Hãy đảm bảo service đang hoạt động và có traffic!")

if __name__ == "__main__":
    test_metrics("recommendationservice")