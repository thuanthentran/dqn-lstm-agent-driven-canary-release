import urllib.request
import urllib.parse
import json

q = urllib.parse.urlencode({'query': 'sum(rate(istio_requests_total{destination_service=~"checkoutservice.*", grpc_response_status!="0"}[1h]))'})
url = 'http://localhost:30090/api/v1/query?' + q
req = urllib.request.urlopen(url)
data = json.loads(req.read())
print("Errors (!= 0):", json.dumps(data['data']['result'], indent=2))
