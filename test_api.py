import requests
import time

print('Testing API performance after optimization...')
start = time.time()
response = requests.post('http://localhost:8000/recommend',
                        json={'user_id':'A0148968UM59JS3Y8D1M','category':'clothing','num_recommendations':5},
                        timeout=30)
end = time.time()

print(f'Response time: {end-start:.2f} seconds')
if response.status_code == 200:
    data = response.json()
    print(f'Success! Got {len(data["recommendations"])} recommendations')
    for rec in data['recommendations'][:3]:
        print(f'  - {rec["product_id"]}: {rec["score"]:.3f}')
else:
    print(f'Error: {response.status_code}')