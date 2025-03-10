import requests

url = "http://127.0.0.1:8000/add_object"
data = {
    "shape": "sphere",
    "position": [0, 0, 1],
    "size": [0.5, 0.5, 0.5],  # Only radius matters for a sphere
    "color": [0, 1, 0]  # Green color
}

response = requests.post(url, json=data)
print(response.json())
