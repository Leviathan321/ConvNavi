import requests
import json

BASE_URL = "http://localhost:8000"
USER_ID = 1

# Step 1: Set initial car state
initial_state_payload = {
    "user_id": USER_ID,
    "car_state": {
        "windows": {
            "front_left": "closed",
            "front_right": "closed",
            "rear_left": "closed",
            "rear_right": "closed"
        },
        "lights": {
            "headlights": "low",
            "fog_lights": "off",
            "interior_front": "off",
            "interior_rear": "off",
            "ambient": "off"
        },
        "climate": {
            "temperature_c": 21,
            "fan_level": 2,
            "mode": "auto"
        },
        "seat_heating": {
            "driver": "off",
            "front_passenger": "off"
        }
    }
}

resp = requests.post(f"{BASE_URL}/carstate/init", json=initial_state_payload)
print(json.dumps(resp.json(), indent=2))

# Step 2: Send query via /query
query_payload = {
    "user_id": USER_ID,
    "query": "Turn on the headlights to low."
}

resp = requests.post(f"{BASE_URL}/query", json=query_payload)
print(json.dumps(resp.json(), indent=2))
