import requests
import json

BASE_URL = "http://localhost:8000"
USER_ID = 1

# Step 1: Set initial car state
initial_state_payload = {
    "user_id": USER_ID,
    "car_state": {
        "windows": {
            "window_front_left": "closed",
            "window_front_right": "closed",
            "window_rear_left": "closed",
            "window_rear_right": "closed"
        },
        "lights": {
            "head_light": "off",
            "fog_light": "off",
            "ambient_light": "off",
            "reading_light_front_left": "off",
            "reading_light_front_right": "off",
            "reading_light_rear_left": "off",
            "reading_light_rear_right": "off"
        },
        "climate": {
            "temperature": 21,
            "climate": "off",
            "fan": "off"
        },
        "seat_heating": {
            "seat_heating_front_left": "off",
            "seat_heating_front_right": "off",
            "seat_heating_rear_left": "off",
            "seat_heating_rear_right": "off"
        }
    }
}

resp = requests.post(f"{BASE_URL}/carstate/init", json=initial_state_payload)
print(json.dumps(resp.json(), indent=2))

# Step 2: Send query via /query
query_payload = {
    "user_id": USER_ID,
    "query": "Turn on the head light."
}

resp = requests.post(f"{BASE_URL}/query", json=query_payload)
print(json.dumps(resp.json(), indent=2))
