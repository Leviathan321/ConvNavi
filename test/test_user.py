import requests
import json
import time
import sys

def test_rag_navigation(query):
    url = "http://127.0.0.1:8000/query"

    payload = {
        "query": query,
        "user_location": [39.955431, -75.154903],
        "llm_type": "llama3.2"
    }

    headers = {
        "Content-Type": "application/json"
    }

    start_time = time.time()

    try:
        print("payload sent:", payload)
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        duration = time.time() - start_time

        print("Response from server:")
        print(json.dumps(response.json(), indent=2))
        print(f"Request completed in {duration:.2f} seconds")
        return response
    except requests.exceptions.RequestException as e:
        duration = time.time() - start_time
        print(e)
        print(f"Request failed after {duration:.2f} seconds")
        sys.exit(0)

if __name__ == "__main__":
    queries = [
        "My hair needs a haircut, rating more than four.",
        "Yo man give me some italian flavours.",
        "I need to bye some new jeans.",
        "Recommend some petrol station closeby.",
        "Need a hotel, 5 stars, not expensive.",
        "Recommend some fancy bars with cocktails tonight?",
        "Need to by some cigarettes, Marlboro.",
        "I am in the mood to eat some chicken burger, not expensive.",
        "Is there some museum close by?",
        # "Need some breakfast now.",
        # "I will have a date today and want try some burger restaurant.",
        # "I am in the mood for some asian food close by rating 4.",
        # "My parents will visit my city, any american restaurant to check out?",
        # "I like to have some english breakfast, not expensive."
    ]
    for query in queries:
        test_rag_navigation(query)
