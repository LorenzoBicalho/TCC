import requests
from config import SERVER_URL

def get(endpoint):
    url = f"{SERVER_URL}{endpoint}"

    response = requests.get(url)

    response.raise_for_status()

    return response.json()

def post(endpoint, data):
    url = f"{SERVER_URL}{endpoint}"

    response = requests.post(
        url,
        json=data
    )

    response.raise_for_status()

    return response.json()