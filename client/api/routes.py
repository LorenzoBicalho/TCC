import requests
from api.client import get, post


def register_client(device_id):

    data = {
        "device_identifier": device_id,
        "description": ''
    }

    response = post("/clients", data)

    print(response)
    return response

def get_latest_model(device_id, model_version):

    data = {
        "device_identifier": device_id,
        "client_version": model_version
    }

    response = post("/model/latest", data)

    print(response)
    return response

def send_local_weights(device_id, model_version, weights, num_samples):

    data = {
        "device_identifier": device_id,
        "version": model_version,
        "weights": weights,
        "num_samples": num_samples
    }

    response = post("/weights", data)

    print(response)
    return response
