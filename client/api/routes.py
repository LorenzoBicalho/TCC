import requests
from api.client import get, post


def register_client(device_id):

    data = {
        "device_identifier": device_id,
        "description": ''
    }

    response = post("/clients", data)

    print(response.json())
    return response

def get_latest_model(device_id, model_version):

    data = {
        "device_identifier": device_id,
        "client_version": model_version
    }

    response = post("/model/latest", data)

    print(response.json())
    return response

def send_local_weights(device_id, trained_params, metrics, num_samples, version):

    data = {
        "device_identifier": device_id,
        "weights": trained_params,
        "metrics": metrics,
        "num_samples": num_samples,
        "version": version
    }

    response = post("/weights", data)

    print(response.json())
    return response

def send_telemetry(device_id, version, rows):

    data = {
        "device_identifier": device_id,
        "version": version,
        "telemetry": rows
    }

    response = post("/weights", data)

    print(response.json())
    return response
