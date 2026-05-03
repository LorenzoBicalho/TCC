import requests
import numpy as np
from api.client import get, post

def _to_json_compatible(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value

def _serialize_telemetry_row(row):
    return {
        "local_id": _to_json_compatible(getattr(row, "id", None)),
        "created_at": _to_json_compatible(getattr(row, "created_at", None)),
        "speed": _to_json_compatible(getattr(row, "speed", None)),
        "acc_long": _to_json_compatible(getattr(row, "acc_long", None)),
        "acc_lat": _to_json_compatible(getattr(row, "acc_lat", None)),
        "engine_speed": _to_json_compatible(getattr(row, "engine_speed", None)),
        "throttle_position": _to_json_compatible(getattr(row, "throttle_position", None)),
        "session_id": _to_json_compatible(getattr(row, "session_id", None)),
    }


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
        "weights": _to_json_compatible(trained_params),
        "metrics": _to_json_compatible(metrics),
        "num_samples": _to_json_compatible(num_samples),
        "version": _to_json_compatible(version)
    }

    response = post("/weights", data)

    print(response.json())
    return response

def send_telemetry(device_id, version, rows):
    telemetry = [_serialize_telemetry_row(row) for row in rows]

    data = {
        "device_identifier": device_id,
        "version": _to_json_compatible(version),
        "telemetry": telemetry
    }

    response = post("/telemetry", data)

    print(response.json())
    return response
