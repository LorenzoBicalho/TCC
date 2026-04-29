import numpy as np

EPSILON = 1e-8

FEATURE_ORDER = [
    "speed",
    "acc_long",
    "acc_lat",
    "engine_speed",
    "throttle_position",
]

def format_data(state):
    accel_x_avg = sum(state.features['accel_x']) / (len(state.features['accel_x']) + EPSILON)
    accel_y_avg = sum(state.features['accel_y']) / (len(state.features['accel_y']) + EPSILON)
    rpm = state.features.get('rpm', 0)
    speed = state.features.get('speed', 0)
    pos_pedal = state.features.get('pos_pedal', 0)

    data = {
        "speed": speed,
        "acc_long": accel_x_avg,
        "acc_lat": accel_y_avg,
        "engine_speed": rpm,
        "throttle_position": pos_pedal
    }

    return data

def get_field(obj, name):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None

def dict_to_feature_vector(data):
    return np.array(
        [float(data[name]) for name in FEATURE_ORDER],
        dtype=float
    )