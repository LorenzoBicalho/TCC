import numpy as np
import config

EPSILON = 1e-8

def format_data(state):
    with state.lock:
        ax = state.features.get("accel_x") or []
        ay = state.features.get("accel_y") or []
        accel_x_avg = sum(ax) / (len(ax) + EPSILON)
        accel_y_avg = sum(ay) / (len(ay) + EPSILON)
        rpm = state.features.get("rpm", 300)
        speed = state.features.get("speed", 10)
        pos_pedal = state.features.get("pos_pedal", 16)
        for key in ("accel_x", "accel_y", "gyro_x", "gyro_y"):
            lst = state.features.get(key)
            if lst is not None:
                lst.clear()

    return {
        "speed": speed,
        "acc_long": accel_x_avg,
        "acc_lat": accel_y_avg,
        "engine_speed": rpm,
        "throttle_position": pos_pedal,
    }

def get_field(obj, name):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name)
    return None

def dict_to_feature_vector(data):
    if not isinstance(data, dict):
        raise TypeError(
            f"Expected 'data' to be dict, got {type(data).__name__}."
        )

    missing_keys = [name for name in config.FEATURE_ORDER if name not in data]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(
            f"Missing required feature keys: {missing}."
        )

    try:
        vector = np.array(
            [float(data[name]) for name in config.FEATURE_ORDER],
            dtype=float
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "All feature values must be numeric and convertible to float."
        ) from exc

    return vector