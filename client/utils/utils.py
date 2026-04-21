EPSILON = 1e-8

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