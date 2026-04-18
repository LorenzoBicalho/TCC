import serial
import time
import struct
import json
from mqtt_client import publish

uart: serial.Serial | None = None


def init(port, baud):
    global uart
    if not uart:
        uart = serial.Serial(
            port=port,
            baudrate=baud,
            timeout=0.1  # leitura não bloqueante
        )


def send_thread(stop_thread, state, serial_port, baudrate, client):
    global uart

    if not uart:
        init(serial_port, baudrate)

    while not stop_thread.is_set():
        if uart and state.features.get('accel_x') and state.features.get('accel_y'):
            with state.lock:
                accel_x_avg = sum(state.features['accel_x']) / len(state.features['accel_x'])
                accel_y_avg = sum(state.features['accel_y']) / len(state.features['accel_y'])
                rpm = state.features.get('rpm', 0)
                speed = state.features.get('speed', 0)
                pos_pedal = state.features.get('pos_pedal', 0)

                packet = struct.pack(
                    '<fffff',
                    sum(state.features['accel_x'])/len(state.features['accel_x']),
                    sum(state.features['accel_y'])/len(state.features['accel_y']),
                    state.features["rpm"],
                    state.features["speed"],
                    state.features["pos_pedal"],
                )

                data = {
                    "speed": speed,
                    "acc_long": accel_x_avg,
                    "acc_lat": accel_y_avg,
                    "engine_speed": rpm,
                    "throttle_position": pos_pedal
                }
                
                print(data)

                payload = json.dumps(data)
                publish(client, payload)
                
                state.features['accel_x'] = []
                state.features['accel_y'] = []
             #  uart.write(packet)

        time.sleep(1)
