import time
import sys
from adafruit_bus_device.i2c_device import I2C
import board
import busio
import adafruit_bno055


i2c: I2C | None = None
sensor: adafruit_bno055.BNO055_I2C | None = None


def init():
    global i2c, sensor

    try:
        i2c = busio.I2C(board.SCL, board.SDA)
    except Exception as e:
        raise RuntimeError("Falha ao inicializar I2C: " + str(e))

    try:
        sensor = adafruit_bno055.BNO055_I2C(i2c)
    except Exception as e:
        raise RuntimeError("Falha ao inicializar BNO055: " + str(e))


def accelerometer_thread(stop_thread, state):
    global i2c, sensor

    state.features["accel_x"] = []
    state.features["accel_y"] = []

    try:
        if not i2c or not sensor:
            init()
            time.sleep(5)

        while not stop_thread.is_set():
            print(f"Reading accelerometer")
            if sensor:
                accel = sensor.acceleration
                if accel:
                    with state.lock:
                        state.features["accel_x"].append(accel[0] or 0)
                        state.features["accel_y"].append(accel[1] or 0)
            time.sleep(0.1)

    except Exception as e:
        print("Erro durante leitura do acelerometro:", e)
        sys.exit(1)
