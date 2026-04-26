import time
import sys
from mpu6050 import mpu6050

sensor = None


def init():
    global sensor

    try:
        # Endereço padrão do MPU-6050
        print(dir(mpu6050))
        print(mpu6050.mpu6050)
        sensor = mpu6050.mpu6050(0x68)
        print("MPU-6050 inicializado com sucesso")

    except Exception as e:
        raise RuntimeError("Falha ao inicializar MPU-6050: " + str(e))


def accelerometer_thread(stop_thread, state):
    global sensor

    state.features["accel_x"] = []
    state.features["accel_y"] = []
    state.features["gyro_x"] = []
    state.features["gyro_y"] = []

    try:
        if not sensor:
            init()
            time.sleep(2)

        while not stop_thread.is_set():

            print("Reading MPU-6050")

            if sensor:

                accel_data = sensor.get_accel_data()
                gyro_data = sensor.get_gyro_data()

                with state.lock:

                    state.features["accel_x"].append(
                        accel_data.get("x", 0)
                    )

                    state.features["accel_y"].append(
                        accel_data.get("y", 0)
                    )

                    state.features["gyro_x"].append(
                        gyro_data.get("x", 0)
                    )

                    state.features["gyro_y"].append(
                        gyro_data.get("y", 0)
                    )

            time.sleep(0.1)

    except Exception as e:
        print("Erro durante leitura do MPU-6050:", e)
        sys.exit(1)


init()
