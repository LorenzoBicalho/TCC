from serial import Serial, SerialException
import threading
import requests
import time
import uuid
from config import SERIAL_NUMBER, SERIAL_PORT, SERVER_URL
from db.repositories import featuresRepository
from utils import utils
from utils import hardware
import services.decoders as decoders_service
import api.routes as api_routes
from services import neurofuzzy as neurofuzzy_service
import services.accelerometer as accelerometer_service
import services.elm as elm_service
from db.repositories import modelRepository

PORTA_SERIAL = SERIAL_PORT
BAUDRATE = 38400

supported_pids = []

stop_thread = threading.Event()
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.features = {}


state = State()

def train_thread(stop_event, device_id):
    while not stop_event.is_set():
        try:
            print("Checking data count")
            data_count = featuresRepository.get_data_count()
            if  data_count >= 1600:
                if hardware.check_internet_connection():
                    print(f"Trainini new model with {data_count} samples")
                    trained_params, metrics, num_samples, version = neurofuzzy_service.train_model()
                    rows = featuresRepository.get_data()

                    weights_sent = False
                    try:
                        print(f"Sent weights")
                        response = api_routes.send_local_weights(device_id, trained_params, metrics, num_samples, version)
                        weights_sent = response.status_code == 200
                    except Exception as e:
                        print(f"Weight send failed: {e}")

                    if weights_sent:
                        print(f"Sent weights")
                        try:
                            print(f"Sending telemetry data to the server")
                            api_routes.send_telemetry(device_id, version, rows)
                        except Exception as e:
                            print(f"Telemetry send failed, keeping data: {e}")
                            # don't delete — will retry next cycle
                        else:
                            print(f"Telemetry data sent")
                            print(f"Deleting all local telemetry data")
                            featuresRepository.delete_all_data()
                            print(f"Local telemetry data deleted")
                else:
                    print("Ready to train new model, but internet connection needed")

        except Exception as e:
            print(f"train_thread error: {e}")

        time.sleep(60)

if __name__ == "__main__":

    read_elm_thread = None
    read_acc_thread = None
    send_thread = None
    train_thread_handle = None

    try:

        if SERIAL_NUMBER in ["UNKNOWN", "ERROR"]:
            raise RuntimeError("Device ID not available")

        device_id = SERIAL_NUMBER.lower()

        print(f"Verifying server {SERVER_URL} connection...")
        is_connected = hardware.check_internet_connection()
        hardware.require_internet_buzz(is_connected)
        if is_connected:
            print(f"Connection established")
            client_info = api_routes.register_client(device_id)
            device_id = client_info.get('device_identifier')
            model = modelRepository.get_global_model()
            model_version = model.version if model is not None else 0
            
            latest_model = api_routes.get_latest_model(device_id, model_version)
            params = latest_model.get('model')
            current_version = latest_model.get('current_version')

            if latest_model['has_update'] == 1:
                print("Global Model has update")
                modelRepository.delete_all_models()
                modelRepository.insert_global_model(params, current_version)
                print("Local Model updated")
        else:
            print(f"No connection established. Initiating offline mode.]")
            params = modelRepository.get_global_model()
            current_version = params['version']
            if params is None:
                raise RuntimeError("No model available in offline mode")

        print(f"Conectando à porta {PORTA_SERIAL}...")

        with Serial(PORTA_SERIAL, BAUDRATE, timeout=1) as serial_conn:

            session_id = str(uuid.uuid4())

            train_thread_handle = threading.Thread(
                name="train_model",
                target=train_thread,
                args=(stop_thread, device_id),
                daemon=True
            )
            train_thread_handle.start()

            elm_service.init_elm(serial_conn)

            print("Conectado. Lendo PIDs disponíveis...")

            elm_service.get_all_supported_pids(serial_conn)

            print(f"Supported PIDs in mode 01: {supported_pids}")

            read_elm_thread = threading.Thread(
                name="read_elm",
                target=elm_service.read_thread,
                args=(stop_thread, serial_conn, state),
                daemon=True,
            )
            read_elm_thread.start()

            read_acc_thread = threading.Thread(
                name="read_accelerometer",
                target=accelerometer_service.accelerometer_thread,
                args=(stop_thread, state),
                daemon=True,
            )
            read_acc_thread.start()

            while not stop_thread.is_set():
                data = utils.format_data(state)
                print(f"Reading complete. Data: {data}")
                if data.get("speed", 0) != 0:
                    classification = neurofuzzy_service.calys(data, params)
                    driver_class = max(1, min(3, round(classification)))

                    if driver_class == 3: hardware.aggressive_buzz()

                    featuresRepository.insert_data(data, session_id)
                    print(f"Data inserted in local database")


    except SerialException as e:
        print(f"Erro na conexão serial: {e}")
    except KeyboardInterrupt:
        print("Programa interrompido pelo usuário.")
        stop_thread.set()
    except RuntimeError as e:
        print(e)
    except requests.exceptions.Timeout:
        print("Timeout")
    except requests.exceptions.ConnectionError:
        print("Sem conexão")
    except requests.exceptions.HTTPError as e:
        print("Erro HTTP:", e)

    finally:
        stop_thread.set()
        if read_elm_thread:
            read_elm_thread.join()
        if read_acc_thread:
            read_acc_thread.join()
        if send_thread:
            send_thread.join()
        if train_thread_handle:
            train_thread_handle.join()