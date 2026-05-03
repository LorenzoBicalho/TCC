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

stop_thread = threading.Event()
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.new_data = threading.Event()
        self.features = {}


state = State()

def train_thread(stop_event, device_id):
    while not stop_event.is_set():
        try:
            print("Checking data count \n")
            data_count = featuresRepository.get_data_count()
            if  data_count >= 300:
                if hardware.check_internet_connection():
                    print(f"Training new model with {data_count} samples")
                    trained_params, metrics, num_samples, version, labels_by_row_id = (
                        neurofuzzy_service.train_model()
                    )
                    labeled_ids = list(labels_by_row_id.keys())
                    labeled_set = set(labeled_ids)
                    rows = [
                        r
                        for r in featuresRepository.get_data()
                        if getattr(r, "id", None) in labeled_set
                    ]

                    weights_sent = False
                    try:
                        print(f"Sending weights to the server")
                        response = api_routes.send_local_weights(
                            device_id, trained_params, metrics, num_samples, version
                        )
                        if response.status_code == 200:
                            body = response.json()
                            weights_sent = body.get("status") == "success"
                            if not weights_sent:
                                print(
                                    f"Weights not stored ({body.get('status')}): {body.get('detail')}"
                                )
                    except Exception as e:
                        print(f"Weight send failed: {e}")

                    if weights_sent:
                        print(f"Sent weights")
                        try:
                            print(f"Sending telemetry data to the server")
                            api_routes.send_telemetry(
                                device_id, version, rows, labels_by_row_id
                            )
                        except Exception as e:
                            print(f"Telemetry send failed, keeping data: {e}")
                            # don't delete — will retry next cycle
                        else:
                            print(f"Telemetry data sent")
                            print(f"Removing uploaded sample rows from local DB")
                            featuresRepository.delete_rows_by_ids(labeled_ids)
                            print(f"Uploaded telemetry rows deleted locally")
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

        print(f"Checking internet connection...")
        is_connected = hardware.check_internet_connection()
        is_connected = hardware.require_internet_buzz(is_connected)

        is_server_on = False
        if is_connected:
            print(f"Hardware is connected to the internet")
            print(f"Verifying server {SERVER_URL} connection...")

            try:
                client_info = api_routes.register_client(device_id)
                is_server_on = client_info.status_code == 200

                if is_server_on:
                    print(f"Connected to server: {SERVER_URL}")
                else:
                    print(f"Failed trying to connect to server: {SERVER_URL}")

            except requests.exceptions.Timeout:
                print(f"Server {SERVER_URL} timed out. Proceeding in offline mode.")
            except requests.exceptions.ConnectionError:
                print(f"Could not reach server {SERVER_URL}. Proceeding in offline mode.")
            except Exception as e:
                print(f"Unexpected error contacting server: {e}. Proceeding in offline mode.")

        if is_connected and is_server_on:
            response_data = client_info.json()
            device_id = response_data.get('device_identifier')

            local_model = modelRepository.get_global_model()
            model_version = local_model.version if local_model is not None else 0

            response = api_routes.get_latest_model(device_id, model_version)
            response_data = response.json()

            global_model = response_data.get('model')
            current_version = response_data.get('current_version')
            latest_model_has_update = response_data.get('has_update')

            if latest_model_has_update:
                print("Global Model has update")
                if global_model is None:
                    raise RuntimeError(
                        "Server reported a model update but sent no model payload."
                    )
                modelRepository.delete_all_models()
                modelRepository.insert_global_model(global_model, current_version)
                print("Local Model updated")

            local_model = modelRepository.get_global_model()
            if local_model is None:
                raise RuntimeError(
                    "No global model on device after sync. "
                    "Publish a model on the server or install weights locally before running online."
                )
            params = modelRepository.get_global_params(local_model)
        else:
            print("Initiating offline mode.")
            local_model = modelRepository.get_global_model()
            if local_model is None:
                raise RuntimeError("No model available in offline mode")
            params = modelRepository.get_global_params(local_model)  
            current_version = local_model.version

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

            print(f"Supported PIDs in mode 01: {elm_service.supported_pids}")

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
                state.new_data.wait(timeout=2)  # blocks until new data arrives
                state.new_data.clear()
                data = utils.format_data(state)
                print(f"Reading complete. Data: {data}")
                if data.get("speed", 0) != 0:
            
                    inputs = neurofuzzy_service.normalize_matrix(data)
                    print(f'x normalized: {inputs}')
                    classification, _, _, _ = neurofuzzy_service.calys(inputs, params)
                    driver_class = max(1, min(3, round(classification)))
                    if driver_class == 3:
                        print(f'Classificação: Agressivo ({classification})')
                    if driver_class == 2:
                        print(f'Classificação: Normal ({classification})')
                    if driver_class == 1:
                        print(f'Classificação: Calma ({classification})')

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