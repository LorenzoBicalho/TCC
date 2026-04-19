import serial
import time
import threading
import json
from client.api.routes import register_client, get_latest_model
from client.db.index import insert_data
from client.services.uart import uart_service
import paho.mqtt.client as mqtt
from decoders import pid_decoders
from mqtt_client import init_client, publish
from config import SERIAL_NUMBER, SERIAL_PORT, SERVER_URL
import accelerometer
from mqtt_client import publish


# [OBD2] --> [Modulo USB] ---> [Processamento (Python)] -- publish ---> [Broker MQTT] ---> [Logger] e [Frontend]

# Set COM port for bluetooth conection with ELM327
PORTA_SERIAL = SERIAL_PORT 
# Set update rate of ELM327
BAUDRATE = 38400
supported_pids = []

stop_thread = threading.Event()

class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.features = {}

state = State()

if __name__ == "__main__":
    try:
        print(f"Verificando conexão com servidor {SERVER_URL}...")

        if SERIAL_NUMBER in ["UNKNOWN", "ERROR"]:
            raise RuntimeError("Device ID not available")

        device_id = SERIAL_NUMBER.lower()

        client = register_client(device_id)

        model_version = '' # TO-DO: GET FROM LOCAL MODEL

        latest_model = get_latest_model(client.device_identifier, model_version)

        update_fpga_model = uart_service.send_global_weiths("/dev/ttyS0", 115200, latest_model)
        
        if latest_model.has_update == true:
            # atualiza global weights do bd
        
        # começa o ciclo. leitura de dados, inferencia, salva no bd dados (sem inferencia). Se der 1600, roda treinamento. envia para servidor 
       
        print(f"Conectando à porta {PORTA_SERIAL}...")

        # TO-DO create thread to read obd data, store and classificate
        with serial.Serial(PORTA_SERIAL, BAUDRATE, timeout=1) as serial:
            init_elm(serial)
            print("Conectado. Lendo PIDs disponíveis...\n")
            get_all_supported_pids(serial)
            print(f"Supported PIDs in mode 01: {[pid for pid in supported_pids]}")
            print("=" * 40)

            read_elm_thread = None
            read_acc_thread = None
            send_thread = None

            try:
                # Initialize MQTT client and keep on first thread
                client = mqtt.Client()
                init_client(client)

                # Set read on second thread
                read_elm_thread = threading.Thread(
                        name="read_elm",
                        target=read_thread,
                        args=(serial, client),
                        daemon=True
                )
                read_elm_thread.start()

                read_acc_thread = threading.Thread(
                        name="read_accelerometer",
                        target=accelerometer.accelerometer_thread,
                        args=(stop_thread, state),
                        daemon=True
                )
                read_acc_thread.start()

                data = format_data(state)
                print(data)

                # TO-DO: thread para salvar os dados
                if data['speed'] not 0:
                    insert_data(data)
                    publish(client, data)

                    # thread para fazer inferencia
                    classificate(data)

            # Stops threads
            except KeyboardInterrupt:
                print("\nExiting...")
                stop_thread.set()
                if read_elm_thread:
                    read_elm_thread.join()
                if read_acc_thread:
                    read_acc_thread.join()
                if send_thread:
                    send_thread.join()

        # if obd2_data table >= 1600:
        # TO-DO create thread to train new local model
        #     train_model()
        #     send_local_weights()
        #     delete data from obd2_data table


    except serial.SerialException as e:
        print(f"Erro na conexão serial: {e}")
    except KeyboardInterrupt:
        print("\nPrograma interrompido pelo usuário.")
        stop_thread.set()
        if read_elm_thread:
            read_elm_thread.join()
        if read_acc_thread:
            read_acc_thread.join()
        if send_thread:
            send_thread.join()
    except RuntimeError as e:
        print(e)
    except requests.exceptions.Timeout:
        print("Timeout")
    except requests.exceptions.ConnectionError:
        print("Sem conexão")
    except requests.exceptions.HTTPError as e:
        print("Erro HTTP:", e)
