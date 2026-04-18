import serial
import time
import threading
import json
import paho.mqtt.client as mqtt
from decoders import pid_decoders
from mqtt_client import init_client, publish
from config import SERIAL_PORT
import accelerometer
import uart

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

# Sends request to OBDII and returns its answer
def send_cmd(serial, cmd, timeout=3):
    serial.write((cmd + "\r").encode())

    buffer = b""
    start_time = time.time()

    while True:
        chunk = serial.read(serial.in_waiting or 1)
        if chunk:
            buffer += chunk

            # Verifica o prompt do ELM327
            if b'>' in buffer:
                break

        # Timeout de segurança
        if time.time() - start_time > timeout:
            print(f"Timeout esperando resposta para {cmd}")
            break
    
        time.sleep(0.01)  # pequena pausa para evitar 100% CPU

    # Processa saída
    lines = buffer.decode(errors='ignore').replace('\r', '\n').split('\n')
    return [line.strip() for line in lines if line.strip()]


def init_elm(serial):
    # ATZ	Reset ELM327
    # ATE0	Turn off echo (no command repetition)
    # ATL0	Turn off long line format
    # ATS0	Turn off spaces between bites
    # ATH0	Turn off headers (CAN adresses)
    # ATSP0	Select automatic protocol
    print("Initializing OBD-II\n")
    init_cmds = ["ATZ", "ATE0", "ATL0", "ATS0", "ATH0", "ATSP0"]  # Reset + standard configs
    for cmd in init_cmds:
        print(f">>> {cmd}")
        resposta = send_cmd(serial, cmd)
        print("\n".join(resposta))
        time.sleep(0.3)

# Gets all the supported PIDs in a given range
# Returns a list of indexes for the supported PIDs
def parse_supported_pids(bytes):
    try:
        # Filter valid data lines (starts with "41")
        data_line = next((line for line in bytes if line.startswith("41")), None)
        if not data_line:
            return []
        
        # Separate answer bytes, removes spaces and joins data
        data = ''.join(data_line.split())[4:]
        bits = bin(int(data, 16))[2:].zfill(32)

        supported = []
        for i, bit in enumerate(bits):
            if bit == '1':
                supported.append(i + 1)
        return supported
    except Exception as e:
        print(f"Error parsing supported PIDs: {e}")
        return []

# Gets all supported PIDs in mode 01 and returns a list of commands
def get_all_supported_pids(serial):
    global supported_pids
    for start in [0x00, 0x20, 0x40, 0x60, 0x80, 0xA0, 0xC0, 0xE0]:
        cmd = f"01{start:02X}"
        answ = send_cmd(serial, cmd)
        pids = parse_supported_pids(answ)
        if not pids:
            break  # Stops if there are no more supported PIDs
        for pid in pids:
            full_pid = start + pid
            supported_pids.append(f"01{full_pid:02X}")

def process_data(cmd, data):
    try:
        # Check if reading was successfull
        print(data)
        data_line = next((line for line in data if line.startswith(f"41{cmd[2:]}")), None)
        if not data_line:
            return None
        
        hexvalue = data_line[4:]  # gets only response value bytes
        decoder = pid_decoders.get(cmd, lambda x: 0) # gets decoder func for command
        return decoder(hexvalue)
    except Exception as e:
        print(f"Erro ao processar dados do PID {cmd}: {e}")
        return None

def read_pid(serial, cmd):
    answ = [] # List of bytes for the request answer

    # Gets answer bytes
    if cmd in supported_pids:
        answ = send_cmd(serial, cmd)
    
    answ = process_data(cmd, answ) # Process answer bytes to generate plotable info
    print(answ)
    return answ

def calc_fuel_usage(serial):
    maf = read_pid(serial, '0111')
    eq_ratio = read_pid(serial, '0144') 
    if maf is not None and eq_ratio is not None:
        afr = 14.7 * eq_ratio
        fuel = maf/afr
        fuel = (fuel*3600) / 754
        return fuel
    return 0.0

# Reading thread. Gets all pid data and inserts in a json
def read_thread(serial, client):
    global state
    while not stop_thread.is_set():
        with state.lock:
            state.features["speed"] = read_pid(serial, "010D") if "010D" in supported_pids else 0.0
            state.features["rpm"] = read_pid(serial, "010C") if "010C" in supported_pids else 0.0
            state.features["pos_pedal"] = read_pid(serial, "0111") if "0111" in supported_pids else 0.0

        # data_queue.put((time.time(), value))
        time.sleep(4)


if __name__ == "__main__":
    try:
        print(f"Conectando à porta {PORTA_SERIAL}...")
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

                send_thread = threading.Thread(
                        name="uart_send",
                        target=uart.send_thread,
                        args=(stop_thread, state, "/dev/ttyS0", 115200, client),
                        daemon=True
                )
                send_thread.start()
                while (read_elm_thread.is_alive()):
                    time.sleep(0.5)

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
