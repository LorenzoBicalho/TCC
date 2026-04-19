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