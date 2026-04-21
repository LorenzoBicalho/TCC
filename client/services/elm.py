import time
from typing import List, Optional

from . import decoders

# Global runtime state containers

supported_pids: List[str] = []

def init_elm(serial) -> None:
    """
    Initialize ELM327 adapter with standard configuration commands.

    ATZ   Reset device
    ATE0  Disable echo
    ATL0  Disable linefeeds
    ATS0  Disable spaces in responses
    ATH0  Disable headers
    ATSP0 Automatic protocol detection
    """

    print("Initializing OBD-II")

    init_cmds = [
        "ATZ",
        "ATE0",
        "ATL0",
        "ATS0",
        "ATH0",
        "ATSP0",
    ]

    for cmd in init_cmds:
        print(f">>> {cmd}")

        response = send_cmd(serial, cmd)

        for line in response:
            print(line)

        time.sleep(0.3)

def send_cmd(serial, cmd: str, timeout: float = 3.0) -> List[str]:
    """
    Send command to the ELM327 device and return response lines.
    """

    serial.write((cmd + "\r").encode())

    buffer = b""

    start_time = time.time()

    while True:
        chunk = serial.read(serial.in_waiting or 1)

        if chunk:
            buffer += chunk

            # Detect ELM327 prompt

            if b">" in buffer:
                break

        if time.time() - start_time > timeout:
            print(f"Timeout waiting response for {cmd}")
            break

        time.sleep(0.01)

    lines = (
        buffer.decode(errors="ignore")
        .replace("\r", "\n")
        .split("\n")
    )

    return [line.strip() for line in lines if line.strip()]

def get_all_supported_pids(serial) -> List[str]:
    """
    Query all supported Mode 01 PIDs from the vehicle.
    """

    global supported_pids

    supported_pids = []

    for start in [0x00, 0x20, 0x40, 0x60, 0x80, 0xA0, 0xC0, 0xE0]:
        cmd = f"01{start:02X}"

        answer = send_cmd(serial, cmd)

        pids = parse_supported_pids(answer)

        if not pids:
            break

        for pid in pids:
            full_pid = start + pid

            supported_pids.append(f"01{full_pid:02X}")

    print(f"Detected {len(supported_pids)} supported PIDs")

    return supported_pids

def read_thread(stop_thread, serial, state) -> None:
    """
    Continuous reading loop.

    This function is intended to run in a dedicated thread.
    """

    global supported_pids

    while not stop_thread.is_set():
        with state.lock:
            state.features["speed"] = read_pid(serial, "010D") if "010D" in supported_pids else 0.0

            state.features["rpm"] = read_pid(serial, "010C") if "010C" in supported_pids else 0.0

            state.features["pos_pedal"] = read_pid(serial, "0111") if "0111" in supported_pids else 0.0

        time.sleep(1)

def parse_supported_pids(lines: List[str]) -> List[int]:
    """
    Parse supported PID bitmap response.

    Example response:
        41 00 BE 3F A8 13

    Returns list of supported PID offsets.
    """

    try:
        data_line = next((line for line in lines if line.startswith("41")), None)

        if not data_line:
            return []

        data = "".join(data_line.split())[4:]

        bits = bin(int(data, 16))[2:].zfill(32)

        supported: List[int] = []

        for i, bit in enumerate(bits):
            if bit == "1":
                supported.append(i + 1)

        return supported

    except Exception as e:
        print(f"Error parsing supported PIDs: {e}")
        return []

def process_data(cmd: str, data: List[str]) -> Optional[float]:
    """
    Decode PID response using registered decoder functions.
    """

    try:
        data_line = next(
            (line for line in data if line.startswith(f"41{cmd[2:]}")),
            None,
        )

        if not data_line:
            return None

        hex_value = data_line[4:]

        decoder = decoders.pid_decoders.get(cmd, lambda x: 0.0)

        return decoder(hex_value)

    except Exception as e:
        print(f"Error processing PID {cmd}: {e}")
        return None

def read_pid(serial, cmd: str) -> float:
    """
    Read a single PID value from the vehicle.
    """

    global supported_pids

    if cmd not in supported_pids:
        return 0.0

    answer = send_cmd(serial, cmd)

    value = process_data(cmd, answer)

    if value is None:
        return 0.0

    return float(value)
