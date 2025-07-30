import socket
import struct
import pandas as pd
import numpy as np

# Configuration
HOST = '127.0.0.1'
PORT = 9000

MAGIC_HEADER = 0xABCD
OPCODE_SEND_WEIGHTS = 0x01
OPCODE_ACK = 0x02
OPCODE_ERROR = 0x03
OPCODE_REQUEST_WEIGHTS = 0x04
OPCODE_SEND_METRICS = 0x05
# NUM_WEIGHTS = 190

HEADER_FORMAT = "<H B H"  # Magic (2 bytes), Opcode (1 byte), Count (2 bytes)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

def load_weights():
    c = pd.read_csv('global_parameters/c.csv', header=None).values
    s = pd.read_csv('global_parameters/s.csv', header=None).values
    p = pd.read_csv('global_parameters/p.csv', header=None).values
    q = pd.read_csv('global_parameters/q.csv', header=None).values.flatten()
    return c.flatten().tolist() + s.flatten().tolist() + p.flatten().tolist() + q.tolist()

def load_metrics():
    metrics = pd.read_csv('local_parameters/metrics.csv') 
    return metrics

def build_packet(weights):
    header = struct.pack(HEADER_FORMAT, MAGIC_HEADER, OPCODE_SEND_WEIGHTS, len(weights))
    body = struct.pack('<' + 'f' * len(weights), *weights)
    return header + body

def build_metrics_packet(metrics):
    MAGIC_HEADER = 0xABCD
    OPCODE_SEND_METRICS = 0x05

    header = struct.pack("<H B H", MAGIC_HEADER, OPCODE_SEND_METRICS, len(metrics))

    body = b""
    for name, value in metrics.items():
        name_bytes = name.encode('ascii')
        name_len = len(name_bytes)
        body += struct.pack("<B", name_len)
        body += name_bytes
        body += struct.pack("<f", value)

    return header + body

def send_weights(sock):
    weights = load_weights()
    packet = build_packet(weights)    
    sock.sendall(packet)
    print(f"[→] Packet with {len(weights)} weights sent")

def send_metrics(sock):
    metrics = load_metrics()
    packet = build_metrics_packet(metrics)
    sock.sendall(packet)
    print("[→] Metrics packet sent")

def save_weights(sock):
    header_bytes = sock.recv(HEADER_SIZE)
    if not header_bytes or len(header_bytes) < HEADER_SIZE:
        print("[!] Invalid or incomplete header.")
        return

    magic, opcode, count = struct.unpack(HEADER_FORMAT, header_bytes)
    if magic != MAGIC_HEADER:
        print(f"[!] Invalid magic header: {hex(magic)}")
        return

    print(f"[>] Opcode: {opcode}, Number of weights: {count}")

    expected_bytes = count * 4
    weight_bytes = b''
    while len(weight_bytes) < expected_bytes:
        chunk = sock.recv(expected_bytes - len(weight_bytes))
        if not chunk:
            print("[!] Connection closed before full data received.")
            return
        weight_bytes += chunk

    if len(weight_bytes) != expected_bytes:
        print("[!] Incomplete weight data.")
        return

    weights = struct.unpack('<' + 'f' * count, weight_bytes)
    print(f"[✓] {len(weights)} weights received.")
    print(weights[:5], "...")

    # Save files
    c = np.array(weights[0:60]).reshape(6, 10) 
    s_arr = np.array(weights[60:120]).reshape(6, 10) 
    p = np.array(weights[120:180]).reshape(6, 10) 
    q = np.array(weights[180:]).reshape(10,) 

    pd.DataFrame(c).to_csv('drivers/d1/c.csv', index=False, header=None)
    pd.DataFrame(s_arr).to_csv('drivers/d1/s.csv', index=False, header=None)
    pd.DataFrame(p).to_csv('drivers/d1/p.csv', index=False, header=None)
    pd.DataFrame(q).to_csv('drivers/d1/q.csv', index=False, header=None)

    # Send ACK
    ack_packet = struct.pack(HEADER_FORMAT, MAGIC_HEADER, OPCODE_ACK, 0)
    sock.sendall(ack_packet)
    print("[←] Ack sent")

def listen_for_commands(sock):
    while True:
        header_bytes = sock.recv(HEADER_SIZE)
        if not header_bytes or len(header_bytes) < HEADER_SIZE:
            print("[X] Connection closed by server.")
            break

        magic, opcode, count = struct.unpack(HEADER_FORMAT, header_bytes)

        if magic != MAGIC_HEADER:
            print(f"[!] Invalid magic header: {hex(magic)}")
            continue

        if opcode == OPCODE_SEND_WEIGHTS:
            print("[⇋] Receiving weights from server")
            save_weights(sock)
        elif opcode == OPCODE_REQUEST_WEIGHTS:
            print("[⇈] Server requested weights")
            send_weights(sock)
            send_metrics(sock)
        elif opcode == OPCODE_ACK:
            print("[✓] Ack received from server")
        else:
            print(f"[!] Unknown opcode: {opcode}")

def connect_to_server():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((HOST, PORT))
            print(f"[↪] Connected to server at {HOST}:{PORT}")
            listen_for_commands(sock)
    except Exception as e:
        print(f"[!] Error connecting to server: {e}")

if __name__ == "__main__":
    connect_to_server()
