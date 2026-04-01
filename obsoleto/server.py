import socket
import struct
import pandas as pd
import numpy as np

# Configurações do protocolo
HOST = '127.0.0.1'  # localhost
PORT = 9000

MAGIC_HEADER = 0xABCD
HEADER_FORMAT = "<H B H"  # Magic (2 bytes), Opcode (1 byte), Quantidade (2 bytes)
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

def handle_client(conn, addr):
    print(f"[+] Conexão recebida de {addr}")
    
    try:
        while True:
            # Ler o cabeçalho
            header_bytes = conn.recv(HEADER_SIZE)
            if not header_bytes:
                break  # Conexão fechada
            
            if len(header_bytes) < HEADER_SIZE:
                print("[!] Header incompleto recebido.")
                break
            
            magic, opcode, qtd_pesos = struct.unpack(HEADER_FORMAT, header_bytes)
            
            if magic != MAGIC_HEADER:
                print(f"[!] Magic header inválido: {hex(magic)}")
                break
            
            print(f"[>] Opcode: {opcode}, Quantidade de pesos: {qtd_pesos}")
            
            # Receber os dados dos pesos (float32 * qtd_pesos)
            pesos_bytes = b''
            expected_bytes = qtd_pesos * 4  # float32 = 4 bytes
            while len(pesos_bytes) < expected_bytes:
                chunk = conn.recv(expected_bytes - len(pesos_bytes))
                if not chunk:
                    print("[!] Cliente desconectado antes de enviar todos os dados.")
                    break
                pesos_bytes += chunk
            
            if len(pesos_bytes) != expected_bytes:
                print("[!] Dados incompletos dos pesos.")
                break
            
            pesos = struct.unpack('<' + 'f' * qtd_pesos, pesos_bytes)
            print(f"[✓] {len(pesos)} pesos recebidos.")
            print(pesos[:5], "...")  # mostra os primeiros 5

            c = np.array(pesos[0:60]).reshape(6,10) 
            s = np.array(pesos[60:120]).reshape(6,10) 
            p = np.array(pesos[120:180]).reshape(6,10) 
            q = np.array(pesos[180:]).reshape(10,) 

            pd.DataFrame(c).to_csv('drivers/d1/c.csv', index=False, header=None)
            pd.DataFrame(s).to_csv('drivers/d1/s.csv', index=False, header=None)
            pd.DataFrame(p).to_csv('drivers/d1/p.csv', index=False, header=None)
            pd.DataFrame(q).to_csv('drivers/d1/q.csv', index=False, header=None)

            # (Opcional) Enviar ack
            ack_packet = struct.pack(HEADER_FORMAT, MAGIC_HEADER, 0x02, 0)
            conn.sendall(ack_packet)
            print("[←] Ack enviado")

    finally:
        conn.close()
        print(f"[-] Conexão encerrada com {addr}")

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"[🔌] Servidor escutando em {HOST}:{PORT}")
        
        while True:
            conn, addr = s.accept()
            handle_client(conn, addr)

if __name__ == "__main__":
    start_server()
