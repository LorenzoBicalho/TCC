import socket
import struct
import random
import pandas as pd

# Configurações
HOST = '127.0.0.1'
PORT = 9000

MAGIC_HEADER = 0xABCD
OPCODE_SEND_PESOS = 0x01
QTD_PESOS = 190

HEADER_FORMAT = "<H B H"  # Magic (2 bytes), Opcode (1 byte), Quantidade (2 bytes)

def get_weights():
    c = pd.read_csv('parametros_locais/c.csv', header=None).values
    s = pd.read_csv('parametros_locais/s.csv', header=None).values
    p = pd.read_csv('parametros_locais/p.csv', header=None).values
    q = pd.read_csv('parametros_locais/q.csv', header=None).values.flatten()
    return c.flatten().tolist() + s.flatten().tolist() + p.flatten().tolist() + q.tolist()

def montar_pacote(pesos):
    header = struct.pack(HEADER_FORMAT, MAGIC_HEADER, OPCODE_SEND_PESOS, len(pesos))
    body = struct.pack('<' + 'f' * len(pesos), *pesos)
    return header + body

def cliente_envia_pesos():
    pesos = get_weights()
    pacote = montar_pacote(pesos)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((HOST, PORT))
        print(f"[↪] Conectado ao servidor {HOST}:{PORT}")
        s.sendall(pacote)
        print(f"[→] Pacote com {len(pesos)} pesos enviado")

        # Espera ack (opcional)
        ack = s.recv(5)  # 2 (magic) + 1 (opcode) + 2 (qtd)
        if ack and len(ack) == 5:
            magic, opcode, qtd = struct.unpack(HEADER_FORMAT, ack)
            if magic == MAGIC_HEADER and opcode == 0x02:
                print("[✓] Ack recebido do servidor")
            else:
                print(f"[!] Resposta inválida: magic={hex(magic)}, opcode={opcode}")
        else:
            print("[!] Nenhuma resposta ou resposta incompleta")

if __name__ == "__main__":
    cliente_envia_pesos()
