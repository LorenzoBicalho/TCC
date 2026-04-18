import requests

def rpm_decoder(bytes):
    A = int(bytes[:2], 16)
    B = int(bytes[2:], 16)
    return ((256 * A) + B) / 4

def speed_decoder(bytes):
    A = int(bytes, 16)
    return A  # km/h

def fuel_usage_decoder(bytes):
    A = int(bytes[:2], 16)
    B = int(bytes[2:], 16)
    return ((256 * A) + B) / 20

def air_flow_decoder(bytes):
    A = int(bytes[:2], 16)
    B = int(bytes[2:], 16)
    return ((256 * A) + B) / 100

def absolute_load_decoder(bytes):
    A = int(bytes[:2], 16)
    B = int(bytes[2:], 16)
    return ((256 * A) + B) * (100/255)

def eq_ratio_decoder(bytes):
    A = int(bytes[:2], 16)
    B = int(bytes[2:], 16)
    return ((256 * A) + B) * (2/65536)

def pedal_pos_decoder(bytes):
    A = int(bytes, 16)
    return A*(100/255)

def vin_decoder(answ):
    bytes= answ[2][8:] + answ[3][2:] + answ[4][2:]
    vin = bytes.fromhex(bytes).decode('ascii')
    return vin


pid_decoders = {
    '010C': rpm_decoder,                 # RPM
    '010D': speed_decoder,               # Speed
    '015E': fuel_usage_decoder,
    '0110': air_flow_decoder,
    '0143': absolute_load_decoder,
    '0144': eq_ratio_decoder,
    '0111': pedal_pos_decoder, # Other possible pids: 45, 47, 48, 49, 4A, 4B, 5A
    '0902': vin_decoder,
}