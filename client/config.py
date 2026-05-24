from dotenv import load_dotenv
import os
import numpy as np
from utils.hardware import get_cpu_serial


load_dotenv()


SERIAL_PORT = os.getenv("SERIAL_PORT", "/dev/ttyUSB0")
SERVER_URL = os.getenv("SERVER_URL", "http://76.13.171.142:8000")
SERIAL_NUMBER = get_cpu_serial()
NUM_RULES = 5
NUM_CLUSTERS = 3
FEATURE_ORDER = [
    "speed",
    "acc_long",
    "engine_speed",
    "throttle_position",
    "acc_lat",

]
MIN_VALUES = np.array([
    0.0,     # speed
    -5.0,    # acc_long
    0.0,     # rpm
    0.0,     # throttle
    -5.0,    # acc_lat
], dtype=float)
MAX_VALUES = np.array([
    120.0,
    5.0,
    10000.0,
    100.0,
    5.0,
], dtype=float)
