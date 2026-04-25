from dotenv import load_dotenv
import os
from utils.hardware import get_cpu_serial

load_dotenv()

SERIAL_PORT = os.getenv("SERIAL_PORT", "/dev/ttyUSB0") # e.g., 'COM1' for Windows or '/dev/ttyUSB0' for Linux
SERVER_URL = os.getenv("SERVER_URL", "http://192.168.0.22:8000")
SERIAL_NUMBER = get_cpu_serial()
NUM_RULES = 5
NUM_CLUSTERS = 3
