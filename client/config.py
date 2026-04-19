from dotenv import load_dotenv
import os
from utils.hardware import get_cpu_serial

load_dotenv()

SERIAL_PORT = os.getenv("SERIAL_PORT", "/dev/ttyUSB0") # e.g., 'COM1' for Windows or '/dev/ttyUSB0' for Linux
MQTT_BROKER = os.getenv("MQTT_BROKER", "test.mosquitto.org")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "futurelab/can")
SERVER_URL = os.getenv("MQTT_TOPIC", "http://76.13.171.142:8000")
SERIAL_NUMBER = get_cpu_serial()
