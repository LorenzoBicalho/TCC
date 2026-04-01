from dotenv import load_dotenv
import os

load_dotenv()

SERIAL_PORT = os.getenv("SERIAL_PORT", "/dev/ttyUSB0") # e.g., 'COM1' for Windows or '/dev/ttyUSB0' for Linux
MQTT_BROKER = os.getenv("MQTT_BROKER", "test.mosquitto.org")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "can/messages")
