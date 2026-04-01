from config import MQTT_BROKER, MQTT_PORT, MQTT_TOPIC

def publish(client, message):
    client.publish(MQTT_TOPIC, message)

def on_connect(client, userdata, flags, rc):
    print(f"Connected sucessfully with code {rc}")
def on_connect_fail(client, userdata, rc):
    print(f"Connection failed with code {rc}")
def on_disconnect(client, userdata, rc):
    print(f"Disconnected from broker with code {rc}")

def init_client(client):
    client.on_connect = on_connect
    client.on_connect_fail = on_connect_fail
    client.on_disconnect = on_disconnect
    client.connect(MQTT_BROKER, MQTT_PORT)
    client.loop_start()

def stop_client(client):
    client.loop_stop()
    client.disconnect()