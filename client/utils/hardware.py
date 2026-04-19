import socket
import time

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except Exception:
    GPIO_AVAILABLE = False


BUZZER_PIN = 18


def _setup_gpio():
    if not GPIO_AVAILABLE:
        return

    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(BUZZER_PIN, GPIO.OUT)


_setup_gpio()


def get_cpu_serial() -> str:
    serial = "UNKNOWN"

    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("Serial"):
                    value = line.split(":")[1].strip()
                    if value:
                        return value

        return serial

    except FileNotFoundError:
        return "NOT_RASPBERRY"

    except Exception:
        return "ERROR"


def _buzz(duration: float, repeat: int = 1, pause: float = 0.1):
    if not GPIO_AVAILABLE:
        print("buzz " * repeat)
        return

    try:
        for _ in range(repeat):
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            time.sleep(duration)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            time.sleep(pause)

    except Exception:
        pass


def aggressive_buzz():
    _buzz(duration=0.15, repeat=3, pause=0.1)


def require_internet_buzz():
    _buzz(duration=0.4, repeat=2, pause=0.2)


def check_internet_connection(timeout: float = 2.0) -> bool:
    try:
        socket.setdefaulttimeout(timeout)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("8.8.8.8", 53))

        return True

    except OSError:
        return False