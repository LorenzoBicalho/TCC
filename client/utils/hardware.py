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

def check_internet_connection(timeout: float = 2.0) -> bool:
    try:
        socket.setdefaulttimeout(timeout)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(("8.8.8.8", 53))

        return True

    except OSError:
        return False


def _buzz(duration: float, repeat: int = 1, pause: float = 0.1, *, frequency_hz: float = 2000.0, duty_cycle: float = 50.0):
    if not GPIO_AVAILABLE:
        print("buzz " * repeat)
        return

    try:
        pwm = GPIO.PWM(BUZZER_PIN, frequency_hz)
        try:
            for _ in range(repeat):
                pwm.start(duty_cycle)
                time.sleep(duration)
                pwm.stop()
                time.sleep(pause)
        finally:
            try:
                pwm.stop()
            except Exception:
                pass
    except Exception as e:
        # Don't fail the app if hardware isn't accessible,
        # but do surface the cause for debugging.
        print(f"Buzz failed: {e!r}")


def aggressive_buzz():
    _buzz(duration=0.2, repeat=5, pause=0.1)


def require_internet_buzz(is_connected=False):
    for i in range(3):
        if not is_connected:
            _buzz(duration=0.5, repeat=3, pause=0.5)
            is_connected = check_internet_connection()
        else:
            break
    return is_connected