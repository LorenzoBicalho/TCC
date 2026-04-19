def get_cpu_serial():
    serial = "UNKNOWN"

    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("Serial"):
                    serial = line.split(":")[1].strip()
                    break
    except Exception as e:
        serial = "ERROR"

    return serial