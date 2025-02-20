import random
from faker import Faker

fake = Faker()

HOSTNAME = "myhostname"

tags_messages = {
    "kernel": [
        lambda: f"usb 1-1: New high-speed USB device number {random.randint(1,20)} using xhci_hcd",
        lambda: "usb 1-1: New USB device found, idVendor=1234, idProduct=5678, bcdDevice=1.00",
        lambda: f"usb 1-1: New USB device strings: Mfr={random.randint(1,5)}, Product={random.randint(1,5)}, SerialNumber={random.randint(100000,999999)}",
        lambda: "usb 1-1: Product: Example USB Device",
        lambda: "usb 1-1: Manufacturer: Example Manufacturer",
        lambda: "eth0: Link is Up - 1Gbps/Full - flow control rx/tx",
        lambda: "EXT4-fs (sda1): mounted filesystem with ordered data mode. Opts: (null)",
    ],
    "systemd": [
        lambda: "Starting Daily apt download activities...",
        lambda: "Started Daily apt download activities.",
        lambda: "Starting Cleanup of Temporary Directories...",
        lambda: "Started Cleanup of Temporary Directories.",
    ],
    "CRON": [
        lambda: "(root) CMD (   cd / && run-parts --report /etc/cron.hourly)",
    ],
    "sshd": [
        lambda: "Accepted password for user from 192.168.1.100 port 22 ssh2",
        lambda: "pam_unix(sshd:session): session opened for user user by (uid=0)",
    ],
    "systemd-logind": [
        lambda: "New session 1 of user user.",
    ],
}

def random_syslog_timestamp():
    return fake.date_time_this_year().strftime("%b %d %H:%M:%S")

def generate_syslog_line():
    timestamp = random_syslog_timestamp()
    tag = random.choice(list(tags_messages.keys()))
    message = random.choice(tags_messages[tag])()
    return f"{timestamp} {HOSTNAME} {tag}: {message}"

def main(entries):
    logs = [generate_syslog_line() for _ in range(entries)]
    log_path = "logs/syslog.log"
    with open(log_path, "w") as file:
        for log in logs:
            file.write(log + "\n")
    print(f"Generated {entries} lines of Syslog logs in '{log_path}'")

if __name__ == "__main__":
    main()