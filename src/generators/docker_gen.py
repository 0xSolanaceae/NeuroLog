import os
import random
from faker import Faker

fake = Faker()

normal_messages = [
    "Container started",
    "Health check passed",
    "Listening on port 80",
    "Connected to database",
    "Request processed successfully",
    "Image pulled successfully",
    "Container stopped",
    "Backup completed successfully",
    "Configuration updated",
    "User logged in",
    "Cache cleared",
    "Service restarted",
    "Log rotation completed",
    "SSL certificate renewed",
    "Scheduled task executed",
    "Disk space checked",
    "Memory usage within limits",
    "CPU usage within limits",
    "Network latency within acceptable range",
    "Database query executed",
]

suspicious_messages = [
    "Failed login attempt from 192.168.1.1",
    "Unauthorized access detected",
    "Port scan detected from 10.0.0.1",
    "Unexpected container termination",
]

def random_timestamp():
    return fake.date_time_this_month().strftime("%Y-%m-%dT%H:%M:%SZ")

def generate_log_line():
    timestamp = random_timestamp()
    if random.random() < 0.02: # 2% chance of generating a suspicious log
        message = random.choice(suspicious_messages)
        level = "WARNING"
    else:
        message = random.choice(normal_messages)
        level = "INFO"
    code = f"c{random.randint(1000, 9999)}"
    return f"{timestamp} {level} [{code}] {message}"

def main(entries):
    logs = [generate_log_line() for _ in range(entries)]
    log_dir = "src/logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "docker.log")
    with open(log_path, "w") as file:
        for log in logs:
            file.write(log + "\n")
    print(f"Generated {entries} lines of Docker logs in '{log_path}'")

if __name__ == "__main__":
    main()