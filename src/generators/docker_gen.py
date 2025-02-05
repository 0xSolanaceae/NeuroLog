
import random
import datetime

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
    now = datetime.datetime.now()
    delta = datetime.timedelta(days=random.randint(0, 7), hours=random.randint(0, 23), minutes=random.randint(0, 59), seconds=random.randint(0, 59))
    return (now - delta).strftime("%Y-%m-%dT%H:%M:%SZ")

def generate_log_line():
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
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
    with open("logs/docker.log", "w") as file:
        for log in logs:
            file.write(log + "\n")
    print(f"Generated {entries} lines of Docker logs in 'logs/docker.log'")

if __name__ == "__main__":
    main()