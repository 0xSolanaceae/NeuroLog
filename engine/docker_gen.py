
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
    "Malicious payload detected",
    "Outbound connection to known malicious IP",
    "Container running in privileged mode",
    "Suspicious file modification detected",
    "Cryptocurrency mining activity detected",
    "Container escape attempt detected",
]

def random_timestamp():
    now = datetime.datetime.now()
    delta = datetime.timedelta(days=random.randint(0, 7), hours=random.randint(0, 23), minutes=random.randint(0, 59), seconds=random.randint(0, 59))
    return (now - delta).strftime("%Y-%m-%dT%H:%M:%SZ")

def generate_log_line():
    if random.random() < 0.02:  # 2% chance of generating a suspicious log
        message = random.choice(suspicious_messages)
    else:
        message = random.choice(normal_messages)
    
    log_level = random.choice(["INFO", "WARNING", "ERROR"]) if "Failed" in message or "Unauthorized" in message else "INFO"
    container_id = f"c{random.randint(1000, 9999)}"
    return f"{random_timestamp()} {log_level} [{container_id}] {message}"

def main(entries):
    logs = [generate_log_line() for _ in range(entries)]
    with open("data/docker.log", "w") as file:
        for log in logs:
            file.write(log + "\n")
    print(f"Generated {entries} lines of Docker logs in 'docker.log'")

if __name__ == "__main__":
    main()