import random
from faker import Faker

fake = Faker()

normal_messages = [
    "Pod started",
    "Health check passed",
    "Listening on port 80",
    "Connected to database",
    "Request processed successfully",
    "Image pulled successfully",
    "Pod stopped",
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
    "Unexpected pod termination",
    "Malicious payload detected",
    "Outbound connection to known malicious IP",
    "Pod running in privileged mode",
    "Suspicious file modification detected",
    "Cryptocurrency mining activity detected",
    "Pod escape attempt detected",
]

namespaces = ["development", "production", "default", "kube-system"]

def random_timestamp():
    return fake.date_time_this_month().strftime("%Y-%m-%dT%H:%M:%SZ")

def generate_log_line():
    if random.random() < 0.02: # 2% chance of generating a suspicious log
        message = random.choice(suspicious_messages)
    else:
        message = random.choice(normal_messages)

    ns = random.choice(namespaces)
    pod_id = f"pod-{random.randint(1000, 9999)}"
    container_id = f"container-{random.randint(1000, 9999)}"
    hostname = fake.hostname()
    return f"{random_timestamp()} {hostname} INFO {ns}/{pod_id}/{container_id} {message}"

def main(entries):
    logs = [generate_log_line() for _ in range(entries)]
    log_path = "logs/kubernetes.log"
    with open(log_path, "w") as file:
        for log in logs:
            file.write(log + "\n")
    print(f"Generated {entries} lines of Kubernetes logs in '{log_path}'")

if __name__ == "__main__":
    main()