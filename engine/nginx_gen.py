import random
import datetime
import urllib.parse

normal_methods = ["GET", "POST", "PUT", "DELETE"]
normal_paths = [
    "/index.html",
    "/about.html",
    "/contact.html",
    "/services.html",
    "/dashboard",
    "/api/data",
    "/static/logo.png",
]

suspicious_messages = [
    "SQL injection attempt detected",
    "Cross-site scripting attack detected",
    "Unauthorized access attempt",
    "Suspicious URL parameter detected",
    "Directory traversal attempt detected",
    "Potential DoS attack detected",
]


def random_timestamp():
    now = datetime.datetime.now()
    delta = datetime.timedelta(
        days=random.randint(0, 7),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
        seconds=random.randint(0, 59),
    )
    timestamp = now - delta
    return timestamp.strftime("%d/%b/%Y:%H:%M:%S +0000")


def random_ip():
    return f"192.168.1.{random.randint(1, 254)}"


def generate_log_line():
    ip = random_ip()
    timestamp = random_timestamp()
    method = random.choice(normal_methods)
    
    # 2% chance of generating a suspicious log
    if random.random() < 0.02:
        path = random.choice(normal_paths)
        msg = random.choice(suspicious_messages)
        query = urllib.parse.urlencode({"error": msg})
        full_path = f"{path}?{query}"
        status = random.choice([400, 403, 404])
        bytes_sent = random.randint(50, 500)
    else:
        full_path = random.choice(normal_paths)
        status = 200
        bytes_sent = random.randint(500, 5000)
    
    protocol = "HTTP/1.1"
    # Common log format: remotehost - - [timestamp] "method path protocol" status bytes
    return f'{ip} - - [{timestamp}] "{method} {full_path} {protocol}" {status} {bytes_sent}'


def main(entries):
    logs = [generate_log_line() for _ in range(entries)]
    with open("data/nginx.log", "w") as file:
        for log in logs:
            file.write(log + "\n")
    print(f"Generated {entries} lines of Nginx logs in 'data/nginx.log'")


if __name__ == "__main__":
    main()