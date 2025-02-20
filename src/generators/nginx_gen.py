import random
import urllib.parse
import time
from faker import Faker

fake = Faker()

normal_methods = ["GET", "POST", "PUT", "DELETE"]
normal_paths = ["/index.html", "/about.html", "/services.html", "/contact.html", "/api/data", "/dashboard"]
suspicious_messages = ["SQL Injection", "XSS Attempt", "Command Injection", "Path Traversal"]

def random_ip():
    return fake.ipv4()

def generate_log_line():
    ip = random_ip()
    method = random.choice(normal_methods)
    protocol = "HTTP/1.1"
    timestamp = time.strftime("%d/%b/%Y:%H:%M:%S +0000", time.gmtime())
    
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
    
    request = f"{method} {full_path} {protocol}"
    referer = "-"
    user_agent = fake.user_agent()
    
    return f'{ip} - - [{timestamp}] "{request}" {status} {bytes_sent} "{referer}" "{user_agent}"'

def main(entries):
    logs = [generate_log_line() for _ in range(entries)]
    log_path = "logs/nginx.log"
    with open(log_path, "w") as file:
        for log in logs:
            file.write(log + "\n")
    print(f"Generated {entries} lines of Nginx logs in '{log_path}'")

if __name__ == "__main__":
    main()