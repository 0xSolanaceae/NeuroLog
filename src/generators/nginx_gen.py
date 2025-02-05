import random
import urllib.parse
import time

normal_methods = ["GET", "POST", "PUT", "DELETE"]
normal_paths = ["/index.html", "/about.html", "/services.html", "/contact.html", "/api/data", "/dashboard"]
suspicious_messages = ["SQL Injection", "XSS Attempt", "Command Injection", "Path Traversal"]

def random_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))

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
    user_agent = random.choice([
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
        "Mozilla/5.0 (Linux; Android 11; SM-G998B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Mobile Safari/537.36"
    ])
    
    return f'{ip} - - [{timestamp}] "{request}" {status} {bytes_sent} "{referer}" "{user_agent}"'

def main(entries):
    logs = [generate_log_line() for _ in range(entries)]
    with open("logs/nginx.log", "w") as file:
        for log in logs:
            file.write(log + "\n")
    print(f"Generated {entries} lines of Nginx logs in 'logs/nginx.log'")

if __name__ == "__main__":
    main(1000)