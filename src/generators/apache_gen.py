import random
import datetime

HTTP_METHODS = ["GET", "POST", "PUT", "DELETE"]
URL_PATHS = ["/index.html", "/about.html", "/services.html", "/contact.html", "/api/data"]
STATUS_CODES = [200, 201, 301, 302, 400, 403, 404, 500]
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (Linux; Android 11; SM-G998B)"
]

def random_ip():
    return ".".join(str(random.randint(0, 255)) for _ in range(4))

def random_apache_timestamp():
    now = datetime.datetime.now()
    delta = datetime.timedelta(seconds=random.randint(0, 7*24*3600))
    ts = now - delta
    return ts.strftime("%d/%b/%Y:%H:%M:%S +0000")

def generate_apache_log_line():
    ip = random_ip()
    ident = "-"  # usually not used
    user = random.choice(["frank", "jane", "bob", "-"])
    timestamp = random_apache_timestamp()
    method = random.choice(HTTP_METHODS)
    path = random.choice(URL_PATHS)
    protocol = "HTTP/1.0"
    request = f"{method} {path} {protocol}"
    status = random.choice(STATUS_CODES)
    bytes_sent = random.randint(200, 5000)
    return f'{ip} {ident} {user} [{timestamp}] "{request}" {status} {bytes_sent}'

def main(entries):
    lines = [generate_apache_log_line() for _ in range(entries)]
    with open("logs/apache.log", "w") as file:
        for log in lines:
            file.write(log + "\n")
    print(f"Generated {entries} lines of Apache logs in 'logs/apache.log'")
    return "\n".join(lines)

if __name__ == "__main__":
    main(100)