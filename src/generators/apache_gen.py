import random
import datetime
from faker import Faker

fake = Faker()

HTTP_METHODS = ["GET", "POST", "PUT", "DELETE"]
URL_PATHS = ["/index.html", "/about.html", "/services.html", "/contact.html", "/api/data"]
STATUS_CODES = [200, 201, 301, 302, 400, 403, 404, 500]

def random_ip():
    return fake.ipv4()

def random_apache_timestamp():
    now = datetime.datetime.now()
    delta = datetime.timedelta(seconds=random.randint(0, 7*24*3600))
    ts = now - delta
    return ts.strftime("%d/%b/%Y:%H:%M:%S +0000")

def generate_apache_log_line():
    ip = random_ip()
    ident = "-"  # usually not used
    user = fake.user_name()
    timestamp = random_apache_timestamp()
    method = random.choice(HTTP_METHODS)
    path = random.choice(URL_PATHS)
    protocol = "HTTP/1.0"
    request = f"{method} {path} {protocol}"
    status = random.choice(STATUS_CODES)
    bytes_sent = random.randint(200, 5000)
    user_agent = fake.user_agent()
    return f'{ip} {ident} {user} [{timestamp}] "{request}" {status} {bytes_sent} "{user_agent}"'

def main(entries):
    logs = [generate_apache_log_line() for _ in range(entries)]
    log_path = "logs/apache.log"
    with open(log_path, "w") as file:
        for log in logs:
            file.write(log + "\n")
    print(f"Generated {entries} lines of Apache logs in '{log_path}'")

if __name__ == "__main__":
    main()