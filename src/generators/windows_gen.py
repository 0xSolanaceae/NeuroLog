import random
import datetime

SOURCES = ["Application", "System", "Security"]
LEVELS = ["Information", "Warning", "Error"]
MESSAGES = [
    "The operation completed successfully.",
    "An unexpected error occurred during startup.",
    "User logon was successful.",
    "The configuration file was updated.",
    "Service failed to start due to a timeout."
]

def random_windows_timestamp():
    now = datetime.datetime.now()
    delta = datetime.timedelta(seconds=random.randint(0, 7*24*3600))
    ts = now - delta
    return ts.strftime("%m/%d/%Y %I:%M:%S %p")

def generate_windows_log_line():
    timestamp = random_windows_timestamp()
    source = random.choice(SOURCES)
    event_id = random.randint(1000, 9999)
    level = random.choice(LEVELS)
    message = random.choice(MESSAGES)
    return f"{timestamp}  {source}  EventID:{event_id}  {level}  {message}"

def main(entries):
    lines = [generate_windows_log_line() for _ in range(entries)]
    with open("logs/windows.log", "w") as file:
        for log in lines:
            file.write(log + "\n")
    print(f"Generated {entries} lines of Windows logs in 'logs/windows.log'")
    return "\n".join(lines)

if __name__ == "__main__":
    main(100)