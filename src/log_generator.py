import sys
import argparse
import generators.docker_gen
import generators.kubernetes_gen
import generators.nginx_gen
import generators.syslog_gen
import generators.apache_gen
import generators.windows_gen

log_generators = {
    "docker": generators.docker_gen.main,
    "kubernetes": generators.kubernetes_gen.main,
    "syslog": generators.syslog_gen.main,
    "nginx": generators.nginx_gen.main,
    "apache": generators.apache_gen.main,
    "windows": generators.windows_gen.main
}

parser = argparse.ArgumentParser(
    description="Generate log entries for various log types.\n\n"
                "Usage example:\n"
                "  python log_generator.py 1000 docker syslog windows",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument("entries", type=int, help="Number of log entries to generate.")
parser.add_argument("log_types", type=str, nargs="+", 
                    help="Log types to generate (options: docker, kubernetes, syslog, nginx, apache, windows).")
args = parser.parse_args()

entries = args.entries
selected_logs = [log.strip().lower() for log in args.log_types]

if not selected_logs:
    sys.exit("No log types selected.")

for log_type in selected_logs:
    if generator := log_generators.get(log_type):
        print(f"Generating {entries} entries for {log_type} logs...")
        generator(entries)
    else:
        print(f"Unknown log type: {log_type}")