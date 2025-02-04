import sys
import engine.docker_gen
import engine.kubernetes_gen
import engine.nginx_gen
import engine.syslog_gen

if len(sys.argv) > 1:
    try:
        entries = int(sys.argv[1])
    except ValueError:
        sys.exit("Please provide a valid integer for entries.")
else:
    entries = int(input("How many entries of logs would you like? "))

engine.docker_gen.main(entries)
engine.kubernetes_gen.main(entries)
engine.syslog_gen.main(entries)
engine.nginx_gen.main(entries)