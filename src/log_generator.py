import sys
import generators.docker_gen
import generators.kubernetes_gen
import generators.nginx_gen
import generators.syslog_gen

if len(sys.argv) > 1:
    try:
        entries = int(sys.argv[1])
    except ValueError:
        sys.exit("Please provide a valid integer for entries.")
else:
    entries = int(input("How many entries of logs would you like? "))

generators.docker_gen.main(entries)
generators.kubernetes_gen.main(entries)
generators.syslog_gen.main(entries)
generators.nginx_gen.main(entries)