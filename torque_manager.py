import subprocess
import argparse



host = subprocess.check_output(["hostname"])

with open('active_hosts.txt', 'a') as f:
    f.write(host)

