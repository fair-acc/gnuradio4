#!/usr/bin/env python3
"""Start a Python http.server in the background and write its PID to a file."""
import subprocess
import sys
import time

directory = sys.argv[1]
port      = sys.argv[2]
pid_file  = sys.argv[3]

proc = subprocess.Popen(
    [sys.executable, '-m', 'http.server', port, '--bind', '127.0.0.1', '--directory', directory],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)
with open(pid_file, 'w') as f:
    f.write(str(proc.pid))

time.sleep(0.4)  # give the server time to bind
sys.exit(0)
