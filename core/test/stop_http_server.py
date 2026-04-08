#!/usr/bin/env python3
"""Terminate the http.server whose PID was written by start_http_server.py."""
import os
import signal
import sys

pid_file = sys.argv[1]
try:
    with open(pid_file) as f:
        pid = int(f.read().strip())
    os.kill(pid, signal.SIGTERM)
except (FileNotFoundError, ProcessLookupError, ValueError):
    pass
