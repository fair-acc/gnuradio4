#!/usr/bin/env python3
"""HTTP server with COOP/COEP headers for SharedArrayBuffer (required for WASM pthreads)."""

import http.server
import os
import socketserver
import sys


class CoopCoepHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    directory = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
    os.chdir(directory)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), CoopCoepHandler) as httpd:
        print(f"Serving {directory} on http://localhost:{port} (COOP/COEP enabled)")
        httpd.serve_forever()
