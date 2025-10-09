# /usr/bin/python3
# serve.py
from http.server import HTTPServer, SimpleHTTPRequestHandler


def make_test_string() -> str:
    return "".join(str(i) for i in range(100))


class CORSHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path == "/getNumbers":
            body = make_test_string()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))
        elif self.path == "/return404":
            self.send_response(404)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Not found")
        else:
            super().do_GET()

    def do_POST(self):
        if self.path == "/postNumbers":
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length)
            print(f"Received POST /postNumbers with {len(body)} bytes")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"OK")
        else:
            self.send_error(404, "Not Found")


if __name__ == "__main__":
    print("Serving on http://localhost:8080")
    HTTPServer(('localhost', 8080), CORSHTTPRequestHandler).serve_forever()
