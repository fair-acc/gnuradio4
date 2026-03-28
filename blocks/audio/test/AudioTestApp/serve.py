#!/usr/bin/env python3
import argparse
import io
import math
import socketserver
import wave
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


def make_tone_wav_bytes() -> bytes:
    sample_rate = 48_000
    duration_seconds = 1.0
    frequency_hz = 440.0
    amplitude = 0.25
    sample_count = int(sample_rate * duration_seconds)

    pcm = bytearray()
    for index in range(sample_count):
        value = int(
            32767
            * amplitude
            * math.sin(2.0 * math.pi * frequency_hz * index / sample_rate)
        )
        pcm += int(value).to_bytes(2, byteorder="little", signed=True)

    with io.BytesIO() as buffer:
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(bytes(pcm))
        return buffer.getvalue()


class AudioHTTPRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        path, _, query = self.path.partition("?")

        if path == "/tone.wav":
            body = make_tone_wav_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if path == "/proxy":
            # Proxy remote WAV URLs through localhost so browser playback tests are not blocked by CORS.
            params = parse_qs(query, keep_blank_values=False)
            target = params.get("url", [""])[0]
            if not target:
                self.send_error(400, "missing url query parameter")
                return

            parsed = urlparse(target)
            if parsed.scheme not in ("http", "https"):
                self.send_error(400, "only http and https URLs are supported")
                return

            try:
                request = Request(target, headers={"User-Agent": "AudioTestApp/1.0"})
                with urlopen(request, timeout=20) as upstream:
                    body = upstream.read()
                    content_type = upstream.headers.get(
                        "Content-Type", "application/octet-stream"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
            except HTTPError as ex:
                self.send_error(ex.code, f"upstream HTTP error: {ex.reason}")
            except URLError as ex:
                self.send_error(502, f"upstream fetch failed: {ex.reason}")
            except Exception as ex:
                self.send_error(502, f"proxy error: {ex}")
            return

        super().do_GET()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="Directory to serve")
    args = parser.parse_args()

    serve_dir = Path(args.dir)
    if not serve_dir.is_dir():
        parser.error(f"--dir does not exist or is not a directory: {args.dir}")

    print(f"Serving {serve_dir} on http://localhost:8080")
    print("Generated WAV endpoint: http://localhost:8080/tone.wav")
    socketserver.TCPServer.allow_reuse_address = True
    handler = partial(AudioHTTPRequestHandler, directory=str(serve_dir))
    HTTPServer(("localhost", 8080), handler).serve_forever()
