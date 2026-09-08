#!/usr/bin/env python3
"""Reference-only HTTPS registry fixture used by product-layer tests.

This process is never invoked by VRhino Native.  It deliberately implements
Range, redirect, truncated-response, and no-Range paths so the native libcurl
client can be tested against real HTTP behavior.
"""

import argparse
import hashlib
import http.server
import json
import os
import pathlib
import ssl
import threading
import time
import urllib.parse


class RegistryFixtureHandler(http.server.BaseHTTPRequestHandler):
    server_version = "VRhinoRegistryFixture/1"

    def _record(self, status: int, sent: int, range_header: str | None) -> None:
        authorization = self.headers.get("Authorization")
        record = {
            "path": self.path,
            "status": status,
            "bytes": sent,
            "range": range_header,
            "authorization": authorization is not None,
            "authorization_matches": (
                authorization is not None
                and self.server.expected_authorization_sha256 is not None
                and hashlib.sha256(authorization.encode("utf-8")).hexdigest()
                == self.server.expected_authorization_sha256
            ),
            "cookie": self.headers.get("Cookie") is not None,
            "host": self.headers.get("Host"),
        }
        with self.server.log_lock:
            with open(self.server.log_path, "a", encoding="utf-8") as output:
                output.write(json.dumps(record, sort_keys=True) + "\n")

    def _resolve(self, url_path: str) -> pathlib.Path | None:
        decoded = urllib.parse.unquote(url_path).lstrip("/")
        candidate = (self.server.root / decoded).resolve()
        try:
            candidate.relative_to(self.server.root)
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        if path.startswith("/redirect-cross/"):
            destination = "/" + path[len("/redirect-cross/") :]
            self.send_response(302)
            self.send_header(
                "Location",
                f"https://127.0.0.1:{self.server.server_port}{destination}",
            )
            self.send_header("Set-Cookie", "vrhino_fixture=must-not-forward")
            self.send_header("Content-Length", "0")
            self.end_headers()
            self._record(302, 0, self.headers.get("Range"))
            return
        if path.startswith("/redirect/"):
            destination = "/" + path[len("/redirect/") :]
            self.send_response(302)
            self.send_header("Location", destination)
            self.send_header("Content-Length", "0")
            self.end_headers()
            self._record(302, 0, self.headers.get("Range"))
            return

        mode = "normal"
        for prefix, selected in (
            ("/no-range/", "no-range"),
            ("/truncate-once/", "truncate-once"),
            ("/wrong-length/", "wrong-length"),
            ("/slow/", "slow"),
            ("/timeout/", "timeout"),
            ("/range-mismatch/", "range-mismatch"),
        ):
            if path.startswith(prefix):
                mode = selected
                path = "/" + path[len(prefix) :]
                break

        if path.startswith("/status-"):
            if path.startswith("/status-503-once/"):
                with self.server.state_lock:
                    key = "status-503-once:" + path
                    first = key not in self.server.truncated
                    self.server.truncated.add(key)
                if first:
                    self.send_response(503)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    self._record(503, 0, self.headers.get("Range"))
                    return
                path = "/" + path[len("/status-503-once/") :]
            else:
                status_text = path[len("/status-") :].partition("/")[0]
                try:
                    status = int(status_text)
                except ValueError:
                    status = 500
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.end_headers()
                self._record(status, 0, self.headers.get("Range"))
                return

        source = self._resolve(path)
        if source is None:
            self.send_error(404)
            self._record(404, 0, self.headers.get("Range"))
            return

        total = source.stat().st_size
        if mode == "timeout":
            time.sleep(3)
        range_header = self.headers.get("Range")
        start = 0
        status = 200
        if range_header and mode != "no-range":
            if not range_header.startswith("bytes=") or not range_header.endswith("-"):
                self.send_error(416)
                self._record(416, 0, range_header)
                return
            try:
                start = int(range_header[6:-1])
            except ValueError:
                self.send_error(416)
                self._record(416, 0, range_header)
                return
            if start >= total:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{total}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                self._record(416, 0, range_header)
                return
            status = 206

        available = total - start
        truncate = False
        if mode == "truncate-once":
            with self.server.state_lock:
                key = str(source)
                if key not in self.server.truncated:
                    self.server.truncated.add(key)
                    truncate = True
        transmitted = max(1, available // 2) if truncate else available
        declared = available + 97 if mode == "wrong-length" else available

        self.send_response(status)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(declared))
        if status == 206:
            content_range_start = start + 1 if mode == "range-mismatch" else start
            self.send_header(
                "Content-Range",
                f"bytes {content_range_start}-{total - 1}/{total}",
            )
        self.end_headers()
        sent = 0
        with source.open("rb") as input_file:
            input_file.seek(start)
            remaining = transmitted
            while remaining:
                chunk_size = 64 * 1024 if mode == "slow" else 1024 * 1024
                data = input_file.read(min(chunk_size, remaining))
                if not data:
                    break
                try:
                    self.wfile.write(data)
                except OSError:
                    break
                sent += len(data)
                remaining -= len(data)
                if mode == "slow":
                    time.sleep(0.02)
        self._record(status, sent, range_header)
        if truncate or mode == "wrong-length":
            try:
                self.connection.shutdown(1)
            except OSError:
                pass

    def log_message(self, format_string: str, *args: object) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=pathlib.Path)
    parser.add_argument("--cert", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--port-file", required=True)
    parser.add_argument("--expected-authorization-sha256")
    arguments = parser.parse_args()

    server = http.server.ThreadingHTTPServer(("127.0.0.1", arguments.port),
                                             RegistryFixtureHandler)
    server.root = arguments.root.resolve()
    server.log_path = arguments.log
    server.log_lock = threading.Lock()
    server.state_lock = threading.Lock()
    server.truncated = set()
    server.expected_authorization_sha256 = arguments.expected_authorization_sha256
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(arguments.cert, arguments.key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    pathlib.Path(arguments.port_file).write_text(str(server.server_port), encoding="utf-8")
    server.serve_forever()


if __name__ == "__main__":
    main()
