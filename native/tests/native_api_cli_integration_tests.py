#!/usr/bin/env python3
"""CPU-only foreground serve, SIGINT, bind-failure, and port-release tests."""

import argparse
import http.client
import json
import pathlib
import signal
import socket
import subprocess
import tempfile
import time


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def get_json(port: int, path: str) -> tuple[int, dict, dict[str, str]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
    connection.request("GET", path, headers={"Connection": "close"})
    response = connection.getresponse()
    body = response.read()
    headers = dict(response.getheaders())
    connection.close()
    return response.status, json.loads(body), headers


def wait_for_server(process: subprocess.Popen[str], port: int) -> None:
    last_error: Exception | None = None
    for _ in range(200):
        require(process.poll() is None, "vrhino serve exited during startup")
        try:
            status, document, headers = get_json(port, "/api/v1/version")
            require(status == 200 and document["schema_version"] == 1,
                    "version endpoint startup probe failed")
            require(headers.get("Content-Type") == "application/json; charset=utf-8",
                    "version endpoint Content-Type drift")
            return
        except (ConnectionError, OSError, TimeoutError) as error:
            last_error = error
            time.sleep(0.01)
    raise RuntimeError(f"Native API did not become ready: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True, type=pathlib.Path)
    arguments = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="vrhino-api-cli-") as temporary:
        root = pathlib.Path(temporary)
        cache = root / "isolated-vrhino-home"
        port = available_port()
        command = [
            str(arguments.cli), "--cache-root", str(cache), "serve",
            "--host", "127.0.0.1", "--port", str(port),
        ]
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        try:
            wait_for_server(process, port)
            status, models, headers = get_json(port, "/api/v1/models")
            require(status == 200 and models == {"models": [], "schema_version": 1},
                    "empty isolated model endpoint mismatch")
            require("Access-Control-Allow-Origin" not in headers,
                    "CLI server enabled CORS")

            process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=5)
            require(process.returncode == 0,
                    f"SIGINT shutdown returned {process.returncode}: {stderr}")
            require("listening on http://127.0.0.1:" in stdout,
                    "foreground startup message missing")
            require(stderr == "", "loopback serve emitted unexpected stderr")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

        released_ok = False
        for _ in range(100):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as released:
                released.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    released.bind(("127.0.0.1", port))
                    released_ok = True
                    break
                except OSError:
                    time.sleep(0.01)
        require(released_ok, "SIGINT shutdown did not release the server port")

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
            occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            occupied.bind(("127.0.0.1", port))
            occupied.listen(1)
            failed = subprocess.run(
                command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=5, check=False
            )
        require(failed.returncode != 0, "occupied-port CLI serve returned success")
        require("could not bind" in failed.stderr and str(cache) not in failed.stderr,
                "occupied-port CLI failure was unsafe or unclear")

    print("Native API CLI foreground/SIGINT/bind tests: PASS")


if __name__ == "__main__":
    main()
