"""Loopback-only fixtures for the real native Windows registry/cache tests."""
import argparse
import hashlib
import http.server
import json
import os
import subprocess
import threading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("executable")
    parser.add_argument("--cross-root", required=True)
    args = parser.parse_args()
    payloads = {"/runtime": b"abc", "/profile": b"{}\n", "/license": b"test license\n"}
    artifacts = [
        {"id": name, "role": role, "path": path, "size": len(payloads["/" + name]),
         "sha256": hashlib.sha256(payloads["/" + name]).hexdigest(), "required": True}
        for name, role, path in [("runtime", "runtime.vrm", "model/model.vrm"),
                                 ("profile", "execution.profile", "execution/default.json"),
                                 ("license", "legal.license", "legal/LICENSE.txt")]
    ]
    manifest = json.dumps({
        "schema_version": 1,
        "identity": {"namespace": "vrhino", "name": "fixture", "version": "v1", "architecture": "test_arch", "publisher": "VRhino"},
        "compatibility": {"runtime_contract": "cuda-v1", "vrm_schema": {"format_major": 0, "format_minor": 1, "metadata_schema": 1}},
        "artifacts": artifacts,
        "entrypoint": {"runtime_artifact": "runtime", "components": [], "default_preset": "default"},
        "defaults": {"default_preset": "default", "presets": {"default": {"profile_artifact": "profile", "inputs": {}}}},
        "hardware": {"presets": {"default": {"minimum_vram_bytes": None}}},
        "source": {"repository": "example/fixture", "revision": "0123456789abcdef", "converter_version": "test"},
        "license": {"identifier": "LicenseRef-Test", "artifact": "license", "upstream_notice": "example.invalid"},
    }).encode()

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            if self.path == "/v1/models/vrhino/fixture/v1/index.json":
                body = json.dumps({
                    "registry_schema_version": 1, "identity": "vrhino/fixture:v1",
                    "manifest": {"url": base + "/manifest", "size": len(manifest), "sha256": hashlib.sha256(manifest).hexdigest()},
                    "artifacts": [{"id": a["id"], "url": base + "/" + a["id"]} for a in artifacts],
                }).encode()
            elif self.path == "/manifest":
                body = manifest
            elif self.path in payloads:
                body = payloads[self.path]
            elif self.path == "/corrupt":
                body = b"bad"
            elif self.path in ("/payload", "/no-range", "/wrong-range") or self.path.endswith("/input.bin"):
                body = b"abc"
            else:
                self.send_error(404)
                return
            status = 200
            content_range = None
            if self.headers.get("Range") and self.path != "/no-range":
                offset = int(self.headers["Range"].split("=")[1].split("-")[0])
                content_range = f"bytes {offset}-{len(body)-1}/{len(body)}"
                if self.path == "/wrong-range":
                    content_range = "bytes 0-2/3"
                body = body[offset:]
                status = 206
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            if content_range:
                self.send_header("Content-Range", content_range)
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{server.server_port}"
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    environment = os.environ.copy()
    environment["VRHINO_TEST_HTTP_URL"] = base
    environment["VRHINO_TEST_CROSS_VOLUME_ROOT"] = args.cross_root
    try:
        result = subprocess.run([os.path.abspath(args.executable)], env=environment,
                                cwd=os.path.dirname(os.path.abspath(args.executable)),
                                timeout=180, creationflags=subprocess.CREATE_NO_WINDOW,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        print(result.stdout, end="", flush=True)
        return result.returncode
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


if __name__ == "__main__":
    raise SystemExit(main())
