#!/usr/bin/env python3
"""Local HTTPS proof for Hugging Face official-to-mirror source fallback."""

import argparse
import hashlib
import json
import os
import pathlib
import select
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time


REVISION = "b" * 40
REPOSITORY = "owner/model"
TOKEN = "hf_endpoint_fixture_secret_must_never_be_logged"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def records(path: pathlib.Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line]


class ConnectProxyHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        request = self.rfile.readline().decode("ascii", errors="replace").strip()
        while self.rfile.readline() not in (b"\r\n", b"\n", b""):
            pass
        if not request.startswith("CONNECT "):
            self.wfile.write(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
            return
        target = request.split()[1]
        host = target.rsplit(":", 1)[0]
        with self.server.log_lock:
            self.server.targets.append(target)
        if host in self.server.failure_hosts:
            self.wfile.write(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n")
            return
        upstream_port = self.server.routes[host]
        upstream = socket.create_connection(("127.0.0.1", upstream_port))
        try:
            self.wfile.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            self.wfile.flush()
            sockets = [self.connection, upstream]
            while True:
                readable, _, _ = select.select(sockets, [], [], 5)
                if not readable:
                    continue
                for source in readable:
                    data = source.recv(64 * 1024)
                    if not data:
                        return
                    (upstream if source is self.connection else self.connection).sendall(data)
        finally:
            upstream.close()


class ThreadingProxy(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquirer", required=True, type=pathlib.Path)
    parser.add_argument("--server", required=True, type=pathlib.Path)
    parser.add_argument("--cli", type=pathlib.Path)
    parser.add_argument("--work-root", required=True, type=pathlib.Path)
    args = parser.parse_args()

    work = args.work_root.resolve()
    shutil.rmtree(work, ignore_errors=True)
    official_root = work / "official-root"
    mirror_root = work / "mirror-root"
    specs = work / "specs"
    logs = work / "logs"
    logs.mkdir(parents=True)

    certificate = work / "certificate.pem"
    key = work / "key.pem"
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-days", "1", "-subj", "/CN=localhost", "-addext",
        "subjectAltName=DNS:localhost,IP:127.0.0.1,DNS:official.test,"
        "DNS:huggingface.co,DNS:hf-mirror.com",
        "-keyout", str(key), "-out", str(certificate),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def make_case(name: str, size: int = 256 * 1024,
                  mirror_data: bytes | None = None) -> tuple[str, bytes, pathlib.Path]:
        pattern = (f"vrhino-{name}-".encode("ascii") * 4096)
        data = (pattern * ((size + len(pattern) - 1) // len(pattern)))[:size]
        upstream = f"{name}.bin"
        relative = pathlib.Path(REPOSITORY) / "resolve" / REVISION / upstream
        official_file = official_root / relative
        mirror_file = mirror_root / relative
        official_file.parent.mkdir(parents=True, exist_ok=True)
        mirror_file.parent.mkdir(parents=True, exist_ok=True)
        official_file.write_bytes(data)
        mirror_file.write_bytes(data if mirror_data is None else mirror_data)
        model = f"vrhino/{name}:1"
        document = {
            "schema_version": 1,
            "model_reference": model,
            "requested_source": {
                "provider": "huggingface", "repository": REPOSITORY,
                "revision": REVISION,
            },
            "artifacts": [{
                "id": name, "role": "checkpoint", "repository": REPOSITORY,
                "revision": REVISION, "upstream_path": upstream,
                "local_path": upstream, "size": len(data), "sha256": digest(data),
            }],
        }
        plan = specs / name / "source-plan.json"
        plan.parent.mkdir(parents=True, exist_ok=True)
        plan.write_text(json.dumps(document) + "\n", encoding="utf-8")
        return model, data, plan

    cases: dict[str, tuple[str, bytes, pathlib.Path]] = {}
    for name in ("official", "connect", "timeout", "server503", "notfound",
                 "unauthorized", "forbidden", "mirror-ok", "partial",
                 "range-mismatch", "no-range", "redirect", "proxy", "ratelimit",
                 "sticky", "dns"):
        cases[name] = make_case(name)
    sticky_model, sticky_data, sticky_plan_path = cases["sticky"]
    sticky_second = b"sticky-second-artifact"
    sticky_second_path = (pathlib.Path(REPOSITORY) / "resolve" / REVISION /
                          "sticky-second.bin")
    (official_root / sticky_second_path).write_bytes(sticky_second)
    (mirror_root / sticky_second_path).write_bytes(sticky_second)
    sticky_plan = json.loads(sticky_plan_path.read_text(encoding="utf-8"))
    sticky_plan["artifacts"].append({
        "id": "sticky-second", "role": "conditioning.weights",
        "repository": REPOSITORY, "revision": REVISION,
        "upstream_path": "sticky-second.bin", "local_path": "sticky-second.bin",
        "size": len(sticky_second), "sha256": digest(sticky_second),
    })
    sticky_plan_path.write_text(json.dumps(sticky_plan) + "\n", encoding="utf-8")
    wrong_size = 256 * 1024
    cases["wrong-content"] = make_case(
        "wrong-content", wrong_size, b"x" * wrong_size)
    cases["cancel"] = make_case("cancel", 16 * 1024 * 1024)

    processes: list[subprocess.Popen[bytes]] = []

    def launch(root: pathlib.Path, prefix: str) -> tuple[int, pathlib.Path]:
        port_file = work / f"{prefix}-port"
        log = logs / f"{prefix}.jsonl"
        process = subprocess.Popen([
            sys.executable, str(args.server), "--root", str(root),
            "--cert", str(certificate), "--key", str(key), "--log", str(log),
            "--port-file", str(port_file),
            "--expected-authorization-sha256",
            digest(f"Bearer {TOKEN}".encode("utf-8")),
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(process)
        for _ in range(100):
            if port_file.exists():
                return int(port_file.read_text(encoding="utf-8")), log
            if process.poll() is not None:
                raise RuntimeError(f"{prefix} fixture exited during startup")
            time.sleep(0.05)
        raise RuntimeError(f"{prefix} fixture did not start")

    official_port, official_log = launch(official_root, "official")
    mirror_port, mirror_log = launch(mirror_root, "mirror")
    official_normal = f"https://localhost:{official_port}"
    mirror_normal = f"https://localhost:{mirror_port}"
    dead = "https://127.0.0.1:9"

    def command(name: str, official: str, mirror: str,
                cache_name: str | None = None) -> tuple[list[str], pathlib.Path]:
        cache = work / "caches" / (cache_name or name)
        model = cases[name][0]
        return ([
            str(args.acquirer), model, f"hf://{REPOSITORY}@{REVISION}",
            str(specs), str(cache / "sources"), official, mirror,
            str(cache / "tmp"), str(certificate), "1",
        ], cache)

    def run_case(name: str, official: str, mirror: str = mirror_normal,
                 expected: int = 0, environment: dict[str, str] | None = None,
                 cache_name: str | None = None) -> tuple[subprocess.CompletedProcess[str], pathlib.Path]:
        cmd, cache = command(name, official, mirror, cache_name)
        env = os.environ.copy() if environment is None else environment
        env["HF_TOKEN"] = TOKEN
        result = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=env, check=False)
        assert result.returncode == expected, result.stdout
        assert TOKEN not in result.stdout
        return result, cache

    def blob(cache: pathlib.Path, data: bytes) -> pathlib.Path:
        sha = digest(data)
        return cache / "sources/blobs/sha256" / sha[:2] / sha

    proxy: ThreadingProxy | None = None
    proxy_thread: threading.Thread | None = None
    try:
        # Official success is primary; the mirror receives no request. The same
        # request proves that HF_TOKEN reaches the official origin.
        mirror_before = len(records(mirror_log))
        result, cache = run_case("official", official_normal)
        assert "transport_endpoint=official" in result.stdout
        assert blob(cache, cases["official"][1]).is_file()
        assert len(records(mirror_log)) == mirror_before
        official_requests = [r for r in records(official_log)
                             if str(r["path"]).endswith("/official.bin")]
        assert official_requests and all(r["authorization"] for r in official_requests)
        assert all(r["authorization_matches"] for r in official_requests)

        # Connect-equivalent failure falls back immediately and anonymously.
        mirror_before = len(records(mirror_log))
        result, cache = run_case("connect", dead)
        assert "transport_endpoint=mirror" in result.stdout
        assert blob(cache, cases["connect"][1]).is_file()
        connect_mirror = records(mirror_log)[mirror_before:]
        assert connect_mirror and not any(r["authorization"] or r["cookie"]
                                          for r in connect_mirror)
        assert not any(r["authorization_matches"] for r in connect_mirror)

        dns_environment = os.environ.copy()
        dns_environment["NO_PROXY"] = "vrhino-source.invalid"
        dns_environment["no_proxy"] = "vrhino-source.invalid"
        result, _ = run_case(
            "dns", "https://vrhino-source.invalid",
            environment=dns_environment)
        assert "transport_endpoint=mirror" in result.stdout

        # A stalled official response switches within the one-second test cap.
        started = time.monotonic()
        result, _ = run_case("timeout", f"{official_normal}/timeout")
        elapsed = time.monotonic() - started
        assert "transport_endpoint=mirror" in result.stdout and elapsed < 2.75, elapsed

        # Selected availability 5xx falls back; semantic 404/401/403 do not.
        result, _ = run_case("server503", f"{official_normal}/status-503")
        assert "transport_endpoint=mirror" in result.stdout
        result, _ = run_case("ratelimit", f"{official_normal}/status-429")
        assert "transport_endpoint=mirror" in result.stdout

        mirror_before = len(records(mirror_log))
        result, _ = run_case(
            "mirror-ok", dead, f"{mirror_normal}/status-503-once",
            cache_name="mirror-retry")
        assert "transport_endpoint=mirror" in result.stdout
        mirror_retry = [r for r in records(mirror_log)[mirror_before:]
                        if str(r["path"]).endswith("/mirror-ok.bin")]
        assert [r["status"] for r in mirror_retry] == [503, 200]

        # The first qualifying failure selects the mirror for all remaining
        # artifacts in this logical acquisition; official is not re-tried.
        official_before = len(records(official_log))
        mirror_before = len(records(mirror_log))
        result, sticky_cache = run_case("sticky", f"{official_normal}/status-503")
        assert "transport_endpoint=mirror" in result.stdout
        sticky_official = records(official_log)[official_before:]
        sticky_mirror = records(mirror_log)[mirror_before:]
        assert len(sticky_official) == 1
        assert {str(r["path"]).rsplit("/", 1)[-1] for r in sticky_mirror} == {
            "sticky.bin", "sticky-second.bin"}
        assert blob(sticky_cache, sticky_data).is_file()
        assert blob(sticky_cache, sticky_second).is_file()
        for name, status in (("notfound", 404), ("unauthorized", 401),
                             ("forbidden", 403)):
            mirror_before = len(records(mirror_log))
            result, _ = run_case(name, f"{official_normal}/status-{status}", expected=1)
            assert f"HTTP {status}" in result.stdout
            assert len(records(mirror_log)) == mirror_before

        # Mirror bytes remain subject to exact size and SHA256.
        result, cache = run_case("mirror-ok", dead)
        assert "transport_endpoint=mirror" in result.stdout
        assert blob(cache, cases["mirror-ok"][1]).read_bytes() == cases["mirror-ok"][1]
        result, cache = run_case("wrong-content", dead, expected=1)
        assert "failed SHA256" in result.stdout
        assert not blob(cache, cases["wrong-content"][1]).exists()

        # Official partial bytes are resumed from the mirror at the exact offset.
        official_before = len(records(official_log))
        mirror_before = len(records(mirror_log))
        result, cache = run_case("partial", f"{official_normal}/truncate-once")
        assert "transport_endpoint=mirror" in result.stdout
        partial_official = [r for r in records(official_log)[official_before:]
                            if str(r["path"]).endswith("/partial.bin")]
        partial_mirror = [r for r in records(mirror_log)[mirror_before:]
                          if str(r["path"]).endswith("/partial.bin")]
        assert partial_official and partial_mirror
        offset = int(partial_official[-1]["bytes"])
        assert 0 < offset < len(cases["partial"][1])
        assert partial_mirror[0]["range"] == f"bytes={offset}-"
        assert partial_mirror[0]["status"] == 206
        assert blob(cache, cases["partial"][1]).read_bytes() == cases["partial"][1]

        # A mismatched 206 fails without appending; a 200 Range ignore restarts.
        mismatch_cmd, mismatch_cache = command(
            "range-mismatch", dead, f"{mirror_normal}/range-mismatch")
        mismatch_data = cases["range-mismatch"][1]
        mismatch_partial = mismatch_cache / "tmp/downloads" / f"{digest(mismatch_data)}.partial"
        mismatch_partial.parent.mkdir(parents=True, exist_ok=True)
        mismatch_offset = len(mismatch_data) // 3
        mismatch_partial.write_bytes(mismatch_data[:mismatch_offset])
        mismatch = subprocess.run(mismatch_cmd, text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  env={**os.environ, "HF_TOKEN": TOKEN}, check=False)
        assert mismatch.returncode == 1 and "mirror (resume protocol)" in mismatch.stdout, mismatch.stdout
        assert mismatch_partial.stat().st_size == mismatch_offset

        no_range_cmd, no_range_cache = command(
            "no-range", dead, f"{mirror_normal}/no-range")
        no_range_data = cases["no-range"][1]
        no_range_partial = no_range_cache / "tmp/downloads" / f"{digest(no_range_data)}.partial"
        no_range_partial.parent.mkdir(parents=True, exist_ok=True)
        no_range_offset = len(no_range_data) // 3
        no_range_partial.write_bytes(no_range_data[:no_range_offset])
        mirror_before = len(records(mirror_log))
        no_range = subprocess.run(no_range_cmd, text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  env={**os.environ, "HF_TOKEN": TOKEN}, check=False)
        assert no_range.returncode == 0, no_range.stdout
        no_range_records = [r for r in records(mirror_log)[mirror_before:]
                            if str(r["path"]).endswith("/no-range.bin")]
        assert any(r["range"] == f"bytes={no_range_offset}-" and r["status"] == 200
                   for r in no_range_records)
        assert any(r["range"] is None and r["status"] == 200 for r in no_range_records)
        assert blob(no_range_cache, no_range_data).read_bytes() == no_range_data

        # Cancellation on mirror retains the partial; rerun resumes it.
        cancel_cmd, cancel_cache = command("cancel", dead, f"{mirror_normal}/slow")
        cancel_process = subprocess.Popen(
            cancel_cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env={**os.environ, "HF_TOKEN": TOKEN})
        cancel_data = cases["cancel"][1]
        cancel_partial = cancel_cache / "tmp/downloads" / f"{digest(cancel_data)}.partial"
        for _ in range(200):
            if cancel_partial.exists() and cancel_partial.stat().st_size > 0:
                break
            if cancel_process.poll() is not None:
                raise RuntimeError("mirror cancellation transfer exited early")
            time.sleep(0.025)
        cancel_process.send_signal(signal.SIGINT)
        cancel_output, _ = cancel_process.communicate(timeout=5)
        assert cancel_process.returncode == 130, cancel_output
        cancel_offset = cancel_partial.stat().st_size
        assert 0 < cancel_offset < len(cancel_data) and TOKEN not in cancel_output
        mirror_before = len(records(mirror_log))
        resumed, _ = run_case("cancel", dead, f"{mirror_normal}/slow",
                              cache_name="cancel")
        assert resumed.returncode == 0
        resumed_records = records(mirror_log)[mirror_before:]
        assert any(r["range"] == f"bytes={cancel_offset}-" and r["status"] == 206
                   for r in resumed_records)

        # Origin-scoped bearer auth and disabled cookies survive a cross-host redirect.
        official_before = len(records(official_log))
        result, _ = run_case("redirect", f"{official_normal}/redirect-cross")
        assert "transport_endpoint=official" in result.stdout
        redirect_records = [r for r in records(official_log)[official_before:]
                            if str(r["path"]).endswith("/redirect.bin")]
        assert len(redirect_records) == 2
        first = next(r for r in redirect_records if r["status"] == 302)
        target = next(r for r in redirect_records if r["status"] == 200)
        assert first["authorization"]
        assert first["authorization_matches"]
        assert not target["authorization"] and not target["cookie"]

        # HTTPS_PROXY is honored for a non-loopback endpoint; the proxy tunnels
        # to the local TLS fixture without seeing or handling origin credentials.
        proxy = ThreadingProxy(("127.0.0.1", 0), ConnectProxyHandler)
        proxy.routes = {"official.test": official_port}
        proxy.failure_hosts = set()
        proxy.targets = []
        proxy.log_lock = threading.Lock()
        proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
        proxy_thread.start()
        proxy_env = os.environ.copy()
        proxy_url = f"http://127.0.0.1:{proxy.server_address[1]}"
        proxy_env.update({
            "HTTPS_PROXY": proxy_url, "https_proxy": proxy_url,
            "NO_PROXY": "", "no_proxy": "",
        })
        result, _ = run_case(
            "proxy", f"https://official.test:{official_port}",
            environment=proxy_env)
        assert "transport_endpoint=official" in result.stdout
        assert proxy.targets == [f"official.test:{official_port}"], proxy.targets

        # Run the normal product command through controlled CONNECT routing:
        # canonical official host is unavailable, canonical mirror host serves
        # the immutable test source, and acquisition reaches verified source CAS.
        if args.cli is not None:
            cli_data = b"test"
            cli_name = "cli-fallback"
            cli_relative = pathlib.Path(REPOSITORY) / "resolve" / REVISION / "test.bin"
            (mirror_root / cli_relative).parent.mkdir(parents=True, exist_ok=True)
            (mirror_root / cli_relative).write_bytes(cli_data)
            cli_spec = specs / cli_name
            cli_spec.mkdir(parents=True, exist_ok=True)
            cli_reference = "vrhino/cli-fallback:1.0.0"
            (cli_spec / "pull-plan.json").write_text(json.dumps({
                "schema_version": 1, "model_reference": cli_reference,
                "distribution": {
                    "kind": "source_backed",
                    "source": {"provider": "huggingface", "repository": REPOSITORY,
                               "revision": REVISION},
                    "source_plan": "source-plan.json", "converter": "native",
                },
            }) + "\n", encoding="utf-8")
            (cli_spec / "source-plan.json").write_text(json.dumps({
                "schema_version": 1, "model_reference": cli_reference,
                "requested_source": {"provider": "huggingface",
                                     "repository": REPOSITORY,
                                     "revision": REVISION},
                "artifacts": [{
                    "id": "fixture", "role": "checkpoint",
                    "repository": REPOSITORY, "revision": REVISION,
                    "upstream_path": "test.bin", "local_path": "test.bin",
                    "size": len(cli_data), "sha256": digest(cli_data),
                }],
            }) + "\n", encoding="utf-8")
            proxy.routes = {
                "official.test": official_port,
                "hf-mirror.com": mirror_port,
            }
            proxy.failure_hosts = {"huggingface.co"}
            proxy.targets.clear()
            cli_cache = work / "cli-cache"
            cli_env = os.environ.copy()
            cli_env.update({
                "HTTPS_PROXY": proxy_url, "https_proxy": proxy_url,
                "NO_PROXY": "", "no_proxy": "", "HF_TOKEN": TOKEN,
            })
            cli_result = subprocess.run([
                str(args.cli), "--cache-root", str(cli_cache),
                "--converter-spec-root", str(specs), "--ca-file", str(certificate),
                "pull", cli_reference,
            ], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                env=cli_env, check=False)
            (logs / "cli-fallback.log").write_text(cli_result.stdout, encoding="utf-8")
            assert cli_result.returncode == 1, cli_result.stdout
            assert "Acquiring source" in cli_result.stdout
            assert TOKEN not in cli_result.stdout
            cli_sha = digest(cli_data)
            cli_blob = cli_cache / "sources/blobs/sha256" / cli_sha[:2] / cli_sha
            assert cli_blob.is_file(), cli_result.stdout + repr(proxy.targets)
            assert cli_blob.read_bytes() == cli_data
            assert proxy.targets[:2] == ["huggingface.co:443", "hf-mirror.com:443"], proxy.targets

        all_logs = official_log.read_text(encoding="utf-8") + mirror_log.read_text(
            encoding="utf-8")
        assert TOKEN not in all_logs
        print("source endpoint fallback tests: PASS")
    finally:
        if proxy is not None:
            proxy.shutdown()
            proxy.server_close()
        if proxy_thread is not None:
            proxy_thread.join(timeout=5)
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
