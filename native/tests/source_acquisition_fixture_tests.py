#!/usr/bin/env python3
"""Local HTTPS tests for native source resume, cancellation, and HF auth."""

import argparse
import hashlib
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_payload(path: pathlib.Path, pattern: bytes, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        while size:
            chunk = pattern[: min(len(pattern), size)]
            output.write(chunk)
            size -= len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--acquirer", required=True, type=pathlib.Path)
    parser.add_argument("--server", required=True, type=pathlib.Path)
    parser.add_argument("--work-root", required=True, type=pathlib.Path)
    arguments = parser.parse_args()

    work = arguments.work_root.resolve()
    shutil.rmtree(work, ignore_errors=True)
    server_root = work / "server"
    cache_root = work / "cache"
    spec_root = work / "specs"
    logs = work / "logs"
    logs.mkdir(parents=True)

    revision = "a" * 40
    first = server_root / "owner/model/resolve" / revision / "first.bin"
    second = server_root / "owner/model/resolve" / revision / "second.bin"
    write_payload(first, b"vrhino-source-first", 16 * 1024 * 1024 + 17)
    write_payload(second, b"vrhino-source-second", 1024 * 1024 + 19)
    first_hash = sha256(first)
    second_hash = sha256(second)
    plan = {
        "schema_version": 1,
        "model_reference": "vrhino/source-fixture:1",
        "requested_source": {
            "provider": "huggingface",
            "repository": "owner/model",
            "revision": revision,
        },
        "artifacts": [
            {
                "id": "first",
                "role": "checkpoint",
                "repository": "owner/model",
                "revision": revision,
                "upstream_path": "first.bin",
                "local_path": "first.bin",
                "size": first.stat().st_size,
                "sha256": first_hash,
            },
            {
                "id": "second",
                "role": "conditioning.weights",
                "repository": "owner/model",
                "revision": revision,
                "upstream_path": "second.bin",
                "local_path": "second.bin",
                "size": second.stat().st_size,
                "sha256": second_hash,
            },
        ],
    }
    (spec_root / "fixture").mkdir(parents=True)
    (spec_root / "fixture/source-plan.json").write_text(
        json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    certificate = work / "certificate.pem"
    key = work / "key.pem"
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
        "-subj", "/CN=localhost", "-addext",
        "subjectAltName=DNS:localhost,IP:127.0.0.1", "-keyout", str(key),
        "-out", str(certificate),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    port_file = work / "port"
    server_log = logs / "server.jsonl"
    process: subprocess.Popen[str] | None = None
    server = subprocess.Popen([
        sys.executable, str(arguments.server), "--root", str(server_root),
        "--cert", str(certificate), "--key", str(key), "--log", str(server_log),
        "--port-file", str(port_file),
    ])
    try:
        for _ in range(100):
            if port_file.exists():
                break
            if server.poll() is not None:
                raise RuntimeError("source fixture server exited during startup")
            time.sleep(0.05)
        port = int(port_file.read_text(encoding="utf-8"))
        base = f"https://localhost:{port}/slow"
        command = [
            str(arguments.acquirer), "vrhino/source-fixture:1",
            f"hf://owner/model@{revision}", str(spec_root),
            str(cache_root / "sources"), base, base,
            str(cache_root / "tmp"), str(certificate), "2",
        ]
        environment = os.environ.copy()
        fake_token = "hf_fixture_token_must_not_be_logged"
        environment["HF_TOKEN"] = fake_token
        process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, env=environment)
        partial = cache_root / "tmp/downloads" / f"{first_hash}.partial"
        for _ in range(200):
            if partial.exists() and partial.stat().st_size > 0:
                break
            if process.poll() is not None:
                raise RuntimeError("source acquisition exited before cancellation")
            time.sleep(0.025)
        else:
            raise RuntimeError("source acquisition did not create a partial")
        process.send_signal(signal.SIGINT)
        interrupted_output, _ = process.communicate(timeout=5)
        assert process.returncode == 130, interrupted_output
        assert interrupted_output.count(
            "Download interrupted. Partial download preserved for resume.") == 1
        assert fake_token not in interrupted_output
        resume_offset = partial.stat().st_size
        assert 0 < resume_offset < first.stat().st_size
        assert not (cache_root / "sources/blobs/sha256" /
                    first_hash[:2] / first_hash).exists()

        resumed = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, env=environment, check=False)
        assert resumed.returncode == 0, resumed.stdout
        assert fake_token not in resumed.stdout
        assert not partial.exists()
        assert (cache_root / "sources/blobs/sha256" /
                first_hash[:2] / first_hash).is_file()
        assert (cache_root / "sources/blobs/sha256" /
                second_hash[:2] / second_hash).is_file()

        records = [json.loads(line) for line in server_log.read_text(
            encoding="utf-8").splitlines() if line]
        assert any(record["path"].endswith("/first.bin") and
                   record["range"] == f"bytes={resume_offset}-" and
                   record["status"] == 206 for record in records)
        assert all(record["authorization"] for record in records
                   if record["path"].endswith(("/first.bin", "/second.bin")))
        first_resume_index = next(
            index for index, record in enumerate(records)
            if record["path"].endswith("/first.bin") and
            record["range"] == f"bytes={resume_offset}-")
        assert not any(record["path"].endswith("/second.bin")
                       for record in records[:first_resume_index])
        assert fake_token not in server_log.read_text(encoding="utf-8")

        print("source acquisition fixture tests: PASS")
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


if __name__ == "__main__":
    main()
