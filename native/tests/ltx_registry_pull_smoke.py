#!/usr/bin/env python3
"""Controlled real-package registry pull for the 24.77 GB LTX package.

Eight artifacts are pre-admitted from the Phase 24B CAS.  The real 5.72 GB
runtime artifact starts from a 256 MiB partial and completes over HTTPS Range.
No local package-install shortcut is used.
"""

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


def digest(path: pathlib.Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{result.stdout}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True, type=pathlib.Path)
    parser.add_argument("--server", required=True, type=pathlib.Path)
    parser.add_argument("--package-source", required=True, type=pathlib.Path)
    parser.add_argument("--source-cache", required=True, type=pathlib.Path)
    parser.add_argument("--work-root", required=True, type=pathlib.Path)
    arguments = parser.parse_args()

    work = arguments.work_root.resolve()
    shutil.rmtree(work, ignore_errors=True)
    registry = work / "registry"
    cache = work / "cache"
    evidence = work / "evidence"
    registry.mkdir(parents=True)
    evidence.mkdir()
    certificate = work / "certificate.pem"
    key = work / "key.pem"
    run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
        "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
        "-keyout", str(key), "-out", str(certificate),
    ])
    port_file = work / "port"
    server_log = evidence / "server.jsonl"
    server = subprocess.Popen([
        sys.executable, str(arguments.server), "--root", str(registry),
        "--cert", str(certificate), "--key", str(key), "--log", str(server_log),
        "--port-file", str(port_file),
    ])
    try:
        for _ in range(100):
            if port_file.exists():
                break
            if server.poll() is not None:
                raise RuntimeError("HTTPS fixture failed to start")
            time.sleep(0.05)
        port = int(port_file.read_text(encoding="utf-8"))
        base = f"https://localhost:{port}"

        source_manifest_path = arguments.package_source / "vrhino-model.json"
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        runtime_id = source_manifest["entrypoint"]["runtime_artifact"]
        artifacts_by_id = {item["id"]: item for item in source_manifest["artifacts"]}
        runtime = artifacts_by_id[runtime_id]

        artifact_root = registry / "artifacts"
        artifact_root.mkdir()
        for item in source_manifest["artifacts"]:
            source = arguments.package_source / item["path"]
            target = artifact_root / item["sha256"]
            os.link(source, target)
            if source.stat().st_size != item["size"]:
                raise RuntimeError(f"source artifact size changed: {item['id']}")

        def publish(version: str) -> None:
            manifest = json.loads(json.dumps(source_manifest))
            manifest["identity"]["version"] = version
            target = registry / "v1" / "models" / manifest["identity"]["namespace"] / \
                     manifest["identity"]["name"] / version
            target.mkdir(parents=True)
            manifest_path = target / "vrhino-model.json"
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            identity = (f"{manifest['identity']['namespace']}/{manifest['identity']['name']}:"
                        f"{version}")
            descriptor = {
                "registry_schema_version": 1,
                "identity": identity,
                "manifest": {
                    "url": f"{base}/v1/models/{manifest['identity']['namespace']}/"
                           f"{manifest['identity']['name']}/{version}/vrhino-model.json",
                    "size": manifest_path.stat().st_size,
                    "sha256": digest(manifest_path),
                },
                "artifacts": [
                    {"id": item["id"], "url": f"{base}/artifacts/{item['sha256']}"}
                    for item in manifest["artifacts"]
                ],
            }
            (target / "index.json").write_text(json.dumps(descriptor, indent=2) + "\n",
                                                encoding="utf-8")

        publish("1.0.0")
        publish("1.0.1")

        reused = 0
        for item in source_manifest["artifacts"]:
            if item["id"] == runtime_id:
                continue
            source = arguments.source_cache / "blobs" / "sha256" / item["sha256"][:2] / \
                     item["sha256"]
            target = cache / "blobs" / "sha256" / item["sha256"][:2] / item["sha256"]
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
            reused += item["size"]

        partial_bytes = 256 * 1024 * 1024
        runtime_source = arguments.package_source / runtime["path"]
        partial = cache / "tmp" / "downloads" / f"{runtime['sha256']}.partial"
        partial.parent.mkdir(parents=True)
        with runtime_source.open("rb") as input_file, partial.open("wb") as output:
            remaining = partial_bytes
            while remaining:
                data = input_file.read(min(8 * 1024 * 1024, remaining))
                if not data:
                    raise RuntimeError("runtime source is shorter than requested partial")
                output.write(data)
                remaining -= len(data)

        common = [str(arguments.cli), "--cache-root", str(cache), "--registry", base,
                  "--ca-file", str(certificate)]
        first = run(common + ["pull", "vrhino/ltx-video-v0.9.1:1.0.0"])
        (evidence / "pull-1.0.0.log").write_text(first.stdout, encoding="utf-8")
        listing = run(common + ["list"])
        info = run(common + ["info", "vrhino/ltx-video-v0.9.1:1.0.0"])
        (evidence / "list.log").write_text(listing.stdout, encoding="utf-8")
        (evidence / "info.log").write_text(info.stdout, encoding="utf-8")
        second = run(common + ["pull", "vrhino/ltx-video-v0.9.1:1.0.1"])
        (evidence / "pull-1.0.1.log").write_text(second.stdout, encoding="utf-8")
        if "Downloaded: 0 B (0 bytes)" not in second.stdout:
            raise RuntimeError("LTX 1.0.1 did not reuse the complete CAS")

        actual_runtime = cache / "blobs" / "sha256" / runtime["sha256"][:2] / runtime["sha256"]
        if actual_runtime.stat().st_size != runtime["size"] or digest(actual_runtime) != runtime["sha256"]:
            raise RuntimeError("pulled real LTX runtime did not pass size/SHA verification")
        log_text = server_log.read_text(encoding="utf-8")
        if f'"range": "bytes={partial_bytes}-"' not in log_text or '"status": 206' not in log_text:
            raise RuntimeError("real LTX transfer did not use HTTPS Range resume")

        summary = {
            "identity": "vrhino/ltx-video-v0.9.1:1.0.0",
            "logical_bytes": sum(item["size"] for item in source_manifest["artifacts"]),
            "controlled_cas_preseed_bytes": reused,
            "initial_partial_bytes": partial_bytes,
            "expected_https_bytes": runtime["size"] - partial_bytes,
            "runtime_artifact_bytes": runtime["size"],
            "runtime_sha256": runtime["sha256"],
            "range_resume": True,
            "size_sha256_verified": True,
            "atomic_install": True,
            "list": True,
            "info": True,
            "version_1_0_1_downloaded_bytes": 0,
            "version_1_0_1_reused_bytes": sum(item["size"] for item in source_manifest["artifacts"]),
            "full_all_artifacts_network_download": False,
        }
        (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                            encoding="utf-8")
        print("LTX real registry pull smoke: PASS")
    finally:
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


if __name__ == "__main__":
    main()
