#!/usr/bin/env python3
"""End-to-end tests for the native registry/pull product layer."""

import argparse
import hashlib
import json
import os
import pathlib
import pty
import shutil
import signal
import subprocess
import sys
import time


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_payload(path: pathlib.Path, pattern: bytes, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as output:
        full, tail = divmod(size, len(pattern))
        for _ in range(full):
            output.write(pattern)
        output.write(pattern[:tail])


def manifest(identity: str, artifacts: list[dict], schema: int = 1) -> dict:
    namespace_name, version = identity.split(":", 1)
    namespace, name = namespace_name.split("/", 1)
    return {
        "schema_version": schema,
        "identity": {
            "namespace": namespace,
            "name": name,
            "version": version,
            "architecture": "test_arch",
            "publisher": "VRhino",
        },
        "compatibility": {
            "runtime_contract": "cuda-v1",
            "vrm_schema": {"format_major": 0, "format_minor": 1, "metadata_schema": 1},
        },
        "artifacts": artifacts,
        "entrypoint": {
            "runtime_artifact": "runtime",
            "components": [],
            "default_preset": "default",
        },
        "defaults": {
            "default_preset": "default",
            "presets": {"default": {"profile_artifact": "profile", "inputs": {}}},
        },
        "hardware": {
            "presets": {
                "default": {"minimum_vram_bytes": None, "recommended_vram_bytes": None}
            }
        },
        "source": {
            "repository": "vrhino/registry-test",
            "revision": "phase24c",
            "converter_version": "test",
        },
        "license": {
            "identifier": "LicenseRef-Test",
            "artifact": "license",
            "upstream_notice": "test fixture",
        },
    }


def artifact(identifier: str, role: str, relative_path: str, source: pathlib.Path,
             declared_hash: str | None = None, declared_size: int | None = None) -> dict:
    return {
        "id": identifier,
        "role": role,
        "path": relative_path,
        "size": source.stat().st_size if declared_size is None else declared_size,
        "sha256": sha256(source) if declared_hash is None else declared_hash,
        "required": True,
    }


def publish(root: pathlib.Path, base: str, identity: str, declarations: list[dict],
            urls: dict[str, str], schema: int = 1) -> pathlib.Path:
    namespace_name, version = identity.split(":", 1)
    namespace, name = namespace_name.split("/", 1)
    target = root / "v1" / "models" / namespace / name / version
    target.mkdir(parents=True, exist_ok=True)
    manifest_path = target / "vrhino-model.json"
    manifest_path.write_text(json.dumps(manifest(identity, declarations, schema), indent=2) + "\n",
                             encoding="utf-8")
    descriptor = {
        "registry_schema_version": 1,
        "identity": identity,
        "manifest": {
            "url": f"{base}/v1/models/{namespace}/{name}/{version}/vrhino-model.json",
            "size": manifest_path.stat().st_size,
            "sha256": sha256(manifest_path),
        },
        "artifacts": [{"id": identifier, "url": url} for identifier, url in urls.items()],
    }
    (target / "index.json").write_text(json.dumps(descriptor, indent=2) + "\n",
                                        encoding="utf-8")
    return manifest_path


def run(command: list[str], expected: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, check=False)
    if result.returncode != expected:
        raise RuntimeError(f"command returned {result.returncode}, expected {expected}: "
                           f"{' '.join(command)}\n{result.stdout}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", required=True, type=pathlib.Path)
    parser.add_argument("--server", required=True, type=pathlib.Path)
    parser.add_argument("--work-root", required=True, type=pathlib.Path)
    arguments = parser.parse_args()
    work = arguments.work_root.resolve()
    shutil.rmtree(work, ignore_errors=True)
    registry_root = work / "registry"
    cache = work / "cache"
    registry_root.mkdir(parents=True)
    (work / "logs").mkdir()

    certificate = work / "certificate.pem"
    key = work / "key.pem"
    run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
        "-subj", "/CN=localhost", "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1",
        "-keyout", str(key), "-out", str(certificate),
    ])
    port_file = work / "port"
    server_log = work / "logs" / "server.jsonl"
    server = subprocess.Popen([
        sys.executable, str(arguments.server), "--root", str(registry_root),
        "--cert", str(certificate), "--key", str(key), "--log", str(server_log),
        "--port-file", str(port_file),
    ])
    try:
        for _ in range(100):
            if port_file.exists():
                break
            if server.poll() is not None:
                raise RuntimeError("registry fixture server terminated during startup")
            time.sleep(0.05)
        port = int(port_file.read_text(encoding="utf-8"))
        base = f"https://localhost:{port}"
        common = [str(arguments.cli), "--cache-root", str(cache), "--registry", base,
                  "--ca-file", str(certificate)]

        sources = work / "sources"
        runtime_a = sources / "runtime-a.vrm"
        runtime_b = sources / "runtime-b.vrm"
        runtime_c = sources / "runtime-c.vrm"
        runtime_d = sources / "runtime-d.vrm"
        runtime_e = sources / "runtime-e.vrm"
        runtime_f = sources / "runtime-f.vrm"
        runtime_g = sources / "runtime-g.vrm"
        runtime_h = sources / "runtime-h.vrm"
        runtime_i = sources / "runtime-i.vrm"
        runtime_j = sources / "runtime-j.vrm"
        profile = sources / "profile.json"
        license_file = sources / "LICENSE.txt"
        cancel_profile = sources / "cancel-profile.json"
        cancel_license = sources / "CANCEL-LICENSE.txt"
        write_payload(runtime_a, b"vrhino-registry-a", 8 * 1024 * 1024 + 17)
        write_payload(runtime_b, b"vrhino-registry-b", 5 * 1024 * 1024 + 31)
        write_payload(runtime_c, b"vrhino-registry-c", 6 * 1024 * 1024 + 47)
        write_payload(runtime_d, b"vrhino-registry-d", 4 * 1024 * 1024 + 59)
        write_payload(runtime_e, b"vrhino-registry-e", 7 * 1024 * 1024 + 71)
        write_payload(runtime_f, b"vrhino-registry-f", 3 * 1024 * 1024 + 89)
        write_payload(runtime_g, b"vrhino-registry-g", 9 * 1024 * 1024 + 97)
        write_payload(runtime_h, b"vrhino-registry-h", 32 * 1024 * 1024 + 113)
        write_payload(runtime_i, b"vrhino-registry-i", 2 * 1024 * 1024 + 127)
        write_payload(runtime_j, b"vrhino-registry-j", 3 * 1024 * 1024 + 139)
        write_payload(profile, b"{}\n", 3)
        write_payload(license_file, b"test license\n", 13)
        write_payload(cancel_profile, b"{\"cancel\":true}\n", 256 * 1024 + 3)
        write_payload(cancel_license, b"cancel license\n", 256 * 1024 + 5)
        artifact_root = registry_root / "artifacts"
        artifact_root.mkdir()
        for source in (runtime_a, runtime_b, runtime_c, runtime_d, runtime_e, runtime_f,
                       runtime_g, runtime_h, runtime_i, runtime_j, profile, license_file,
                       cancel_profile, cancel_license):
            os.link(source, artifact_root / sha256(source))

        def declarations(runtime: pathlib.Path, bad_hash: bool = False,
                         huge_size: int | None = None) -> list[dict]:
            return [
                artifact("runtime", "runtime.vrm", "model/model.vrm", runtime,
                         "0" * 64 if bad_hash else None, huge_size),
                artifact("profile", "execution.profile", "execution/default.json", profile),
                artifact("license", "legal.license", "legal/LICENSE.txt", license_file),
            ]

        def urls(runtime: pathlib.Path, prefix: str = "") -> dict[str, str]:
            return {
                "runtime": f"{base}/{prefix}artifacts/{sha256(runtime)}",
                "profile": f"{base}/artifacts/{sha256(profile)}",
                "license": f"{base}/artifacts/{sha256(license_file)}",
            }

        publish(registry_root, base, "vrhino/fixture:1.0.0", declarations(runtime_a),
                urls(runtime_a))
        publish(registry_root, base, "vrhino/fixture:1.0.1", declarations(runtime_a),
                urls(runtime_a))
        publish(registry_root, base, "vrhino/resume:1.0.0", declarations(runtime_b),
                urls(runtime_b))
        publish(registry_root, base, "vrhino/no-range:1.0.0", declarations(runtime_c),
                urls(runtime_c, "no-range/"))
        publish(registry_root, base, "vrhino/redirect:1.0.0", declarations(runtime_d),
                urls(runtime_d, "redirect/"))
        publish(registry_root, base, "vrhino/truncated:1.0.0", declarations(runtime_e),
                urls(runtime_e, "truncate-once/"))
        publish(registry_root, base, "vrhino/wrong-hash:1.0.0",
                declarations(runtime_b, bad_hash=True), urls(runtime_b))
        publish(registry_root, base, "vrhino/wrong-length:1.0.0", declarations(runtime_f),
                urls(runtime_f, "wrong-length/"))
        publish(registry_root, base, "vrhino/unsupported:1.0.0", declarations(runtime_b),
                urls(runtime_b), schema=2)
        publish(registry_root, base, "vrhino/no-space:1.0.0",
                declarations(runtime_b, huge_size=10**15), urls(runtime_b))
        publish(registry_root, base, "vrhino/concurrent:1.0.0", declarations(runtime_g),
                urls(runtime_g))
        publish(registry_root, base, "vrhino/oversized:1.0.0", declarations(runtime_i),
                urls(runtime_i))
        publish(registry_root, base, "vrhino/tty:1.0.0", declarations(runtime_j),
                urls(runtime_j))
        publish(registry_root, base, "vrhino/environment-home:1.0.0",
                declarations(runtime_j), urls(runtime_j))
        cancel_declarations = [
            artifact("runtime", "runtime.vrm", "model/model.vrm", runtime_h),
            artifact("profile", "execution.profile", "execution/default.json",
                     cancel_profile),
            artifact("license", "legal.license", "legal/LICENSE.txt", cancel_license),
        ]
        cancel_urls = {
            "runtime": f"{base}/slow/artifacts/{sha256(runtime_h)}",
            "profile": f"{base}/artifacts/{sha256(cancel_profile)}",
            "license": f"{base}/artifacts/{sha256(cancel_license)}",
        }
        publish(registry_root, base, "vrhino/cancel:1.0.0", cancel_declarations,
                cancel_urls)

        untrusted = run([str(arguments.cli), "--cache-root", str(cache), "--registry", base,
                         "pull", "vrhino/fixture:1.0.0"], expected=1)
        assert "REGISTRY_UNAVAILABLE" in untrusted.stdout
        insecure = run([str(arguments.cli), "--cache-root", str(cache), "--registry",
                        f"http://127.0.0.1:{port}", "pull", "vrhino/fixture:1.0.0"], expected=1)
        assert "REGISTRY_UNAVAILABLE" in insecure.stdout

        result = run(common + ["pull", "vrhino/fixture:1.0.0"])
        (work / "logs" / "pull-normal.log").write_text(result.stdout, encoding="utf-8")
        assert "Installed vrhino/fixture:1.0.0" in result.stdout
        normal_progress = [line for line in result.stdout.splitlines()
                           if line.startswith("Acquiring source")]
        assert normal_progress and "100%" in normal_progress[-1]
        assert len(normal_progress) <= 11
        assert "\r" not in result.stdout and "\x1b" not in result.stdout
        result = run(common + ["pull", "vrhino/fixture:1.0.1"])
        (work / "logs" / "pull-cas-reuse.log").write_text(result.stdout, encoding="utf-8")
        assert "Registry downloaded: 0 B (0 bytes)" in result.stdout
        result = run(common + ["pull", "vrhino/fixture:1.0.0"])
        assert "Already installed: yes" in result.stdout

        resume_hash = sha256(runtime_b)
        partial = cache / "tmp" / "downloads" / f"{resume_hash}.partial"
        partial.parent.mkdir(parents=True, exist_ok=True)
        with runtime_b.open("rb") as source, partial.open("wb") as output:
            output.write(source.read(runtime_b.stat().st_size // 2))
        resume_offset = partial.stat().st_size
        result = run(common + ["pull", "vrhino/resume:1.0.0"])
        (work / "logs" / "pull-resume.log").write_text(result.stdout, encoding="utf-8")
        assert not partial.exists()
        assert (cache / "blobs" / "sha256" / resume_hash[:2] / resume_hash).is_file()

        no_range_hash = sha256(runtime_c)
        partial = cache / "tmp" / "downloads" / f"{no_range_hash}.partial"
        if (partial.exists()):
            partial.unlink()
        with runtime_c.open("rb") as source, partial.open("wb") as output:
            output.write(source.read(runtime_c.stat().st_size // 3))
        result = run(common + ["pull", "vrhino/no-range:1.0.0"])
        (work / "logs" / "pull-no-range.log").write_text(result.stdout, encoding="utf-8")
        assert not partial.exists()

        oversized_hash = sha256(runtime_i)
        oversized_partial = cache / "tmp" / "downloads" / f"{oversized_hash}.partial"
        oversized_partial.parent.mkdir(parents=True, exist_ok=True)
        oversized_partial.write_bytes(b"x" * (runtime_i.stat().st_size + 1))
        run(common + ["pull", "vrhino/oversized:1.0.0"])
        assert not oversized_partial.exists()

        cancel_hash = sha256(runtime_h)
        cancel_partial = cache / "tmp" / "downloads" / f"{cancel_hash}.partial"
        cancelled = subprocess.Popen(
            common + ["pull", "vrhino/cancel:1.0.0"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for _ in range(200):
            if cancel_partial.exists() and cancel_partial.stat().st_size > 0:
                break
            if cancelled.poll() is not None:
                raise RuntimeError("cancellation fixture pull exited before transfer started")
            time.sleep(0.025)
        else:
            raise RuntimeError("cancellation fixture did not create a partial")
        cancelled.send_signal(signal.SIGINT)
        cancelled_output, _ = cancelled.communicate(timeout=5)
        assert cancelled.returncode == 130, cancelled_output
        assert cancelled_output.count(
            "Download interrupted. Partial download preserved for resume.") == 1
        assert cancel_partial.is_file()
        cancel_offset = cancel_partial.stat().st_size
        assert 0 < cancel_offset < runtime_h.stat().st_size
        assert not (cache / "blobs" / "sha256" / cancel_hash[:2] / cancel_hash).exists()
        assert not (cache / "models/vrhino/cancel/1.0.0/vrhino-model.json").exists()
        (work / "logs" / "pull-cancel.log").write_text(
            cancelled_output, encoding="utf-8")

        resumed_cancel = run(common + ["pull", "vrhino/cancel:1.0.0"])
        assert "Installed vrhino/cancel:1.0.0" in resumed_cancel.stdout
        assert "\r" not in resumed_cancel.stdout and "\x1b" not in resumed_cancel.stdout
        assert len([line for line in resumed_cancel.stdout.splitlines()
                    if line.startswith("Acquiring source")]) <= 11
        assert not cancel_partial.exists()
        assert (cache / "blobs" / "sha256" / cancel_hash[:2] / cancel_hash).is_file()

        master_fd, slave_fd = pty.openpty()
        tty_process = subprocess.Popen(
            common + ["pull", "vrhino/tty:1.0.0"], stdout=slave_fd,
            stderr=slave_fd, close_fds=True)
        os.close(slave_fd)
        tty_chunks: list[bytes] = []
        try:
            while True:
                try:
                    chunk = os.read(master_fd, 4096)
                except OSError:
                    break
                if not chunk:
                    break
                tty_chunks.append(chunk)
        finally:
            os.close(master_fd)
        tty_process.wait(timeout=5)
        tty_output = b"".join(tty_chunks)
        assert tty_process.returncode == 0, tty_output.decode(errors="replace")
        assert b"\rAcquiring source" in tty_output
        assert b"\x1b" not in tty_output
        assert len([line for line in tty_output.split(b"\n")
                    if b"Acquiring source" in line]) == 1

        environment_cache = work / "environment-cache"
        environment = os.environ.copy()
        environment["VRHINO_HOME"] = str(environment_cache)
        environment["HOME"] = str(work / "unused-home")
        environment_pull = subprocess.run([
            str(arguments.cli), "--registry", base, "--ca-file", str(certificate),
            "pull", "vrhino/environment-home:1.0.0",
        ], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            env=environment, check=False)
        assert environment_pull.returncode == 0, environment_pull.stdout
        assert (environment_cache /
                "models/vrhino/environment-home/1.0.0/vrhino-model.json").is_file()
        assert not (pathlib.Path(environment["HOME"]) / ".vrhino").exists()

        run(common + ["pull", "vrhino/redirect:1.0.0"])
        run(common + ["pull", "vrhino/truncated:1.0.0"])

        records = [json.loads(line) for line in server_log.read_text(
            encoding="utf-8").splitlines() if line]
        assert any(record["path"].endswith(f"/artifacts/{resume_hash}") and
                   record["range"] == f"bytes={resume_offset}-" and
                   record["status"] == 206 for record in records)
        assert any(record["path"].startswith("/no-range/") and
                   record["range"] is not None and record["status"] == 200
                   for record in records)
        assert any(record["path"].startswith("/slow/") and
                   record["range"] == f"bytes={cancel_offset}-" and
                   record["status"] == 206 for record in records)
        assert not any(record["path"].endswith(f"/artifacts/{sha256(cancel_profile)}")
                       for record in records[:next(
                           index for index, record in enumerate(records)
                           if record["path"].startswith("/slow/") and
                           record["range"] == f"bytes={cancel_offset}-")])

        wrong_hash = run(common + ["pull", "vrhino/wrong-hash:1.0.0"], expected=1)
        assert "CHECKSUM_MISMATCH" in wrong_hash.stdout
        wrong_length = run(common + ["pull", "vrhino/wrong-length:1.0.0"], expected=1)
        assert "DOWNLOAD_" in wrong_length.stdout
        unsupported = run(common + ["pull", "vrhino/unsupported:1.0.0"], expected=1)
        assert "PACKAGE_VERSION_UNSUPPORTED" in unsupported.stdout
        no_space = run(common + ["pull", "vrhino/no-space:1.0.0"], expected=1)
        assert "INSUFFICIENT_DISK_SPACE" in no_space.stdout

        first = subprocess.Popen(common + ["pull", "vrhino/concurrent:1.0.0"],
                                 text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        second = subprocess.Popen(common + ["pull", "vrhino/concurrent:1.0.0"],
                                  text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        first_output, _ = first.communicate()
        second_output, _ = second.communicate()
        if first.returncode != 0 or second.returncode != 0:
            raise RuntimeError(f"concurrent pull failed\n{first_output}\n{second_output}")
        (work / "logs" / "pull-concurrent.log").write_text(
            first_output + "\n--- second ---\n" + second_output, encoding="utf-8")

        list_result = run(common + ["list"])
        info_result = run(common + ["info", "vrhino/fixture:1.0.0"])
        run(common + ["rm", "vrhino/fixture:1.0.0"])
        (work / "logs" / "list.log").write_text(list_result.stdout, encoding="utf-8")
        (work / "logs" / "info.log").write_text(info_result.stdout, encoding="utf-8")

        installed = list((cache / "models").rglob("vrhino-model.json"))
        blobs = list((cache / "blobs" / "sha256").rglob("*"))
        summary = {
            "https": True,
            "certificate_verification_default": True,
            "plain_http_rejected_default": True,
            "range_resume": True,
            "no_range_restart": True,
            "oversized_partial_restart": True,
            "single_sigint_cancellation": True,
            "cancelled_partial_preserved": True,
            "cancelled_pull_exit_code": 130,
            "aggregate_progress_throttled": True,
            "tty_progress_in_place": True,
            "vrhino_home_cli_pull": True,
            "redirect": True,
            "truncated_resume": True,
            "checksum_mismatch_fail_closed": True,
            "wrong_length_fail_closed": True,
            "unsupported_schema_fail_closed": True,
            "insufficient_disk_fail_closed": True,
            "concurrent_pull": True,
            "cas_reuse": True,
            "installed_manifests": len(installed),
            "cas_files": sum(path.is_file() for path in blobs),
        }
        (work / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                            encoding="utf-8")
        print("registry fixture tests: PASS")
    finally:
        server.send_signal(signal.SIGTERM)
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


if __name__ == "__main__":
    main()
