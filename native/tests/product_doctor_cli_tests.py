#!/usr/bin/env python3
"""Small command-surface and redaction checks for the native doctor command."""

import os
import pathlib
import subprocess
import sys
import tempfile


def run(command: list[str], environment: dict[str, str], expected: int) -> str:
    result = subprocess.run(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != expected:
        raise RuntimeError(
            f"unexpected exit {result.returncode}, expected {expected}\n{result.stdout}"
        )
    return result.stdout


def main() -> int:
    if len(sys.argv) != 2:
        raise RuntimeError("usage: product_doctor_cli_tests.py VRHINO")
    executable = str(pathlib.Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix="vrhino-doctor-cli-") as temporary:
        environment = os.environ.copy()
        environment["HOME"] = temporary
        environment.pop("VRHINO_HOME", None)
        token = "doctor-cli-token-unique-sentinel"
        password = "doctor-cli-proxy-password-unique-sentinel"
        prompt = "doctor-cli-private-prompt-sentinel"
        environment["HF_TOKEN"] = token
        environment["HTTPS_PROXY"] = (
            f"https://user:{password}@example.invalid:443"
        )
        environment["VRHINO_TEST_PROMPT"] = prompt

        report = run([executable, "doctor"], environment, 0)
        required = (
            "VRhino Doctor",
            "HF_TOKEN: set",
            "HTTPS_PROXY: set",
            "Root: ~/.vrhino",
            "Overall\nREADY",
        )
        if any(value not in report for value in required):
            raise RuntimeError(f"doctor report omitted a required field\n{report}")
        if any(secret in report for secret in (token, password, prompt)):
            raise RuntimeError("doctor report exposed a sentinel secret")

        custom = pathlib.Path(temporary) / "private-cache-name-sentinel"
        custom_report = run(
            [executable, "--cache-root", str(custom), "doctor"], environment, 0
        )
        if "Root: <custom>" not in custom_report or str(custom) in custom_report:
            raise RuntimeError("doctor did not redact an explicit cache root")

        usage = run([executable, "doctor", "one", "two"], environment, 2)
        if "Usage:" not in usage:
            raise RuntimeError("invalid doctor arguments did not show usage")

        not_installed = run(
            [executable, "doctor", "vrhino/not-installed:1.0.0"], environment, 1
        )
        if "Installed: no" not in not_installed or "Overall\nFAILED" not in not_installed:
            raise RuntimeError("doctor model argument was not parsed diagnostically")

    print("product doctor CLI tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
