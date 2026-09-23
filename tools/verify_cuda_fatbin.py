#!/usr/bin/env python3
"""Check the CUDA images in a release executable before packaging it."""

import argparse
from pathlib import Path
import re
import subprocess
import sys


DEFAULT_SM = (80, 86, 89, 90, 120)


def images(cuobjdump: str, option: str, executable: Path) -> set[int]:
    try:
        result = subprocess.run(
            [cuobjdump, option, str(executable)],
            text=True, capture_output=True, check=False,
        )
    except OSError as error:
        raise RuntimeError(f"cannot run cuobjdump: {error}") from error
    if result.returncode:
        raise RuntimeError(f"cuobjdump {option} failed: {result.stderr.strip()}")
    suffix = "cubin" if option == "--list-elf" else "ptx"
    return {int(value) for value in re.findall(r"\.sm_(\d+)\." + suffix + r"\b", result.stdout)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executables", type=Path, nargs="+")
    parser.add_argument("--cuobjdump", default="cuobjdump")
    parser.add_argument("--required-sm", type=int, nargs="+", default=DEFAULT_SM)
    parser.add_argument("--required-ptx", type=int, default=80)
    args = parser.parse_args()
    for executable in args.executables:
        if not executable.is_file():
            parser.error(f"missing executable: {executable}")
        try:
            cubins = images(args.cuobjdump, "--list-elf", executable)
            ptx = images(args.cuobjdump, "--list-ptx", executable)
        except RuntimeError as error:
            print(f"CUDA_IMAGE_AUDIT=FAIL binary={executable} {error}", file=sys.stderr)
            return 1
        missing = set(args.required_sm) - cubins
        print(f"BINARY={executable}")
        print(f"CUBIN_SM={','.join(map(str, sorted(cubins)))}")
        print(f"PTX_COMPUTE={','.join(map(str, sorted(ptx)))}")
        if missing or args.required_ptx not in ptx:
            print(f"CUDA_IMAGE_AUDIT=FAIL missing_cubin={sorted(missing)} "
                  f"missing_ptx={args.required_ptx not in ptx}", file=sys.stderr)
            return 1
    print("CUDA_IMAGE_AUDIT=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
