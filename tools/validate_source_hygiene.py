#!/usr/bin/env python3
"""Check tracked source boundaries; full secret/provenance review remains separate."""
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {'.git', '__pycache__', 'test-assets', 'private-assets', 'checkpoints'}
FORBIDDEN_SUFFIXES = {'.safetensors', '.ckpt', '.pt', '.pth', '.vrm', '.mp4', '.wav'}
SECRET_PATTERNS = (
    rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    rb'gh[pousr]_[A-Za-z0-9]{36,}',
    rb'github_pat_[A-Za-z0-9_]{70,}',
    rb'AKIA[0-9A-Z]{16}',
    rb'https?://[^\s/:]+:[^\s/@]+@',
)


def main():
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    errors = []
    count = 0
    for name in filter(None, names):
        path = ROOT / name
        relative = Path(name)
        vendor = name.startswith('native/third_party/')
        count += 1
        if not path.exists() or not path.resolve().is_relative_to(ROOT):
            errors.append(f'{name}: missing file or escaping symlink')
            continue
        if (FORBIDDEN_PARTS.intersection(relative.parts) or relative.name == 'banner.png'
                or (not vendor and relative.suffix.lower() in FORBIDDEN_SUFFIXES)):
            errors.append(f'{name}: forbidden source artifact')
        if path.is_symlink() and path.is_dir():
            continue
        data = path.read_bytes()
        if len(data) > 10 * 1024 * 1024:
            errors.append(f'{name}: large file requires explicit source-boundary review')
        # Upstream fixtures include deliberate sample keys and binary model tests;
        # their pinned inventories and license review define the vendored boundary.
        if vendor:
            continue
        if data.startswith(b'\x7fELF'):
            errors.append(f'{name}: compiled ELF in project source')
        if re.search(rb'/root/autodl-tmp/' + rb'vrhino', data):
            errors.append(f'{name}: obsolete machine-local path')
        for pattern in SECRET_PATTERNS:
            matches = re.findall(pattern, data)
            # Explicit non-routable credential-redaction fixture, not a secret.
            if name == 'native/tests/product_doctor_cli_tests.py':
                matches = [m for m in matches if m != b'https://user:{password}@']
            if matches:
                errors.append(f'{name}: possible credential; inspect privately')
                break
    for name in ('LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md',
                 'native/third_party/tokenizer-build-sources/SOURCE-MANIFEST.json',
                 'native/third_party/cpp-httplib/LICENSE',
                 'native/third_party/cudnn_frontend/LICENSE.txt'):
        if not (ROOT / name).is_file():
            errors.append(f'{name}: required license/provenance input missing')
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print(f'PASS: {count} tracked files; source artifact and basic credential checks')
    return 0


if __name__ == '__main__':
    sys.exit(main())
