#!/usr/bin/env python3
"""Assemble a local source build with an explicit, audited runtime dependency bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--dependency-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check', action='store_true', help='validate inputs without producing output')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    build = args.build_dir.resolve(strict=True)
    deps = args.dependency_root.resolve(strict=True)
    output = args.output.absolute()
    if output.exists():
        parser.error('output must be a new directory')
    required_files = [build / 'vrhino', build / 'vrhino-native',
                      deps / 'media/bin/vrhino-ffmpeg', source / 'LICENSE',
                      source / 'NOTICE', source / 'THIRD_PARTY_NOTICES.md']
    required_dirs = [deps / p for p in ('lib', 'media/sources', 'media/licenses', 'licenses')]
    for path in required_files:
        if not path.is_file():
            parser.error(f'missing required file: {path}')
    for path in required_dirs:
        if not path.is_dir():
            parser.error(f'missing audited dependency directory: {path}')
    for name in ('vrhino', 'vrhino-native'):
        with (build / name).open('rb') as stream:
            if stream.read(4) != b'\x7fELF':
                parser.error(f'{name} must be a freshly built ELF executable')
    if any((deps / 'lib').glob('libcuda.so*')):
        parser.error('dependency bundle must not contain the host NVIDIA driver')
    directories = [name for name in ('lib', 'media', 'licenses', 'sbom') if (deps / name).is_dir()]
    for directory in directories:
        for path in (deps / directory).rglob('*'):
            if path.is_symlink() and not path.resolve().is_relative_to(deps):
                parser.error(f'dependency symlink escapes its bundle: {path}')
    patcher = shutil.which('patchelf')
    if not patcher:
        parser.error('patchelf is required for VRhino-owned executable RUNPATH')
    if args.check:
        print('Packaging inputs: PASS (binary/model qualification is a separate gate)')
        return
    output.mkdir(parents=True)
    for directory in directories:
        shutil.copytree(deps / directory, output / directory, symlinks=True)
    for old_license in (output / 'licenses').glob('*BINARY*LICENSE*'):
        history = output / 'licenses/historical-release-origin'
        history.mkdir(exist_ok=True)
        old_license.rename(history / old_license.name)
    for directory in ('bin', 'libexec', 'share/vrhino'):
        (output / directory).mkdir(parents=True, exist_ok=True)
    # Retain the dependency SBOM as provenance, without mislabeling old executable hashes.
    (output / 'sbom').mkdir(exist_ok=True)
    for path in (output / 'sbom').iterdir():
        if path.is_file():
            path.rename(path.with_name('dependency-origin-' + path.name))
    if (deps / 'THIRD_PARTY_NOTICES.txt').is_file():
        shutil.copy2(deps / 'THIRD_PARTY_NOTICES.txt', output / 'licenses/dependency-origin-NOTICES.txt')
    shutil.copytree(source / 'native/specs', output / 'share/vrhino/converters', symlinks=True)
    for name in ('vrhino', 'vrhino-native'):
        binary = output / 'libexec' / name
        shutil.copy2(build / name, binary)
        subprocess.run([patcher, '--set-rpath', '$ORIGIN/../lib', str(binary)], check=True)
        launcher = output / 'bin' / name
        launcher.write_text('#!/bin/sh\nset -eu\n'
            'root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)\n'
            'export LD_LIBRARY_PATH="$root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
            'export VRHINO_FFMPEG="$root/media/bin/vrhino-ffmpeg"\n'
            f'exec "$root/libexec/{name}" "$@"\n')
        launcher.chmod(0o755)
    # A sibling helper also supports direct libexec invocation and product preflight.
    helper = output / 'libexec/vrhino-ffmpeg'
    helper.write_text('#!/bin/sh\nset -eu\n'
        'root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)\n'
        'export LD_LIBRARY_PATH="$root/media/lib:$root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
        'exec "$root/media/bin/vrhino-ffmpeg" "$@"\n')
    helper.chmod(0o755)
    for name in ('LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md', 'VERSION'):
        shutil.copy2(source / name, output / name)
    manifest = {'kind': 'local-source-build-not-a-published-release',
                'project_license': 'Apache-2.0',
                'version': (source / 'VERSION').read_text().strip(),
                'dependency_licenses': 'licenses/ and media/licenses/',
                'dependency_provenance': 'licenses/dependency-origin-NOTICES.txt and optional sbom/dependency-origin-*',
                'binaries': {name: digest(output / 'libexec' / name)
                             for name in ('vrhino', 'vrhino-native')}}
    (output / 'SOURCE-BUILD.json').write_text(json.dumps(manifest, indent=2) + '\n')
    # Package input libraries/media are copied without patching or stripping.
    for directory in ('lib', 'media'):
        for path in (deps / directory).rglob('*'):
            if path.is_file() and not path.is_symlink():
                if digest(path) != digest(output / path.relative_to(deps)):
                    raise RuntimeError('dependency payload changed during assembly')
    checksums = []
    for path in sorted(output.rglob('*')):
        if path.is_file() and not path.is_symlink():
            checksums.append(f'{digest(path)}  {path.relative_to(output)}\n')
    (output / 'SHA256SUMS').write_text(''.join(checksums))
    print(f'Local package assembled: {output}; release qualification remains required')


if __name__ == '__main__':
    main()
