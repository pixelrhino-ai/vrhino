#!/usr/bin/env python3
"""Create a versioned, SHA-pinned v2 catalog for two frozen artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from urllib.parse import urlparse


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_kv(path):
    result = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition('=')
        if not sep or not key or key in result:
            raise ValueError(f'invalid or duplicate catalog key: {line}')
        result[key] = value
    return result


def member_json(archive, relative):
    with tarfile.open(archive, 'r:gz') as file:
        members = [m for m in file if m.name.endswith('/' + relative)]
        if len(members) != 1:
            raise ValueError(f'missing or duplicate {relative} in {archive}')
        return json.load(file.extractfile(members[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle-dir', type=Path, required=True)
    parser.add_argument('--base-url', required=True, help='HTTPS directory serving the frozen archives')
    parser.add_argument('--catalog-version', required=True,
                        help='Immutable publication identity, for example v0.9.2-alpha')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    base = args.base_url.rstrip('/')
    parsed = urlparse(base)
    if parsed.scheme != 'https' or not parsed.netloc or parsed.query or parsed.fragment:
        parser.error('base URL must be a fixed HTTPS directory')
    if not args.catalog_version or not all(c.isalnum() or c in '._-' for c in args.catalog_version):
        parser.error('catalog version must use only letters, digits, dot, underscore, or dash')
    if args.output.exists():
        parser.error('catalog output must not exist')
    bundle = args.bundle_dir.resolve(strict=True)
    original = read_kv(bundle / 'catalog.txt')
    for kind in ('core', 'runtime'):
        path = bundle / original[kind + '_name']
        if sha(path) != original[kind + '_sha256']:
            parser.error(f'{kind} archive differs from frozen catalog')
    core = member_json(bundle / original['core_name'], 'manifest.json')
    runtime = member_json(bundle / original['runtime_name'], 'manifest.json')
    requirement = core['requires']['nvidia_runtime']
    if (requirement['id'] != runtime['id'] or
            requirement['sha256'] != original['runtime_sha256'] or
            requirement['runtime_api'] != runtime['runtime_api'] or
            runtime['id'] != original['runtime_id']):
        parser.error('core/vendor manifest identity mismatch')
    lines = [
        'schema=vrhino-distribution-v2-catalog-v1',
        f'catalog_version={args.catalog_version}',
        'platform=linux-x86_64',
        'runtime_api=1',
        f'source_archive_sha256={core["source_archive_sha256"]}',
    ]
    for key in ('core_name', 'core_sha256', 'core_root', 'core_version',
                'runtime_name', 'runtime_sha256', 'runtime_root', 'runtime_id'):
        lines.append(f'{key}={original[key]}')
    lines += [f'core_url={base}/{original["core_name"]}',
              f'runtime_url={base}/{original["runtime_name"]}']
    args.output.write_text('\n'.join(lines) + '\n')
    print(json.dumps({'catalog': str(args.output), 'sha256': sha(args.output),
                      'core_sha256': original['core_sha256'],
                      'runtime_sha256': original['runtime_sha256']}, indent=2))


if __name__ == '__main__':
    main()
