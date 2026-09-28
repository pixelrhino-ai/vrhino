#!/usr/bin/env python3
"""Rename the qualified core package root without changing its binary payloads."""
import argparse
import gzip
import hashlib
import io
import tarfile
from pathlib import Path

OLD_SHA = '70ff9060357310b4c865790b92cb14749a91247534e282bbf8afbe97ca8594b0'
OLD_ROOT = 'vrhino-core-v0.9.1-alpha-split-prototype'
RELEASE = 'v0.9.3-alpha'
NEW_ROOT = f'vrhino-core-{RELEASE}'
SOURCE_COMMIT = 'df27ee5eb4b987fa4cb6e83d744f001d61883595'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('qualified_core', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must not exist')
    if digest(args.qualified_core) != OLD_SHA:
        parser.error('qualified core SHA256 mismatch')
    notice = (f'VRhino distribution release: {RELEASE}\n'
              'Binary build version: v0.9.1-alpha (vrhino --version reports this build)\n'
              f'Binary source commit: {SOURCE_COMMIT}\n'
              'This package changes distribution layout and metadata only.\n').encode()
    notice_sha = hashlib.sha256(notice).hexdigest()
    with tarfile.open(args.qualified_core, 'r:gz') as source, args.output.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode='w') as target:
                seen = set()
                for member in source:
                    old_name = member.name
                    if not old_name.startswith(OLD_ROOT + '/'):
                        raise ValueError(f'unexpected member: {old_name}')
                    relative = old_name[len(OLD_ROOT) + 1:]
                    if not relative or relative in seen or relative == 'RELEASE.txt':
                        raise ValueError(f'duplicate or reserved member: {old_name}')
                    seen.add(relative)
                    member.name = NEW_ROOT + '/' + relative
                    # The qualified tar uses PAX path records for long names.
                    # A stale PAX path overrides TarInfo.name during extraction.
                    member.pax_headers.pop('path', None)
                    if member.isfile():
                        data = source.extractfile(member).read()
                        if relative == 'SHA256SUMS':
                            data += f'{notice_sha}  RELEASE.txt\n'.encode()
                            member.size = len(data)
                        target.addfile(member, io.BytesIO(data))
                    else:
                        target.addfile(member)
                if 'SHA256SUMS' not in seen or 'SOURCE-BUILD.json' not in seen:
                    raise ValueError('required provenance or checksums missing')
                extra = tarfile.TarInfo(NEW_ROOT + '/RELEASE.txt')
                extra.mode = 0o644
                extra.size = len(notice)
                extra.mtime = 0
                target.addfile(extra, io.BytesIO(notice))
    with tarfile.open(args.output, 'r:gz') as check:
        for member in check:
            if not member.name.startswith(NEW_ROOT + '/'):
                raise ValueError(f'old package root remains: {member.name}')
    print(f'{args.output}: {digest(args.output)} ({args.output.stat().st_size} bytes)')


if __name__ == '__main__':
    main()
