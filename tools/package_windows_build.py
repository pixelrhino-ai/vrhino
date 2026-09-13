#!/usr/bin/env python3
"""Assemble an unpublished Windows x64 CUDA candidate (stdlib build tool only)."""
import argparse
import hashlib
import io
import json
import mmap
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import tempfile
import zipfile

DLLS = tuple(sorted('''archive.dll charset.dll iconv.dll LIBBZ2.dll libcrypto-3-x64.dll
libcurl.dll liblz4.dll liblzma.dll libssh2.dll libxml2.dll zlib.dll zstd.dll
cudart64_12.dll cublas64_12.dll cublasLt64_12.dll nvrtc64_120_0.dll nvrtc-builtins64_128.dll
cudnn64_9.dll cudnn_adv64_9.dll cudnn_cnn64_9.dll cudnn_engines_precompiled64_9.dll
cudnn_engines_runtime_compiled64_9.dll cudnn_graph64_9.dll cudnn_heuristic64_9.dll cudnn_ops64_9.dll
MSVCP140.dll VCRUNTIME140.dll VCRUNTIME140_1.dll
avcodec-58.dll avfilter-7.dll avformat-58.dll avutil-56.dll swresample-3.dll swscale-5.dll
libx264-163.dll libsoxr.dll libgomp-1.dll libwinpthread-1.dll zlib1.dll'''.split(), key=str.casefold))
HOST_DLLS = frozenset('''advapi32.dll bcrypt.dll bcryptprimitives.dll cfgmgr32.dll
combase.dll crypt32.dll cryptbase.dll cryptnet.dll dbghelp.dll devobj.dll drvstore.dll
dxcore.dll gdi32.dll gdi32full.dll imm32.dll kernel.appcore.dll kernel32.dll kernelbase.dll
msasn1.dll msvcp_win.dll msvcrt.dll ntdll.dll ole32.dll oleaut32.dll rpcrt4.dll sechost.dll
setupapi.dll shcore.dll shell32.dll shlwapi.dll ucrtbase.dll user32.dll uxtheme.dll
version.dll win32u.dll windows.storage.dll wldap32.dll wldp.dll ws2_32.dll wsock32.dll
secur32.dll winmm.dll wintrust.dll iphlpapi.dll normaliz.dll nvcuda.dll'''.split())
COMPILER_HASHES = {
    'nvrtc64_120_0.dll': 'ec2a860422b7d0396b45c29a1848faecc3788041ba52f166f0fc52c1b47e8245',
    'nvrtc-builtins64_128.dll': '96a28201dbffd4eff6dc93f310417b981bf7a141d46f2839036f5dbfd83b33ae',
}
CONVERTER_FILE_COUNT = 83
ROOT_RESOURCES = ('VERSION', 'LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md')


def canonical_resources(source, commit):
    """Read immutable Git blobs; checkout EOL conversion cannot affect payloads."""
    require(re.fullmatch(r'[0-9a-f]{40}', commit), 'full source commit required')
    listing = subprocess.check_output(['git', '-C', str(source), 'ls-tree', '-rz', '--full-tree', commit,
                                       '--', 'native/specs', *ROOT_RESOURCES])
    entries = []
    for item in filter(None, listing.split(b'\0')):
        metadata, name = item.split(b'\t', 1)
        mode, kind, oid = metadata.split()
        name = name.decode('utf-8')
        safe_relative(name)
        require(mode in (b'100644', b'100755') and kind == b'blob', 'regular tracked resource required')
        entries.append((name, oid))
    require(sum(n.startswith('native/specs/') for n, _ in entries) == CONVERTER_FILE_COUNT,
            'converter/spec allowlist must contain exactly 83 tracked files')
    require({n for n, _ in entries if not n.startswith('native/specs/')} == set(ROOT_RESOURCES),
            'required root resource missing from source commit')
    output = subprocess.check_output(['git', '-C', str(source), 'cat-file', '--batch'],
                                     input=b''.join(oid + b'\n' for _, oid in entries))
    stream = io.BytesIO(output)
    result = {}
    for name, oid in entries:
        actual, kind, size = stream.readline().split()
        require(actual == oid and kind == b'blob', 'Git resource identity mismatch')
        data = stream.read(int(size))
        require(len(data) == int(size) and stream.read(1) == b'\n', 'truncated Git resource')
        result[name] = data
    require(not stream.read(), 'unexpected Git resource output')
    return result


def public_metadata(value):
    """Reject recognizable private paths/credentials without echoing their values."""
    if isinstance(value, dict):
        for key, item in value.items():
            require(key.casefold() not in {'password', 'secret', 'token', 'access_token', 'authorization', 'credentials'},
                    'credential field forbidden in provenance')
            public_metadata(key)
            public_metadata(item)
    elif isinstance(value, list):
        for item in value:
            public_metadata(item)
    elif isinstance(value, str):
        require(not re.search(r'(?i)(?<![a-z0-9])[a-z]:[\\/]|\\\\|file://|(?:^|\s)/(?!/)[^\s]', value),
                'absolute local path forbidden in provenance')
        require(not re.search(r'https?://[^\s/]+@|-----BEGIN .*PRIVATE KEY|gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{70,}', value),
                'credential-like value forbidden in provenance')


def validate_build_info(build_info, commit):
    public_metadata(build_info)
    require(build_info['source_commit'] == commit, 'build source mismatch')
    config = build_info['configuration']
    require(config['build_type'] == 'Release', 'Release build required')
    require(all(config[x] is True for x in ('cuda', 'tokenizer', 'product_cli')), 'CUDA/tokenizer/product required')
    require('86' in [str(x) for x in config['gpu_architectures']], 'SM86 support required')
    require(config['cuda_runtime_linkage'] == 'Shared', 'shared CUDA runtime required')
    require(re.fullmatch(r'[0-9a-f]{64}', build_info['executable_sha256']), 'executable SHA256 required')
    toolchain = build_info['toolchain']
    for name in ('msvc', 'msvc_toolset', 'cmake', 'ninja', 'rust', 'cuda_toolkit', 'nvcc', 'cudnn'):
        require(isinstance(toolchain.get(name), str) and toolchain[name].strip(), 'missing toolchain field: ' + name)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + '\n', encoding='utf-8', newline='\n')


def safe_relative(name):
    require(isinstance(name, str) and name and '\\' not in name, 'invalid relative path')
    p = PurePosixPath(name)
    require(not p.is_absolute() and all(x not in ('', '.', '..') for x in name.split('/')), 'unsafe relative path: ' + name)
    for part in p.parts:
        require(not any(c in part for c in ':*?"<>|\t\r\n') and not part.endswith((' ', '.')), 'invalid Windows name: ' + name)
        require(part.split('.')[0].upper() not in {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(1, 10)), *(f'LPT{i}' for i in range(1, 10))}, 'reserved Windows name: ' + name)
    return p


def regular_file(root, name):
    rel = safe_relative(name)
    path = root.joinpath(*rel.parts)
    require(path.is_file(), 'missing file: ' + str(path))
    for part in (path, *path.parents):
        require(not part.is_symlink() and not (hasattr(part, 'is_junction') and part.is_junction()), 'reparse path rejected: ' + str(part))
        if part == root:
            break
    require(path.resolve().is_relative_to(root.resolve()), 'input escapes root')
    return path


def pe_imports(path):
    """Read x64 PE import and delay-import descriptors without executing the file."""
    with open(path, 'rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        def unpack(fmt, offset):
            require(0 <= offset <= len(data) - struct.calcsize(fmt), 'truncated PE: ' + str(path))
            return struct.unpack_from(fmt, data, offset)
        require(data[:2] == b'MZ', 'not PE: ' + str(path))
        pe, = unpack('<I', 0x3c)
        require(data[pe:pe + 4] == b'PE\0\0', 'invalid PE signature')
        machine, count = unpack('<HH', pe + 4)
        optsize, = unpack('<H', pe + 20)
        opt = pe + 24
        require(machine == 0x8664 and unpack('<H', opt)[0] == 0x20b, 'Windows x64 PE required')
        require(optsize >= 112 and count <= 96, 'invalid PE optional/section headers')
        imagebase, = unpack('<Q', opt + 24)
        headers, = unpack('<I', opt + 60)
        directories, = unpack('<I', opt + 108)
        sections = []
        for index in range(count):
            at = opt + optsize + index * 40
            virtual_size, rva, rawsize, raw = unpack('<IIII', at + 8)
            sections.append((rva, max(virtual_size, rawsize), raw, rawsize))

        def offset(rva, length=1):
            if rva < headers and rva + length <= min(headers, len(data)):
                return rva
            for base, span, raw, size in sections:
                if base <= rva < base + span:
                    delta = rva - base
                    require(delta + length <= size and raw + delta + length <= len(data), 'PE RVA outside file data')
                    return raw + delta
            raise ValueError('unmapped PE RVA')

        def dll_name(rva):
            at = offset(rva)
            end = data.find(b'\0', at, min(len(data), at + 512))
            require(end >= 0, 'unterminated PE import name')
            name = data[at:end].decode('ascii')
            require(name.lower().endswith('.dll') and all(c.isalnum() or c in '_.-' for c in name), 'invalid DLL import name')
            return name

        result = []
        for index, width, kind in ((1, 20, 'direct'), (13, 32, 'delay')):
            if directories <= index:
                continue
            require(112 + (index + 1) * 8 <= optsize, 'directory beyond optional header')
            start, size = unpack('<II', opt + 112 + index * 8)
            if start == 0 and size == 0:
                continue
            require(start and size >= width, 'invalid PE import directory')
            terminated = False
            for position in range(0, size - width + 1, width):
                values = unpack('<' + 'I' * (width // 4), offset(start + position, width))
                if not any(values):
                    terminated = True
                    break
                name_rva = values[3] if kind == 'direct' else values[1]
                if kind == 'delay' and not (values[0] & 1):
                    name_rva -= imagebase
                result.append({'name': dll_name(name_rva), 'kind': kind})
            require(terminated, 'unterminated PE import directory')
        return sorted(result, key=lambda row: (row['name'].lower(), row['kind']))


def audit_pe(files):
    local = {name.lower() for name in files}
    records = {}
    for name, path in sorted(files.items()):
        imports = pe_imports(path)
        for dep in imports:
            key = dep['name'].lower()
            host = key in HOST_DLLS or key.startswith(('api-ms-win-', 'ext-ms-win-'))
            require(key in local or host, f'unexpected/missing runtime dependency: {name} -> {dep["name"]}')
            dep['boundary'] = 'application' if key in local else ('driver' if key == 'nvcuda.dll' else 'Windows')
        records[name] = imports
    return records


def inventory(root):
    return [{'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)}
            for p in sorted(root.rglob('*')) if p.is_file()]


def write_zip(stage, target):
    require(not target.exists(), 'ZIP already exists')
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in sorted(stage.rglob('*')):
            if not path.is_file():
                continue
            info = zipfile.ZipInfo(stage.name + '/' + path.relative_to(stage).as_posix(), (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            with open(path, 'rb') as src, archive.open(info, 'w', force_zip64=True) as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
    target.with_name(target.name + '.sha256').write_text(digest(target) + '  ' + target.name + '\n', encoding='ascii')


def dependency_paths(deps, manifest):
    public_metadata(manifest)
    require(manifest['schema_version'] == 1, 'unsupported dependency manifest')
    runtime, resources = manifest['runtime'], manifest['resources']
    expected = set(DLLS) | {'vrhino-ffmpeg.exe'}
    require(len(runtime) == len(expected) and {x['path'] for x in runtime} == expected,
            'runtime allowlist must match exactly 39 DLLs and helper')
    all_records = runtime + resources
    names = [r['path'] for r in all_records]
    require(len({n.casefold() for n in names}) == len(names), 'case-colliding/duplicate payload paths')
    paths = {}
    for record in all_records:
        name = record['path']
        path = regular_file(deps, name)
        if record in resources:
            require(PurePosixPath(name).parts[0] in ('licenses', 'sources'), 'resource outside allowlisted trees')
            # GNU license texts conventionally use .LIB; these are not import libraries.
            license_text = path.name in ('COPYING.LIB', 'COPYING3.LIB') and 'GNU' in path.read_text(encoding='utf-8')[:256]
            require(license_text or path.suffix.lower() not in
                    ('.exe', '.dll', '.pdb', '.obj', '.lib', '.mp4', '.wav', '.avi', '.mov', '.vrm',
                     '.safetensors', '.pt', '.pth', '.ckpt', '.log', '.pyc'), 'forbidden resource payload: ' + name)
        require(digest(path) == record['sha256'], 'dependency hash mismatch: ' + name)
        paths[name] = path
    for record in runtime:
        require(record.get('version') and record.get('origin') and record.get('license'), 'runtime provenance incomplete: ' + record['path'])
        require(record.get('license_files') and all(x in paths and x.startswith('licenses/') for x in record['license_files']), 'missing runtime license closure')
    actual = {p.relative_to(deps).as_posix() for p in deps.rglob('*') if p.is_file()}
    require(actual == set(names) | {'dependency-manifest.json'}, 'unexpected dependency-root file')
    for name, sha in COMPILER_HASHES.items():
        require(digest(paths[name]) == sha, 'NVRTC identity differs from qualified closure')
    closure = manifest['dynamic_closure']
    require(closure['nvrtc_required'] is True and closure['nvjitlink_included'] is False
            and closure['nvrtc_component_version'] == '12.8.61', 'dynamic closure provenance mismatch')
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('build-dir', 'dependency-root', 'build-info', 'output', 'zip'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    source = Path(__file__).resolve().parents[1]
    git = lambda *a: subprocess.check_output(['git', '-C', str(source), *a], text=True).strip()
    require(git('rev-parse', 'HEAD') == args.source_commit, 'source HEAD mismatch')
    require(not git('status', '--porcelain', '--', 'native'), 'native source must match HEAD, including untracked inputs')
    build, deps = args.build_dir.resolve(strict=True), args.dependency_root.resolve(strict=True)
    output, zip_path = args.output.absolute(), args.zip.absolute()
    require(not output.exists() and not zip_path.exists() and not zip_path.with_name(zip_path.name + '.sha256').exists(), 'output must be new')
    require(not zip_path.is_relative_to(output), 'ZIP must be outside stage')
    build_info = json.loads(args.build_info.read_text(encoding='utf-8-sig'))
    validate_build_info(build_info, args.source_commit)
    manifest = json.loads(regular_file(deps, 'dependency-manifest.json').read_text(encoding='utf-8-sig'))
    paths = dependency_paths(deps, manifest)
    runtime, resources = manifest['runtime'], manifest['resources']
    exe = regular_file(build, 'vrhino.exe')
    require(digest(exe) == build_info['executable_sha256'], 'build executable hash mismatch')
    binaries = {r['path']: paths[r['path']] for r in runtime}
    binaries['vrhino.exe'] = exe
    imports = audit_pe(binaries)
    source_resources = canonical_resources(source, args.source_commit)
    if args.check:
        print('Windows packaging inputs: PASS; runtime execution qualification remains separate')
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.vrhino-stage-', dir=output.parent))
    try:
        for name, path in {**paths, 'vrhino.exe': exe}.items():
            dest = temp / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            require(digest(dest) == digest(path), 'copy changed bytes: ' + name)
        for name, data in source_resources.items():
            relative = ('share/vrhino/converters/' + name.removeprefix('native/specs/')) if name.startswith('native/specs/') else name
            dest = temp / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
        version = (temp / 'VERSION').read_text(encoding='utf-8').strip()
        env = {k: os.environ[k] for k in ('SystemRoot', 'WINDIR', 'TEMP', 'TMP', 'USERPROFILE', 'LOCALAPPDATA') if k in os.environ}
        env['PATH'] = str(temp) + ';' + env.get('SystemRoot', 'C:\\Windows') + '\\System32;' + env.get('SystemRoot', 'C:\\Windows')
        started = subprocess.run([str(temp / 'vrhino.exe'), '--version'], cwd=temp, env=env, capture_output=True, text=True, timeout=30, check=True)
        require('Git HEAD: ' + args.source_commit in started.stdout.splitlines()
                and 'VRhino ' + version in started.stdout.splitlines(), 'built product version/source identity mismatch')
        provenance = {'schema_version': 1, 'kind': 'unpublished-windows-cuda-candidate', 'source_commit': args.source_commit,
                      'native_tree': git('rev-parse', 'HEAD:native'), 'version': version,
                      'build': build_info, 'assembler_sha256': digest(Path(__file__)),
                      'runtime': runtime, 'resource_origins': resources, 'pe_imports': imports,
                      'dynamic_closure': manifest['dynamic_closure'], 'layout': 'EXEs and 39 DLLs at root; executable-relative share/vrhino/converters',
                      'qualification': {'clean_machine': 'NOT_RUN', 'model_inference_on_this_candidate': 'NOT_RUN'},
                      'files': inventory(temp)}
        write_json(temp / 'SOURCE-BUILD.json', provenance)
        sums = inventory(temp)
        (temp / 'SHA256SUMS').write_text(''.join(r['sha256'] + '  ' + r['path'] + '\n' for r in sums), encoding='utf-8', newline='\n')
        require(not output.exists(), 'output appeared during assembly')
        temp.rename(output)
    finally:
        if temp.exists():
            require(temp.resolve().parent == output.parent.resolve() and temp.name.startswith('.vrhino-stage-'), 'unsafe temporary cleanup')
            shutil.rmtree(temp)
    write_zip(output, zip_path)
    print(json.dumps({'stage': str(output), 'zip': str(zip_path), 'zip_bytes': zip_path.stat().st_size,
                      'zip_sha256': digest(zip_path), 'files': len(inventory(output)), 'dlls': len(DLLS)}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as error:
        raise SystemExit('Windows package rejected: ' + str(error))
