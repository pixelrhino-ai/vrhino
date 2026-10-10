#!/usr/bin/env python3
"""Opt-in Stage 2A fixture: Native source pull, isolated HTTPS package pull, full run.

Python only observes processes and serves fixture bytes. Native owns acquisition,
conversion, admission, text conditioning, sampling, decoding and media encoding.
No existing model cache is used as an installation shortcut.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def put(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def gpu():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version,memory.used',
                                   '--format=csv,noheader'], text=True).strip()


def stop_owned_process(process):
    # This Linux fixture owns the whole process group, including media helpers.
    # Terminate the group even if its leader already exited after an observer error.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=30)
    # A helper may ignore TERM and outlive a leader that exited promptly.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def main():
    parser = argparse.ArgumentParser()
    for name in ('cli', 'source', 'package', 'specs', 'work', 'encoder'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.work.exists(), 'fixture work must be fresh'
    args.work.mkdir(parents=True)
    evidence = args.work / 'evidence'
    evidence.mkdir()
    cli_hash = digest(args.cli)
    manifest_path = args.package / 'vrhino-model.json'
    manifest = json.loads(manifest_path.read_text())
    identity = manifest['identity']
    reference = f"{identity['namespace']}/{identity['name']}:{identity['version']}"
    assert reference == 'review/ltx-video-v0.9.1-schema2:0.2.0'
    rows = []

    def run(name, command, timeout=1200):
        before = gpu()
        started = time.monotonic()
        samples = []
        with (evidence / (name + '.log')).open('wb') as log:
            environment = os.environ.copy()
            environment.pop('VRHINO_CONVERTER_SPEC_ROOT', None)
            process = subprocess.Popen([str(x) for x in command], stdout=log, stderr=subprocess.STDOUT,
                                       env=environment, start_new_session=True)
            try:
                while process.poll() is None:
                    samples.append({'elapsed_seconds': round(time.monotonic()-started, 3), 'gpu': gpu()})
                    if time.monotonic()-started > timeout:
                        raise RuntimeError(f'{name} timed out without reduced workload')
                    time.sleep(1)
            finally:
                stop_owned_process(process)
        row = {'name': name, 'command': [str(x) for x in command], 'exit_code': process.returncode,
               'pid': process.pid, 'seconds': round(time.monotonic()-started, 3),
               'executable_sha256': cli_hash, 'gpu_before': before, 'gpu_after': gpu()}
        rows.append(row)
        put(evidence / 'native-processes.json', rows)
        put(evidence / (name + '-gpu.json'), samples)
        assert digest(args.cli) == cli_hash, 'binary changed'
        assert process.returncode == 0, f'{name} failed; preserve evidence'
        print(name, 'PASS', row['seconds'], flush=True)
        return (evidence / (name + '.log')).read_text()

    source_cache = args.work / 'source-pull-cache'
    plan = json.loads((args.specs / 'ltx_schema2_v1/source-plan.json').read_text())
    seeded = []
    for artifact in plan['artifacts']:
        original = args.source / artifact['local_path']
        assert original.is_file() and not original.is_symlink()
        assert original.stat().st_size == artifact['size']
        destination = source_cache / 'sources/blobs/sha256' / artifact['sha256'][:2] / artifact['sha256']
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.link(original, destination)
        seeded.append({'source_path': str(original), 'source_blob': str(destination),
                       'size': artifact['size'], 'declared_sha256': artifact['sha256']})
    assert not (source_cache / 'models').exists() and not (source_cache / 'blobs').exists()
    put(evidence / 'initial-caches.json', {'source_pull_models_empty': True,
        'source_pull_product_CAS_empty': True, 'source_only_preseed': seeded,
        'verification_owner': 'Native LocalSourceCache acquire and LTX converter'})
    text = run('source-pull', [args.cli, '--cache-root', source_cache, 'pull', reference])
    assert 'Converter invoked: yes' in text and 'Conversion performed: yes' in text, text
    assert 'Source downloaded: 0 B (0 bytes)' in text, text
    installed = source_cache / 'models' / identity['namespace'] / identity['name'] / identity['version'] / 'vrhino-model.json'
    assert json.loads(installed.read_text()) == manifest
    # The standalone Package and source pull generated the same fixed artifact.
    # Merge this task's redundant fresh VRM storage only after verified CAS identity.
    runtime = next(x for x in manifest['artifacts'] if x['id'] == 'runtime')
    source_vrm = source_cache / 'blobs/sha256' / runtime['sha256'][:2] / runtime['sha256']
    assert digest(source_vrm) == runtime['sha256']
    assert digest(args.package / runtime['path']) == runtime['sha256']
    temporary = args.package / 'model.vrm.dedup'
    os.link(source_vrm, temporary)
    os.replace(temporary, args.package / runtime['path'])

    registry = args.work / 'registry'
    artifact_root = registry / 'artifacts'
    artifact_root.mkdir(parents=True)
    for artifact in manifest['artifacts']:
        os.link(args.package / artifact['path'], artifact_root / artifact['sha256'])
    cert, key = args.work / 'certificate.pem', args.work / 'key.pem'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
                    '-subj', '/CN=localhost', '-addext', 'subjectAltName=DNS:localhost,IP:127.0.0.1',
                    '-keyout', str(key), '-out', str(cert)], check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    port_file = args.work / 'port'
    server = subprocess.Popen([sys.executable, str(Path(__file__).with_name('registry_fixture_server.py')),
        '--root', str(registry), '--cert', str(cert), '--key', str(key),
        '--log', str(evidence / 'https-requests.jsonl'), '--port-file', str(port_file)], start_new_session=True)
    try:
        for _ in range(100):
            if port_file.exists(): break
            assert server.poll() is None, 'fixture server failed'
            time.sleep(.05)
        base = 'https://localhost:' + port_file.read_text().strip()
        model_dir = registry / 'v1/models' / identity['namespace'] / identity['name'] / identity['version']
        model_dir.mkdir(parents=True)
        os.link(manifest_path, model_dir / 'vrhino-model.json')
        model_url = f"{base}/v1/models/{identity['namespace']}/{identity['name']}/{identity['version']}"
        put(model_dir / 'index.json', {'registry_schema_version': 1, 'identity': reference,
            'manifest': {'url': model_url + '/vrhino-model.json', 'size': manifest_path.stat().st_size,
                         'sha256': digest(manifest_path)},
            'artifacts': [{'id': a['id'], 'url': base + '/artifacts/' + a['sha256']} for a in manifest['artifacts']]})
        registry_cache = args.work / 'registry-pull-cache'
        empty_specs = args.work / 'empty-specs'
        empty_specs.mkdir()
        assert not registry_cache.exists()
        common = [args.cli, '--cache-root', registry_cache, '--registry', base, '--ca-file', cert,
                  '--converter-spec-root', empty_specs]
        run('registry-pull', common + ['pull', reference])
        # All bytes came through Native RegistryClient; no model CAS preseed.
        verified = []
        for artifact in manifest['artifacts']:
            path = registry_cache / 'blobs/sha256' / artifact['sha256'][:2] / artifact['sha256']
            assert path.stat().st_size == artifact['size'] and digest(path) == artifact['sha256']
            verified.append(artifact)
        put(evidence / 'registry-artifacts.json', {'initial_model_cache_empty': True,
            'initial_product_CAS_empty': True, 'artifacts': verified,
            'scope': 'isolated loopback HTTPS fixture with private CA; no public distribution qualification'})
        run('registry-info', common + ['info', reference])
        run('registry-pull-repeat', common + ['pull', reference])
        output = args.work / 'schema2-stage2a.mp4'
        text = run('product-run', common + ['run', reference, '--prompt',
            'A red cube rests on a table in soft daylight.', '--seed', '5703',
            '--output', output, '--encoder', args.encoder, '--debug'])
        assert text.splitlines().count('Sampling') == 41, 'one stage entry plus 40 completion callbacks required'
        for expected in ('Encoding prompt', 'Decoding video', 'Encoding MP4', 'Done',
                         'Resolution: 704x480', 'Frames: 121', 'FPS: 25'):
            assert expected in text, expected
        assert ' 0 MiB' in gpu(), 'GPU resources remain'
        assert output.stat().st_size > 0
        put(evidence / 'smoke-result.json', {'status': 'PASS', 'reference': reference,
            'manifest_sha256': digest(manifest_path), 'cli_sha256': cli_hash,
            'sampling_stage_entries': 1, 'completed_step_callbacks': 40,
            'output_bytes': output.stat().st_size, 'output_sha256': digest(output),
            'gpu_after': gpu(), 'public_distribution_qualified': False})
        print('STAGE2A_NATIVE_SOURCE_PULL_REGISTRY_PULL_FULL_RUN=PASS', flush=True)
    finally:
        stop_owned_process(server)


if __name__ == '__main__':
    main()
