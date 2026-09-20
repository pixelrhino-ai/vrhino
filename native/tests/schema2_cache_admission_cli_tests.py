#!/usr/bin/env python3
"""Installed/local schema2 admission equivalence; ordinary run remains fail closed.

Uses the small synthetic fixture emitted by schema2-request-wiring-tests.
No GPU execution or external model asset is needed.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

p = argparse.ArgumentParser()
p.add_argument('--binary', required=True, type=Path)
fixtures = p.add_mutually_exclusive_group(required=True)
fixtures.add_argument('--fixture', type=Path)
fixtures.add_argument('--fixture-builder', type=Path)
p.add_argument('--specs', type=Path)
p.add_argument('--output', type=Path)
a = p.parse_args()
if a.output is None:
    a.output = Path(tempfile.mkdtemp(prefix='vrhino-schema2-cache-cli-')) / 'results'
a.output.mkdir(parents=True, exist_ok=False)
records = []
if a.fixture_builder:
    assert a.specs is not None, '--fixture-builder requires --specs'
    a.fixture = a.output / 'fixture'
    fixture_result = subprocess.run([str(a.fixture_builder), str(a.fixture), str(a.specs)],
                                    capture_output=True, text=True, timeout=120)
    (a.output / 'fixture.log').write_text(fixture_result.stdout + fixture_result.stderr)
    assert fixture_result.returncode == 0, fixture_result.stdout + fixture_result.stderr

def run(name, args, success=True):
    command = [str(a.binary), '--cache-root', str(a.fixture / 'cache'), *args]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    record = dict(name=name, command=command, returncode=result.returncode,
                  stdout=result.stdout, stderr=result.stderr)
    records.append(record)
    assert (result.returncode == 0) == success, record
    return result

cached = json.loads(run('cached-preflight', ['preflight', 'test/catalog:1']).stdout)
local = json.loads(run('local-preflight', ['preflight', str(a.fixture / 'vrhino-model.json'),
                                         str(a.fixture / 'local.json')]).stdout)
assert cached['vrm_path'] != local['vrm_path']
assert {k: v for k, v in cached.items() if k != 'vrm_path'} == {
    k: v for k, v in local.items() if k != 'vrm_path'}
assert cached['structurally_runnable'] and not cached['numerically_qualified']
assert cached['device_upload_bytes'] == 0 and cached['numerical_status'] == 'HOLD'
request = str(a.fixture / 'request.json')
cached_request = json.loads(run('cached-dry-run', ['dry-run', 'test/catalog:1', request]).stdout)
local_request = json.loads(run('local-dry-run', ['dry-run', str(a.fixture / 'vrhino-model.json'),
                                             str(a.fixture / 'local.json'), request]).stdout)
assert cached_request == local_request
assert cached_request['native_tokenization_executed']
assert not cached_request['conditioning_executed'] and not cached_request['denoising_executed']
assert not cached_request['numerically_qualified']
media = a.output / 'must-not-exist.mp4'
rejected = run('ordinary-run-retains-HOLD', ['run', 'test/catalog:1', '--prompt', 'hello',
                                         '--output', str(media)], False)
assert 'numerical qualification is HOLD' in rejected.stderr, rejected.stderr
assert not media.exists()
resources = a.output / 'run-resources.json'
resources.write_text(json.dumps({'weight_cache_budget_bytes': 4096}))
rejected = run('resource-control-cannot-bypass-HOLD',
               ['run', 'test/catalog:1', '--prompt', 'hello', '--resources', str(resources),
                '--output', str(media)], False)
assert 'numerical qualification is HOLD' in rejected.stderr, rejected.stderr
assert not media.exists()
resources.write_text(json.dumps({'weight_cache_budget_bytes': 0}))
rejected = run('invalid-run-resource', ['run', 'test/catalog:1', '--prompt', 'hello',
                                     '--resources', str(resources)], False)
assert 'positive integer' in rejected.stderr, rejected.stderr
resources.write_text('{}')
run('duplicate-run-resource-flag', ['run', 'test/catalog:1', '--resources', str(resources),
                                  '--resources', str(resources)], False)
run('unknown-cached-package', ['preflight', 'test/absent:1'], False)
bad_request = a.output / 'override.json'
bad_request.write_text(json.dumps({**json.loads(Path(request).read_text()), 'guidance': 5}))
run('cached-request-rejects-override', ['dry-run', 'test/catalog:1', str(bad_request)], False)
(a.output / 'results.json').write_text(json.dumps(dict(status='PASS', cases=records,
    scope='synthetic schema2 CLI admission/preparation; numerical execution rejected'), indent=2))
print(f'PASS {len(records)} CLI cases; cache/local equivalence; ordinary run HOLD; no GPU execution')
