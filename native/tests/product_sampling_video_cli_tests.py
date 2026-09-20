"""Fail-closed Product complete-sampling declarations; no model payload or GPU."""
import argparse
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--cli', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
base = dict(schema='vrhino.sampling-video-check.v1', precision='fp32', max_steps=40,
            device_budget_bytes=60 << 30, host_budget_bytes=200 << 30,
            fps=8, media_range=[-1, 1])

def call(name, options, command='sampling-video-check', existing=False):
    document = a.output / (name + '.json')
    document.write_text(json.dumps(options))
    out = a.output / (name + '-output')
    if existing:
        out.mkdir(exist_ok=True)
        (out / 'marker').write_text('preserve')
    result = subprocess.run([a.cli, command, 'absent-manifest', 'absent-resources',
                             'absent-request', str(document), str(out)],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode != 0, result.stdout
    assert ((out / 'marker').read_text() == 'preserve') if existing else not out.exists()
    return result.stderr

cases = [
    ('schema', 'unknown', 'Invalid sampling-step-check declaration'),
    ('precision', 'bf16', 'explicit FP32 qualification'),
    ('max_steps', 0, 'two to 64 steps'),
    ('max_steps', 1, 'two to 64 steps'),
    ('max_steps', 65, 'two to 64 steps'),
    ('max_steps', -1, 'two to 64 steps'),
    ('device_budget_bytes', 0, 'Invalid sampling resource budget'),
    ('host_budget_bytes', 0, 'Invalid sampling resource budget'),
    ('fps', 0, 'Invalid media fps'), ('fps', 121, 'Invalid media fps'),
    ('media_range', [1, -1], 'Invalid media range'),
    ('media_range', [0, 0], 'Invalid media range'),
    ('media_range', [0], 'Expected declared media range'),
    ('binding', 1, 'Invalid sampling-step-check declaration'),
    ('guidance', 4, 'Invalid sampling-step-check declaration'),
    ('model', 'fixture', 'Invalid sampling-step-check declaration'),
    ('latent_bundle', 'fixture', 'Invalid sampling-step-check declaration'),
    ('completed_step_limit', 3, 'Invalid sampling-step-check declaration'),
]
for i, (key, value, error) in enumerate(cases):
    assert error in call('negative-' + str(i), dict(base, **{key: value}))
    print('PASS reject', key, value)
assert 'already exists' in call('existing', base, existing=True)
# Closed entrypoints: neither declaration can escape its own execution scope.
assert 'Invalid sampling-step-check declaration' in call('wrong-entry', base, 'sampling-step-check')
prefix = dict(schema='vrhino.sampling-step-check.v3', precision='fp32',
              completed_step_limit=27, device_budget_bytes=60 << 30, host_budget_bytes=200 << 30)
assert 'Invalid sampling-step-check declaration' in call('prefix-entry', prefix)
# Valid complete declarations reach the same ordinary package rejection.
errors = [call('valid-' + str(bound), dict(base, max_steps=bound)) for bound in [2, 40, 64]]
assert errors[0] and all(e == errors[0] for e in errors)
assert 'sampling-step-check declaration' not in errors[0] and 'two to 64' not in errors[0]
print('PASS 21 rejection/preservation cases; 3 valid declarations reach missing-package boundary')

# New declared-policy versions must not reinterpret legacy options or widen
# execution scope. Missing packages still fail before creating evidence/GPU work.
declared = dict(base, schema='vrhino.sampling-video-check.v2', precision='bf16',
                precision_policy_artifact='precision-policy')
assert call('declared-valid', declared) == errors[0]
for key, value, error in [
        ('precision', 'fp32', 'requires BF16 mode'),
        ('precision', 'fp16', 'requires BF16 mode'),
        ('precision_policy_artifact', '', 'artifact ID'),
        ('binding', 1, 'Invalid sampling-step-check declaration'),
        ('tolerance', 1.0, 'Invalid sampling-step-check declaration')]:
    assert error in call('declared-' + key + '-' + str(value), dict(declared, **{key: value}))
missing = dict(declared)
del missing['precision_policy_artifact']
assert 'Invalid sampling-step-check declaration' in call('declared-missing-policy', missing)
assert 'Invalid sampling-step-check declaration' in call('declared-wrong-entry', declared, 'sampling-step-check')
declared_prefix = dict(prefix, schema='vrhino.sampling-step-check.v4', precision='bf16',
                       precision_policy_artifact='precision-policy')
assert call('declared-prefix-valid', declared_prefix, 'sampling-step-check') == errors[0]
assert 'Invalid sampling-step-check declaration' in call('declared-prefix-wrong-entry', declared_prefix)
assert 'one to 32' in call('declared-prefix-bound', dict(declared_prefix, completed_step_limit=33), 'sampling-step-check')
print('PASS declared-policy closed schemas, precision selection, bounded scope, legacy isolation')
