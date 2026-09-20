"""Evaluation admission is explicit and never overrides ordinary run qualification."""
import argparse
import json
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--cli', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
base = dict(schema='vrhino.product-evaluation.v1', precision='bf16',
            precision_policy_artifact='policy', mode='prefix', max_steps=1,
            device_budget_bytes=8 << 30, host_budget_bytes=2 << 30,
            max_output_bytes=128 << 20, max_latent_elements=4194304,
            max_video_elements=100000000, timeout_seconds=5,
            fps=16, media_range=[-1, 1])
for i, (key, value) in enumerate([
    ('schema', 'unknown'), ('precision', 'fp32'), ('mode', 'run'),
    ('model', 'fixture'), ('binding', 0), ('tolerance', 1.0),
    ('precision_override', True), ('trace_enabled', True),
    ('max_steps', 0), ('timeout_seconds', 0), ('max_output_bytes', 0),
    ('max_latent_elements', 0), ('max_video_elements', 0),
    ('media_range', [1, -1]), ('precision_policy_artifact', '')]):
    doc = a.output / f'invalid-{i}.json'
    doc.write_text(json.dumps(dict(base, **{key: value})))
    out = a.output / f'invalid-{i}-output'
    result = subprocess.run([a.cli, 'evaluate', 'missing', 'missing', 'missing',
                             str(doc), str(out)], capture_output=True, text=True, timeout=15)
    assert result.returncode != 0 and not out.exists(), result.stdout
    assert 'Usage:' not in result.stderr, result.stderr

doc = a.output / 'valid.json'
doc.write_text(json.dumps(base))
out = a.output / 'preserved'
out.mkdir(exist_ok=True)
(out / 'marker').write_text('keep')
r = subprocess.run([a.cli, 'evaluate', 'missing', 'missing', 'missing', str(doc), str(out)],
                   capture_output=True, text=True, timeout=15)
assert r.returncode != 0 and 'already exists' in r.stderr
assert (out / 'marker').read_text() == 'keep'
out = a.output / 'missing-package'
if out.exists():
    import shutil
    shutil.rmtree(out)
r = subprocess.run([a.cli, 'evaluate', 'missing', 'missing', 'missing', str(doc), str(out)],
                   capture_output=True, text=True, timeout=15)
assert r.returncode != 0, r.stdout
supervision = json.loads((out / 'supervision.json').read_text())
assert supervision['status'] == 'HOLD'
assert supervision['production_numerically_qualified'] is False
assert supervision['reference_numerically_qualified'] is False
assert not (out / 'evaluation.mp4').exists()
print('PASS evaluation CLI: closed options, explicit scope, output preservation, package failure')
