#!/usr/bin/env python3
"""Research-only native/reference initial-state audit; never runs a model.

Reference: CUDA float32 randn, followed by the requested storage conversion.
Same seed is tested separately from an explicit same-input tensor handoff.
No implementation, precision policy or tolerance is changed by this probe.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from bf16_policy_reference import read


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.cuda.get_device_properties(0)
    # A test coordinate derived from device capacity, not a model boundary.
    capacity = device.multi_processor_count * device.max_threads_per_multi_processor
    counts = [17, capacity - 1, capacity, capacity + 1, 2 * capacity + 1, 2096640]
    rows = []
    for seed in [5701, 173]:
        for count in counts:
            for precision, dtype in [('f32', torch.float32), ('bf16', torch.bfloat16)]:
                label = f's{seed}_n{count}_{precision}'
                path = args.output / (label + '.bundle')
                command = [str(args.binary), str(count), str(seed), precision, str(path)]
                with (args.output / (label + '.log')).open('w') as log:
                    subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=60)
                values = read(path)
                assert torch.equal(values['first'], values['replay']), 'native replay is not deterministic'
                generator = torch.Generator(device='cuda').manual_seed(seed)
                steps = []
                for key in ['first', 'second']:
                    reference = torch.randn(count, dtype=torch.float32, device='cuda', generator=generator).to(dtype).cpu()
                    actual = values[key]
                    assert actual.dtype == dtype and actual.shape == reference.shape
                    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
                    mismatch = actual != reference
                    delta = actual.double() - reference.double()
                    # Explicit common-input replay supplies the captured native
                    # tensor to the reference; it does not regenerate by seed.
                    handed_off = actual.cuda().cpu()
                    assert torch.equal(handed_off, actual), 'same-input reference handoff lost data'
                    steps.append(dict(step=key, exact=not bool(mismatch.any()),
                        first_different_index=int(mismatch.nonzero()[0]) if mismatch.any() else None,
                        different_elements=int(mismatch.sum()), max_abs=float(delta.abs().max()),
                        mean_abs=float(delta.abs().mean()),
                        relative_l2=float(delta.norm() / reference.double().norm()),
                        reference_offset=int(generator.get_offset()),
                        native_offset=int(values['offset_after_' + key]),
                        same_input_handoff_exact=True))
                rows.append(dict(seed=seed, count=count, dtype=precision,
                    command=command, native_bundle_sha256=sha(path), native_replay_exact=True, steps=steps))
    result = dict(route='CURRENT_PRODUCTION_DEFAULT', scope='RNG and tensor handoff only; no denoiser execution',
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        device=device.name, tested_capacity=capacity, binary_sha256=sha(args.binary),
        reference_contract='CUDA F32 randn then storage cast',
        same_seed_reference_exact=all(s['exact'] for r in rows for s in r['steps']),
        native_replay_exact=True, same_input_handoff_exact=True, cases=rows)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != 'cases'}, indent=2))


if __name__ == '__main__':
    main()
