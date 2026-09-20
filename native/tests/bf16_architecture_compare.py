#!/usr/bin/env python3
"""Apply a predeclared v3 architecture gate, preserving all diagnostics."""
import argparse
import json
from pathlib import Path
import torch
from bf16_policy_reference import read
from bf16_acceptance_contract import qualify
from short_chain_compare import validate_continuity


def compare(actual, reference, declaration, chain=None):
    reference = dict(reference)
    aliases = []
    continuity = None
    if chain:
        for i in range(chain['steps']):
            for branch in [0, 1]:
                key = f'step.{i}.branch.{branch}.prediction'
                canonical = f'step.{i}.prediction.{branch}'
                if key not in reference or not torch.equal(reference[key], reference[canonical]):
                    raise ValueError('Reference prediction alias mismatch')
                if reference[key].dtype != reference[canonical].dtype or reference[key].shape != reference[canonical].shape:
                    raise ValueError('Reference prediction alias metadata mismatch')
                del reference[key]; aliases.append(key)
        continuity = dict(native=validate_continuity(actual, chain),
                          reference=validate_continuity(reference, chain))
    result = qualify(actual, reference, declaration)
    result.update(exact_aliases=aliases, continuity=continuity)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ['actual', 'reference', 'declaration', 'output']: parser.add_argument('--'+key, required=True)
    parser.add_argument('--chain-contract')
    args = parser.parse_args()
    result = compare(read(args.actual), read(args.reference), json.loads(Path(args.declaration).read_text()),
                     json.loads(Path(args.chain_contract).read_text()) if args.chain_contract else None)
    Path(args.output).write_text(json.dumps(result, indent=2)+'\n')
    failed = {key:value for key,value in result['checkpoints'].items() if not value['passed']}
    print('CHECKPOINTS', len(result['checkpoints']), 'FAILED_STRICT', len(failed))
    for key,value in failed.items(): print(key, 'max_abs', value['max_abs'], 'relative_l2', value['relative_l2'], 'failing', value['pointwise_bad'])
    raise SystemExit(0 if result['passed'] else 1)
