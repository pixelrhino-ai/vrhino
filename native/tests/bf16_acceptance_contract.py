#!/usr/bin/env python3
"""Fail-closed scope/classification wrapper for the generic v3 research gate."""
import argparse
import json
from pathlib import Path
import torch
from bf16_operator_acceptance import SCHEMA, compare


def qualify(actual, reference, declaration):
    if set(declaration) != {'schema', 'scope', 'checkpoints'} or declaration['schema'] != SCHEMA:
        raise ValueError('Unknown acceptance declaration')
    required = {'operator': 'operator_output', 'architecture': 'observable_output'}.get(declaration['scope'])
    if required is None:
        raise ValueError('Unknown qualification scope')
    checkpoints = declaration['checkpoints']
    if not checkpoints or actual.keys() != reference.keys() or actual.keys() != checkpoints.keys():
        raise ValueError('Incomplete checkpoint set')
    results = {}
    seen = False
    for name, item in checkpoints.items():
        if set(item) != {'category', 'precision', 'operator', 'reason'} or not isinstance(item['reason'], str) or not item['reason'].strip():
            raise ValueError('Incomplete classification')
        category, precision = item['category'], item['precision']
        if category not in ['operator_output', 'observable_output', 'internal_diagnostic', 'exact_control']:
            raise ValueError('Unknown boundary category')
        if precision not in ['bf16', 'fp32', 'exact'] or (category == 'exact_control') != (precision == 'exact'):
            raise ValueError('Invalid precision category')
        if precision != 'exact' and actual[name].dtype != {'bf16': torch.bfloat16, 'fp32': torch.float32}[precision]:
            raise ValueError('Declared precision mismatch')
        operator = item['operator']
        if category == 'operator_output':
            if operator not in ['linear', 'attention', 'softmax', 'layernorm', 'ffn', 'residual', 'rms', 'gelu']:
                raise ValueError('Unknown operator')
            gate = 'operator_bf16' if precision == 'bf16' else 'f32_semantic'
        else:
            if operator is not None:
                raise ValueError('Operator scope mismatch')
            gate = {'observable_output': 'observable', 'internal_diagnostic': 'diagnostic', 'exact_control': 'exact'}[category]
        seen |= category == required
        results[name] = compare(actual[name], reference[name], gate, operator)
    if not seen:
        raise ValueError('No required strict output for scope')
    return dict(schema=SCHEMA, passed=all(row['passed'] for row in results.values()),
                declaration=declaration, checkpoints=results)


if __name__ == '__main__':
    from bf16_policy_reference import read
    parser = argparse.ArgumentParser()
    for key in ['actual', 'reference', 'declaration', 'output']:
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    result = qualify(read(args.actual), read(args.reference), json.loads(Path(args.declaration).read_text()))
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    raise SystemExit(0 if result['passed'] else 1)
