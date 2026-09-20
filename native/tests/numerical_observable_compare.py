#!/usr/bin/env python3
"""Research comparison of predeclared outputs; preserve every internal metric.

Primitive qualification is independent and mandatory. This tool never turns
its diagnostic results into evidence that an operator passed a local gate.
"""
import argparse
import json
from pathlib import Path
import torch
from f32_attention_reference import read, metrics


def compare(expected, actual, observables):
    if not observables or len(set(observables)) != len(observables):
        raise ValueError('Observable set must be nonempty and unique')
    if expected.keys() != actual.keys() or not set(observables) <= expected.keys():
        raise ValueError('Trace/observable key mismatch')
    rows = []
    for name in sorted(expected):
        e, a = expected[name], actual[name]
        if e.shape != a.shape or e.dtype != a.dtype:
            raise ValueError('Trace shape/dtype mismatch: '+name)
        if not bool(torch.isfinite(e).all() and torch.isfinite(a).all()):
            raise ValueError('Nonfinite checkpoint: '+name)
        rows.append(dict(name=name, role='observable' if name in observables else 'diagnostic',
                         **metrics(a,e)))
    return dict(passed=all(x['passed'] for x in rows if x['role']=='observable'),
                atol=2e-5, rtol=2e-5, checkpoints=rows,
                diagnostic_exceedances=[x['name'] for x in rows if x['role']=='diagnostic' and not x['passed']])


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--expected',required=True);p.add_argument('--actual',required=True)
    p.add_argument('--observable',action='append',required=True);p.add_argument('--output',required=True)
    args=p.parse_args()
    result=compare(read(args.expected),read(args.actual),args.observable)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    for row in result['checkpoints']:
        if row['role']=='observable': print(json.dumps(row))
    print('diagnostic exceedances (retained):',len(result['diagnostic_exceedances']))
    raise SystemExit(0 if result['passed'] else 1)
