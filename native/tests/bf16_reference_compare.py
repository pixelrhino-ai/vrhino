#!/usr/bin/env python3
"""Frozen role-based numerical gate. Internal metrics are never dropped."""
import argparse,json
from pathlib import Path
import torch
from bf16_policy_reference import read


def compare(actual,expected,roles):
    if actual.keys()!=expected.keys() or roles.keys()!=expected.keys():raise ValueError('Trace/role keys mismatch')
    rows=[]
    for name,e in sorted(expected.items()):
        a=actual[name];role=roles[name]
        if role not in ['bf16_path','local_f32','diagnostic','exact']:raise ValueError('Unknown role')
        if a.shape!=e.shape or a.dtype!=e.dtype:raise ValueError('Shape/dtype mismatch '+name)
        if not bool(torch.isfinite(a).all() and torch.isfinite(e).all()):raise ValueError('Nonfinite '+name)
        x=a.double();y=e.double();d=(x-y).abs();atol=rtol=2e-5 if role=='local_f32' else .005
        bad=int((d>atol+rtol*y.abs()).sum())
        if role=='exact':bad=int((a!=e).sum())
        rows.append(dict(name=name,role=role,shape=list(a.shape),dtype=str(a.dtype),max_abs=float(d.max()),mean_abs=float(d.mean()),max_relative=float((d/y.abs().clamp_min(1e-8)).max()),relative_l2=float(d.norm()/y.norm().clamp_min(1e-30)),bad=bad,count=a.numel(),passed=bad==0,atol=atol,rtol=rtol))
    if not any(x['role']!='diagnostic' for x in rows):raise ValueError('Empty strict gate')
    return dict(passed=all(x['passed'] for x in rows if x['role']!='diagnostic'),checkpoints=rows,diagnostic_exceedances=[x['name'] for x in rows if x['role']=='diagnostic' and not x['passed']])


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['actual','expected','roles','output']:p.add_argument('--'+k,required=True)
    a=p.parse_args();r=compare(read(a.actual),read(a.expected),json.loads(Path(a.roles).read_text()));Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
    for v in r['checkpoints']:
        if v['role']!='diagnostic':print(json.dumps(v))
    raise SystemExit(0 if r['passed'] else 1)
