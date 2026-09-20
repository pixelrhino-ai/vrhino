#!/usr/bin/env python3
"""Compare explicit semantic lowering with the frozen mixed-scalar oracle/gates."""
import argparse,json
from pathlib import Path
import torch
from bf16_policy_reference import read,configure
from bf16_operator_acceptance import compare
from mixed_scalar_solver_qualification import solvers

def parts(root,prefix):
    result={}
    for path in sorted(root.glob(prefix+'.bundle.[0-9]*')):
        values=read(path);assert not result.keys() & values.keys();result.update(values)
    assert result
    return result

def qualify(root,fixture):
    configure();raw=read(fixture/'inputs.bundle');decl=json.loads((fixture/'declarations.json').read_text())
    legacy=parts(root,'legacy');previous=parts(fixture,'native');assert legacy.keys()==previous.keys()
    exact=all(torch.equal(legacy[k],previous[k]) for k in legacy)
    native=parts(root,'semantic');rows={}
    for p,d in decl.items():
        x=raw[p+'.x'].cuda();scalar=raw[p+'.scalar'].cuda()
        if d['kind']=='scalar':
            for location in ['host','device']:
                for op in ['add','mul','div']:
                    expected=(x.float()+scalar if op=='add' else x.float()*scalar if op=='mul' else x.float()/scalar).to(x.dtype).cpu()
                    name=p+'.semantic.'+location+'.'+op
                    rows[name]=compare(native[name],expected,'operator_bf16' if x.dtype==torch.bfloat16 else 'f32_semantic')
                    rows[name]['bitwise']=torch.equal(native[name],expected)
        else:
            expected=solvers(x,raw[p+'.y'].cuda(),raw[p+'.z'].cuda(),float(scalar),False)
            for k,v in expected.items():
                name=p+'.'+k;rows[name]=compare(native[name],v.cpu(),'observable');rows[name]['bitwise']=torch.equal(native[name],v.cpu())
    assert len(native)==len(legacy)+576
    summary=dict(legacy_1380_outputs_bitwise=exact,checks=len(rows),passed=all(v['passed'] for v in rows.values()),non_bitwise=sum(not v['bitwise'] for v in rows.values()),failed=[k for k,v in rows.items() if not v['passed']])
    (root/'scalar-qualification.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2)+'\n');print(json.dumps(summary,indent=2))
    return exact and summary['passed']
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('fixture',type=Path);a=p.parse_args();raise SystemExit(0 if qualify(a.root,a.fixture) else 1)
