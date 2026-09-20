#!/usr/bin/env python3
import argparse,json
from pathlib import Path
from collections import defaultdict
import torch
from bf16_policy_reference import read
from bf16_operator_acceptance import compare,metrics,SCHEMA

def main(a):
    actual,expected=read(a.actual),read(a.reference);decl=json.loads(Path(a.declarations).read_text())
    if actual.keys()!=expected.keys() or actual.keys()!=decl.keys():raise ValueError('Incomplete boundary set')
    rows=[];groups=defaultdict(list)
    for name in sorted(actual):
        d=decl[name];result=compare(actual[name],expected[name],d['category'],d['operator']);result.update(name=name,operator=d['operator'],profile=d.get('profile',''))
        rows.append(result);groups[d['operator']].append(name)
    summary={}
    for op,names in groups.items():
        x=torch.cat([actual[n].double().flatten() for n in names]);y=torch.cat([expected[n].double().flatten() for n in names]);summary[op]=metrics(x,y)
        summary[op]['cases']=len(names);summary[op]['failed_cases']=[r['name'] for r in rows if r['operator']==op and not r['passed']]
    result=dict(schema=SCHEMA,passed=all(r['passed'] for r in rows),cases=len(rows),checkpoints=rows,summary=summary)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print('CASES',len(rows),'FAILED',sum(not r['passed'] for r in rows))
    for r in rows:
        if not r['passed']:print(json.dumps(r))
    return 0 if result['passed'] else 1

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['actual','reference','declarations','output']:p.add_argument('--'+n,required=True)
    raise SystemExit(main(p.parse_args()))
