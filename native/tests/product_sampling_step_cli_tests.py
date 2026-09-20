#!/usr/bin/env python3
"""Bounded qualification options fail closed before package I/O or GPU use."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('cli');p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
base={'schema':'vrhino.sampling-step-check.v1','precision':'fp32','completed_step_limit':1,'device_budget_bytes':60<<30,'host_budget_bytes':200<<30}
cases=[('schema','unknown','Invalid sampling-step-check declaration'),('precision','bf16','explicit FP32 qualification'),('completed_step_limit',0,'exactly one transition'),('completed_step_limit',2,'exactly one transition'),('completed_step_limit',-1,'exactly one transition'),('device_budget_bytes',1,'Invalid sampling resource budget'),('host_budget_bytes',0,'Invalid sampling resource budget'),('binding',1,'Invalid sampling-step-check declaration'),('guidance',1,'Invalid sampling-step-check declaration')]
for i,(key,value,message) in enumerate(cases):
 j=dict(base);j[key]=value;f=a.output/f'bad-{i}.json';f.write_text(json.dumps(j));out=a.output/f'out-{i}'
 r=subprocess.run([a.cli,'sampling-step-check','absent-manifest','absent-resources','absent-request',str(f),str(out)],capture_output=True,text=True,timeout=15)
 assert r.returncode!=0 and message in r.stderr,(i,r.stderr)
 assert not out.exists()
 print('PASS reject',key,repr(value))
f=a.output/'valid.json';f.write_text(json.dumps(base));out=a.output/'existing';out.mkdir(exist_ok=True);marker=out/'marker';marker.write_text('retain')
r=subprocess.run([a.cli,'sampling-step-check','absent-manifest','absent-resources','absent-request',str(f),str(out)],capture_output=True,text=True,timeout=15)
assert r.returncode!=0 and 'already exists' in r.stderr and marker.read_text()=='retain'
print('PASS 10 fail-closed cases')
# v1 remains frozen. v2 widens only the bounded prefix length, not numerical intent.
v2=dict(base,schema='vrhino.sampling-step-check.v2',completed_step_limit=3)
for i,(key,value,message) in enumerate([
 ('completed_step_limit',0,'one to three transitions'),
 ('completed_step_limit',4,'one to three transitions'),
 ('completed_step_limit',-1,'one to three transitions'),
 ('precision','bf16','explicit FP32 qualification'),
 ('binding',1,'Invalid sampling-step-check declaration'),
 ('guidance',4,'Invalid sampling-step-check declaration'),
 ('device_budget_bytes',0,'Invalid sampling resource budget'),
 ('host_budget_bytes',0,'Invalid sampling resource budget')]):
 j=dict(v2);j[key]=value;f=a.output/f'v2-bad-{i}.json';f.write_text(json.dumps(j));out=a.output/f'v2-out-{i}'
 r=subprocess.run([a.cli,'sampling-step-check','absent-manifest','absent-resources','absent-request',str(f),str(out)],capture_output=True,text=True,timeout=15)
 assert r.returncode!=0 and message in r.stderr,(i,r.stderr)
 assert not out.exists()
 print('PASS v2 reject',key,repr(value))
# Valid v1/v2 options reach the same package fail-closed boundary, without a GPU.
errors=[]
for i,(schema,limit) in enumerate([(base['schema'],1),(v2['schema'],1),(v2['schema'],2),(v2['schema'],3)]):
 j=dict(base,schema=schema,completed_step_limit=limit);f=a.output/f'accepted-{i}.json';f.write_text(json.dumps(j));out=a.output/f'accepted-out-{i}'
 r=subprocess.run([a.cli,'sampling-step-check','absent-manifest','absent-resources','absent-request',str(f),str(out)],capture_output=True,text=True,timeout=15)
 assert r.returncode!=0 and not out.exists();errors.append(r.stderr)
assert all(e==errors[0] for e in errors) and 'transition' not in errors[0]
print('PASS v1 plus v2 limits 1/2/3 reach identical missing-package rejection')
print('PASS total 18 negative options/output cases plus 4 package rejection cases')
v3=dict(base,schema='vrhino.sampling-step-check.v3',completed_step_limit=27)
for i,(key,value,message) in enumerate([
 ('completed_step_limit',0,'one to 32 transitions'),('completed_step_limit',33,'one to 32 transitions'),
 ('completed_step_limit',-1,'one to 32 transitions'),('precision','bf16','explicit FP32 qualification'),
 ('binding',1,'Invalid sampling-step-check declaration'),('model','fixture','Invalid sampling-step-check declaration'),
 ('guidance',3,'Invalid sampling-step-check declaration'),('device_budget_bytes',0,'Invalid sampling resource budget'),
 ('host_budget_bytes',0,'Invalid sampling resource budget')]):
 j=dict(v3);j[key]=value;f=a.output/f'v3-bad-{i}.json';f.write_text(json.dumps(j));out=a.output/f'v3-out-{i}'
 r=subprocess.run([a.cli,'sampling-step-check','absent-manifest','absent-resources','absent-request',str(f),str(out)],capture_output=True,text=True,timeout=15)
 assert r.returncode!=0 and message in r.stderr,(i,r.stderr)
 assert not out.exists()
 print('PASS v3 reject',key,repr(value))
for limit in [1,27,32]:
 f=a.output/f'v3-valid-{limit}.json';f.write_text(json.dumps(dict(v3,completed_step_limit=limit)));out=a.output/f'v3-valid-out-{limit}'
 r=subprocess.run([a.cli,'sampling-step-check','absent-manifest','absent-resources','absent-request',str(f),str(out)],capture_output=True,text=True,timeout=15)
 assert r.returncode!=0 and r.stderr==errors[0] and not out.exists(),r.stderr
print('PASS total 27 negative options/output cases and 7 accepted declarations reaching missing-package rejection')
