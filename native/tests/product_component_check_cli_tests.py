#!/usr/bin/env python3
"""Fail-closed product component-check option validation, without opening weights."""
import argparse,json,subprocess
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('cli');p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 base={'schema':'vrhino.component-check.v1','precision':'fp32','actions':['conditioning','decoder'],'device_budget_bytes':60<<30,'host_budget_bytes':200<<30,'fps':8,'latent_bundle':'fixture.vrt','media_range':[-1,1]}
 cases=[('schema','unknown','Invalid component-check declaration'),('precision','bf16','explicit FP32 qualification'),('actions',['unknown'],'Unknown component check'),('actions',['conditioning','conditioning'],'Duplicate component check'),('actions',[],'Empty component check'),('device_budget_bytes',1,'Invalid component resource budget'),('host_budget_bytes',0,'Invalid component resource budget'),('fps',0,'Invalid media fps'),('media_range',[1,-1],'Invalid media range'),('latent_bundle','','explicit synthetic latent fixture')]
 for i,(key,value,message) in enumerate(cases):
  j=dict(base);j[key]=value;f=a.output/f'bad-{i}.json';f.write_text(json.dumps(j))
  r=subprocess.run([a.cli,'component-check','absent-manifest','absent-resources','absent-request',str(f),str(a.output/f'out-{i}')],capture_output=True,text=True,timeout=15)
  assert r.returncode!=0 and message in r.stderr,(i,r.stderr)
  assert not (a.output/f'out-{i}').exists()
  print('PASS reject',key,repr(value))
 # Existing evidence must never be overwritten, even before package I/O.
 f=a.output/'valid.json';f.write_text(json.dumps(base));out=a.output/'existing';out.mkdir(exist_ok=True);marker=out/'marker';marker.write_text('retain')
 r=subprocess.run([a.cli,'component-check','absent-manifest','absent-resources','absent-request',str(f),str(out)],capture_output=True,text=True,timeout=15)
 assert r.returncode!=0 and 'already exists' in r.stderr and marker.read_text()=='retain'
 print('PASS 11 negative admission cases; no package I/O or component numerical execution')
if __name__=='__main__':main()
