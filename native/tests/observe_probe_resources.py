#!/usr/bin/env python3
"""Lightweight research process/RSS/device observer; no production memory policy."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def device():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    return [dict(uuid=x[0],name=x[1],total_mib=int(x[2]),used_mib=int(x[3])) for x in ([v.strip() for v in row.split(',')] for row in rows)]


def host(pid):
    try:lines=Path(f'/proc/{pid}/status').read_text().splitlines()
    except FileNotFoundError:return {}
    return {line.split(':')[0]:int(line.split()[1])*1024 for line in lines if line.split(':')[0] in ['VmRSS','VmHWM','RssAnon','RssFile','RssShmem']}


def main(args):
    before=device();samples=[];started=time.monotonic()
    with open(args.log,'w') as log:
        p=subprocess.Popen(args.command,stdout=log,stderr=subprocess.STDOUT)
        while p.poll() is None:
            samples.append(dict(elapsed_seconds=time.monotonic()-started,host=host(p.pid),device=device()))
            time.sleep(0.5)
        code=p.wait()
    after=device()
    result=dict(command=args.command,returncode=code,before=before,after=after,samples=samples,
                peak_host_bytes={k:max((s['host'].get(k,0) for s in samples),default=0) for k in ['VmRSS','VmHWM','RssAnon','RssFile']},
                peak_device_used_mib=max((d['used_mib'] for s in samples for d in s['device']),default=0),
                teardown_returned_to_baseline=all(a['uuid']==b['uuid'] and a['used_mib']==b['used_mib'] for a,b in zip(after,before)),
                limitations='Sampled RSS includes mapped file pages; device usage includes CUDA libraries/context. Not an allocation counter or full-inference bound.')
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    return code


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--log',required=True);p.add_argument('--output',required=True)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    if a.command and a.command[0]=='--':a.command=a.command[1:]
    if not a.command:p.error('probe command required')
    raise SystemExit(main(a))
