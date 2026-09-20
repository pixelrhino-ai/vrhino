#!/usr/bin/env python3
"""Cross-precision diagnostics and strict finite/dtype checks, no model tolerance."""
import argparse
import json
from pathlib import Path
import struct
import numpy as np
import torch


def read(path):
    output={}
    with open(path,'rb') as f:
        def take(n):
            x=f.read(n)
            if len(x)!=n:raise ValueError('Truncated bundle')
            return x
        if take(8)!=b'VRINPUT1':raise ValueError('Bad bundle magic')
        count=struct.unpack('<I',take(4))[0]
        if count>1024:raise ValueError('Oversize bundle')
        for _ in range(count):
            name=take(struct.unpack('<H',take(2))[0]).decode()
            dt,rank=struct.unpack('<BB',take(2));shape=struct.unpack('<'+'q'*rank,take(8*rank))
            n=struct.unpack('<Q',take(8))[0]
            dtype={0:'<f4',1:'<f2',2:'<u2',3:'<i8',4:'<i4',5:'u1',6:'?',7:'i1'}[dt]
            if n!=int(np.prod(shape))*np.dtype(dtype).itemsize:raise ValueError('Byte count mismatch')
            x=torch.from_numpy(np.frombuffer(take(n),dtype=dtype).copy().reshape(shape))
            if dt==2:x=x.view(torch.bfloat16)
            if name in output:raise ValueError('Duplicate tensor name')
            output[name]=x
        if f.read(1):raise ValueError('Trailing bundle bytes')
    return output


def analyze(actual,reference):
    if actual.keys()!=reference.keys():raise ValueError('Trace key mismatch')
    rows=[]
    for name,a in sorted(actual.items()):
        b=reference[name]
        if a.shape!=b.shape:raise ValueError('Shape drift '+name)
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):raise ValueError('NaN/Inf '+name)
        if not a.is_floating_point() or not b.is_floating_point():
            if a.dtype!=b.dtype or not torch.equal(a,b):raise ValueError('Discrete input drift '+name)
        x=a.double();y=b.double();d=(x-y).abs()
        rows.append(dict(name=name,shape=list(a.shape),dtype=str(a.dtype),reference_dtype=str(b.dtype),finite=True,
            max_abs=float(d.max()),mean_abs=float(d.mean()),max_relative=float((d/y.abs().clamp_min(1e-8)).max()),
            relative_l2=float(d.norm()/y.norm().clamp_min(1e-30)),max_magnitude=float(x.abs().max()),rms=float(x.square().mean().sqrt()),
            reference_rms=float(y.square().mean().sqrt()),diagnostic_exceedances_5e3=int((d>0.005+0.005*y.abs()).sum()),elements=a.numel()))
    return dict(finite=True,shape_pass=True,checkpoints=rows,
        comparison_role='Cross-precision diagnostics, not an end-to-end tolerance gate; 5e-3 marker is retained for visibility only')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--actual',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=analyze(read(a.actual),read(a.reference))
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print('FINITE_SHAPE_PASS',len(result['checkpoints']))
