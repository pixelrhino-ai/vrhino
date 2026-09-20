#!/usr/bin/env python3
"""Research corpus/oracle for generic mixed scalar semantics; immutable V3 gates."""
import argparse,ctypes,json,math,subprocess
from pathlib import Path
import numpy as np
import torch
from bf16_policy_reference import read,write,configure
from bf16_operator_acceptance import compare,metrics

F=np.float32
libm=ctypes.CDLL('libm.so.6')
for name in ['logf','expm1f','sqrtf']:
    fn=getattr(libm,name);fn.argtypes=[ctypes.c_float];fn.restype=ctypes.c_float

def fixture(root):
    torch.manual_seed(20260918);data={};decl={}
    for dt in [torch.bfloat16,torch.float32]:
      for shape in [(17,),(2,3,7),(1,3,128)]:
       for rank in [0,1]:
        for c in [1.0,0.5,-1.0,0.8778772950172424,0.9862433671951294,3.1415927,1000.25,1.00390625]:
         p=f'scalar{len(decl):03}';x=(torch.randn(shape)*3).to(dt)
         data.update({p+'.kind':torch.tensor(0),p+'.x':x,p+'.y':x,p+'.scalar':torch.tensor(c,dtype=torch.float32).reshape(()) if rank==0 else torch.tensor([c],dtype=torch.float32)})
         decl[p]=dict(kind='scalar',shape=shape,dtype=str(dt),rank=rank,coefficient=float(F(c)))
    for dt in [torch.bfloat16,torch.float32]:
      for shape in [(17,),(2,3,7),(1,3,128)]:
       for c in [0.8778772950172424,0.031415927]:
        p=f'solver{len(decl):03}';data[p+'.kind']=torch.tensor(1)
        for k in ['x','y','z']:data[p+'.'+k]=(torch.randn(shape)*2).to(dt)
        data[p+'.scalar']=torch.tensor(c,dtype=torch.float32);decl[p]=dict(kind='solver',shape=shape,dtype=str(dt),coefficient=float(F(c)))
    write(root/'inputs.bundle',data);(root/'declarations.json').write_text(json.dumps(decl,indent=2)+'\n')

class Arithmetic:
    def __init__(self,dtype,narrow):self.dtype=dtype;self.narrow=narrow
    def add(self,a,b):return (a.float()+b.float()).to(self.dtype)
    def mul(self,a,c):
        t=torch.tensor(float(c),dtype=self.dtype if self.narrow else torch.float32,device=a.device)
        return (a.float()*t.float()).to(self.dtype)
    def sub(self,a,b):return (a.float()-b.float()).to(self.dtype)

def solvers(x,y,z,c,narrow):
    op=Arithmetic(x.dtype,narrow);add,mul,sub=op.add,op.mul,op.sub;c=F(c)
    signal=F(libm.sqrtf(c));noise=F(libm.sqrtf(F(1-c)))
    r={'flow':add(x,mul(y,-c)),'v':add(mul(x,signal),mul(y,-noise)),
       'epsilon':mul(add(x,mul(y,-noise)),F(1/signal)),
       'affine':add(mul(x,c),mul(y,F(1-c))),
       'cfg':add(x,mul(sub(y,x),F(3.1415927))),
       'linear':add(add(mul(x,c),mul(y,F(1-c))),mul(z,F(-.1234567))),
       'euler':add(x,mul(y,float(torch.tensor(-.012076616287231445).bfloat16()) if narrow else F(-.012076616287231445)))}
    ss=list(map(F,[.8778772950172424,.865800678730011,.8522725701332092,.8370144367218018]))
    def lm(s):return F(F(libm.logf(F(1-s)))-F(libm.logf(s)))
    for correct,order,idx,name in [(False,1,0,'predictor1'),(False,2,1,'predictor2'),(True,1,1,'corrector1'),(True,2,2,'corrector2')]:
        st=ss[idx if correct else idx+1];s0=ss[idx-1 if correct else idx];at=F(1-st);h=F(lm(st)-lm(s0));hh=-h;ph=F(libm.expm1f(hh));coeff=F(-at*ph)
        base=add(mul(x,F(st/s0)),mul(y,coeff))
        if order==1:
            r[name]=add(base,mul(mul(sub(z,y),F(.5)),coeff)) if correct else base;continue
        si=ss[idx-2 if correct else idx-1];rk=F(F(lm(si)-lm(s0))/h);d=mul(sub(z,y),F(1/rk))
        if not correct:res=mul(d,F(.5))
        else:
            phi2=F(F(ph/hh)-F(1));b1=F(phi2/ph);phi3=F(F(phi2/hh)-F(.5));b2=F(F(phi3*F(2))/ph);det=F(1-rk)
            c0=F(F(b1-b2)/det);c1=F(F(b2-F(b1*rk))/det);res=add(mul(d,c0),mul(sub(z,y),c1))
        r[name]=add(base,mul(res,coeff))
    return r

def analyze(root):
    configure();inputs=read(root/'inputs.bundle');actual={}
    for part in sorted(root.glob('native.bundle.[0-9]*')):
        chunk=read(part);assert not actual.keys() & chunk.keys();actual.update(chunk)
    assert actual
    decl=json.loads((root/'declarations.json').read_text());rows={};invariants=[]
    for p,d in decl.items():
        x=inputs[p+'.x'].cuda();y=inputs[p+'.y'].cuda();s=inputs[p+'.scalar'].cuda();c=float(s.flatten()[0]);dt=x.dtype
        if d['kind']=='scalar':
            for location in ['host','device']:
                for opname in ['mul','mul_reverse','add','add_reverse','div']:
                    reverse=opname.endswith('_reverse');dtype=torch.float32 if location=='device' and reverse else dt
                    a=x.to(dtype).float();b=s.to(dtype).float()
                    legacy=(a*b if opname.startswith('mul') else a+b if opname.startswith('add') else a/b).to(dtype).cpu()
                    desired=(x.float()*s if opname.startswith('mul') else x.float()+s if opname.startswith('add') else x.float()/s).to(dt).cpu()
                    name=p+'.'+location+'.'+opname;v=actual[name]
                    match=v.dtype==desired.dtype and v.shape==desired.shape
                    rows[name]=dict(legacy=compare(v,legacy,'operator_bf16' if dtype==torch.bfloat16 else 'f32_semantic'),candidate_shape_dtype_match=match,candidate=compare(v,desired,'operator_bf16' if dt==torch.bfloat16 else 'f32_semantic') if match else None)
            for opname in ['mul','add','div']:
                v=actual[p+'.preserve.'+opname];expected=(x.float()*s if opname=='mul' else x.float()+s if opname=='add' else x.float()/s).to(dt).cpu()
                rows[p+'.preserve.'+opname]=dict(candidate=compare(v,expected,'operator_bf16' if dt==torch.bfloat16 else 'f32_semantic'),bitwise=torch.equal(v,expected))
            invariants.append(dict(case=p,dtype=d['dtype'],device_commutative_dtype=actual[p+'.device.mul'].dtype==actual[p+'.device.mul_reverse'].dtype,host_device_forward_exact=torch.equal(actual[p+'.host.mul'],actual[p+'.device.mul'])))
        else:
            z=inputs[p+'.z'].cuda();legacy=solvers(x,y,z,c,True);desired=solvers(x,y,z,c,False)
            for k in legacy:
                name=p+'.'+k;v=actual[name];rows[name]=dict(legacy=compare(v,legacy[k].cpu(),'observable'),legacy_bitwise=torch.equal(v,legacy[k].cpu()),candidate=compare(v,desired[k].cpu(),'observable'))
    assert rows.keys()==actual.keys(), 'Incomplete or unexpected output set'
    summary=dict(outputs=len(rows),legacy_failed=sum(not x['legacy']['passed'] for x in rows.values() if 'legacy' in x),candidate_numeric_failed=sum(not x['candidate']['passed'] for x in rows.values() if x.get('candidate')),candidate_dtype_failed=sum(x.get('candidate_shape_dtype_match') is False for x in rows.values()),preserve_bridge_bitwise=all(x.get('bitwise',True) for x in rows.values()),solver_legacy_bitwise=all(x.get('legacy_bitwise',True) for x in rows.values()),device_operand_order_dtype_failures=sum(not x['device_commutative_dtype'] for x in invariants))
    (root/'comparison.json').write_text(json.dumps(dict(summary=summary,rows=rows,invariants=invariants),indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('mode',choices=['fixture','analyze']);q.add_argument('root',type=Path);a=q.parse_args();globals()[a.mode](a.root)
