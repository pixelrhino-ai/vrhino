#!/usr/bin/env python3
"""Model-independent F32 attention fixtures, stage analysis, and fixed-gate checks."""
import argparse,json,math,struct,subprocess,os
from pathlib import Path
import numpy as np
import torch


def read(path):
    out={}
    with open(path,'rb') as f:
        assert f.read(8)==b'VRINPUT1'
        for _ in range(struct.unpack('<I',f.read(4))[0]):
            name=f.read(struct.unpack('<H',f.read(2))[0]).decode()
            dt,rank=struct.unpack('<BB',f.read(2));shape=struct.unpack('<'+'q'*rank,f.read(8*rank));n=struct.unpack('<Q',f.read(8))[0]
            out[name]=torch.from_numpy(np.frombuffer(f.read(n),dtype={0:'<f4',3:'<i8',6:'?'}[dt]).copy().reshape(shape))
        assert not f.read(1)
    return out


def write(path,tensors):
    with open(path,'wb') as f:
        f.write(b'VRINPUT1'+struct.pack('<I',len(tensors)))
        for name,t in sorted(tensors.items()):
            a=t.detach().cpu().contiguous().numpy();dt={np.dtype('float32'):0,np.dtype('int64'):3,np.dtype('bool'):6}[a.dtype]
            n=name.encode();data=a.tobytes()
            f.write(struct.pack('<H',len(n))+n+struct.pack('<BB',dt,a.ndim)+struct.pack('<'+'q'*a.ndim,*a.shape)+struct.pack('<Q',len(data))+data)


def metrics(actual,expected):
    a=actual.double();e=expected.double();assert a.shape==e.shape
    delta=(a-e).abs();bad=int((delta>2e-5+2e-5*e.abs()).sum())
    return dict(max_abs=float(delta.max()),mean_abs=float(delta.mean()),max_relative=float((delta/e.abs().clamp_min(1e-8)).max()),relative_l2=float(delta.norm()/e.norm().clamp_min(1e-30)),bad=bad,count=e.numel(),passed=bool(torch.isfinite(a).all() and torch.isfinite(e).all()) and bad==0)


def generate(args):
    torch.manual_seed(7211);root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    cases={}
    if args.capture:
        source=read(args.capture)
        for prefix in ['reference','native']:
            cases['captured_'+prefix]={n:source[prefix+'.'+n] for n in ['q','k','v']}
    for name,b,q,k,h,d in [('single',1,1,1,1,8),('tail_scalar',2,3,101,2,17),('vector',1,7,193,3,64),('cross',1,4,512,2,128),('ltx_shape',1,8,8,32,64),('wan21_shape',1,8,8,12,128),('long_k',1,2,4097,1,17),('bias_mask',2,5,103,2,32),('causal',1,8,8,2,24)]:
        x={n:torch.randn(b,s,h,d)*.3 for n,s in [('q',q),('k',k),('v',k)]}
        if name=='bias_mask':x.update(mask=torch.arange(k)[None,None,:].expand(b,q,k)%3!=1,bias=torch.randn(1,h,q,k)*.05)
        if name=='causal':x['causal']=torch.tensor([1],dtype=torch.int64)
        cases[name]=x
    for name,keys in [('repeated',512),('repeated_long',4097)]:
        x=dict(q=torch.zeros(1,4,2,128),k=torch.zeros(1,keys,2,128),v=torch.full((1,keys,2,128),6.4))
        x['v'][:,0]=-6.4*(keys-1);cases[name]=x
    for name,x in cases.items():
        x['scale']=torch.tensor([1/math.sqrt(x['q'].shape[-1])],dtype=torch.float32)
        x['projection']=torch.randn(17,x['q'].shape[2]*x['q'].shape[3])/math.sqrt(x['q'].shape[2]*x['q'].shape[3])
        write(root/(name+'.input.bundle'),x)
    print('fixtures',len(cases))


@torch.no_grad()
def analyze(args):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    x=read(args.input);y=read(args.actual);q,k,v=[x[n].double() for n in ['q','k','v']]
    b,qs,hs,d=q.shape;ks=k.shape[1];scale=float(x['scale'].item())
    scores=q.transpose(1,2)@k.transpose(1,2).transpose(-1,-2)
    raw=y['qk'].double()
    def logits(s):
        s=s*scale
        if 'bias' in x:s=s+x['bias'].double()
        if 'mask' in x:s=s.masked_fill(~x['mask'][:,None],-float('inf'))
        if 'causal' in x and bool(x['causal'].item()):s=s.masked_fill(torch.arange(ks)[None,:]>torch.arange(qs)[:,None],-float('inf'))
        return s
    exact_p=logits(scores).softmax(-1)
    native_score_p=logits(raw).softmax(-1)
    denom=y['denominator'].reshape(b,qs,hs).transpose(1,2).double().unsqueeze(-1)
    cur=y['current'].double();prev=y['previous'].double()
    suffix=torch.ones_like(denom);effective=torch.empty_like(cur)
    for j in range(ks-1,-1,-1):
        effective[...,j:j+1]=cur[...,j:j+1]*suffix/denom
        suffix*=prev[...,j:j+1]
    exact=(exact_p@v.transpose(1,2)).transpose(1,2)
    controlled_pv=(effective@v.transpose(1,2)).transpose(1,2)
    out=y['output'].double()
    denom_expected=torch.exp(logits(raw)-logits(raw).amax(-1,keepdim=True)).sum(-1,keepdim=True)
    result={
        'qk_vs_f64':metrics(raw,scores),
        'softmax_denominator':metrics(denom,denom_expected),
        'softmax_probability_controlled_scores':metrics(effective,native_score_p),
        'qk_propagated_to_output':metrics((native_score_p@v.transpose(1,2)).transpose(1,2),exact),
        'softmax_propagated_to_output':metrics(controlled_pv,(native_score_p@v.transpose(1,2)).transpose(1,2)),
        'pv_controlled_probabilities':metrics(out,controlled_pv),
        'output_vs_f64':metrics(out,exact),
    }
    # Compare a separate F32 oracle without TF32 or fused half-precision SDPA.
    fq,fk,fv=[x[n].cuda() for n in ['q','k','v']]
    fs=(fq.transpose(1,2)@fk.transpose(1,2).transpose(-1,-2))*scale
    if 'bias' in x:fs=fs+x['bias'].cuda()
    if 'mask' in x:fs=fs.masked_fill(~x['mask'].cuda()[:,None],-float('inf'))
    if 'causal' in x and bool(x['causal'].item()):fs=fs.masked_fill(torch.arange(ks,device='cuda')[None,:]>torch.arange(qs,device='cuda')[:,None],-float('inf'))
    ref=(fs.softmax(-1)@fv.transpose(1,2)).transpose(1,2).cpu()
    result['output_vs_f32']=metrics(out,ref)
    result['f32_oracle_vs_f64']=metrics(ref,exact)
    if 'projection' in y:
        w=x['projection'].double().T
        result['projection_controlled_input']=metrics(y['projection'],out.reshape(b,qs,hs*d)@w)
        result['projection_end_to_end']=metrics(y['projection'],exact.reshape(b,qs,hs*d)@w)
    result['gate']={'atol':2e-5,'rtol':2e-5,'nonfinite':'reject','dtype':'F32'}
    result['passed']=all(v['passed'] for v in result.values() if isinstance(v,dict) and 'passed' in v)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(args.input, 'PASS' if result['passed'] else 'FAIL')
    for name,value in result.items():
        if isinstance(value,dict) and 'bad' in value:print(name,json.dumps(value))
    return 0 if result['passed'] else 1

def qualify(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    fixtures=root/'fixtures';generate(argparse.Namespace(output=str(fixtures),capture=args.capture))
    cases=[(p,96) for p in sorted(fixtures.glob('*.input.bundle'))]
    for name in ['tail_scalar','repeated_long']+(['captured_reference'] if args.capture else []):
        cases.extend((fixtures/(name+'.input.bundle'),t) for t in [8,256])
    rows=[]
    for path,tile in cases:
        name=path.name.split('.')[0]+'-tile'+str(tile)
        env=dict(os.environ,VRHINO_CUDA_ATTENTION_KEY_TILE=str(tile))
        # No research variant may silently replace the production F32 default.
        env.pop('VRHINO_CUDA_ATTENTION_VARIANT',None)
        run=subprocess.run([args.binary,str(path),str(root/(name+'.bundle'))],capture_output=True,text=True,env=env)
        (root/(name+'.native.log')).write_text(run.stdout+run.stderr)
        result=analyze(argparse.Namespace(input=str(path),actual=str(root/(name+'.bundle')),output=str(root/(name+'.json')))) if run.returncode==0 else 1
        rows.append(dict(name=name,native_exit=run.returncode,reference_exit=result))
    (root/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    return int(any(r['native_exit'] or r['reference_exit'] for r in rows))


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='mode',required=True)
    r=sub.add_parser('generate');r.add_argument('--capture');r.add_argument('--output',required=True)
    r=sub.add_parser('analyze');r.add_argument('--input',required=True);r.add_argument('--actual',required=True);r.add_argument('--output',required=True)
    r=sub.add_parser('qualify');r.add_argument('--capture');r.add_argument('--binary',required=True);r.add_argument('--output',required=True)
    a=p.parse_args()
    if a.mode=='generate':generate(a)
    elif a.mode=='qualify':raise SystemExit(qualify(a))
    else:raise SystemExit(analyze(a))
