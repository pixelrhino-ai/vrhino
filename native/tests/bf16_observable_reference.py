#!/usr/bin/env python3
"""Research preparation/head replay; causal diagnostics, never an acceptance waiver."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from bf16_policy_reference import PolicyMath, configure, read, write
from wan_numerical_reference import Source, oracle


@torch.no_grad()
def run(args):
    configure(); p=PolicyMath(args.policy); env=oracle(args.official)
    source=Source(args.source); weights={}
    def w(name):
        if name not in weights: weights[name]=source.tensor(name)
        return weights[name]
    x={k:v.cuda() for k,v in read(args.inputs).items()}; out={}
    # Preserve the full reference's token-expanded timestep GEMM geometry.
    tokens=x['native.0.hidden'].shape[1]; width=x['native.0.hidden'].shape[-1]
    time=env['sinusoidal_embedding_1d'](w('time_embedding.0.weight').shape[1],x['timestep']).float()
    out['preparation.sinusoidal']=time
    time=time.unsqueeze(1).repeat(1,tokens,1)
    time=p.linear(time,w('time_embedding.0.weight'),w('time_embedding.0.bias'),'MODULATION');out['preparation.time_linear0']=time[:,0]
    time=F.silu(time);out['preparation.time_activation']=time[:,0]
    time=p.linear(time,w('time_embedding.2.weight'),w('time_embedding.2.bias'),'MODULATION');out['preparation.time']=time[:,0]
    mod=p.linear(F.silu(time),w('time_projection.1.weight'),w('time_projection.1.bias'),'MODULATION');out['preparation.modulation']=mod[:,0].reshape(1,6,width)
    cfg=json.loads((Path(args.source)/'config.json').read_text()); text_len=cfg.get('text_len',512)
    for branch in [0,1]:
        context=x['raw_context'] if branch else torch.zeros_like(x['raw_context'])
        context=F.pad(context,(0,0,0,text_len-context.shape[1]));prefix=f'branch.{branch}.context.'
        context=p.linear(context,w('text_embedding.0.weight'),w('text_embedding.0.bias'));out[prefix+'linear0']=context
        context=F.gelu(context,approximate='tanh');out[prefix+'activation']=context
        out[prefix+'output']=p.linear(context,w('text_embedding.2.weight'),w('text_embedding.2.bias'))
    for prefix in ['native.0','native.1','reference.0','reference.1','hybrid.0','hybrid.1']:
        b=prefix[-1]; h=x[('reference.'+b if prefix.startswith('hybrid') else prefix)+'.hidden']
        t=x[('native.'+b if prefix.startswith('hybrid') else prefix)+'.time']
        parts=(w('head.modulation')+t.reshape(1,1,width)).chunk(2,dim=1)
        norm=p.norm(h,eps=cfg.get('eps',1e-6));out[prefix+'.norm']=norm
        m=norm*(1+parts[1])+parts[0];out[prefix+'.modulated']=m
        out[prefix+'.output']=p.linear(m,w('head.head.weight'),w('head.head.bias'))
    if args.native_boundaries:
        actual=read(args.native_boundaries)
        for b in [0,1]:
            m=actual[f'native.{b}.modulated'].cuda()
            out[f'linear_same_input.{b}']=p.linear(m,w('head.head.weight'),w('head.head.bias'))
    write(args.output,out)
    Path(args.output+'.json').write_text(json.dumps(dict(source_hashes=source.hashes,
        scope='Independent same-input and swapped-input replay; no native intermediate used for full reference qualification'),indent=2)+'\n')


if __name__=='__main__':
    q=argparse.ArgumentParser()
    for k in ['policy','official','source','inputs','output']:q.add_argument('--'+k,required=True)
    q.add_argument('--native-boundaries');run(q.parse_args())
