#!/usr/bin/env python3
"""Deterministic model-independent corpus; coverage is frozen before measurement."""
import argparse,json,math
from pathlib import Path
import torch
import torch.nn.functional as F
from bf16_policy_reference import configure,PolicyMath,write

@torch.no_grad()
def run(a):
    configure();p=PolicyMath(a.policy);root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    fixtures={};expected={};declarations={}
    def case(name,kind,v,operator,category='operator_bf16',profile='generic'):
        fixtures[name+'.kind']=torch.tensor(kind,dtype=torch.int64)
        fixtures.update({name+'.'+k:t.cpu() for k,t in v.items()});v={k:t.cuda() for k,t in v.items()}
        if kind==0:y=p.norm(v['x'],v['w'],v['b'])
        elif kind==1:y=p.linear(v['x'],v['w'],v['b'])
        elif kind==2:y=p.attention(v['q'],v['k'],v['v'])
        elif kind==3:y=p.linear(F.gelu(p.linear(v['x'],v['w'],v['b']),approximate='tanh'),v['w2'],v['b2'])
        elif kind==4:y=p.residual(v['x'],v['branch'],v['gate'])
        elif kind==7:y=F.softmax(v['x'].float(),-1).to(v['x'].dtype)
        else:raise ValueError('Kind')
        expected[name]=y.cpu();declarations[name]=dict(operator=operator,category=category,profile=profile,
            input_shapes={k:list(t.shape) for k,t in v.items()},dtype=str(y.dtype))
    for seed,amplitude in [(20261001,.25),(20261002,1.),(20261003,8.)]:
        torch.manual_seed(seed);tag=f's{seed}'
        for h,n in [(31,67),(127,193),(512,256),(1536,1536),(2048,2048),(5120,256)]:
            v=dict(x=torch.randn(3,h)*amplitude,w=torch.randn(n,h)/math.sqrt(h),b=torch.randn(n)*.1)
            case(f'{tag}_linear_{h}',1,v,'linear',profile=f'hidden_{h}')
        for h,f in [(31,67),(127,193),(512,1024),(1536,8960),(2048,8192)]:
            v=dict(x=torch.randn(3,h)*amplitude,w=torch.randn(f,h)/math.sqrt(h),b=torch.randn(f)*.1,
                w2=torch.randn(h,f)/math.sqrt(f),b2=torch.randn(h)*.1)
            case(f'{tag}_ffn_{h}',3,v,'ffn',profile=f'hidden_{h}_ffn_{f}')
        for h in [31,127,1536,2048,5120]:
            for dt in [torch.float32,torch.bfloat16]:
                v=dict(x=(torch.randn(3,h)*amplitude).to(dt),w=torch.randn(h)*.1+1,b=torch.randn(h)*.1)
                case(f'{tag}_norm_{h}_{dt}',0,v,'layernorm','f32_semantic' if dt==torch.float32 else 'operator_bf16')
            case(f'{tag}_residual_{h}',4,dict(x=torch.randn(3,h)*amplitude,branch=torch.randn(3,h).bfloat16(),gate=torch.randn(1,h)),
                'residual','f32_semantic')
        for heads,d,qs,ks in [(2,32,7,19),(12,128,3,257),(32,64,3,129),(40,128,4,512),(1,128,1,4096)]:
            v=dict(q=(torch.randn(1,qs,heads,d)*amplitude).bfloat16(),k=(torch.randn(1,ks,heads,d)*amplitude).bfloat16(),v=torch.randn(1,ks,heads,d).bfloat16())
            case(f'{tag}_attention_{heads}_{d}_{ks}',2,v,'attention',profile=f'heads_{heads}_width_{d}')
        for h in [19,127,512,4096]:
            case(f'{tag}_softmax_{h}',7,dict(x=(torch.randn(3,h)*amplitude).bfloat16()),'softmax')
    # Degenerate but valid distributions, constant values and high-offset norm.
    case('uniform_softmax',7,dict(x=torch.zeros(3,257,dtype=torch.bfloat16)),'softmax')
    case('dominant_softmax',7,dict(x=torch.tensor([[80.,-80.,0.,-1.]],dtype=torch.bfloat16)),'softmax')
    case('constant_v_attention',2,dict(q=torch.zeros(1,3,2,32,dtype=torch.bfloat16),k=torch.zeros(1,129,2,32,dtype=torch.bfloat16),v=torch.full((1,129,2,32),4.,dtype=torch.bfloat16)),'attention')
    case('zero_ffn',3,dict(x=torch.zeros(3,31),w=torch.zeros(67,31),b=torch.zeros(67),w2=torch.zeros(31,67),b2=torch.zeros(31)),'ffn')
    case('offset_norm',0,dict(x=(torch.randn(3,127)*.01+64).bfloat16(),w=torch.ones(127),b=torch.zeros(127)),'layernorm')
    write(root/'inputs.bundle',fixtures);write(root/'reference.bundle',expected)
    (root/'declarations.json').write_text(json.dumps(declarations,sort_keys=True,indent=2)+'\n')
    print('CASES',len(expected))

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--policy',required=True);q.add_argument('--output',required=True);run(q.parse_args())
