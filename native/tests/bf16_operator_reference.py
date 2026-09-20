#!/usr/bin/env python3
"""Generic synthetic and captured-operand operator oracle using PyTorch math."""
import argparse,json
from pathlib import Path
import torch
import torch.nn.functional as F
from bf16_policy_reference import PolicyMath,configure,write,read
from wan_numerical_reference import Source


@torch.no_grad()
def run(a):
    configure();torch.manual_seed(20260917);root=Path(a.output);root.mkdir(parents=True,exist_ok=True);p=PolicyMath(a.policy)
    fixtures={};expected={};roles={}
    def case(name,kind,values,local_f32=False):
        fixtures[name+'.kind']=torch.tensor(kind,dtype=torch.int64)
        fixtures.update({name+'.'+k:v.cpu() for k,v in values.items()});roles[name]='local_f32' if local_f32 else 'bf16_path'
        v={k:x.cuda() for k,x in values.items()}
        if kind==0:y=p.norm(v['x'],v['w'],v['b'])
        elif kind==1:y=p.linear(v['x'],v['w'],v['b'])
        elif kind==2:y=p.attention(v['q'],v['k'],v['v'])
        elif kind==3:y=p.linear(F.gelu(p.linear(v['x'],v['w'],v['b']),approximate='tanh'),v['w2'],v['b2'])
        elif kind==4:y=p.residual(v['x'],v['branch'],v['gate'])
        elif kind==5:y=p.rms(v['x'],v['w'],1e-6)
        expected[name]=y.cpu()
    rand=lambda *s:torch.randn(*s)*.2
    for dtype in [torch.float32,torch.bfloat16]:
        tag='f32_storage' if dtype==torch.float32 else 'bf16_storage'
        case('norm_'+tag,0,dict(x=rand(3,127).to(dtype),w=rand(127)+1,b=rand(127)),dtype==torch.float32)
        case('residual_'+tag,4,dict(x=rand(3,127).to(dtype),branch=rand(3,127).to(torch.bfloat16),gate=rand(1,127)),True)
    case('linear_tail',1,dict(x=rand(3,127),w=rand(67,127),b=rand(67)))
    case('ffn_tail',3,dict(x=rand(3,127),w=rand(67,127),b=rand(67),w2=rand(127,67),b2=rand(127)))
    case('attention_tail',2,dict(q=rand(1,7,2,32),k=rand(1,19,2,32),v=rand(1,19,2,32)))
    if a.capture:
        x=read(a.capture);src=Source(a.source)
        case('captured_linear',1,dict(x=x['block.self_attention_input'],w=src.tensor('blocks.0.self_attn.q.weight'),b=src.tensor('blocks.0.self_attn.q.bias')))
        case('captured_norm',0,dict(x=x['block.first_residual'],w=src.tensor('blocks.0.norm3.weight'),b=src.tensor('blocks.0.norm3.bias')),True)
        case('captured_rms',5,dict(x=x['block.self_attention.q_linear'],w=src.tensor('blocks.0.self_attn.norm_q.weight')),True)
        for label in ['self_attention','cross_attention']:
            case('captured_'+label,2,{k:x[f'block.{label}.{k}_positioned' if k!='v' else f'block.{label}.v_heads'] for k in ['q','k','v']})
        case('captured_ffn',3,dict(x=x['block.ffn_input'],w=src.tensor('blocks.0.ffn.0.weight'),b=src.tensor('blocks.0.ffn.0.bias'),w2=src.tensor('blocks.0.ffn.2.weight'),b2=src.tensor('blocks.0.ffn.2.bias')))
        case('captured_residual',4,dict(x=x['block.input'],branch=x['block.self_attention.output'],gate=x['block.gate']),True)
        (root/'source-hashes.json').write_text(json.dumps(src.hashes,sort_keys=True,indent=2)+'\n')
    write(root/'inputs.bundle',fixtures);write(root/'reference.bundle',expected)
    (root/'roles.json').write_text(json.dumps(roles,sort_keys=True,indent=2)+'\n');print('REFERENCE_OPERATORS',len(expected))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--policy',required=True);p.add_argument('--output',required=True);p.add_argument('--capture');p.add_argument('--source');run(p.parse_args())
