#!/usr/bin/env python3
"""Independent research math for the shared operation policy, no runtime/model routing."""
import json,struct
from pathlib import Path
import torch
import torch.nn.functional as F
from bf16_bringup_analysis import read


def write(path, tensors):
    with open(path,'wb') as f:
        f.write(b'VRINPUT1'+struct.pack('<I',len(tensors)))
        for name,x in sorted(tensors.items()):
            x=x.detach().cpu().contiguous();dt={torch.float32:0,torch.bfloat16:2,torch.int64:3,torch.bool:6}[x.dtype]
            data=(x.view(torch.uint16) if dt==2 else x).numpy().tobytes();n=name.encode()
            f.write(struct.pack('<H',len(n))+n+struct.pack('<BB',dt,x.ndim)+struct.pack('<'+'q'*x.ndim,*x.shape)+struct.pack('<Q',len(data))+data)


class PolicyMath:
    def __init__(self,path):
        self.document=json.loads(Path(path).read_text());d=self.document['effective']
        assert self.document['schema']=='vrhino.precision.policy.v1' and d['requested_mode']=='bf16'
        self.ops=d['operations'];self.roles=d['roles']
        assert all(v['accumulator_dtype']=='fp32' for v in self.ops.values())
    @staticmethod
    def dtype(value):return {'fp32':torch.float32,'bf16':torch.bfloat16}[value]
    def role(self,name):return self.dtype(self.roles[name])
    def linear(self,x,w,b=None,operation='LINEAR'):
        # Round operands first, accumulate and apply bias in F32, then round
        # the output once. An opaque BF16 F.linear may round GEMM before bias
        # on some shapes. That is a different epilogue contract. Promoting
        # already-rounded operands here represents the accumulator in the
        # independent research oracle; it is not a production compute fallback.
        d=torch.float32 if operation=='MODULATION' else self.dtype(self.ops[operation]['compute_dtype'])
        out=torch.float32 if operation=='MODULATION' else self.dtype(self.ops[operation]['output_dtype'])
        return F.linear(x.to(d).float(),w.to(d).float(),
                        None if b is None else b.to(d).float()).to(out)
    def norm(self,x,w=None,b=None,eps=1e-6):
        w=None if w is None else w.to(x.dtype).float();b=None if b is None else b.to(x.dtype).float()
        return F.layer_norm(x.float(),(x.shape[-1],),w,b,eps).to(x.dtype)
    def rms(self,x,w,eps):
        y=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+eps)
        # Normalized storage follows the input dtype before F32 affine output.
        # A F32 output role does not eliminate this intermediate rounding.
        return (y.to(x.dtype).float()*w.float()).float()
    def attention(self,q,k,v):
        d=self.dtype(self.ops['ATTENTION_QK']['compute_dtype'])
        q=q.to(d).float().transpose(1,2);k=k.to(d).float().transpose(1,2);v=v.to(d).float().transpose(1,2)
        p=torch.softmax((q@k.transpose(-1,-2))*(q.shape[-1]**-.5),-1)
        return (p@v).transpose(1,2).contiguous().to(self.dtype(self.ops['ATTENTION_PV']['output_dtype']))
    def residual(self,x,branch,gate=None):
        d=self.role('RESIDUAL_STATE');x=x.to(d);branch=branch.to(d)
        if gate is not None:branch=branch*gate.to(d)
        return x+branch
    def rope(self,x,cos,sin):
        a=x.float()[...,0::2];b=x.float()[...,1::2]
        cos=cos.float()[...,0::2];sin=sin.float()[...,0::2]
        return torch.stack((a*cos-b*sin,a*sin+b*cos),-1).flatten(-2)


def configure():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.use_deterministic_algorithms(True)
