#!/usr/bin/env python3
"""Candidate v3 research acceptance; no production precision or model routing."""
import torch

SCHEMA='generic.bf16.operator.acceptance.v3'
U_BF16=2.0**-8

def radius_bf16(x):
    if x.dtype!=torch.bfloat16 or not bool(torch.isfinite(x).all()):
        raise ValueError('BF16 finite storage required')
    # Exponent endpoints use the larger neighboring gap; subnormal/zero use
    # the representable subnormal spacing. Work in F64 to retain those gaps.
    y=x.double();up=torch.nextafter(x,torch.full_like(x,float('inf'))).double()
    down=torch.nextafter(x,torch.full_like(x,-float('inf'))).double()
    gap_up=up-y;gap_down=y-down
    gap_up=torch.where(torch.isfinite(gap_up),gap_up,gap_down)
    gap_down=torch.where(torch.isfinite(gap_down),gap_down,gap_up)
    return .5*torch.maximum(gap_up,gap_down)

def metrics(a,r):
    a=a.double().flatten();r=r.double().flatten();d=(a-r).abs();rel=d/r.abs().clamp_min(1e-8)
    return dict(max_abs=float(d.max()),mean_abs=float(d.mean()),max_relative=float(rel.max()),
        relative_l2=float(d.norm()/r.norm().clamp_min(1e-30)),rms_error=float(d.square().mean().sqrt()),
        reference_rms=float(r.square().mean().sqrt()),count=d.numel(),different=int((d!=0).sum()),
        abs_quantiles={str(q):float(torch.quantile(d,q)) for q in [0.,.5,.9,.99,.999,1.]},
        relative_quantiles={str(q):float(torch.quantile(rel,q)) for q in [.5,.9,.99,1.]})

def compare(a,r,category,operator=None):
    if a.shape!=r.shape or a.dtype!=r.dtype or a.numel()==0:raise ValueError('Shape/dtype/empty contract')
    if not bool(torch.isfinite(a).all() and torch.isfinite(r).all()):raise ValueError('Nonfinite boundary')
    if category!='exact' and not a.is_floating_point():raise ValueError('Discrete values require exact role')
    d=(a.double()-r.double()).abs();result=metrics(a,r)
    if category=='operator_bf16':
        bound=.005+radius_bf16(a)+radius_bf16(r)
        rms_bound=.005+U_BF16*result['reference_rms']
    elif category=='f32_semantic':
        if a.dtype!=torch.float32:raise ValueError('F32 role dtype')
        bound=2e-5+2e-5*r.double().abs();rms_bound=None
    elif category in ['observable','diagnostic']:
        bound=.005+.005*r.double().abs();rms_bound=None
    elif category=='exact':
        bound=torch.zeros_like(d);rms_bound=None
    else:raise ValueError('Unknown category')
    pointwise_bad=int((a!=r).sum()) if category=='exact' else int((d>bound).sum())
    rms_pass=rms_bound is None or result['rms_error']<=rms_bound
    invariant_pass=True
    if operator=='softmax':
        invariant_pass=bool((a>=0).all() and (a<=1).all() and ((a.double().sum(-1)-1).abs()<=U_BF16+2e-5).all())
    old_bad=int((d>.005+.005*r.double().abs()).sum())
    result.update(category=category,pointwise_bad=pointwise_bad,rms_bound=rms_bound,rms_pass=rms_pass,
        invariant_pass=invariant_pass,old_gate_exceedances=old_bad,
        passed=invariant_pass and (category=='diagnostic' or (pointwise_bad==0 and rms_pass)))
    return result
