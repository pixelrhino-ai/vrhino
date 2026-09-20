#!/usr/bin/env python3
"""Research-only same-input localization; higher precision is diagnostic only."""
import argparse,json
import hashlib
from pathlib import Path
import torch
import torch.nn.functional as F
from bf16_policy_reference import PolicyMath,configure,read,write
from bf16_reference_compare import compare


@torch.no_grad()
def prepare(a):
    configure();p=PolicyMath(a.policy);root=Path(a.root)
    x=read(root/'operators/inputs.bundle');r=read(root/'operators/reference.bundle')
    r['captured_rms']=p.rms(x['captured_rms.x'].cuda(),x['captured_rms.w'].cuda(),1e-6).cpu()
    write(root/'operators/reference.bundle',r)
    block=read(a.capture);fixtures={};expected={};roles={}
    def add(name,kind,values,y):
        fixtures[name+'.kind']=torch.tensor(kind,dtype=torch.int64)
        fixtures.update({name+'.'+k:v.cpu() for k,v in values.items()})
        expected[name]=y.cpu();roles[name]='bf16_path'
    v={k:x['captured_ffn.'+k].cuda() for k in ['x','w','b','w2','b2']}
    h=p.linear(v['x'],v['w'],v['b']);g=F.gelu(h,approximate='tanh')
    add('ffn_linear0',1,{k:v[k] for k in ['x','w','b']},h)
    hn=block['block.ffn_projection_in'].cuda();gn=block['block.ffn_activation'].cuda()
    add('gelu_native_input',6,dict(x=hn),F.gelu(hn,approximate='tanh'))
    add('gelu_reference_input',6,dict(x=h),g)
    add('linear1_native_input',1,dict(x=gn,w=v['w2'],b=v['b2']),p.linear(gn,v['w2'],v['b2']))
    add('linear1_reference_input',1,dict(x=g,w=v['w2'],b=v['b2']),p.linear(g,v['w2'],v['b2']))
    # An independent F64 calculation on already rounded operands measures the
    # rounding boundary. It is NOT substituted for the frozen F32-accumulator oracle.
    q,k,val=[x['captured_cross_attention.'+n].cuda().to(torch.bfloat16).double().transpose(1,2) for n in ['q','k','v']]
    y=(torch.softmax((q@k.transpose(-1,-2))*(q.shape[-1]**-.5),-1)@val).transpose(1,2).contiguous()
    # Bundle format intentionally has no F64: JSON preserves the diagnostic scalars.
    n=read(root/'operators/native.bundle')['captured_cross_attention'].double().cuda()
    ref=r['captured_cross_attention'].double().cuda();bad=(n-ref).abs()>.005+.005*ref.abs()
    rows=[]
    for idx in bad.nonzero().tolist():
        i=tuple(idx);rows.append(dict(index=idx,native=float(n[i]),reference=float(ref[i]),f64_diagnostic=float(y[i]),
            midpoint=float((n[i]+ref[i])/2),native_error_f64=float(abs(n[i]-y[i])),reference_error_f64=float(abs(ref[i]-y[i])),
            nearest_bf16_f64=float(y[i].bfloat16())))
    (root/'attention-rounding.json').write_text(json.dumps(dict(failing_elements=rows,diagnostic_only=True),indent=2)+'\n')
    write(root/'local-inputs.bundle',fixtures);write(root/'local-reference.bundle',expected)
    (root/'local-roles.json').write_text(json.dumps(roles,sort_keys=True,indent=2)+'\n')
    result=compare(read(root/'operators/native.bundle'),r,json.loads((root/'operators/roles.json').read_text()))
    (root/'operators-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print('corrected_operator_pass',result['passed']);print(json.dumps(rows,indent=2))


@torch.no_grad()
def analyze(a):
    configure();p=PolicyMath(a.policy);root=Path(a.root)
    x=read(root/'operators/inputs.bundle');n=read(root/'local-native.bundle');r=read(root/'local-reference.bundle')
    block=read(a.capture);opn=read(root/'operators/native.bundle');opr=read(root/'operators/reference.bundle')
    v={k:x['captured_ffn.'+k].cuda().to(torch.bfloat16) for k in ['x','w','b','w2','b2']}
    exact=F.linear(v['x'].double(),v['w'].double(),v['b'].double()).cpu()
    hn=n['ffn_linear0'].double();hr=r['ffn_linear0'].double()
    rows=[]
    for idx in (hn!=hr).nonzero().tolist():
        i=tuple(idx);rows.append(dict(index=idx,native=float(hn[i]),reference=float(hr[i]),f64_diagnostic=float(exact[i]),
            nearest_bf16_f64=float(exact[i].bfloat16()),native_error_f64=float(abs(hn[i]-exact[i])),reference_error_f64=float(abs(hr[i]-exact[i]))))
    replay=p.linear(F.gelu(n['ffn_linear0'].cuda(),approximate='tanh'),v['w2'],v['b2']).cpu()
    native=opn['captured_ffn'];reference=opr['captured_ffn'];d=(native.double()-reference.double()).abs()
    bad=d>.005+.005*reference.double().abs()
    failures=[]
    for idx in bad.nonzero().tolist():
        i=tuple(idx);failures.append(dict(index=idx,native=float(native[i]),reference=float(reference[i]),abs_error=float(d[i]),gate=float(.005+.005*abs(reference[i].double()))))
    out=dict(first_linear_mismatches=rows,first_linear_mismatch_count=len(rows),
             linear0_matches_block_capture=torch.equal(n['ffn_linear0'],block['block.ffn_projection_in']),
             gelu_matches_block_capture=torch.equal(n['gelu_native_input'],block['block.ffn_activation']),
             final_matches_block_capture=torch.equal(native,block['block.ffn_projection_out']),
             reference_replayed_from_native_linear0_exact=torch.equal(replay,native),
             ffn_failed_elements=failures,diagnostic_only=True)
    (root/'ffn-localization.json').write_text(json.dumps(out,indent=2)+'\n')
    repeat=read(root/'operators/native-repeat.bundle')
    hashes=lambda file:hashlib.sha256(Path(file).read_bytes()).hexdigest()
    consistency=dict(native_repeat_all_tensors_exact=all(torch.equal(v,repeat[k]) for k,v in opn.items()),
                     native_sha256=hashes(root/'operators/native.bundle'),repeat_sha256=hashes(root/'operators/native-repeat.bundle'))
    (root/'determinism.json').write_text(json.dumps(consistency,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ['first_linear_mismatches','ffn_failed_elements']},indent=2));print(json.dumps(consistency,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--policy',required=True);parser.add_argument('--capture',required=True)
    parser.add_argument('--analyze',action='store_true');a=parser.parse_args();(analyze if a.analyze else prepare)(a)
