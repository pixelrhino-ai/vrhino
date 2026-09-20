#!/usr/bin/env python3
"""Stage attribution on captured BF16 operands; no acceptance threshold changes."""
import argparse,json
from pathlib import Path
import torch
from bf16_policy_reference import configure,read

def metrics(a,b):
    a=a.double();b=b.double();d=(a-b).abs()
    return dict(max_abs=float(d.max()),mean_abs=float(d.mean()),relative_l2=float(d.norm()/b.norm().clamp_min(1e-30)),different=int((a!=b).sum()),elements=a.numel())

@torch.no_grad()
def main(a):
    configure();inputs=read(a.inputs);traces=read(a.trace);report=[]
    for prefix in sorted(k[:-5] for k in inputs if k.endswith('.kind')):
        q,k,v=[inputs[prefix+'.'+n].cuda().bfloat16().float().reshape(-1,inputs[prefix+'.'+n].shape[-1]) for n in ['q','k','v']]
        t={n:traces[prefix+'.'+n].cuda() for n in ['qk','score','current','previous','denominator','maximum','accumulation','pre_round','native','observed','scale']}
        qkv_equal=all(torch.equal(traces[prefix+'.'+n],inputs[prefix+'.'+n].bfloat16()) for n in ['q','k','v'])
        if not qkv_equal or not all(bool(torch.isfinite(v).all()) for v in t.values()):
            raise ValueError('Operand or finite contract failed')
        qk=q@k.T;scale=torch.tensor(q.shape[-1]**-.5,device='cuda',dtype=torch.float32);score=qk*scale
        probability=torch.softmax(score,-1);ref=probability@v
        ns=t['score'].reshape(1,-1);same_score_p=torch.softmax(ns,-1);same_score_out=same_score_p@v
        stable_exp=torch.exp(ns-ns.max());reference_den=stable_exp.sum();den=t['denominator'].flatten()[-1]
        # Effective weights in final online-max coordinates from captured native
        # exp/rescaling values. F64 isolates F32 accumulation error; not a fallback.
        cur=t['current'].double().flatten();prev=t['previous'].double().flatten()
        previous_max=torch.cat([torch.full_like(t['maximum'][:,:1],-torch.inf),t['maximum'][:,:-1]],1)
        same_input_exp=torch.exp(t['score']-t['maximum'])
        same_input_rescale=torch.exp(previous_max-t['maximum'])
        future=torch.ones_like(prev);future[:-1]=prev[1:].flip(0).cumprod(0).flip(0)
        weights=cur*future;ideal_den=weights.sum();ideal_acc=weights@v.double()
        acc=t['accumulation'].flatten().double();pre=t['pre_round'].flatten().double()
        ideal_out=ideal_acc/ideal_den
        # Conditional forward bound for the observed exp/rescale operands, NOT
        # a full exp/softmax correctness bound. Each recurrence has <=2K rounded
        # multiply/add operations. Positive denominator, normal finite values.
        keys=k.shape[0];u=2.0**-24;n=2*keys;gamma=n*u/(1-n*u)
        normal=bool((weights>=torch.finfo(torch.float32).tiny).all())
        if not normal or n*u>=1:raise ValueError('Conditional rounding bound inapplicable')
        den_bound=gamma*weights.abs().sum()
        acc_bound=gamma*(weights.abs()@v.double().abs())
        qk_bound=(2*q.shape[-1]*u)/(1-2*q.shape[-1]*u)*(q.double().abs()@k.double().abs().T)
        exact_score=q.double()@k.double().T
        native=t['native'].flatten();reference=ref.bfloat16().flatten()
        bad=(native.double()-reference.double()).abs()>.005+.005*reference.double().abs()
        rows=[]
        for j in bad.nonzero().flatten().tolist():
            rows.append(dict(index=j,native=float(native[j]),reference=float(reference[j]),
                native_pre_round=float(pre[j]),reference_pre_round=float(ref.flatten()[j]),
                reference_with_native_scores=float(same_score_out.flatten()[j]),
                online_weights_f64_result=float(ideal_out[j]),midpoint=float((native[j].double()+reference[j].double())/2),
                native_accumulator=float(acc[j]),ideal_accumulator=float(ideal_acc[j]),
                denominator_only_counterfactual=float(acc[j]/ideal_den),
                denominator_only_bf16=float((acc[j]/ideal_den).bfloat16()),
                numerator_only_counterfactual=float(ideal_acc[j]/den.double()),
                numerator_only_bf16=float((ideal_acc[j]/den.double()).bfloat16()),
                exact_online_weights_bf16=float(ideal_out[j].bfloat16())))
        report.append(dict(case=prefix,qkv_equal=qkv_equal,accumulator_dtype='F32',
            output_exact_observer=torch.equal(t['native'],t['observed']),
            pre_round_cast_exact=torch.equal(t['pre_round'].bfloat16(),t['native']),
            reference_scale=float(scale),native_scale=float(t['scale']),
            qk_vs_reference=metrics(t['qk'],qk),native_qk_vs_f64=metrics(t['qk'],exact_score),
            reference_qk_vs_f64=metrics(qk,exact_score),score_vs_reference=metrics(t['score'],score),
            score_using_native_qk_reference_scale=metrics(t['score'],t['qk']*scale),
            exp_same_scores_maxima=metrics(t['current'],same_input_exp),
            rescale_same_maxima=metrics(t['previous'],same_input_rescale),
            first_qk_different_indices=(t['qk']!=qk).nonzero().cpu().tolist(),
            reference_native_scores_output=metrics(same_score_out,ref),
            native_pre_round_vs_reference=metrics(t['pre_round'].flatten(),ref.flatten()),
            native_pre_round_f32_gate_exceedances=int(((t['pre_round'].flatten()-ref.flatten()).abs()>2e-5+2e-5*ref.flatten().abs()).sum()),
            native_denominator=float(den),reference_native_score_denominator=float(reference_den),
            effective_online_probability_mass=float(weights.sum()/den.double()),
            effective_online_probability_vs_reference=metrics(weights/den.double(),same_score_p.flatten()),
            exact_captured_weights_denominator=float(ideal_den),denominator_error=float(den.double()-ideal_den),
            accumulator_vs_exact_captured_weights=metrics(acc,ideal_acc),
            conditional_rounding_bound=dict(unit_roundoff=u,operations=n,gamma=gamma,normal_weights_checked=normal,
                denominator_bound=float(den_bound),denominator_within=bool(abs(den.double()-ideal_den)<=den_bound),
                numerator_max_bound=float(acc_bound.max()),numerator_outside=int(((acc-ideal_acc).abs()>acc_bound).sum()),
                qk_native_outside=int(((t['qk'].double()-exact_score).abs()>qk_bound).sum()),
                qk_reference_outside=int(((qk.double()-exact_score).abs()>qk_bound).sum()),
                limitation='conditions on captured exponential/rescale values; does not certify exp implementation or observable output'),
            strict_old_gate_exceedances=int(bad.sum()),failed_outputs=rows))
    Path(a.output).write_text(json.dumps(dict(cases=report,diagnostic_only=True),indent=2)+'\n')
    for r in report:print(json.dumps({k:v for k,v in r.items() if k!='first_qk_different_indices'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ['inputs','trace','output']:p.add_argument('--'+n,required=True)
    main(p.parse_args())
