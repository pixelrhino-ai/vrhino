#!/usr/bin/env python3
"""Fail-closed layered short-chain comparison, with exact continuity invariants."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from wan_numerical_reference import read_bundle
from numerical_observable_compare import compare


def require(value, message):
    if not value: raise ValueError(message)


def validate_continuity(trace, contract):
    n=contract['steps']; rows=contract['transitions']; selected=contract['selection']
    require(3<=n<=8 and len(rows)==len(selected)==n,'Short-chain declaration')
    def exact(a,b,label):
        require(a.shape==b.shape and a.dtype==b.dtype and torch.equal(a,b),label)
    def scalar(key,value):
        x=trace[key];require(x.ndim==0 and x.item()==value,'Control mismatch: '+key)
    orders=[]
    for i in range(n):
        prefix=f'step.{i}.';sp=prefix+'solver.';order=min(2,n-i,i+1);orders.append(order)
        controls=dict(state_id=0,step=i,instance_id=selected[i],timestep=rows[i]['model_timestep'],
                      rng_seed=contract['seed'],rng_offset=0,history_before=min(i,2),history_after=min(i+1,2),
                      warmup_before=min(i,2),warmup_after=min(i+1,2),order=order,previous_order=orders[i-1] if i else 1)
        for k,v in controls.items():scalar(sp+k,v)
        for k in ['sigma','next_sigma','guidance_scale']:scalar(sp+k,rows[i][k])
        exact(trace[prefix+'input_latent'],trace['initial_noise'] if i==0 else trace[f'step.{i-1}.latent'],'Latent continuity')
        for j in range(min(i,2)):
            exact(trace[sp+f'history_before.{j}'],trace[f'step.{i-1}.solver.history_after.{j}'],'Previous history was reset or changed')
        for j in range(min(i+1,2)):
            origin=i+1-min(i+1,2)+j
            exact(trace[sp+f'history_after.{j}'],trace[f'step.{origin}.solver.converted'],'History output provenance')
    exact(trace['final_latent'],trace[f'step.{n-1}.latent'],'Final output continuity')
    for k,v in {'cfg_combine':n,'flow_to_x0':n,'multistep_predictor':n,'multistep_corrector':n-1,'state_advance':n,'rng_normal':0}.items():scalar('calls.'+k,v)
    scalar('rng_after_initialization.seed',contract['seed']);scalar('rng_after_initialization.offset',0);scalar('shared_graph',1)
    return dict(steps=n,orders=orders,history_lengths=[min(i+1,2) for i in range(n)],
                a_to_b=any(a==0 and b==1 for a,b in zip(selected,selected[1:])),
                b_to_a=any(a==1 and b==0 for a,b in zip(selected,selected[1:])),
                a_to_b_to_a=any(selected[i:i+3]==[0,1,0] for i in range(n-2)),
                exact_latent_links=True,exact_history_provenance=True,rng_draws=0)


def qualification(native,reference,contract):
    require(contract['atol']==contract['rtol']==2e-5,'Frozen numerical contract changed')
    # The denoiser-only oracle names returns branch.N.prediction; the sampler
    # names the same returns prediction.N. Its raw evidence contains both.
    # Validate redundant aliases exactly before canonicalizing trace keys;
    # no internal checkpoint or prediction value is discarded from evidence.
    reference=dict(reference)
    aliases=[]
    for i in range(contract['steps']):
        for branch in [0,1]:
            alias=f'step.{i}.branch.{branch}.prediction'
            if alias in reference:
                value=reference[alias];canonical=reference[f'step.{i}.prediction.{branch}']
                require(value.shape==canonical.shape and value.dtype==canonical.dtype and
                        torch.equal(value,canonical),'Reference prediction alias mismatch')
                del reference[alias];aliases.append(alias)
    observations=['initial_noise','final_latent']
    for i in range(contract['steps']):
        for suffix in ['input_latent','prediction.0','prediction.1','guidance','latent']:
            observations.append(f'step.{i}.'+suffix)
    metrics=compare(reference,native,observations)
    metrics['exact_redundant_reference_aliases']=aliases
    # Control values are exact, not subject to a floating-point tolerance.
    for key in reference:
        if reference[key].ndim==0:
            require(torch.equal(reference[key],native[key]),'Exact control drift: '+key)
    metrics['native_continuity']=validate_continuity(native,contract)
    metrics['reference_continuity']=validate_continuity(reference,contract)
    require(torch.equal(native['initial_noise'],reference['initial_noise']),'Initial input replay')
    return metrics


def main(args):
    contract_path=Path(args.contract);c=json.loads(contract_path.read_text())
    require(hashlib.sha256(Path(c['input']).read_bytes()).hexdigest()==c['input_sha256'],'Input identity drift')
    native=read_bundle(args.native);reference=read_bundle(args.reference)
    result=qualification(native,reference,c)
    frozen=read_bundle(c['input'])
    require(torch.equal(frozen['latent'],native['initial_noise']),'Frozen input not used')
    result['contract_sha256']=hashlib.sha256(contract_path.read_bytes()).hexdigest()
    result['input_replay_pass']=True
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    for row in result['checkpoints']:
        if row['role']=='observable':print(json.dumps(row))
    print('diagnostic exceedances retained:',len(result['diagnostic_exceedances']))
    print('SHORT_CHAIN_PASS=',result['passed'])
    return 0 if result['passed'] else 1


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['contract','native','reference','output']:p.add_argument('--'+name,required=True)
    raise SystemExit(main(p.parse_args()))
