#!/usr/bin/env python3
"""Research-only continuous scheduler oracle; subprocess denoisers own no solver state.

Original pinned scheduler methods execute unchanged. The short schedule, guidance
and opaque instance IDs are explicit experiment inputs, not model phase rules.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import List, Optional, Union, Tuple
import numpy as np
import torch
from wan_numerical_reference import read_bundle, write_bundle


def scheduler(official, contract):
    path=Path(official)/'wan/utils/fm_solvers_unipc.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='FlowUniPCMultistepScheduler')
    names=['__init__','step_index','begin_index','_sigma_to_alpha_sigma_t','convert_model_output',
           'multistep_uni_p_bh_update','multistep_uni_c_bh_update','index_for_timestep','_init_step_index','step']
    selected=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert len(selected)==len(names)
    tree=ast.Module(body=[ast.ClassDef(name='ReferenceScheduler',bases=[],keywords=[],body=selected,decorator_list=[])],type_ignores=[])
    env=dict(np=np,torch=torch,math=math,List=List,Optional=Optional,Union=Union,Tuple=Tuple,SchedulerMixin=object,SchedulerOutput=object,register_to_config=lambda f:f)
    def deprecated(*a,**k): raise RuntimeError('Unexpected deprecated reference path')
    env['deprecate']=deprecated
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),env)
    s=env['ReferenceScheduler'].__new__(env['ReferenceScheduler'])
    s.config=SimpleNamespace(num_train_timesteps=1000,solver_order=2,prediction_type='flow_prediction',shift=1.0,use_dynamic_shifting=False,final_sigmas_type='zero',solver_type='bh2',thresholding=False,lower_order_final=True)
    s.__init__(num_train_timesteps=1000,shift=1.0)
    rows=contract['transitions']
    # Explicit short-chain input schedule; terminal sigma is nonzero because
    # this is a controlled prefix, not a complete generation trajectory.
    s.num_inference_steps=len(rows)
    s.timesteps=torch.tensor([r['model_timestep'] for r in rows],dtype=torch.int64)
    s.sigmas=torch.tensor([r['sigma'] for r in rows]+[rows[-1]['next_sigma']],dtype=torch.float32)
    assert all(rows[i]['next_sigma']==rows[i+1]['sigma'] for i in range(len(rows)-1))
    return s,dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),methods=names)


@torch.no_grad()
def run(args):
    # Optional research policy changes operand/storage boundaries only. The
    # pinned scheduler, its one continuous state and F32 default stay intact.
    if args.policy:
        from bf16_policy_reference import read as load_bundle, write as save_bundle, PolicyMath, configure
        configure(); policy = PolicyMath(args.policy)
    else:
        load_bundle, save_bundle = read_bundle, write_bundle
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    contract=json.loads(Path(args.contract).read_text());input_path=Path(contract['input'])
    assert hashlib.sha256(input_path.read_bytes()).hexdigest()==contract['input_sha256']
    assert contract['steps']==len(contract['selection'])==len(contract['transitions'])
    assert subprocess.check_output(['git','-C',args.official,'rev-parse','HEAD'],text=True).strip()=='42bf4cfaa384bc21833865abc2f9e6c0e67233dc'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.manual_seed(contract['seed']);torch.cuda.manual_seed_all(contract['seed'])
    initial=load_bundle(input_path);latent=initial['latent'].cuda()
    if args.policy: latent=latent.to(policy.role('SAMPLING_STATE'))
    rng_cpu=torch.get_rng_state().clone();rng_cuda=torch.cuda.get_rng_state().clone()
    s,provenance=scheduler(args.official,contract)
    result={'initial_noise':latent.cpu().clone()};sources={0:args.component_a,1:args.component_b}
    state_identity=id(s);history_records=[]
    for i,(selected,row) in enumerate(zip(contract['selection'],contract['transitions'])):
        assert id(s)==state_identity
        prefix=f'step.{i}.';sp=prefix+'solver.'
        x=dict(initial);x['latent']=latent.cpu();x['timestep']=torch.tensor([row['model_timestep']],dtype=torch.int64)
        inp=root/f'step-{i}-input.bundle';save_bundle(inp,x)
        result[prefix+'input_latent']=latent.cpu().clone()
        history=[v for v in s.model_outputs if v is not None]
        before=len(history);previous=s.this_order if i else 1;warmup=s.lower_order_nums
        for j,value in enumerate(history):result[sp+f'history_before.{j}']=value.cpu().clone()
        out=root/f'step-{i}'
        command=[sys.executable,str(Path(__file__).with_name('bf16_architecture_reference.py'))] if args.policy else [sys.executable,str(Path(__file__).with_name('wan_numerical_reference.py')),'full']
        command+=['--official',args.official,'--source',sources[selected],'--inputs',str(inp),'--output',str(out),'--timestep',str(row['model_timestep'])]
        if args.policy: command+=['--policy',args.policy]
        with (root/f'step-{i}.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        predictions=load_bundle(out/'reference-full.bundle')
        result.update({prefix+k:v for k,v in predictions.items()})
        p0=predictions['branch.0.prediction'].cuda();p1=predictions['branch.1.prediction'].cuda()
        result[prefix+'prediction.0']=p0.cpu();result[prefix+'prediction.1']=p1.cpu()
        guided=p0+row['guidance_scale']*(p1-p0);result[prefix+'guidance']=guided.cpu()
        latent=s.step(guided,s.timesteps[i],latent,return_dict=False)[0]
        assert s.step_index==i+1
        history=[v for v in s.model_outputs if v is not None]
        for j,value in enumerate(history):result[sp+f'history_after.{j}']=value.cpu().clone()
        result[sp+'converted']=history[-1].cpu().clone();result[sp+'last_sample']=s.last_sample.cpu().clone()
        controls=dict(state_id=0,step=i,instance_id=selected,timestep=row['model_timestep'],rng_seed=contract['seed'],rng_offset=0,history_before=before,warmup_before=warmup,previous_order=previous,order=s.this_order,warmup_after=s.lower_order_nums,history_after=len(history))
        result.update({sp+k:torch.tensor(v,dtype=torch.int64) for k,v in controls.items()})
        result.update({sp+k:torch.tensor(row[k],dtype=torch.float32) for k in ['sigma','next_sigma','guidance_scale']})
        result[prefix+'latent']=latent.cpu().clone()
        assert torch.equal(rng_cpu,torch.get_rng_state()) and torch.equal(rng_cuda,torch.cuda.get_rng_state())
        history_records.append(dict(step=i,instance_id=selected,history_before=before,history_after=len(history),order=s.this_order,warmup=s.lower_order_nums,rng_unchanged=True))
        print('Reference completed',history_records[-1],flush=True)
        save_bundle(root/'partial.bundle',result)
    result['final_latent']=latent.cpu()
    for key,value in {'calls.cfg_combine':len(contract['selection']),'calls.flow_to_x0':s.step_index,'calls.multistep_predictor':s.step_index,'calls.multistep_corrector':s.step_index-1,'calls.state_advance':s.step_index,'calls.rng_normal':0,'rng_after_initialization.seed':contract['seed'],'rng_after_initialization.offset':0,'shared_graph':1}.items():result[key]=torch.tensor(value,dtype=torch.int64)
    save_bundle(root/'reference.bundle',result)
    (root/'reference.json').write_text(json.dumps(dict(scheduler=provenance,steps=history_records,continuous_scheduler=True,rng_cpu_unchanged=True,rng_cuda_unchanged=True,scope='One continuous reference solver; isolated denoiser processes release parameter resources only',
        state_dtype=str(latent.dtype), policy_sha256=hashlib.sha256(Path(args.policy).read_bytes()).hexdigest() if args.policy else None,
        scalar_arithmetic='Pinned upstream scheduler expressions, no rounding rewrite'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['contract','official','component-a','component-b','output']:p.add_argument('--'+k,required=True)
    p.add_argument('--policy',help='Optional explicit BF16 research operation contract')
    run(p.parse_args())
