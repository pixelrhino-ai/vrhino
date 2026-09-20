#!/usr/bin/env python3
"""Pinned upstream scheduler methods as a research oracle, never production."""
import argparse
import ast
import hashlib
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Union
import numpy as np
import torch
from f32_attention_reference import read, write


def assignment(path, field):
    values=[]
    for n in ast.parse(path.read_text()).body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Attribute) and n.targets[0].attr==field:
            values.append(ast.literal_eval(n.value))
    assert len(values)==1,field
    return values[0]


@torch.no_grad()
def main(args):
    official=Path(args.official)
    assert subprocess.check_output(['git','-C',str(official),'rev-parse','HEAD'],text=True).strip()=='42bf4cfaa384bc21833865abc2f9e6c0e67233dc'
    path=official/'wan/utils/fm_solvers_unipc.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='FlowUniPCMultistepScheduler')
    methods=['__init__','step_index','set_timesteps','_sigma_to_alpha_sigma_t','convert_model_output','multistep_uni_p_bh_update']
    selected=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in methods]
    assert len(selected)==len(methods)
    tree=ast.Module(body=[ast.ClassDef(name='ReferenceScheduler',bases=[],keywords=[],body=selected,decorator_list=[])],type_ignores=[])
    env=dict(np=np,torch=torch,math=math,List=List,Optional=Optional,Union=Union,SchedulerMixin=object,register_to_config=lambda f:f)
    def deprecated(*a,**k): raise RuntimeError('Unexpected deprecated reference path')
    env['deprecate']=deprecated
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),env)
    config=official/'wan/configs/wan_t2v_A14B.py';shared=official/'wan/configs/shared_config.py'
    train=assignment(shared,'num_train_timesteps');steps=assignment(config,'sample_steps')
    shift=assignment(config,'sample_shift');boundary=assignment(config,'boundary')*train
    scales=assignment(config,'sample_guide_scale')
    scheduler=env['ReferenceScheduler'].__new__(env['ReferenceScheduler'])
    scheduler.config=SimpleNamespace(num_train_timesteps=train,solver_order=2,prediction_type='flow_prediction',shift=1.0,use_dynamic_shifting=False,final_sigmas_type='zero',solver_type='bh2',thresholding=False)
    scheduler.__init__(num_train_timesteps=train,shift=1.0)
    scheduler.set_timesteps(steps,device='cpu',shift=shift)
    programs=json.loads(Path(args.programs).read_text());transitions=programs['sampling']['transitions']
    assert len(transitions)==steps
    for i,t in enumerate(transitions):
        assert t['model_timestep']==int(scheduler.timesteps[i])
        assert t['sigma']==float(scheduler.sigmas[i]) and t['next_sigma']==float(scheduler.sigmas[i+1])
        selected=int(scheduler.timesteps[i])>=boundary
        assert programs['execution']['instance_ids'][i]==(0 if selected else 1)
        assert t['guidance_scale']==scales[1 if selected else 0]
    x=read(args.inputs)['latent'].cuda();p=read(args.predictions)
    scale=transitions[0]['guidance_scale']
    guided=p['branch.0.prediction'].cuda()+scale*(p['branch.1.prediction'].cuda()-p['branch.0.prediction'].cuda())
    scheduler._step_index=0
    x0=scheduler.convert_model_output(guided,sample=x)
    scheduler.model_outputs=[x0];scheduler.timestep_list=[scheduler.timesteps[0]]
    nxt=scheduler.multistep_uni_p_bh_update(guided,sample=x,order=1)
    output=dict(guided_prediction=guided,x0=x0,next_latent=nxt,
        timestep=scheduler.timesteps[0],sigma=scheduler.sigmas[0],next_sigma=scheduler.sigmas[1],
        guidance_scale=torch.tensor(scale),instance_id=torch.tensor(programs['execution']['instance_ids'][0],dtype=torch.int64))
    write(args.output,output)
    Path(args.output+'.json').write_text(json.dumps(dict(reference_file=str(path),reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),methods=methods,all_schedule_rows_exact=True,rows=steps,solver_order_this_transition=1),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['official','programs','inputs','predictions','output']:p.add_argument('--'+name,required=True)
    main(p.parse_args())
