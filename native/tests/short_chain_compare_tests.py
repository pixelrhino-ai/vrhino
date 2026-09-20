#!/usr/bin/env python3
"""Synthetic negative tests of trace/continuity admission, no model resources."""
import unittest
import torch
from short_chain_compare import qualification


def fixture():
    c=dict(steps=3,selection=[0,1,0],seed=11,atol=2e-5,rtol=2e-5,
           transitions=[dict(model_timestep=t,sigma=s,next_sigma=n,guidance_scale=g)
                        for t,s,n,g in [(9,.875,.75,4.),(7,.75,.5,3.),(5,.5,.25,2.)]])
    x={'initial_noise':torch.tensor([.1,.2])};history=[]
    for i,row in enumerate(c['transitions']):
        p=f'step.{i}.';s=p+'solver.'
        x[p+'input_latent']=x['initial_noise'] if not i else x[f'step.{i-1}.latent']
        for j,h in enumerate(history):x[s+f'history_before.{j}']=h
        converted=torch.tensor([float(i+1),float(i+2)])
        history=(history+[converted])[-2:]
        for j,h in enumerate(history):x[s+f'history_after.{j}']=h
        x[s+'converted']=converted;x[s+'last_sample']=x[p+'input_latent']
        for name in ['prediction.0','prediction.1','guidance','latent','diagnostic']:
            x[p+name]=torch.tensor([.1*(i+1),.2*(i+1)])
        controls=dict(state_id=0,step=i,instance_id=c['selection'][i],timestep=row['model_timestep'],rng_seed=11,rng_offset=0,
                      history_before=min(i,2),history_after=min(i+1,2),warmup_before=min(i,2),warmup_after=min(i+1,2),
                      order=[1,2,1][i],previous_order=[1,1,2][i])
        x.update({s+k:torch.tensor(v,dtype=torch.int64) for k,v in controls.items()})
        x.update({s+k:torch.tensor(row[k],dtype=torch.float32) for k in ['sigma','next_sigma','guidance_scale']})
    x['final_latent']=x['step.2.latent']
    controls={'calls.cfg_combine':3,'calls.flow_to_x0':3,'calls.multistep_predictor':3,'calls.multistep_corrector':2,'calls.state_advance':3,'calls.rng_normal':0,'rng_after_initialization.seed':11,'rng_after_initialization.offset':0,'shared_graph':1}
    x.update({k:torch.tensor(v,dtype=torch.int64) for k,v in controls.items()})
    return x,c


class QualificationTests(unittest.TestCase):
    def test_valid(self):
        x,c=fixture();self.assertTrue(qualification(x,x,c)['passed'])

    def test_exact_continuity_and_control_failures(self):
        mutations={
            'latent_link':('step.1.input_latent',torch.tensor([.100001,.2])),
            'history_reset':('step.2.solver.history_before',torch.tensor(0)),
            'history_replaced':('step.2.solver.history_before.0',torch.tensor([99.,99.])),
            'warmup_reset':('step.2.solver.warmup_before',torch.tensor(0)),
            'order_reset':('step.2.solver.previous_order',torch.tensor(1)),
            'wrong_instance':('step.1.solver.instance_id',torch.tensor(0)),
            'guidance_from_binding':('step.2.solver.guidance_scale',torch.tensor(4.)),
            'rng_draw':('step.1.solver.rng_offset',torch.tensor(2)),
            'extra_solver':('step.2.solver.state_id',torch.tensor(1)),
            'nonfinite_diagnostic':('step.0.diagnostic',torch.tensor([float('nan'),0.])),
            'dtype_drift':('step.0.diagnostic',torch.tensor([.1,.2],dtype=torch.float64)),
            'shape_drift':('step.0.diagnostic',torch.tensor([[.1,.2]])),
        }
        for name,(key,value) in mutations.items():
            with self.subTest(name=name):
                x,c=fixture();bad=dict(x);bad[key]=value
                with self.assertRaises(ValueError):qualification(bad,x,c)

    def test_missing_or_extra_checkpoint(self):
        x,c=fixture();bad=dict(x);del bad['step.1.diagnostic']
        with self.assertRaises(ValueError):qualification(bad,x,c)
        bad=dict(x);bad['extra']=torch.tensor(1)
        with self.assertRaises(ValueError):qualification(bad,x,c)

    def test_observable_cannot_be_waived(self):
        x,c=fixture();bad=dict(x);bad['step.1.prediction.0']=x['step.1.prediction.0']+1
        self.assertFalse(qualification(bad,x,c)['passed'])

    def test_diagnostic_exceedance_is_retained(self):
        x,c=fixture();bad=dict(x);bad['step.1.diagnostic']=x['step.1.diagnostic']+1
        r=qualification(bad,x,c)
        self.assertTrue(r['passed']);self.assertIn('step.1.diagnostic',r['diagnostic_exceedances'])

    def test_tolerance_drift_rejected(self):
        x,c=fixture();c['atol']=1e-3
        with self.assertRaises(ValueError):qualification(x,x,c)

    def test_redundant_prediction_alias_must_be_exact(self):
        x,c=fixture();ref=dict(x);ref['step.0.branch.0.prediction']=x['step.0.prediction.0']
        self.assertTrue(qualification(x,ref,c)['passed'])
        ref['step.0.branch.0.prediction']=x['step.0.prediction.0']+1
        with self.assertRaises(ValueError):qualification(x,ref,c)
        ref['step.0.branch.0.prediction']=x['step.0.prediction.0'].double()
        with self.assertRaises(ValueError):qualification(x,ref,c)


if __name__=='__main__':unittest.main()
