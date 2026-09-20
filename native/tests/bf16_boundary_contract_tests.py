import unittest
import torch
from bf16_boundary_contract import qualify

class Boundaries(unittest.TestCase):
    def declaration(self,scope='architecture'):
        category='observable_output' if scope=='architecture' else 'operator_output'
        return dict(schema='generic.bf16.boundaries.v2',scope=scope,checkpoints={
            'out':dict(category=category,precision='bf16_path',reason='Declared public result'),
            'inner':dict(category='internal_diagnostic',precision='bf16_path',reason='Composed internal value')})
    def inputs(self):return {'out':torch.ones(2,dtype=torch.bfloat16),'inner':torch.ones(2,dtype=torch.bfloat16)}
    def test_internal_metrics_retained(self):
        a=self.inputs();b=self.inputs();a['inner']*=10
        r=qualify(a,b,self.declaration());self.assertTrue(r['passed']);self.assertEqual(r['diagnostic_exceedances'],['inner'])
    def test_observable_stays_strict(self):
        a=self.inputs();a['out']*=2;self.assertFalse(qualify(a,self.inputs(),self.declaration())['passed'])
    def test_operator_stays_strict(self):
        a=self.inputs();a['out']*=2;self.assertFalse(qualify(a,self.inputs(),self.declaration('operator'))['passed'])
    def test_no_observable_cannot_pass(self):
        d=self.declaration();d['checkpoints']['out']['category']='internal_diagnostic'
        with self.assertRaises(ValueError):qualify(self.inputs(),self.inputs(),d)
    def test_no_nan_diagnostic_waiver(self):
        a=self.inputs();a['inner'][0]=float('nan')
        with self.assertRaises(ValueError):qualify(a,self.inputs(),self.declaration())
    def test_no_missing_checkpoint(self):
        d=self.declaration();del d['checkpoints']['inner']
        with self.assertRaises(ValueError):qualify(self.inputs(),self.inputs(),d)
    def test_no_custom_tolerance(self):
        d=self.declaration();d['checkpoints']['out']['atol']=1
        with self.assertRaises(ValueError):qualify(self.inputs(),self.inputs(),d)
    def test_no_unexplained_classification(self):
        d=self.declaration();d['checkpoints']['out']['reason']=''
        with self.assertRaises(ValueError):qualify(self.inputs(),self.inputs(),d)

if __name__=='__main__':unittest.main()
