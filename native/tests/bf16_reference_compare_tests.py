#!/usr/bin/env python3
"""CPU-only negative tests for BF16 qualification and oracle rounding semantics."""
from pathlib import Path
import unittest
import torch
from bf16_policy_reference import PolicyMath
from bf16_reference_compare import compare


class ContractTests(unittest.TestCase):
    def run_case(self,a,b,role='bf16_path'):
        return compare({'x':a},{'x':b},{'x':role})
    def test_equal(self):
        x=torch.tensor([1.,2.],dtype=torch.bfloat16)
        self.assertTrue(self.run_case(x,x)['passed'])
    def test_adjacent_rounding_is_not_silently_accepted(self):
        a=torch.tensor([5.15625],dtype=torch.bfloat16);b=torch.tensor([5.1875],dtype=torch.bfloat16)
        self.assertFalse(self.run_case(a,b)['passed'])
    def test_dtype_rejected(self):
        with self.assertRaises(ValueError):self.run_case(torch.ones(1),torch.ones(1,dtype=torch.bfloat16))
    def test_shape_rejected(self):
        with self.assertRaises(ValueError):self.run_case(torch.ones(1),torch.ones(2))
    def test_nonfinite_rejected_even_diagnostic(self):
        for value in [float('nan'),float('inf'),-float('inf')]:
            with self.assertRaises(ValueError):self.run_case(torch.tensor([value]),torch.ones(1),'diagnostic')
    def test_missing_key_rejected(self):
        with self.assertRaises(ValueError):compare({}, {'x':torch.ones(1)}, {'x':'bf16_path'})
    def test_missing_role_rejected(self):
        with self.assertRaises(ValueError):compare({'x':torch.ones(1)}, {'x':torch.ones(1)}, {})
    def test_unknown_role_rejected(self):
        with self.assertRaises(ValueError):self.run_case(torch.ones(1),torch.ones(1),'custom_exception')
    def test_f32_role_is_stricter(self):
        self.assertFalse(self.run_case(torch.tensor([1.001]),torch.ones(1),'local_f32')['passed'])
    def test_exact_controls(self):
        self.assertFalse(self.run_case(torch.tensor([2]),torch.tensor([1]),'exact')['passed'])
    def test_diagnostic_record_retained(self):
        r=compare({'x':torch.ones(1),'d':torch.zeros(1)},{'x':torch.ones(1),'d':torch.ones(1)}, {'x':'bf16_path','d':'diagnostic'})
        self.assertTrue(r['passed']);self.assertEqual(r['diagnostic_exceedances'],['d'])
        with self.assertRaises(ValueError):self.run_case(torch.ones(1),torch.ones(1),'diagnostic')
    def test_rms_rounds_normalized_storage_before_affine(self):
        p=PolicyMath(Path(__file__).parent/'fixtures/precision/bf16-operation-contract-v1.json')
        x=torch.tensor([[.3,1.1,-2.7]],dtype=torch.bfloat16);w=torch.tensor([1.125,2.375,.625])
        y=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)
        expected=y.bfloat16().float()*w
        self.assertTrue(torch.equal(p.rms(x,w,1e-6),expected))
        self.assertFalse(torch.equal(expected,y*w))


if __name__=='__main__':unittest.main()
