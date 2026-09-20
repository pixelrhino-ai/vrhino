import unittest
import torch
from bf16_operator_acceptance import compare,radius_bf16

class Acceptance(unittest.TestCase):
    def test_sparse_one_spacing_needs_aggregate_guard(self):
        r=torch.full((1024,),64.,dtype=torch.bfloat16);a=r.clone();a[0]=64.5
        self.assertTrue(compare(a,r,'operator_bf16')['passed'])
        self.assertFalse(compare(a,r,'observable')['passed'])
        a.fill_(64.5);result=compare(a,r,'operator_bf16')
        self.assertEqual(result['pointwise_bad'],0);self.assertFalse(result['rms_pass']);self.assertFalse(result['passed'])
    def test_two_spacing_outlier_rejected(self):
        r=torch.full((1024,),64.,dtype=torch.bfloat16);a=r.clone();a[0]=65.
        self.assertFalse(compare(a,r,'operator_bf16')['passed'])
    def test_zero_and_power_boundaries(self):
        x=torch.tensor([0.,-0.,1.,-1.,64.],dtype=torch.bfloat16);radius=radius_bf16(x)
        self.assertEqual(radius[0],2.**-134);self.assertEqual(radius[1],2.**-134)
        self.assertEqual(radius[2],2.**-8);self.assertEqual(radius[3],2.**-8);self.assertEqual(radius[4],.25)
    def test_max_finite_radius(self):
        x=torch.tensor([torch.finfo(torch.bfloat16).max],dtype=torch.bfloat16)
        self.assertTrue(torch.isfinite(radius_bf16(x)).all());self.assertTrue(compare(x,x,'operator_bf16')['passed'])
    def test_subnormal(self):
        x=torch.tensor([2.**-133,-2.**-133],dtype=torch.bfloat16)
        self.assertTrue(torch.equal(radius_bf16(x),torch.full((2,),2.**-134,dtype=torch.float64)))
    def test_nan_inf_reject_diagnostics(self):
        for v in [float('nan'),float('inf'),-float('inf')]:
            with self.assertRaises(ValueError):compare(torch.tensor([v]),torch.zeros(1),'diagnostic')
    def test_dtype_shape_reject(self):
        with self.assertRaises(ValueError):compare(torch.ones(1),torch.ones(1).bfloat16(),'operator_bf16')
        with self.assertRaises(ValueError):compare(torch.ones(1),torch.ones(2),'f32_semantic')
    def test_f32_gate_unchanged(self):
        self.assertFalse(compare(torch.tensor([1.001]),torch.ones(1),'f32_semantic')['passed'])
    def test_large_exact_integer(self):
        self.assertFalse(compare(torch.tensor([2**60+1]),torch.tensor([2**60]),'exact')['passed'])
    def test_unknown_category_reject(self):
        with self.assertRaises(ValueError):compare(torch.ones(1),torch.ones(1),'custom_override')
    def test_softmax_invariant(self):
        a=torch.zeros(1,4096,dtype=torch.bfloat16);r=torch.full_like(a,1/4096)
        self.assertFalse(compare(a,r,'operator_bf16','softmax')['passed'])
    def test_diagnostic_retains_bad_count(self):
        result=compare(torch.ones(1)*10,torch.ones(1),'diagnostic')
        self.assertTrue(result['passed']);self.assertEqual(result['pointwise_bad'],1)

if __name__=='__main__':unittest.main()
