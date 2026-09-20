import unittest
import torch
from numerical_observable_compare import compare


class BoundaryComparisonTests(unittest.TestCase):
    def fixture(self): return {'internal':torch.zeros(2),'output':torch.ones(2)}
    def test_diagnostic_error_is_retained(self):
        e=self.fixture();a=self.fixture();a['internal']+=0.1
        r=compare(e,a,['output']);self.assertTrue(r['passed'])
        self.assertEqual(r['diagnostic_exceedances'],['internal'])
        self.assertEqual(len(r['checkpoints']),2)
    def test_observable_error_fails(self):
        e=self.fixture();a=self.fixture();a['output']+=0.1
        self.assertFalse(compare(e,a,['output'])['passed'])
    def test_local_operator_output_remains_strict(self):
        self.assertFalse(compare({'op':torch.zeros(1)},{'op':torch.ones(1)},['op'])['passed'])
    def test_invalid_role_set(self):
        for roles in [[],['missing'],['output','output']]:
            with self.assertRaises(ValueError): compare(self.fixture(),self.fixture(),roles)
    def test_missing_diagnostic_rejected(self):
        with self.assertRaises(ValueError):compare(self.fixture(),{'output':torch.ones(2)},['output'])
    def test_nonfinite_diagnostic_rejected(self):
        for bad in [float('nan'),float('inf')]:
            a=self.fixture();a['internal'][0]=bad
            with self.assertRaises(ValueError): compare(self.fixture(),a,['output'])
    def test_shape_dtype_rejected(self):
        for value in [torch.zeros(1,2),torch.zeros(2,dtype=torch.float64)]:
            a=self.fixture();a['internal']=value
            with self.assertRaises(ValueError):compare(self.fixture(),a,['output'])


if __name__=='__main__':unittest.main()
