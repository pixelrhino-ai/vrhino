import tempfile
from pathlib import Path
import struct
import unittest
import torch
from bf16_bringup_analysis import analyze, read


class BF16AnalysisTests(unittest.TestCase):
    def test_raw_bf16_roundtrip_known_bits(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.bundle'
            p.write_bytes(b'VRINPUT1'+struct.pack('<IH',1,1)+b'x'+struct.pack('<BBqQ4H',2,1,4,8,0x3f80,0xbf80,0x0000,0x8000))
            x=read(p)['x'];self.assertEqual(x.dtype,torch.bfloat16)
            self.assertEqual(x.float().tolist(),[1.,-1.,0.,-0.]);self.assertTrue(torch.signbit(x[-1]))

    def test_nan_inf_rejected(self):
        ref={'x':torch.ones(2)}
        for v in [float('nan'),float('inf'),-float('inf')]:
            with self.subTest(value=v),self.assertRaises(ValueError):
                analyze({'x':torch.tensor([1.,v],dtype=torch.bfloat16)},ref)

    def test_shape_and_control_drift_rejected(self):
        with self.assertRaises(ValueError):analyze({'x':torch.ones(1,2,dtype=torch.bfloat16)},{'x':torch.ones(2)})
        with self.assertRaises(ValueError):analyze({'id':torch.tensor(1)},{'id':torch.tensor(0)})

    def test_all_diagnostics_retained_no_silent_agreement_claim(self):
        result=analyze({'x':torch.tensor([1.,2.],dtype=torch.bfloat16)},{'x':torch.tensor([1.,1.])})
        self.assertTrue(result['finite']);self.assertNotIn('numerical_equivalence_pass',result)
        self.assertEqual(result['checkpoints'][0]['diagnostic_exceedances_5e3'],1)
        self.assertEqual(result['checkpoints'][0]['max_abs'],1.)

    def test_keys_fail_closed(self):
        with self.assertRaises(ValueError):analyze({'x':torch.ones(2)},{'y':torch.ones(2)})


if __name__=='__main__':unittest.main()
