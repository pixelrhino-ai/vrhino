#!/usr/bin/env python3
"""Research comparison gates reject metadata drift/nonfinite/numerical failures."""
import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import torch
import wan_numerical_reference as research

class ComparisonTests(unittest.TestCase):
    def run_pair(self, expected, actual):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root)
            research.write_bundle(root/'e',expected); research.write_bundle(root/'a',actual)
            with contextlib.redirect_stdout(io.StringIO()):
                return research.compare(SimpleNamespace(expected=root/'e',actual=root/'a',output=root/'result'))
    def test_exact(self):
        x={'param.0':torch.arange(8,dtype=torch.float32)}
        self.assertEqual(self.run_pair(x,x),0)
    def test_small_roundoff(self):
        self.assertEqual(self.run_pair({'x':torch.ones(8)},{'x':torch.ones(8)+1e-6}),0)
    def test_error(self):
        self.assertEqual(self.run_pair({'x':torch.ones(8)},{'x':torch.zeros(8)}),1)
    def test_passing_output_does_not_hide_failed_intermediate(self):
        expected={'intermediate':torch.zeros(1),'output':torch.ones(1)}
        actual={'intermediate':torch.full((1,),3e-5),'output':torch.ones(1)+1e-6}
        self.assertEqual(self.run_pair(expected,actual),1)
    def test_mixed_gate_is_not_absolute_only(self):
        self.assertEqual(self.run_pair({'x':torch.ones(1)},
                                      {'x':torch.ones(1)+3e-5}),0)
    def test_nonfinite(self):
        for v in [float('nan'),float('inf'),float('-inf')]:
            self.assertEqual(self.run_pair({'x':torch.ones(1)},{'x':torch.tensor([v])}),1)
    def test_shape(self):
        with self.assertRaises(AssertionError):self.run_pair({'x':torch.ones(2)},{'x':torch.ones(1,2)})
    def test_dtype(self):
        with self.assertRaises(AssertionError):self.run_pair({'x':torch.ones(2)},{'x':torch.ones(2,dtype=torch.int64)})
    def test_names(self):
        with self.assertRaises(AssertionError):self.run_pair({'x':torch.ones(2)},{'y':torch.ones(2)})
    def test_trailing_bytes(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)/'a';research.write_bundle(p,{'x':torch.ones(2)})
            with p.open('ab') as f:f.write(b'x')
            with self.assertRaises(AssertionError):research.read_bundle(p)
if __name__=='__main__':unittest.main()
