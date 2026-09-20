#!/usr/bin/env python3
"""Model-independent counterexamples for the compositional contract design.

These do not replace the strict all-checkpoint comparator or set tolerances.
Powers of two make the simple linear arithmetic exactly representable in F32.
"""
import unittest
import torch
from f32_attention_reference import metrics


class CompositionalExamples(unittest.TestCase):
    def setUp(self):
        self.reference = torch.zeros(1)
        self.rounded = torch.tensor([2.0**-20])
        self.gain = 64.0

    def test_local_gate_does_not_imply_composed_internal_gate(self):
        self.assertTrue(metrics(self.rounded, self.reference)['passed'])
        self.assertFalse(metrics(self.rounded*self.gain, self.reference)['passed'])
        self.assertTrue(metrics((self.rounded*self.gain)/self.gain, self.reference)['passed'])

    def test_signed_linear_propagation_and_envelope(self):
        actual = self.rounded*self.gain
        propagated = (self.rounded.double()-self.reference.double())*self.gain
        self.assertTrue(torch.equal(actual.double(), propagated))
        envelope = (self.rounded.double()-self.reference.double()).abs()*self.gain
        self.assertTrue(bool((actual.abs() <= envelope).all()))

    def test_final_pass_cannot_hide_local_bug(self):
        expected_local = self.rounded*self.gain
        broken_local = expected_local+2.0**-10
        self.assertFalse(metrics(broken_local, expected_local)['passed'])
        self.assertTrue(metrics(broken_local/self.gain, self.reference)['passed'])

    def test_local_pass_cannot_waive_observable_failure(self):
        local = self.rounded*self.gain
        self.assertTrue(metrics(local, local.clone())['passed'])
        self.assertFalse(metrics(local, self.reference)['passed'])

    def test_nonfinite_is_not_propagated_roundoff(self):
        for value in [float('nan'), float('inf'), -float('inf')]:
            self.assertFalse(metrics(torch.tensor([value]), self.reference)['passed'])


if __name__ == '__main__': unittest.main()
