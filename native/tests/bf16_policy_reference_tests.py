import unittest
import torch
import torch.nn.functional as F
from bf16_policy_reference import PolicyMath


class LinearRounding(unittest.TestCase):
    def policy(self):
        p = PolicyMath.__new__(PolicyMath)
        p.ops = {'LINEAR': {'compute_dtype': 'bf16', 'output_dtype': 'bf16'}}
        return p

    def test_bias_precedes_output_rounding(self):
        x = torch.tensor([[1., 1.]])
        w = torch.tensor([[1., 1/256]])
        b = torch.tensor([1/256])
        result = self.policy().linear(x, w, b)
        double_round = (F.linear(x, w).bfloat16().float() + b).bfloat16()
        self.assertEqual(result.item(), 1 + 1/128)
        self.assertEqual(double_round.item(), 1.)

    def test_operand_rounding_is_not_bypassed(self):
        x = torch.tensor([[1.001]])
        w = torch.tensor([[1.]])
        self.assertEqual(self.policy().linear(x, w).item(), 1.)

    def test_modulation_remains_f32(self):
        x = torch.tensor([[1.001]])
        w = torch.tensor([[1.]])
        self.assertTrue(torch.equal(self.policy().linear(x, w, operation='MODULATION'), x))


if __name__ == '__main__':
    unittest.main()
