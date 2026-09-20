import unittest
import torch
from bf16_acceptance_contract import qualify
from bf16_operator_acceptance import SCHEMA


class Contract(unittest.TestCase):
    def declaration(self, category='observable_output', precision='fp32', operator=None, scope='architecture'):
        return dict(schema=SCHEMA, scope=scope, checkpoints={'output': dict(
            category=category, precision=precision, operator=operator, reason='Declared test result')})

    def test_observable_cannot_use_operator_allowance(self):
        ref = {'output': torch.full((1024,), 64., dtype=torch.bfloat16)}
        actual = {'output': ref['output'].clone()}; actual['output'][0] = 64.5
        self.assertFalse(qualify(actual, ref, self.declaration(precision='bf16'))['passed'])
        self.assertTrue(qualify(actual, ref, self.declaration('operator_output', 'bf16', 'linear', 'operator'))['passed'])

    def test_diagnostic_only_fails(self):
        values = {'output': torch.ones(1)}
        with self.assertRaises(ValueError):
            qualify(values, values, self.declaration('internal_diagnostic'))

    def test_missing_extra_checkpoints_fail(self):
        values = {'output': torch.ones(1)}
        with self.assertRaises(ValueError): qualify({}, values, self.declaration())
        with self.assertRaises(ValueError): qualify({**values, 'extra': torch.ones(1)}, values, self.declaration())

    def test_no_per_run_tolerance(self):
        values = {'output': torch.ones(1)}; d = self.declaration()
        d['checkpoints']['output']['atol'] = 1.
        with self.assertRaises(ValueError): qualify(values, values, d)

    def test_nonfinite_diagnostic_with_strict_output_fails(self):
        values = {'output': torch.ones(1), 'internal': torch.tensor([float('nan')])}
        d = self.declaration(); d['checkpoints']['internal'] = dict(
            category='internal_diagnostic', precision='fp32', operator=None, reason='Localization')
        with self.assertRaises(ValueError): qualify(values, values, d)

    def test_declared_dtype_and_schema_fail_closed(self):
        values = {'output': torch.ones(1)}
        with self.assertRaises(ValueError): qualify(values, values, self.declaration(precision='bf16'))
        d = self.declaration(); d['schema'] = 'unknown'
        with self.assertRaises(ValueError): qualify(values, values, d)


if __name__ == '__main__': unittest.main()
