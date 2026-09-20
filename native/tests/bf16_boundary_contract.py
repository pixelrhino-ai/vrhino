#!/usr/bin/env python3
"""Versioned qualification roles, independent of production PrecisionPolicy."""
from bf16_reference_compare import compare


def qualify(actual,expected,declaration):
    if set(declaration)!={'schema','scope','checkpoints'} or declaration['schema']!='generic.bf16.boundaries.v2':
        raise ValueError('Unknown boundary contract')
    scope=declaration['scope'];checkpoints=declaration['checkpoints']
    required={'operator':'operator_output','architecture':'observable_output'}.get(scope)
    if required is None:raise ValueError('Unknown qualification scope')
    if checkpoints.keys()!=expected.keys():raise ValueError('Incomplete checkpoint declaration')
    roles={};seen=False
    for name,item in checkpoints.items():
        if set(item)!={'category','precision','reason'} or not item['reason']:
            raise ValueError('Incomplete boundary classification')
        category=item['category'];precision=item['precision']
        if category not in ['operator_output','observable_output','internal_diagnostic','exact_control']:
            raise ValueError('Unknown boundary category')
        if precision not in ['bf16_path','local_f32','exact']:raise ValueError('Unknown precision contract')
        if (category=='exact_control')!=(precision=='exact'):raise ValueError('Exact role mismatch')
        seen |= category==required
        roles[name]='diagnostic' if category=='internal_diagnostic' else precision
    if not seen:raise ValueError('No required strict output for scope')
    # Intentionally preserves the frozen comparator; role documentation is not
    # a new tolerance and cannot turn a failed strict output into a pass.
    result=compare(actual,expected,roles)
    result['contract']=declaration
    return result
