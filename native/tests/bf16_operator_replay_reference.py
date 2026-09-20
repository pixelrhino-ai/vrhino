#!/usr/bin/env python3
"""Replay frozen operands through independent policy math; never resample inputs."""
import argparse
import torch
import torch.nn.functional as F
from bf16_policy_reference import PolicyMath, configure, read, write


@torch.no_grad()
def replay(inputs, policy):
    p = PolicyMath(policy)
    result = {}
    for key in sorted(inputs):
        if not key.endswith('.kind'):
            continue
        name = key[:-5]
        kind = inputs[key]
        if kind.dtype != torch.int64 or kind.numel() != 1:
            raise ValueError('Invalid operator kind')
        v = {k[len(name)+1:]: x.cuda() for k, x in inputs.items()
             if k.startswith(name + '.') and k != key and not k.endswith('.trace')}
        kind = int(kind.item())
        if kind == 0: y = p.norm(v['x'], v['w'], v['b'])
        elif kind == 1: y = p.linear(v['x'], v['w'], v['b'])
        elif kind == 2: y = p.attention(v['q'], v['k'], v['v'])
        elif kind == 3:
            h = p.linear(v['x'], v['w'], v['b'])
            y = p.linear(F.gelu(h, approximate='tanh'), v['w2'], v['b2'])
        elif kind == 4: y = p.residual(v['x'], v['branch'], v['gate'])
        elif kind == 5: y = p.rms(v['x'], v['w'], 1e-6)
        elif kind == 6: y = F.gelu(v['x'], approximate='tanh')
        elif kind == 7: y = F.softmax(v['x'].float(), -1).to(v['x'].dtype)
        else: raise ValueError('Unknown operator kind')
        result[name] = y.cpu()
    if not result:
        raise ValueError('Empty operator set')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for field in ['input', 'policy', 'output']:
        parser.add_argument('--' + field, required=True)
    args = parser.parse_args()
    configure()
    outputs = replay(read(args.input), args.policy)
    write(args.output, outputs)
    print('REFERENCE_OPERATORS', len(outputs))
