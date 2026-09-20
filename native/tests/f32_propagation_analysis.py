#!/usr/bin/env python3
"""Research-only fixed-input arithmetic qualification; no model identity rules."""
import argparse
import json
from pathlib import Path
import torch
from f32_attention_reference import read, write, metrics


@torch.no_grad()
def analyze(args):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    source, native = read(args.input), read(args.native)
    expected, accurate, rows = {}, {}, []
    for name in sorted(source):
        if not name.endswith('.kind'): continue
        prefix, kind = name[:-5], int(source[name].item())
        for dtype, target in [(torch.float32, expected), (torch.float64, accurate)]:
            def get(key): return source[prefix+'.'+key].cuda().to(dtype)
            x = get('input')
            if kind == 0:
                weight = get('weight') if prefix+'.weight' in source else None
                bias = get('bias') if prefix+'.bias' in source else None
                y = torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias,
                    float(source[prefix+'.eps'].item()))
            elif kind == 1: y = x * (get('one') + get('scale')) + get('shift')
            elif kind == 2: y = x + get('other')
            elif kind == 3: y = x + get('other') * get('gate')
            else: raise ValueError('Unsupported replay operator')
            target[prefix] = y.cpu()
        actual = native[prefix]
        assert actual.dtype == torch.float32 and actual.shape == expected[prefix].shape
        rows.append(dict(name=prefix, f32=metrics(actual, expected[prefix]),
                         f64=metrics(actual, accurate[prefix]),
                         reference_f64=metrics(expected[prefix], accurate[prefix])))
    assert rows and native.keys() == expected.keys()
    output = Path(args.output)
    write(output.with_suffix('.f32.bundle'), expected)
    # F64 metrics above are evaluated without rounding the diagnostic oracle.
    write(output.with_suffix('.f64-rounded.bundle'), {k:v.float() for k,v in accurate.items()})
    result = dict(passed=all(r['f32']['passed'] and r['f64']['passed'] and
                           r['reference_f64']['passed'] for r in rows),
                  atol=2e-5, rtol=2e-5, checkpoints=rows)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print('fixed-input operators',len(rows),'PASS' if result['passed'] else 'FAIL')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--native', required=True)
    parser.add_argument('--output', required=True)
    raise SystemExit(analyze(parser.parse_args()))
