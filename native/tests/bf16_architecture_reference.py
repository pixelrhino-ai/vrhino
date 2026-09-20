#!/usr/bin/env python3
"""Research-only pinned family forward with explicit generic policy boundaries.

No native output is used as an input. No production module imports this oracle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import types
import torch
import torch.nn.functional as F
from wan_numerical_reference import Source, oracle
from bf16_policy_reference import PolicyMath, configure, read, write


def install_policy(model, env, policy):
    replacements = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            operation = 'MODULATION' if name.startswith(('time_embedding.', 'time_projection.')) else 'LINEAR'
            def linear(m, x, op=operation):
                return policy.linear(x, m.weight, m.bias, op)
            module.forward = types.MethodType(linear, module)
            replacements[name] = operation
        elif isinstance(module, torch.nn.Conv3d):
            def convolution(m, x):
                dtype = policy.dtype(policy.ops['CONVOLUTION']['compute_dtype'])
                output = F.conv3d(x.to(dtype), m.weight.to(dtype), None,
                                  m.stride, m.padding, m.dilation, m.groups)
                # Existing convolution boundary: output rounding before bias.
                if m.bias is not None:
                    output = output + m.bias.to(dtype).view(1, -1, 1, 1, 1)
                return output.to(policy.dtype(policy.ops['CONVOLUTION']['output_dtype']))
            module.forward = types.MethodType(convolution, module)
            replacements[name] = 'CONVOLUTION'
        elif isinstance(module, env['WanRMSNorm']):
            module.forward = types.MethodType(lambda m, x: policy.rms(x, m.weight, m.eps), module)
            replacements[name] = 'RMS_NORMALIZATION'
        elif isinstance(module, torch.nn.LayerNorm):
            module.forward = types.MethodType(lambda m, x: policy.norm(x, m.weight, m.bias, m.eps), module)
            replacements[name] = 'LAYER_NORMALIZATION'

    def attention(q, k, v, k_lens=None, window_size=(-1, -1), **kwargs):
        assert window_size == (-1, -1) and not kwargs
        assert k_lens is None or bool((k_lens == k.shape[1]).all())
        return policy.attention(q, k, v)

    def rope(x, grids, freqs):
        half = x.shape[-1] // 2
        parts = freqs.split([half - 2*(half//3), half//3, half//3], dim=1)
        outputs = []
        for i, (t, h, w) in enumerate(grids.tolist()):
            angles = torch.cat([
                parts[0][:t].view(t, 1, 1, -1).expand(t, h, w, -1),
                parts[1][:h].view(1, h, 1, -1).expand(t, h, w, -1),
                parts[2][:w].view(1, 1, w, -1).expand(t, h, w, -1)], -1).reshape(t*h*w, 1, -1)
            dtype = policy.dtype(policy.ops['ACTIVATION']['output_dtype'])
            cos = angles.real.float().to(dtype).float().repeat_interleave(2, -1)
            sin = angles.imag.float().to(dtype).float().repeat_interleave(2, -1)
            rotated = policy.rope(x[i, :t*h*w], cos, sin)
            outputs.append(torch.cat([rotated, x[i, t*h*w:].float()]))
        return torch.stack(outputs)

    env['flash_attention'] = attention
    env['rope_apply'] = rope
    return replacements


@torch.no_grad()
def run(args):
    configure(); torch.manual_seed(20260917); torch.cuda.manual_seed_all(20260917)
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git', '-C', args.official, 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == '42bf4cfaa384bc21833865abc2f9e6c0e67233dc'
    policy = PolicyMath(args.policy); env = oracle(args.official); source = Source(args.source)
    cfg = json.loads((Path(args.source)/'config.json').read_text())
    with torch.device('meta'):
        model = env['WanModel'](**{k:v for k,v in cfg.items() if not k.startswith('_')})
    for name in ['patch_embedding', 'time_embedding', 'time_projection', 'text_embedding', 'head']:
        source.load(getattr(model, name), name+'.')
    for i, block in enumerate(model.blocks):
        source.load(block, f'blocks.{i}.')
        print('loaded block', i, flush=True)
    assert len(source.hashes) == 1095
    model.eval()
    # Non-parameter positional frequencies were constructed under meta along
    # with the module. Materialize them from the pinned upstream definition.
    d = model.dim // model.num_heads
    model.freqs = torch.cat([env['rope_params'](1024, width)
        for width in [d - 4*(d//6), 2*(d//6), 2*(d//6)]], dim=1).cuda()
    replacements = install_policy(model, env, policy)
    x = {k:v.cuda() for k,v in read(args.inputs).items()}
    x['latent'] = x['latent'].to(policy.role('SAMPLING_STATE'))
    x['timestep'] = torch.tensor([args.timestep], dtype=torch.int64, device='cuda')
    tokens = x['latent'].shape[2]*x['latent'].shape[3]*x['latent'].shape[4] // (model.patch_size[0]*model.patch_size[1]*model.patch_size[2])
    result = {}; branch = 0
    def hook(module, name, transform=lambda v:v):
        def callback(m, inp, out): result[f'branch.{branch}.{name}'] = transform(out).detach().cpu().clone()
        module.register_forward_hook(callback)
    hook(model.patch_embedding, 'input_projection')
    hook(model.time_embedding, 'conditioning.time', lambda v:v[:, 0])
    hook(model.time_projection, 'conditioning.modulation', lambda v:v[:, 0].reshape(1, 6, model.dim))
    hook(model.text_embedding, 'conditioning.context')
    for i, block in enumerate(model.blocks): hook(block, f'block.{i}.output')
    hook(model.head, 'final_projection')
    rng_cpu = torch.get_rng_state().clone(); rng_gpu = torch.cuda.get_rng_state().clone()
    for branch, context in enumerate([torch.zeros_like(x['raw_context']), x['raw_context']]):
        y = model([x['latent'][0]], x['timestep'], [context[0]], tokens)
        result[f'branch.{branch}.prediction'] = torch.stack(y).to(policy.role('DENOISER_OUTPUT')).cpu()
        for alias, i in [('block_0', 0), ('middle_block', len(model.blocks)//2), ('last_block', len(model.blocks)-1)]:
            result[f'branch.{branch}.{alias}'] = result[f'branch.{branch}.block.{i}.output']
        print('evaluated branch', branch, flush=True)
    assert torch.equal(rng_cpu, torch.get_rng_state()) and torch.equal(rng_gpu, torch.cuda.get_rng_state())
    assert all(torch.isfinite(v).all() for v in result.values())
    torch.cuda.synchronize(); write(root/'reference-full.bundle', result)
    metadata = dict(source_revision=revision, source_tensor_hashes=source.hashes, timestep=args.timestep,
        policy_sha256=hashlib.sha256(Path(args.policy).read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),
        forward_sha256=hashlib.sha256((Path(args.official)/'wan/modules/model.py').read_bytes()).hexdigest(),
        operator_lowering=replacements, precision='BF16 operands/output, existing F32 semantic roles',
        peak_device_allocated=torch.cuda.max_memory_allocated(), rng_unchanged=True,
        reference='Pinned official forward; explicit test-only policy operator boundaries', checkpoints=len(result))
    (root/'reference-full.json').write_text(json.dumps(metadata, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ['official', 'source', 'inputs', 'output', 'policy']: parser.add_argument('--'+key, required=True)
    parser.add_argument('--timestep', required=True, type=int)
    run(parser.parse_args())
