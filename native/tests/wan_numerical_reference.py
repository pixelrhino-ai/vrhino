#!/usr/bin/env python3
"""Research-only oracle. Executes pinned upstream numerical definitions, never imports production.

The upstream flash-attention entry casts F32 to BF16; this F32 experiment substitutes
explicit scaled dot-product math at that ONE boundary. Diffusers serialization mixins
are inert metadata stubs. No block/model forward AST is rewritten.
"""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import numpy as np
import torch


def write_bundle(path, tensors):
    with open(path, 'wb') as f:
        f.write(b'VRINPUT1' + struct.pack('<I', len(tensors)))
        for name, value in sorted(tensors.items()):
            a = value.detach().cpu().contiguous().numpy()
            dt = {np.dtype('float32'): 0, np.dtype('int64'): 3}[a.dtype]
            n = name.encode(); f.write(struct.pack('<H', len(n)) + n)
            f.write(struct.pack('<BB', dt, a.ndim))
            f.write(struct.pack('<'+'q'*a.ndim, *a.shape))
            data = a.tobytes(); f.write(struct.pack('<Q', len(data)) + data)


def read_bundle(path):
    out = {}
    with open(path, 'rb') as f:
        assert f.read(8) == b'VRINPUT1'
        for _ in range(struct.unpack('<I', f.read(4))[0]):
            name = f.read(struct.unpack('<H', f.read(2))[0]).decode()
            dt, rank = struct.unpack('<BB', f.read(2))
            shape = struct.unpack('<'+'q'*rank, f.read(rank*8))
            size = struct.unpack('<Q', f.read(8))[0]
            out[name] = torch.from_numpy(np.frombuffer(f.read(size), dtype={0:'<f4',3:'<i8'}[dt]).copy().reshape(shape))
        assert not f.read(1)
    return out


class Source:
    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
        self.headers = {}; self.hashes = {}
    def tensor(self, name):
        shard = self.index[name]
        if shard not in self.headers:
            with (self.root/shard).open('rb') as f:
                length = struct.unpack('<Q', f.read(8))[0]
                self.headers[shard] = (8+length, json.loads(f.read(length)))
        start, header = self.headers[shard]; entry = header[name]
        assert entry['dtype'] == 'F32'
        lo, hi = entry['data_offsets']; assert hi-lo == math.prod(entry['shape'])*4
        with (self.root/shard).open('rb') as f:
            f.seek(start+lo); data = bytearray(f.read(hi-lo))
        assert len(data) == hi-lo
        self.hashes[name] = hashlib.sha256(data).hexdigest()
        return torch.frombuffer(data, dtype=torch.float32).reshape(entry['shape']).to('cuda')
    def load(self, module, prefix):
        state = {n:self.tensor(prefix+n) for n in module.state_dict()}
        module.load_state_dict(state, strict=True, assign=True)


def oracle(source):
    path = Path(source)/'wan/modules/model.py'
    tree = ast.parse(path.read_text())
    # Remove import statements only; inject their explicit equivalents below.
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    class ConfigMixin: pass
    env = dict(math=math, torch=torch, nn=torch.nn, ConfigMixin=ConfigMixin,
               ModelMixin=torch.nn.Module, register_to_config=lambda f:f)
    exec(compile(tree, str(path), 'exec'), env)
    return env


@torch.no_grad()
def reference(args):
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    revision = subprocess.check_output(['git','-C',args.official,'rev-parse','HEAD'], text=True).strip()
    assert revision == '42bf4cfaa384bc21833865abc2f9e6c0e67233dc'
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260917)
    # All inputs are created ONCE on CPU and serialized; Native reads identical bytes.
    x = dict(latent=torch.randn(1,16,2,2,4), timestep=torch.tensor([999], dtype=torch.int64),
             raw_context=torch.randn(1,3,4096), input=torch.randn(1,4,5120),
             modulation_input=torch.randn(1,6,5120)*0.1, context=torch.randn(1,512,5120)*0.1)
    if args.replay: x=read_bundle(args.replay)
    write_bundle(root/'inputs.bundle', x)
    x = {n:v.cuda() for n,v in x.items()}
    env = oracle(args.official); src = Source(args.source)
    cfg = json.loads((Path(args.source)/'config.json').read_text())
    with torch.device('meta'):
        model = env['WanModel'](**{k:v for k,v in cfg.items() if not k.startswith('_')})
    for name in ['patch_embedding','time_embedding','time_projection','text_embedding']:
        src.load(getattr(model,name),name+'.')
    src.load(model.blocks[0], 'blocks.0.')
    trace = {}; save = lambda name,t: trace.__setitem__(name,t.detach().clone())
    for n,t in x.items(): save('input.'+n,t)
    time = env['sinusoidal_embedding_1d'](256,x['timestep']).float()
    save('preparation.sinusoidal', time)
    save('preparation.patch',model.patch_embedding(x['latent']))
    time = model.time_embedding(time); save('preparation.time',time)
    save('preparation.modulation',model.time_projection(time).unflatten(1,(6,5120)))
    padded = torch.nn.functional.pad(x['raw_context'], (0,0,0,509))
    save('preparation.padded_context',padded)
    save('preparation.context',model.text_embedding(padded))
    freqs = torch.cat([env['rope_params'](4,d) for d in [44,42,42]],dim=1).cuda()
    grid = torch.tensor([[2,1,2]],device='cuda'); lens = torch.tensor([4],device='cuda')
    parts = freqs.split([22,21,21],dim=1)
    angles = torch.cat([parts[0][:2].view(2,1,1,22).expand(2,1,2,22),parts[1][:1].view(1,1,1,21).expand(2,1,2,21),parts[2][:2].view(1,1,2,21).expand(2,1,2,21)],dim=-1).reshape(1,4,1,64)
    save('block.cosine',angles.real.repeat_interleave(2,-1).float())
    save('block.sine',angles.imag.repeat_interleave(2,-1).float())
    b = model.blocks[0]; combined = b.modulation + x['modulation_input']
    save('block.combined',combined); save('block.input',x['input'])
    save('block.modulation_input',x['modulation_input']); save('block.context',x['context'])
    for i,n in enumerate(['shift','scale','gate']): save('block.'+n,combined[:,i:i+1])
    # Hooks observe the actual upstream forward; no copied block implementation.
    handles = []
    def hook(module,name,pre=False):
        def cb(m,inputs,*output): save('block.'+name,inputs[0] if pre else output[0])
        handles.append(module.register_forward_pre_hook(cb) if pre else module.register_forward_hook(cb))
    for module,name in [(b.norm1,'pre_attention_norm'),(b.norm3,'pre_cross_attention_norm'),(b.norm2,'pre_ffn_norm'),(b.ffn[0],'ffn_projection_in'),(b.ffn[1],'ffn_activation'),(b.ffn[2],'ffn_projection_out')]: hook(module,name)
    hook(b.norm3,'first_residual',True); hook(b.norm2,'cross_residual',True)
    hook(b.ffn[0],'ffn_input',True); hook(b.self_attn,'self_attention_input',True)
    for attr,label in [('self_attn','self_attention'),('cross_attn','cross_attention')]:
        att = getattr(b,attr)
        for short,long in [('q','q_linear'),('k','k_linear'),('v','v_linear'),('norm_q','q_norm'),('norm_k','k_norm'),('o','output')]: hook(getattr(att,short),label+'.'+long)
        hook(att.o,label+'.attention_merged',True)
    count = 0
    def attention(q,k,v,k_lens=None,window_size=(-1,-1),**kwargs):
        nonlocal count
        assert q.dtype==k.dtype==v.dtype==torch.float32 and window_size==(-1,-1) and not kwargs
        assert k_lens is None or bool(torch.all(k_lens == k.shape[1]))
        label = 'self_attention' if count==0 else 'cross_attention'; count+=1
        save('block.'+label+'.q_positioned',q); save('block.'+label+'.k_positioned',k)
        out = (torch.softmax((q.transpose(1,2) @ k.transpose(1,2).transpose(-1,-2))*(q.shape[-1]**-0.5),dim=-1) @ v.transpose(1,2)).transpose(1,2).contiguous()
        save('block.'+label+'.attention_heads',out)
        return out
    env['flash_attention'] = attention
    result = b(x['input'],x['modulation_input'].unsqueeze(1),lens,grid,freqs,x['context'],None)
    save('block.output',result)
    for label in ['self_attention','cross_attention']:
        for n in ['q','k','v']:
            value=trace['block.'+label+'.'+(n+'_norm' if n!='v' else 'v_linear')]
            save('block.'+label+'.'+n+'_heads',value.reshape(1,-1,40,128))
    save('block.gated',trace['block.self_attention.output']*combined[:,2:3])
    torch.cuda.synchronize(); write_bundle(root/'reference.bundle',trace)
    for h in handles: h.remove()
    meta = dict(source_revision=revision,source_file_sha256=hashlib.sha256((Path(args.official)/'wan/modules/model.py').read_bytes()).hexdigest(),seed=20260917,torch=torch.__version__,device=torch.cuda.get_device_name(),dtype='F32',tf32=False,attention='research-only explicit F32 scaled dot-product; NOT official BF16 FlashAttention qualification',loaded_tensor_sha256=src.hashes,checkpoint_count=len(trace),peak_device_allocated=torch.cuda.max_memory_allocated())
    (root/'reference.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:v for k,v in meta.items() if k!='loaded_tensor_sha256'},indent=2))


@torch.no_grad()
def full_reference(args):
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    assert subprocess.check_output(['git','-C',args.official,'rev-parse','HEAD'],text=True).strip()=='42bf4cfaa384bc21833865abc2f9e6c0e67233dc'
    env=oracle(args.official); src=Source(args.source)
    cfg=json.loads((Path(args.source)/'config.json').read_text())
    with torch.device('meta'):
        model=env['WanModel'](**{k:v for k,v in cfg.items() if not k.startswith('_')})
    for name in ['patch_embedding','time_embedding','time_projection','text_embedding','head']:
        src.load(getattr(model,name),name+'.')
    for i,b in enumerate(model.blocks):
        src.load(b,'blocks.'+str(i)+'.')
        print('loaded block',i,flush=True)
    assert len(src.hashes)==1095
    model.eval(); model.freqs=torch.cat([env['rope_params'](1024,d) for d in [44,42,42]],dim=1).cuda()
    def attention(q,k,v,k_lens=None,window_size=(-1,-1),**kwargs):
        assert q.dtype==k.dtype==v.dtype==torch.float32 and window_size==(-1,-1) and not kwargs
        assert k_lens is None or bool(torch.all(k_lens==k.shape[1]))
        return (torch.softmax((q.transpose(1,2)@k.transpose(1,2).transpose(-1,-2))*(q.shape[-1]**-0.5),dim=-1)@v.transpose(1,2)).transpose(1,2).contiguous()
    env['flash_attention']=attention
    x={n:v.cuda() for n,v in read_bundle(args.inputs).items()}
    x['timestep']=torch.tensor([args.timestep],dtype=torch.int64,device='cuda')
    result={}; branch=0; handles=[]
    def hook(module,name,transform=lambda x:x):
        def cb(m,inputs,output): result['branch.'+str(branch)+'.'+name]=transform(output).detach().clone()
        handles.append(module.register_forward_hook(cb))
    hook(model.patch_embedding,'input_projection')
    hook(model.time_embedding,'conditioning.time',lambda x:x[:,0])
    hook(model.time_projection,'conditioning.modulation',lambda x:x[:,0].reshape(1,6,5120))
    hook(model.text_embedding,'conditioning.context')
    for i,b in enumerate(model.blocks): hook(b,'block.'+str(i)+'.output')
    hook(model.head,'final_projection')
    for branch,context in enumerate([torch.zeros_like(x['raw_context']),x['raw_context']]):
        output=model([x['latent'][0]],x['timestep'],[context[0]],4)
        prefix='branch.'+str(branch)+'.'
        result[prefix+'prediction']=torch.stack(output)
        for alias,i in [('block_0',0),('middle_block',20),('last_block',39)]: result[prefix+alias]=result[prefix+'block.'+str(i)+'.output']
        print('evaluated branch',branch,flush=True)
    torch.cuda.synchronize(); write_bundle(root/'reference-full.bundle',result)
    (root/'reference-full.json').write_text(json.dumps(dict(timestep=args.timestep,source_tensor_hashes=src.hashes,peak_device_allocated=torch.cuda.max_memory_allocated(),attention='explicit F32 reference boundary'),indent=2,sort_keys=True)+'\n')


@torch.no_grad()
def attention_fixture(args):
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=False
    inputs={}; f32={}; f64={}
    for prefix,file in [('reference','reference.bundle'),('native','native.bundle')]:
        if args.synthetic:
            # Model-independent cancellation test with repeated keys/values.
            q=torch.zeros(1,4,2,128)
            k=torch.zeros(1,512,2,128)
            v=torch.full_like(k,6.4);v[:,0]=-6.4*511
        else:
            bundle=read_bundle(Path(args.input_dir)/file)
            q,k,v=[bundle['block.cross_attention.'+n] for n in ['q_positioned','k_positioned','v_heads']]
        for name,t in zip(['q','k','v'],[q,k,v]):inputs[prefix+'.'+name]=t
        def attention(q,k,v):
            return (torch.softmax((q.transpose(1,2)@k.transpose(1,2).transpose(-1,-2))*(q.shape[-1]**-0.5),-1)@v.transpose(1,2)).transpose(1,2).contiguous()
        f32[prefix+'.attention']=attention(q.cuda(),k.cuda(),v.cuda()).cpu()
        # Higher accuracy diagnostic only; all qualification still uses F32 execution.
        f64[prefix+'.attention']=attention(q.cuda().double(),k.cuda().double(),v.cuda().double()).float().cpu()
    write_bundle(root/'attention-inputs.bundle',inputs)
    write_bundle(root/'attention-reference.bundle',f32)
    write_bundle(root/'attention-f64.bundle',f64)


def verify_weights(args):
    evidence=json.loads(Path(args.expected).read_text())
    hashes=evidence.get('source_tensor_hashes',evidence.get('loaded_tensor_sha256'))
    with open(args.vrm,'rb') as f:
        header=f.read(128); assert header[:8]==b'VRHINO\x00\x01'
        mo,ml,to,tl,go,gl,data,total=struct.unpack_from('<8Q',header,48)
        assert Path(args.vrm).stat().st_size==total
        f.seek(to); records={v['name']:v for v in json.loads(f.read(tl))['tensors']}
        checked={}
        for slot,expected in sorted(hashes.items()):
            record=records['parameters.'+str(args.binding)+'.'+slot]
            assert record['dtype']=='f32'
            f.seek(data+record['offset']); left=record['byte_length']; digest=hashlib.sha256()
            while left:
                chunk=f.read(min(left,8*1024*1024)); assert chunk
                digest.update(chunk); left-=len(chunk)
            actual=digest.hexdigest(); assert actual==expected, slot
            checked[slot]=actual
    Path(args.output).write_text(json.dumps(dict(binding=args.binding,count=len(checked),byte_identity=True,sha256=checked),sort_keys=True,indent=2)+'\n')
    print('SOURCE_TO_VRM_EXACT_PARAMETER_BYTES',len(checked))


@torch.no_grad()
def ffn_local(args):
    """Hold operator inputs fixed; diagnostic only, never substitutes production."""
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    src=Source(args.source)
    weights={n:src.tensor('blocks.0.ffn.'+n) for n in ['0.weight','0.bias','2.weight','2.bias']}
    inputs={}; expected={}; accurate={}
    for prefix,path in [('reference',args.reference),('native',args.native)]:
        trace=read_bundle(path)
        for key,checkpoint in [('input','ffn_input'),('preactivation','ffn_projection_in'),('activated','ffn_activation')]:
            inputs[prefix+'.'+key]=trace['block.'+checkpoint]
        for dtype,target in [(torch.float32,expected),(torch.float64,accurate)]:
            x={n:inputs[prefix+'.'+n].cuda().to(dtype) for n in ['input','preactivation','activated']}
            w={n:v.to(dtype) for n,v in weights.items()}
            target[prefix+'.projection_in']=torch.nn.functional.linear(x['input'],w['0.weight'],w['0.bias']).float()
            target[prefix+'.activation']=torch.nn.functional.gelu(x['preactivation'],approximate='tanh').float()
            target[prefix+'.projection_out']=torch.nn.functional.linear(x['activated'],w['2.weight'],w['2.bias']).float()
    write_bundle(root/'inputs.bundle',inputs)
    write_bundle(root/'reference.bundle',expected)
    write_bundle(root/'f64-rounded.bundle',accurate)
    (root/'weights.json').write_text(json.dumps(dict(source_tensor_hashes=src.hashes),indent=2,sort_keys=True)+'\n')


def compare(args):
    expected=read_bundle(args.expected); actual=read_bundle(args.actual); rows=[]
    assert expected.keys()==actual.keys(), (expected.keys()-actual.keys(),actual.keys()-expected.keys())
    for name,e in expected.items():
        a=actual[name]; assert a.dtype==e.dtype and a.shape==e.shape, name
        a=a.double(); e=e.double(); delta=(a-e).abs(); finite=bool(torch.isfinite(a).all() and torch.isfinite(e).all())
        limit=2e-5+2e-5*e.abs()  # Frozen before the first run; never fit to observed results.
        bad=int((delta>limit).sum())
        rows.append(dict(name=name,max_absolute=float(delta.max()),mean_absolute=float(delta.mean()),max_relative=float((delta/e.abs().clamp_min(1e-8)).max()),relative_l2=float(delta.norm()/e.norm().clamp_min(1e-30)),failures=bad,elements=e.numel(),pass_=finite and bad==0))
    Path(args.output).write_text(json.dumps(dict(atol=2e-5,rtol=2e-5,passed=all(r['pass_'] for r in rows),checkpoints=rows),indent=2)+'\n')
    for r in rows: print(r['name'], 'PASS' if r['pass_'] else 'FAIL', 'max_abs=',r['max_absolute'],'mean_abs=',r['mean_absolute'],'bad=',r['failures'])
    return 0 if all(r['pass_'] for r in rows) else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='mode',required=True)
    r=sub.add_parser('reference'); r.add_argument('--official',required=True); r.add_argument('--source',required=True); r.add_argument('--output',required=True); r.add_argument('--replay')
    r=sub.add_parser('full'); r.add_argument('--official',required=True); r.add_argument('--source',required=True); r.add_argument('--inputs',required=True); r.add_argument('--output',required=True); r.add_argument('--timestep',type=int,default=999)
    r=sub.add_parser('attention-fixture'); r.add_argument('--output',required=True); g=r.add_mutually_exclusive_group(required=True); g.add_argument('--input-dir'); g.add_argument('--synthetic',action='store_true')
    r=sub.add_parser('weights'); r.add_argument('--expected',required=True); r.add_argument('--vrm',required=True); r.add_argument('--binding',type=int,required=True); r.add_argument('--output',required=True)
    r=sub.add_parser('ffn-local'); r.add_argument('--source',required=True); r.add_argument('--reference',required=True); r.add_argument('--native',required=True); r.add_argument('--output',required=True)
    r=sub.add_parser('compare'); r.add_argument('--expected',required=True); r.add_argument('--actual',required=True); r.add_argument('--output',required=True)
    args=p.parse_args()
    if args.mode=='reference': reference(args)
    elif args.mode=='full': full_reference(args)
    elif args.mode=='weights': verify_weights(args)
    elif args.mode=='ffn-local': ffn_local(args)
    elif args.mode=='attention-fixture': attention_fixture(args)
    else: raise SystemExit(compare(args))
