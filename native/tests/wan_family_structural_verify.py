#!/usr/bin/env python3
"""Independent bounded structural verifier. Never imports tensor frameworks."""
import argparse, csv, hashlib, json, math, struct
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('package',type=Path)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--spec',type=Path,required=True)
p.add_argument('--semantic-source',type=Path,required=True)
a=p.parse_args();root=a.package
copy_samples=0

def raw_copy_edges(source,source_offset,destination,destination_offset,length):
    # Byte-copy validation only, not a numerical tensor/reference comparison.
    global copy_samples
    n=min(64,length)
    with source.open('rb') as left,destination.open('rb') as right:
        for relative in sorted({0,length-n}):
            left.seek(source_offset+relative);right.seek(destination_offset+relative)
            assert left.read(n)==right.read(n), ('raw byte copy mismatch',source,source_offset)
            copy_samples+=1

def safe_header(path):
    with path.open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]
        assert 0<n<=64*1024*1024
        h=json.loads(f.read(n))
    return h,8+n

def full_hash(path,vrm=False):
    sha=hashlib.sha256();blake=hashlib.blake2b(digest_size=16);size=0
    with path.open('rb') as f:
        if vrm:
            header=f.read(128);assert len(header)==128;sha.update(header);size=128
        while chunk:=f.read(8*1024*1024):
            sha.update(chunk);size+=len(chunk)
            if vrm:blake.update(chunk)
    if vrm:assert blake.digest()==header[112:128], 'VRM payload checksum'
    return sha.hexdigest(),size

with (root/'model.vrm').open('rb') as f:
    header=f.read(128)
    assert header[:8]==b'VRHINO\x00\x01' and struct.unpack_from('<HH',header,8)==(0,1)
    mo,ml,to,tl,go,gl,do,size=struct.unpack_from('<8Q',header,48)
    assert mo==128 and to==mo+ml and go==to+tl and do==(go+gl+63)//64*64
    assert size==(root/'model.vrm').stat().st_size
    def section(offset,length):
        assert length<8*1024*1024
        f.seek(offset);return f.read(length)
    metadata_bytes=section(mo,ml);table_bytes=section(to,tl);graph_bytes=section(go,gl)
    metadata=json.loads(metadata_bytes);table=json.loads(table_bytes);graph=json.loads(graph_bytes)
    assert section(go+gl,do-go-gl)==bytes(do-go-gl)
    assert metadata_bytes==(root/'metadata.json').read_bytes() and graph_bytes==(root/'graph.json').read_bytes()
    expected=0;names=[];records={}
    for t in table['tensors']:
        aligned=(expected+63)//64*64
        assert t['offset']==aligned and t['alignment']==64 and t['layout']=='contiguous'
        assert t['dtype']=='f32' and t['quantization']['type']=='none'
        assert t['byte_length']==math.prod(t['shape'])*4
        assert section(do+expected,aligned-expected)==bytes(aligned-expected)
        expected=aligned+t['byte_length'];names.append(t['name']);records[t['name']]=t
    assert names==sorted(set(names)) and do+expected==size and len(names)==2384
assert metadata['architecture']=='wan'
assert graph['schema_version']==2 and graph['required_capabilities']==['binding_catalog.v1']
assert len(graph['graphs'])==1 and len(graph['bindings'])==len(graph['instances'])==2
canonical=graph['graphs'][0]['declaration'];d=canonical['dimensions']
assert d==json.loads((a.spec/'conversion.json').read_text())['expected_dimensions']
assert d['hidden_size']//d['head_count']==128 and canonical['patch']['size']==[1,2,2]
assert canonical['position']=={'encoding':'rope_3d_adjacent','axis_partition':'temporal_remainder_spatial_sixths','theta':10000}
assert canonical['conditioning']['text_feature_size']==4096 and canonical['conditioning']['text_token_limit']==512
assert canonical['normalization']['query_key']=='rms_norm' and canonical['normalization']['epsilon']==1e-6
assert canonical['attention']['head_layout']=='BSHD' and canonical['parameters']['storage_dtype']=='float32'
all_source_headers=[]
for binding_id,directory in enumerate(['high_noise_model','low_noise_model']):
    index=json.loads((a.source/directory/'diffusion_pytorch_model.safetensors.index.json').read_text())
    tensors={}; source_offsets={}
    for shard in sorted(set(index['weight_map'].values())):
        h,source_start=safe_header(a.source/directory/shard)
        for name,t in h.items():
            if name=='__metadata__':continue
            assert name not in tensors and index['weight_map'][name]==shard
            tensors[name]=t
            source_offsets[name]=source_start+t['data_offsets'][0]
    assert len(tensors)==1095
    binding=graph['bindings'][binding_id]
    assert binding['id']==binding_id and binding['graph']==0 and set(binding['parameters'])==set(tensors)
    for slot,t in tensors.items():
        name=binding['parameters'][slot]
        assert name==f'parameters.{binding_id}.{slot}'
        assert records[name]['shape']==t['shape'] and t['dtype']=='F32'
        assert records[name]['byte_length']==t['data_offsets'][1]-t['data_offsets'][0]
        raw_copy_edges(a.source/directory/index['weight_map'][slot],source_offsets[slot],
                       root/'model.vrm',do+records[name]['offset'],records[name]['byte_length'])
    assert graph['instances'][binding_id]=={'id':binding_id,'binding':binding_id,'graph':0}
    all_source_headers.append(tensors)
assert all_source_headers[0]==all_source_headers[1]
# Shared byte extraction rules are independently checked against their canonical tables.
vae_rows=list(csv.DictReader((a.spec/'vae-mapping.tsv').open(),delimiter='\t'))
assert len(vae_rows)==194
for row in vae_rows:
    assert records[row['destination_name']]['shape']==[int(x) for x in row['destination_shape'].split(',')]
    assert row['source_dtype']==row['destination_dtype']=='F32'
    record=records[row['destination_name']]
    raw_copy_edges(a.source/'Wan2.1_VAE.pth',int(row['source_offset']),root/'model.vrm',
                   do+record['offset'],record['byte_length'])
decoder=metadata['shared_components']['decoder']
assert decoder==json.loads((a.spec/'decoder.json').read_text())
assert set(decoder['runtime_tensor_bindings'].values())=={r['destination_name'] for r in vae_rows}
umt5,start=safe_header(root/'conditioning.safetensors');umt5={k:v for k,v in umt5.items() if k!='__metadata__'}
rows=list(csv.DictReader((a.spec/'umt5-mapping.tsv').open(),delimiter='\t'))
assert len(umt5)==len(rows)==242
for row in rows:
    t=umt5[row['destination_name']]
    assert t['dtype']==row['destination_dtype']=='BF16'
    assert t['shape']==[int(x) for x in row['destination_shape'].split(',')]
    assert t['data_offsets'][1]-t['data_offsets'][0]==math.prod(t['shape'])*2
    raw_copy_edges(a.source/'models_t5_umt5-xxl-enc-bf16.pth',int(row['source_offset']),
                   root/'conditioning.safetensors',start+t['data_offsets'][0],math.prod(t['shape'])*2)
assert start+max(t['data_offsets'][1] for t in umt5.values())==(root/'conditioning.safetensors').stat().st_size
index=json.loads((root/'conditioning-index.json').read_text())
assert set(index['weight_map'])==set(umt5) and set(index['weight_map'].values())=={'conditioning.safetensors'}
conditioning=json.loads((root/'conditioning.json').read_text())
assert len(conditioning['blocks'])==24 and conditioning['blocks'][0]['attention']['query_heads']==64
assert umt5['token_embedding.weight']['shape'][1]==4096
# Only parse literal assignment text; never execute the qualified Python modules.
import re,ast
cfg=(a.semantic_source/'wan/configs/wan_t2v_A14B.py').read_text()
shared=(a.semantic_source/'wan/configs/shared_config.py').read_text()
def literal(text,object_name,name):
    values=re.findall(r'^'+re.escape(object_name+'.'+name)+r'\s*=\s*([^#\n]+)',text,re.M)
    assert len(values)==1;return ast.literal_eval(values[0].strip())
steps=literal(cfg,'t2v_A14B','sample_steps');shift=literal(cfg,'t2v_A14B','sample_shift')
scales=literal(cfg,'t2v_A14B','sample_guide_scale');train=literal(shared,'wan_shared_cfg','num_train_timesteps')
boundary=literal(cfg,'t2v_A14B','boundary')*train
program=metadata['programs'];assert program==json.loads((root/'programs.json').read_text())
ids=program['execution']['instance_ids'];transitions=program['sampling']['transitions']
assert len(ids)==len(transitions)==steps and set(ids)=={0,1}
f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
initial=f32(1-1/train);sigmas=[]
for i in range(steps):
    s=initial*(1-i/steps);sigmas.append(shift*s/(1+(shift-1)*s))
sigmas.append(0)
for i,t in enumerate(transitions):
    model_time=int(sigmas[i]*train);selected=0 if model_time>=boundary else 1
    assert ids[i]==selected and t['model_timestep']==model_time
    assert t['guidance_scale']==scales[1 if selected==0 else 0]
    assert t['sigma']==f32(sigmas[i]) and t['next_sigma']==f32(sigmas[i+1])
assert program['sampling']['branch_order']=='unconditional_then_conditional'
assert set(program['required_capabilities'])=={'execution.per_step.v1','sampling.flow_sigma_cfg.v1'}
# Canonical execution-bearing sections have no upstream identity or path semantics.
for blob in [graph_bytes,table_bytes,(root/'programs.json').read_bytes()]:
    for forbidden in [b'high_noise',b'low_noise',b'ModelScope',b'HuggingFace',b'_diffusers',b'WanModel']:
        assert forbidden not in blob
provenance=metadata['conversion_provenance']
assert provenance['actual_repository']=='https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B'
assert provenance['actual_revision']=='3f42affa3a1f1c6bd1f14f4cd01cdb90373af3d7'
assert provenance['core_reference_revision']=='c8c270b13ee05bfa474194ac9fb07a5868a97cea'
assert provenance['whole_directory_hf_identity'] is False
assert hashlib.sha256((root/'source-manifest.tsv').read_bytes()).hexdigest()==provenance['manifest_sha256']=='4ea6995c08502841565b9dd2e612af0dd97ddfb49ab17721339fc5c077338911'
package=json.loads((root/'package.json').read_text());resource_paths=[x['path'] for x in package['resources']]
assert resource_paths==sorted(set(resource_paths))
assert set(package['required_capabilities'])=={'binding_catalog.v1','execution.per_step.v1','sampling.flow_sigma_cfg.v1'}
for resource in package['resources']:
    path=Path(resource['path']);assert not path.is_absolute() and '..' not in path.parts
    print('VERIFY_RESOURCE',str(path),flush=True)
    sha,n=full_hash(root/path,path.name=='model.vrm')
    assert sha==resource['sha256'] and n==resource['size']
    if str(path)=='model.vrm':vrm_sha=sha
    if str(path)=='conditioning.safetensors':assert sha=='1c137395973a8a717bd5d467fec816e77af973474c6d4afd2cb83c78cc489bd6'
manifest=list(csv.DictReader((root/'source-manifest.tsv').open(),delimiter='\t'))
for row in manifest:
    if row['relative_path'].startswith('model/google/umt5-xxl/'):
        sha,n=full_hash(root/'tokenizer'/Path(row['relative_path']).name)
        assert sha==row['sha256'] and n==int(row['size_bytes'])
result={'status':'PASS','vrm_size':size,'vrm_sha256':vrm_sha,'transformer_slots':[1095,1095],'tensor_count':2384,'vae_slots':194,'umt5_slots':242,'binding_step_counts':[ids.count(0),ids.count(1)],'steps':steps,'payload_checksum':'verified_streaming_blake2b128','resource_hashes_verified':len(resource_paths),'raw_copy_boundary_samples':copy_samples,'model_payload_loaded':False,'numerical_execution':False,'structural_sections_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in [('metadata',metadata_bytes),('tensor_table',table_bytes),('graph',graph_bytes)]}}
(root.parent/'independent-structural-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
