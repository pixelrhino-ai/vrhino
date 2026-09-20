#!/usr/bin/env python3
"""Exercise the real converter's admission without reading real tensor payloads."""
import argparse,json,subprocess,tempfile,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--converter',type=Path,required=True);p.add_argument('--spec',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--semantic-source',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);a=p.parse_args()
a.scratch.mkdir(parents=True,exist_ok=True)
contract=json.loads((a.spec/'source-contract.json').read_text())
with tempfile.TemporaryDirectory(prefix='source-rejections-',dir=a.scratch) as tmp:
    root=Path(tmp);model=root/'source';model.mkdir()
    # Sparse files only satisfy size checks; the README checksum must reject
    # before any placeholder large payload is hashed or converted.
    for row in contract['artifacts']:
        if not row['relative_path'].startswith('model/'):continue
        rel=Path(row['relative_path'][6:]);f=model/rel;f.parent.mkdir(parents=True,exist_ok=True)
        with f.open('wb') as out:out.truncate(int(row['size_bytes']))
    def reject(label,spec,needle):
        output=root/'rejected-output'
        result=subprocess.run([str(a.converter),str(model),str(a.semantic_source),str(spec),str(output)],capture_output=True,text=True)
        assert result.returncode!=0 and needle in result.stderr,(label,result.stderr)
        assert not output.exists() and not Path(str(output)+'.partial').exists()
        print('REJECT',label,'no_output=yes')
    reject('wrong source SHA256',a.spec,'identity mismatch')
    shard=model/'high_noise_model/diffusion_pytorch_model-00001-of-00006.safetensors';shard.unlink()
    reject('missing shard',a.spec,'Missing/nonregular qualified source')
    changed=root/'spec';shutil.copytree(a.spec,changed)
    text=(changed/'source-contract.json').read_text().replace('3f42affa3a1f1c6bd1f14f4cd01cdb90373af3d7','0'*40)
    (changed/'source-contract.json').write_text(text)
    reject('revision provenance drift',changed,'Converter spec drift')
    shutil.copyfile(a.spec/'source-contract.json',changed/'source-contract.json')
    with (changed/'source-manifest.tsv').open('a') as f:f.write('drift\n')
    reject('manifest drift',changed,'Converter spec drift')
print('PASS qualified source converter gate; no real tensor payload reads; no source changes')
