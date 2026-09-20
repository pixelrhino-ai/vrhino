"""Bounded malformed containers with real metadata, never a model payload copy."""
import argparse,json,struct
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('vrm',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
with a.vrm.open('rb') as f:
 header=bytearray(f.read(128));mo,ml,to,tl,go,gl,do,size=struct.unpack_from('<8Q',header,48)
 assert ml+tl+gl<8*1024*1024
 f.seek(mo);metadata=f.read(ml);f.seek(go);graph=f.read(gl)
for name in ['out-of-range','overlap']:
 record={'name':'fixture.0','dtype':'f32','shape':[1024] if name=='out-of-range' else [1],'offset':0,'byte_length':4096 if name=='out-of-range' else 4,'alignment':64,'layout':'contiguous','component':'parameter','role':'weight','quantization':{'type':'none'}}
 records=[record]
 if name=='overlap':records.append(dict(record,name='fixture.1',offset=0))
 table=json.dumps({'schema_version':1,'tensors':records},sort_keys=True,separators=(',',':')).encode()
 m=128;t=m+len(metadata);g=t+len(table);d=(g+len(graph)+63)//64*64
 h=bytearray(header);struct.pack_into('<8Q',h,48,m,len(metadata),t,len(table),g,len(graph),d,d+4);h[112:128]=bytes(16)
 (a.output/(name+'.vrm')).write_bytes(h+metadata+table+graph+bytes(d-g-len(graph))+bytes(4))
print('PASS small corrupt containers generated; payload bytes=8; original VRM unchanged')
