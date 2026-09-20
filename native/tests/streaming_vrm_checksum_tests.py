#!/usr/bin/env python3
"""Independent hashlib golden fixtures straddling the native 8 MiB read window."""
import argparse, hashlib, json, struct, subprocess
from pathlib import Path

def fixture(path, hashed_bytes):
    metadata=b'{"architecture":"fixture"}'; graph=b'{"schema_version":1}'
    n=hashed_bytes
    for _ in range(8):
        table=json.dumps({'schema_version':1,'tensors':[{'name':'x','dtype':'u8','shape':[n],
            'offset':0,'byte_length':n,'alignment':64,'layout':'contiguous','quantization':{'type':'none'}}]},separators=(',',':')).encode()
        start=(128+len(metadata)+len(table)+len(graph)+63)//64*64
        updated=hashed_bytes+128-start
        if updated==n:break
        n=updated
    assert n>0
    header=bytearray(128);header[:8]=b'VRHINO\x00\x01'
    struct.pack_into('<HHBBH',header,8,0,1,1,6,0)
    header[16:25]=b'component';header[32:39]=b'fixture'
    cur=128
    for offset,section in zip((48,64,80),(metadata,table,graph)):
        struct.pack_into('<QQ',header,offset,cur,len(section));cur+=len(section)
    struct.pack_into('<QQ',header,96,start,hashed_bytes+128)
    body=metadata+table+graph+b'\x00'*(start-cur)
    h=hashlib.blake2b(digest_size=16)
    with path.open('wb') as f:
        f.write(header);f.write(body);h.update(body)
        chunk=bytes(range(256))*256
        while n:
            data=chunk[:min(n,len(chunk))];f.write(data);h.update(data);n-=len(data)
        f.seek(112);f.write(h.digest())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('inspector');ap.add_argument('output',type=Path);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    for n in [1024,1025,8*1024*1024-1,8*1024*1024,8*1024*1024+1,16*1024*1024,16*1024*1024+129]:
        p=args.output/f'{n}.vrm';fixture(p,n)
        result=subprocess.run([args.inspector,str(p)],capture_output=True,text=True);assert result.returncode==0,result.stderr
        with p.open('r+b') as f:f.seek(-1,2);b=f.read(1);f.seek(-1,2);f.write(bytes([b[0]^1]))
        result=subprocess.run([args.inspector,str(p)],capture_output=True,text=True)
        assert result.returncode!=0 and 'checksum mismatch' in result.stderr,result.stderr
        print('PASS hashlib golden + corruption rejection: hashed bytes',n)
    print('PASS streaming checksum: 7 independent boundary goldens, 7 corruptions')
if __name__=='__main__':main()
