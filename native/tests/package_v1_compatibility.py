#!/usr/bin/env python3
"""Build the actual frozen v1 reader/writer; only synthetic fixture bytes are used."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import tempfile

p = argparse.ArgumentParser()
p.add_argument('--build', required=True, type=Path)
p.add_argument('--fixtures', required=True, type=Path)
p.add_argument('--baseline', default='b1d5ac2941353e6dbc82f8224ca41fb1da3c98f4')
a = p.parse_args()
repo = Path(__file__).resolve().parents[2]
def run(cmd):
    subprocess.run([str(x) for x in cmd], check=True, cwd=repo)
with tempfile.TemporaryDirectory(prefix='vrhino-v1-compat-') as temporary:
    t = Path(temporary)
    for source in ('loader.cpp', 'product/converter.cpp'):
        data = subprocess.check_output(['git', 'show', f'{a.baseline}:native/src/{source}'], cwd=repo)
        (t / Path(source).name).write_bytes(data)
    (t / 'main.cpp').write_text(r'''
#include "vrhino/loader.h"
#include "vrhino/error.h"
#include "vrhino/product/converter.h"
#include <iostream>
using namespace vrhino;
namespace p=vrhino::product;
int main(int argc,char** argv) {
    try {
        require(argc==5,"compatibility paths");
        VrmModel legacy(argv[1]);
        bool refused=false;
        try { VrmModel unsupported(argv[2]); }
        catch(const Error& e) { refused=std::string(e.what()).find("Unsupported graph schema")!=std::string::npos; }
        require(refused,"Old reader did not reject v2 at schema admission");
        p::SafeTensorReader source(argv[3]);std::vector<p::TensorMapping> mappings;
        for(const auto& [name,dest]:std::vector<std::pair<std::string,std::string>>{
            {"param.0","canonical.b"},{"param.2","canonical.c"},{"param.1","canonical.a"},{"param.3","canonical.d"}}) {
            const auto& x=source.tensors().at(name);
            mappings.emplace_back(name,x.dtype,x.shape,"identity_bytes",dest,x.dtype,x.shape,"parameter","weight");
        }
        p::write_vrm_streaming(argv[4],"dit-flow","synthetic",Json::parse(R"({"architecture":"synthetic"})"),
            Json::parse(R"({"schema_version":1})"),source,mappings);
        std::cout<<"PASS actual baseline reader: v1 accepted / v2 rejected; baseline writer emitted\n";
    }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
''')
    run(['c++', '-std=c++20', '-O1', '-I', repo / 'native/include',
         t / 'main.cpp', t / 'loader.cpp', t / 'converter.cpp',
         a.build / 'libvrhino_model_package.a', a.build / 'libvrhino_native_common.a',
         '-pthread', '-o', t / 'compat'])
    run([t / 'compat', a.fixtures / 'legacy.vrm', a.fixtures / 'multi.vrm',
         a.fixtures / 'single.safetensors', t / 'baseline.vrm'])
    current = (a.fixtures / 'legacy-single.vrm').read_bytes()
    assert current == (t / 'baseline.vrm').read_bytes(), 'Frozen writer byte drift'
    print('PASS baseline/current VRM byte equality sha256=' + hashlib.sha256(current).hexdigest())
