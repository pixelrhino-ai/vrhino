#include "vrhino/product/converter.h"
#include "vrhino/loader.h"
#include "vrhino/error.h"
#include <fstream>
#include <algorithm>
#include <iostream>
using namespace vrhino;
namespace p=vrhino::product;
namespace fs=std::filesystem;
namespace {
uint64_t u64(const std::string& b,size_t offset){uint64_t n=0;for(int i=7;i>=0;--i)n=(n<<8)|static_cast<unsigned char>(b.at(offset+i));return n;}
std::string region(const fs::path& p,uint64_t offset,size_t n) {
    std::ifstream f(p,std::ios::binary);require(f.good(),"Missing structure file");f.seekg(offset);
    std::string s(n,'\0');f.read(s.data(),n);require(f.good(),"Structure read failed");return s;
}
struct EndOfStructure:std::exception {};
class StructureSource final:public p::TensorSource {
public:
    std::map<std::string,p::SourceTensorDescriptor> catalog;
    fs::path partial;std::string expected;mutable bool checked=false;
    const std::map<std::string,p::SourceTensorDescriptor>& tensors()const noexcept override{return catalog;}
    void read_tensor(const p::SourceTensorDescriptor&,uint64_t,void*,size_t)const override {
        require(region(partial,128,expected.size())==expected,"Re-emitted metadata/table/graph bytes differ");
        checked=true;throw EndOfStructure{};
    }
};
}
int main(int argc,char** argv) {
    try {
        require(argc==3,"usage: structure-reproduction VRM EMPTY_PARTIAL_PATH");
        VrmModel model(argv[1],false);auto header=region(argv[1],0,128);auto offset=u64(header,96);
        require(offset<8*1024*1024,"Unbounded structural prefix");
        auto table=Json::parse(region(argv[1],u64(header,64),u64(header,72)));
        StructureSource source;source.partial=argv[2];source.expected=region(argv[1],128,offset-128);
        std::vector<p::TensorMapping> mapping;
        for(const auto& t:table.at("tensors").array()) {
            auto name=t.at("name").string();std::vector<int64_t> shape;
            for(auto& n:t.at("shape").array())shape.push_back(n.integer());auto dtype=dtype_from_vrm(t.at("dtype").string());
            source.catalog.emplace(name,p::SourceTensorDescriptor{name,dtype,"",shape,0,0,uint64_t(t.at("byte_length").integer())});
            mapping.emplace_back(name,dtype,shape,"identity_bytes",name,dtype,shape,t.at("component").string(),t.at("role").string());
        }
        // A second native writer run, reversed input mapping order, no payload
        // reads/writes or second large allocation. Compare every structural byte.
        std::reverse(mapping.begin(),mapping.end());
        try {p::write_vrm_streaming(source.partial,model.profile_id(),model.architecture_id(),model.metadata(),model.graph(),source,mapping);}
        catch(const EndOfStructure&) {}
        require(source.checked && !fs::exists(source.partial),"Structural stop did not clean partial output");
        std::cout<<"PASS deterministic_structure_bytes="<<source.expected.size()<<" reversed_mapping_order=yes payload_bytes_read=0 failure_cleanup=yes\n";return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
