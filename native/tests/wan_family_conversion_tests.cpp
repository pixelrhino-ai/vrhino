#include "vrhino/product/wan_family_conversion.h"
#include "vrhino/error.h"
#include "../src/architectures/wan_family.h"
#include <iostream>
#include <fstream>
#include <chrono>
using namespace vrhino;
namespace p=vrhino::product;
namespace fs=std::filesystem;
namespace {
int failures=0;
template<class F> void reject(const char* name,F operation) {
    bool rejected=false;try{operation();}catch(const std::exception& e){rejected=true;std::cout<<"REJECT "<<name<<": "<<e.what()<<'\n';}
    require(rejected,std::string("Expected rejection: ")+name);++failures;
}
std::string change(std::string s,const std::string& from,const std::string& to) {
    auto at=s.find(from);require(at!=std::string::npos,"Missing mutation target");s.replace(at,from.size(),to);return s;
}
class Catalog:public p::TensorSource {
public:
    std::map<std::string,p::SourceTensorDescriptor> entries;
    const std::map<std::string,p::SourceTensorDescriptor>& tensors()const noexcept override{return entries;}
    void read_tensor(const p::SourceTensorDescriptor&,uint64_t,void*,size_t)const override {throw Error("Mapping validation must not read payload");}
};
}
int main() {
    try {
        const auto config=Json::parse(R"({"model_type":"t2v","qk_norm":true,"cross_attn_norm":true,"eps":0.000001,"dim":12,"num_heads":2,"num_layers":1,"ffn_dim":16,"freq_dim":4,"in_dim":2,"out_dim":2})");
        const auto canonical=p::lower_wan_source_config(config,Json::parse("[1,2,2]"),4);
        const auto family=wan_family::lower(CanonicalArchitectureDeclaration::parse(canonical));Catalog source;
        for(const auto& s:family->parameters->slots())source.entries.emplace(s.role,p::SourceTensorDescriptor{s.role,s.dtype,"F32",s.shape,0,0,uint64_t(shape_numel(s.shape))*4});
        auto a=p::map_wan_parameters(source,canonical,0),b=p::map_wan_parameters(source,canonical,1);
        require(a.size()==42 && b.size()==42,"One-block family slot count");
        for(size_t i=0;i<a.size();++i)require(a[i].source_name==b[i].source_name && a[i].destination_name!=b[i].destination_name,"Graph/binding separation");
        auto bad=source;bad.entries.erase(bad.entries.begin());reject("missing tensor",[&]{p::map_wan_parameters(bad,canonical,0);});
        bad=source;bad.entries.emplace("illegal",bad.entries.begin()->second);reject("extra illegal mapping",[&]{p::map_wan_parameters(bad,canonical,0);});
        bad=source;bad.entries.begin()->second.shape={1};reject("shape mismatch",[&]{p::map_wan_parameters(bad,canonical,0);});
        bad=source;bad.entries.begin()->second.dtype=DType::BF16;reject("dtype mismatch",[&]{p::map_wan_parameters(bad,canonical,0);});
        bad=source;bad.entries.begin()->second.byte_length++;reject("element byte mismatch",[&]{p::map_wan_parameters(bad,canonical,0);});
        reject("invalid architecture config",[&]{p::lower_wan_source_config(Json::parse(change(p::canonical_json(config),"\"dim\":12","\"dim\":13")),Json::parse("[1,2,2]"),4);});
        const std::string graph=R"({"schema_version":2,"required_capabilities":["binding_catalog.v1"],"graphs":[{"id":0,"declaration":{"interface":"synthetic"}}],"bindings":[{"id":0,"graph":0,"parameters":{"p":"parameters.0.p"}},{"id":1,"graph":0,"parameters":{"p":"parameters.1.p"}}],"instances":[{"id":0,"graph":0,"binding":0},{"id":1,"graph":0,"binding":1}]})";
        auto catalog=PackageDeclaration::parse(Json::parse(graph));
        const std::string programs=R"({"schema":"vrhino.programs.v1","required_capabilities":["execution.per_step.v1","sampling.flow_sigma_cfg.v1"],"execution":{"kind":"per_step","instance_ids":[0,1,0]},"sampling":{"prediction":"flow","solver":"multistep_predictor_corrector","maximum_order":2,"schedule":"flow_sigma","branch_order":"unconditional_then_conditional","transitions":[{"model_timestep":999,"sigma":0.999,"next_sigma":0.6,"guidance_scale":4},{"model_timestep":600,"sigma":0.6,"next_sigma":0.2,"guidance_scale":3},{"model_timestep":200,"sigma":0.2,"next_sigma":0,"guidance_scale":4}]}})";
        auto decoded=p::admit_program_declaration(Json::parse(programs),catalog);
        require(decoded.sampling.steps==3 && decoded.sampling.guidance_schedule->at(1).scale==3,"Program wire lowering");
        reject("invalid execution reference",[&]{p::admit_program_declaration(Json::parse(change(programs,"[0,1,0]","[0,9,0]")),catalog);});
        reject("invalid guidance declaration",[&]{p::admit_program_declaration(Json::parse(change(programs,"\"guidance_scale\":3","\"guidance_scale\":-1")),catalog);});
        reject("step length",[&]{p::admit_program_declaration(Json::parse(change(programs,"[0,1,0]","[0,1]")),catalog);});
        reject("unsupported program capability",[&]{p::admit_program_declaration(Json::parse(programs),catalog,{});});
        reject("missing required program capability",[&]{p::admit_program_declaration(Json::parse(change(programs,"\"execution.per_step.v1\",","")),catalog);});
        reject("unknown sampling fields",[&]{p::admit_program_declaration(Json::parse(change(programs,"\"maximum_order\":2","\"maximum_order\":2,\"override\":true")),catalog);});
        reject("broken schedule chain",[&]{p::admit_program_declaration(Json::parse(change(programs,"\"sigma\":0.6","\"sigma\":0.5")),catalog);});
        reject("duplicate binding",[&]{PackageDeclaration::parse(Json::parse(change(graph,"\"id\":1,\"graph\":0,\"parameters\"","\"id\":0,\"graph\":0,\"parameters\"")));});
        reject("unsupported package capability",[&]{PackageDeclaration::parse(Json::parse(change(graph,"binding_catalog.v1","unknown.v1")));});
        auto root=fs::temp_directory_path()/("family-source-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));fs::create_directory(root);
        {std::ofstream out(root/"data");out<<"abcd";}
        {
            p::FrozenTensorSource lease({root/"data"},{{"p",{"p",DType::U8,"U8",{4},0,0,4}}});
            lease.verify_identity(0,4,p::sha256_file(root/"data"));
            reject("wrong source SHA256",[&]{lease.verify_identity(0,4,std::string(64,'0'));});
            reject("wrong source size",[&]{lease.verify_identity(0,5,p::sha256_file(root/"data"));});
            fs::rename(root/"data",root/"retired");{std::ofstream out(root/"data");out<<"xxxx";}
            char bytes[4];lease.read_tensor(lease.tensors().at("p"),0,bytes,4);require(std::string(bytes,4)=="abcd","Source lease pathname replacement");
        }
        fs::remove_all(root);
        std::cout<<"PASS family mapping, programs, qualified file lease; rejections="<<failures<<"; numerical_execution=none\n";return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
