#include "vrhino/product/converter.h"
#include "vrhino/package_declaration.h"
#include "vrhino/loader.h"
#include "step_execution_test_support.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <chrono>

using namespace vrhino;
namespace p=vrhino::product;
namespace fs=std::filesystem;
namespace st=vrhino::step_test;
namespace {
int rejections=0;
size_t descriptors() {
#ifdef __linux__
    return std::distance(fs::directory_iterator("/proc/self/fd"),fs::directory_iterator{});
#else
    return 0;
#endif
}
template<class F> void reject(const std::string& label,F f) {
    const auto before=descriptors(); bool failed=false;
    std::string reason;
    try { f(); } catch(const std::exception& e) { failed=true; reason=e.what(); }
    require(failed,"Expected rejection: "+label);
    require(descriptors()==before,"File descriptor leak: "+label);
    ++rejections; std::cout<<"REJECT "<<label<<": "<<reason<<'\n';
}
std::string replace(std::string s,const std::string& a,const std::string& b) {
    auto n=s.find(a); require(n!=std::string::npos,"Fixture replacement missing"); s.replace(n,a.size(),b); return s;
}
void safe(const fs::path& file,std::string header,std::vector<float> payload) {
    while(header.size()%8)header+=' ';
    std::ofstream out(file,std::ios::binary|std::ios::trunc);
    uint64_t n=header.size(); for(int i=0;i<8;++i)out.put(static_cast<char>(n>>(8*i)));
    out<<header;out.write(reinterpret_cast<const char*>(payload.data()),payload.size()*sizeof(float));
    require(out.good(),"Fixture write failed");
}
const std::string h1=R"({"param.2":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}})";
const std::string h2=R"({"param.0":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]},"param.3":{"dtype":"I32","shape":[1,2],"data_offsets":[8,16]}})";
const std::string h3=R"({"param.1":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]}})";
const std::string index_text=R"({"metadata":{"total_size":32},"weight_map":{"param.0":"shard-2.safetensors","param.1":"shard-3.safetensors","param.2":"shard-1.safetensors","param.3":"shard-2.safetensors"}})";
using Artifacts=std::map<std::string,p::IndexedSourceArtifact>;
Artifacts artifacts(const fs::path& root) {
    Artifacts a;
    for(int i=1;i<=3;++i) { auto name="shard-"+std::to_string(i)+".safetensors";auto file=root/name;
        a.emplace(name,p::IndexedSourceArtifact{file,fs::file_size(file),p::sha256_file(file)}); }
    return a;
}
void fixtures(const fs::path& root) {
    fs::create_directories(root);
    std::ofstream(root/"index.json")<<index_text;
    safe(root/"shard-1.safetensors",h1,{7,8});
    safe(root/"shard-2.safetensors",h2,{1,2,9,10});
    safe(root/"shard-3.safetensors",h3,{3,4});
}
std::vector<p::TensorMapping> mappings(const p::TensorSource& s) {
    std::vector<p::TensorMapping> result;
    for(const auto& [name,dest]:std::vector<std::pair<std::string,std::string>>{
            {"param.0","canonical.b"},{"param.2","canonical.c"},
            {"param.1","canonical.a"},{"param.3","canonical.d"}}) {
        const auto& t=s.tensors().at(name);
        result.emplace_back(name,t.dtype,t.shape,"identity_bytes",dest,t.dtype,t.shape,"parameter","weight");
    }
    return result;
}
const std::string catalog=R"({"schema_version":2,"required_capabilities":["binding_catalog.v1"],
"graphs":[{"id":0,"declaration":{"topology":"test.affine.v1"}}],
"bindings":[{"id":10,"graph":0,"parameters":{"param.0":"canonical.b"}},{"id":20,"graph":0,"parameters":{"param.0":"canonical.a"}}],
"instances":[{"id":0,"graph":0,"binding":10},{"id":1,"graph":0,"binding":20}]})";
p::VrmWriteResult emit(const fs::path& file,const p::TensorSource& source,const std::string& graph) {
    fs::remove(file); // Replace only this test-owned generated output.
    return p::write_vrm_streaming(file,"dit-flow","synthetic",Json::parse(R"({"architecture":"synthetic"})"),
        Json::parse(graph),source,mappings(source));
}
void indexed(const fs::path& root) {
    fixtures(root);auto a=artifacts(root);const auto baseline=descriptors();
    {
        p::IndexedTensorSource source(index_text,a);
        require(descriptors()==baseline+3,"Source does not own exactly three descriptors");
        float v[2]; source.read_tensor(source.tensors().at("param.1"),0,v,8);
        require(v[0]==3 && v[1]==4,"Cross-shard logical lookup");
        source.read_tensor(source.tensors().at("param.0"),4,v,4);require(v[0]==2,"Relative slice lookup");
        auto forged=source.tensors().at("param.0");forged.source_index=0;
        reject("forged source identity",[&]{source.read_tensor(forged,0,v,8);});
        reject("tensor read range",[&]{source.read_tensor(source.tensors().at("param.0"),7,v,4);});
        const auto out=emit(root/"multi.vrm",source,catalog);
        require(out.tensor_count==4 && out.largest_buffer_bytes<=8*1024*1024,"Unbounded streaming buffer");
        auto reverse=mappings(source);std::reverse(reverse.begin(),reverse.end());
        p::write_vrm_streaming(root/"multi-reordered.vrm","dit-flow","synthetic",
            Json::parse(R"({"architecture":"synthetic"})"),Json::parse(catalog),source,reverse);
        require(p::sha256_file(root/"multi.vrm")==p::sha256_file(root/"multi-reordered.vrm"),"Non deterministic streaming");
        VrmModel model((root/"multi.vrm").string());
        require(model.tensor("canonical.a").data_as<float>()[0]==3 &&
                model.tensor("canonical.b").data_as<float>()[1]==2 &&
                model.tensor("canonical.c").data_as<float>()[1]==8,"Incorrect streamed bytes");
        emit(root/"legacy.vrm",source,R"({"schema_version":1})");
        VrmModel legacy((root/"legacy.vrm").string());
        st::exact(legacy.tensor("canonical.a"),model.tensor("canonical.a"));
        // One-shard conversion remains an independent legacy reader path.
        safe(root/"single.safetensors",R"({"param.0":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]},"param.1":{"dtype":"F32","shape":[1,2],"data_offsets":[8,16]},"param.2":{"dtype":"F32","shape":[2],"data_offsets":[16,24]},"param.3":{"dtype":"I32","shape":[1,2],"data_offsets":[24,32]}})",{1,2,3,4,7,8,9,10});
        p::SafeTensorReader single(root/"single.safetensors");
        emit(root/"legacy-single.vrm",single,R"({"schema_version":1})");
        require(p::sha256_file(root/"legacy.vrm")==p::sha256_file(root/"legacy-single.vrm"),"Legacy writer byte drift");
        int polls=0;
        reject("streaming cancellation",[&]{p::write_vrm_streaming(root/"cancel.vrm","dit-flow","synthetic",
            Json::parse(R"({"architecture":"synthetic"})"),Json::parse(catalog),source,mappings(source),[&]{return ++polls==3;});});
        require(!fs::exists(root/"cancel.vrm"),"Cancelled partial output survived");
        // Replacement of a pathname cannot redirect an admitted open source.
        fs::rename(root/"shard-3.safetensors",root/"retired.safetensors");
        safe(root/"shard-3.safetensors",h3,{99,99});
        source.read_tensor(source.tensors().at("param.1"),0,v,8);
        require(v[0]==3 && v[1]==4,"Path replacement corrupted admitted source");
    }
    require(descriptors()==baseline,"Source teardown descriptor leak");fixtures(root);a=artifacts(root);
    for(const auto& [label,text]:std::vector<std::pair<std::string,std::string>>{
        {"malformed index","{"}, {"duplicate index key",replace(index_text,"\"param.0\":", "\"param.0\":\"shard-2.safetensors\",\"param.0\":")},
        {"missing shard",replace(index_text,"shard-3.safetensors","absent.safetensors")},
        {"missing tensor",replace(index_text,"param.0","param.99")},
        {"wrong shard",replace(index_text,"\"param.0\":\"shard-2.safetensors\"","\"param.0\":\"shard-1.safetensors\"")},
        {"unindexed tensor",replace(index_text,"\"param.3\":\"shard-2.safetensors\"", "\"param.9\":\"shard-2.safetensors\"")},
        {"total size mismatch",replace(index_text,"32","33")},
        {"path traversal",replace(index_text,"shard-3.safetensors","../shard-3.safetensors")},
        {"non-string shard",replace(index_text,"\"shard-3.safetensors\"","3")},
        {"invalid metadata",replace(index_text,"{\"total_size\":32}","[]")}})
        reject(label,[&]{p::IndexedTensorSource invalid(text,a);});
    for(const auto& [label,header]:std::vector<std::pair<std::string,std::string>>{
        {"malformed header","{"}, {"duplicate header key",replace(h3,"\"param.1\":","\"param.1\":{},\"param.1\":")},
        {"duplicate tensor in shards",replace(h3,"param.1","param.0")},
        {"negative offset",replace(h3,"[0,8]","[-1,7]")},
        {"offset outside file",replace(h3,"[0,8]","[8,16]")},
        {"shape byte mismatch",replace(h3,"[1,2]","[1,3]")},
        {"negative shape",replace(h3,"[1,2]","[-1,2]")},
        {"unsupported dtype",replace(h3,"F32","BAD")}}) {
        safe(root/"shard-3.safetensors",header,{3,4});auto invalid=artifacts(root);
        reject(label,[&]{p::IndexedTensorSource source(index_text,invalid);});
    }
    std::string deep(64,'[');deep+='0';deep+=std::string(64,']');
    safe(root/"shard-3.safetensors",deep,{3,4});a=artifacts(root);
    reject("bounded header nesting",[&]{p::IndexedTensorSource invalid(index_text,a);});
    fixtures(root);
    fs::resize_file(root/"shard-3.safetensors",7);a=artifacts(root);
    reject("truncated header prefix",[&]{p::IndexedTensorSource invalid(index_text,a);});
    fixtures(root);
    safe(root/"shard-2.safetensors",replace(h2,"[8,16]","[4,12]"),{1,2,9,10});a=artifacts(root);
    reject("overlap ranges",[&]{p::IndexedTensorSource invalid(index_text,a);});
    fixtures(root);a=artifacts(root);a.at("shard-3.safetensors").sha256=std::string(64,'0');
    reject("checksum failure after partial acquisition",[&]{p::IndexedTensorSource invalid(index_text,a);});
    a=artifacts(root);a.at("shard-2.safetensors").size++;
    reject("file size identity",[&]{p::IndexedTensorSource invalid(index_text,a);});
    a=artifacts(root);a.emplace("unused",a.begin()->second);
    reject("extra shard",[&]{p::IndexedTensorSource invalid(index_text,a);});
    a=artifacts(root);int checks=0;
    reject("verification cancellation after partial acquisition",[&] {
        p::IndexedTensorSource cancelled(index_text,a,[&](uint64_t) {
            if(++checks==3) throw p::ModelPackageError(p::ModelPackageErrorCode::Cancelled,"synthetic cancellation");
        });
    });
    require(checks==3,"Verification cancellation did not reach third shard");
    std::cout<<"PASS indexed source, streaming, immutable open identity, legacy bytes, cleanup\n";
}
PackageGraphLowerer lowerer(st::Backend& backend,int& factories) {
    return [&](const Json& j) {
        require(j.object().size()==1 && j.at("topology").string()=="test.affine.v1","Unknown architecture semantics");
        ComponentInterface i; i.latent={{1,2},DType::F32};i.timestep={{},DType::I64};
        i.predictions={i.latent,i.latent}; i.guidance_mode=GuidanceMode::CFG;
        auto slots=std::make_shared<const ArchitectureBindingDeclaration>(
            std::vector<ArchitectureParameterSlot>{{"param.0",{1,2}}});
        auto graph=std::make_shared<const ComponentGraphDefinition>(i,
            std::vector<ExecutionTensorContract>{{{1,2},DType::F32}},
            [&](const std::vector<Tensor>& p){++factories;return std::make_unique<st::LegacyDenoiser>(backend,p.at(0));});
        return LoweredPackageGraph{slots,graph};
    };
}
void packages(const fs::path& root) {
    fixtures(root);p::IndexedTensorSource source(index_text,artifacts(root));
    auto model=std::make_shared<VrmModel>((root/"multi.vrm").string());
    auto declaration=PackageDeclaration::parse(model->graph());
    require(declaration.graphs().size()==1 && declaration.bindings().size()==2 && declaration.instances().size()==2,"Catalog counts");
    st::Backend backend;int factories=0;auto lower=lowerer(backend,factories);
    auto admitted=declaration.admit(model,lower);
    require(factories==0 && admitted.graph_count()==1 && admitted.binding_count()==2,"Premature endpoint execution");
    auto context=admitted.create_context();
    require(context.at({0}).graph()==context.at({1}).graph(),"Graph identity not shared");
    require(context.at({0}).parameters()[0].data()!=context.at({1}).parameters()[0].data(),"Binding storage alias");
    const auto plan=ExecutionProgram::per_step({{0},{1},{0}});auto schedule=st::program();
    auto selected=plan.admit(context,schedule,DType::F32);
    require(selected[0]->binding_id().value==10 && selected[1]->binding_id().value==20 && selected[2]==selected[0],"Selection identity");
    auto result=SamplingRuntime(backend).run(context,plan,schedule);
    for(int i=0;i<3;++i) {
        const auto& parameter=result.trace.at("step."+std::to_string(i)+".endpoint.parameter");
        require(parameter.data_as<float>()[0]==(i==1?3:1),"A B A numerical binding trace");
    }
    require(backend.rng_calls==1,"RNG restarted on binding switch");
    st::Backend reference_backend;
    st::LegacyDenoiser reference(reference_backend,model->tensor("canonical.b"));
    auto old=SamplingRuntime(reference_backend).run(reference,schedule);
    st::Backend single_backend;int single_factories=0;
    auto single=declaration.admit(model,lowerer(single_backend,single_factories)).create_context();
    auto modern=SamplingRuntime(single_backend).run(single,ExecutionProgram::uniform({0}),schedule);
    // Same numerical trace; the generic catalog also declares unused binding B.
    st::exact(old.final_latent,modern.final_latent);
    for(const auto& [name,value]:old.trace) st::exact(value,modern.trace.at(name));
    for(const auto& [label,wire]:std::vector<std::pair<std::string,std::string>>{
        {"missing binding",replace(catalog,"\"bindings\":[", "\"missing\":[")},
        {"duplicate binding ID",replace(catalog,"\"id\":20","\"id\":10")},
        {"duplicate instance ID",replace(catalog,"\"id\":1,\"graph\":0,\"binding\":20","\"id\":0,\"graph\":0,\"binding\":20")},
        {"unknown graph ID",replace(catalog,"\"id\":10,\"graph\":0","\"id\":10,\"graph\":9")},
        {"missing slot",replace(catalog,"\"param.0\":\"canonical.b\"","\"param.1\":\"canonical.b\"")},
        {"extra slot",replace(catalog,"\"param.0\":\"canonical.b\"","\"param.0\":\"canonical.b\",\"param.1\":\"canonical.a\"")},
        {"shape mismatch",replace(catalog,"canonical.b","canonical.c")},
        {"dtype mismatch",replace(catalog,"canonical.b","canonical.d")},
        {"unknown tensor",replace(catalog,"canonical.b","absent")},
        {"unknown instance binding",replace(catalog,"\"binding\":20","\"binding\":99")},
        {"unknown capability",replace(catalog,"binding_catalog.v1","unsupported.v1")},
        {"missing capability",replace(catalog,"[\"binding_catalog.v1\"]","[]")},
        {"selection in package",replace(catalog,"\"schema_version\":2","\"selection\":{},\"schema_version\":2")},
        {"guidance in package",replace(catalog,"\"schema_version\":2","\"guidance\":4,\"schema_version\":2")},
        {"unknown architecture semantics",replace(catalog,"test.affine.v1","unknown.v1")},
        {"invalid ID",replace(catalog,"\"id\":10","\"id\":-1")}}) {
        int before=factories;
        reject(label,[&]{emit(root/"invalid.vrm",source,wire);
            auto invalid=std::make_shared<VrmModel>((root/"invalid.vrm").string());
            auto parsed=PackageDeclaration::parse(invalid->graph());parsed.admit(invalid,lower);});
        require(factories==before,"Invalid package constructed numerical endpoint");
    }
    // Cross-graph interface compatibility is an Execution Program property,
    // never an existence-only catalog restriction or a numerical fallback.
    auto two=replace(catalog,"\"declaration\":{\"topology\":\"test.affine.v1\"}}]",
        "\"declaration\":{\"topology\":\"test.affine.v1\"}},{\"id\":1,\"declaration\":{\"topology\":\"test.affine.v1\"}}]");
    two=replace(two,"\"id\":20,\"graph\":0","\"id\":20,\"graph\":1");
    two=replace(two,"\"id\":1,\"graph\":0,\"binding\":20","\"id\":1,\"graph\":1,\"binding\":20");
    int calls=0;
    auto incompatible=PackageDeclaration::parse(Json::parse(two)).admit(model,[&](const Json& j) {
        auto g=lower(j);if(++calls==2) {auto i=g.executable->interface();i.conditioning_contract=7;
            g.executable=std::make_shared<const ComponentGraphDefinition>(i,g.executable->parameters(),
                [&](const std::vector<Tensor>& p){return std::make_unique<st::LegacyDenoiser>(backend,p[0]);});}return g;
    }).create_context();
    const auto evaluations=backend.rng_calls;
    reject("incompatible selectable interfaces",[&]{SamplingRuntime(backend).run(incompatible,plan,schedule);});
    require(backend.rng_calls==evaluations,"Invalid execution reached RNG/numerics");
    reject("unknown selected instance",[&]{ExecutionProgram::uniform({99}).admit(context,schedule,DType::F32);});
    // An execution context independently retains the real file owner.
    std::weak_ptr<VrmModel> weak;
    {
        auto owned=std::make_shared<VrmModel>((root/"multi.vrm").string());weak=owned;
        auto lease=PackageDeclaration::parse(owned->graph()).admit(owned,lower).create_context();
        owned.reset();require(!weak.expired(),"Missing source lease");
        require(lease.at({1}).parameters()[0].data_as<float>()[0]==3,"Stale borrowed tensor");
    }
    require(weak.expired(),"Context owner leak");
    std::cout<<"PASS package admission, one graph/two bindings, A B A, legacy trace, lease, execution separation\n";
}
}
int main(int argc,char** argv) {
    try {
        require(argc==2,"Expected synthetic output directory");fs::path root=fs::path(argv[1])/std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        std::cout<<"FIXTURE_ROOT="<<root.string()<<'\n';
        const auto baseline=descriptors();indexed(root);packages(root);
        require(descriptors()==baseline,"Final descriptor count not at baseline");
        std::cout<<"PASS total_rejections="<<rejections<<" owned_fds_after_teardown=0\n";
        return 0;
    }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
}
