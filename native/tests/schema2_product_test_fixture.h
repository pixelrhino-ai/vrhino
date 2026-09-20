#pragma once
#include "vrhino/product/package_preflight.h"
#include "vrhino/product/converter.h"
#include "vrhino/product/program_declaration.h"
#include "../src/architectures/wan_family.h"
#include "neural_graph_test_backend.h"
#include "vrhino/tensor_util.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cstring>
#include <functional>

using namespace vrhino;
namespace p=vrhino::product;
namespace fs=std::filesystem;
Json j(Json::Value v){return Json(std::move(v));}
Json set(Json x,const std::string& key,Json value){auto o=x.object();o[key]=std::move(value);return j(o);}
Json read(const fs::path& path){std::ifstream f(path);require(f.good(),"Missing fixture");return Json::parse(std::string(std::istreambuf_iterator<char>(f),{}));}
void write(const fs::path& path,const Json& value){std::ofstream(path)<<value.serialize();}
int rejected=0;
size_t descriptors(){
#ifdef __linux__
    return std::distance(fs::directory_iterator("/proc/self/fd"),fs::directory_iterator{});
#else
    return 0;
#endif
}
template<class F>void reject(const std::string& name,F f){const auto before=descriptors();bool failed=false;try{f();}catch(const std::exception& e){failed=true;std::cout<<"REJECT "<<name<<": "<<e.what()<<'\n';}require(failed,"Unexpected acceptance: "+name);require(descriptors()==before,"Descriptor leak: "+name);++rejected;}
struct Source:p::TensorSource {
    std::map<std::string,p::SourceTensorDescriptor> values;
    const std::map<std::string,p::SourceTensorDescriptor>& tensors()const noexcept override{return values;}
    void read_tensor(const p::SourceTensorDescriptor&,uint64_t,void* out,size_t bytes)const override{std::memset(out,0,bytes);}
};
struct Fixture {
    fs::path root; Json graph,metadata,manifest,local;Source source;std::vector<p::TensorMapping> mappings;
    Fixture(fs::path r,const fs::path& specs):root(std::move(r)){
        fs::create_directories(root);
        auto legacy=read(specs/"wan2_1_t2v_1_3b/graph.json").at("architecture_graph");
        auto c=legacy.at("config");
        for(auto [key,value]:std::initializer_list<std::pair<const char*,int>>{{"dim",12},{"num_heads",2},{"num_layers",2},{"ffn_dim",20},{"freq_dim",6},{"text_dim",8},{"text_len",4}})c=set(c,key,j(int64_t(value)));
        auto canonical=canonical_architecture_from_vrm_descriptor(set(legacy,"config",c));
        auto definition=wan_family::lower(canonical);
        Json::Array bindings,instances;
        for(int n=0;n<2;++n){Json::Object parameters;
            for(const auto& s:definition->parameters->slots()){
                const std::string name="binding."+std::to_string(n)+"."+s.role;parameters.emplace(s.role,j(name));add(name,s.shape,s.dtype);
            }
            bindings.push_back(Json::parse("{\"id\":"+std::to_string(n)+",\"graph\":0,\"parameters\":"+j(parameters).serialize()+"}"));
            instances.push_back(Json::parse("{\"id\":"+std::to_string(n)+",\"graph\":0,\"binding\":"+std::to_string(n)+"}"));
        }
        graph=Json::parse(R"({"schema_version":2,"required_capabilities":["binding_catalog.v1"],"graphs":[],"bindings":[],"instances":[]})");
        graph=set(graph,"graphs",j(Json::Array{j(Json::Object{{"id",j(int64_t(0))},{"declaration",canonical.declaration()}})}));graph=set(graph,"bindings",j(bindings));graph=set(graph,"instances",j(instances));
        add("decoder.mock",{1},DType::F32);
        auto programs=Json::parse(R"({"schema":"vrhino.programs.v1","required_capabilities":["execution.per_step.v1","sampling.flow_sigma_cfg.v1"],"execution":{"kind":"per_step","instance_ids":[0,1,0]},"sampling":{"prediction":"flow","solver":"multistep_predictor_corrector","maximum_order":2,"schedule":"flow_sigma","branch_order":"unconditional_then_conditional","transitions":[{"model_timestep":900,"sigma":0.9,"next_sigma":0.6,"guidance_scale":4},{"model_timestep":600,"sigma":0.6,"next_sigma":0.3,"guidance_scale":3},{"model_timestep":300,"sigma":0.3,"next_sigma":0,"guidance_scale":4}]}})");
        metadata=Json::parse(R"({"architecture":"wan","product":"test/catalog:1","conversion_provenance":{"actual_repository":"test/source","actual_revision":"fixed","manifest_sha256":""},"shared_components":{"decoder":{"schema_version":1,"implementation_id":"dit_flow.wan.vae_decoder.v1","latent_contract":{"channels":16,"layout":"BCTHW"},"runtime_tensor_bindings":{"mock":"decoder.mock"}},"decoder_slots":[{"name":"decoder.mock","shape":[1],"dtype":"float32"}],"text_encoder_declaration":"conditioning.json","text_encoder_resource":"conditioning.safetensors","tokenizer_resource":"tokenizer.json"}})");
        metadata=set(metadata,"programs",programs);
        write(root/"source-manifest.tsv",j(std::string("synthetic source identity")));
        metadata=set(metadata,"conversion_provenance",set(metadata.at("conversion_provenance"),"manifest_sha256",j(p::sha256_file(root/"source-manifest.tsv"))));
        write(root/"conditioning.json",Json::parse(R"({"weight":"embedding"})"));
        write(root/"conditioning-index.json",Json::parse(R"({"weight_map":{"embedding":"conditioning.safetensors"}})"));
        write(root/"tokenizer.json",Json::parse(R"({"model":{"type":"Unigram"}})"));
        std::string header=R"({"embedding":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}})";while(header.size()%8)header+=' ';
        {std::ofstream f(root/"conditioning.safetensors",std::ios::binary);uint64_t size=header.size();f.write(reinterpret_cast<const char*>(&size),8);f<<header;float zero=0;f.write(reinterpret_cast<const char*>(&zero),4);}
        write(root/"profile.json",Json::parse(R"({"latent_shape":[1,16,1,4,4],"seed":0})"));
        manifest=Json::parse(R"({"schema_version":2,"identity":{"namespace":"test","name":"catalog","version":"1","architecture":"wan","publisher":"test"},"compatibility":{"runtime_contract":"cuda-v1","vrm_schema":{"format_major":0,"format_minor":1,"metadata_schema":1}},"artifacts":[],"entrypoint":{"runtime_artifact":"runtime","components":[],"default_preset":"structural"},"defaults":{"default_preset":"structural","presets":{"structural":{"profile_artifact":"profile"}}},"hardware":{"presets":{"structural":{"minimum_vram_bytes":null,"recommended_vram_bytes":null}}},"source":{"repository":"test/source","revision":"fixed","converter_version":"fixture"},"license":{"identifier":"test-only","artifact":"source","upstream_notice":"synthetic"},"admission":{"graph_artifact":"graph","metadata_artifact":"metadata","programs_artifact":"programs","source_manifest_artifact":"source","structural_only":true,"required_capabilities":["binding_catalog.v1","execution.per_step.v1","sampling.flow_sigma_cfg.v1"],"resources":{"conditioning_declaration":"conditioning","conditioning_index":"index","conditioning_weights":"weights","tokenizer":"tokenizer"}}})");
        emit();
    }
    void add(const std::string& name,const std::vector<int64_t>& shape,DType dtype){
        p::SourceTensorDescriptor d;d.name=name;d.dtype=dtype;d.source_dtype=dtype_name(dtype);d.shape=shape;d.byte_length=shape_numel(shape)*dtype_size(dtype);source.values.emplace(name,d);
        mappings.emplace_back(name,dtype,shape,"identity_bytes",name,dtype,shape,"parameter","weight");
    }
    void emit(){
        fs::remove(root/"model.vrm");p::write_vrm_streaming(root/"model.vrm","dit-flow","wan",metadata,graph,source,mappings);
        write(root/"graph.json",graph);write(root/"metadata.json",metadata);write(root/"programs.json",metadata.at("programs"));refresh();
    }
    void refresh(){
        Json::Array arts;Json::Object paths;
        for(const auto& [id,file]:std::initializer_list<std::pair<const char*,const char*>>{{"runtime","model.vrm"},{"graph","graph.json"},{"metadata","metadata.json"},{"programs","programs.json"},{"source","source-manifest.tsv"},{"conditioning","conditioning.json"},{"index","conditioning-index.json"},{"weights","conditioning.safetensors"},{"tokenizer","tokenizer.json"},{"profile","profile.json"}}){
            arts.push_back(j(Json::Object{{"id",j(std::string(id))},{"path",j(std::string(file))},{"role",j(std::string("package.resource"))},{"required",j(true)},{"size",j(int64_t(fs::file_size(root/file)))},{"sha256",j(p::sha256_file(root/file))}}));paths[id]=j(std::string(file));
        }
        manifest=set(manifest,"artifacts",j(arts));local=j(Json::Object{{"schema",j(std::string("vrhino.local-resources.v1"))},{"resources",j(paths)}});save();
    }
    void save(){write(root/"vrhino-model.json",manifest);write(root/"local.json",local);}
    auto preflight(){return p::preflight_local_product(root/"vrhino-model.json",root/"local.json");}
};
