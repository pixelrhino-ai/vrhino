#include "vrhino/product/wan_family_conversion.h"
#include "vrhino/error.h"
#include "vrhino/loader.h"
#include "../architectures/wan_family.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>

namespace vrhino::product {
namespace {
namespace fs=std::filesystem;
Json j(Json::Value v) { return Json(std::move(v)); }
Json obj(Json::Object v) { return j(std::move(v)); }
Json arr(Json::Array v) { return j(std::move(v)); }
Json ints(const std::vector<int64_t>& v) { Json::Array a; for(auto n:v)a.push_back(j(n)); return arr(std::move(a)); }
std::string text(const fs::path& p) {
    require(fs::file_size(p)<=8*1024*1024,"Converter text input exceeds bound");
    std::ifstream in(p,std::ios::binary);require(in.good(),"Missing converter text input: "+p.string());
    return std::string(std::istreambuf_iterator<char>(in),{});
}
void put(const fs::path& p,const std::string& s) {
    require(!fs::exists(p),"Output already exists");std::ofstream f(p,std::ios::binary);
    f<<s;f.flush();require(f.good(),"Converter output write failure");
}
std::vector<std::string> split(const std::string& s,char c) {
    std::vector<std::string> v; std::stringstream in(s);std::string part;
    while(std::getline(in,part,c)) v.push_back(part);
    return v;
}
struct Artifact { fs::path path; uint64_t size; std::string sha; size_t index; fs::file_time_type modified; };
// Owns stable-open descriptors for all admitted sources. No source decoding or
// object deserialization; PyTorch ranges are frozen inert byte regions.
class QualifiedFiles {
public:
    std::map<std::string,Artifact> artifacts;
    std::unique_ptr<FrozenTensorSource> bytes;
    QualifiedFiles(const Json& contract,const fs::path& model,const fs::path& semantic,
                   const WorkProgressCallback& progress) {
        std::vector<fs::path> paths;std::map<std::string,SourceTensorDescriptor> ranges;
        for(const auto& row:contract.at("artifacts").array()) {
            auto name=row.at("relative_path").string();auto slash=name.find('/');
            require(slash!=std::string::npos,"Invalid qualified artifact namespace");
            const auto space=name.substr(0,slash);require(space=="model" || space=="official_source","Unknown source namespace");
            const auto root=fs::canonical(space=="model"?model:semantic);auto relative=fs::path(name.substr(slash+1));
            require(!relative.is_absolute(),"Absolute source path");for(auto part:relative)require(part!="..","Escaping source path");
            auto path=root/relative;require(fs::is_regular_file(fs::symlink_status(path)),"Missing/nonregular qualified source: "+name);
            auto resolved=fs::relative(fs::canonical(path),root);for(auto part:resolved)require(part!="..","Escaping source symlink");
            auto n=std::stoull(row.at("size_bytes").string());require(n>0 && fs::file_size(path)==n,"Qualified source size drift: "+name);
            const size_t index=paths.size();paths.push_back(path);
            require(artifacts.emplace(name,Artifact{path,n,row.at("sha256").string(),index,fs::last_write_time(path)}).second,"Duplicate source artifact");
            ranges.emplace(name,SourceTensorDescriptor{name,DType::U8,"U8",{static_cast<int64_t>(n)},index,0,n});
        }
        bytes=std::make_unique<FrozenTensorSource>(paths,std::move(ranges));
        // IndexedTensorSource verifies each shard on the descriptor it owns.
        // Do not hash those same large files twice. All other artifacts are
        // verified here; no output is opened until every source is admitted.
        for(const auto& [name,a]:artifacts) if(!name.ends_with(".safetensors")) {
            std::cerr<<"VERIFY "<<name<<'\n';bytes->verify_identity(a.index,a.size,a.sha,progress);
        }
        unchanged();
    }
    void unchanged() const {
        for(const auto& [name,a]:artifacts)require(fs::is_regular_file(fs::symlink_status(a.path)) &&
            fs::file_size(a.path)==a.size && fs::last_write_time(a.path)==a.modified,"Source changed during conversion: "+name);
    }
    std::string read(const std::string& name) const {
        const auto& a=artifacts.at(name);require(a.size<=8*1024*1024,"Bounded source text required");
        std::string s(a.size,'\0');bytes->read_tensor(bytes->tensors().at(name),0,s.data(),s.size());return s;
    }
    void copy(const std::string& name,const fs::path& dest) const {
        const auto& a=artifacts.at(name);std::ofstream f(dest,std::ios::binary);std::vector<char> b(8*1024*1024);
        for(uint64_t offset=0;offset<a.size;) { size_t n=std::min<uint64_t>(b.size(),a.size-offset);
            bytes->read_tensor(bytes->tensors().at(name),offset,b.data(),n);f.write(b.data(),n);offset+=n; }
        f.flush();require(f.good(),"Resource copy failed");require(sha256_file(dest)==a.sha,"Resource copy identity mismatch");
    }
};
class JoinedSource final:public TensorSource {
public:
    struct Region {const TensorSource* source;SourceTensorDescriptor descriptor;};
    std::map<std::string,SourceTensorDescriptor> catalog;std::map<std::string,Region> regions;
    void add(std::string name,const TensorSource& source,SourceTensorDescriptor d) {
        auto published=d;published.name=name;
        require(catalog.emplace(name,published).second,"Duplicate joined tensor");regions.emplace(name,Region{&source,std::move(d)});
    }
    const std::map<std::string,SourceTensorDescriptor>& tensors() const noexcept override {return catalog;}
    void read_tensor(const SourceTensorDescriptor& d,uint64_t o,void* p,size_t n)const override {
        const auto& r=regions.at(d.name);r.source->read_tensor(r.descriptor,o,p,n);
    }
};
Json assignment(const std::string& source,const std::string& object,const std::string& field) {
    // Deliberately bounded literal extraction from checksum-qualified Python;
    // this never imports Python, evaluates expressions or accepts arbitrary code.
    std::regex line("(^|\\n)"+object+"\\."+field+"[ \\t]*=[ \\t]*([^\\n#]+)");
    std::sregex_iterator begin(source.begin(),source.end(),line),end;require(begin!=end,"Missing qualified config literal: "+field);
    auto value=(*begin)[2].str();require(++begin==end,"Ambiguous qualified config literal");
    std::replace(value.begin(),value.end(),'(','[');std::replace(value.begin(),value.end(),')',']');
    value=std::regex_replace(value,std::regex("\\bTrue\\b"),"true");
    value=std::regex_replace(value,std::regex("\\bFalse\\b"),"false");
    return Json::parse(value);
}
std::vector<TensorMapping> frozen_map(const fs::path& path,const std::string& name,
                                    const QualifiedFiles& files,JoinedSource& joined) {
    auto lines=split(text(path),'\n');require(!lines.empty() && split(lines[0],'\t').size()==11,"Invalid frozen mapping header");
    std::vector<TensorMapping> result; const auto& artifact=files.artifacts.at(name);
    for(size_t i=1;i<lines.size();++i) {
        const auto f=split(lines[i],'\t');require(f.size()==11,"Invalid frozen map row");
        std::vector<int64_t> shape;for(const auto& v:split(f[3],','))shape.push_back(std::stoll(v));
        auto dtype=f[2]=="F32"?DType::F32:DType::BF16;
        require(f[2]=="F32" || f[2]=="BF16","Unsupported frozen dtype");
        require(f[2]==f[7] && f[3]==f[8] && f[5]=="identity_bytes","Frozen mapping must be identity");
        const uint64_t count=shape_numel(shape)*dtype_size(dtype),offset=std::stoull(f[4]);
        require(offset<=artifact.size && count<=artifact.size-offset,"Frozen range outside source");
        const auto source_name=name+":"+f[1];
        // FrozenTensorSource's stable file owners also serve these validated
        // subregions. No reopen of a possibly replaced PTH pathname.
        SourceTensorDescriptor d{source_name,dtype,f[2],shape,artifact.index,offset,count};
        joined.add(source_name,*files.bytes,d);
        result.emplace_back(source_name,dtype,shape,"identity_bytes",f[6],dtype,shape,f[9],f[10]);
    }
    return result;
}
Json make_programs(const std::string& config,const std::string& shared) {
    auto number=[&](const char* name){return assignment(config,"t2v_A14B",name).number();};
    const int steps=static_cast<int>(number("sample_steps"));
    const auto train=assignment(shared,"wan_shared_cfg","num_train_timesteps").integer();
    const double shift=number("sample_shift"),boundary=number("boundary")*train;
    const auto scales=assignment(config,"t2v_A14B","sample_guide_scale").array();
    require(steps>0 && steps<=10000 && train>1 && shift>0 && boundary>=0 && boundary<=train && scales.size()==2,"Invalid source program parameters");
    std::vector<double> sigmas;
    const double start=static_cast<double>(static_cast<float>(1.0-1.0/train));
    for(int i=0;i<steps;++i) { const double sigma=start*(1.0-static_cast<double>(i)/steps);sigmas.push_back(shift*sigma/(1+(shift-1)*sigma)); }
    sigmas.push_back(0.0);Json::Array ids,transitions;
    for(int i=0;i<steps;++i) {
        int64_t time=static_cast<int64_t>(sigmas[i]*train);bool selected=time>=boundary;
        ids.push_back(j(int64_t(selected?0:1)));
        transitions.push_back(obj({{"model_timestep",j(time)},{"sigma",j(double(float(sigmas[i])))},
            {"next_sigma",j(double(float(sigmas[i+1])))},{"guidance_scale",scales[selected?1:0]}}));
    }
    return obj({{"schema",j(std::string("vrhino.programs.v1"))},
        {"required_capabilities",Json::parse("[\"execution.per_step.v1\",\"sampling.flow_sigma_cfg.v1\"]")},
        {"execution",obj({{"kind",j(std::string("per_step"))},{"instance_ids",arr(std::move(ids))}})},
        {"sampling",obj({{"prediction",j(std::string("flow"))},{"solver",j(std::string("multistep_predictor_corrector"))},
            {"maximum_order",j(int64_t(2))},{"schedule",j(std::string("flow_sigma"))},
            {"branch_order",j(std::string("unconditional_then_conditional"))},{"transitions",arr(std::move(transitions))}})}});
}
void pin_spec(const fs::path& spec) {
    require(sha256_file(spec/"spec-files.sha256")=="ee88b4b3cbdf34539b16e318898ab43701118d44991c5b9696d79d356d384eed","Unqualified converter spec identity");
    for(const auto& line:split(text(spec/"spec-files.sha256"),'\n')) {
        require(line.size()>66 && line.substr(64,2)=="  ","Invalid spec manifest");
        const auto name=line.substr(66);require(name.find('/')==std::string::npos && name.find('\\')==std::string::npos,"Invalid spec filename");
        require(sha256_file(spec/name)==line.substr(0,64),"Converter spec drift: "+name);
    }
}
}
Json lower_wan_source_config(const Json& c,const Json& patch,int64_t limit) {
    require(c.at("model_type").string()=="t2v" && c.at("qk_norm").boolean() && c.at("cross_attn_norm").boolean(),"Unsupported source architecture semantics");
    auto d=Json::parse(R"({"schema":"vrhino.architecture.v1","topology":"modulated_self_cross_ffn.v1","dimensions":{},"normalization":{"query_key":"rms_norm","self_pre":"layer_norm","cross_pre":"affine_layer_norm","feed_forward_pre":"layer_norm","epsilon":0.000001},"attention":{"self":"global_noncausal","cross":"global_noncausal","head_layout":"BSHD"},"patch":{"size":[1,2,2],"latent_layout":"BCTHW"},"position":{"encoding":"rope_3d_adjacent","axis_partition":"temporal_remainder_spatial_sixths","theta":10000},"conditioning":{"text_feature_size":4096,"text_token_limit":512,"text_padding":"zero_before_projection","timestep":"per_sample_sinusoidal","branch_order":"unconditional_then_conditional"},"parameters":{"slot_schema":"modulated_self_cross_ffn.slots.v1","storage_dtype":"float32","layout":"contiguous"}})").object();
    Json::Object dims;
    for(const auto& [canonical,upstream]:std::map<std::string,std::string>{{"hidden_size","dim"},{"head_count","num_heads"},{"block_count","num_layers"},{"feed_forward_size","ffn_dim"},{"time_frequency_size","freq_dim"},{"input_channels","in_dim"},{"output_channels","out_dim"}}) dims.emplace(canonical,c.at(upstream));
    d["dimensions"]=obj(dims);auto norm=d.at("normalization").object();norm["epsilon"]=c.at("eps");d["normalization"]=obj(norm);
    d["patch"]=obj({{"size",patch},{"latent_layout",j(std::string("BCTHW"))}});
    auto condition=d.at("conditioning").object();condition["text_token_limit"]=j(limit);d["conditioning"]=obj(condition);
    auto result=obj(d);(void)wan_family::lower(CanonicalArchitectureDeclaration::parse(result));return result;
}
std::vector<TensorMapping> map_wan_parameters(const TensorSource& source,const Json& canonical,uint32_t binding) {
    const auto family=wan_family::lower(CanonicalArchitectureDeclaration::parse(canonical));
    require(source.tensors().size()==family->parameters->slots().size(),"Unmapped or extra transformer tensor");
    std::vector<TensorMapping> mappings;
    for(const auto& slot:family->parameters->slots()) {
        auto found=source.tensors().find(slot.role);require(found!=source.tensors().end(),"Missing canonical source slot: "+slot.role);
        const auto& t=found->second;require(t.shape==slot.shape && t.dtype==slot.dtype &&
            t.byte_length==uint64_t(shape_numel(slot.shape))*dtype_size(slot.dtype),"Transformer shape/dtype/byte contract mismatch: "+slot.role);
        mappings.emplace_back(t.name,t.dtype,t.shape,"identity_bytes","parameters."+std::to_string(binding)+"."+slot.role,t.dtype,t.shape,"denoiser","weight");
    }
    return mappings;
}
Json verify_wan_family_structure(const fs::path& path) {
    // No payload traversal by mmap: checksums are verified by bounded streaming
    // separately. Creating borrowed tensor descriptors does not load weights.
    auto model=std::make_shared<VrmModel>(path.string(),false);
    require(model->architecture_id()=="wan" && model->profile_id()=="dit-flow","Family VRM identity mismatch");
    auto catalog=PackageDeclaration::parse(model->graph());
    require(catalog.graphs().size()==1 && catalog.bindings().size()==2 && catalog.instances().size()==2,"Dual binding catalog cardinality");
    const auto definition=wan_family::lower(CanonicalArchitectureDeclaration::parse(catalog.graphs().at(0)));
    require(definition->config.dim==5120 && definition->config.heads==40 && definition->config.layers==40 &&
        definition->config.ffn==13824 && definition->head_dim==128 && definition->rope_axes==std::vector<int>({44,42,42}) &&
        definition->parameters->slots().size()==1095,"Qualified family dimensions mismatch");
    std::set<const void*> addresses;
    for(uint32_t id:{0U,1U}) {
        const auto& binding=catalog.bindings().at(id);require(binding.graph==0,"Graph not shared");
        require(catalog.instances().at(id).binding==id,"Instance binding mismatch");
        std::map<std::string,const Tensor*> tensors;
        for(const auto& [slot,name]:binding.parameters) {
            require(name=="parameters."+std::to_string(id)+"."+slot,"Canonical binding namespace mismatch");
            const auto& tensor=model->tensor(name);require(addresses.insert(tensor.data()).second,"Binding storage aliased");tensors.emplace(slot,&tensor);
        }
        (void)AdmittedArchitectureBinding::admit(definition->parameters,tensors,BorrowedBindingLifetime::ExplicitOwners,{model});
    }
    auto declared=admit_program_declaration(model->metadata().at("programs"),catalog);
    const auto& decoder=model->metadata().at("shared_components").at("decoder");
    require(decoder.at("implementation_id").string()=="dit_flow.wan.vae_decoder.v1","Decoder family contract mismatch");
    const auto& slots=model->metadata().at("shared_components").at("decoder_slots").array();
    require(slots.size()==194 && decoder.at("runtime_tensor_bindings").object().size()==194,"Decoder slot count mismatch");
    for(const auto& slot:slots) {
        std::vector<int64_t> shape;for(const auto& n:slot.at("shape").array())shape.push_back(n.integer());
        validate_architecture_tensor(model->tensor(slot.at("name").string()),shape,DType::F32);
    }
    for(const auto& [slot,name]:decoder.at("runtime_tensor_bindings").object()) { (void)slot; (void)model->tensor(name.string()); }
    require(model->tensors().size()==2384,"Unexpected canonical tensor count");
    const auto& provenance=model->metadata().at("conversion_provenance");
    require(provenance.at("manifest_sha256").string()=="4ea6995c08502841565b9dd2e612af0dd97ddfb49ab17721339fc5c077338911" &&
            provenance.at("actual_revision").string()=="3f42affa3a1f1c6bd1f14f4cd01cdb90373af3d7","Source provenance mismatch");
    return obj({{"status",j(std::string("PASS"))},{"tensor_count",j(int64_t(model->tensors().size()))},
        {"graph_count",j(int64_t(1))},{"binding_count",j(int64_t(2))},{"instance_count",j(int64_t(2))},
        {"slots_per_binding",j(int64_t(1095))},{"steps",j(int64_t(declared.sampling.steps))},
        {"file_size",j(int64_t(model->file_size()))},{"numerical_execution",j(false)}});
}
Json convert_wan_family_package(const fs::path& model_root,const fs::path& semantic_root,
                                const fs::path& spec,const fs::path& output,const WorkProgressCallback& progress) {
    pin_spec(spec);const auto plan=Json::parse(text(spec/"conversion.json"));
    const auto contract=Json::parse(text(spec/"source-contract.json"));
    const auto manifest_sha=sha256_file(spec/"source-manifest.tsv");
    require(manifest_sha==plan.at("manifest_sha256").string() && manifest_sha==contract.at("manifest_sha256").string() &&
        contract.at("qualification_status").string()=="PASS","Source manifest admission failed");
    require(contract.at("source_roots").at("model").at("revision").string()=="3f42affa3a1f1c6bd1f14f4cd01cdb90373af3d7" &&
        contract.at("source_roots").at("model").at("repository").string()=="https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B" &&
        contract.at("expected_huggingface_reference").at("revision").string()=="c8c270b13ee05bfa474194ac9fb07a5868a97cea","Qualified source provider/revision drift");
    require(!fs::exists(output),"Immutable conversion output directory already exists");
    fs::create_directories(output.parent_path());require_conversion_disk_space(127000000000ULL,fs::space(output.parent_path()).available);
    QualifiedFiles files(contract,model_root,semantic_root,progress);
    const auto config_text=files.read(plan.at("semantic_config").string());const auto shared=files.read(plan.at("shared_config").string());
    auto patch=assignment(config_text,"t2v_A14B","patch_size");auto limit=assignment(shared,"wan_shared_cfg","text_len").integer();
    Json canonical;JoinedSource source;std::vector<std::unique_ptr<IndexedTensorSource>> owners;std::vector<TensorMapping> mappings;
    Json::Array bindings,instances;std::string mapping_text="binding\tsource_tensor\tcanonical_slot\tdestination\tdtype\tshape\tbytes\n";
    for(uint32_t id=0;id<2;++id) {
        const auto prefix=plan.at("transformer_sources").array().at(id).string();
        auto config=Json::parse(files.read(prefix+"/config.json")).object();
        // The checkpoint JSON omits constructor/config defaults. Resolve them
        // at the qualified source boundary, never in Architecture or Runtime.
        for(const auto* key:{"qk_norm","cross_attn_norm"})
            config[key]=assignment(config_text,"t2v_A14B",key);
        for(const auto* key:{"dim","num_heads","num_layers","ffn_dim","freq_dim","eps"})
            require(config.at(key).number()==assignment(config_text,"t2v_A14B",key).number(),"Source configuration disagreement");
        require(config.at("text_len").integer()==limit &&
            canonical_json(assignment(config_text,"t2v_A14B","window_size"))=="[-1,-1]","Source attention/text semantics mismatch");
        const auto lowered=lower_wan_source_config(obj(config),patch,limit);
        require(canonical_json(lowered.at("dimensions"))==canonical_json(plan.at("expected_dimensions")),"Product family dimensions drift");
        if(id==0)canonical=lowered;else require(canonical_json(canonical)==canonical_json(lowered),"Bindings cannot share graph");
        const auto index=files.read(prefix+"/diffusion_pytorch_model.safetensors.index.json");
        std::map<std::string,IndexedSourceArtifact> shards;
        for(const auto& [name,value]:files.artifacts) if(name.starts_with(prefix+"/") && name.ends_with(".safetensors"))
            shards.emplace(fs::path(name).filename().string(),IndexedSourceArtifact{value.path,value.size,value.sha});
        auto indexed=std::make_unique<IndexedTensorSource>(index,shards,progress);
        auto map=map_wan_parameters(*indexed,canonical,id);require(map.size()==1095,"Incomplete real tensor map");
        Json::Object parameters;
        for(auto& m:map) {
            const auto slot=m.source_name;const auto joined_name="source."+std::to_string(id)+"."+slot;
            const auto& d=indexed->tensors().at(slot);source.add(joined_name,*indexed,d);m.source_name=joined_name;
            parameters.emplace(slot,j(m.destination_name));
            mapping_text+=std::to_string(id)+"\t"+slot+"\t"+slot+"\t"+m.destination_name+"\tF32\t"+canonical_json(ints(d.shape))+"\t"+std::to_string(d.byte_length)+"\n";
            mappings.push_back(m);
        }
        owners.push_back(std::move(indexed));bindings.push_back(obj({{"id",j(int64_t(id))},{"graph",j(int64_t(0))},{"parameters",obj(parameters)}}));
        instances.push_back(obj({{"id",j(int64_t(id))},{"graph",j(int64_t(0))},{"binding",j(int64_t(id))}}));
    }
    auto vae=frozen_map(spec/"vae-mapping.tsv",plan.at("vae_source").string(),files,source);
    require(vae.size()==194,"VAE mapping count");mappings.insert(mappings.end(),vae.begin(),vae.end());
    JoinedSource conditioning;auto umt5=frozen_map(spec/"umt5-mapping.tsv",plan.at("text_source").string(),files,conditioning);require(umt5.size()==242,"UMT5 mapping count");
    auto graph=obj({{"schema_version",j(int64_t(2))},{"required_capabilities",Json::parse("[\"binding_catalog.v1\"]")},
        {"graphs",arr({obj({{"id",j(int64_t(0))},{"declaration",canonical}})})},{"bindings",arr(bindings)},{"instances",arr(instances)}});
    auto programs=make_programs(config_text,shared);(void)admit_program_declaration(programs,PackageDeclaration::parse(graph));
    Json::Array decoder_slots;for(const auto& m:vae)decoder_slots.push_back(obj({{"name",j(m.destination_name)},{"shape",ints(m.destination_shape)},{"dtype",j(std::string("float32"))}}));
    auto provenance=obj({{"actual_repository",j(std::string("https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B"))},
        {"actual_revision",j(std::string("3f42affa3a1f1c6bd1f14f4cd01cdb90373af3d7"))},
        {"core_reference_repository",j(std::string("https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B"))},
        {"core_reference_revision",j(std::string("c8c270b13ee05bfa474194ac9fb07a5868a97cea"))},
        {"whole_directory_hf_identity",j(false)},{"manifest_sha256",j(manifest_sha)},
        {"semantic_source_revision",j(contract.at("source_roots").at("official_source").at("revision").string())},
        {"converter",j(std::string("wan_family.structural.v1"))},{"spec_sha256",j(sha256_file(spec/"spec-files.sha256"))}});
    auto decoder=Json::parse(text(spec/"decoder.json"));
    require(decoder.at("implementation_id").string()=="dit_flow.wan.vae_decoder.v1" &&
        decoder.at("runtime_tensor_bindings").object().size()==vae.size(),"Decoder declaration admission failed");
    for(const auto& [slot,name]:decoder.at("runtime_tensor_bindings").object()) {
        (void)slot;
        require(std::any_of(vae.begin(),vae.end(),[&](const TensorMapping& m){return m.destination_name==name.string();}),
                "Decoder declaration references unknown canonical tensor");
    }
    auto metadata=obj({{"architecture",j(std::string("wan"))},{"product",plan.at("product")},{"programs",programs},
        {"conversion_provenance",provenance},{"shared_components",obj({{"decoder",decoder},
            {"decoder_slots",arr(decoder_slots)},{"text_encoder_resource",j(std::string("conditioning.safetensors"))},
            {"text_encoder_declaration",j(std::string("conditioning.json"))},{"tokenizer_resource",j(std::string("tokenizer/tokenizer.json"))}})}});
    const fs::path staging=output.string()+".partial";require(!fs::exists(staging),"Staging output already exists");fs::create_directory(staging);
    try {
        put(staging/"graph.json",canonical_json(graph));put(staging/"metadata.json",canonical_json(metadata));
        put(staging/"programs.json",canonical_json(programs));put(staging/"tensor-mapping.tsv",mapping_text);
        put(staging/"source-manifest.tsv",text(spec/"source-manifest.tsv"));put(staging/"source-contract.json",text(spec/"source-contract.json"));
        put(staging/"conditioning.json",text(spec/"conditioning.json"));
        auto conditioning_index=Json::parse(text(spec/"conditioning-index.json")).object();
        auto weight_map=conditioning_index.at("weight_map").object();
        require(weight_map.size()==umt5.size(),"Conditioning index slot coverage mismatch");
        for(auto& [slot,resource]:weight_map) {
            require(conditioning.tensors().contains(plan.at("text_source").string()+":"+slot),"Unknown conditioning index tensor");
            resource=j(std::string("conditioning.safetensors"));
        }
        conditioning_index["weight_map"]=obj(weight_map);
        put(staging/"conditioning-index.json",canonical_json(obj(conditioning_index)));
        fs::create_directory(staging/"tokenizer");
        for(const auto& [name,a]:files.artifacts) { (void)a;if(name.starts_with("model/google/umt5-xxl/"))files.copy(name,staging/"tokenizer"/fs::path(name).filename()); }
        std::cerr<<"EMIT conditioning\n";
        write_safetensors_streaming(staging/"conditioning.safetensors",conditioning,umt5,
            {{"vrhino.phase","13C"},{"vrhino.component","text_encoder"},{"vrhino.extraction","sorted-tensor-names-v1"},{"format","pt"}},{},progress);
        require(sha256_file(staging/"conditioning.safetensors",progress)=="1c137395973a8a717bd5d467fec816e77af973474c6d4afd2cb83c78cc489bd6","Shared canonical UMT5 identity drift");
        std::cerr<<"EMIT canonical VRM\n";auto written=write_vrm_streaming(staging/"model.vrm","dit-flow","wan",metadata,graph,source,mappings,{},progress);
        files.unchanged();auto verified=verify_wan_family_structure(staging/"model.vrm");
        // Byte determinism of declarations independent of mapping insertion order.
        require(text(staging/"graph.json")==canonical_json(Json::parse(canonical_json(graph))) &&
                text(staging/"metadata.json")==canonical_json(Json::parse(canonical_json(metadata))),"Noncanonical declaration bytes");
        auto summary=verified.object();summary["sha256"]=j(sha256_file(staging/"model.vrm",progress));
        summary["payload_blake2b128"]=j(written.payload_blake2b128);summary["largest_copy_buffer_bytes"]=j(int64_t(written.largest_buffer_bytes));
        summary["source_manifest_sha256"]=j(manifest_sha);
        put(staging/"conversion-result.json",canonical_json(obj(summary)));
        Json::Array resources;for(auto& e:fs::recursive_directory_iterator(staging))if(e.is_regular_file())
            resources.push_back(obj({{"path",j(fs::relative(e.path(),staging).generic_string())},{"size",j(int64_t(e.file_size()))},
                {"sha256",j(e.path().filename()=="model.vrm"?summary.at("sha256").string():sha256_file(e.path()))}}));
        std::sort(resources.begin(),resources.end(),[](const Json& a,const Json& b){return a.at("path").string()<b.at("path").string();});
        put(staging/"package.json",canonical_json(obj({{"schema",j(std::string("vrhino.structural-package.v1"))},
            {"required_capabilities",Json::parse("[\"binding_catalog.v1\",\"execution.per_step.v1\",\"sampling.flow_sigma_cfg.v1\"]")},
            {"product",plan.at("product")},{"architecture",j(std::string("wan"))},{"resources",arr(resources)}})));
        fs::rename(staging,output);return obj(summary);
    } catch(...) {std::error_code error;fs::remove_all(staging,error);throw;}
}
} // namespace vrhino::product
