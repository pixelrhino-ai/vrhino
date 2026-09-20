// Structural qualification only: real family factories, borrowed parameters,
// no evaluation. The test Backend forbids numeric and device work.
#include "../src/architectures/wan_execution.h"
#include "../src/architectures/wan_self_attention_graph.h"
#include "neural_graph_test_backend.h"
#include "vrhino/package_declaration.h"
#include "vrhino/product/program_declaration.h"
#include "vrhino/product/safetensors.h"
#include "vrhino/product/converter.h"
#include "vrhino/conditioning.h"
#include "vrhino/components.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <limits>
#ifdef __linux__
#include <unistd.h>
#endif

using namespace vrhino;
namespace fs=std::filesystem;
namespace wf=vrhino::wan_family;
namespace product=vrhino::product;
namespace {
Json j(Json::Value v){return Json(std::move(v));}
Json read(const fs::path& p){std::ifstream f(p);require(f.good(),"Missing declaration");return Json::parse(std::string(std::istreambuf_iterator<char>(f),{}));}
size_t rss(){
#ifdef __linux__
    std::ifstream f("/proc/self/statm");size_t total=0,resident=0;f>>total>>resident;return resident*size_t(sysconf(_SC_PAGESIZE));
#else
    return 0; // RSS observation is optional; no platform-specific production code.
#endif
}
struct StructuralBackend final:neural_graph::test::TinyBackend {
    size_t conditioning_calls=0,conditioning_bytes=0;
    std::map<const void*,ExecutionTensorContract> allowed_preparation;
    Tensor copy_to_device(const Tensor& t,DType d) override {
        const auto found=allowed_preparation.find(t.data());
        require(found!=allowed_preparation.end() && t.device().is_host() && t.dtype()==d &&
            t.dtype()==found->second.dtype && t.shape()==found->second.shape && t.bytes()<=1024*1024,
            "Unexpected preparation/upload");
        ++conditioning_calls;conditioning_bytes+=t.bytes();return t; // host alias, zero GPU transfer
    }
    Tensor add(const Tensor&,const Tensor&) override {throw Error("Numerical operation forbidden");}
    Tensor mul(const Tensor&,const Tensor&) override {throw Error("Numerical operation forbidden");}
    Tensor reshape(const Tensor&,const std::vector<int64_t>&) override {throw Error("Numerical operation forbidden");}
};
size_t rejects_count=0;
template<class F>void rejects(const char* name,F action){
    bool caught=false;try{action();}catch(const std::exception& e){caught=true;std::cout<<"REJECT "<<name<<": "<<e.what()<<'\n';}
    require(caught,std::string("Unexpected admission: ")+name);++rejects_count;
}
void provenance(const Json& metadata,const std::string& manifest){
    const auto& p=metadata.at("conversion_provenance");
    require(manifest=="4ea6995c08502841565b9dd2e612af0dd97ddfb49ab17721339fc5c077338911" && p.at("manifest_sha256").string()==manifest,"Source manifest identity mismatch");
    require(p.at("actual_repository").string()=="https://modelscope.cn/models/Wan-AI/Wan2.2-T2V-A14B" && p.at("actual_revision").string()=="3f42affa3a1f1c6bd1f14f4cd01cdb90373af3d7","Primary provenance mismatch");
    require(p.at("core_reference_revision").string()=="c8c270b13ee05bfa474194ac9fb07a5868a97cea" && !p.at("whole_directory_hf_identity").boolean(),"Core equivalence provenance mismatch");
}
void weight_references(const Json& v,const WeightMap& weights,std::set<std::string>& refs){
    if(v.is_object())for(const auto& [k,value]:v.object()){
        if(k=="weight" || k=="bias"){(void)weights.at(value.string());refs.insert(value.string());}
        else weight_references(value,weights,refs);
    }else if(v.is_array())for(const auto& value:v.array())weight_references(value,weights,refs);
}
}
int main(int argc,char** argv){try{
    require(argc==3,"usage: native-load-qualification PACKAGE NEGATIVE_VRM_DIRECTORY");
    const fs::path root=argv[1];const size_t before=rss();
    auto model=std::make_shared<VrmModel>((root/"model.vrm").string(),false);
    const size_t loaded=rss();std::weak_ptr<VrmModel> weak=model;
    require(model->file_size()==114816482368ULL && model->tensors().size()==2384,"Real VRM identity/count");
    require(model->architecture_id()=="wan" && model->profile_id()=="dit-flow","Architecture/profile mismatch");
    require(model->metadata().at("product").string()=="vrhino/wan2.2-t2v-a14b:0.9.0-structural","Product mismatch");
    const auto manifest=product::sha256_file(root/"source-manifest.tsv");provenance(model->metadata(),manifest);
    const auto declaration=PackageDeclaration::parse(model->graph());declaration.validate_tensor_references(*model);
    auto definition=wf::lower(CanonicalArchitectureDeclaration::parse(declaration.graphs().at(0)));
    const auto& c=definition->config;
    require(c.dim==5120 && c.heads==40 && c.layers==40 && c.ffn==13824 && c.frequency==256 && c.text_dim==4096 && c.text_length==512 && c.input_channels==16 && c.output_channels==16,"Canonical dimensions");
    require(c.patch==std::array<int,3>{1,2,2} && c.rope_theta==10000 && c.epsilon==1e-6f && definition->head_dim==128 && definition->rope_axes==std::vector<int>({44,42,42}),"Patch/norm/position semantics");
    require(definition->parameters->slots().size()==1095,"Family slot count");
    std::map<std::string,std::vector<int64_t>> shapes;
    for(const auto& slot:definition->parameters->slots())shapes.emplace(slot.role,slot.shape);
    for(int n=0;n<40;++n){
        const auto p="blocks."+std::to_string(n)+".";
        require(shapes.at(p+"modulation")==std::vector<int64_t>({1,6,5120}),"Block modulation");
        require(shapes.at(p+"cross_attn.k.weight")==std::vector<int64_t>({5120,5120}),"Cross attention");
        require(shapes.at(p+"ffn.0.weight")==std::vector<int64_t>({13824,5120}) && shapes.at(p+"ffn.2.weight")==std::vector<int64_t>({5120,13824}),"FFN topology");
    }
    require(shapes.at("patch_embedding.weight")==std::vector<int64_t>({5120,16,1,2,2}) && shapes.at("head.head.weight")==std::vector<int64_t>({64,5120}),"Patch/head contract");
    auto attention=wan_internal::self_attention_graph(1,1,DType::F32,DType::F32,true,c);
    size_t attention_nodes=0;
    for(const auto& n:attention.description().nodes)if(const auto* a=std::get_if<neural_graph::Attention>(&n.op)){
        require(attention.description().values[n.result].shape==std::vector<int64_t>({1,1,40,128}) && a->scale==1/std::sqrt(128.0f),"Constructed attention dimensions");++attention_nodes;
    }
    require(attention_nodes==1,"Missing self attention node");
    std::cout<<"PASS canonical lowering; 40 block contracts; production self-attention graph nodes="<<attention.description().nodes.size()<<"\n";
    auto policy=PrecisionPolicy::fp32();
    auto backend=std::make_unique<StructuralBackend>();
    auto positive=Tensor::host({1,2,4096},DType::F32),negative=Tensor::host({1,2,4096},DType::F32);
    for(const auto* t:{&positive,&negative})backend->allowed_preparation.emplace(t->data(),ExecutionTensorContract{t->shape(),t->dtype()});
    const std::vector<int64_t> latent{1,16,1,2,2};size_t lower_calls=0;
    auto lower=[&](const Json& config){++lower_calls;auto d=wf::lower(CanonicalArchitectureDeclaration::parse(config));auto e=wf::realize(d,*backend,policy,latent,positive,negative,true,0,0);return LoweredPackageGraph{d->parameters,e.graph};};
    auto admitted=std::make_unique<AdmittedPackage>(declaration.admit(model,lower));
    require(lower_calls==1 && backend->conditioning_calls==0,"Admission must lower once before endpoint creation");
    auto context=std::make_unique<ExecutionContext>(admitted->create_context());
    require(context->at({0}).graph()==context->at({1}).graph() && context->at({0}).binding_id().value==0 && context->at({1}).binding_id().value==1,"Shared graph/separate binding identity");
    require(context->at({0}).graph()->component_interface()==context->at({1}).graph()->component_interface(),"Interface mismatch");
    uint64_t bound_bytes=0;std::set<const void*> addresses;
    for(const auto& instance:context->instances()){
        require(instance.parameters().size()==1095,"Real instance slots");
        for(const auto& t:instance.parameters()){require(addresses.insert(t.data()).second,"Parameters alias across bindings");bound_bytes+=t.bytes();}
    }
    require(backend->conditioning_calls==4 && backend->conditioning_bytes==131072 && backend->calls.empty(),"Unexpected numerical/weight upload");
    auto programs=product::admit_program_declaration(model->metadata().at("programs"),declaration);
    programs.sampling.latent_shape=latent;
    auto plan=programs.execution.admit(*context,programs.sampling,DType::F32);
    require(plan.size()==40 && programs.sampling.contract->solver.maximum_order==2,"Schedule/solver declaration");
    Json::Array table;size_t switches=0;
    for(size_t i=0;i<plan.size();++i){
        const auto time=*programs.sampling.model_timestep_at(i).data_as<int64_t>();
        const auto& flow=programs.sampling.contract->schedule.flow_at(i);
        const auto scale=programs.sampling.guidance_schedule->at(i).scale;
        if(i && plan[i]!=plan[i-1])++switches;
        table.push_back(j(Json::Object{{"step",j(int64_t(i))},{"timestep",j(time)},{"instance",j(int64_t(plan[i]->id().value))},{"sigma",j(double(flow.sigma))},{"guidance",j(double(scale))}}));
    }
    require(switches==1 && plan[25]->id().value==0 && plan[26]->id().value==1,"Resolved plan transition");
    // This is a generic synthetic selector test, not an invented real schedule row.
    auto synthetic=programs.sampling;
    synthetic.contract.reset();synthetic.steps=3;synthetic.model_timesteps.clear();
    // Use the actual flow contract so component interface remains compatible.
    std::vector<FlowScheduleTransition> transitions;
    for(int64_t n:{876,875,874}){auto t=Tensor::host({},DType::I64);*t.data_as<int64_t>()=n;float s=float(3-transitions.size())/4;transitions.push_back({t,s,s-0.25f});}
    synthetic.contract=SamplingContract{{PredictionSemantic::Flow},{SolverSemantic::MultistepPredictorCorrector,2},ScheduleContract::flow_sigma(transitions)};
    synthetic.guidance_schedule=GuidanceSchedule::constant(GuidanceParameters::cfg(1));
    ScalarPartition partition{SelectionCoordinate::ModelTimestepScalar,SelectionComparison::GreaterOrEqual,int64_t(875),{0},{1}};
    auto boundary=ExecutionProgram::partition(partition).admit(*context,synthetic,DType::F32);
    require(boundary[0]->id().value==0 && boundary[1]->id().value==0 && boundary[2]->id().value==1,"Inclusive scalar selector");
    rejects("invalid selector coordinate",[&]{auto x=partition;x.coordinate=static_cast<SelectionCoordinate>(99);ExecutionProgram::partition(x).admit(*context,synthetic,DType::F32);});
    rejects("invalid threshold dtype",[&]{auto x=partition;x.threshold=875.0f;ExecutionProgram::partition(x).admit(*context,synthetic,DType::F32);});
    rejects("unknown component instance",[&]{(void)context->at({99});});
    auto graph_negative=[&](const char* label,auto mutate){rejects(label,[&]{auto root_object=model->graph().object();mutate(root_object);auto d=PackageDeclaration::parse(j(root_object));(void)d.admit(model,lower);});};
    graph_negative("unknown binding",[](auto& g){auto v=g.at("instances").array();auto x=v[0].object();x["binding"]=j(int64_t(99));v[0]=j(x);g["instances"]=j(v);});
    graph_negative("missing binding",[](auto& g){auto v=g.at("bindings").array();v.pop_back();g["bindings"]=j(v);});
    graph_negative("duplicate binding",[](auto& g){auto v=g.at("bindings").array();v.push_back(v[0]);g["bindings"]=j(v);});
    graph_negative("instance wrong graph",[](auto& g){auto v=g.at("instances").array();auto x=v[0].object();x["graph"]=j(int64_t(99));v[0]=j(x);g["instances"]=j(v);});
    graph_negative("binding missing slot",[](auto& g){auto v=g.at("bindings").array();auto b=v[0].object();auto p=b.at("parameters").object();p.erase(p.begin());b["parameters"]=j(p);v[0]=j(b);g["bindings"]=j(v);});
    graph_negative("unknown capability",[](auto& g){g["required_capabilities"]=Json::parse("[\"unavailable.v1\"]");});
    std::map<std::string,const Tensor*> refs;
    for(const auto& [slot,name]:declaration.bindings().at(0).parameters)refs.emplace(slot,&model->tensor(name));
    for(const bool dtype:{false,true})rejects(dtype?"binding dtype mismatch":"binding shape mismatch",[&]{
        const auto& t=*refs.begin()->second;auto altered=Tensor::borrowed(t.data(),t.bytes(),dtype?t.shape():std::vector<int64_t>{1},dtype?DType::BF16:t.dtype());auto copy=refs;copy.begin()->second=&altered;
        AdmittedArchitectureBinding::admit(definition->parameters,copy,BorrowedBindingLifetime::ExplicitOwners,{model});
    });
    auto program_negative=[&](const char* label,auto mutate){rejects(label,[&]{auto p=model->metadata().at("programs").object();mutate(p);product::admit_program_declaration(j(p),declaration);});};
    program_negative("selection unknown instance",[](auto& p){auto e=p.at("execution").object();auto ids=e.at("instance_ids").array();ids[0]=j(int64_t(99));e["instance_ids"]=j(ids);p["execution"]=j(e);});
    program_negative("selection length mismatch",[](auto& p){auto e=p.at("execution").object();auto ids=e.at("instance_ids").array();ids.pop_back();e["instance_ids"]=j(ids);p["execution"]=j(e);});
    program_negative("guidance step count mismatch",[](auto& p){auto s=p.at("sampling").object();auto rows=s.at("transitions").array();rows.pop_back();s["transitions"]=j(rows);p["sampling"]=j(s);});
    program_negative("nonfinite guidance",[](auto& p){auto s=p.at("sampling").object();auto rows=s.at("transitions").array();auto t=rows[0].object();t["guidance_scale"]=j(std::numeric_limits<double>::infinity());rows[0]=j(t);s["transitions"]=j(rows);p["sampling"]=j(s);});
    rejects("reader lacks step guidance capability",[&]{product::admit_program_declaration(model->metadata().at("programs"),declaration,{"execution.per_step.v1"});});
    rejects("source identity corruption",[&]{provenance(model->metadata(),std::string(64,'0'));});
    rejects("provenance corruption",[&]{auto m=model->metadata().object();auto p=m.at("conversion_provenance").object();p["actual_revision"]=j(std::string(40,'0'));m["conversion_provenance"]=j(p);provenance(j(m),manifest);});
    for(const auto name:{"out-of-range.vrm","overlap.vrm"})rejects(name,[&]{VrmModel invalid((fs::path(argv[2])/name).string(),false);});
    auto text=std::make_shared<product::SafeTensorAsset>(product::SafeTensorAsset::single(root/"conditioning.safetensors"));
    auto weights=text->weights();auto conditioning=read(root/"conditioning.json");std::set<std::string> references;weight_references(conditioning,weights,references);
    require(references.size()==242 && conditioning.at("blocks").array().size()==24 && weights.at("token_embedding.weight").dim(1)==4096,"UMT5 wiring contract");
    { ConditioningComponentExecutor endpoint(*backend,weights); (void)endpoint; }
    for(const auto& block:conditioning.at("blocks").array()){
        const auto& a=block.at("attention");
        require(a.at("query_heads").integer()==64 && a.at("kv_heads").integer()==64 &&
            conditioning_mask_semantic(a)==MaskSemantic::AdditiveFinite,"Conditioning attention contract");
        require(weights.at(a.at("q").at("weight").string()).shape()==std::vector<int64_t>({4096,4096}),"Conditioning projection interface");
    }
    std::set<std::string> decoder_names;
    for(const auto& [slot,name]:model->metadata().at("shared_components").at("decoder").at("runtime_tensor_bindings").object()){(void)slot;require(model->tensor(name.string()).dtype()==DType::F32,"VAE storage");decoder_names.insert(name.string());}
    require(decoder_names.size()==194,"VAE wiring contract");
    const auto& decoder=model->metadata().at("shared_components").at("decoder");
    require(decoder.at("implementation_id").string()=="dit_flow.wan.vae_decoder.v1" &&
        decoder.at("latent_contract").at("channels").integer()==16,"Decoder implementation/latent interface");
    { ComponentExecutor endpoint(*backend,WeightMap(model->bindings(decoder)));(void)endpoint; }
    for(const auto& slot:model->metadata().at("shared_components").at("decoder_slots").array()){
        std::vector<int64_t> shape;for(const auto& n:slot.at("shape").array())shape.push_back(n.integer());
        validate_architecture_tensor(model->tensor(slot.at("name").string()),shape,DType::F32);
    }
    auto tokenizer=read(root/"tokenizer/tokenizer.json");require(tokenizer.at("model").at("type").string()=="Unigram","Tokenizer declaration");
    auto index=read(root/"conditioning-index.json");require(index.at("weight_map").object().size()==242,"UMT5 index");
    for(const auto& [slot,path]:index.at("weight_map").object()){require(path.string()=="conditioning.safetensors","Component resource path");(void)weights.at(slot);}
    ResourceEstimate estimate;estimate.source_backing=model->file_size()+text->mapped_bytes();
    // This request describes HOST-ONLY preflight, not future inference residency.
    estimate.host_staging=positive.bytes()+negative.bytes();
    // Generic admission requires positive configured limits even when the
    // declared device usage is zero. A one-byte envelope admits no real tensor.
    ResourceAdmissionRequest request{estimate,{1,estimate.host_peak()}};backend->set_resource_admission(request);backend->admit_execution_resources();
    rejects("insufficient declared budget",[&]{auto x=request;--x.budget.estimated_host_peak_limit;x.validate();});
    std::weak_ptr<product::SafeTensorAsset> weak_text=text;
    for(const auto& instance:context->instances())backend->retain_resource_owners(instance.resource_owners());
    backend->retain_resource_owners({text});require(backend->resource_owner_count()==2,"Session owners deduplicate");
    const size_t rss_ready=rss();
    auto stats=context->prepared_tensor_stats();require(stats.host_materializations==0 && stats.host_to_device_bytes==0,"Unexpected prepared weights");
    // Drop caller and catalog owners first. Endpoints and then session leases
    // must independently preserve borrowed backing until their release.
    model.reset();text.reset();admitted.reset();require(!weak.expired() && !weak_text.expired(),"Premature backing release");
    context.reset();require(!weak.expired() && !weak_text.expired(),"Session lease released too early");
    require(backend->calls.empty() && backend->conditioning_calls==4,"Numerical execution detected");
    backend.reset();require(weak.expired() && weak_text.expired(),"Session backing leak");
    Json result=j(Json::Object{{"status",j(std::string("PASS"))},{"graph_count",j(int64_t(1))},{"bindings",j(int64_t(2))},{"slots_per_binding",j(int64_t(1095))},{"bound_parameter_bytes",j(int64_t(bound_bytes))},{"host_mapped_backing_bytes",j(int64_t(estimate.source_backing))},{"declared_device_peak",j(int64_t(estimate.device_peak()))},{"rss_before",j(int64_t(before))},{"rss_after_load",j(int64_t(loaded))},{"rss_ready",j(int64_t(rss_ready))},{"negative_tests",j(int64_t(rejects_count))},{"numerical_operations",j(int64_t(0))},{"gpu_upload_bytes",j(int64_t(0))},{"schedule",j(table)}});
    std::cout<<"RESULT_JSON="<<result.serialize()<<'\n';
    return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}}
