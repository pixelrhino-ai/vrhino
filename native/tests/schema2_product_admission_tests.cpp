#include "schema2_product_test_fixture.h"
#include "vrhino/product/run_request.h"
struct NoCompute:neural_graph::test::TinyBackend{
    Tensor copy_to_device(const Tensor& t,DType d)override{require(t.dtype()==d,"Unexpected cast");return t;}
};
int main(int argc,char**argv){try{
    require(argc==3,"usage: schema2-tests OUTPUT SPECS");Fixture f(argv[1],argv[2]);
    auto admitted=f.preflight();require(admitted.evidence.at("structurally_runnable").boolean()&&!admitted.evidence.at("numerically_qualified").boolean(),"Qualification status leak");
    NoCompute backend;TensorBundle input{{"seed",scalar_i64(0)},{"positive",Tensor::host({1,2,8},DType::F32)},{"negative",Tensor::host({1,2,8},DType::F32)}};
    auto setup=admitted.architecture->create_execution_setup(backend,PrecisionPolicy::fp32(),input);
    auto sampling=admitted.architecture->create_program(input);
    const auto selected=setup.program.admit(setup.context,sampling,DType::F32);
    require(selected.size()==3 && selected[0]==selected[2] && selected[0]!=selected[1],"Lost selection");
    require(selected[0]->graph()==selected[1]->graph(),"Duplicated topology");
    require(backend.calls.empty(),"Preflight executed numerics");
    std::weak_ptr<const VrmModel> weak=admitted.model;admitted.model.reset();require(!weak.expired(),"Lost mmap ownership");
    reject("singleton fallback",[&]{admitted.architecture->create_denoiser(backend,PrecisionPolicy::fp32(),input);});
    input["guidance_scale"]=scalar_f32(1);reject("legacy guidance override",[&]{admitted.architecture->create_program(input);});
    std::cout<<"PASS normal product preflight; managed factory; shared graph; distinct bindings; A/B/A intent; no numerical calls\n";
    require(selected[0]->parameters().front().data()!=selected[1]->parameters().front().data(), "Binding backing alias");
    require(sampling.guidance_schedule->at(0).scale == 4 && sampling.guidance_schedule->at(1).scale == 3,
            "Guidance schedule collapsed");
    reject("structural admission cannot grant numerical execution", [&]{p::require_numerical_product_admission(admitted.resources.manifest);});
    const auto base_graph=f.graph,base_metadata=f.metadata,base_manifest=f.manifest,base_local=f.local;
    // A public Product profile references the existing canonical program. It
    // never fabricates constant guidance or numerical execution permission.
    auto product=read(fs::path(argv[2])/"wan2_1_t2v_1_3b/successors/1.0.1/vrhino-model.json").at("product");
    auto frozen=product.at("frozen_profile");
    frozen=set(frozen,"sampling",Json::parse(R"({"program_artifact":"programs"})"));
    product=set(product,"frozen_profile",frozen);
    f.manifest=set(base_manifest,"product",product);f.save();
    auto with_product=f.preflight();
    require(with_product.resources.manifest.product.frozen_profile->sampling->program_artifact=="programs",
            "Lost Product program reference");
    require(with_product.sampling.guidance_schedule->at(0).scale==4 &&
            with_product.sampling.guidance_schedule->at(1).scale==3,
            "Product profile collapsed per-step guidance");
    auto normal_request=p::parse_product_run_document(Json::parse(
        R"({"model":"test/catalog:1","inputs":{"prompt":"hello"},"parameters":{"seed":19},"resources":{"weight_cache_budget_bytes":4096}})"));
    auto normal_options=p::map_product_run_options(with_product.resources,normal_request);
    require(normal_options.prompt=="hello" && normal_options.seed==19 &&
            normal_options.resources.weight_cache_budget_bytes==4096,
            "Canonical Product request did not map for program-backed package");
    reject("program profile does not grant numerical admission",[&]{p::require_numerical_product_admission(with_product.resources.manifest);});
    for(const auto* id:{"absent","graph"}){
        auto bad=set(product,"frozen_profile",set(frozen,"sampling",j(Json::Object{{"program_artifact",j(std::string(id))}})));
        f.manifest=set(base_manifest,"product",bad);f.save();
        reject("unadmitted Product program reference",[&]{f.preflight();});
    }
    auto schema1=read(fs::path(argv[2])/"wan2_1_t2v_1_3b/successors/1.0.1/vrhino-model.json");
    schema1=set(schema1,"product",product);write(f.root/"invalid-schema1-program.json",schema1);
    reject("schema1 cannot acquire schema2 program semantics",[&]{p::load_model_package_manifest(f.root/"invalid-schema1-program.json");});
    f.manifest=base_manifest;f.save();
    auto graph_negative=[&](const char*name,std::function<Json(Json)> mutate){f.graph=mutate(base_graph);reject(name,[&]{f.emit();f.preflight();});f.graph=base_graph;f.metadata=base_metadata;f.manifest=base_manifest;f.emit();};
    graph_negative("missing binding",[](Json g){auto a=g.at("bindings").array();a.pop_back();return set(g,"bindings",j(a));});
    graph_negative("duplicate component ID",[](Json g){auto a=g.at("instances").array();a.push_back(a[0]);return set(g,"instances",j(a));});
    graph_negative("unknown component binding",[](Json g){auto a=g.at("instances").array();a[0]=set(a[0],"binding",j(int64_t(99)));return set(g,"instances",j(a));});
    graph_negative("missing canonical slot",[](Json g){auto a=g.at("bindings").array();auto p=a[0].at("parameters").object();p.erase(p.begin());a[0]=set(a[0],"parameters",j(p));return set(g,"bindings",j(a));});
    graph_negative("wrong slot shape",[](Json g){auto a=g.at("bindings").array();auto p=a[0].at("parameters").object();p[p.begin()->first]=j(std::string("decoder.mock"));a[0]=set(a[0],"parameters",j(p));return set(g,"bindings",j(a));});
    for(int mode=0;mode<4;++mode){auto programs=base_metadata.at("programs");
        if(mode==0)programs=set(programs,"execution",Json::parse(R"({"kind":"per_step","instance_ids":[0,99,0]})"));
        if(mode==1)programs=set(programs,"execution",Json::parse(R"({"kind":"per_step","instance_ids":[0]})"));
        if(mode==2)programs=set(programs,"sampling",set(programs.at("sampling"),"solver",j(std::string("unknown"))));
        if(mode==3)programs=set(programs,"required_capabilities",j(Json::Array{j(std::string("unsupported"))}));
        f.metadata=set(base_metadata,"programs",programs);reject("invalid execution/sampling "+std::to_string(mode),[&]{f.emit();f.preflight();});f.metadata=base_metadata;f.emit();
    }
    for(int mode=0;mode<6;++mode){f.manifest=base_manifest;f.local=base_local;
        if(mode==0)f.manifest=set(f.manifest,"schema_version",j(int64_t(99)));
        if(mode==1)f.manifest=set(f.manifest,"admission",set(f.manifest.at("admission"),"required_capabilities",j(Json::Array{j(std::string("unknown"))})));
        if(mode==2)f.local=set(f.local,"resources",set(f.local.at("resources"),"runtime",j(std::string("absent.vrm"))));
        if(mode==3)f.local=set(f.local,"extra",j(true));
        if(mode==4)f.manifest=set(f.manifest,"admission",set(f.manifest.at("admission"),"structural_only",j(false)));
        if(mode==5)f.manifest=set(f.manifest,"admission",set(f.manifest.at("admission"),"programs_artifact",j(std::string("absent"))));
        f.save();reject("invalid product/local "+std::to_string(mode),[&]{f.preflight();});
    }
    f.manifest=base_manifest;f.local=base_local;f.emit();
    write(f.root/"graph.json",set(base_graph,"schema_version",j(int64_t(1))));f.refresh();reject("graph/package mismatch",[&]{f.preflight();});f.emit();
    auto arts=f.manifest.at("artifacts").array();arts[0]=set(arts[0],"sha256",j(std::string(64,'0')));f.manifest=set(f.manifest,"artifacts",j(arts));f.save();reject("wrong SHA256",[&]{f.preflight();});f.manifest=base_manifest;f.emit();
    for(const auto* family:{"wan2_1_t2v_1_3b","ltx_v0_9_1","mochi_1_preview"}){
        auto path=fs::path(argv[2])/family/"vrhino-model.json";auto old=p::load_model_package_manifest(path);require(old.schema_version==1,"Schema1 changed");p::require_numerical_product_admission(old);std::cout<<"PASS schema1 manifest "<<family<<'\n';
    }
    // Exercise the unchanged caller-retained schema1 family factory using the
    // same small canonical graph and tensor bytes (no numerical execution).
    auto descriptor=Json::parse(R"({"schema_version":1,"canonical_architecture":{},"runtime_tensor_bindings":{}})");
    descriptor=set(descriptor,"canonical_architecture",base_graph.at("graphs").array()[0].at("declaration"));
    descriptor=set(descriptor,"runtime_tensor_bindings",base_graph.at("bindings").array()[0].at("parameters"));
    auto legacy_graph=j(Json::Object{{"schema_version",j(int64_t(1))},{"architecture_graph",descriptor},
        {"component_graphs",j(Json::Array{base_metadata.at("shared_components").at("decoder")})}});
    const auto legacy_path=f.root/"legacy.vrm";fs::remove(legacy_path);
    p::write_vrm_streaming(legacy_path,"dit-flow","wan",base_metadata,legacy_graph,f.source,f.mappings);
    VrmModel legacy_model(legacy_path.string());auto legacy_architecture=create_architecture(legacy_model);
    TensorBundle legacy_request{{"seed",scalar_i64(0)}};
    require(legacy_architecture->create_program(legacy_request).steps==3,"Legacy program behavior changed");
    p::LocalModelCache cache(f.root/"cache");cache.install(f.root);
    auto cached=cache.resolve("test/catalog:1",true);require(cached.manifest.schema_version==2,"Cache lost schema2");
    auto cached_admission=p::preflight_resolved_product(cached);
    require(cached_admission.evidence.at("vrm_sha256").string()==admitted.evidence.at("vrm_sha256").string() &&
        cached_admission.evidence.at("bindings").integer()==2 &&
        !cached_admission.evidence.at("numerically_qualified").boolean(),"Cache/local admission drift");
    reject("missing resolved binding resource",[&]{auto bad=cached;bad.artifacts.erase("runtime");p::preflight_resolved_product(std::move(bad));});
    reject("extra resolved resource",[&]{auto bad=cached;bad.artifacts.emplace("extra",bad.artifacts.at("runtime"));p::preflight_resolved_product(std::move(bad));});
    reject("resolved declaration identity",[&]{auto bad=cached;bad.artifacts.at("runtime").declaration.sha256=std::string(64,'0');p::preflight_resolved_product(std::move(bad));});
    reject("resolved runtime pointer mismatch",[&]{auto bad=cached;bad.runtime_model_path=bad.artifacts.at("tokenizer").path;p::preflight_resolved_product(std::move(bad));});
    reject("stale manifest resolution",[&]{auto bad=cached;bad.manifest.raw_json+=' ';p::preflight_resolved_product(std::move(bad));});
    const auto corrupted=f.root/"same-size-corrupt-tokenizer.json";
    {std::ifstream in(cached.artifacts.at("tokenizer").path,std::ios::binary);std::string bytes((std::istreambuf_iterator<char>(in)),{});
     require(!bytes.empty(),"Empty fixture");bytes[0]^=1;std::ofstream(corrupted,std::ios::binary)<<bytes;}
    reject("resolved same-size content drift",[&]{auto bad=cached;bad.artifacts.at("tokenizer").path=corrupted;p::preflight_resolved_product(std::move(bad));});
    reject("cached schema2 cannot execute",[&]{p::require_numerical_product_admission(cached.manifest);});
    std::cout<<"PASS schema1 factory and schema2 cache publication/resolve\n";
    std::cout<<"PASS negative cases="<<rejected<<"; structural fixture only, no text/decode qualification\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
