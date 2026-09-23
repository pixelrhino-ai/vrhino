#include "schema2_product_test_fixture.h"
#include "vrhino/product/request_wiring.h"
#include "vrhino/product/component_preparation.h"
#include "vrhino/product/declared_run.h"
#include "vrhino/product/run_request.h"
#include "neural_graph_test_backend.h"
#include <algorithm>

void extra(Fixture& f,const char* id,const char* file) {
    auto a=f.manifest.at("artifacts").array();a.push_back(j(Json::Object{{"id",j(std::string(id))},{"path",j(std::string(file))},
        {"role",j(std::string("product.request"))},{"required",j(true)},{"size",j(int64_t(fs::file_size(f.root/file)))},
        {"sha256",j(p::sha256_file(f.root/file))}}));f.manifest=set(f.manifest,"artifacts",j(a));
    f.local=set(f.local,"resources",set(f.local.at("resources"),id,j(std::string(file))));
}
void prepare(Fixture& f) {
    auto shared=f.metadata.at("shared_components");auto d=shared.at("decoder");
    auto latent=set(set(d.at("latent_contract"),"spatial_scale",j(int64_t(8))),"temporal_decode",j(std::string("1+4*(F-1)")));
    f.metadata=set(f.metadata,"shared_components",set(shared,"decoder",set(d,"latent_contract",latent)));
    write(f.root/"conditioning.json",Json::parse(R"({"schema_version":1,"kind":"pre_norm_transformer","output_trim_to_mask":true,"embedding":{"weight":"embedding"},"final_norm":{"weight":"norm","kind":"rms_norm","eps":0.000001},"blocks":[]})"));
    write(f.root/"conditioning-index.json",Json::parse(R"({"weight_map":{"embedding":"conditioning.safetensors","norm":"conditioning.safetensors"}})"));
    std::string h=R"({"embedding":{"dtype":"F32","shape":[6,8],"data_offsets":[0,192]},"norm":{"dtype":"F32","shape":[8],"data_offsets":[192,224]}})";while(h.size()%8)h+=' ';
    {std::ofstream out(f.root/"conditioning.safetensors",std::ios::binary);uint64_t n=h.size();out.write(reinterpret_cast<const char*>(&n),8);out<<h;std::vector<float> zero(56);out.write(reinterpret_cast<const char*>(zero.data()),224);}
    write(f.root/"tokenizer.json",Json::parse(R"({"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":0,"content":"<pad>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":1,"content":"</s>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<pad>":0,"</s>":1,"[UNK]":2,"hello":3,"bad":4,"猫":5},"unk_token":"[UNK]"}})"));
    std::ofstream(f.root/"tokenizer-config.json")<<R"({"pad_token":"<pad>","eos_token":"</s>","model_max_length":1000000000000000019884624838656})";
    write(f.root/"wiring.json",Json::parse(R"({"schema":"vrhino.text-product-wiring.v1","tokenizer_spec":{"format":"huggingface_json","max_length":4,"pad_id":0,"suffix_ids":[1]},"tokenizer_config_artifact":"tokenizer-config","geometry":{"spatial_scale":8,"temporal_scale":4,"temporal_origin":1},"conditioning_bindings":[{"component_id":"positive","text_source":"prompt","target":"positive"},{"component_id":"negative","text_source":"negative_prompt","target":"negative"}]})"));
    Json::Array components;for(const auto* id:{"positive","negative"})components.push_back(j(Json::Object{{"id",j(std::string(id))},{"kind",j(std::string("conditioning.text_encoder"))},{"artifacts",Json::parse(R"(["conditioning","index","weights","tokenizer"])" )}}));
    f.manifest=set(f.manifest,"entrypoint",set(f.manifest.at("entrypoint"),"components",j(components)));
    f.emit();extra(f,"tokenizer-config","tokenizer-config.json");extra(f,"wiring","wiring.json");
    f.manifest=set(f.manifest,"admission",set(f.manifest.at("admission"),"request_artifact",j(std::string("wiring"))));f.save();
}
int main(int argc,char** argv){try{
    const std::string oversized="1000000000000000019884624838656";
    reject("strict integer overflow",[&]{Json::parse(oversized);});
    JsonParseLimits metadata_limits;metadata_limits.oversized_integer_as_float=true;
    auto sentinel=Json::parse(oversized,metadata_limits);
    require(sentinel.is_number() && !sentinel.is_int(),"Metadata sentinel type");
    reject("metadata float cannot become integer",[&]{sentinel.integer();});
    reject("nonfinite metadata sentinel",[&]{Json::parse(std::string(400,'9'),metadata_limits);});
    require(argc==3,"usage: schema2-request-tests OUTPUT SPECS");Fixture f(argv[1],argv[2]);prepare(f);
    auto product=std::make_shared<p::AdmittedLocalProduct>(f.preflight());
    auto request=Json::parse(R"({"schema":"vrhino.product-text-request.v1","prompt":"hello 猫","negative_prompt":"bad","seed":19,"width":32,"height":32,"frames":5})");
    auto prepared=p::prepare_text_product_request(product,request);
    require(prepared.runtime_inputs.size()==2 && !prepared.runtime_inputs.contains("positive"),"Dry-run manufactured conditioning");
    require(prepared.sampling.latent_shape==std::vector<int64_t>({1,16,2,4,4}) && prepared.sampling.seed==19 && prepared.sampling.steps==3,"Request lowering lost shape/seed/program");
    require(prepared.conditioning[0].input_ids.data_as<int64_t>()[0]==3 && prepared.conditioning[0].input_ids.data_as<int64_t>()[1]==5 && prepared.conditioning[0].input_ids.data_as<int64_t>()[2]==1,"Native tokenizer exact IDs");
    require(prepared.conditioning[0].attention_mask.data_as<uint8_t>()[3]==0,"Tokenizer padding mask");
    require(prepared.conditioning[0].expected_hidden_shape==std::vector<int64_t>({1,3,8}),"Trimmed output interface");
    auto empty=p::prepare_text_product_request(product,set(request,"negative_prompt",j(std::string(""))));
    require(empty.conditioning[1].expected_hidden_shape[1]==1 && empty.conditioning[1].input_ids.data_as<int64_t>()[0]==1,"Empty negative prompt EOS");
    auto long_input=p::prepare_text_product_request(product,set(request,"prompt",j(std::string("hello hello hello hello hello"))));
    require(long_input.conditioning[0].input_ids.data_as<int64_t>()[3]==1,"Truncation must retain EOS");
    for(const auto* key:{"steps","cfg","binding","precision"})reject(std::string("legacy semantic override ")+key,[&]{p::prepare_text_product_request(product,set(request,key,j(int64_t(1))));});
    reject("invalid width",[&]{p::prepare_text_product_request(product,set(request,"width",j(int64_t(31))));});
    reject("invalid frames",[&]{p::prepare_text_product_request(product,set(request,"frames",j(int64_t(2))));});
    reject("negative seed",[&]{p::prepare_text_product_request(product,set(request,"seed",j(int64_t(-1))));});
    reject("invalid utf8",[&]{p::prepare_text_product_request(product,set(request,"prompt",j(std::string("\xff"))));});
    reject("missing result",[&]{prepared.bind_conditioning_outputs({});});
    reject("wrong result shape",[&]{prepared.bind_conditioning_outputs({Tensor::host({1},DType::F32),Tensor::host({1},DType::F32)});});
    std::vector<Tensor> fake;for(const auto& c:prepared.conditioning){
        fake.push_back(Tensor::host(c.expected_hidden_shape,DType::F32));
        std::fill_n(fake.back().data_as<float>(),fake.back().numel(),0.0f);
    }
    auto inputs=prepared.bind_conditioning_outputs(fake);require(inputs.contains("positive")&&inputs.contains("negative"),"Result wiring failed");
    // Shape-only outputs are test fixtures, never emitted by product dry-run.
    struct HostOnly:neural_graph::test::TinyBackend {Tensor copy_to_device(const Tensor&t,DType d)override{require(t.dtype()==d,"Unexpected conversion");return t;}} backend;
    auto setup=product->architecture->create_execution_setup(backend,PrecisionPolicy::fp32(),inputs);
    auto selected=prepared.execution.admit(setup.context,prepared.sampling,DType::F32);
    require(selected.size()==3 && selected[0]==selected[2] && selected[0]!=selected[1] && backend.calls.empty(),"Execution intent changed");
    require(prepared.sampling.guidance_schedule->at(0).scale==4 && prepared.sampling.guidance_schedule->at(1).scale==3,"Guidance overwritten");
    const auto admission=p::admit_conditioned_sampling(backend,PrecisionPolicy::fp32(),prepared,inputs);
    require(admission.at("admitted").boolean() && !admission.at("denoiser_evaluated").boolean(),"Conditioned admission executed denoiser");
    auto changed_seed=inputs;changed_seed["seed"]=scalar_i64(20);
    reject("conditioned request seed drift",[&]{p::admit_conditioned_sampling(backend,PrecisionPolicy::fp32(),prepared,changed_seed);});
    auto added_cfg=inputs;added_cfg.emplace("guidance",scalar_f32(9.0f));
    reject("conditioned request injected guidance",[&]{p::admit_conditioned_sampling(backend,PrecisionPolicy::fp32(),prepared,added_cfg);});
    std::weak_ptr<p::AdmittedLocalProduct> weak=product;product.reset();require(!weak.expired(),"Request lost owners");
    require(prepared.conditioning_weights().at("embedding").shape()==std::vector<int64_t>({6,8}) && prepared.decoder_weights().contains("mock"),"Resource owner wiring");
    reject("numerical qualification must remain HOLD",[&]{p::require_numerical_product_admission(prepared.owner->resources.manifest);});
    const auto base_wiring=read(f.root/"wiring.json");const auto base_manifest=f.manifest;const auto base_local=f.local;
    auto declaration=read(fs::path(argv[2])/"wan2_1_t2v_1_3b/successors/1.0.1/vrhino-model.json").at("product");
    auto frozen=declaration.at("frozen_profile");
    frozen=set(frozen,"output",Json::parse(R"({"width":32,"height":32,"frames":5,"fps":{"numerator":16,"denominator":1},"duration":"fixed","audio":"none"})"));
    frozen=set(frozen,"sampling",Json::parse(R"({"program_artifact":"programs"})"));
    f.manifest=set(base_manifest,"product",set(declaration,"frozen_profile",frozen));f.save();
    auto declared=std::make_shared<p::AdmittedLocalProduct>(f.preflight());
    auto from_declared=p::prepare_text_product_request(declared,request);
    require(from_declared.evidence.serialize()==prepared.evidence.serialize(),
            "Product declaration changed prepared tokens or program intent");
    for(const auto* dimension:{"width","height","frames"})
        reject("frozen Product geometry cannot be overridden",[&]{p::prepare_text_product_request(declared,
            set(request,dimension,j(request.at(dimension).integer()+8)));});
    reject("declared Product prompt minimum",[&]{p::prepare_text_product_request(declared,set(request,"prompt",j(std::string(""))));});
    reject("declared Product still HOLD",[&]{p::require_numerical_product_admission(declared->resources.manifest);});
    // Ordinary Product envelope lowers to the same tokenizer/program request;
    // numerical admission remains closed, even when declarations are complete.
    const auto run_plan=Json::parse(R"({"schema":"vrhino.product-text-run.v1","negative_prompt":"bad","precision_artifact":"policy","memory":{"device_budget_bytes":1048576,"host_budget_bytes":1048576,"workspace_bytes":1024,"safety_margin_bytes":1024,"weight_cache_budget_bytes":65536},"media_range":[-1,1]})");
    write(f.root/"run.json",run_plan);
    write(f.root/"policy.json",read(fs::path(argv[2]).parent_path()/"tests/fixtures/precision/bf16-semantic-scalar-v2.json"));
    extra(f,"run","run.json");extra(f,"policy","policy.json");
    auto artifacts=f.manifest.at("artifacts").array();
    for(auto& a:artifacts) {
        if(a.at("id").string()=="run")a=set(a,"role",j(std::string("product.execution")));
        if(a.at("id").string()=="policy")a=set(a,"role",j(std::string("precision.policy")));
    }
    f.manifest=set(f.manifest,"artifacts",j(artifacts));
    f.manifest=set(f.manifest,"product",set(f.manifest.at("product"),"execution_artifact",j(std::string("run"))));f.save();
    auto run_product=std::make_shared<p::AdmittedLocalProduct>(f.preflight());
    auto envelope=Json::parse(R"({"model":"test/catalog:1","inputs":{"prompt":"hello 猫"},"parameters":{"seed":19},"resources":{"weight_cache_budget_bytes":32768}})");
    auto options=p::map_product_run_options(run_product->resources,p::parse_product_run_document(envelope));
    auto lowered=p::lower_declared_text_run(run_product->resources,options);
    require(lowered.request.serialize()==request.serialize(),"Ordinary request lost defaults or frozen geometry");
    require(lowered.memory.weight_cache_budget_bytes==32768 && lowered.memory.reserved_device_workspace_bytes==1024 &&
            lowered.fps==16 && lowered.video_minimum==-1 && lowered.video_maximum==1,"Run resource/media lowering changed semantics");
    write(f.root/"normal-request.json",envelope);
    auto normal=p::dry_run_resolved_product(run_product->resources,f.root/"normal-request.json");
    require(normal.evidence.serialize()==prepared.evidence.serialize(),"Normal request changed prepared execution intent");
    auto defaults=options;defaults.seed.reset();defaults.resources={};
    require(p::lower_declared_text_run(run_product->resources,defaults).request.at("seed").integer()==5701,
            "Product seed default lost");
    auto wide=options;wide.seed=UINT64_MAX;
    auto wide_request=p::lower_declared_text_run(run_product->resources,wide).request;
    auto wide_prepared=p::prepare_text_product_request(run_product,wide_request);
    require(wide_request.at("seed").string()=="18446744073709551615" &&
            wide_prepared.sampling.seed==UINT64_MAX && wide_prepared.runtime_inputs.at("seed").data_as<int64_t>()[0]==-1,
            "Full uint64 seed did not survive Product/Architecture lowering");
    for(const auto* seed:{"", "01", "-1", "+1", "18446744073709551616", "1.0"})
        reject("invalid canonical seed",[&]{p::prepare_text_product_request(run_product,set(request,"seed",j(std::string(seed))));});
    for(int mode=0;mode<6;++mode) {
        auto bad=options;
        if(mode==0)bad.model_reference="test/other:1";
        if(mode==1)bad.preset="other";
        if(mode==2)bad.prompt="";
        if(mode==3)bad.video="video.mp4";
        if(mode==4)bad.audio="audio.wav";
        if(mode==5)bad.resources.weight_cache_budget_bytes=65537;
        reject("invalid declared Product options",[&]{p::lower_declared_text_run(run_product->resources,bad);});
    }
    const auto run_resources=run_product->resources;
    const auto invalid_plan=[&](Json plan) {
        write(f.root/"run.json",plan);auto resources=run_resources;
        auto& a=resources.artifacts.at("run").declaration;
        a.size=fs::file_size(f.root/"run.json");a.sha256=p::sha256_file(f.root/"run.json");
        reject("invalid immutable run declaration",[&]{p::lower_declared_text_run(resources,options);});
        write(f.root/"run.json",run_plan);
    };
    for(const auto* field:{"cfg","steps","binding","precision_override"})invalid_plan(set(run_plan,field,j(int64_t(1))));
    invalid_plan(set(run_plan,"schema",j(std::string("unknown"))));
    invalid_plan(set(run_plan,"precision_artifact",j(std::string("missing"))));
    invalid_plan(set(run_plan,"media_range",Json::parse("[1,-1]")));
    invalid_plan(set(run_plan,"media_range",Json::parse("[0,1e100]")));
    invalid_plan(set(run_plan,"memory",set(run_plan.at("memory"),"weight_cache_budget_bytes",j(int64_t(0)))));
    invalid_plan(set(run_plan,"memory",set(run_plan.at("memory"),"workspace_bytes",j(int64_t(-1)))));
    invalid_plan(set(run_plan,"memory",set(run_plan.at("memory"),"workspace_bytes",j(int64_t(1048577)))));
    for(int mode=0;mode<8;++mode) {
        auto bad=run_resources;
        if(mode==0)bad.artifacts.erase("run");
        if(mode==1)bad.artifacts.at("run").declaration.required=false;
        if(mode==2)bad.artifacts.at("run").declaration.role="other";
        if(mode==3)bad.artifacts.at("run").declaration.size=65537;
        if(mode==4)bad.artifacts.at("run").declaration.sha256=std::string(64,'0');
        if(mode==5)bad.artifacts.at("policy").declaration.required=false;
        if(mode==6)bad.artifacts.at("policy").declaration.role="other";
        if(mode==7)bad.artifacts.at("policy").declaration.sha256=std::string(64,'0');
        reject("unqualified run resource",[&]{p::lower_declared_text_run(bad,options);});
    }
    std::ofstream(f.root/"run.json",std::ios::app)<<' ';
    reject("run resource byte drift",[&]{p::lower_declared_text_run(run_resources,options);});write(f.root/"run.json",run_plan);
    auto invalid_policy=run_resources;write(f.root/"policy.json",Json::parse(R"({"schema":"unknown"})"));
    invalid_policy.artifacts.at("policy").declaration.size=fs::file_size(f.root/"policy.json");
    invalid_policy.artifacts.at("policy").declaration.sha256=p::sha256_file(f.root/"policy.json");
    reject("malformed precision declaration during dry lowering",[&]{p::lower_declared_text_run(invalid_policy,options);});
    write(f.root/"policy.json",read(fs::path(argv[2]).parent_path()/"tests/fixtures/precision/bf16-semantic-scalar-v2.json"));
    write(f.root/"normal-request.json",set(envelope,"model",j(std::string("test/other:1"))));
    reject("normal request/package mismatch",[&]{p::dry_run_resolved_product(run_resources,f.root/"normal-request.json");});
    reject("complete declaration cannot grant numerical qualification",[&]{p::require_numerical_product_admission(run_resources.manifest);});
    reject("complete declaration does not implicitly grant execution",[&]{p::require_product_execution_admission(run_resources.manifest);});
    f.manifest=set(f.manifest,"admission",set(f.manifest.at("admission"),
        "execution_eligibility",j(std::string("alpha_unqualified"))));f.save();
    const auto alpha_product=f.preflight();
    p::require_product_execution_admission(alpha_product.resources.manifest);
    require(p::product_execution_is_alpha_unqualified(alpha_product.resources.manifest),
            "Explicit alpha execution declaration was lost");
    require(alpha_product.evidence.at("product_execution_eligible").boolean() &&
            alpha_product.evidence.at("product_execution_status").string()==
                "alpha_unqualified",
            "Preflight did not distinguish alpha execution from qualification");
    reject("alpha execution must not grant numerical qualification",[&]{
        p::require_numerical_product_admission(alpha_product.resources.manifest);});
    f.manifest=base_manifest;f.local=base_local;f.save();
    for(int mode=0;mode<5;++mode){
        auto w=base_wiring;
        if(mode==0)w=set(w,"tokenizer_spec",set(w.at("tokenizer_spec"),"max_length",j(int64_t(3))));
        if(mode==1)w=set(w,"tokenizer_spec",set(w.at("tokenizer_spec"),"pad_id",j(int64_t(2))));
        if(mode==2)w=set(w,"geometry",set(w.at("geometry"),"spatial_scale",j(int64_t(16))));
        if(mode>=3){auto bindings=w.at("conditioning_bindings").array();
            if(mode==3)bindings[1]=bindings[0];
            else bindings[0]=set(bindings[0],"component_id",j(std::string("absent")));
            w=set(w,"conditioning_bindings",j(bindings));
        }
        write(f.root/"wiring.json",w);auto artifacts=base_manifest.at("artifacts").array();
        for(auto& a:artifacts)if(a.at("id").string()=="wiring"){
            a=set(set(a,"size",j(int64_t(fs::file_size(f.root/"wiring.json")))),"sha256",j(p::sha256_file(f.root/"wiring.json")));
        }
        f.manifest=set(base_manifest,"artifacts",j(artifacts));f.save();
        auto bad=std::make_shared<p::AdmittedLocalProduct>(f.preflight());
        reject("invalid wiring "+std::to_string(mode),[&]{p::prepare_text_product_request(bad,request);});
    }
    write(f.root/"wiring.json",base_wiring);f.manifest=base_manifest;f.save();
    std::weak_ptr<p::AdmittedLocalProduct> released;
    {auto local=std::make_shared<p::AdmittedLocalProduct>(f.preflight());released=local;
     auto held=p::prepare_text_product_request(local,request);local.reset();require(!released.expired(),"Early resource release");}
    require(released.expired(),"Prepared request leaked owners");
    write(f.root/"request.json",request);
    p::LocalModelCache cache(f.root/"cache");cache.install(f.root);
    auto cached=p::dry_run_resolved_product(cache.resolve("test/catalog:1",false),f.root/"request.json");
    require(cached.evidence.serialize()==prepared.evidence.serialize(),"Cached request changed tokens or programs");
    require(cached.owner->model && cached.owner->architecture && cached.conditioning_weights().contains("embedding"),
            "Cached request lost owned resources");
    reject("cached request is not numerical authorization",[&]{p::require_numerical_product_admission(cached.owner->resources.manifest);});
    std::ofstream(f.root/"wiring.json",std::ios::app)<<' ';reject("request declaration drift",[&]{p::prepare_text_product_request(prepared.owner,request);});
    write(f.root/"wiring.json",base_wiring);
    std::cout<<"PASS native tokenizer; Unicode/empty/truncation; geometry; declared selection/guidance; conditioning output admission; retained component/decoder owners; negatives="<<rejected<<"; denoiser_calls=0\n";
    return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
