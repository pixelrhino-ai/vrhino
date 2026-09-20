#include "vrhino/product/request_wiring.h"
#include "vrhino/product/declared_run.h"
#include "vrhino/product/run_request.h"
#include "vrhino/product/program_declaration.h"
#include "vrhino/conditioning.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"
#include <fstream>
#include <set>
#include <cmath>
#include <limits>
#include <algorithm>
#include <bit>
#include <charconv>

#ifndef VRHINO_PRODUCT_TOKENIZERS
#define VRHINO_PRODUCT_TOKENIZERS 0
#endif
namespace vrhino::product {
namespace {
void keys(const Json& j, std::initializer_list<const char*> names) {
    require(j.object().size()==names.size(), "Request declaration field set mismatch");
    for (const auto* key:names) require(j.find(key), "Missing request declaration field");
}
std::string text_file(const std::filesystem::path& path, size_t bound=32*1024*1024) {
    require(std::filesystem::file_size(path)<=bound, "Oversized request resource");
    std::ifstream f(path,std::ios::binary); require(f.good(), "Cannot open request resource");
    std::string text(bound+1,'\0'); f.read(text.data(),static_cast<std::streamsize>(text.size()));
    text.resize(static_cast<size_t>(f.gcount())); require(text.size()<=bound && !f.bad(), "Invalid request resource read");
    return text;
}
Json parse(const std::string& text, bool external_metadata=false) {
    JsonParseLimits l; l.maximum_document_bytes=32*1024*1024;
    l.maximum_container_entries=300000;l.maximum_total_values=2000000;
    l.oversized_integer_as_float=external_metadata;
    return Json::parse(text,l);
}
const ResolvedArtifact& artifact(const AdmittedLocalProduct& p,const std::string& id) {
    const auto it=p.resources.artifacts.find(id); require(it!=p.resources.artifacts.end(), "Unknown request artifact");
    require(it->second.declaration.required,"Request requires mandatory artifact"); return it->second;
}
std::string verified_text(const AdmittedLocalProduct& p,const std::string& id) {
    const auto& a=artifact(p,id);
    // Recheck the small request-time resource against the admitted identity.
    require(std::filesystem::file_size(a.path)==a.declaration.size && sha256_file(a.path)==a.declaration.sha256,
            "Request resource identity drift");
    return text_file(a.path);
}
int64_t positive(const Json& j,const char* key,int64_t limit) {
    const auto n=j.at(key).integer();require(n>0 && n<=limit,"Request dimension outside declared bound");return n;
}
Json shape_json(const std::vector<int64_t>& shape) {
    Json::Array a;for(auto n:shape)a.emplace_back(n);return Json(std::move(a));
}
uint64_t request_seed(const Json& value) {
    if(value.is_int()) {
        require(value.integer()>=0,"Negative seed");return static_cast<uint64_t>(value.integer());
    }
    require(value.is_string(),"Seed must be an integer or canonical uint64 string");
    const auto& text=value.string();uint64_t seed=0;
    require(!text.empty() && !(text.size()>1 && text.front()=='0') &&
        std::all_of(text.begin(),text.end(),[](char c){return c>='0'&&c<='9';}),"Invalid seed string");
    const auto parsed=std::from_chars(text.data(),text.data()+text.size(),seed);
    require(parsed.ec==std::errc() && parsed.ptr==text.data()+text.size(),"Seed outside uint64 domain");return seed;
}
PreparedTextProductRequest prepare_document(std::shared_ptr<AdmittedLocalProduct> admitted,const Json& document) {
    if(document.find("model")) {
        const auto request=parse_product_run_document(document);
        require(request.model_reference==admitted->resources.manifest.identity.reference(),"Request/package identity mismatch");
        const auto options=map_product_run_options(admitted->resources,request);
        const auto lowered=lower_declared_text_run(admitted->resources,options);
        return prepare_text_product_request(std::move(admitted),lowered.request);
    }
    return prepare_text_product_request(std::move(admitted),document);
}
}
bool native_text_request_available(){return VRHINO_PRODUCT_TOKENIZERS != 0;}
WeightMap PreparedTextProductRequest::conditioning_weights()const {
    require(owner && owner->conditioning,"Missing conditioning resource owner");return owner->conditioning->weights();
}
WeightMap PreparedTextProductRequest::decoder_weights()const {
    require(owner && owner->model,"Missing decoder resource owner");
    return WeightMap(owner->model->bindings(owner->model->metadata().at("shared_components").at("decoder")));
}
TensorBundle PreparedTextProductRequest::bind_conditioning_outputs(const std::vector<Tensor>& outputs)const {
    require(outputs.size()==conditioning.size(),"Conditioning result count mismatch");
    auto inputs=runtime_inputs;
    for(size_t n=0;n<outputs.size();++n){
        const auto& t=outputs[n];require(t.dtype()==DType::F32 || t.dtype()==DType::BF16,"Unsupported conditioning result dtype");
        validate_architecture_tensor(t,conditioning[n].expected_hidden_shape,t.dtype());
        require(inputs.emplace(conditioning[n].target,t).second,"Duplicate conditioning result target");
    }
    return inputs;
}
PreparedTextProductRequest prepare_text_product_request(std::shared_ptr<AdmittedLocalProduct> admitted,const Json& request) {
    require(native_text_request_available(),"Native tokenizer support is unavailable in this build");
    require(admitted && admitted->architecture && admitted->conditioning,"Missing admitted product owner");
    keys(request,{"schema","prompt","negative_prompt","seed","width","height","frames"});
    require(request.at("schema").string()=="vrhino.product-text-request.v1","Unsupported product request schema");
    for(const auto* k:{"prompt","negative_prompt"})require(request.at(k).string().size()<=65536,"Text request exceeds byte limit");
    const auto seed=request_seed(request.at("seed"));
    const auto width=positive(request,"width",16384),height=positive(request,"height",16384),frames=positive(request,"frames",4096);
    const auto& frozen=admitted->resources.manifest.product.frozen_profile;
    if(frozen && frozen->sampling && frozen->sampling->program_artifact) {
        require(frozen->output.width==static_cast<uint64_t>(width) &&
            frozen->output.height==static_cast<uint64_t>(height) &&
            frozen->output.frames==static_cast<uint64_t>(frames),
            "Request geometry contradicts program-backed Product profile");
        require(!request.at("prompt").string().empty(),"Product prompt must be non-empty");
    }
    PreparedTextProductRequest out;out.owner=std::move(admitted);
    const auto manifest=Json::parse(out.owner->resources.manifest.raw_json);
    const auto& admission=manifest.at("admission");
    const auto wiring=parse(verified_text(*out.owner,admission.at("request_artifact").string()));
    keys(wiring,{"schema","tokenizer_spec","tokenizer_config_artifact","geometry","conditioning_bindings"});
    require(wiring.at("schema").string()=="vrhino.text-product-wiring.v1","Unsupported text product wiring");
    const auto& shared=out.owner->model->metadata().at("shared_components");
    const auto& decoder=shared.at("decoder");const auto& latent=decoder.at("latent_contract");
    const auto& g=wiring.at("geometry");keys(g,{"spatial_scale","temporal_scale","temporal_origin"});
    const auto spatial=positive(g,"spatial_scale",1024),temporal=positive(g,"temporal_scale",1024),origin=positive(g,"temporal_origin",1024);
    require(latent.at("layout").string()=="BCTHW" && latent.at("spatial_scale").integer()==spatial &&
        latent.at("temporal_decode").string()==std::to_string(origin)+"+"+std::to_string(temporal)+"*(F-1)","Decoder/request geometry mismatch");
    require(width%spatial==0 && height%spatial==0 && frames>=origin && (frames-origin)%temporal==0,"Request does not fit decoder geometry");
    std::vector<int64_t> latent_shape{1,latent.at("channels").integer(),(frames-origin)/temporal+1,height/spatial,width/spatial};
    out.runtime_inputs={{"seed",scalar_i64(std::bit_cast<int64_t>(seed))},{"latent_shape",host_i64({5},latent_shape)}};
    out.sampling=out.owner->architecture->create_program(out.runtime_inputs);
    const auto package=PackageDeclaration::parse(out.owner->model->graph());
    out.execution=admit_program_declaration(out.owner->model->metadata().at("programs"),package).execution;
    require(package.graphs().size()==1,"Text request requires shared graph interface");
    const auto& conditioning_contract=package.graphs().begin()->second.at("conditioning");
    const int64_t features=conditioning_contract.at("text_feature_size").integer();
    const auto& roles=admission.at("resources");
    const auto graph=parse(verified_text(*out.owner,roles.at("conditioning_declaration").string()));
    require(graph.at("schema_version").integer()==1 && graph.at("kind").string()=="pre_norm_transformer" &&
            graph.at("output_trim_to_mask").boolean(),"Unsupported conditioning output contract");
    const auto weights=out.conditioning_weights();
    const auto& embedding=weights.at(graph.at("embedding").at("weight").string());
    require(embedding.ndim()==2 && embedding.dim(1)==features,"Text embedding/graph feature mismatch");
    for(const auto& b:graph.at("blocks").array()) (void)conditioning_mask_semantic(b.at("attention"));
    const auto tokenizer_blob=verified_text(*out.owner,roles.at("tokenizer").string());
    const auto tokenizer_json=parse(tokenizer_blob);
    // External config may carry an unbounded model_max_length sentinel. The
    // actual sequence bound comes from the strict admitted graph/spec above.
    const auto config=parse(verified_text(*out.owner,wiring.at("tokenizer_config_artifact").string()),true);
    const auto& ts=wiring.at("tokenizer_spec");keys(ts,{"format","max_length","pad_id","suffix_ids"});
    require(ts.at("format").string()=="huggingface_json","Unsupported text tokenizer format");
    TokenizerSpec spec;spec.max_length=positive(ts,"max_length",1048576);
    require(spec.max_length==conditioning_contract.at("text_token_limit").integer(),"Tokenizer/graph sequence bound mismatch");
    const auto pad=ts.at("pad_id").integer();require(pad>=0 && pad<embedding.dim(0) && pad<=INT32_MAX,"Invalid pad ID");spec.pad_id=static_cast<int32_t>(pad);
    const auto& suffix=ts.at("suffix_ids").array();require(suffix.size()==1,"Expected declared EOS suffix");
    const auto eos=suffix[0].integer();require(eos>=0 && eos<embedding.dim(0) && eos<=INT32_MAX,"Invalid EOS ID");spec.suffix_ids={static_cast<int32_t>(eos)};
    // Validate declared special-token IDs against actual resource contents, not
    // a model identity or values copied from a different product's spec.
    const auto special_id=[&](const std::string& token){
        for(const auto& item:tokenizer_json.at("added_tokens").array())
            if(item.at("content").string()==token)return item.at("id").integer();
        throw Error("Special token absent from tokenizer resource");
    };
    require(special_id(config.at("pad_token").string())==pad && special_id(config.at("eos_token").string())==eos,"Tokenizer special-token identity mismatch");
    const auto& bindings=wiring.at("conditioning_bindings").array();require(bindings.size()==2,"CFG conditioning requires two declared inputs");
    std::set<std::string> targets;Json::Array token_evidence;
#if VRHINO_PRODUCT_TOKENIZERS
    NativeTokenizer tokenizer(spec,tokenizer_blob);
#endif
    for(const auto& binding:bindings){
        keys(binding,{"component_id","text_source","target"});
        const auto target=binding.at("target").string(),source=binding.at("text_source").string();
        require((target=="positive" && source=="prompt") || (target=="negative" && source=="negative_prompt"),"Invalid conditioning binding contract");
        require(targets.insert(target).second,"Duplicate conditioning target");
        const auto component_id=binding.at("component_id").string();
        const ComponentDeclaration* component=nullptr;
        for(const auto& c:out.owner->resources.manifest.components)if(c.id==component_id)component=&c;
        require(component && component->kind=="conditioning.text_encoder","Unknown conditioning component");
        for(const auto* key:{"conditioning_declaration","conditioning_index","conditioning_weights","tokenizer"})
            require(std::find(component->artifact_ids.begin(),component->artifact_ids.end(),roles.at(key).string())!=component->artifact_ids.end(),"Component resource reference mismatch");
#if VRHINO_PRODUCT_TOKENIZERS
        const auto tokens=tokenizer.encode(request.at(source).string());
        require(tokens.valid_length>0 && tokens.valid_length<=spec.max_length,"Invalid conditioning token count");
        std::vector<int64_t> ids(tokens.input_ids.begin(),tokens.input_ids.end());
        for(auto id:ids)require(id>=0 && id<embedding.dim(0),"Tokenizer ID outside embedding table");
        out.conditioning.push_back({target,graph,host_i64({1,spec.max_length},ids),
            host_bool({1,spec.max_length},tokens.attention_mask),{1,tokens.valid_length,features}});
        Json::Array ids_json;for(auto id:ids)ids_json.emplace_back(id);
        token_evidence.emplace_back(Json::Object{{"target",Json(target)},{"valid_length",Json(tokens.valid_length)},
            {"input_ids",Json(std::move(ids_json))},{"expected_hidden_shape",shape_json(out.conditioning.back().expected_hidden_shape)}});
#endif
    }
    out.expected_video_shape={1,3,frames,height,width};(void)shape_numel(out.expected_video_shape);
    Json::Array trace;
    const auto& ids=out.owner->model->metadata().at("programs").at("execution").at("instance_ids").array();
    for(size_t n=0;n<ids.size();++n){const auto instance=static_cast<uint32_t>(ids[n].integer());
        trace.emplace_back(Json::Object{{"step",Json(int64_t(n))},{"instance_id",Json(int64_t(instance))},
            {"binding_id",Json(int64_t(package.instances().at(instance).binding))},
            {"model_timestep",Json(read_scalar_i64(out.sampling.model_timestep_at(static_cast<int>(n))))},
            {"guidance",Json(double(out.sampling.guidance_schedule->at(n).scale))}});
    }
    out.evidence=Json(Json::Object{{"request_prepared",Json(true)},{"native_tokenization_executed",Json(true)},
        {"conditioning_executed",Json(false)},{"denoising_executed",Json(false)},{"decode_executed",Json(false)},
        {"numerically_qualified",Json(false)},{"numerical_status",Json(std::string("HOLD"))},
        {"latent_shape",shape_json(latent_shape)},{"expected_video_shape",shape_json(out.expected_video_shape)},
        {"seed",seed<=static_cast<uint64_t>(INT64_MAX)?Json(static_cast<int64_t>(seed)):Json(std::to_string(seed))},
        {"conditioning",Json(std::move(token_evidence))},{"execution_intent",Json(std::move(trace))}});
    (void)out.decoder_weights();
    return out;
}
PreparedTextProductRequest dry_run_local_product(const std::filesystem::path& manifest,
    const std::filesystem::path& resources,const std::filesystem::path& request) {
    require(native_text_request_available(),"Native tokenizer support is unavailable in this build");
    const auto document=parse(text_file(request,256*1024));
    auto admitted=std::make_shared<AdmittedLocalProduct>(preflight_local_product(manifest,resources));
    return prepare_document(std::move(admitted),document);
}
PreparedTextProductRequest dry_run_resolved_product(ResolvedRunnableModel resources,
    const std::filesystem::path& request) {
    require(native_text_request_available(),"Native tokenizer support is unavailable in this build");
    const auto document=parse(text_file(request,256*1024));
    auto admitted=std::make_shared<AdmittedLocalProduct>(preflight_resolved_product(std::move(resources)));
    return prepare_document(std::move(admitted),document);
}
} // namespace vrhino::product
