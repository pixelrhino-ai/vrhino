#include "vrhino/canonical_architecture.h"
#include "vrhino/error.h"
#include "../src/architectures/wan_family.h"
#include <fstream>
#include <iostream>
#include <tuple>
using namespace vrhino;
namespace {
int rejected=0;
template<class F> void reject(F f) { bool caught=false; try { f(); } catch(const Error&) { caught=true; }
    require(caught,"Invalid canonical input accepted"); ++rejected; }
Json changed(Json j,const std::string& group,const std::string& field,Json value) {
    auto root=j.object(); auto child=root.at(group).object(); child[field]=std::move(value);
    root[group]=Json(std::move(child)); return Json(std::move(root));
}
Json text(const char* s) { return Json(std::string(s)); }
}
int main() {
    try {
        // Independent canonical producer: no upstream aliases, paths or class.
        const auto canonical=Json::parse(R"({
          "schema":"vrhino.architecture.v1","topology":"modulated_self_cross_ffn.v1",
          "dimensions":{"hidden_size":12,"head_count":2,"block_count":2,"feed_forward_size":20,"time_frequency_size":6,"input_channels":2,"output_channels":2},
          "normalization":{"query_key":"rms_norm","self_pre":"layer_norm","cross_pre":"affine_layer_norm","feed_forward_pre":"layer_norm","epsilon":0.000001},
          "attention":{"self":"global_noncausal","cross":"global_noncausal","head_layout":"BSHD"},
          "patch":{"size":[1,2,1],"latent_layout":"BCTHW"},
          "position":{"encoding":"rope_3d_adjacent","axis_partition":"temporal_remainder_spatial_sixths","theta":10000.0},
          "conditioning":{"text_feature_size":8,"text_token_limit":4,"text_padding":"zero_before_projection","timestep":"per_sample_sinusoidal","branch_order":"unconditional_then_conditional"},
          "parameters":{"slot_schema":"modulated_self_cross_ffn.slots.v1","storage_dtype":"float32","layout":"contiguous"}
        })");
        auto lowered=wan_family::lower(CanonicalArchitectureDeclaration::parse(canonical));
        require(lowered->config.dim==12 && lowered->config.layers==2 && lowered->parameters->slots().size()==69,
                "Independent canonical producer did not lower");
        auto envelope=Json(Json::Object{{"canonical_architecture",canonical}});
        require(wan_family::lower(canonical_architecture_from_vrm_descriptor(envelope))->config==lowered->config,
                "Canonical ingress drift");
        for(const auto& alias:{"dim","model_type","_class_name","source_file","shard_layout","repository"}) {
            reject([&]{ auto j=canonical.object(); j[alias]=text("untrusted"); CanonicalArchitectureDeclaration::parse(Json(j)); });
            reject([&]{ CanonicalArchitectureDeclaration::parse(changed(canonical,"dimensions",alias,text("untrusted"))); });
        }
        for(const auto& [group,field,value]:std::vector<std::tuple<std::string,std::string,Json>>{
            {"normalization","query_key",text("layer_norm")},{"attention","self",text("causal")},
            {"patch","latent_layout",text("BTHWC")},{"position","axis_partition",text("unknown")},
            {"conditioning","timestep",text("per_token")},{"parameters","layout",text("strided")},
            {"parameters","storage_dtype",text("bfloat16")},{"parameters","slot_schema",text("unknown")},
            {"dimensions","head_count",Json(int64_t(0))},{"dimensions","hidden_size",Json(12.5)},
            {"patch","size",Json(Json::Array{Json(int64_t(1))})}})
            reject([&]{ CanonicalArchitectureDeclaration::parse(changed(canonical,group,field,value)); });
        reject([&]{ wan_family::lower(CanonicalArchitectureDeclaration::parse(changed(canonical,"dimensions","hidden_size",Json(int64_t(13))))); });
        for(const auto& [key,value]:canonical.object()) {
            (void)value; reject([&]{ auto j=canonical.object(); j.erase(key); CanonicalArchitectureDeclaration::parse(Json(j)); });
        }
        std::ifstream file(VRHINO_FAMILY_SPEC); require(file.good(),"Missing legacy fixture");
        auto descriptor=Json::parse(std::string(std::istreambuf_iterator<char>(file),{})).at("architecture_graph");
        auto migrated=canonical_architecture_from_vrm_descriptor(descriptor);
        require(wan_family::lower(migrated)->config==wan_family::Config{},"Legacy defaults changed");
        auto roundtrip=CanonicalArchitectureDeclaration::parse(Json::parse(migrated.declaration().serialize()));
        require(wan_family::lower(roundtrip)->config==wan_family::Config{},"Canonical roundtrip changed semantics");
        reject([&]{ CanonicalArchitectureDeclaration::parse(descriptor.at("config")); });
        reject([&]{ canonical_architecture_from_vrm_descriptor(descriptor.at("config")); });
        reject([&]{ auto j=descriptor.object(); j["implementation_id"]=text("unknown"); canonical_architecture_from_vrm_descriptor(Json(j)); });
        reject([&]{ auto j=descriptor.object(); j["schema_version"]=Json(int64_t(2)); canonical_architecture_from_vrm_descriptor(Json(j)); });
        reject([&]{ auto j=descriptor.object(); j["canonical_architecture"]=canonical; canonical_architecture_from_vrm_descriptor(Json(j)); });
        std::cout<<"canonical architecture qualification=PASS negative_cases="<<rejected<<" independent_producer=PASS legacy_migration=PASS\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
