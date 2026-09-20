#include "vrhino/canonical_architecture.h"
#include "vrhino/loader.h"
#include "vrhino/error.h"
#include <set>

namespace vrhino {
CanonicalArchitectureDeclaration canonical_architecture_from_vrm_descriptor(const Json& descriptor) {
    if (const auto* canonical=descriptor.find("canonical_architecture")) {
        require(!descriptor.find("config"),"Ambiguous canonical and legacy VRM architecture inputs");
        return CanonicalArchitectureDeclaration::parse(*canonical);
    }
    // Compatibility for an already canonical-name-mapped, historical VRM
    // envelope. This is not a general upstream/Diffusers config reader.
    require(descriptor.at("schema_version").integer()==1 &&
            descriptor.at("implementation_id").string()=="dit_flow.wan.denoiser.v1" &&
            !descriptor.at("runtime_tensor_bindings").object().empty(),"Unsupported legacy VRM architecture descriptor");
    const auto& old=descriptor.at("config");
    const std::set<std::string> allowed{"dim","num_heads","num_layers","ffn_dim","freq_dim","text_dim","text_len",
        "in_dim","out_dim","patch_size","eps","qk_norm","cross_attn_norm","model_type","window_size","rope_theta"};
    for (const auto& [key,value]:old.object()) { (void)value; require(allowed.contains(key),"Unknown legacy VRM config field: "+key); }
    require(old.at("model_type").string()=="t2v" && old.at("qk_norm").boolean() && old.at("cross_attn_norm").boolean(),
            "Unsupported legacy VRM normalization/conditioning");
    const auto& window=old.at("window_size").array();
    require(window.size()==2 && window[0].integer()==-1 && window[1].integer()==-1,"Unsupported legacy VRM attention window");
    auto j=Json::parse(R"({
      "schema":"vrhino.architecture.v1", "topology":"modulated_self_cross_ffn.v1",
      "dimensions":{},
      "normalization":{"query_key":"rms_norm","self_pre":"layer_norm","cross_pre":"affine_layer_norm","feed_forward_pre":"layer_norm","epsilon":0.000001},
      "attention":{"self":"global_noncausal","cross":"global_noncausal","head_layout":"BSHD"},
      "patch":{"size":[1,2,2],"latent_layout":"BCTHW"},
      "position":{"encoding":"rope_3d_adjacent","axis_partition":"temporal_remainder_spatial_sixths","theta":10000.0},
      "conditioning":{"text_feature_size":4096,"text_token_limit":512,"text_padding":"zero_before_projection","timestep":"per_sample_sinusoidal","branch_order":"unconditional_then_conditional"},
      "parameters":{"slot_schema":"modulated_self_cross_ffn.slots.v1","storage_dtype":"float32","layout":"contiguous"}
    })").object();
    Json::Object dims;
    for (const auto& [to,from]:std::initializer_list<std::pair<const char*,const char*>>{
        {"hidden_size","dim"},{"head_count","num_heads"},{"block_count","num_layers"},{"feed_forward_size","ffn_dim"},
        {"time_frequency_size","freq_dim"},{"input_channels","in_dim"},{"output_channels","out_dim"}}) dims.emplace(to,old.at(from));
    j["dimensions"]=Json(std::move(dims));
    auto norm=j.at("normalization").object(); norm["epsilon"]=old.at("eps"); j["normalization"]=Json(std::move(norm));
    auto patch=j.at("patch").object(); patch["size"]=old.at("patch_size"); j["patch"]=Json(std::move(patch));
    auto position=j.at("position").object(); if (const auto* theta=old.find("rope_theta")) position["theta"]=*theta;
    j["position"]=Json(std::move(position));
    auto conditioning=j.at("conditioning").object(); conditioning["text_feature_size"]=old.at("text_dim");
    conditioning["text_token_limit"]=old.at("text_len"); j["conditioning"]=Json(std::move(conditioning));
    return CanonicalArchitectureDeclaration::parse(Json(std::move(j)));
}
CanonicalArchitectureDeclaration VrmModel::canonical_architecture() const {
    return canonical_architecture_from_vrm_descriptor(graph().at("architecture_graph"));
}
}  // namespace vrhino
