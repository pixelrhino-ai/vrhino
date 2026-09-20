#include "vrhino/canonical_architecture.h"
#include "vrhino/error.h"
#include <cmath>
#include <set>

namespace vrhino {
namespace {
void keys(const Json& j, std::initializer_list<const char*> expected) {
    std::set<std::string> names;
    for (const auto* name : expected) names.insert(name);
    require(j.object().size()==names.size(), "Canonical architecture field set mismatch");
    for (const auto& [key, value] : j.object()) {
        (void)value;
        require(names.contains(key), "Unknown canonical architecture field: " + key);
    }
}
void tag(const Json& j, const char* key, const char* expected) {
    require(j.at(key).string()==expected, std::string("Unsupported canonical semantics: ")+key);
}
void positive(const Json& j, const char* key, int64_t limit) {
    const auto n=j.at(key).integer();
    require(n>0 && n<=limit, std::string("Canonical dimension out of range: ")+key);
}
}
CanonicalArchitectureDeclaration CanonicalArchitectureDeclaration::parse(const Json& j) {
    keys(j,{"schema","topology","dimensions","normalization","attention","patch","position","conditioning","parameters"});
    tag(j,"schema","vrhino.architecture.v1");
    // This topology version fixes six-way modulation, tanh-GELU FFN, SiLU
    // timestep projection and two-way output modulation. It is not a model ID.
    tag(j,"topology","modulated_self_cross_ffn.v1");
    const auto& d=j.at("dimensions");
    keys(d,{"hidden_size","head_count","block_count","feed_forward_size","time_frequency_size","input_channels","output_channels"});
    for (const auto& [key,value]:d.object()) { (void)value; positive(d,key.c_str(),key=="block_count"?1024:1048576); }
    const auto& n=j.at("normalization");
    keys(n,{"query_key","self_pre","cross_pre","feed_forward_pre","epsilon"});
    tag(n,"query_key","rms_norm"); tag(n,"self_pre","layer_norm");
    tag(n,"cross_pre","affine_layer_norm"); tag(n,"feed_forward_pre","layer_norm");
    require(std::isfinite(n.at("epsilon").number()) && n.at("epsilon").number()>0,"Invalid canonical normalization epsilon");
    const auto& a=j.at("attention"); keys(a,{"self","cross","head_layout"});
    tag(a,"self","global_noncausal"); tag(a,"cross","global_noncausal"); tag(a,"head_layout","BSHD");
    const auto& p=j.at("patch"); keys(p,{"size","latent_layout"}); tag(p,"latent_layout","BCTHW");
    require(p.at("size").array().size()==3,"Canonical patch rank mismatch");
    for (const auto& v:p.at("size").array()) require(v.integer()>0 && v.integer()<=1024,"Canonical patch size out of range");
    const auto& r=j.at("position"); keys(r,{"encoding","axis_partition","theta"});
    tag(r,"encoding","rope_3d_adjacent"); tag(r,"axis_partition","temporal_remainder_spatial_sixths");
    require(std::isfinite(r.at("theta").number()) && r.at("theta").number()>0,"Invalid canonical RoPE base");
    const auto& c=j.at("conditioning"); keys(c,{"text_feature_size","text_token_limit","text_padding","timestep","branch_order"});
    positive(c,"text_feature_size",1048576); positive(c,"text_token_limit",1048576);
    tag(c,"text_padding","zero_before_projection"); tag(c,"timestep","per_sample_sinusoidal");
    tag(c,"branch_order","unconditional_then_conditional");
    const auto& s=j.at("parameters"); keys(s,{"slot_schema","storage_dtype","layout"});
    // Slot schema declares the complete role/shape equations; lowering expands
    // it into ArchitectureBindingDeclaration before any binding is admitted.
    tag(s,"slot_schema","modulated_self_cross_ffn.slots.v1");
    tag(s,"storage_dtype","float32"); tag(s,"layout","contiguous");
    return CanonicalArchitectureDeclaration(j);
}
}  // namespace vrhino
