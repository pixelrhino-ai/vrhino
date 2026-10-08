#include "ltx_declaration.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"
#include <bit>
#include <limits>
#include <set>

namespace vrhino::ltx_internal {
Json token_flow_declaration(DType storage) {
    require(storage == DType::F32 || storage == DType::BF16, "Unsupported LTX parameter storage");
    auto j = Json::parse(
        R"({"schema":"vrhino.architecture.ltx_token_flow.v1","topology":"ltx_rms_self_masked_cross_ffn.v1",
        "dimensions":{"latent_channels":128,"hidden_size":2048,"head_count":32,"block_count":28,"feed_forward_size":8192,"time_frequency_size":256},
        "normalization":{"self_pre":"rms_norm","cross_pre":"identity","feed_forward_pre":"rms_norm","final":"layer_norm","epsilon":0.000001,"query_key_epsilon":0.00001},
        "position":{"encoding":"fractional_rope_hidden_width","theta":10000.0,"max_positions":[20.0,2048.0,2048.0],"coordinate_scale":[0.32,32.0,32.0],"centered":true},
        "conditioning":{"text_feature_size":4096,"text_token_limit":128,"text_padding":"preserve_mask","timestep":"sigma_f32_1x1","branch_order":"unconditional_then_conditional"},
        "parameters":{"slot_schema":"ltx_token_flow.slots.v1","storage_dtype":"float32","layout":"contiguous"}})");
    auto o = j.object();
    auto p = j.at("parameters").object();
    p["storage_dtype"] = Json(std::string(storage == DType::F32 ? "float32" : "bfloat16"));
    o["parameters"] = Json(p);
    return Json(o);
}
TokenFlowDefinition lower_token_flow(const Json &declaration) {
    for (const auto &[name, value] : declaration.at("dimensions").object()) {
        (void)name;
        (void)value.integer();
    }
    (void)declaration.at("conditioning").at("text_feature_size").integer();
    (void)declaration.at("conditioning").at("text_token_limit").integer();
    const auto &dtype = declaration.at("parameters").at("storage_dtype").string();
    require(dtype == "float32" || dtype == "bfloat16", "Unsupported LTX slot dtype");
    const auto storage = dtype == "float32" ? DType::F32 : DType::BF16;
    require(declaration.serialize() == token_flow_declaration(storage).serialize(),
            "Unsupported or incomplete closed LTX topology");
    std::vector<ArchitectureParameterSlot> slots;
    auto slot = [&](std::string role, std::vector<int64_t> shape) {
        slots.push_back({std::move(role), std::move(shape), storage});
    };
    auto linear = [&](std::string name, int64_t out, int64_t in) {
        slot(name + ".weight", {out, in});
        slot(name + ".bias", {out});
    };
    linear("patchify_proj", 2048, 128);
    linear("caption_projection.linear_1", 2048, 4096);
    linear("caption_projection.linear_2", 2048, 2048);
    linear("adaln_single.emb.timestep_embedder.linear_1", 2048, 256);
    linear("adaln_single.emb.timestep_embedder.linear_2", 2048, 2048);
    linear("adaln_single.linear", 12288, 2048);
    slot("scale_shift_table", {2, 2048});
    linear("proj_out", 128, 2048);
    for (int i = 0; i < 28; ++i) {
        const std::string prefix = "transformer_blocks." + std::to_string(i) + ".";
        slot(prefix + "scale_shift_table", {6, 2048});
        for (const auto *attention : {"attn1.", "attn2."}) {
            for (const auto *projection : {"to_q", "to_k", "to_v", "to_out.0"})
                linear(prefix + attention + projection, 2048, 2048);
            slot(prefix + attention + "q_norm.weight", {2048});
            slot(prefix + attention + "k_norm.weight", {2048});
        }
        linear(prefix + "ff.net.0.proj", 8192, 2048);
        linear(prefix + "ff.net.2", 2048, 8192);
    }
    require(slots.size() == 715, "LTX slot schema implementation mismatch");
    return {storage, std::make_shared<const ArchitectureBindingDeclaration>(std::move(slots))};
}
std::shared_ptr<const ArchitectureBindingDeclaration> decoder_slots(DType storage) {
    require(storage == DType::F32 || storage == DType::BF16, "Unsupported decoder slot storage");
    std::vector<ArchitectureParameterSlot> slots;
    auto slot = [&](std::string role, std::vector<int64_t> shape) {
        slots.push_back({std::move(role), std::move(shape), storage});
    };
    // Frozen role/shape inventory from the existing 0.9.1 converter spec.
    // This is architecture lowering data, not an operator/workflow declaration.
#include "ltx_decoder_slots.inc"
    require(slots.size() == 297, "LTX decoder slot schema mismatch");
    return std::make_shared<const ArchitectureBindingDeclaration>(std::move(slots));
}
void validate_token_flow_inputs(const TensorBundle &input, bool conditioning) {
    const std::set<std::string> fields{
        "seed",     "decode_seed", "latent_grid",   "latent_shape",  "coordinates",
        "positive", "negative",    "positive_mask", "negative_mask", "audit_trace"};
    for (const auto &[name, t] : input) {
        (void)t;
        require(fields.contains(name), "Unknown managed LTX request field");
    }
    if (input.contains("audit_trace")) {
        validate_architecture_tensor(input.at("audit_trace"), {}, DType::I64);
        require(read_scalar_i64(input.at("audit_trace")) == 0 ||
                    read_scalar_i64(input.at("audit_trace")) == 1,
                "Invalid audit flag");
    }
    for (const auto *key :
         {"sampling_steps", "guidance_scale", "flow_shift", "resolution_shift_min_tokens",
          "resolution_shift_max_tokens", "resolution_shift_min", "resolution_shift_max"})
        require(!input.contains(key), "Managed LTX program cannot be overridden");
    for (const auto *name : {"seed", "decode_seed"})
        validate_architecture_tensor(input.at(name), {}, DType::I64);
    require(std::bit_cast<uint64_t>(read_scalar_i64(input.at("decode_seed"))) ==
                std::bit_cast<uint64_t>(read_scalar_i64(input.at("seed"))) + uint64_t(100),
            "LTX decode seed mismatch");
    for (const auto *name : {"latent_grid", "latent_shape"})
        validate_architecture_tensor(input.at(name), {3}, DType::I64);
    const auto *g = input.at("latent_grid").data_as<int64_t>();
    int64_t n = 1;
    for (int i = 0; i < 3; ++i) {
        require(g[i] > 0 && g[i] <= 512 && n <= INT64_MAX / g[i], "LTX grid overflow/bounds");
        n *= g[i];
    }
    const auto *s = input.at("latent_shape").data_as<int64_t>();
    require(s[0] == 1 && s[1] == n && s[2] == 128, "LTX token latent/grid mismatch");
    const auto &c = input.at("coordinates");
    validate_architecture_tensor(c, {1, 3, n}, DType::F32);
    for (int64_t i = 0; i < n; ++i)
        require(c.data_as<float>()[i] == float(i / (g[1] * g[2])) &&
                    c.data_as<float>()[n + i] == float((i / g[2]) % g[1]) &&
                    c.data_as<float>()[2 * n + i] == float(i % g[2]),
                "LTX raw coordinate order/content mismatch");
    if (!conditioning)
        return;
    for (const auto *name : {"positive", "negative"}) {
        const auto &t = input.at(name);
        require(t.dtype() == DType::F32 || t.dtype() == DType::BF16, "LTX conditioning dtype");
        validate_architecture_tensor(t, {1, 128, 4096}, t.dtype());
    }
    for (const auto *name : {"positive_mask", "negative_mask"}) {
        const auto &m = input.at(name);
        validate_architecture_tensor(m, {1, 128}, DType::Bool);
        bool any = false;
        for (int i = 0; i < 128; ++i) {
            require(m.data_as<uint8_t>()[i] <= 1, "Invalid validity mask byte");
            any |= m.data_as<uint8_t>()[i] != 0;
        }
        require(any, "LTX conditioning has no valid token");
    }
}
} // namespace vrhino::ltx_internal
