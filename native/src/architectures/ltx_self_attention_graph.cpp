#include "ltx_self_attention_graph.h"

namespace vrhino::ltx_internal {
SelfAttentionContract self_attention_contract(DType execution, DType hidden) {
    if (execution == DType::F32 && hidden == DType::F32) return SelfAttentionContract::F32;
    if (execution == DType::BF16 && hidden == DType::F32) return SelfAttentionContract::BF16F32Hidden;
    if (execution == DType::BF16 && hidden == DType::BF16) return SelfAttentionContract::BF16Hidden;
    throw Error("Unsupported LTX self-attention dtype contract");
}
neural_graph::Graph self_attention_graph(int64_t b, int64_t s, int64_t m,
                                         bool diagnostics, bool block_state,
                                         SelfAttentionContract contract) {
    namespace ng = neural_graph;
    require(contract == SelfAttentionContract::F32 || contract == SelfAttentionContract::BF16F32Hidden ||
            contract == SelfAttentionContract::BF16Hidden, "Unknown LTX self-attention dtype contract");
    require(b > 0 && s > 0 && (m == 1 || m == s), "LTX self-attention shape contract");
    const std::vector<int64_t> hidden{b,s,2048}, part{b,m,2048}, heads{b,s,32,64};
    ng::Description d;
    d.schema = ng::schema_v1;
    d.values = {{hidden},{{b,m,12288}},{hidden},{hidden},
        {{6,2048}},{{2048,2048}},{{2048}},{{2048,2048}},{{2048}},
        {{2048,2048}},{{2048}},{{2048}},{{2048}},{{2048,2048}},{{2048}},{{1}},
        {{1,1,6,2048}},{{b,m,6,2048}},{{b,m,6,2048}},
        {{b,m,1,2048}},{part},{{b,m,1,2048}},{part},{{b,m,1,2048}},{part},
        {hidden},{part},{hidden},{hidden},{hidden},{hidden},{hidden},
        {hidden},{hidden},{hidden},{hidden},{heads},{heads},{heads},{heads},
        {hidden},{hidden},{hidden},{hidden}};
    d.inputs = {0,1,2,3}; d.parameters = {4,5,6,7,8,9,10,11,12,13,14};
    d.constants = {{15,1.0f}};
    d.nodes = {
        {16,ng::Reshape{4,{1,1,6,2048}}}, {17,ng::Reshape{1,{b,m,6,2048}}},
        {18,ng::Add{16,17}},
        {19,ng::Slice{18,2,0,1}}, {20,ng::Reshape{19,part}},
        {21,ng::Slice{18,2,1,2}}, {22,ng::Reshape{21,part}},
        {23,ng::Slice{18,2,2,3}}, {24,ng::Reshape{23,part}},
        {25,ng::RmsNorm{0,{},2,1e-6f}}, {26,ng::Add{15,22}},
        {27,ng::Mul{25,26}}, {28,ng::Add{27,20}},
        {29,ng::Linear{28,5,6}}, {30,ng::Linear{28,7,8}}, {31,ng::Linear{28,9,10}},
        {32,ng::RmsNorm{29,11,2,1e-5f}}, {33,ng::RmsNorm{30,12,2,1e-5f}},
        {34,ng::Rope{32,2,3}}, {35,ng::Rope{33,2,3}},
        {36,ng::Reshape{34,heads}}, {37,ng::Reshape{35,heads}},
        {38,ng::Reshape{31,heads}}, {39,ng::Attention{36,37,38,0.125f}},
        {40,ng::Reshape{39,hidden}}, {41,ng::Linear{40,13,14}},
        {42,ng::Mul{41,24}}, {43,ng::Add{0,42}}};
    if (contract != SelfAttentionContract::F32) {
        const DType hidden_dtype = contract == SelfAttentionContract::BF16Hidden ? DType::BF16 : DType::F32;
        d.execution_dtype = DType::BF16; d.values[0].dtype = hidden_dtype;
        for (auto id : {2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,19,20,21,22,23,24,29,30,31,38,39,40,41})
            d.values[id].dtype = DType::BF16;
        d.values[25].dtype = hidden_dtype;
        auto original = std::move(d.nodes); d.nodes.clear();
        auto cast = [&](ng::ValueId source, DType destination) {
            const auto id = static_cast<ng::ValueId>(d.values.size());
            d.values.push_back({d.values[source].shape, destination});
            d.nodes.push_back({id, ng::Cast{source, destination}});
            return id;
        };
        // Preserve the admitted explicit boundaries and value IDs.
        for (auto node : original) {
            if (auto* op = std::get_if<ng::Linear>(&node.op)) op->compute_dtype = DType::BF16;
            if (node.result == 18) std::get<ng::Add>(node.op).b = cast(17, DType::BF16);
            if (node.result == 26) std::get<ng::Add>(node.op).b = cast(22, DType::F32);
            if (node.result == 27 && hidden_dtype == DType::BF16)
                std::get<ng::Mul>(node.op).a = cast(25, DType::F32);
            if (node.result == 28) std::get<ng::Add>(node.op).b = cast(20, DType::F32);
            if (node.result == 42) {
                std::get<ng::Mul>(node.op).a = cast(41, DType::F32);
                std::get<ng::Mul>(node.op).b = cast(24, DType::F32);
            }
            if (node.result == 43 && hidden_dtype == DType::BF16)
                std::get<ng::Add>(node.op).a = cast(0, DType::F32);
            d.nodes.push_back(std::move(node));
        }
    }
    d.outputs = diagnostics ? std::vector<ng::ValueId>{
        0,18,20,22,24,25,28,29,30,31,32,33,34,35,36,37,38,39,40,41,43}
        : std::vector<ng::ValueId>{43};
    // Export existing shared modulation value; no extra computation/nodes.
    if (block_state && !diagnostics) d.outputs = {43,18};
    return ng::admit(d);
}
}
