#include "wan_family.h"
#include "vrhino/error.h"
#include <cmath>

namespace vrhino::wan_family {
std::shared_ptr<const Definition> lower(const CanonicalArchitectureDeclaration& declaration) {
    const auto& j=declaration.declaration();
    const auto& d=j.at("dimensions");
    Config c;
    c.dim=d.at("hidden_size").integer(); c.heads=d.at("head_count").integer(); c.layers=d.at("block_count").integer();
    c.ffn=d.at("feed_forward_size").integer(); c.frequency=d.at("time_frequency_size").integer();
    c.input_channels=d.at("input_channels").integer(); c.output_channels=d.at("output_channels").integer();
    c.text_dim=j.at("conditioning").at("text_feature_size").integer();
    c.text_length=j.at("conditioning").at("text_token_limit").integer();
    c.epsilon=static_cast<float>(j.at("normalization").at("epsilon").number());
    c.rope_theta=j.at("position").at("theta").number();
    const auto& patch=j.at("patch").at("size").array();
    for (size_t i=0;i<3;++i) c.patch[i]=static_cast<int>(patch[i].integer());
    return lower(c);
}

std::shared_ptr<const Definition> lower(const Config& c) {
    for (auto n : {c.dim,c.heads,c.ffn,c.frequency,c.text_dim,c.text_length,c.input_channels,c.output_channels})
        require(n>0 && n<=1048576, "Family dimension outside supported bound");
    require(c.layers>0 && c.layers<=1024, "Family layer count outside supported bound");
    require(c.dim%c.heads==0 && c.dim/c.heads>=6 && (c.dim/c.heads)%2==0, "Family head/RoPE dimension mismatch");
    require(c.frequency%2==0 && c.input_channels==c.output_channels, "Family frequency/flow channel mismatch");
    require(std::isfinite(c.epsilon) && c.epsilon>0 && std::isfinite(c.rope_theta) && c.rope_theta>0,
            "Invalid family normalization/RoPE configuration");
    for (auto p:c.patch) require(p>0 && p<=1024, "Family patch outside supported bound");
    const int64_t d=c.dim/c.heads, D=c.dim, F=c.ffn;
    std::vector<int> axes{static_cast<int>(d-4*(d/6)),static_cast<int>(2*(d/6)),static_cast<int>(2*(d/6))};
    std::vector<ArchitectureParameterSlot> slots;
    auto slot=[&](std::string role,std::vector<int64_t> shape) { slots.push_back({std::move(role),std::move(shape)}); };
    auto linear=[&](const std::string& role,int64_t out,int64_t in) { slot(role+".weight",{out,in}); slot(role+".bias",{out}); };
    slot("patch_embedding.weight",{D,c.input_channels,c.patch[0],c.patch[1],c.patch[2]}); slot("patch_embedding.bias",{D});
    linear("text_embedding.0",D,c.text_dim); linear("text_embedding.2",D,D);
    linear("time_embedding.0",D,c.frequency); linear("time_embedding.2",D,D); linear("time_projection.1",6*D,D);
    for (int64_t i=0;i<c.layers;++i) {
        const std::string prefix="blocks."+std::to_string(i)+".";
        slot(prefix+"modulation",{1,6,D});
        for (const std::string attention : {"self_attn","cross_attn"}) {
            for (const std::string projection : {"q","k","v","o"}) linear(prefix+attention+"."+projection,D,D);
            slot(prefix+attention+".norm_q.weight",{D}); slot(prefix+attention+".norm_k.weight",{D});
        }
        slot(prefix+"norm3.weight",{D}); slot(prefix+"norm3.bias",{D});
        linear(prefix+"ffn.0",F,D); linear(prefix+"ffn.2",D,F);
    }
    slot("head.modulation",{1,2,D});
    const int64_t output=shape_numel({c.output_channels,c.patch[0],c.patch[1],c.patch[2]});
    linear("head.head",output,D);
    return std::shared_ptr<const Definition>(new Definition(c,d,std::move(axes),
        std::make_shared<const ArchitectureBindingDeclaration>(std::move(slots))));
}

void validate_latent_shape(const Definition& d,const std::vector<int64_t>& shape) {
    const auto& c=d.config;
    require(shape.size()==5 && shape[0]==1 && shape[1]==c.input_channels,"Family latent BCTHW contract mismatch");
    for (size_t i=0;i<3;++i) require(shape[i+2]>0 && shape[i+2]%c.patch[i]==0,"Family latent/patch grid mismatch");
    (void)shape_numel(shape);
}

void validate_request(const Definition& d,const std::vector<int64_t>& shape,const Tensor& positive,const Tensor& negative) {
    validate_latent_shape(d,shape);
    const auto& c=d.config;
    for (const Tensor* context : {&positive,&negative}) {
        require(context->defined() && context->ndim()==3 && context->dim(0)==shape[0] &&
            context->dim(1)>0 && context->dim(1)<=c.text_length && context->dim(2)==c.text_dim &&
            (context->dtype()==DType::F32 || context->dtype()==DType::BF16),"Family text conditioning contract mismatch");
        validate_architecture_tensor(*context,context->shape(),context->dtype());
    }
}
}  // namespace vrhino::wan_family
