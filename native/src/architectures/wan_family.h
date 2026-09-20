#pragma once
#include <array>
#include "vrhino/architecture_binding.h"
#include "vrhino/canonical_architecture.h"

namespace vrhino::wan_family {
struct Config {
    int64_t dim = 1536, heads = 12, layers = 30, ffn = 8960;
    int64_t frequency = 256, text_dim = 4096, text_length = 512, input_channels = 16, output_channels = 16;
    std::array<int, 3> patch{1, 2, 2};
    double rope_theta = 10000.0;
    float epsilon = 1e-6f;
    bool operator==(const Config&) const = default;
};

class Definition {
public:
    const Config config;
    const int64_t head_dim;
    const std::vector<int> rope_axes;
    const std::shared_ptr<const ArchitectureBindingDeclaration> parameters;
private:
    Definition(Config c, int64_t head, std::vector<int> axes,
               std::shared_ptr<const ArchitectureBindingDeclaration> slots)
        : config(std::move(c)), head_dim(head), rope_axes(std::move(axes)), parameters(std::move(slots)) {}
    friend std::shared_ptr<const Definition> lower(const Config&);
};

// Lowering adapter: metadata only. It cannot allocate tensors, invoke Backend,
// choose a component, create a sampling loop, or override PrecisionPolicy.
std::shared_ptr<const Definition> lower(const CanonicalArchitectureDeclaration& config);
std::shared_ptr<const Definition> lower(const Config& config);
void validate_latent_shape(const Definition& definition, const std::vector<int64_t>& latent_shape);
void validate_request(const Definition& definition, const std::vector<int64_t>& latent_shape,
                      const Tensor& positive, const Tensor& negative);
}  // namespace vrhino::wan_family
