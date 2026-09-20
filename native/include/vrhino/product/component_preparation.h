#pragma once
#include "vrhino/product/request_wiring.h"

namespace vrhino::product {
// Product preparation owns component orchestration. Returned host tensors do
// not retain device caches; source leases stay with the executing Backend.
TensorBundle execute_text_conditioning(Backend&, const PreparedTextProductRequest&);
Json admit_conditioned_sampling(Backend&, const PrecisionPolicy&,
    const PreparedTextProductRequest&, const TensorBundle&);
void require_finite_component_output(const Tensor&);
// Bounded normal-product component qualification, never a sampling entrypoint.
Json check_local_components(const std::filesystem::path& manifest,
    const std::filesystem::path& resources, const std::filesystem::path& request,
    const std::filesystem::path& options, const std::filesystem::path& output);
// Explicit qualification of a bounded prefix of the original admitted
// schedule. Does not grant full-run admission or change package qualification.
// Legacy v1-v3 use F32; v4 consumes an admitted precision-policy artifact.
Json check_local_sampling_step(const std::filesystem::path& manifest,
    const std::filesystem::path& resources, const std::filesystem::path& request,
    const std::filesystem::path& options, const std::filesystem::path& output);
// Bounded complete sampling plus native decoder/media qualification.
// Legacy v1 uses F32; v2 consumes an admitted precision-policy artifact.
// Does not grant unrestricted run admission or change numerical qualification.
Json check_local_sampling_video(const std::filesystem::path& manifest,
    const std::filesystem::path& resources, const std::filesystem::path& request,
    const std::filesystem::path& options, const std::filesystem::path& output);
} // namespace vrhino::product
