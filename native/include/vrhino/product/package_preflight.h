#pragma once
#include "vrhino/architecture.h"
#include "vrhino/product/model_package.h"
#include "vrhino/product/safetensors.h"

namespace vrhino::product {
// This object owns verified backing and the admitted adapter, not a device or
// numerical session. Structural eligibility never grants numerical qualification.
struct AdmittedLocalProduct {
    ResolvedRunnableModel resources;
    std::shared_ptr<const VrmModel> model;
    std::unique_ptr<SafeTensorAsset> conditioning;
    std::unique_ptr<Architecture> architecture;
    SamplingProgram sampling;
    Json evidence;
};
// Local map schema: {schema:"vrhino.local-resources.v1", resources:{id:path}}.
// Physical paths belong only to this machine-local document. No downloads/CAS copies.
ResolvedRunnableModel resolve_local_product(const std::filesystem::path& manifest,
                                            const std::filesystem::path& local_resources);
// Full streaming SHA256 verification is mandatory; no trust/skip-hash switch.
// Accepts either cache-resolved or machine-local resources. Resolution alone is
// not admission: revalidate the manifest, inventory and payload identities here.
AdmittedLocalProduct preflight_resolved_product(ResolvedRunnableModel resources);
AdmittedLocalProduct preflight_local_product(const std::filesystem::path& manifest,
                                            const std::filesystem::path& local_resources);
} // namespace vrhino::product
