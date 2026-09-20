#include "vrhino/product/package_preflight.h"
#include "vrhino/product/program_declaration.h"
#include "vrhino/product/vrm_verification.h"
#include "vrhino/error.h"
#include "vrhino/tensor_util.h"
#include <fstream>
#include <set>
#include <limits>

namespace vrhino::product {
namespace fs = std::filesystem;
namespace {
Json read_json(const fs::path& path, size_t limit = 32*1024*1024) {
    require(fs::is_regular_file(path) && fs::file_size(path) <= limit, "Missing/oversized JSON resource");
    std::ifstream f(path, std::ios::binary);
    require(f.good(), "Cannot open declaration");
    std::string text((std::istreambuf_iterator<char>(f)), {});
    require(text.size() <= limit, "Declaration grew beyond limit");
    JsonParseLimits limits;
    limits.maximum_document_bytes = limit; limits.maximum_container_entries = 300000;
    limits.maximum_total_values = 2000000;
    return Json::parse(text, limits);
}
void reference_weights(const Json& j, const WeightMap& weights, std::set<std::string>& names) {
    if (j.is_object()) for (const auto& [key, value] : j.object()) {
        if (key == "weight" || key == "bias") { (void)weights.at(value.string()); names.insert(value.string()); }
        else reference_weights(value, weights, names);
    }
    else if (j.is_array()) for (const auto& v : j.array()) reference_weights(v, weights, names);
}
}
ResolvedRunnableModel resolve_local_product(const fs::path& manifest_path, const fs::path& local) {
    ResolvedRunnableModel result;
    result.manifest = load_model_package_manifest(manifest_path);
    require(result.manifest.schema_version == 2, "Local catalog resolver requires product schema2");
    result.manifest_path = fs::absolute(manifest_path);
    const auto j = read_json(local, 1024*1024);
    require(j.object().size() == 2 && j.at("schema").string() == "vrhino.local-resources.v1", "Malformed local resource declaration");
    const auto& map = j.at("resources").object();
    require(map.size() == result.manifest.artifacts.size(), "Local resource inventory mismatch");
    for (const auto& a : result.manifest.artifacts) {
        const auto it = map.find(a.id);
        require(it != map.end() && !it->second.string().empty(), "Missing local resource ID");
        fs::path path(it->second.string());
        require(it->second.string().find('\0') == std::string::npos, "NUL in local resource path");
        if (path.is_relative()) path = fs::absolute(local).parent_path()/path;
        require(fs::is_regular_file(path), "External resource missing: " + a.id);
        require(fs::file_size(path) == a.size, "External resource size mismatch: " + a.id);
        result.artifacts.emplace(a.id, ResolvedArtifact{a, fs::canonical(path)});
    }
    result.runtime_model_path = result.artifacts.at(result.manifest.runtime_artifact_id).path;
    return result;
}
AdmittedLocalProduct preflight_local_product(const fs::path& manifest, const fs::path& local) {
    return preflight_resolved_product(resolve_local_product(manifest, local));
}
AdmittedLocalProduct preflight_resolved_product(ResolvedRunnableModel resources) {
    // Do not trust mutable parsed fields or an earlier cache lookup as authority.
    // A changed manifest requires a new resolution, not silently new semantics.
    auto manifest = load_model_package_manifest(resources.manifest_path);
    require(manifest.schema_version == 2, "Catalog product admission requires schema2");
    require(manifest.raw_json == resources.manifest.raw_json,
            "Product manifest changed after resource resolution");
    require(resources.artifacts.size() == manifest.artifacts.size(),
            "Resolved resource inventory mismatch");
    for (const auto& expected : manifest.artifacts) {
        const auto it = resources.artifacts.find(expected.id);
        require(it != resources.artifacts.end(), "Missing resolved resource: " + expected.id);
        const auto& actual = it->second.declaration;
        require(actual.id == expected.id && actual.role == expected.role &&
                actual.relative_path == expected.relative_path && actual.size == expected.size &&
                actual.sha256 == expected.sha256 && actual.required == expected.required,
                "Resolved resource declaration mismatch: " + expected.id);
        require(fs::is_regular_file(it->second.path) && fs::file_size(it->second.path) == expected.size,
                "Resolved resource missing or size mismatch: " + expected.id);
    }
    require(fs::is_regular_file(resources.runtime_model_path) &&
            fs::equivalent(resources.runtime_model_path, resources.artifacts.at(manifest.runtime_artifact_id).path),
            "Resolved runtime resource identity mismatch");
    AdmittedLocalProduct out;
    out.resources = std::move(resources);
    out.resources.manifest = std::move(manifest);
    const auto& m = out.resources.manifest;
    const auto& artifacts = out.resources.artifacts;
    const auto declaration = Json::parse(m.raw_json);
    const auto& admission = declaration.at("admission");
    uint64_t total = 0;
    std::map<std::string, fs::file_time_type> observed_stamps;
    // Verify small declarations and external resources first. No tensor payload
    // is materialized; hashing has a bounded buffer and the VRM loader mmaps.
    for (const auto& [id, a] : artifacts) {
        require(a.declaration.size <= std::numeric_limits<uint64_t>::max()-total, "Resource size overflow");
        total += a.declaration.size;
        if (id == m.runtime_artifact_id) continue;
        const auto stamp = fs::last_write_time(a.path);
        observed_stamps.emplace(id, stamp);
        require(sha256_file(a.path) == a.declaration.sha256, "External resource SHA256 mismatch: " + id);
        require(fs::file_size(a.path) == a.declaration.size && fs::last_write_time(a.path) == stamp,
                "External resource changed during verification: " + id);
    }
    const auto& runtime = artifacts.at(m.runtime_artifact_id);
    out.model = load_verified_vrm_component(runtime.path,
        {m.runtime_artifact_id, m.identity.architecture, runtime.declaration.size, runtime.declaration.sha256});
    const auto read_artifact = [&](const std::string& id) { return read_json(artifacts.at(id).path); };
    const auto graph = read_artifact(admission.at("graph_artifact").string());
    const auto metadata = read_artifact(admission.at("metadata_artifact").string());
    const auto programs = read_artifact(admission.at("programs_artifact").string());
    require(graph.serialize() == out.model->graph().serialize(), "Graph/package identity mismatch");
    require(metadata.serialize() == out.model->metadata().serialize(), "Metadata/package identity mismatch");
    require(metadata.at("product").string() == m.identity.reference(), "Product identity mismatch");
    require(programs.serialize() == metadata.at("programs").serialize(), "Program/package identity mismatch");
    const auto& provenance = metadata.at("conversion_provenance");
    require(provenance.at("actual_repository").string() == m.source_repository &&
            provenance.at("actual_revision").string() == m.source_revision, "Source provenance mismatch");
    require(provenance.at("manifest_sha256").string() == artifacts.at(
        admission.at("source_manifest_artifact").string()).declaration.sha256, "Source manifest identity mismatch");
    auto catalog = PackageDeclaration::parse(graph);
    auto lowered_programs = admit_program_declaration(programs, catalog);
    (void)lowered_programs;
    out.architecture = create_architecture(out.model); // all canonical slots admitted here
    const auto& preset = declaration.at("defaults").at("presets").at(m.default_preset);
    const auto profile = read_artifact(preset.at("profile_artifact").string());
    require(profile.object().size() == 2 && profile.at("latent_shape").array().size() == 5,
            "Malformed structural run profile");
    auto shape = Tensor::host({5}, DType::I64);
    for (size_t i=0; i<5; ++i) shape.data_as<int64_t>()[i] = profile.at("latent_shape").array()[i].integer();
    TensorBundle request{{"seed", scalar_i64(profile.at("seed").integer())}, {"latent_shape", shape}};
    out.sampling = out.architecture->create_program(request);
    require(out.sampling.steps == static_cast<int>(programs.at("execution").at("instance_ids").array().size()),
            "Adapter lost declared program");
    const auto& roles = admission.at("resources");
    const auto resource = [&](const char* role) -> const ResolvedArtifact& { return artifacts.at(roles.at(role).string()); };
    const auto& shared = metadata.at("shared_components");
    for (const auto& pair : {std::pair{"text_encoder_declaration", "conditioning_declaration"},
                            std::pair{"text_encoder_resource", "conditioning_weights"},
                            std::pair{"tokenizer_resource", "tokenizer"}})
        require(shared.at(pair.first).string() == resource(pair.second).declaration.relative_path.generic_string(),
                "Shared resource logical identity mismatch");
    const auto index = read_json(resource("conditioning_index").path);
    const auto& weight_resource = resource("conditioning_weights");
    out.conditioning = std::make_unique<SafeTensorAsset>(SafeTensorAsset::indexed(
        resource("conditioning_index").path, {{weight_resource.declaration.relative_path.generic_string(), weight_resource.path}}));
    const auto conditioning = read_json(resource("conditioning_declaration").path);
    std::set<std::string> names;
    reference_weights(conditioning, out.conditioning->weights(), names);
    require(!names.empty() && names.size() == index.at("weight_map").object().size(), "Incomplete conditioning tensor references");
    const auto tokenizer = read_json(resource("tokenizer").path);
    require(tokenizer.at("model").is_object(), "Invalid tokenizer declaration");
    // Metadata inspection is inside the same qualification interval. The
    // declaration is not a reusable permission to run later-mutated resources.
    for (const auto& [id, stamp] : observed_stamps) {
        const auto& a = artifacts.at(id);
        require(fs::file_size(a.path) == a.declaration.size && fs::last_write_time(a.path) == stamp,
                "Resource changed during admission: " + id);
    }
    ResourceEstimate estimate;
    estimate.source_backing = checked_memory_add(out.model->file_size(), out.conditioning->mapped_bytes(), "source backing");
    // This is a host-only structural request, not an inference residency budget.
    ResourceAdmissionRequest resource_request{estimate, {1, estimate.host_peak()}};
    resource_request.validate();
    out.evidence = Json(Json::Object{
        {"structurally_runnable", Json(true)}, {"numerically_qualified", Json(false)},
        {"numerical_status", Json(std::string("HOLD"))}, {"route", Json(std::string("production_default_structural"))},
        {"graphs", Json(int64_t(catalog.graphs().size()))}, {"bindings", Json(int64_t(catalog.bindings().size()))},
        {"instances", Json(int64_t(catalog.instances().size()))}, {"sampling_steps", Json(int64_t(out.sampling.steps))},
        {"resource_bytes", Json(std::to_string(total))}, {"vrm_bytes", Json(std::to_string(out.model->file_size()))},
        {"vrm_sha256", Json(runtime.declaration.sha256)}, {"vrm_path", Json(runtime.path.string())},
        {"host_only_resource_admission", Json(true)}, {"device_upload_bytes", Json(int64_t(0))}, {"total_device_memory_bound_guaranteed", Json(false)},
        {"load_strategy", Json(std::string("verified_descriptor_read_only_mmap"))}});
    return out;
}
} // namespace vrhino::product
