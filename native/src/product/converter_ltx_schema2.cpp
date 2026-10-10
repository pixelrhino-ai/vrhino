#include "../architectures/ltx_declaration.h"
#include "converter_ltx_internal.h"
#include "vrhino/error.h"
#include "vrhino/product/converter.h"
#include "vrhino/product/package_preflight.h"
#include "vrhino/product/program_declaration.h"
#include "vrhino/product/token_grid_geometry.h"
#include "vrhino/product/vrm_verification.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>

namespace v = vrhino;
namespace p = vrhino::product;
namespace fs = std::filesystem;
namespace vrhino::product {
namespace {
v::Json read(const fs::path &f) {
  std::ifstream in(f);
  v::require(in.good(), "Missing converter declaration");
  return v::Json::parse(std::string(std::istreambuf_iterator<char>(in), {}));
}
void put(const fs::path &f, const v::Json &j) {
  std::ofstream out(f);
  out << j.serialize() << '\n';
  v::require(out.good(), "Converter write failure");
}
v::Json str(const std::string &s) { return v::Json(s); }
v::Json num(int64_t n) { return v::Json(n); }
} // namespace
VrmWriteResult convert_ltx_schema2_package(const fs::path &source_directory,
                                           const fs::path &spec_root,
                                           const fs::path &output_directory,
                                           const ImportOptions &options) {
  const fs::path source = fs::canonical(source_directory),
                 spec = fs::canonical(spec_root / "ltx_schema2_v1"),
                 out = fs::absolute(output_directory);
  v::require(!fs::exists(out), "Converter output must be fresh");
  const auto work = [&](uint64_t) {
    if (options.cancellation_requested && options.cancellation_requested())
      throw ModelPackageError(ModelPackageErrorCode::Cancelled,
                              "LTX Schema2 conversion interrupted");
  };
  work(0);
  v::require(
      p::sha256_file(spec / "direct-source-contract.json", work) ==
          "17c5e0fb064d5edb88dfb80b78bd6fbde4f3fd366b14e1ece649042ff6846bd5",
      "Frozen direct source contract drift");
  v::require(
      p::sha256_file(spec_root / "ltx_v0_9_1/tensor-mapping.tsv", work) ==
          "b0ae4b597e617a287d62a66546a395f43e83631b9602b43c9726afec2a10f53b",
      "Frozen LTX tensor mapping drift");
  const auto contract = read(spec / "direct-source-contract.json");
  v::require(contract.at("schema").string() == "vrhino.ltx-schema2-source.v2" &&
                 contract.at("source_repository").string() ==
                     "Lightricks/LTX-Video" &&
                 contract.at("source_revision").string() ==
                     "8984fa25007f376c1a299016d0957a37a2f797bb",
             "Unsupported frozen source identity");
  // Reserve enough for a standalone Package even when resource hard links
  // require the existing cross-filesystem copy fallback.
  uint64_t required = 5718000000ULL;
  for (const auto &artifact : contract.at("artifacts").array())
    if (artifact.at("id").string() != "checkpoint") {
      const auto size = artifact.at("size").integer();
      v::require(size > 0 && uint64_t(size) <= UINT64_MAX - required,
                 "Invalid source artifact size");
      required += uint64_t(size);
    }
  require_conversion_disk_space(required,
                                options.available_space_override.value_or(
                                    fs::space(out.parent_path()).available),
                                64ULL * 1024 * 1024, out.parent_path());
  VrmWriteResult result;
  bool output_created = false;
  try {
    v::Json::Object source_paths;
    for (const auto &a : contract.at("artifacts").array()) {
      const fs::path relative = a.at("source_path").string();
      v::require(relative.is_relative(), "Unsafe source path");
      for (const auto &part : relative)
        v::require(part != "..", "Unsafe source path");
      const auto path = checked_ltx_source(source, relative);
      v::require(fs::file_size(path) == uint64_t(a.at("size").integer()) &&
                     p::sha256_file(path, work) == a.at("sha256").string(),
                 "Frozen source artifact drift: " + a.at("id").string());
      source_paths.emplace(a.at("id").string(), str(path.string()));
    }
    v::require(source_paths.size() == 6 &&
                   source_paths.contains("checkpoint") &&
                   source_paths.contains("t5-index") &&
                   source_paths.contains("t5-shard-1") &&
                   source_paths.contains("t5-shard-2") &&
                   source_paths.contains("tokenizer") &&
                   source_paths.contains("license"),
               "Frozen source inventory mismatch");
    const auto checkpoint =
        checked_ltx_source(source, "ltx-video-2b-v0.9.1.safetensors");
    SafeTensorReader reader(checkpoint);
    const auto mappings =
        load_ltx_source_mappings(spec_root / "ltx_v0_9_1/tensor-mapping.tsv");
    validate_ltx_source_mappings(reader, mappings);
    const auto programs = read(spec / "programs.json");
    const std::string reference = "review/ltx-video-v0.9.1-schema2:0.2.0";
    const auto denoiser =
        v::Json(ltx_source_bindings(mappings, "model.diffusion_model."));
    const auto decoder = v::Json(ltx_source_bindings(mappings, "vae."));
    auto j = v::Json::parse(
        R"({"schema_version":2,"required_capabilities":["binding_catalog.v1"],"graphs":[{"id":0,"declaration":{}}],"bindings":[{"id":0,"graph":0,"parameters":{}}],"instances":[{"id":0,"graph":0,"binding":0}]})");
    auto go = j.object();
    auto gs = j.at("graphs").array();
    auto g = gs[0].object();
    g["declaration"] = v::ltx_internal::token_flow_declaration(v::DType::BF16);
    gs[0] = v::Json(g);
    go["graphs"] = v::Json(gs);
    auto bs = j.at("bindings").array();
    auto binding = bs[0].object();
    binding["parameters"] = denoiser;
    bs[0] = v::Json(binding);
    go["bindings"] = v::Json(bs);
    j = v::Json(go);
    const auto catalog = v::PackageDeclaration::parse(j);
    (void)p::admit_program_declaration(programs, catalog);
    auto component = v::Json::parse(
        R"JSON({"schema_version":1,"implementation_id":"dit_flow.ltx_v0_9_1.vae_decoder.v1","latent_contract":{"channels":128,"layout":"BCTHW","sampling_layout":"BLC","spatial_scale":32,"temporal_decode":"1+8*(F-1)"},"runtime_tensor_bindings":{}})JSON");
    auto co = component.object();
    co["runtime_tensor_bindings"] = decoder;
    component = v::Json(co);
    const auto provenance = v::Json(v::Json::Object{
        {"actual_repository", contract.at("source_repository")},
        {"actual_revision", contract.at("source_revision")},
        {"manifest_sha256",
         str(p::sha256_file(spec / "direct-source-contract.json"))},
        {"checkpoint_sha256", str("a23200896c5eddf215c7cb9517820c5763a2b054eb62"
                                  "ba86cbce6b871a4577e3")},
        {"tensor_mapping_sha256",
         str(p::sha256_file(spec_root / "ltx_v0_9_1/tensor-mapping.tsv"))},
        {"converter_version", str("ltx-schema2-native-source-v1")},
        {"direct_safetensors_source", v::Json(true)},
        {"identity_tensor_copy", v::Json(true)}});
    const auto meta = v::Json(v::Json::Object{
        {"architecture", str("ltx_v0_9_1")},
        {"product", str(reference)},
        {"programs", programs},
        {"conversion_provenance", provenance},
        {"shared_components",
         v::Json(v::Json::Object{
             {"decoder", component},
             {"text_encoder_declaration", str("conditioning.json")},
             {"tokenizer_resource", str("tokenizer/spiece.model")},
             {"text_encoder_resource",
              v::Json(v::Json::Object{
                  {"model-00001-of-00002.safetensors",
                   str("conditioning/model-00001-of-00002.safetensors")},
                  {"model-00002-of-00002.safetensors",
                   str("conditioning/"
                       "model-00002-of-00002.safetensors")}})}})}});
    v::require(fs::create_directory(out), "Converter output must be fresh");
    output_created = true;
    result = p::write_vrm_streaming(out / "model.vrm", "dit-flow", "ltx_v0_9_1",
                                    meta, j, reader, mappings,
                                    options.cancellation_requested, work);
    auto converted =
        std::make_shared<v::VrmModel>((out / "model.vrm").string());
    std::vector<uint8_t> buffer(8 * 1024 * 1024);
    for (const auto &mapping : mappings) {
      const auto &t = converted->tensor(mapping.destination_name);
      const auto &descriptor = reader.tensors().at(mapping.source_name);
      v::require(t.shape() == descriptor.shape && t.dtype() == descriptor.dtype,
                 "Tensor dtype/shape changed");
      for (uint64_t offset = 0; offset < t.bytes();) {
        const auto bytes =
            std::min<uint64_t>(buffer.size(), t.bytes() - offset);
        work(bytes);
        reader.read_tensor(descriptor, offset, buffer.data(), bytes);
        v::require(std::memcmp(buffer.data(),
                               static_cast<const uint8_t *>(t.data()) + offset,
                               bytes) == 0,
                   "Tensor payload changed");
        offset += bytes;
      }
    }
    (void)v::create_architecture(std::shared_ptr<const v::VrmModel>(converted));
    put(out / "metadata.json", meta);
    put(out / "graph.json", j);
    for (const auto *f :
         {"programs.json", "conditioning.json", "precision.json",
          "request-wiring.json", "profile.json", "product-run.json"})
      fs::copy_file(spec / f, out / f);
    fs::copy_file(spec / "direct-source-contract.json",
                  out / "source-contract.json");
    v::Json::Array artifacts;
    v::Json::Object locations;
    const auto add = [&](std::string id, std::string role, std::string path,
                         const fs::path &physical) {
      artifacts.emplace_back(
          v::Json::Object{{"id", str(id)},
                          {"role", str(role)},
                          {"path", str(path)},
                          {"size", num(fs::file_size(physical))},
                          {"sha256", str(p::sha256_file(physical, work))},
                          {"required", v::Json(true)}});
      const auto destination = out / path;
      if (fs::absolute(physical) != fs::absolute(destination)) {
        fs::create_directories(destination.parent_path());
        std::error_code error;
        fs::create_hard_link(physical, destination, error);
        if (error)
          fs::copy_file(physical, destination);
      }
      locations.emplace(id, str(path));
    };
    for (const auto &row : std::vector<std::vector<std::string>>{
             {"runtime", "runtime.vrm", "model.vrm"},
             {"metadata", "declaration.metadata", "metadata.json"},
             {"graph", "declaration.graph", "graph.json"},
             {"programs", "declaration.programs", "programs.json"},
             {"conditioning-declaration", "conditioning.graph",
              "conditioning.json"},
             {"precision", "precision.policy", "precision.json"},
             {"request-wiring", "declaration.request", "request-wiring.json"},
             {"profile", "execution.profile", "profile.json"},
             {"product-run", "product.execution", "product-run.json"},
             {"source-manifest", "source.manifest", "source-contract.json"}})
      add(row[0], row[1], row[2], out / row[2]);
    add("t5-index", "conditioning.weights.index",
        "conditioning/model.safetensors.index.json",
        source_paths.at("t5-index").string());
    for (int i = 1; i <= 2; ++i) {
      const auto id = "t5-shard-" + std::to_string(i);
      add(id, "conditioning.weights.shard",
          "conditioning/model-0000" + std::to_string(i) +
              "-of-00002.safetensors",
          source_paths.at(id).string());
    }
    add("tokenizer", "tokenizer.model", "tokenizer/spiece.model",
        source_paths.at("tokenizer").string());
    add("license", "legal.license", "legal/license.txt",
        source_paths.at("license").string());
    auto manifest = v::Json::parse(
        R"({"schema_version":2,"identity":{"namespace":"review","name":"ltx-video-v0.9.1-schema2","version":"0.2.0","architecture":"ltx_v0_9_1","publisher":"VRhino Review"},"compatibility":{"runtime_contract":"cuda-v1","vrm_schema":{"format_major":0,"format_minor":1,"metadata_schema":1}},"product":{"family":"text_to_video","status":"alpha_unqualified","public_distribution":"local_test_only","input_schema":{"schema":"vrhino.product.input-schema.v1","inputs":[{"name":"prompt","type":"text","required":true,"validation":{"min_length":1}}],"parameters":[{"name":"seed","type":"integer","required":false,"default":5703,"validation":{"minimum":0,"maximum":"18446744073709551615"}}],"outputs":[{"name":"output","type":"media.mp4","required":false,"default":"output.mp4","validation":{"parent_creatable_and_writable":true}}]},"frozen_profile":{"output":{"width":704,"height":480,"frames":121,"fps":{"numerator":25,"denominator":1},"duration":"fixed","audio":"none"},"sampling":{"program_artifact":"programs"}},"execution_artifact":"product-run"},"artifacts":[],"entrypoint":{"runtime_artifact":"runtime","components":[{"id":"positive-conditioning","kind":"conditioning.text_encoder","artifacts":["conditioning-declaration","t5-index","t5-shard-1","t5-shard-2","tokenizer"]},{"id":"negative-conditioning","kind":"conditioning.text_encoder","artifacts":["conditioning-declaration","t5-index","t5-shard-1","t5-shard-2","tokenizer"]}],"default_preset":"default"},"defaults":{"default_preset":"default","presets":{"default":{"profile_artifact":"profile"}}},"hardware":{"presets":{"default":{"minimum_vram_bytes":null,"recommended_vram_bytes":25757220864}}},"source":{"repository":"Lightricks/LTX-Video","revision":"8984fa25007f376c1a299016d0957a37a2f797bb","converter_version":"ltx-schema2-native-source-v1"},"license":{"identifier":"LicenseRef-LTX-Video-0.9.1-RAIL-M","artifact":"license","upstream_notice":"Frozen LTX 0.9.1 source license; review package only"},"admission":{"graph_artifact":"graph","metadata_artifact":"metadata","programs_artifact":"programs","source_manifest_artifact":"source-manifest","required_capabilities":["binding_catalog.v1","execution.per_step.v1","sampling.flow_euler_cfg.v1"],"structural_only":true,"execution_eligibility":"alpha_unqualified","resources":{"conditioning_declaration":"conditioning-declaration","conditioning_index":"t5-index","conditioning_weights":{"model-00001-of-00002.safetensors":"t5-shard-1","model-00002-of-00002.safetensors":"t5-shard-2"},"tokenizer":"tokenizer"},"request_artifact":"request-wiring"}})");
    auto mo = manifest.object();
    mo["artifacts"] = v::Json(artifacts);
    manifest = v::Json(mo);
    put(out / "vrhino-model.json", manifest);
    put(out / "local.json",
        v::Json(v::Json::Object{{"schema", str("vrhino.local-resources.v1")},
                                {"resources", v::Json(locations)}}));
    (void)p::load_model_package_manifest(out / "vrhino-model.json");
    work(0);
    (void)p::preflight_local_product(out / "vrhino-model.json",
                                     out / "local.json");
    return result;
  } catch (...) {
    std::error_code ignored;
    if (output_created)
      fs::remove_all(out, ignored);
    throw;
  }
}
} // namespace vrhino::product
