#include "../src/architectures/ltx_declaration.h"
#include "vrhino/error.h"
#include "vrhino/product/converter.h"
#include "vrhino/product/package_preflight.h"
#include "vrhino/product/program_declaration.h"
#include "vrhino/product/token_grid_geometry.h"
#include "vrhino/product/vrm_verification.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>

namespace v = vrhino;
namespace p = vrhino::product;
namespace fs = std::filesystem;
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
class MappedSource final : public p::TensorSource {
  public:
    explicit MappedSource(const v::VrmModel &m) : model(m) {
        for (const auto &[name, r] : m.tensors())
            descriptors.emplace(name, p::SourceTensorDescriptor{name, r.tensor.dtype(), "",
                                                                r.tensor.shape(), 0, 0,
                                                                r.tensor.bytes()});
    }
    const std::map<std::string, p::SourceTensorDescriptor> &tensors() const noexcept override {
        return descriptors;
    }
    void read_tensor(const p::SourceTensorDescriptor &d, uint64_t offset, void *dest,
                     size_t bytes) const override {
        const auto &t = model.tensor(d.name);
        v::require(offset <= t.bytes() && bytes <= t.bytes() - offset, "VRM copy range");
        std::memcpy(dest, static_cast<const uint8_t *>(t.data()) + offset, bytes);
    }

  private:
    const v::VrmModel &model;
    std::map<std::string, p::SourceTensorDescriptor> descriptors;
};
int main(int argc, char **argv) {
    try {
        v::require(argc == 5,
                   "usage: ltx-schema2-convert OLD_VRM SOURCE_DIR SPEC_DIR NEW_OUTPUT_DIR");
        const fs::path source = fs::canonical(argv[2]), spec = fs::canonical(argv[3]),
                       out = fs::absolute(argv[4]);
        v::require(!fs::exists(out), "Converter output must be fresh");
        const auto contract = read(spec / "source-contract.json");
        v::require(contract.at("schema").string() == "vrhino.ltx-schema2-source.v1" &&
                       contract.at("legacy_vrm_sha256").string() ==
                           "267a95330f48dbe2134220e6116c60cddddf54f7548a661e20b18531fb70fa7d" &&
                       contract.at("legacy_vrm_size").integer() == 5717174080LL,
                   "Unsupported frozen source identity");
        std::shared_ptr<const v::VrmModel> old(
            p::load_verified_vrm_component(argv[1], {"legacy-source", "ltx_v0_9_1", 5717174080ULL,
                                                     contract.at("legacy_vrm_sha256").string()}));
        v::Json::Object source_paths;
        v::Json::Array source_records;
        for (const auto &a : contract.at("artifacts").array()) {
            const fs::path relative = a.at("source_path").string();
            v::require(relative.is_relative(), "Unsafe source path");
            for (const auto &part : relative)
                v::require(part != "..", "Unsafe source path");
            const auto path = fs::canonical(source / relative);
            v::require(fs::file_size(path) == uint64_t(a.at("size").integer()) &&
                           p::sha256_file(path) == a.at("sha256").string(),
                       "Frozen source artifact drift: " + a.at("id").string());
            source_paths.emplace(a.at("id").string(), str(path.string()));
            source_records.push_back(a);
        }
        const auto programs = read(spec / "programs.json");
        const std::string reference = "review/ltx-video-v0.9.1-schema2:0.1.0";
        auto denoiser = old->graph().at("architecture_graph").at("runtime_tensor_bindings");
        auto decoder =
            old->graph().at("component_graphs").array().at(0).at("runtime_tensor_bindings");
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
        const auto def = v::ltx_internal::lower_token_flow(gs[0].at("declaration"));
        std::map<std::string, const v::Tensor *> roles;
        for (const auto &[role, name] : denoiser.object())
            roles.emplace(role, &old->tensor(name.string()));
        (void)v::AdmittedArchitectureBinding::admit(
            def.parameters, roles, v::BorrowedBindingLifetime::ExplicitOwners, {old});
        auto component = v::Json::parse(
            R"JSON({"schema_version":1,"implementation_id":"dit_flow.ltx_v0_9_1.vae_decoder.v1","latent_contract":{"channels":128,"layout":"BCTHW","sampling_layout":"BLC","spatial_scale":32,"temporal_decode":"1+8*(F-1)"},"runtime_tensor_bindings":{}})JSON");
        auto co = component.object();
        co["runtime_tensor_bindings"] = decoder;
        component = v::Json(co);
        const auto provenance = v::Json(
            v::Json::Object{{"actual_repository", contract.at("source_repository")},
                            {"actual_revision", contract.at("source_revision")},
                            {"manifest_sha256", str(p::sha256_file(spec / "source-contract.json"))},
                            {"legacy_vrm_sha256", contract.at("legacy_vrm_sha256")},
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
                       str("conditioning/model-00002-of-00002.safetensors")}})}})}});
        fs::create_directories(out);
        MappedSource bytes(*old);
        std::vector<p::TensorMapping> maps;
        for (const auto &[name, r] : old->tensors())
            maps.push_back({name, r.tensor.dtype(), r.tensor.shape(), "identity_bytes", name,
                            r.tensor.dtype(), r.tensor.shape(), "", ""});
        std::cout << "Writing identity tensor payload: " << maps.size() << " tensors\n"
                  << std::flush;
        p::write_vrm_streaming(out / "model.vrm", "dit-flow", "ltx_v0_9_1", meta, j, bytes, maps);
        auto converted = std::make_shared<v::VrmModel>((out / "model.vrm").string());
        for (const auto &[name, r] : old->tensors()) {
            const auto &t = converted->tensor(name);
            v::require(t.shape() == r.tensor.shape() && t.dtype() == r.tensor.dtype() &&
                           std::memcmp(t.data(), r.tensor.data(), t.bytes()) == 0,
                       "Tensor payload changed");
        }
        (void)v::create_architecture(std::shared_ptr<const v::VrmModel>(converted));
        put(out / "metadata.json", meta);
        put(out / "graph.json", j);
        for (const auto *f :
             {"programs.json", "conditioning.json", "precision.json", "request-wiring.json",
              "profile.json", "product-run.json", "source-contract.json"})
            fs::copy_file(spec / f, out / f);
        v::Json::Array artifacts;
        v::Json::Object locations;
        const auto add = [&](std::string id, std::string role, std::string path,
                             const fs::path &physical) {
            artifacts.emplace_back(v::Json::Object{{"id", str(id)},
                                                   {"role", str(role)},
                                                   {"path", str(path)},
                                                   {"size", num(fs::file_size(physical))},
                                                   {"sha256", str(p::sha256_file(physical))},
                                                   {"required", v::Json(true)}});
            locations.emplace(id, str(fs::canonical(physical).string()));
        };
        for (const auto &row : std::vector<std::vector<std::string>>{
                 {"runtime", "runtime.vrm", "model.vrm"},
                 {"metadata", "declaration.metadata", "metadata.json"},
                 {"graph", "declaration.graph", "graph.json"},
                 {"programs", "declaration.programs", "programs.json"},
                 {"conditioning-declaration", "conditioning.graph", "conditioning.json"},
                 {"precision", "precision.policy", "precision.json"},
                 {"request-wiring", "declaration.request", "request-wiring.json"},
                 {"profile", "execution.profile", "profile.json"},
                 {"product-run", "product.execution", "product-run.json"},
                 {"source-manifest", "source.manifest", "source-contract.json"}})
            add(row[0], row[1], row[2], out / row[2]);
        add("t5-index", "conditioning.weights.index", "conditioning/model.safetensors.index.json",
            source_paths.at("t5-index").string());
        for (int i = 1; i <= 2; ++i) {
            const auto id = "t5-shard-" + std::to_string(i);
            add(id, "conditioning.weights.shard",
                "conditioning/model-0000" + std::to_string(i) + "-of-00002.safetensors",
                source_paths.at(id).string());
        }
        add("tokenizer", "tokenizer.model", "tokenizer/spiece.model",
            source_paths.at("tokenizer").string());
        add("license", "legal.license", "legal/license.txt", source_paths.at("license").string());
        auto manifest = v::Json::parse(
            R"({"schema_version":2,"identity":{"namespace":"review","name":"ltx-video-v0.9.1-schema2","version":"0.1.0","architecture":"ltx_v0_9_1","publisher":"VRhino Review"},"compatibility":{"runtime_contract":"cuda-v1","vrm_schema":{"format_major":0,"format_minor":1,"metadata_schema":1}},"product":{"family":"text_to_video","status":"alpha_unqualified","public_distribution":"supported","input_schema":{"schema":"vrhino.product.input-schema.v1","inputs":[{"name":"prompt","type":"text","required":true,"validation":{"min_length":1}}],"parameters":[{"name":"seed","type":"integer","required":false,"default":5703,"validation":{"minimum":0,"maximum":"18446744073709551615"}}],"outputs":[{"name":"output","type":"media.mp4","required":false,"default":"output.mp4","validation":{"parent_creatable_and_writable":true}}]},"frozen_profile":{"output":{"width":704,"height":480,"frames":121,"fps":{"numerator":25,"denominator":1},"duration":"fixed","audio":"none"},"sampling":{"program_artifact":"programs"}},"execution_artifact":"product-run"},"artifacts":[],"entrypoint":{"runtime_artifact":"runtime","components":[{"id":"positive-conditioning","kind":"conditioning.text_encoder","artifacts":["conditioning-declaration","t5-index","t5-shard-1","t5-shard-2","tokenizer"]},{"id":"negative-conditioning","kind":"conditioning.text_encoder","artifacts":["conditioning-declaration","t5-index","t5-shard-1","t5-shard-2","tokenizer"]}],"default_preset":"default"},"defaults":{"default_preset":"default","presets":{"default":{"profile_artifact":"profile"}}},"hardware":{"presets":{"default":{"minimum_vram_bytes":null,"recommended_vram_bytes":25757220864}}},"source":{"repository":"Lightricks/LTX-Video","revision":"8984fa25007f376c1a299016d0957a37a2f797bb","converter_version":"ltx-schema2-review-v1"},"license":{"identifier":"LicenseRef-LTX-Video-0.9.1-RAIL-M","artifact":"license","upstream_notice":"Frozen LTX 0.9.1 source license; review package only"},"admission":{"graph_artifact":"graph","metadata_artifact":"metadata","programs_artifact":"programs","source_manifest_artifact":"source-manifest","required_capabilities":["binding_catalog.v1","execution.per_step.v1","sampling.flow_euler_cfg.v1"],"structural_only":true,"execution_eligibility":"alpha_unqualified","resources":{"conditioning_declaration":"conditioning-declaration","conditioning_index":"t5-index","conditioning_weights":{"model-00001-of-00002.safetensors":"t5-shard-1","model-00002-of-00002.safetensors":"t5-shard-2"},"tokenizer":"tokenizer"},"request_artifact":"request-wiring"}})");
        auto mo = manifest.object();
        mo["artifacts"] = v::Json(artifacts);
        manifest = v::Json(mo);
        put(out / "vrhino-model.json", manifest);
        put(out / "local.json",
            v::Json(v::Json::Object{{"schema", str("vrhino.local-resources.v1")},
                                    {"resources", v::Json(locations)}}));
        (void)p::load_model_package_manifest(out / "vrhino-model.json");
        std::cout << "LTX_SCHEMA2_CONVERSION=PASS tensor_copy_bitwise=PASS slots=715 "
                     "decoder_slots=297 Product qualification=HOLD\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
