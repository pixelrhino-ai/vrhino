#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "vrhino/product/converter.h"
#include "vrhino/product/model_package.h"
#include "vrhino/product/pull_orchestration.h"
#include "vrhino/product/source_acquisition.h"

namespace fs = std::filesystem;
namespace product = vrhino::product;

namespace {

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string read_bytes(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    require_test(static_cast<bool>(input), "cannot read fixture: " + path.string());
    return {std::istreambuf_iterator<char>(input), {}};
}

void write_bytes(const fs::path& path, const std::string& bytes) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    require_test(static_cast<bool>(output), "cannot write fixture: " + path.string());
}

struct SymlinkUnavailable : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Exercise the real package converters/finalizers without acquiring model weights.
// Only neural component declarations/payloads and unrelated static payloads are
// tiny fixtures. Both execution profiles and their declared hashes stay frozen.
struct ResourceFixture {
    bool musetalk;
    fs::path root, specs, family, source, selected;
    std::string reference;
    std::unique_ptr<product::LocalModelCache> cache;
    product::ImportOptions options;
    product::ArtifactDeclaration execution;

    ResourceFixture(const fs::path& fixture_root, const fs::path& real_specs,
                    const bool is_musetalk, const std::string& version)
        : musetalk(is_musetalk), root(fixture_root), specs(root / "specs"),
          family(specs / (musetalk ? "public_musetalk_v15" : "public_latentsync_16")),
          source(root / "source"),
          selected(family / (version == "1.0.0" ? fs::path{} : fs::path("successors") / version)
                   / product::kModelManifestName),
          reference(std::string("vrhino/") + (musetalk ? "musetalk-v1.5:" : "latentsync-1.6:") + version),
          cache(std::make_unique<product::LocalModelCache>(root / "cache")) {
        const fs::path real_family = real_specs / family.filename();
        for (const fs::path& relative : {fs::path("execution.json"),
                 fs::path("successors/1.0.1/execution.json")})
            write_bytes(family / relative, read_bytes(real_family / relative));
        fs::create_directories(source);
        auto document = vrhino::Json::parse(read_bytes(
            real_family / selected.lexically_relative(family))).object();
        std::set<std::string> neural;
        for (const auto& component : document.at("entrypoint").at("components").array())
            for (const auto& id : component.at("artifacts").array())
                neural.insert(id.string());
        std::map<std::string, fs::path> paths = {
            {"whisper-preprocessor", source / "whisper/preprocessor_config.json"},
            {"workflow-config", specs / (musetalk ? "musetalk_v15_workflow_v2/workflow.json"
                                                  : "latentsync_16_workflow/workflow.json")},
            {"third-party-notices", family / "THIRD_PARTY_NOTICES.txt"},
            {"license-apache", family / "licenses/Apache-2.0.txt"},
            {"license-mit", family / "licenses/MIT.txt"},
            {"license-openrail", family / "licenses/CreativeML-OpenRAIL-M.txt"},
            {"license-openrail-plus", family / "licenses/CreativeML-Open-RAIL++-M.txt"},
            {"precision-policy", family / "policy/bf16-heavy-consumer-fp32-state-v1.json"},
            {"mel-filters", source / "whisper/mel_filters.npz"},
            {"unet-config", source / "temporal_unet/stage2_512.yaml"},
            {"scheduler-config", source / "temporal_unet/scheduler_config.json"},
            {"fixed-mask", source / "workflow/mask.png"},
        };
        auto artifacts = document.at("artifacts").array();
        for (auto& value : artifacts) {
            auto artifact = value.object();
            const std::string id = artifact.at("id").string();
            if (id == "execution-profile") continue;
            const fs::path path = neural.count(id) ? root / "seeds" / id : paths.at(id);
            const std::string bytes = "bounded resource fixture: " + id + "\n";
            write_bytes(path, bytes);
            paths[id] = path;
            artifact["size"] = vrhino::Json(static_cast<int64_t>(bytes.size()));
            artifact["sha256"] = vrhino::Json(product::sha256_file(path));
            value = vrhino::Json(std::move(artifact));
        }
        document["artifacts"] = vrhino::Json(std::move(artifacts));
        write_bytes(selected, vrhino::Json(std::move(document)).serialize());
        const auto manifest = product::load_model_package_manifest(selected);
        for (const auto& artifact : manifest.artifacts) {
            if (neural.count(artifact.id))
                cache->admit_local_blob(paths.at(artifact.id), artifact);
            if (artifact.id == "execution-profile") execution = artifact;
        }
        const bool successor = version != "1.0.0";
        const uint64_t expected_size = musetalk ? (successor ? 348 : 347)
                                                : (successor ? 487 : 486);
        const std::string expected_hash = musetalk
            ? (successor ? "a5e1592fec66c5fb2a69b0e965e2dde8a7a4d45319535123712ca5c7b04d4a68"
                         : "c770d7ecbfe3c6cf21617176ebdf054a3c021c2bfc787f024ef8d919a5ca3cde")
            : (successor ? "f867e628bf4d2d588e265f480c30d0c60e9727d2e341a4300cc7c3081ce85b81"
                         : "aedba58531ffbba784f8d9ac7bc38666dae756c6f07e62616d49dbb6acbc7332");
        require_test(execution.size == expected_size && execution.sha256 == expected_hash,
                     "frozen execution profile declaration changed");
        options.converter_spec_root = specs;
        options.package_manifest = selected;
        options.expected_model_reference = reference;
    }

    ~ResourceFixture() {
        // Windows CAS files are read-only. All fixture bytes are private copies.
        std::error_code error;
        for (fs::recursive_directory_iterator it(root, error), end; it != end && !error;
             it.increment(error))
            if (!it->is_symlink(error) && it->is_regular_file(error))
                fs::permissions(it->path(), fs::perms::owner_write, fs::perm_options::add, error);
        fs::remove_all(root, error);
    }

    product::ImportResult run() {
        return musetalk ? product::import_public_musetalk_v15(source, *cache, options)
                        : product::import_public_latentsync_16(source, *cache, options);
    }

    void reject(const product::ModelPackageErrorCode expected) {
        try {
            (void)run();
        } catch (const product::ModelPackageError& error) {
            require_test(error.code() == expected, "unexpected rejection: " + std::string(error.what()));
            require_test(cache->list().empty(), "failed finalization published a manifest");
            return;
        }
        throw std::runtime_error("invalid resource was accepted");
    }
};

void resource_regressions(const fs::path& temporary, const fs::path& real_specs) {
    size_t failures = 0;
    size_t sequence = 0;
    auto test = [&](const std::string& name, const auto& body) {
        try {
            body(temporary / std::to_string(++sequence));
            std::cout << name << ": PASS\n";
        } catch (const SymlinkUnavailable& error) {
            std::cout << name << ": SKIP: " << error.what() << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << name << ": FAIL: " << error.what() << '\n';
        }
    };
    for (const bool musetalk : {true, false}) {
        const std::string model = musetalk ? "MuseTalk" : "LatentSync";
        for (const std::string version : {"1.0.0", "1.0.1"}) {
            test(model + " " + version + " selected profile", [&](const fs::path& root) {
                ResourceFixture fixture(root, real_specs, musetalk, version);
                const auto installed = fixture.run().installation;
                require_test(installed.identity.reference() == fixture.reference,
                             "selected manifest identity changed");
                const fs::path profile = fixture.cache->artifact_path(fixture.execution.sha256);
                require_test(fs::file_size(profile) == fixture.execution.size &&
                             product::sha256_file(profile) == fixture.execution.sha256,
                             "installed execution profile size/hash differs");
                require_test(read_bytes(profile) == read_bytes(fixture.selected.parent_path() / "execution.json"),
                             "converter selected another execution profile");
            });
        }
        for (const std::string fault : {"wrong-root", "wrong-hash", "malformed", "missing"}) {
            test(model + " successor rejects " + fault, [&](const fs::path& root) {
                ResourceFixture fixture(root, real_specs, musetalk, "1.0.1");
                const fs::path profile = fixture.selected.parent_path() / "execution.json";
                if (fault == "missing") fs::remove(profile);
                else if (fault == "wrong-root") write_bytes(profile, read_bytes(fixture.family / "execution.json"));
                else if (fault == "wrong-hash") write_bytes(profile, read_bytes(profile) + "\n");
                else {auto bytes = read_bytes(profile); bytes[0] = '!'; write_bytes(profile, bytes);}
                fixture.reject(fault == "missing" ? product::ModelPackageErrorCode::ArtifactMissing
                                                  : product::ModelPackageErrorCode::ChecksumMismatch);
            });
        }
        test(model + " selected identity is authoritative", [&](const fs::path& root) {
            ResourceFixture fixture(root, real_specs, musetalk, "1.0.1");
            fixture.options.expected_model_reference = "vrhino/unrelated:1.0.1";
            fixture.reject(product::ModelPackageErrorCode::PackageInvalid);
        });
        test(model + " manifest-relative directory, not hardcoded version", [&](const fs::path& root) {
            ResourceFixture fixture(root, real_specs, musetalk, "1.0.1");
            const fs::path selected = fixture.family / "versions/selected/vrhino-model.json";
            write_bytes(selected, read_bytes(fixture.selected));
            write_bytes(selected.parent_path() / "execution.json", read_bytes(fixture.selected.parent_path() / "execution.json"));
            fixture.options.package_manifest = selected;
            require_test(fixture.run().installation.identity.reference() == fixture.reference,
                         "directory selection replaced manifest identity");
        });
        test(model + " rejects manifest outside package root", [&](const fs::path& root) {
            ResourceFixture fixture(root, real_specs, musetalk, "1.0.1");
            const fs::path outside = fixture.specs /
                (fixture.family.filename().string() + "-outside") / "vrhino-model.json";
            write_bytes(outside, read_bytes(fixture.selected));
            write_bytes(outside.parent_path() / "execution.json", read_bytes(fixture.selected.parent_path() / "execution.json"));
            fixture.options.package_manifest = outside;
            fixture.reject(product::ModelPackageErrorCode::PackageInvalid);
        });
        test(model + " rejects profile symlink outside package root", [&](const fs::path& root) {
            ResourceFixture fixture(root, real_specs, musetalk, "1.0.1");
            const fs::path profile = fixture.selected.parent_path() / "execution.json";
            const fs::path outside = fixture.specs / "outside-profile.json";
            write_bytes(outside, read_bytes(profile));
            fs::remove(profile);
            std::error_code error;
            fs::create_symlink(outside, profile, error);
            if (error) throw SymlinkUnavailable(error.message());
            fixture.reject(product::ModelPackageErrorCode::PackageInvalid);
        });
    }
    require_test(failures == 0, "successor resource regression failures: " + std::to_string(failures));
}

}  // namespace

int main() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path temporary = fs::temp_directory_path() /
        ("vrhino-product-schema-successor-" + std::to_string(stamp));
    try {
        const fs::path spec_root = fs::path(VRHINO_TEST_SOURCE_ROOT) / "specs";
        resource_regressions(temporary / "resources", spec_root);
        const std::vector<std::string> references = {
            "vrhino/ltx-video-v0.9.1:1.1.1",
            "vrhino/wan2.1-t2v-1.3b:1.0.1",
            "vrhino/mochi-1-preview:1.0.1",
            "vrhino/musetalk-v1.5:1.0.1",
            "vrhino/latentsync-1.6:1.0.1",
        };
        product::LocalModelCache cache(temporary / "cache");
        for (const std::string& reference : references) {
            const std::optional<product::PullDistributionPlan> plan =
                product::find_pull_distribution_plan(reference, spec_root);
            require_test(plan.has_value(), "successor pull plan missing: " + reference);
            require_test(plan->model_reference == reference,
                         "successor pull plan identity mismatch: " + reference);
            const product::SourceArtifactPlanDocument source =
                product::load_pull_source_artifact_plan(*plan);
            require_test(product::load_source_artifact_plan(reference, spec_root).model_reference == reference,
                         "packaged-tree discovery collided with component provenance");
            require_test(source.model_reference == reference,
                         "successor source plan identity mismatch: " + reference);

            product::ImportOptions options;
            options.converter_spec_root = spec_root;
            try {
                (void)product::import_local_model(
                    reference, temporary / "missing-source", cache, options);
                throw std::runtime_error(
                    "successor converter unexpectedly accepted missing source: " +
                    reference);
            } catch (const product::ModelPackageError& error) {
                require_test(
                    error.code() !=
                        product::ModelPackageErrorCode::PackageVersionUnsupported,
                    "successor was not registered with Native converter: " + reference);
                require_test(
                    error.code() == product::ModelPackageErrorCode::SourceInvalid ||
                        error.code() == product::ModelPackageErrorCode::SourceNotFound ||
                        error.code() == product::ModelPackageErrorCode::ArtifactMissing,
                    "successor converter failed before bounded source admission: " +
                        reference + ": " + error.what());
            }
        }
        fs::remove_all(temporary);
        std::cout << "successor pull/source/converter dispatch tests: PASS\n";
        return 0;
    } catch (const std::exception& error) {
        fs::remove_all(temporary);
        std::cerr << "successor pull/source/converter dispatch tests: FAIL: "
                  << error.what() << '\n';
        return 1;
    }
}
