#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <chrono>
#include <sstream>

#include "vrhino/product/pull_orchestration.h"

namespace fs = std::filesystem;
namespace product = vrhino::product;

namespace {

constexpr const char* kRevision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
constexpr const char* kTestSha =
    "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void write_text(const fs::path& path, const std::string& value) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << value;
    require_test(static_cast<bool>(output), "cannot write test fixture");
}

template <typename Operation>
void expect_code(const product::ModelPackageErrorCode expected,
                 Operation&& operation, const std::string& context) {
    try {
        operation();
        throw std::runtime_error(context + ": unexpectedly succeeded");
    } catch (const product::ModelPackageError& error) {
        require_test(error.code() == expected, context + ": wrong error code");
    }
}

void write_source_backed_plan(const fs::path& spec, const std::string& reference) {
    write_text(spec / "pull-plan.json",
        "{\"schema_version\":1,\"model_reference\":\"" + reference +
        "\",\"distribution\":{\"kind\":\"source_backed\","
        "\"source\":{\"provider\":\"huggingface\","
        "\"repository\":\"owner/model\",\"revision\":\"" + kRevision +
        "\"},\"source_plan\":\"source-plan.json\",\"converter\":\"native\"}}\n");
    write_text(spec / "source-plan.json",
        "{\"schema_version\":1,\"model_reference\":\"" + reference +
        "\",\"requested_source\":{\"provider\":\"huggingface\","
        "\"repository\":\"owner/model\",\"revision\":\"" + kRevision +
        "\"},\"artifacts\":[{\"id\":\"fixture\",\"role\":\"checkpoint\","
        "\"repository\":\"owner/model\",\"revision\":\"" + kRevision +
        "\",\"upstream_path\":\"test.bin\",\"local_path\":\"test.bin\","
        "\"size\":4,\"sha256\":\"" + kTestSha + "\"}]}\n");
}

void write_package_manifest(const fs::path& spec) {
    write_text(spec / "vrhino-model.json", R"({
      "schema_version":1,
      "identity":{"namespace":"vrhino","name":"fixture","version":"1.0.0",
                  "architecture":"fixture","publisher":"VRhino"},
      "compatibility":{"runtime_contract":"cuda-v1","vrm_schema":{
        "format_major":0,"format_minor":1,"metadata_schema":1}},
      "artifacts":[
        {"id":"runtime","role":"runtime.vrm","path":"model/model.vrm",
         "size":8,"sha256":"0000000000000000000000000000000000000000000000000000000000000000",
         "required":true},
        {"id":"profile","role":"execution.profile","path":"execution/default.json",
         "size":4,"sha256":"9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
         "required":true}],
      "entrypoint":{"runtime_artifact":"runtime","components":[],
                    "default_preset":"default"},
      "defaults":{"default_preset":"default","presets":{"default":{
        "profile_artifact":"profile","inputs":{}}}},
      "hardware":{"presets":{"default":{"minimum_vram_bytes":null,
        "recommended_vram_bytes":null,"minimum_compute_capability":null,
        "qualifications":[]}}},
      "source":{"repository":"owner/model",
                "revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "converter_version":"fixture-v1"},
      "license":{"identifier":"Apache-2.0","artifact":"profile",
                 "upstream_notice":"fixture"}
    })");
}

}  // namespace

int main(int argc, char** argv) {
    const fs::path root = fs::temp_directory_path() /
        ("vrhino-pull-orchestration-tests-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::error_code error;
    fs::remove_all(root, error);
    try {
        const fs::path specs = root / "specs";
        const std::string reference = "vrhino/fixture:1.0.0";
        // Create the component first, including a lexically earlier sibling.
        // The original recursive parser fails even if it visits the model first.
        const std::string component =
            R"({"schema_version":1,"component":"semantic_segmenter_2d","sources":[]})";
        write_text(specs / "00-component/source-plan.json", component);
        write_text(specs / "fixture/component-a/source-plan.json", component);
        write_source_backed_plan(specs / "fixture", reference);
        write_package_manifest(specs / "fixture");
        const auto plan = product::find_pull_distribution_plan(reference, specs);
        require_test(plan.has_value(), "declarative pull plan was not found");
        require_test(plan->source.repository == "owner/model" &&
                         plan->source.revision == kRevision &&
                         plan->converter == "native",
                     "declarative source binding mismatch");
        require_test(!product::find_pull_distribution_plan(
                          "vrhino/absent:1.0.0", specs).has_value(),
                     "unexpected pull plan match");

        const fs::path declared = specs / "fixture/source-plan.json";
        const auto exact = product::load_pull_source_artifact_plan(*plan);
        require_test(fs::equivalent(exact.document_path, declared) &&
                     exact.model_reference == reference, "exact declared model plan");
        require_test(product::load_source_artifact_plan(reference, specs).model_reference ==
                     reference, "component-aware model discovery");
        write_text(specs / "duplicate/source-plan.json", exact.raw_json);
        expect_code(product::ModelPackageErrorCode::SourceInvalid, [&] {
            (void)product::load_source_artifact_plan(reference, specs);
        }, "duplicate model discovery");
        require_test(product::load_pull_source_artifact_plan(*plan).model_reference == reference,
                     "exact loading must not discover another plan");
        fs::remove_all(specs / "duplicate");

        auto reject_declared = [&](const std::string& bytes,
                                   product::ModelPackageErrorCode code) {
            write_text(declared, bytes);
            expect_code(code, [&] {
                (void)product::load_pull_source_artifact_plan(*plan);
            }, "invalid explicit model plan");
            write_text(declared, exact.raw_json);
        };
        reject_declared("{", product::ModelPackageErrorCode::SourceInvalid);
        reject_declared(component, product::ModelPackageErrorCode::SourceInvalid);
        for (const auto& replacement : {
                std::pair<std::string, std::string>{reference, "vrhino/wrong:1.0.0"},
                {"\"schema_version\":1", "\"schema_version\":2"},
                {"test.bin", "../test.bin"},
                {kTestSha, std::string(64, 'z')}}) {
            std::string invalid = exact.raw_json;
            invalid.replace(invalid.find(replacement.first), replacement.first.size(),
                            replacement.second);
            reject_declared(invalid, product::ModelPackageErrorCode::SourceInvalid);
        }
        std::string floating = exact.raw_json;
        floating.replace(floating.find(kRevision), std::string(kRevision).size(), "main");
        reject_declared(floating, product::ModelPackageErrorCode::SourceRevisionRequired);
        auto wrong_source = *plan;
        wrong_source.source.revision = std::string(40, 'b');
        expect_code(product::ModelPackageErrorCode::PackageInvalid, [&] {
            (void)product::load_pull_source_artifact_plan(wrong_source);
        }, "immutable source binding mismatch");
        fs::remove(declared);
        expect_code(product::ModelPackageErrorCode::SourceInvalid, [&] {
            (void)product::load_pull_source_artifact_plan(*plan);
        }, "absent declared model plan");
        write_text(declared, exact.raw_json);
        for (const std::string& unsafe : {"../source-plan.json", "/source-plan.json",
                                          "C:/source-plan.json", "..\\source-plan.json"}) {
            auto invalid = *plan;
            invalid.source_plan = unsafe;
            expect_code(product::ModelPackageErrorCode::PackageInvalid, [&] {
                (void)product::load_pull_source_artifact_plan(invalid);
            }, "unsafe declared path");
        }
        // Symlink rejection where supported; Windows CI also covers portable
        // traversal/drive paths without requiring developer-mode symlinks.
        write_text(root / "outside.json", exact.raw_json);
        fs::create_symlink(root / "outside.json", specs / "fixture/escape.json", error);
        if (!error) {
            auto escaped = *plan; escaped.source_plan = "escape.json";
            expect_code(product::ModelPackageErrorCode::SourceInvalid, [&] {
                (void)product::load_pull_source_artifact_plan(escaped);
            }, "symlink escaped declared path");
        }
        // A component discriminator must not hide a malformed model plan.
        write_text(specs / "00-component/source-plan.json",
            R"({"schema_version":1,"component":"semantic_segmenter_2d","model_reference":null})");
        expect_code(product::ModelPackageErrorCode::SourceInvalid, [&] {
            (void)product::load_source_artifact_plan(reference, specs);
        }, "model fields cannot be skipped as component provenance");
        write_text(specs / "00-component/source-plan.json", component);

        product::LocalModelCache cache(root / "cache");
        product::UnifiedPullOptions low_space;
        low_space.converter_spec_root = specs;
        low_space.conversion_available_space_override = 0;
        expect_code(product::ModelPackageErrorCode::InsufficientDiskSpace, [&] {
            (void)product::pull_runnable_model(reference, cache, low_space);
        }, "source-backed pull disk admission");
        require_test(!fs::exists(cache.layout().temporary / "downloads"),
                     "disk admission started source acquisition");

        const fs::path source_blob = cache.layout().root / "sources/blobs/sha256/9f" /
                                     kTestSha;
        if (argc == 2) {
            // The Python fixture serves immutable HF-style paths over loopback.
            product::LocalSourceCache acquisition(cache.layout().root / "sources");
            product::AcquisitionOptions remote;
            remote.huggingface_official.base_url = argv[1];
            remote.automatic_huggingface_fallback = false;
            remote.network.allow_development_http = true;
            const auto acquired = acquisition.acquire(plan->source, exact, remote);
            require_test(acquired.downloaded_bytes == 4 && acquired.reused_bytes == 0,
                         "remote acquisition did not download the declared artifact");
        } else {
            require_test(argc == 1, "unexpected arguments");
            write_text(source_blob, "test");
        }
        product::UnifiedPullOptions options;
        options.converter_spec_root = specs;
        std::ostringstream progress;
        expect_code(product::ModelPackageErrorCode::PackageVersionUnsupported, [&] {
            (void)product::pull_runnable_model(reference, cache, options, &progress);
        }, "unregistered converter failure");
        require_test(progress.str().find("Acquiring source") != std::string::npos,
                     "remote pull never reached source acquisition");
        require_test(cache.list().empty(),
                     "failed pull published an installed package");
        require_test(fs::is_regular_file(source_blob) &&
                         product::sha256_file(source_blob) == kTestSha,
                     "failed pull damaged reusable source CAS");

        const fs::path no_plan = root / "no-plan";
        fs::create_directories(no_plan);
        options.converter_spec_root = no_plan;
        expect_code(product::ModelPackageErrorCode::PullPlanNotFound, [&] {
            (void)product::pull_runnable_model(
                "vrhino/no-plan:1.0.0", cache, options);
        }, "missing distribution plan");

        fs::remove_all(root, error);
        std::cout << "native pull orchestration tests: PASS\n";
        return 0;
    } catch (const std::exception& exception) {
        fs::remove_all(root, error);
        std::cerr << "native pull orchestration tests: FAIL: "
                  << exception.what() << '\n';
        return 1;
    }
}
