#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

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

int main() {
    const fs::path root = fs::temp_directory_path() /
        ("vrhino-pull-orchestration-tests-" + std::to_string(getpid()));
    std::error_code error;
    fs::remove_all(root, error);
    try {
        const fs::path specs = root / "specs";
        const std::string reference = "vrhino/fixture:1.0.0";
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
        write_text(source_blob, "test");
        product::UnifiedPullOptions options;
        options.converter_spec_root = specs;
        expect_code(product::ModelPackageErrorCode::PackageVersionUnsupported, [&] {
            (void)product::pull_runnable_model(reference, cache, options);
        }, "unregistered converter failure");
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
