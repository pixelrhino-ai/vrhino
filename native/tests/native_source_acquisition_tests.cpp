#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

#include "vrhino/product/source_acquisition.h"

namespace fs = std::filesystem;
namespace product = vrhino::product;

namespace {

volatile std::sig_atomic_t interrupted = 0;

void interrupt_handler(int) { interrupted = 1; }

void require_test(const bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Operation>
void expect_code(const product::ModelPackageErrorCode expected,
                 Operation&& operation, const std::string& context) {
    try {
        operation();
        throw std::runtime_error(context + ": operation unexpectedly succeeded");
    } catch (const product::ModelPackageError& error) {
        require_test(error.code() == expected, context + ": wrong error code");
    }
}

void write_text(const fs::path& path, const std::string& value) {
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output << value;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 5 && argc <= 10) {
        try {
            interrupted = 0;
            std::signal(SIGINT, interrupt_handler);
            const std::string model_reference = argv[1];
            const product::SourceReference source = product::parse_source_reference(argv[2]);
            const product::SourceArtifactPlanDocument plan =
                product::load_source_artifact_plan(model_reference, argv[3]);
            product::LocalSourceCache cache(
                argv[4], argc >= 8 ? fs::path(argv[7]) : fs::path{});
            product::AcquisitionOptions options;
            if (argc >= 6) options.huggingface_official.base_url = argv[5];
            if (argc >= 7) options.huggingface_mirror.base_url = argv[6];
            if (argc >= 9) options.network.ca_file = argv[8];
            if (argc == 10) {
                options.network.connect_timeout_seconds = std::stol(argv[9]);
                options.network.low_speed_timeout_seconds = std::stol(argv[9]);
                options.official_availability_connect_timeout_seconds =
                    std::stol(argv[9]);
                options.official_availability_low_speed_timeout_seconds =
                    std::stol(argv[9]);
            }
            options.network.cancellation_requested = [] {
                return interrupted != 0;
            };
            if (const char* token = std::getenv("HF_TOKEN");
                token != nullptr && *token != '\0')
                options.huggingface_token = token;
            const product::AcquisitionResult result =
                cache.acquire(source, plan, options, &std::cout);
            std::cout << "source=" << result.source.canonical() << '\n'
                      << "materialized=" << result.materialized_directory.string() << '\n'
                      << "downloaded_bytes=" << result.downloaded_bytes << '\n'
                      << "reused_bytes=" << result.reused_bytes << '\n'
                      << "resumed_bytes=" << result.resumed_bytes << '\n'
                      << "temporary_disk_peak_bytes="
                      << result.temporary_disk_peak_bytes << '\n'
                      << "acquisition_seconds=" << result.acquisition_seconds << '\n';
            if (result.transport_endpoint.has_value())
                std::cout << "transport_endpoint="
                          << (*result.transport_endpoint ==
                                  product::HuggingFaceEndpointKind::Official
                              ? "official" : "mirror") << '\n';
            for (const product::AcquiredSourceArtifact& artifact : result.artifacts)
                std::cout << "artifact=" << artifact.declaration.id << '\t'
                          << artifact.declaration.size << '\t'
                          << artifact.declaration.sha256 << '\t'
                          << (artifact.downloaded ? "downloaded" : "reused") << '\n';
            return 0;
        } catch (const product::ModelPackageError& error) {
            std::cerr << error.what() << '\n';
            return error.code() == product::ModelPackageErrorCode::Cancelled ? 130 : 1;
        } catch (const std::exception& exception) {
            std::cerr << exception.what() << '\n';
            return 1;
        }
    }
    if (argc != 1) {
        std::cerr << "usage: vrhino-native-source-acquisition-tests "
                     "[MODEL SOURCE SPEC-ROOT SOURCE-CACHE "
                     "[HF-OFFICIAL-BASE HF-MIRROR-BASE TEMP-ROOT CA-FILE "
                     "[TIMEOUT-SECONDS]]]\n";
        return 2;
    }
    const fs::path root = fs::temp_directory_path() /
        ("vrhino-source-acquisition-tests-" + std::to_string(getpid()));
    std::error_code error;
    fs::remove_all(root, error);
    fs::create_directories(root);
    try {
        const std::string revision(40, 'a');
        const product::SourceReference valid = product::parse_source_reference(
            "hf://owner/model@" + revision);
        require_test(valid.repository == "owner/model" && valid.revision == revision,
                     "valid source reference");
        require_test(valid.canonical() == "hf://owner/model@" + revision,
                     "canonical source reference");

        expect_code(product::ModelPackageErrorCode::SourceRevisionRequired, [&] {
            product::parse_source_reference("hf://owner/model");
        }, "missing revision");
        expect_code(product::ModelPackageErrorCode::SourceInvalid, [&] {
            product::parse_source_reference("hf://owner/model/extra@" + revision);
        }, "malformed repository");
        expect_code(product::ModelPackageErrorCode::SourceRevisionRequired, [&] {
            product::parse_source_reference("hf://owner/model@main");
        }, "floating revision");

        const fs::path specs = root / "specs";
        write_text(specs / "fixture" / "source-plan.json",
            "{\"schema_version\":1,"
            "\"model_reference\":\"vrhino/fixture:1\","
            "\"requested_source\":{\"provider\":\"huggingface\","
            "\"repository\":\"owner/model\",\"revision\":\"" + revision + "\"},"
            "\"artifacts\":[{\"id\":\"a\",\"role\":\"checkpoint\","
            "\"repository\":\"owner/model\",\"revision\":\"" + revision + "\","
            "\"upstream_path\":\"a.bin\",\"local_path\":\"a.bin\","
            "\"size\":4,\"sha256\":\"" + std::string(64, '0') + "\"}]}\n");
        const product::SourceArtifactPlanDocument plan =
            product::load_source_artifact_plan("vrhino/fixture:1", specs);
        require_test(plan.artifacts.size() == 1, "plan artifact count");
        expect_code(product::ModelPackageErrorCode::SourceNotFound, [&] {
            product::load_source_artifact_plan("vrhino/missing:1", specs);
        }, "missing source plan");

        product::LocalSourceCache cache(root / "sources");
        product::AcquisitionOptions options;
        options.available_space_override = 0;
        options.huggingface_official.base_url = "http://127.0.0.1:9";
        options.huggingface_mirror.base_url = "http://127.0.0.1:9";
        options.network.allow_development_http = true;
        expect_code(product::ModelPackageErrorCode::SourceDiskFull, [&] {
            cache.acquire(valid, plan, options);
        }, "source disk admission");

        const fs::path product_cache = root / "product-cache";
        product::LocalSourceCache cleanup_cache(
            product_cache / "sources", product_cache / "tmp");
        require_test(cleanup_cache.layout().temporary == product_cache / "tmp",
                     "source partials do not use the product cache tmp root");
        const fs::path unique_fixture = root / "unique-fixture";
        write_text(unique_fixture, "abc");
        const std::string unique_sha = product::sha256_file(unique_fixture);
        const fs::path shared_fixture = root / "shared-fixture";
        write_text(shared_fixture, "shared");
        const std::string shared_sha = product::sha256_file(shared_fixture);
        const fs::path unique_blob = cleanup_cache.layout().blobs /
            unique_sha.substr(0, 2) / unique_sha;
        const fs::path shared_blob = cleanup_cache.layout().blobs /
            shared_sha.substr(0, 2) / shared_sha;
        write_text(unique_blob, "abc");
        write_text(shared_blob, "shared");
        const fs::path tree = cleanup_cache.layout().trees / "owner/model" / revision /
            "vrhino/cleanup/1";
        fs::create_directories(tree);
        fs::create_hard_link(unique_blob, tree / "unique.bin");
        fs::create_hard_link(shared_blob, tree / "shared.bin");
        write_text(tree / "vrhino-source-plan.json", "{}\n");
        const fs::path installed_shared = product_cache / "blobs/sha256" /
            shared_sha.substr(0, 2) / shared_sha;
        fs::create_directories(installed_shared.parent_path());
        fs::create_hard_link(shared_blob, installed_shared);
        const fs::path stale_partial = product_cache / "tmp/downloads" /
            (unique_sha + ".partial");
        write_text(stale_partial, "x");

        product::AcquisitionResult cleanup_acquisition;
        cleanup_acquisition.model_reference = "vrhino/cleanup:1";
        cleanup_acquisition.source = valid;
        cleanup_acquisition.materialized_directory = tree;
        product::SourceArtifactPlan unique_artifact;
        unique_artifact.id = "unique";
        unique_artifact.local_path = "unique.bin";
        unique_artifact.size = 3;
        unique_artifact.sha256 = unique_sha;
        product::SourceArtifactPlan shared_artifact;
        shared_artifact.id = "shared";
        shared_artifact.local_path = "shared.bin";
        shared_artifact.size = 6;
        shared_artifact.sha256 = shared_sha;
        cleanup_acquisition.artifacts.push_back(
            {unique_artifact, unique_blob, true});
        cleanup_acquisition.artifacts.push_back(
            {shared_artifact, shared_blob, true});

        const product::SourceCleanupResult cleaned =
            cleanup_cache.reclaim_after_install(cleanup_acquisition);
        require_test(!fs::exists(tree), "completed source tree was not removed");
        require_test(!fs::exists(unique_blob),
                     "unshared source blob was not reclaimed");
        require_test(fs::is_regular_file(shared_blob) &&
                         fs::is_regular_file(installed_shared),
                     "shared/installed data was removed by source cleanup");
        require_test(!fs::exists(stale_partial),
                     "stale completed partial was not removed");
        require_test(cleaned.reclaimed_bytes >= 7,
                     "source cleanup did not report reclaimed unique bytes");

        const fs::path cancel_root = root / "cancel-cache";
        product::LocalSourceCache cancel_cache(
            cancel_root / "sources", cancel_root / "tmp");
        product::SourceArtifactPlanDocument cancel_plan;
        cancel_plan.model_reference = "vrhino/cancel:1";
        cancel_plan.requested_source = valid;
        product::SourceArtifactPlan first_artifact = unique_artifact;
        first_artifact.repository = valid.repository;
        first_artifact.revision = valid.revision;
        first_artifact.upstream_path = "unique.bin";
        product::SourceArtifactPlan second_artifact = shared_artifact;
        second_artifact.repository = valid.repository;
        second_artifact.revision = valid.revision;
        second_artifact.upstream_path = "shared.bin";
        cancel_plan.artifacts = {first_artifact, second_artifact};
        const fs::path first_blob = cancel_cache.layout().blobs /
            unique_sha.substr(0, 2) / unique_sha;
        write_text(first_blob, "abc");
        bool cancel_requested = false;
        product::AcquisitionOptions cancel_options;
        cancel_options.network.cancellation_requested = [&] {
            return cancel_requested;
        };
        cancel_options.network.progress = [&](const uint64_t completed,
                                              const uint64_t) {
            if (completed >= first_artifact.size) cancel_requested = true;
        };
        expect_code(product::ModelPackageErrorCode::Cancelled, [&] {
            cancel_cache.acquire(valid, cancel_plan, cancel_options);
        }, "source cancellation between artifacts");
        require_test(fs::is_regular_file(first_blob),
                     "cancellation removed a completed source blob");
        require_test(!fs::exists(cancel_cache.layout().temporary / "downloads" /
                                 (shared_sha + ".partial")),
                     "cancellation started a subsequent artifact");

        fs::remove_all(root, error);
        std::cout << "native source acquisition tests: PASS\n";
        return 0;
    } catch (const std::exception& exception) {
        fs::remove_all(root, error);
        std::cerr << "native source acquisition tests: FAIL: "
                  << exception.what() << '\n';
        return 1;
    }
}
