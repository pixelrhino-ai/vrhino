#include <chrono>
#include <filesystem>
#include <iostream>
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

}  // namespace

int main() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path temporary = fs::temp_directory_path() /
        ("vrhino-product-schema-successor-" + std::to_string(stamp));
    try {
        const fs::path spec_root = fs::path(VRHINO_TEST_SOURCE_ROOT) / "specs";
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
                product::load_source_artifact_plan(reference,
                                                    plan->document_path.parent_path());
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
