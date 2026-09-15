#pragma once

#include <filesystem>

#include "vrhino/product/model_package.h"

namespace vrhino::product::converter_detail {

// Execution profiles are version-scoped: they accompany the selected manifest.
// Family-scoped policies, workflows and legal resources retain their own paths.
inline std::filesystem::path manifest_execution_profile(
        const std::filesystem::path& package_root,
        const std::filesystem::path& selected_manifest) {
    namespace fs = std::filesystem;
    std::error_code error;
    const fs::path root = fs::canonical(package_root, error);
    if (error)
        throw ModelPackageError(ModelPackageErrorCode::PackageInvalid,
                                "cannot resolve converter package root");
    const auto contained = [&](const fs::path& path) {
        // Compare directory identity, not a string prefix or case folding.
        // This also respects Windows case-insensitive directory names.
        for (fs::path parent = path.parent_path(); !parent.empty();) {
            std::error_code comparison_error;
            if (fs::equivalent(parent, root, comparison_error) && !comparison_error)
                return true;
            const fs::path next = parent.parent_path();
            if (next == parent) break;
            parent = next;
        }
        return false;
    };
    const fs::path manifest = fs::canonical(selected_manifest, error);
    if (error || !contained(manifest))
        throw ModelPackageError(ModelPackageErrorCode::PackageInvalid,
                                "selected manifest is outside converter package root");
    const fs::path profile = fs::canonical(manifest.parent_path() / "execution.json", error);
    if (error)
        throw ModelPackageError(ModelPackageErrorCode::ArtifactMissing,
                                "selected manifest execution profile is missing");
    if (!contained(profile))
        throw ModelPackageError(ModelPackageErrorCode::PackageInvalid,
                                "execution profile is outside converter package root");
    if (!fs::is_regular_file(profile, error) || error)
        throw ModelPackageError(ModelPackageErrorCode::ArtifactMissing,
                                "selected manifest execution profile is not a regular file");
    return profile;
}

}  // namespace vrhino::product::converter_detail
