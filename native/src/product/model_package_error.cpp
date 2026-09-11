#include "vrhino/product/model_package.h"

namespace vrhino::product {

const char* model_package_error_code_name(const ModelPackageErrorCode code) {
    switch (code) {
        case ModelPackageErrorCode::ModelNotFound: return "MODEL_NOT_FOUND";
        case ModelPackageErrorCode::PackageInvalid: return "PACKAGE_INVALID";
        case ModelPackageErrorCode::PackageVersionUnsupported:
            return "PACKAGE_VERSION_UNSUPPORTED";
        case ModelPackageErrorCode::ArtifactMissing: return "ARTIFACT_MISSING";
        case ModelPackageErrorCode::ChecksumMismatch: return "CHECKSUM_MISMATCH";
        case ModelPackageErrorCode::InstallFailed: return "INSTALL_FAILED";
        case ModelPackageErrorCode::CacheError: return "CACHE_ERROR";
        case ModelPackageErrorCode::RegistryUnavailable: return "REGISTRY_UNAVAILABLE";
        case ModelPackageErrorCode::DownloadFailed: return "DOWNLOAD_FAILED";
        case ModelPackageErrorCode::DownloadResumeFailed: return "DOWNLOAD_RESUME_FAILED";
        case ModelPackageErrorCode::NetworkError: return "NETWORK_ERROR";
        case ModelPackageErrorCode::InsufficientDiskSpace: return "INSUFFICIENT_DISK_SPACE";
        case ModelPackageErrorCode::UnsupportedGpu: return "UNSUPPORTED_GPU";
        case ModelPackageErrorCode::InsufficientVram: return "INSUFFICIENT_VRAM";
        case ModelPackageErrorCode::DriverIncompatible: return "DRIVER_INCOMPATIBLE";
        case ModelPackageErrorCode::InvalidInput: return "INVALID_INPUT";
        case ModelPackageErrorCode::RuntimeError: return "RUNTIME_ERROR";
        case ModelPackageErrorCode::OutOfMemory: return "OUT_OF_MEMORY";
        case ModelPackageErrorCode::OutputExists: return "OUTPUT_EXISTS";
        case ModelPackageErrorCode::OutputInvalid: return "OUTPUT_INVALID";
        case ModelPackageErrorCode::VideoEncodingFailed: return "VIDEO_ENCODING_FAILED";
        case ModelPackageErrorCode::Cancelled: return "CANCELLED";
        case ModelPackageErrorCode::ComponentNotFound: return "COMPONENT_NOT_FOUND";
        case ModelPackageErrorCode::ComponentInvalid: return "COMPONENT_INVALID";
        case ModelPackageErrorCode::ComponentVersionUnsupported:
            return "COMPONENT_VERSION_UNSUPPORTED";
        case ModelPackageErrorCode::SourceInvalid: return "SOURCE_INVALID";
        case ModelPackageErrorCode::SourceRevisionRequired:
            return "SOURCE_REVISION_REQUIRED";
        case ModelPackageErrorCode::SourceNotFound: return "SOURCE_NOT_FOUND";
        case ModelPackageErrorCode::SourceDownloadFailed:
            return "SOURCE_DOWNLOAD_FAILED";
        case ModelPackageErrorCode::SourceDownloadResumeFailed:
            return "SOURCE_DOWNLOAD_RESUME_FAILED";
        case ModelPackageErrorCode::SourceIntegrityFailed:
            return "SOURCE_INTEGRITY_FAILED";
        case ModelPackageErrorCode::SourceDiskFull: return "SOURCE_DISK_FULL";
        case ModelPackageErrorCode::PullPlanNotFound: return "PULL_PLAN_NOT_FOUND";
    }
    return "CACHE_ERROR";
}

ModelPackageError::ModelPackageError(const ModelPackageErrorCode code,
                                     const std::string& message)
    : std::runtime_error(std::string(model_package_error_code_name(code)) + ": " + message),
      code_(code) {}

}  // namespace vrhino::product
