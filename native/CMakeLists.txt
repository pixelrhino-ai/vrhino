cmake_minimum_required(VERSION 3.22)

set(VRHINO_RELEASE_VERSION_FILE "${CMAKE_CURRENT_LIST_DIR}/../VERSION")
if(NOT EXISTS "${VRHINO_RELEASE_VERSION_FILE}")
    message(FATAL_ERROR "Missing canonical VRhino release VERSION file")
endif()
file(STRINGS "${VRHINO_RELEASE_VERSION_FILE}" VRHINO_RELEASE_VERSION_LINES)
list(LENGTH VRHINO_RELEASE_VERSION_LINES VRHINO_RELEASE_VERSION_LINE_COUNT)
if(NOT VRHINO_RELEASE_VERSION_LINE_COUNT EQUAL 1)
    message(FATAL_ERROR "VRhino release VERSION must contain exactly one line")
endif()
list(GET VRHINO_RELEASE_VERSION_LINES 0 VRHINO_RELEASE_VERSION)
if(NOT VRHINO_RELEASE_VERSION MATCHES
        "^v([0-9]+)\\.([0-9]+)\\.([0-9]+)-alpha$")
    message(FATAL_ERROR
        "VRhino release VERSION must use vMAJOR.MINOR.PATCH-alpha")
endif()
set(VRHINO_RELEASE_SEMVER
    "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}.${CMAKE_MATCH_3}")
project(vrhino_native VERSION "${VRHINO_RELEASE_SEMVER}" LANGUAGES CXX)

set(VRHINO_PRODUCT_VERSION "${VRHINO_RELEASE_VERSION}")
set(VRHINO_RELEASE_ARCHIVE_NAME
    "vrhino-linux-x86_64-cuda-${VRHINO_RELEASE_VERSION}.tar.gz")
set(VRHINO_RELEASE_METADATA_DIR "${CMAKE_CURRENT_BINARY_DIR}/release")
file(MAKE_DIRECTORY "${VRHINO_RELEASE_METADATA_DIR}")
file(WRITE "${VRHINO_RELEASE_METADATA_DIR}/VERSION"
    "${VRHINO_RELEASE_VERSION}\n")
file(WRITE "${VRHINO_RELEASE_METADATA_DIR}/archive-name.txt"
    "${VRHINO_RELEASE_ARCHIVE_NAME}\n")
find_package(Threads REQUIRED)

if(APPLE)
    set(VRHINO_CUDA_DEFAULT OFF)
else()
    set(VRHINO_CUDA_DEFAULT ON)
endif()
option(VRHINO_ENABLE_CUDA "Build the CUDA backend and CUDA validation targets" ${VRHINO_CUDA_DEFAULT})
option(VRHINO_ENABLE_METAL "Build the Metal backend (Phase 12B and later)" OFF)
option(VRHINO_ENABLE_TOKENIZERS "Build the Phase 13 native tokenizer layer" OFF)
option(VRHINO_ENABLE_PRODUCT_CLI "Build the native registry and unified product CLI" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Standalone allocation contract: also buildable with MSVC without linking the
# not-yet-ported file loader or the rest of the production library.
add_executable(vrhino-host-allocation-tests
    tests/host_allocation_tests.cpp src/tensor.cpp src/json.cpp)
target_include_directories(vrhino-host-allocation-tests PRIVATE include)
if(MSVC)
    target_compile_options(vrhino-host-allocation-tests PRIVATE /W4 /WX /utf-8)
else()
    target_compile_options(vrhino-host-allocation-tests PRIVATE -Wall -Wextra -Werror)
endif()

# Standalone loader ownership validation without Product or Runtime linkage.
add_executable(vrhino-file-mapping-tests
    tests/file_mapping_tests.cpp src/loader.cpp src/tensor.cpp src/json.cpp)
target_include_directories(vrhino-file-mapping-tests PRIVATE include)
if(MSVC)
    target_compile_options(vrhino-file-mapping-tests PRIVATE /W4 /WX /utf-8)
else()
    target_compile_options(vrhino-file-mapping-tests PRIVATE -Wall -Wextra -Werror)
endif()

if(WIN32)
    # Exercise the real verification/SHA/loader boundary without cache/CUDA linkage.
    add_executable(vrhino-stable-file-verification-tests
        tests/stable_file_verification_tests.cpp
        src/product/vrm_verification.cpp src/product/sha256.cpp
        src/product/model_package_error.cpp src/loader.cpp src/tensor.cpp src/json.cpp)
    target_include_directories(vrhino-stable-file-verification-tests PRIVATE include)
    target_link_libraries(vrhino-stable-file-verification-tests PRIVATE Threads::Threads)
    target_compile_definitions(vrhino-stable-file-verification-tests PRIVATE
        VRHINO_STABLE_VERIFICATION_TESTING=1)
    if(MSVC)
        target_compile_options(vrhino-stable-file-verification-tests PRIVATE /W4 /WX /utf-8)
    endif()
endif()

add_library(vrhino_native_common
    src/json.cpp
    src/tensor.cpp
    src/quantization/reference.cpp
    src/loader.cpp
    src/bundle.cpp
    src/tensor_util.cpp
    src/precision/precision.cpp
    src/components/executor.cpp
    src/components/tiling.cpp
    src/components/conditioning.cpp
    src/components/audio_encoder.cpp
    src/components/autoencoder_kl.cpp
    src/components/conditional_unet_2d.cpp
    src/components/temporal_conditional_unet_2d.cpp
    src/components/vision_detector.cpp
    src/components/pose_estimator.cpp
    src/components/semantic_segmenter.cpp
    src/sampling/sampling.cpp
    src/runtime/runtime.cpp
    src/runtime/prepared_tensor.cpp
    src/runtime/memory/memory.cpp
    src/architectures/common.cpp
    src/architectures/registry.cpp
    src/architectures/wan.cpp
    src/architectures/hunyuan_video.cpp
    src/architectures/ltx.cpp
    src/architectures/mochi.cpp
    src/architectures/cogvideox.cpp
)
target_include_directories(vrhino_native_common PUBLIC include)
target_compile_options(vrhino_native_common PRIVATE -Wall -Wextra -Werror)

# Internal bounded neural DAG facility.
add_library(vrhino_neural_graph STATIC
    src/runtime/neural_graph_validation.cpp
    src/runtime/neural_graph_evaluator.cpp)
target_link_libraries(vrhino_neural_graph PRIVATE vrhino_native_common)
target_compile_options(vrhino_neural_graph PRIVATE -Wall -Wextra -Werror)
# LTX production self-attention composition.
add_library(vrhino_ltx_graph STATIC src/architectures/ltx_self_attention_graph.cpp)
target_link_libraries(vrhino_ltx_graph PUBLIC vrhino_neural_graph vrhino_native_common)
target_compile_options(vrhino_ltx_graph PRIVATE -Wall -Wextra -Werror)
# Static dependency cycle reflects the existing combined Tensor/architecture
# library; CMake repeats these archives to resolve LTX Graph composition.
target_link_libraries(vrhino_native_common PRIVATE vrhino_ltx_graph)

add_executable(vrhino-component-executor-identity-tests
    tests/component_executor_identity_tests.cpp)
target_link_libraries(vrhino-component-executor-identity-tests PRIVATE
    vrhino_native_common)
target_compile_options(vrhino-component-executor-identity-tests PRIVATE
    -Wall -Wextra -Werror)

add_executable(vrhino-neural-graph-tests tests/neural_graph_tests.cpp)
target_include_directories(vrhino-neural-graph-tests PRIVATE src/runtime)
target_link_libraries(vrhino-neural-graph-tests PRIVATE
    vrhino_neural_graph vrhino_native_common)
target_compile_options(vrhino-neural-graph-tests PRIVATE -Wall -Wextra -Werror)
add_executable(vrhino-neural-graph-extension-tests tests/neural_graph_extension_tests.cpp)
target_include_directories(vrhino-neural-graph-extension-tests PRIVATE src/runtime)
target_link_libraries(vrhino-neural-graph-extension-tests PRIVATE vrhino_neural_graph vrhino_native_common)
target_compile_options(vrhino-neural-graph-extension-tests PRIVATE -Wall -Wextra -Werror)

add_executable(vrhino-neural-graph-bf16-tests tests/neural_graph_bf16_tests.cpp)
target_include_directories(vrhino-neural-graph-bf16-tests PRIVATE src/runtime)
target_link_libraries(vrhino-neural-graph-bf16-tests PRIVATE vrhino_neural_graph vrhino_native_common)
target_compile_options(vrhino-neural-graph-bf16-tests PRIVATE -Wall -Wextra -Werror)

add_executable(vrhino-neural-graph-layernorm-tests tests/neural_graph_layernorm_tests.cpp)
add_executable(vrhino-wan-graph-dual-path-tests tests/wan_graph_dual_path_tests.cpp)
add_executable(vrhino-wan-graph-optin-tests tests/wan_graph_optin_tests.cpp)
target_link_libraries(vrhino-wan-graph-optin-tests PRIVATE vrhino_neural_graph vrhino_native_common)
target_compile_options(vrhino-wan-graph-optin-tests PRIVATE -Wall -Wextra -Werror -ffp-contract=off)
target_link_libraries(vrhino-wan-graph-dual-path-tests PRIVATE vrhino_neural_graph vrhino_native_common)
target_compile_options(vrhino-wan-graph-dual-path-tests PRIVATE -Wall -Wextra -Werror -ffp-contract=off)
target_include_directories(vrhino-neural-graph-layernorm-tests PRIVATE src/runtime)
target_link_libraries(vrhino-neural-graph-layernorm-tests PRIVATE vrhino_neural_graph vrhino_native_common)
target_compile_options(vrhino-neural-graph-layernorm-tests PRIVATE -Wall -Wextra -Werror -ffp-contract=off)

add_executable(vrhino-neural-graph-bf16-ltx-admission-tests tests/neural_graph_bf16_ltx_admission_tests.cpp)
target_include_directories(vrhino-neural-graph-bf16-ltx-admission-tests PRIVATE src/runtime)
target_link_libraries(vrhino-neural-graph-bf16-ltx-admission-tests PRIVATE vrhino_ltx_graph vrhino_native_common)
target_compile_options(vrhino-neural-graph-bf16-ltx-admission-tests PRIVATE -Wall -Wextra -Werror)

# Bounded product/audio conditioning operations. They compose generic Tensor
# and Backend semantics but are deliberately separate from Shared Runtime.
add_library(vrhino_audio_conditioning src/workflow/audio_conditioning.cpp)
target_include_directories(vrhino_audio_conditioning PUBLIC include)
target_link_libraries(vrhino_audio_conditioning PUBLIC vrhino_native_common)
target_compile_options(vrhino_audio_conditioning PRIVATE -Wall -Wextra -Werror)

# Bounded detector preprocessing/decode/NMS/ROI operations remain outside the
# neural Runtime and Backend.
add_library(vrhino_face_roi src/workflow/face_roi.cpp)
target_include_directories(vrhino_face_roi PUBLIC include)
target_compile_options(vrhino_face_roi PRIVATE -Wall -Wextra -Werror)

# Bounded top-down affine, SimCC decode, and face-geometry policy. Neural pose
# execution remains in the generic component/Backend layer.
add_library(vrhino_pose_geometry src/workflow/pose_geometry.cpp)
target_include_directories(vrhino_pose_geometry PUBLIC include)
target_link_libraries(vrhino_pose_geometry PUBLIC vrhino_native_common vrhino_face_roi)
target_compile_options(vrhino_pose_geometry PRIVATE -Wall -Wextra -Werror)

add_library(vrhino_face_mask src/workflow/face_mask.cpp)
target_include_directories(vrhino_face_mask PUBLIC include)
target_link_libraries(vrhino_face_mask PUBLIC vrhino_native_common)
target_compile_options(vrhino_face_mask PRIVATE -Wall -Wextra -Werror)

# Narrow typed digital-human workflow operations. This is fixed orchestration
# over known component contracts and media/image operations, not a DAG engine.
add_library(vrhino_lip_sync_workflow src/workflow/lip_sync_workflow.cpp)
target_include_directories(vrhino_lip_sync_workflow PUBLIC include)
target_link_libraries(vrhino_lip_sync_workflow PUBLIC
    vrhino_native_common vrhino_face_mask vrhino_model_package)
target_compile_options(vrhino_lip_sync_workflow PRIVATE -Wall -Wextra -Werror)

# Typed non-overlapping temporal lip-sync workflow operations. This composes
# existing generic components and Sampling without a model Runtime or DAG.
add_library(vrhino_lip_sync_diffusion_workflow
    src/workflow/image_filter.cpp
    src/workflow/lip_sync_diffusion_workflow.cpp)
target_include_directories(vrhino_lip_sync_diffusion_workflow PUBLIC include)
target_link_libraries(vrhino_lip_sync_diffusion_workflow PUBLIC
    vrhino_lip_sync_workflow vrhino_audio_conditioning vrhino_pose_geometry)
target_compile_options(vrhino_lip_sync_diffusion_workflow PRIVATE
    -Wall -Wextra -Werror)

if(VRHINO_ENABLE_TOKENIZERS)
    # The complete tokenizer build-source graph is repository-local and
    # content-verified. Production configure must fail closed rather than
    # falling back to GitHub, crates.io, a system package, or a warm user
    # cache. Runtime model acquisition is a separate Product operation.
    set(VRHINO_TOKENIZER_BUILD_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/tokenizer-build-sources")
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/VerifyTokenizerBuildSources.cmake")
    vrhino_verify_tokenizer_build_sources(
        "${VRHINO_TOKENIZER_BUILD_SOURCES}")

    find_program(VRHINO_REAL_CARGO_EXECUTABLE NAMES cargo REQUIRED)
    set(VRHINO_TOKENIZER_CARGO_HOME
        "${CMAKE_CURRENT_BINARY_DIR}/tokenizer-cargo-home")
    set(VRHINO_TOKENIZER_CARGO_VENDOR
        "${VRHINO_TOKENIZER_BUILD_SOURCES}/cargo-vendor")
    file(MAKE_DIRECTORY "${VRHINO_TOKENIZER_CARGO_HOME}")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tokenizer-cargo-config.toml.in"
        "${VRHINO_TOKENIZER_CARGO_HOME}/config.toml"
        @ONLY)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/tokenizer-cargo-offline.sh.in"
        "${CMAKE_CURRENT_BINARY_DIR}/vrhino-tokenizer-cargo"
        @ONLY)
    file(CHMOD "${CMAKE_CURRENT_BINARY_DIR}/vrhino-tokenizer-cargo"
        PERMISSIONS
            OWNER_READ OWNER_WRITE OWNER_EXECUTE
            GROUP_READ GROUP_EXECUTE
            WORLD_READ WORLD_EXECUTE)
    set(CARGO_HOME "${VRHINO_TOKENIZER_CARGO_HOME}" CACHE PATH
        "Build-local Cargo home for the hermetic tokenizer build" FORCE)
    set(CARGO_EXECUTABLE
        "${CMAKE_CURRENT_BINARY_DIR}/vrhino-tokenizer-cargo"
        CACHE FILEPATH "Locked/offline Cargo wrapper" FORCE)

    # SentencePiece's module provider performs an unconditional nested
    # FetchContent population. Add the exact local Abseil source at the parent
    # level and force SentencePiece's supported package-provider branch. The
    # small Find module exposes only the already-defined local targets.
    set(ABSL_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(ABSL_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    add_subdirectory(
        "${VRHINO_TOKENIZER_BUILD_SOURCES}/abseil-cpp"
        "${CMAKE_CURRENT_BINARY_DIR}/tokenizer-abseil"
        EXCLUDE_FROM_ALL)
    list(PREPEND CMAKE_MODULE_PATH
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
    set(SPM_ABSL_PROVIDER "package" CACHE STRING "" FORCE)
    set(MLC_ENABLE_SENTENCEPIECE_TOKENIZER ON CACHE BOOL "" FORCE)
    set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
    set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
    add_subdirectory(
        "${VRHINO_TOKENIZER_BUILD_SOURCES}/tokenizers-cpp"
        "${CMAKE_CURRENT_BINARY_DIR}/tokenizers-cpp"
        EXCLUDE_FROM_ALL)

    add_custom_target(vrhino-tokenizer-source-integrity
        COMMAND ${CMAKE_COMMAND}
            -DVRHINO_TOKENIZER_BUILD_SOURCES=${VRHINO_TOKENIZER_BUILD_SOURCES}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunTokenizerSourceVerification.cmake
        COMMENT "Verifying repository-local tokenizer build sources")
    add_custom_target(vrhino-tokenizer-source-fail-closed-tests
        COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/tokenizer_source_fail_closed_tests.sh
            ${VRHINO_TOKENIZER_BUILD_SOURCES}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunTokenizerSourceVerification.cmake
        COMMENT "Testing missing/corrupt tokenizer source failures")

    add_library(vrhino_native_tokenizer src/input/tokenizer.cpp)
    target_include_directories(vrhino_native_tokenizer PUBLIC include)
    target_link_libraries(vrhino_native_tokenizer PUBLIC vrhino_native_common PRIVATE tokenizers_cpp)
    target_compile_options(vrhino_native_tokenizer PRIVATE -Wall -Wextra -Werror)

    add_executable(vrhino-tokenizer-tests tests/tokenizer_tests.cpp)
    target_link_libraries(vrhino-tokenizer-tests PRIVATE vrhino_native_tokenizer)

    add_executable(vrhino-tokenizer-probe tests/tokenizer_probe.cpp)
    target_link_libraries(vrhino-tokenizer-probe PRIVATE vrhino_native_tokenizer)
endif()

add_executable(vrhino-memory-runtime-tests tests/memory_runtime_tests.cpp)
target_link_libraries(vrhino-memory-runtime-tests PRIVATE vrhino_native_common)

add_executable(vrhino-precision-policy-tests tests/precision_policy_tests.cpp)
target_link_libraries(vrhino-precision-policy-tests PRIVATE vrhino_native_common)

add_executable(vrhino-vrm-inspect tests/inspect_vrm.cpp)
target_link_libraries(vrhino-vrm-inspect PRIVATE vrhino_native_common)

add_executable(vrhino-hunyuan-sampling-declaration-tests
    tests/hunyuan_sampling_declaration_tests.cpp)
target_link_libraries(vrhino-hunyuan-sampling-declaration-tests PRIVATE
    vrhino_native_common)

add_executable(vrhino-phase21b-sampling-declaration-tests
    tests/phase21b_sampling_declaration_tests.cpp)
target_link_libraries(vrhino-phase21b-sampling-declaration-tests PRIVATE
    vrhino_native_common)

add_executable(vrhino-generic-attention-mask-binding-tests
    tests/generic_attention_mask_binding_tests.cpp)
target_link_libraries(vrhino-generic-attention-mask-binding-tests PRIVATE
    vrhino_native_common)

# Output-only utility: converts an already-generated contiguous FP32/BF16
# BCTHW RGB tensor in [0, 1] to an RGB24 frame stream. It is not linked into the
# Runtime or any Backend.
add_executable(vrhino-export-bcthw-f32-rgb24 tools/export_bcthw_f32_rgb24.cpp)
target_link_libraries(vrhino-export-bcthw-f32-rgb24 PRIVATE vrhino_native_common)

# Product-layer runnable model package and immutable local cache. This target
# is deliberately independent of every hardware backend and tokenizer library.
add_library(vrhino_model_package
    src/product/info.cpp
    src/product/input_schema.cpp
    src/product/model_package.cpp
    src/product/model_package_error.cpp
    src/product/sha256.cpp
    src/product/vrm_verification.cpp)
target_include_directories(vrhino_model_package PUBLIC include)
target_link_libraries(vrhino_model_package PUBLIC vrhino_native_common PRIVATE Threads::Threads)
target_compile_options(vrhino_model_package PRIVATE -Wall -Wextra -Werror)

# Transport-neutral, synchronous Product execution lifecycle. The executor is
# injected so cancellation/event state remains CPU-testable and independent of
# Runtime, Backend, CUDA, HTTP, and future job scheduling.
add_library(vrhino_product_run_session
    src/product/run_event.cpp
    src/product/run_session.cpp)
target_include_directories(vrhino_product_run_session PUBLIC include)
target_link_libraries(vrhino_product_run_session PUBLIC
    vrhino_model_package Threads::Threads)
target_compile_options(vrhino_product_run_session PRIVATE -Wall -Wextra -Werror)

# Canonical ProductInputSchema-v1 request validation and RunOptions mapping.
# This remains transport-neutral and contains no HTTP or job policy.
add_library(vrhino_product_run_request src/product/run_request.cpp)
target_include_directories(vrhino_product_run_request PUBLIC include)
target_link_libraries(vrhino_product_run_request PUBLIC
    vrhino_model_package vrhino_product_run_session)
target_compile_options(vrhino_product_run_request PRIVATE -Wall -Wextra -Werror)

# Bounded process-memory application jobs: one Product worker, a four-entry
# FIFO, output reservations, and per-job RunSession ownership. Runtime and
# Backend remain unaware of scheduling and identifiers.
add_library(vrhino_application_jobs src/application/job_manager.cpp)
target_include_directories(vrhino_application_jobs PUBLIC include)
target_link_libraries(vrhino_application_jobs PUBLIC
    vrhino_product_run_request vrhino_product_run_session Threads::Threads)
target_compile_options(vrhino_application_jobs PRIVATE -Wall -Wextra -Werror)

add_executable(vrhino-run-session-tests tests/run_session_tests.cpp)
target_link_libraries(vrhino-run-session-tests PRIVATE
    vrhino_product_run_session Threads::Threads)
target_compile_options(vrhino-run-session-tests PRIVATE -Wall -Wextra -Werror)

add_executable(vrhino-json-request-tests tests/json_request_tests.cpp)
target_link_libraries(vrhino-json-request-tests PRIVATE vrhino_native_common)
target_compile_options(vrhino-json-request-tests PRIVATE -Wall -Wextra -Werror)

add_executable(vrhino-run-request-tests tests/run_request_tests.cpp)
target_link_libraries(vrhino-run-request-tests PRIVATE vrhino_product_run_request)
target_compile_definitions(vrhino-run-request-tests PRIVATE
    VRHINO_TEST_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/specs")
target_compile_options(vrhino-run-request-tests PRIVATE -Wall -Wextra -Werror)

add_executable(vrhino-job-manager-tests tests/job_manager_tests.cpp)
target_link_libraries(vrhino-job-manager-tests PRIVATE
    vrhino_application_jobs Threads::Threads)
target_compile_definitions(vrhino-job-manager-tests PRIVATE
    VRHINO_TEST_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_options(vrhino-job-manager-tests PRIVATE -Wall -Wextra -Werror)

add_executable(vrhino-sha256-tests tests/sha256_tests.cpp)
target_link_libraries(vrhino-sha256-tests PRIVATE vrhino_model_package)
target_compile_options(vrhino-sha256-tests PRIVATE -Wall -Wextra -Werror)

add_library(vrhino_product_progress src/product/progress.cpp)
target_include_directories(vrhino_product_progress PUBLIC include)
target_compile_options(vrhino_product_progress PRIVATE -Wall -Wextra -Werror)

add_executable(vrhino-native-progress-tests tests/native_progress_tests.cpp)
target_link_libraries(vrhino-native-progress-tests PRIVATE vrhino_product_progress)

# Product-layer generic component package/cache. This remains outside the
# frozen Runtime, Backend, architecture, and .vrm targets.
add_library(vrhino_component_package src/product/component_package.cpp)
target_include_directories(vrhino_component_package PUBLIC include)
target_link_libraries(vrhino_component_package PUBLIC vrhino_model_package)
target_compile_options(vrhino_component_package PRIVATE -Wall -Wextra -Werror)

# Product-only source conversion framework. It emits the frozen VRM/package
# contracts and has no dependency on Runtime execution, CUDA, Python, or
# PyTorch. Architecture-specific checkpoint mapping lives behind this target.
add_library(vrhino_converter
    src/product/converter.cpp
    src/product/pytorch_zip.cpp
    src/product/converter_ltx.cpp
    src/product/converter_wan.cpp
    src/product/converter_mochi.cpp)
target_sources(vrhino_converter PRIVATE src/product/converter_whisper.cpp)
target_sources(vrhino_converter PRIVATE src/product/converter_openai_whisper.cpp)
target_sources(vrhino_converter PRIVATE src/product/converter_sd_vae.cpp)
target_sources(vrhino_converter PRIVATE src/product/converter_musetalk_unet.cpp)
target_sources(vrhino_converter PRIVATE src/product/converter_temporal_unet.cpp)
target_sources(vrhino_converter PRIVATE src/product/converter_s3fd.cpp)
target_sources(vrhino_converter PRIVATE
    src/product/tflite.cpp
    src/product/converter_blazeface.cpp
    src/product/converter_selfie_multiclass.cpp)
target_sources(vrhino_converter PRIVATE src/product/converter_dwpose.cpp)
target_sources(vrhino_converter PRIVATE src/product/converter_bisenet.cpp)
target_sources(vrhino_converter PRIVATE src/product/converter_musetalk_package.cpp)
target_sources(vrhino_converter PRIVATE src/product/converter_latentsync_package.cpp)
target_include_directories(vrhino_converter PUBLIC include)
target_link_libraries(vrhino_converter PUBLIC vrhino_model_package)
target_compile_options(vrhino_converter PRIVATE -Wall -Wextra -Werror)

add_executable(vrhino-native-converter-tests tests/native_converter_tests.cpp)
target_link_libraries(vrhino-native-converter-tests PRIVATE vrhino_converter)

add_executable(vrhino-preconditioned-int8-tests
    tests/preconditioned_int8_tests.cpp)
target_link_libraries(vrhino-preconditioned-int8-tests PRIVATE vrhino_converter)
target_compile_options(vrhino-preconditioned-int8-tests PRIVATE
    -Wall -Wextra -Werror)

add_executable(vrhino-vrm-verification-tests tests/vrm_verification_tests.cpp)
target_link_libraries(vrhino-vrm-verification-tests PRIVATE vrhino_converter)
target_compile_options(vrhino-vrm-verification-tests PRIVATE -Wall -Wextra -Werror)

add_executable(vrhino-audio-conditioning-tests tests/audio_conditioning_tests.cpp)
target_link_libraries(vrhino-audio-conditioning-tests PRIVATE
    vrhino_audio_conditioning vrhino_converter)

add_executable(vrhino-autoencoder-kl-tests tests/autoencoder_kl_tests.cpp)
target_link_libraries(vrhino-autoencoder-kl-tests PRIVATE vrhino_converter)

add_executable(vrhino-conditional-unet-2d-tests
    tests/conditional_unet_2d_tests.cpp)
target_link_libraries(vrhino-conditional-unet-2d-tests PRIVATE vrhino_converter)

add_executable(vrhino-temporal-conditional-unet-tests
    tests/temporal_conditional_unet_tests.cpp)
target_link_libraries(vrhino-temporal-conditional-unet-tests PRIVATE
    vrhino_converter)

add_executable(vrhino-vision-detector-tests tests/vision_detector_tests.cpp)
target_link_libraries(vrhino-vision-detector-tests PRIVATE
    vrhino_converter vrhino_face_roi)

add_executable(vrhino-blazeface-tests tests/blazeface_tests.cpp)
target_link_libraries(vrhino-blazeface-tests PRIVATE
    vrhino_converter vrhino_face_roi vrhino_pose_geometry)

add_executable(vrhino-pose-estimator-tests tests/pose_estimator_tests.cpp)
target_link_libraries(vrhino-pose-estimator-tests PRIVATE
    vrhino_converter vrhino_pose_geometry)

add_executable(vrhino-semantic-segmenter-tests
    tests/semantic_segmenter_tests.cpp)
target_link_libraries(vrhino-semantic-segmenter-tests PRIVATE
    vrhino_converter vrhino_face_mask)

add_executable(vrhino-selfie-multiclass-tests
    tests/selfie_multiclass_tests.cpp)
target_link_libraries(vrhino-selfie-multiclass-tests PRIVATE
    vrhino_converter vrhino_face_mask)

add_executable(vrhino-lip-sync-workflow-tests tests/lip_sync_workflow_tests.cpp)
target_link_libraries(vrhino-lip-sync-workflow-tests PRIVATE
    vrhino_lip_sync_workflow)

add_executable(vrhino-lip-sync-diffusion-workflow-tests
    tests/lip_sync_diffusion_workflow_tests.cpp)
target_link_libraries(vrhino-lip-sync-diffusion-workflow-tests PRIVATE
    vrhino_lip_sync_diffusion_workflow)

add_executable(vrhino-image-filter-tests tests/image_filter_tests.cpp)
target_link_libraries(vrhino-image-filter-tests PRIVATE
    vrhino_lip_sync_diffusion_workflow)
target_compile_options(vrhino-image-filter-tests PRIVATE -Wall -Wextra -Werror)

add_executable(vrhino-openai-whisper-converter-tests
    tests/openai_whisper_converter_tests.cpp)
target_link_libraries(vrhino-openai-whisper-converter-tests PRIVATE
    vrhino_converter)

add_executable(vrhino-model tools/model_cache_cli.cpp)
target_link_libraries(vrhino-model PRIVATE vrhino_model_package)

add_executable(vrhino-model-package-tests tests/model_package_tests.cpp)
target_link_libraries(vrhino-model-package-tests PRIVATE vrhino_model_package)

add_executable(vrhino-model-info-tests tests/model_info_tests.cpp)
target_link_libraries(vrhino-model-info-tests PRIVATE vrhino_model_package)
target_compile_options(vrhino-model-info-tests PRIVATE -Wall -Wextra -Werror)
target_compile_definitions(vrhino-model-info-tests PRIVATE
    VRHINO_TEST_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(vrhino-product-input-schema-tests
    tests/product_input_schema_tests.cpp)
target_link_libraries(vrhino-product-input-schema-tests PRIVATE
    vrhino_model_package)
target_compile_options(vrhino-product-input-schema-tests PRIVATE
    -Wall -Wextra -Werror)

add_executable(vrhino-product-schema-spec-tests
    tests/product_schema_spec_tests.cpp)
target_link_libraries(vrhino-product-schema-spec-tests PRIVATE
    vrhino_model_package)
target_compile_options(vrhino-product-schema-spec-tests PRIVATE
    -Wall -Wextra -Werror)
target_compile_definitions(vrhino-product-schema-spec-tests PRIVATE
    VRHINO_TEST_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

if(VRHINO_ENABLE_PRODUCT_CLI)
    # Product-layer registry, HTTPS downloader, and unified model CLI. Registry
    # metadata and downloads are deliberately outside Runtime/Backend targets.
    execute_process(
        COMMAND git rev-parse HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE VRHINO_BUILD_GIT_HEAD
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(NOT VRHINO_BUILD_GIT_HEAD)
        set(VRHINO_BUILD_GIT_HEAD "unknown")
    endif()

    # Reusable, privacy-safe Product/application metadata projections. Both
    # the human CLI and Native API derive version/model responses here.
    add_library(vrhino_product_metadata
        src/product/model_list.cpp
        src/product/version.cpp)
    target_include_directories(vrhino_product_metadata PUBLIC include)
    target_link_libraries(vrhino_product_metadata PUBLIC vrhino_model_package)
    target_compile_definitions(vrhino_product_metadata PRIVATE
        VRHINO_PRODUCT_VERSION="${VRHINO_PRODUCT_VERSION}"
        VRHINO_GIT_HEAD="${VRHINO_BUILD_GIT_HEAD}")
    target_compile_options(vrhino_product_metadata PRIVATE -Wall -Wextra -Werror)

    add_executable(vrhino-release-version-tests
        tests/release_version_tests.cpp)
    target_link_libraries(vrhino-release-version-tests PRIVATE
        vrhino_product_metadata)
    target_compile_definitions(vrhino-release-version-tests PRIVATE
        VRHINO_TEST_CANONICAL_VERSION_FILE="${VRHINO_RELEASE_VERSION_FILE}"
        VRHINO_TEST_PACKAGE_VERSION_FILE="${VRHINO_RELEASE_METADATA_DIR}/VERSION"
        VRHINO_TEST_ARCHIVE_NAME_FILE="${VRHINO_RELEASE_METADATA_DIR}/archive-name.txt"
        VRHINO_TEST_PROJECT_VERSION="${PROJECT_VERSION}")
    target_compile_options(vrhino-release-version-tests PRIVATE
        -Wall -Wextra -Werror)

    # Bounded HTTP/1.1 metadata transport. cpp-httplib is vendored and linked
    # header-only with TLS and compression support disabled. These bounds are
    # internal alpha server limits, not Native API schema fields.
    add_library(vrhino_native_api src/api/server.cpp)
    target_include_directories(vrhino_native_api
        PUBLIC include
        PRIVATE third_party/cpp-httplib)
    target_link_libraries(vrhino_native_api PUBLIC
        vrhino_product_metadata vrhino_model_package
        vrhino_product_run_request vrhino_application_jobs Threads::Threads)
    target_compile_definitions(vrhino_native_api PRIVATE
        CPPHTTPLIB_REQUEST_URI_MAX_LENGTH=2048
        CPPHTTPLIB_HEADER_MAX_LENGTH=4096
        CPPHTTPLIB_HEADER_MAX_COUNT=64
        CPPHTTPLIB_MAX_LINE_LENGTH=4096
        CPPHTTPLIB_PAYLOAD_MAX_LENGTH=65536
        CPPHTTPLIB_LISTEN_BACKLOG=16)
    target_compile_options(vrhino_native_api PRIVATE -Wall -Wextra -Werror)

    add_executable(vrhino-native-api-tests tests/native_api_tests.cpp)
    target_link_libraries(vrhino-native-api-tests PRIVATE
        vrhino_native_api vrhino_product_metadata Threads::Threads)
    target_compile_definitions(vrhino-native-api-tests PRIVATE
        VRHINO_TEST_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
        VRHINO_REPOSITORY_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
    target_compile_options(vrhino-native-api-tests PRIVATE -Wall -Wextra -Werror)

    add_executable(vrhino-native-api-job-tests
        tests/native_api_job_tests.cpp)
    target_link_libraries(vrhino-native-api-job-tests PRIVATE
        vrhino_native_api vrhino_product_metadata Threads::Threads)
    target_compile_definitions(vrhino-native-api-job-tests PRIVATE
        VRHINO_TEST_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(vrhino-native-api-job-tests PRIVATE
        -Wall -Wextra -Werror)

    add_executable(vrhino-native-api-event-tests
        tests/native_api_event_tests.cpp)
    target_link_libraries(vrhino-native-api-event-tests PRIVATE
        vrhino_native_api vrhino_product_metadata Threads::Threads)
    target_compile_definitions(vrhino-native-api-event-tests PRIVATE
        VRHINO_TEST_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    target_compile_options(vrhino-native-api-event-tests PRIVATE
        -Wall -Wextra -Werror)

    add_executable(vrhino-cpp-httplib-legal-tests
        tests/cpp_httplib_legal_tests.cpp)
    target_link_libraries(vrhino-cpp-httplib-legal-tests PRIVATE
        vrhino_model_package)
    target_compile_definitions(vrhino-cpp-httplib-legal-tests PRIVATE
        VRHINO_TEST_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
        VRHINO_REPOSITORY_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
    target_compile_options(vrhino-cpp-httplib-legal-tests PRIVATE
        -Wall -Wextra -Werror)

    if(VRHINO_ENABLE_TOKENIZERS)
        add_executable(vrhino-tokenizer-build-source-tests
            tests/tokenizer_build_source_tests.cpp)
        target_link_libraries(vrhino-tokenizer-build-source-tests PRIVATE
            vrhino_model_package)
        target_compile_definitions(vrhino-tokenizer-build-source-tests PRIVATE
            VRHINO_TEST_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}"
            VRHINO_REPOSITORY_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
        target_compile_options(vrhino-tokenizer-build-source-tests PRIVATE
            -Wall -Wextra -Werror)
    endif()

    find_package(CURL REQUIRED)
    add_library(vrhino_registry src/product/registry.cpp)
    target_include_directories(vrhino_registry PUBLIC include)
    target_link_libraries(vrhino_registry PUBLIC vrhino_model_package vrhino_component_package
        PRIVATE CURL::libcurl)
    target_compile_options(vrhino_registry PRIVATE -Wall -Wextra -Werror)

    # Provider-extensible upstream source acquisition and immutable source
    # cache. This remains product orchestration and never enters Runtime/CUDA.
    add_library(vrhino_source_acquisition src/product/source_acquisition.cpp)
    target_include_directories(vrhino_source_acquisition PUBLIC include)
    target_link_libraries(vrhino_source_acquisition PUBLIC vrhino_registry)
    target_compile_options(vrhino_source_acquisition PRIVATE -Wall -Wextra -Werror)

    add_executable(vrhino-native-source-acquisition-tests
        tests/native_source_acquisition_tests.cpp)
    target_link_libraries(vrhino-native-source-acquisition-tests PRIVATE
        vrhino_source_acquisition)

    # Generic product pull resolver: declarative source-backed plans reuse the
    # qualified acquisition/converter paths; ready package fallback reuses the
    # existing registry client. It has no Runtime or architecture dependency.
    add_library(vrhino_pull_orchestration src/product/pull_orchestration.cpp)
    target_include_directories(vrhino_pull_orchestration PUBLIC include)
    target_link_libraries(vrhino_pull_orchestration PUBLIC
        vrhino_source_acquisition vrhino_converter)
    target_compile_options(vrhino_pull_orchestration PRIVATE -Wall -Wextra -Werror)

    add_executable(vrhino-native-pull-orchestration-tests
        tests/native_pull_orchestration_tests.cpp)
    target_link_libraries(vrhino-native-pull-orchestration-tests PRIVATE
        vrhino_pull_orchestration)

    add_executable(vrhino-product-schema-successor-orchestration-tests
        tests/product_schema_successor_orchestration_tests.cpp)
    target_link_libraries(
        vrhino-product-schema-successor-orchestration-tests PRIVATE
        vrhino_pull_orchestration)
    target_compile_options(
        vrhino-product-schema-successor-orchestration-tests PRIVATE
        -Wall -Wextra -Werror)
    target_compile_definitions(
        vrhino-product-schema-successor-orchestration-tests PRIVATE
        VRHINO_TEST_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

    add_executable(vrhino-public-musetalk-product-tests
        tests/public_musetalk_product_tests.cpp)
    target_link_libraries(vrhino-public-musetalk-product-tests PRIVATE
        vrhino_model_package vrhino_pull_orchestration)
    target_compile_options(vrhino-public-musetalk-product-tests PRIVATE
        -Wall -Wextra -Werror)
    target_compile_definitions(vrhino-public-musetalk-product-tests PRIVATE
        VRHINO_TEST_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    add_executable(vrhino-public-latentsync-product-tests
        tests/public_latentsync_product_tests.cpp)
    target_link_libraries(vrhino-public-latentsync-product-tests PRIVATE
        vrhino_model_package vrhino_pull_orchestration)
    target_compile_options(vrhino-public-latentsync-product-tests PRIVATE
        -Wall -Wextra -Werror)
    target_compile_definitions(vrhino-public-latentsync-product-tests PRIVATE
        VRHINO_TEST_SOURCE_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
    if(VRHINO_ENABLE_CUDA)
        add_executable(vrhino-latentsync-product-rng-oracle-tests
            tests/latentsync_product_rng_oracle_tests.cpp)
        target_include_directories(vrhino-latentsync-product-rng-oracle-tests
            PRIVATE tests)
        target_link_libraries(vrhino-latentsync-product-rng-oracle-tests PRIVATE
            vrhino_cuda_backend vrhino_lip_sync_diffusion_workflow)
        target_compile_options(vrhino-latentsync-product-rng-oracle-tests PRIVATE
            -Wall -Wextra -Werror)
        add_executable(vrhino-musetalk-successor-oracle-tests
            tests/musetalk_successor_oracle_tests.cpp)
        target_include_directories(vrhino-musetalk-successor-oracle-tests PRIVATE tests)
        target_link_libraries(vrhino-musetalk-successor-oracle-tests PRIVATE
            vrhino_cuda_backend vrhino_face_mask vrhino_lip_sync_workflow)
        target_compile_options(vrhino-musetalk-successor-oracle-tests PRIVATE
            -Wall -Wextra -Werror)
    endif()

    add_executable(vrhino-wan-product-integration-tests
        tests/wan_product_integration_tests.cpp)
    target_link_libraries(vrhino-wan-product-integration-tests PRIVATE
        vrhino_pull_orchestration)

    add_executable(vrhino-mochi-product-integration-tests
        tests/mochi_product_integration_tests.cpp)
    target_link_libraries(vrhino-mochi-product-integration-tests PRIVATE
        vrhino_pull_orchestration)

    add_executable(vrhino tools/vrhino_cli.cpp)
    target_link_libraries(vrhino PRIVATE vrhino_registry vrhino_converter
        vrhino_source_acquisition vrhino_pull_orchestration vrhino_product_progress
        vrhino_native_api vrhino_product_metadata)
    add_custom_command(TARGET vrhino POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/specs"
            "$<TARGET_FILE_DIR:vrhino>/share/vrhino/converters"
        COMMENT "Staging Native converter specifications")
endif()

if(VRHINO_ENABLE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "The Metal backend requires Apple platforms")
    endif()
    enable_language(OBJCXX)
    find_library(METAL_FRAMEWORK Metal REQUIRED)
    find_library(FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders REQUIRED)

    add_library(vrhino_metal_backend src/backend/metal/metal_backend.mm)
    target_include_directories(vrhino_metal_backend PUBLIC include)
    target_link_libraries(vrhino_metal_backend PUBLIC vrhino_native_common
        ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK} ${MPS_FRAMEWORK})
    target_compile_options(vrhino_metal_backend PRIVATE
        $<$<COMPILE_LANGUAGE:OBJCXX>:-Wall;-Wextra;-Werror;-fno-objc-arc>)

    add_executable(vrhino-metal cli/main.cpp)
    target_compile_definitions(vrhino-metal PRIVATE VRHINO_USE_METAL_BACKEND=1)
    target_link_libraries(vrhino-metal PRIVATE vrhino_metal_backend)

    add_executable(vrhino-metal-tests tests/primitive_tests.cpp)
    target_compile_definitions(vrhino-metal-tests PRIVATE VRHINO_USE_METAL_BACKEND=1)
    target_link_libraries(vrhino-metal-tests PRIVATE vrhino_metal_backend)

    add_executable(vrhino-metal-negative-tests tests/metal_negative_tests.cpp)
    target_link_libraries(vrhino-metal-negative-tests PRIVATE vrhino_metal_backend)

    # Research-only gate: executes the first canonical architecture graph on
    # Metal before the full VRM E2E target is admitted.
    add_executable(vrhino-metal-first-graph tests/metal_first_graph.cpp)
    target_link_libraries(vrhino-metal-first-graph PRIVATE vrhino_metal_backend)

    add_executable(vrhino-metal-memory-smoke tests/metal_memory_smoke.cpp)
    target_link_libraries(vrhino-metal-memory-smoke PRIVATE vrhino_metal_backend)

    # Observation-only generic Linear reproducer for fixed numerical audits.
    add_executable(vrhino-metal-linear-reproducer
        tests/metal_linear_reproducer.cpp tests/phase13_safetensors.cpp)
    target_include_directories(vrhino-metal-linear-reproducer PRIVATE tests)
    target_link_libraries(vrhino-metal-linear-reproducer PRIVATE vrhino_metal_backend)

    # Observation-only generic Attention reproducer for fixed numerical audits.
    add_executable(vrhino-metal-attention-reproducer tests/metal_attention_reproducer.cpp)
    target_link_libraries(vrhino-metal-attention-reproducer PRIVATE vrhino_metal_backend)

    # Observation-only generic RMSNorm reproducer for fixed numerical audits.
    add_executable(vrhino-metal-rms-norm-reproducer
        tests/metal_rms_norm_reproducer.cpp tests/phase13_safetensors.cpp)
    target_include_directories(vrhino-metal-rms-norm-reproducer PRIVATE tests)
    target_link_libraries(vrhino-metal-rms-norm-reproducer PRIVATE vrhino_metal_backend)

    if(VRHINO_ENABLE_TOKENIZERS)
        add_executable(vrhino-metal-phase13-conditioning-probe
            tests/phase13_conditioning_probe.cpp tests/phase13_safetensors.cpp)
        target_include_directories(vrhino-metal-phase13-conditioning-probe PRIVATE tests)
        target_compile_definitions(vrhino-metal-phase13-conditioning-probe PRIVATE
            VRHINO_USE_METAL_BACKEND=1)
        target_link_libraries(vrhino-metal-phase13-conditioning-probe PRIVATE
            vrhino_metal_backend vrhino_native_tokenizer)

        add_executable(vrhino-metal-phase13d-prompt-video
            tests/phase13d_prompt_video.cpp tests/phase13_safetensors.cpp)
        target_include_directories(vrhino-metal-phase13d-prompt-video PRIVATE tests)
        target_compile_definitions(vrhino-metal-phase13d-prompt-video PRIVATE
            VRHINO_USE_METAL_BACKEND=1)
        target_link_libraries(vrhino-metal-phase13d-prompt-video PRIVATE
            vrhino_metal_backend vrhino_native_tokenizer)
    endif()
endif()

if(VRHINO_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89)
    endif()

    find_package(CUDAToolkit REQUIRED)
    find_library(CUDNN_LIBRARY cudnn REQUIRED)

    add_library(vrhino_cuda_backend
        src/backend/cuda/cuda_backend.cu
        src/backend/cuda/cudnn_conv.cpp
        src/backend/cuda/cudnn_sdpa.cpp)
    target_include_directories(vrhino_cuda_backend PUBLIC include PRIVATE
        third_party/cudnn_frontend/include)
    target_link_libraries(vrhino_cuda_backend PUBLIC
        vrhino_native_common CUDA::cudart CUDA::cublas CUDA::cublasLt CUDA::curand ${CUDNN_LIBRARY})
    target_compile_options(vrhino_cuda_backend PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

    add_executable(vrhino-neural-graph-cuda-tests tests/neural_graph_cuda_tests.cpp)
    add_executable(vrhino-wan-graph-dual-path-cuda-tests tests/wan_graph_dual_path_tests.cpp)
    add_executable(vrhino-wan-graph-optin-cuda-tests tests/wan_graph_optin_tests.cpp)
    target_compile_definitions(vrhino-wan-graph-optin-cuda-tests PRIVATE VRHINO_TEST_CUDA=1)
    target_link_libraries(vrhino-wan-graph-optin-cuda-tests PRIVATE vrhino_neural_graph vrhino_cuda_backend)
    target_compile_options(vrhino-wan-graph-optin-cuda-tests PRIVATE -Wall -Wextra -Werror -ffp-contract=off)
    target_link_options(vrhino-wan-graph-optin-cuda-tests PRIVATE -Wl,--wrap=cudaGetLastError -Wl,--wrap=cudaDeviceSynchronize)
    target_compile_definitions(vrhino-wan-graph-dual-path-cuda-tests PRIVATE VRHINO_TEST_CUDA=1)
    target_link_libraries(vrhino-wan-graph-dual-path-cuda-tests PRIVATE vrhino_neural_graph vrhino_cuda_backend)
    target_compile_options(vrhino-wan-graph-dual-path-cuda-tests PRIVATE -Wall -Wextra -Werror -ffp-contract=off)
    target_include_directories(vrhino-neural-graph-cuda-tests PRIVATE src/runtime)
    target_link_libraries(vrhino-neural-graph-cuda-tests PRIVATE
        vrhino_neural_graph vrhino_cuda_backend)
    target_compile_options(vrhino-neural-graph-cuda-tests PRIVATE
        -Wall -Wextra -Werror -ffp-contract=off)
    # Test-only observation; the wrapper always calls the real CUDA function.
    target_link_options(vrhino-neural-graph-cuda-tests PRIVATE
        -Wl,--wrap=cudaDeviceSynchronize)

    add_executable(vrhino-ltx-graph-cfg-cuda-tests tests/ltx_graph_cfg_cuda_tests.cpp)
    target_include_directories(vrhino-ltx-graph-cfg-cuda-tests PRIVATE src/architectures)
    target_link_libraries(vrhino-ltx-graph-cfg-cuda-tests PRIVATE vrhino_ltx_graph vrhino_cuda_backend)
    target_compile_options(vrhino-ltx-graph-cfg-cuda-tests PRIVATE -Wall -Wextra -Werror -ffp-contract=off)
    # Test-only call observer always forwards to the real unchanged executor.
    target_link_options(vrhino-ltx-graph-cfg-cuda-tests PRIVATE
        -Wl,--wrap=_ZN6vrhino12neural_graph8evaluateERNS_7BackendERKNS0_5GraphERKSt6vectorINS_6TensorESaIS7_EESB_)

    add_executable(vrhino-ltx-graph-optin-cuda-tests tests/ltx_graph_optin_cuda_tests.cpp)
    target_include_directories(vrhino-ltx-graph-optin-cuda-tests PRIVATE src/architectures)
    target_link_libraries(vrhino-ltx-graph-optin-cuda-tests PRIVATE vrhino_ltx_graph vrhino_cuda_backend)
    target_compile_options(vrhino-ltx-graph-optin-cuda-tests PRIVATE -Wall -Wextra -Werror -ffp-contract=off)

    add_executable(vrhino-ltx-graph-binding-cuda-tests tests/neural_graph_bf16_ltx_admission_tests.cpp)
    target_compile_definitions(vrhino-ltx-graph-binding-cuda-tests PRIVATE VRHINO_LTX_BINDING_CUDA=1)
    target_link_libraries(vrhino-ltx-graph-binding-cuda-tests PRIVATE vrhino_ltx_graph vrhino_cuda_backend)
    target_compile_options(vrhino-ltx-graph-binding-cuda-tests PRIVATE -Wall -Wextra -Werror)

    add_executable(vrhino-ltx-graph-first-cuda-tests tests/ltx_graph_first_cuda_tests.cpp)
    target_include_directories(vrhino-ltx-graph-first-cuda-tests PRIVATE src/architectures)
    target_link_libraries(vrhino-ltx-graph-first-cuda-tests PRIVATE vrhino_ltx_graph vrhino_cuda_backend)
    target_compile_options(vrhino-ltx-graph-first-cuda-tests PRIVATE -Wall -Wextra -Werror -ffp-contract=off)

    add_executable(vrhino-neural-graph-ltx-cuda-tests tests/neural_graph_ltx_cuda_tests.cpp)
    target_include_directories(vrhino-neural-graph-ltx-cuda-tests PRIVATE src/runtime)
    target_link_libraries(vrhino-neural-graph-ltx-cuda-tests PRIVATE
        vrhino_neural_graph vrhino_cuda_backend)
    target_compile_options(vrhino-neural-graph-ltx-cuda-tests PRIVATE
        -Wall -Wextra -Werror -ffp-contract=off)
    target_link_options(vrhino-neural-graph-ltx-cuda-tests PRIVATE
        -Wl,--wrap=cudaDeviceSynchronize)

    add_executable(vrhino-neural-graph-extension-cuda-tests tests/neural_graph_extension_tests.cpp)
    target_include_directories(vrhino-neural-graph-extension-cuda-tests PRIVATE src/runtime)
    target_compile_definitions(vrhino-neural-graph-extension-cuda-tests PRIVATE VRHINO_TEST_CUDA)
    target_link_libraries(vrhino-neural-graph-extension-cuda-tests PRIVATE vrhino_neural_graph vrhino_cuda_backend)
    target_compile_options(vrhino-neural-graph-extension-cuda-tests PRIVATE -Wall -Wextra -Werror -ffp-contract=off)

    add_executable(vrhino-native cli/main.cpp)
    target_link_libraries(vrhino-native PRIVATE vrhino_cuda_backend)

    add_executable(vrhino-neural-graph-bf16-cuda-tests tests/neural_graph_bf16_tests.cpp)
    target_include_directories(vrhino-neural-graph-bf16-cuda-tests PRIVATE src/runtime)
    target_compile_definitions(vrhino-neural-graph-bf16-cuda-tests PRIVATE VRHINO_TEST_CUDA=1)
    target_link_libraries(vrhino-neural-graph-bf16-cuda-tests PRIVATE vrhino_neural_graph vrhino_cuda_backend)
    target_compile_options(vrhino-neural-graph-bf16-cuda-tests PRIVATE -Wall -Wextra -Werror)

    add_executable(vrhino-native-tests tests/primitive_tests.cpp)

    add_executable(vrhino-neural-graph-layernorm-cuda-tests tests/neural_graph_layernorm_tests.cpp)
    target_include_directories(vrhino-neural-graph-layernorm-cuda-tests PRIVATE src/runtime)
    target_compile_definitions(vrhino-neural-graph-layernorm-cuda-tests PRIVATE VRHINO_TEST_CUDA=1)
    target_link_libraries(vrhino-neural-graph-layernorm-cuda-tests PRIVATE vrhino_neural_graph vrhino_cuda_backend)
    target_compile_options(vrhino-neural-graph-layernorm-cuda-tests PRIVATE -Wall -Wextra -Werror -ffp-contract=off)
    target_link_libraries(vrhino-native-tests PRIVATE vrhino_cuda_backend)

    add_executable(vrhino-native-negative-tests tests/negative_tests.cpp)
    target_link_libraries(vrhino-native-negative-tests PRIVATE vrhino_cuda_backend)

    add_executable(vrhino-whisper-tiny-oracle-tests
        tests/whisper_tiny_oracle_tests.cpp)
    target_link_libraries(vrhino-whisper-tiny-oracle-tests PRIVATE
        vrhino_cuda_backend vrhino_audio_conditioning vrhino_converter)

    add_executable(vrhino-autoencoder-kl-oracle-tests
        tests/autoencoder_kl_oracle_tests.cpp)
    target_link_libraries(vrhino-autoencoder-kl-oracle-tests PRIVATE
        vrhino_cuda_backend vrhino_converter)

    add_executable(vrhino-conditional-unet-2d-oracle-tests
        tests/conditional_unet_2d_oracle_tests.cpp)
    target_link_libraries(vrhino-conditional-unet-2d-oracle-tests PRIVATE
        vrhino_cuda_backend vrhino_converter)

    add_executable(vrhino-temporal-conditional-unet-oracle-tests
        tests/temporal_conditional_unet_oracle_tests.cpp)
    target_link_libraries(vrhino-temporal-conditional-unet-oracle-tests PRIVATE
        vrhino_cuda_backend vrhino_converter)

    add_executable(vrhino-vision-detector-oracle-tests
        tests/vision_detector_oracle_tests.cpp)
    target_link_libraries(vrhino-vision-detector-oracle-tests PRIVATE
        vrhino_cuda_backend vrhino_converter vrhino_face_roi)

    add_executable(vrhino-blazeface-oracle-tests
        tests/blazeface_oracle_tests.cpp)
    target_link_libraries(vrhino-blazeface-oracle-tests PRIVATE
        vrhino_cuda_backend vrhino_converter vrhino_face_roi
        vrhino_pose_geometry)

    add_executable(vrhino-pose-estimator-oracle-tests
        tests/pose_estimator_oracle_tests.cpp)
    target_link_libraries(vrhino-pose-estimator-oracle-tests PRIVATE
        vrhino_cuda_backend vrhino_converter vrhino_pose_geometry vrhino_face_roi)

    add_executable(vrhino-semantic-segmenter-oracle-tests
        tests/semantic_segmenter_oracle_tests.cpp)
    target_link_libraries(vrhino-semantic-segmenter-oracle-tests PRIVATE
        vrhino_cuda_backend vrhino_converter vrhino_face_mask)

    add_executable(vrhino-selfie-multiclass-oracle-tests
        tests/selfie_multiclass_oracle_tests.cpp)
    target_link_libraries(vrhino-selfie-multiclass-oracle-tests PRIVATE
        vrhino_cuda_backend vrhino_converter vrhino_face_mask)

    add_executable(vrhino-lip-sync-workflow-oracle-tests
        tests/lip_sync_workflow_oracle_tests.cpp)
    target_link_libraries(vrhino-lip-sync-workflow-oracle-tests PRIVATE
        vrhino_lip_sync_workflow vrhino_audio_conditioning
        vrhino_pose_geometry vrhino_face_roi vrhino_face_mask
        vrhino_cuda_backend)

    add_executable(vrhino-latentsync-alignment-oracle-tests
        tests/latentsync_alignment_oracle_tests.cpp)
    target_link_libraries(vrhino-latentsync-alignment-oracle-tests PRIVATE
        vrhino_lip_sync_diffusion_workflow vrhino_cuda_backend)

    add_executable(vrhino-latentsync-workflow-oracle-tests
        tests/latentsync_workflow_oracle_tests.cpp)
    target_link_libraries(vrhino-latentsync-workflow-oracle-tests PRIVATE
        vrhino_lip_sync_diffusion_workflow vrhino_cuda_backend)

    add_executable(vrhino-phase20k-large-conv-tests
        tests/phase20k_large_conv_tests.cu)
    target_link_libraries(vrhino-phase20k-large-conv-tests PRIVATE
        vrhino_cuda_backend)

    add_executable(vrhino-phase20p-last-use-tests
        tests/phase20p_last_use_tests.cpp)
    target_link_libraries(vrhino-phase20p-last-use-tests PRIVATE
        vrhino_cuda_backend)

    add_executable(vrhino-phase20q-component-tiling-tests
        tests/phase20q_component_tiling_tests.cpp)
    target_link_libraries(vrhino-phase20q-component-tiling-tests PRIVATE
        vrhino_cuda_backend)

    add_executable(vrhino-phase22d-recursive-tiling-tests
        tests/phase22d_recursive_tiling_tests.cpp)
    target_link_libraries(vrhino-phase22d-recursive-tiling-tests PRIVATE
        vrhino_cuda_backend)

    add_executable(vrhino-phase22d-mochi-tiled-decode
        tests/phase22d_mochi_tiled_decode.cpp)
    target_link_libraries(vrhino-phase22d-mochi-tiled-decode PRIVATE
        vrhino_cuda_backend)

    add_executable(vrhino-phase22e-mochi-binding-tests
        tests/phase22e_mochi_binding_tests.cpp)
    target_link_libraries(vrhino-phase22e-mochi-binding-tests PRIVATE
        vrhino_cuda_backend)

    add_executable(vrhino-phase22f-sdpa-workspace
        tests/phase22f_sdpa_workspace.cu)
    target_link_libraries(vrhino-phase22f-sdpa-workspace PRIVATE
        vrhino_cuda_backend)

    add_executable(vrhino-phase20w-component-execution-binding-tests
        tests/phase20w_component_execution_binding_tests.cpp)
    target_link_libraries(vrhino-phase20w-component-execution-binding-tests PRIVATE
        vrhino_cuda_backend)

    add_executable(vrhino-phase20w-production-decode
        tests/phase20w_production_decode.cpp)
    target_link_libraries(vrhino-phase20w-production-decode PRIVATE
        vrhino_cuda_backend)

    add_executable(vrhino-phase20w-fast-canary
        tests/phase20w_fast_canary.cpp)
    target_link_libraries(vrhino-phase20w-fast-canary PRIVATE
        vrhino_cuda_backend)

    add_executable(vrhino-conditioning-mask-contract-tests
        tests/conditioning_mask_contract_tests.cpp)
    target_link_libraries(vrhino-conditioning-mask-contract-tests PRIVATE
        vrhino_cuda_backend)

    add_executable(vrhino-hunyuan-sampling-short-chain
        tests/hunyuan_sampling_short_chain.cpp)
    target_link_libraries(vrhino-hunyuan-sampling-short-chain PRIVATE
        vrhino_cuda_backend)

    add_executable(vrhino-phase21b-ltx-sampling-short-chain
        tests/phase21b_ltx_sampling_short_chain.cpp)
    target_link_libraries(vrhino-phase21b-ltx-sampling-short-chain PRIVATE
        vrhino_cuda_backend)

    # Model-neutral, data-driven PrecisionPolicy qualification harness. It is
    # not part of the production CLI and accepts no architecture identity.
    add_executable(vrhino-precision-canary tests/precision_canary.cpp)
    target_link_libraries(vrhino-precision-canary PRIVATE vrhino_cuda_backend)

    add_executable(vrhino-precision-operation-contract-tests
        tests/precision_operation_contract_tests.cpp)
    target_link_libraries(vrhino-precision-operation-contract-tests PRIVATE
        vrhino_cuda_backend)

    # Research-only validation harnesses; these are not product CLI targets.
    add_executable(vrhino-phase8-batch tests/phase8_batch.cpp)
    target_link_libraries(vrhino-phase8-batch PRIVATE vrhino_cuda_backend)

    add_executable(vrhino-phase9-quant-tests tests/phase9_quant_tests.cpp)
    target_link_libraries(vrhino-phase9-quant-tests PRIVATE vrhino_cuda_backend)

    add_executable(vrhino-phase9-batch tests/phase9_batch.cpp)
    target_link_libraries(vrhino-phase9-batch PRIVATE vrhino_cuda_backend)

    add_executable(vrhino-phase10-cuda-memory-tests tests/phase10_cuda_memory_tests.cpp)
    target_link_libraries(vrhino-phase10-cuda-memory-tests PRIVATE vrhino_cuda_backend)

    add_executable(vrhino-phase10-batch tests/phase10_batch.cpp)
    target_link_libraries(vrhino-phase10-batch PRIVATE vrhino_cuda_backend)

    # Model-neutral numerical/resource probe for bounded-workspace Attention.
    add_executable(vrhino-phase14-attention-capacity tests/phase14_attention_capacity.cpp)
    target_link_libraries(vrhino-phase14-attention-capacity PRIVATE vrhino_cuda_backend)

    add_executable(vrhino-phase15-attention-benchmark tests/phase15_attention_benchmark.cpp)
    target_link_libraries(vrhino-phase15-attention-benchmark PRIVATE vrhino_cuda_backend)

    add_executable(vrhino-phase15-attention-audit tests/phase15_attention_audit.cpp)
    target_link_libraries(vrhino-phase15-attention-audit PRIVATE vrhino_cuda_backend)

    # Model-neutral Phase 16R same-input BF16 long-Attention gate.
    add_executable(vrhino-phase16r-bf16-attention tests/phase16r_bf16_attention.cpp)
    target_link_libraries(vrhino-phase16r-bf16-attention PRIVATE vrhino_cuda_backend)

    add_executable(vrhino-phase18c-bf16-attention tests/phase18c_bf16_attention.cpp)
    target_link_libraries(vrhino-phase18c-bf16-attention PRIVATE vrhino_cuda_backend)

    add_executable(vrhino-phase19c-cudnn-sdpa tests/phase19c_cudnn_sdpa.cpp)
    target_link_libraries(vrhino-phase19c-cudnn-sdpa PRIVATE vrhino_cuda_backend)
    add_executable(vrhino-phase20d-prefix-mask-sdpa tests/phase20d_prefix_mask_sdpa.cpp)
    target_link_libraries(vrhino-phase20d-prefix-mask-sdpa PRIVATE vrhino_cuda_backend)
    add_executable(vrhino-phase21h-attention-mask-cache
        tests/phase21h_attention_mask_cache.cpp)
    target_link_libraries(vrhino-phase21h-attention-mask-cache PRIVATE vrhino_cuda_backend)
    add_executable(vrhino-phase21j-prepared-tensor-tests
        tests/phase21j_prepared_tensor_tests.cpp)
    target_link_libraries(vrhino-phase21j-prepared-tensor-tests PRIVATE
        vrhino_cuda_backend)
    add_executable(vrhino-phase20d-sdpa-crossover tests/phase20d_sdpa_crossover.cpp)
    target_link_libraries(vrhino-phase20d-sdpa-crossover PRIVATE vrhino_cuda_backend)

    add_executable(vrhino-phase18c-bf16-linear-benchmark
        tests/phase18c_bf16_linear_benchmark.cpp)
    target_link_libraries(vrhino-phase18c-bf16-linear-benchmark PRIVATE vrhino_cuda_backend)

    # Phase 16R2 Native-only VAE precision decomposition harness.
    add_executable(vrhino-phase16r2-vae-replay tests/phase16r2_vae_replay.cpp)
    target_link_libraries(vrhino-phase16r2-vae-replay PRIVATE vrhino_cuda_backend)

    # Phase 16R2 Native-only same-conditioning precision replay harness.
    add_executable(vrhino-phase16r2-sampling-replay tests/phase16r2_sampling_replay.cpp)
    target_link_libraries(vrhino-phase16r2-sampling-replay PRIVATE vrhino_cuda_backend)

    # Reference-only calibration harness: execute exactly one architecture
    # denoiser evaluation from an externally frozen latent/timestep. It bypasses
    # SamplingRuntime and VAE and does not alter production execution semantics.
    add_executable(vrhino-precision-denoiser-calibration
        tests/precision_denoiser_calibration.cpp)
    target_link_libraries(vrhino-precision-denoiser-calibration PRIVATE
        vrhino_cuda_backend)

    if(VRHINO_ENABLE_TOKENIZERS)
        # Product prompt-to-MP4 orchestration. The package declaration drives
        # every component; this layer contains no architecture-name dispatch.
        add_library(vrhino_product_run
            src/product/run.cpp
            src/product/lip_sync_product.cpp
            src/product/lip_sync_diffusion_product.cpp
            src/product/safetensors.cpp)
        target_include_directories(vrhino_product_run PUBLIC include)
        target_link_libraries(vrhino_product_run PUBLIC
            vrhino_product_run_session vrhino_cuda_backend vrhino_native_tokenizer
            vrhino_audio_conditioning vrhino_face_roi vrhino_pose_geometry
            vrhino_face_mask vrhino_lip_sync_workflow
            vrhino_lip_sync_diffusion_workflow)
        target_compile_options(vrhino_product_run PRIVATE -Wall -Wextra -Werror)
        if(VRHINO_ENABLE_PRODUCT_CLI)
            target_link_libraries(vrhino PRIVATE vrhino_product_run)
            target_compile_definitions(vrhino PRIVATE VRHINO_PRODUCT_RUN_ENABLED=1)
            # Product `run` resolves only a sibling executable and never the
            # user's PATH. Build/release staging copies the selected native
            # encoder so an installed user is not required to install FFmpeg.
            set(VRHINO_FFMPEG_EXECUTABLE "" CACHE FILEPATH
                "Native FFmpeg executable staged beside the vrhino product CLI")
            if(VRHINO_FFMPEG_EXECUTABLE)
                add_custom_command(TARGET vrhino POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${VRHINO_FFMPEG_EXECUTABLE}"
                        "$<TARGET_FILE_DIR:vrhino>/vrhino-ffmpeg"
                    COMMENT "Staging bundled native MP4 encoder")
            else()
                message(WARNING
                    "vrhino run is enabled but no FFmpeg encoder was selected; "
                    "set VRHINO_FFMPEG_EXECUTABLE before product packaging")
            endif()
        endif()
        add_executable(vrhino-product-run-preflight-tests
            tests/product_run_preflight_tests.cpp)
        target_link_libraries(vrhino-product-run-preflight-tests PRIVATE vrhino_product_run)

        # Read-only, privacy-safe support report. This composes the existing
        # package, pull-plan, device-admission, and media readiness layers and
        # never enters Runtime execution or performs network access.
        if(VRHINO_ENABLE_PRODUCT_CLI)
            add_library(vrhino_product_doctor src/product/doctor.cpp)
            target_include_directories(vrhino_product_doctor PUBLIC include)
            target_link_libraries(vrhino_product_doctor PUBLIC
                vrhino_product_run vrhino_pull_orchestration)
            target_compile_options(vrhino_product_doctor PRIVATE -Wall -Wextra -Werror)
            target_link_libraries(vrhino PRIVATE vrhino_product_doctor)

            add_executable(vrhino-product-doctor-tests
                tests/product_doctor_tests.cpp)
            target_link_libraries(vrhino-product-doctor-tests PRIVATE
                vrhino_product_doctor)
        endif()

        add_executable(vrhino-phase13-conditioning-probe
            tests/phase13_conditioning_probe.cpp tests/phase13_safetensors.cpp)
        target_include_directories(vrhino-phase13-conditioning-probe PRIVATE tests)
        target_link_libraries(vrhino-phase13-conditioning-probe PRIVATE
            vrhino_cuda_backend vrhino_native_tokenizer)

        add_executable(vrhino-phase13d-prompt-video
            tests/phase13d_prompt_video.cpp tests/phase13_safetensors.cpp)
        target_include_directories(vrhino-phase13d-prompt-video PRIVATE tests)
        target_link_libraries(vrhino-phase13d-prompt-video PRIVATE
            vrhino_cuda_backend vrhino_native_tokenizer)

        # Audit-only exact-shape replay for the Phase 22I conditioning repair.
        # It keeps reference assets outside the Native production dependency
        # graph and does not provide a production execution path.
        add_executable(vrhino-phase22i-conditioning-fix-tests
            tests/phase22i_conditioning_fix_tests.cpp tests/phase13_safetensors.cpp)
        target_include_directories(vrhino-phase22i-conditioning-fix-tests PRIVATE tests)
        target_link_libraries(vrhino-phase22i-conditioning-fix-tests PRIVATE
            vrhino_cuda_backend vrhino_native_tokenizer)

    endif()
endif()

if(VRHINO_ENABLE_CUDA)
    add_executable(vrhino-wan-graph-synthetic-cuda-tests tests/wan_graph_synthetic_cuda_tests.cpp)
    target_compile_definitions(vrhino-wan-graph-synthetic-cuda-tests PRIVATE
        VRHINO_TEST_PRECISION_POLICY="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/precision/bf16-operation-contract-v1.json")
    target_link_libraries(vrhino-wan-graph-synthetic-cuda-tests PRIVATE vrhino_neural_graph vrhino_cuda_backend)
    target_compile_options(vrhino-wan-graph-synthetic-cuda-tests PRIVATE -Wall -Wextra -Werror -ffp-contract=off)
endif()
include(cmake/PublicTests.cmake)
