# Build-time source names only. Keep diagnostics and all execution policies.
# Apply after dependency discovery so generated targets and SDK headers share
# the same mappings. Debug builds retain their normal local source locations.
function(vrhino_privacy_targets directory)
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS targets)
        get_target_property(kind "${target}" TYPE)
        if(kind STREQUAL "INTERFACE_LIBRARY" OR kind STREQUAL "UTILITY")
            continue()
        endif()
        foreach(flag IN LISTS VRHINO_PRIVACY_MSVC_FLAGS)
            target_compile_options("${target}" PRIVATE
                "$<$<AND:$<CONFIG:Release,RelWithDebInfo,MinSizeRel>,$<COMPILE_LANGUAGE:C,CXX>>:${flag}>"
                "$<$<AND:$<CONFIG:Release,RelWithDebInfo,MinSizeRel>,$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=\"${flag}\">")
        endforeach()
        if(kind MATCHES "^(EXECUTABLE|SHARED_LIBRARY|MODULE_LIBRARY)$")
            target_link_options("${target}" PRIVATE
                "$<$<CONFIG:Release,RelWithDebInfo,MinSizeRel>:/PDBALTPATH:%_PDB%>")
        endif()
    endforeach()
    get_property(children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(child IN LISTS children)
        vrhino_privacy_targets("${child}")
    endforeach()
endfunction()

function(vrhino_configure_windows_privacy)
    get_filename_component(repo "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
    # Rust uses the last match. MSVC uses the first match, so reverse its list
    # below: an out-of-source build nested in a checkout must still map to build/.
    set(mappings "$ENV{USERPROFILE}|toolchain/user" "${repo}|vrhino"
        "${CMAKE_CURRENT_BINARY_DIR}|build"
        "$ENV{VCToolsInstallDir}|toolchain/msvc"
        "$ENV{WindowsSdkDir}|toolchain/windows-sdk"
        "${CUDAToolkit_ROOT}|toolchain/cuda" "$ENV{CUDA_PATH}|toolchain/cuda"
        "${CUDAToolkit_INCLUDE_DIRS}|toolchain/cuda/include"
        "${CUDNN_INCLUDE_DIR}|toolchain/cudnn/include")
    set(VRHINO_PRIVACY_MSVC_FLAGS)
    set(rust_flags)
    set(roots)
    foreach(mapping IN LISTS mappings)
        if(NOT mapping MATCHES "^(.+)\\|([^|]+)$")
            continue()
        endif()
        set(physical "${CMAKE_MATCH_1}")
        set(logical "${CMAKE_MATCH_2}")
        string(REPLACE "\\" "/" physical "${physical}")
        if(NOT IS_ABSOLUTE "${physical}")
            continue()
        endif()
        cmake_path(NORMAL_PATH physical OUTPUT_VARIABLE physical)
        string(REGEX REPLACE "[/\\]+$" "" physical "${physical}")
        file(TO_NATIVE_PATH "${physical}" native_path)
        list(APPEND roots "${physical}")
        list(APPEND VRHINO_PRIVACY_MSVC_FLAGS "/pathmap:${native_path}=${logical}")
        # rustc matches literal separators; cover native and CMake spellings.
        list(APPEND rust_flags "--remap-path-prefix=${physical}=${logical}"
            "--remap-path-prefix=${native_path}=${logical}")
    endforeach()
    list(REMOVE_DUPLICATES VRHINO_PRIVACY_MSVC_FLAGS)
    list(REVERSE VRHINO_PRIVACY_MSVC_FLAGS)
    list(PREPEND VRHINO_PRIVACY_MSVC_FLAGS /experimental:deterministic)
    list(REMOVE_DUPLICATES rust_flags)
    list(REMOVE_DUPLICATES roots)
    vrhino_privacy_targets("${CMAKE_CURRENT_SOURCE_DIR}")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/windows-build-privacy.cmake"
        "set(VRHINO_PRIVACY_RUST_FLAGS [==[${rust_flags}]==])\n"
        "set(VRHINO_PRIVACY_MSVC_FLAGS [==[${VRHINO_PRIVACY_MSVC_FLAGS}]==])\n")
    string(REPLACE ";" "\n" root_lines "${roots}")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/windows-build-privacy-roots.txt" "${root_lines}\n")
endfunction()

cmake_language(DEFER CALL vrhino_configure_windows_privacy)
