function(_vrhino_verify_source_inventory root inventory expected_inventory_hash)
    if(NOT IS_DIRECTORY "${root}")
        message(FATAL_ERROR "Required tokenizer source directory is missing: ${root}")
    endif()
    if(NOT EXISTS "${inventory}")
        message(FATAL_ERROR "Required tokenizer inventory is missing: ${inventory}")
    endif()

    file(SHA256 "${inventory}" actual_inventory_hash)
    if(NOT actual_inventory_hash STREQUAL expected_inventory_hash)
        message(FATAL_ERROR
            "Tokenizer inventory hash mismatch: ${inventory}\n"
            "expected ${expected_inventory_hash}\n"
            "actual   ${actual_inventory_hash}")
    endif()

    file(STRINGS "${inventory}" inventory_lines)
    set(expected_paths)
    foreach(line IN LISTS inventory_lines)
        if(NOT line MATCHES "^([0-9a-f]+)  (\\./.+)$")
            message(FATAL_ERROR "Malformed tokenizer inventory line: ${line}")
        endif()
        set(expected_hash "${CMAKE_MATCH_1}")
        string(LENGTH "${expected_hash}" expected_hash_length)
        if(NOT expected_hash_length EQUAL 64)
            message(FATAL_ERROR "Malformed tokenizer source hash: ${line}")
        endif()
        set(relative_with_dot "${CMAKE_MATCH_2}")
        string(SUBSTRING "${relative_with_dot}" 2 -1 relative)
        set(path "${root}/${relative}")
        if(NOT EXISTS "${path}" OR IS_DIRECTORY "${path}")
            message(FATAL_ERROR "Required tokenizer source file is missing: ${path}")
        endif()
        file(SHA256 "${path}" actual_hash)
        if(NOT actual_hash STREQUAL expected_hash)
            message(FATAL_ERROR
                "Tokenizer source hash mismatch: ${path}\n"
                "expected ${expected_hash}\n"
                "actual   ${actual_hash}")
        endif()
        list(APPEND expected_paths "${relative}")
    endforeach()

    cmake_policy(PUSH)
    cmake_policy(SET CMP0009 NEW)
    file(GLOB_RECURSE actual_entries
        LIST_DIRECTORIES FALSE
        RELATIVE "${root}"
        "${root}/*")
    cmake_policy(POP)
    set(actual_paths)
    foreach(relative IN LISTS actual_entries)
        if(NOT IS_SYMLINK "${root}/${relative}")
            list(APPEND actual_paths "${relative}")
        endif()
    endforeach()
    list(SORT expected_paths)
    list(SORT actual_paths)
    if(NOT expected_paths STREQUAL actual_paths)
        message(FATAL_ERROR
            "Tokenizer source file set differs from the pinned inventory: ${root}")
    endif()
endfunction()

function(vrhino_verify_tokenizer_build_sources source_root)
    _vrhino_verify_source_inventory(
        "${source_root}/tokenizers-cpp"
        "${source_root}/inventories/tokenizers-cpp-expanded.sha256"
        "63c37881d795895fc65322f648552c5cfeaa2db8438cb0c59fa9537d2ddbc04e")
    _vrhino_verify_source_inventory(
        "${source_root}/abseil-cpp"
        "${source_root}/inventories/abseil-cpp.sha256"
        "f5ff13fdf9f770f0f07c295b0ca1e9fca2d88038f49ebf20e83616c2c8f22346")
    _vrhino_verify_source_inventory(
        "${source_root}/cargo-vendor"
        "${source_root}/inventories/cargo-vendor.sha256"
        "ebb8d6b25210908e1c45b0a585819c8fff177240d356bc5c76600628db1a76dd")

    set(absl_link
        "${source_root}/tokenizers-cpp/sentencepiece/third_party/absl")
    if(NOT IS_SYMLINK "${absl_link}")
        message(FATAL_ERROR "Pinned SentencePiece Abseil include link is missing")
    endif()
    file(READ_SYMLINK "${absl_link}" absl_link_target)
    if(NOT absl_link_target STREQUAL "../../../abseil-cpp/absl")
        message(FATAL_ERROR "Pinned SentencePiece Abseil include link changed")
    endif()

    file(SHA256 "${source_root}/tokenizers-cpp/rust/Cargo.lock" cargo_lock_hash)
    if(NOT cargo_lock_hash STREQUAL
            "ea028c09e0ac3a242df874856b34b1fbedd1eda0dfb6b6160c0e4598229669a1")
        message(FATAL_ERROR "Pinned tokenizer Cargo.lock changed")
    endif()
endfunction()
