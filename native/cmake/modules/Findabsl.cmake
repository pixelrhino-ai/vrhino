# SentencePiece's supported package-provider branch calls find_package(absl).
# VRhino adds the exact repository-local Abseil source before that call, so
# this module can expose the existing targets without a system package search.
if(NOT TARGET absl::base)
    set(absl_FOUND FALSE)
    if(absl_FIND_REQUIRED)
        message(FATAL_ERROR
            "Repository-local Abseil targets were not configured")
    endif()
    return()
endif()

set(absl_FOUND TRUE)
