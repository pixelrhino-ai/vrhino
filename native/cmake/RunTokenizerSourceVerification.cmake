if(NOT DEFINED VRHINO_TOKENIZER_BUILD_SOURCES)
    message(FATAL_ERROR "VRHINO_TOKENIZER_BUILD_SOURCES is required")
endif()

include("${CMAKE_CURRENT_LIST_DIR}/VerifyTokenizerBuildSources.cmake")
vrhino_verify_tokenizer_build_sources("${VRHINO_TOKENIZER_BUILD_SOURCES}")
message(STATUS "Repository-local tokenizer build-source verification: PASS")
