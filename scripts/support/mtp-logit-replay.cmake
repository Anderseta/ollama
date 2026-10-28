cmake_minimum_required(VERSION 3.20)
project(mtp_logit_replay LANGUAGES CXX)
if(NOT DEFINED NATIVE_SOURCE OR NOT DEFINED NATIVE_BUILD)
    message(FATAL_ERROR "Set NATIVE_SOURCE and NATIVE_BUILD to the exact existing native source/build")
endif()
add_executable(mtp-logit-replay "${CMAKE_CURRENT_LIST_DIR}/mtp-logit-replay.cpp")
target_compile_features(mtp-logit-replay PRIVATE cxx_std_17)
target_include_directories(mtp-logit-replay PRIVATE
    "${NATIVE_SOURCE}/include" "${NATIVE_SOURCE}/src" "${NATIVE_SOURCE}/ggml/include"
    "${NATIVE_SOURCE}/vendor/nlohmann")
if(MSVC)
    target_compile_definitions(mtp-logit-replay PRIVATE LLAMA_SHARED GGML_SHARED GGML_BACKEND_SHARED)
    target_link_libraries(mtp-logit-replay PRIVATE
        "${NATIVE_BUILD}/_deps/llama_cpp-build/src/llama.lib"
        "${NATIVE_BUILD}/_deps/llama_cpp-build/ggml/src/ggml.lib"
        "${NATIVE_BUILD}/_deps/llama_cpp-build/ggml/src/ggml-base.lib")
else()
    target_link_directories(mtp-logit-replay PRIVATE "${NATIVE_BUILD}/bin")
    target_link_libraries(mtp-logit-replay PRIVATE llama ggml ggml-base)
endif()
