cmake_minimum_required(VERSION 3.24)
project(wildllama_gpu_first_validation C CXX)

get_filename_component(WILDLLAMA_REPO "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(WILDLLAMA_NATIVE "${WILDLLAMA_REPO}/.build/gpu-first-moe/native" CACHE PATH "Prepared native source tree")
if(MSVC)
    # A toolchain with only a localized language pack can ignore VSLANG. Older
    # CMake versions decode its /showIncludes prefix incorrectly. Preserve the
    # compiler's raw bytes so Ninja can match the installed compiler locale.
    set(_prefix_probe_dir "${CMAKE_BINARY_DIR}/showincludes-probe")
    file(MAKE_DIRECTORY "${_prefix_probe_dir}")
    file(WRITE "${_prefix_probe_dir}/probe.h" "")
    file(WRITE "${_prefix_probe_dir}/probe.cpp" "#include \"probe.h\"\n")
    execute_process(COMMAND "${CMAKE_CXX_COMPILER}" /nologo /utf-8 /showIncludes /EP /TP "${_prefix_probe_dir}/probe.cpp"
        WORKING_DIRECTORY "${_prefix_probe_dir}" RESULT_VARIABLE _prefix_probe_status
        OUTPUT_VARIABLE _prefix_probe_out ERROR_VARIABLE _prefix_probe_err ENCODING UTF8)
    file(TO_NATIVE_PATH "${_prefix_probe_dir}/probe.h" _prefix_probe_header)
    set(_prefix_probe_text "${_prefix_probe_err}${_prefix_probe_out}")
    string(FIND "${_prefix_probe_text}" "${_prefix_probe_header}" _prefix_probe_position)
    if(NOT _prefix_probe_status EQUAL 0 OR _prefix_probe_position LESS 0)
        message(FATAL_ERROR "Cannot detect the installed MSVC /showIncludes prefix")
    endif()
    string(SUBSTRING "${_prefix_probe_text}" 0 ${_prefix_probe_position} _prefix_probe_lines)
    string(REPLACE "\r" "" _prefix_probe_lines "${_prefix_probe_lines}")
    string(REPLACE "\n" ";" _prefix_probe_lines "${_prefix_probe_lines}")
    list(GET _prefix_probe_lines -1 _gpu_first_msvc_include_prefix)
    set(CMAKE_CL_SHOWINCLUDES_PREFIX "${_gpu_first_msvc_include_prefix}")
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>
                        $<$<COMPILE_LANGUAGE:C,CXX>:/diagnostics:caret>)
endif()
set(FETCHCONTENT_SOURCE_DIR_LLAMA_CPP "${WILDLLAMA_NATIVE}" CACHE PATH "" FORCE)
set(OLLAMA_LLAMA_CPP_SKIP_COMPAT_PATCH ON CACHE BOOL "Source is already patched" FORCE)
set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL "Never download during this isolated build" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "Only build this host CPU backend" FORCE)
set(OLLAMA_GPU_BACKEND cuda CACHE STRING "CUDA module install filter" FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
enable_testing()
add_subdirectory("${WILDLLAMA_REPO}/llama/server" server)

function(validation_target name library)
    if(NOT EXISTS "${WILDLLAMA_NATIVE}/tests/${name}.cpp")
        message(FATAL_ERROR "Missing prepared test source: ${name}.cpp")
    endif()
    add_executable(${name} "${WILDLLAMA_NATIVE}/tests/${name}.cpp" ${ARGN})
    target_compile_features(${name} PRIVATE cxx_std_17)
    target_link_libraries(${name} PRIVATE ${library})
    set_target_properties(${name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endfunction()

# Cache classes/internal helpers are not part of the shared-library ABI.
set(RAM_TEST_INTERNALS
    "${WILDLLAMA_NATIVE}/src/llama-model-ram.cpp"
    "${WILDLLAMA_NATIVE}/src/llama-mmap.cpp"
    "${WILDLLAMA_NATIVE}/src/llama-impl.cpp")
validation_target(test-moe-ram llama-common ${RAM_TEST_INTERNALS})
validation_target(test-moe-ram-metrics llama ${RAM_TEST_INTERNALS})
validation_target(test-gpu-exclusive-scheduler ggml)
validation_target(test-gpu-exclusive llama)
validation_target(test-gpu-exclusive-op ggml)
validation_target(test-gpu-exclusive-ssm ggml)
validation_target(test-gpu-layer-diagnostics ggml)
target_include_directories(test-gpu-layer-diagnostics PRIVATE "${WILDLLAMA_NATIVE}/include")
validation_target(test-cuda-memory-budget ggml)
validation_target(test-moe-admission ggml)
validation_target(test-moe-budget ggml)
validation_target(test-moe-partition ggml)
validation_target(test-moe-readiness ggml)
validation_target(test-moe-read-plan llama ${RAM_TEST_INTERNALS})
validation_target(test-moe-copy-plan llama ${RAM_TEST_INTERNALS})
validation_target(test-moe-work-batch ggml)
validation_target(test-moe-ram-admission llama ${RAM_TEST_INTERNALS})
validation_target(test-moe-packed-rows llama ${RAM_TEST_INTERNALS})
validation_target(test-moe-expert-plan ggml)
validation_target(test-speculative-depth llama-common)
add_dependencies(test-speculative-depth ggml-cpu)
target_compile_definitions(test-speculative-depth PRIVATE GGML_TEST_CPU_BACKEND_PATH="$<TARGET_FILE:ggml-cpu>")
validation_target(test-state-transfer ggml)
validation_target(test-scheduler-exceptions ggml)
add_dependencies(test-scheduler-exceptions ggml-cpu)
target_compile_definitions(test-scheduler-exceptions PRIVATE GGML_TEST_CPU_BACKEND_PATH="$<TARGET_FILE:ggml-cpu>")
target_include_directories(test-state-transfer PRIVATE "${WILDLLAMA_NATIVE}/src")
validation_target(test-cuda-upload ggml)
validation_target(test-cuda-host-op ggml)
validation_target(test-cuda-dense ggml)
validation_target(test-dense-embedding llama)
target_include_directories(test-dense-embedding PRIVATE "${WILDLLAMA_NATIVE}/src")
validation_target(test-cuda-kv-growth ggml)
validation_target(test-kv-growth llama)
validation_target(test-llama-context llama-common
    "${WILDLLAMA_NATIVE}/tests/llama-context/cases-kv.cpp"
    "${WILDLLAMA_NATIVE}/tests/llama-context/cases-rs.cpp"
    "${WILDLLAMA_NATIVE}/tests/llama-context/test-case.cpp"
    "${WILDLLAMA_NATIVE}/tests/llama-context/test-log.cpp"
    "${WILDLLAMA_NATIVE}/tests/llama-context/test-runner.cpp"
    "${WILDLLAMA_NATIVE}/tests/llama-context/test-util.cpp")
validation_target(test-prepare-async ggml)
target_include_directories(test-prepare-async PRIVATE "${WILDLLAMA_NATIVE}/ggml/src")
validation_target(test-moe-ram-pool llama ${RAM_TEST_INTERNALS})
add_executable(test-expert-trace "${CMAKE_CURRENT_LIST_DIR}/tests/test-expert-trace.cpp")
find_package(Threads REQUIRED)
target_link_libraries(test-expert-trace PRIVATE Threads::Threads)
target_compile_features(test-expert-trace PRIVATE cxx_std_17)
target_include_directories(test-expert-trace PRIVATE "${WILDLLAMA_NATIVE}/src")
validation_target(test-server-component server-context)
target_include_directories(test-server-component PRIVATE
    "${WILDLLAMA_NATIVE}/tools/server" "${WILDLLAMA_NATIVE}/tests")

# Offline dense calibration is explicit; its policy tests never initialize CUDA.
add_executable(test-cuda-tuning-policy "${CMAKE_CURRENT_LIST_DIR}/tests/test-cuda-tuning-policy.cpp")
target_compile_features(test-cuda-tuning-policy PRIVATE cxx_std_17)
target_include_directories(test-cuda-tuning-policy PRIVATE "${WILDLLAMA_NATIVE}/ggml/include" "${WILDLLAMA_NATIVE}/ggml/src")
add_test(NAME cuda-tuning-policy COMMAND test-cuda-tuning-policy)
add_executable(wildllama-cuda-calibrate "${CMAKE_CURRENT_LIST_DIR}/cuda-calibrate.cpp")
target_compile_features(wildllama-cuda-calibrate PRIVATE cxx_std_17)
target_include_directories(wildllama-cuda-calibrate PRIVATE "${WILDLLAMA_NATIVE}/ggml/include" "${WILDLLAMA_NATIVE}/ggml/src")
target_link_libraries(wildllama-cuda-calibrate PRIVATE ggml)
add_test(NAME gpu-exclusive-scheduler COMMAND test-gpu-exclusive-scheduler)
add_test(NAME moe-admission COMMAND test-moe-admission)
add_test(NAME moe-budget COMMAND test-moe-budget)
add_test(NAME moe-partition COMMAND test-moe-partition)
add_test(NAME moe-readiness COMMAND test-moe-readiness)
add_test(NAME moe-read-plan COMMAND test-moe-read-plan "${CMAKE_BINARY_DIR}/moe-read-plan-fixtures")
add_test(NAME moe-copy-plan COMMAND test-moe-copy-plan "${CMAKE_BINARY_DIR}/moe-copy-plan-fixtures")
add_test(NAME moe-copy-plan-cuda COMMAND test-moe-copy-plan "${CMAKE_BINARY_DIR}/moe-copy-plan-fixtures" --gpu)
add_test(NAME moe-work-batch COMMAND test-moe-work-batch)
add_test(NAME moe-work-batch-cuda COMMAND test-moe-work-batch --gpu)
add_test(NAME moe-ram-admission COMMAND test-moe-ram-admission "${CMAKE_BINARY_DIR}/moe-ram-admission-fixtures")
add_test(NAME moe-packed-rows COMMAND test-moe-packed-rows "${CMAKE_BINARY_DIR}/moe-packed-row-fixtures")
add_test(NAME moe-packed-rows-cuda COMMAND test-moe-packed-rows "${CMAKE_BINARY_DIR}/moe-packed-row-fixtures" --gpu)
add_test(NAME moe-expert-plan COMMAND test-moe-expert-plan)
add_test(NAME moe-expert-plan-cuda COMMAND test-moe-expert-plan --gpu)
add_test(NAME speculative-depth COMMAND test-speculative-depth)
set_tests_properties(moe-packed-rows-cuda moe-expert-plan-cuda PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
set_tests_properties(moe-work-batch-cuda moe-copy-plan-cuda PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
add_test(NAME state-transfer COMMAND test-state-transfer)
add_test(NAME scheduler-exceptions COMMAND test-scheduler-exceptions)
add_test(NAME prepare-async COMMAND test-prepare-async)
add_test(NAME cuda-memory-budget COMMAND test-cuda-memory-budget)
set_tests_properties(cuda-memory-budget PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
add_test(NAME cuda-dense COMMAND test-cuda-dense)
add_test(NAME dense-embedding COMMAND test-dense-embedding "${CMAKE_BINARY_DIR}/dense-embedding-fixtures")
set_tests_properties(dense-embedding PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)
add_test(NAME cuda-kv-growth COMMAND test-cuda-kv-growth)
set_tests_properties(cuda-dense cuda-kv-growth PROPERTIES SKIP_RETURN_CODE 77 RUN_SERIAL TRUE)

# One CUDA build also supplies the base runner libraries. The normal project
# installs these in a separate CPU build; reuse the identical shared targets.
set(GPU_FIRST_BASE_RUNTIME llama-server ggml-base ggml llama mtmd llama-common ggml-cpu)
foreach(implementation IN ITEMS llama-server-impl llama-quantize llama-quantize-impl)
    if(TARGET ${implementation})
        list(APPEND GPU_FIRST_BASE_RUNTIME ${implementation})
    endif()
endforeach()
install(TARGETS ${GPU_FIRST_BASE_RUNTIME}
    RUNTIME DESTINATION lib/ollama COMPONENT gpu-first-runtime OPTIONAL
    LIBRARY DESTINATION lib/ollama COMPONENT gpu-first-runtime OPTIONAL
    PUBLIC_HEADER DESTINATION include/gpu-first COMPONENT gpu-first-development
    PRIVATE_HEADER DESTINATION include/gpu-first COMPONENT gpu-first-development)
ollama_install_windows_runtime_dlls("lib/ollama")
install(FILES "${WILDLLAMA_NATIVE}/LICENSE" DESTINATION lib/ollama
    RENAME LLAMA_CPP_LICENSE COMPONENT gpu-first-runtime)
install(FILES "${WILDLLAMA_NATIVE}/vendor/cpp-httplib/LICENSE" DESTINATION lib/ollama
    RENAME CPP_HTTPLIB_LICENSE COMPONENT gpu-first-runtime)
install(FILES "${CMAKE_BINARY_DIR}/ollama-vendor-licenses/LLAMA_CPP_VENDORS_LICENSE"
    DESTINATION lib/ollama COMPONENT gpu-first-runtime)

# enable_language(CUDA) in the child project can replace the root definition.
# Ninja writes its single shared prefix from the root directory at generation.
if(MSVC)
    set(CMAKE_CL_SHOWINCLUDES_PREFIX "${_gpu_first_msvc_include_prefix}")
endif()

# Header-only cache-safety tests also work in standalone CPU builds.
foreach(test_name IN ITEMS test-kv-trim-ranges test-prompt-state-transaction)
    add_executable(${test_name} "${WILDLLAMA_NATIVE}/tests/${test_name}.cpp")
    target_compile_features(${test_name} PRIVATE cxx_std_17)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Real tiny CPU MTP/carryover regressions (no model downloads or GPU use).
validation_target(test-mtp-state llama-common)
add_dependencies(test-mtp-state ggml-cpu)
target_compile_definitions(test-mtp-state PRIVATE GGML_TEST_CPU_BACKEND_PATH="$<TARGET_FILE:ggml-cpu>")
add_test(NAME mtp-state COMMAND test-mtp-state)
add_executable(test-mtp-carry-state "${WILDLLAMA_NATIVE}/tests/test-mtp-carry-state.cpp")
target_compile_features(test-mtp-carry-state PRIVATE cxx_std_17)
add_test(NAME mtp-carry-state COMMAND test-mtp-carry-state)

# The output-view fixture uses an explicit CPU backend and a sub-MiB model.
validation_target(test-sampler-output-view llama-common)
add_dependencies(test-sampler-output-view ggml-cpu)
target_compile_definitions(test-sampler-output-view PRIVATE GGML_TEST_CPU_BACKEND_PATH="$<TARGET_FILE:ggml-cpu>")
add_test(NAME sampler-output-view COMMAND test-sampler-output-view)
