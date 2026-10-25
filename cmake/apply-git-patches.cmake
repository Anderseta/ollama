# Idempotent external-source patch applier.
#
# Invocation (from a CMake PATCH_COMMAND):
#   cmake -DPATCH_DIR=<dir of *.patch> [-DPATCH_LABEL=<name>]
#     -P cmake/apply-git-patches.cmake
#
# Every *.patch under PATCH_DIR is applied in numeric filename order in the
# current working directory (which ExternalProject / FetchContent sets to the
# fetched source's SOURCE_DIR). A patch already applied — detected via
# `git apply --reverse --check` — is skipped. This makes re-configuring and
# re-building safe when the source directory outlives its CMake patch stamp.

if(NOT DEFINED PATCH_DIR)
    message(FATAL_ERROR "apply-git-patches.cmake: PATCH_DIR not set")
endif()

if(NOT DEFINED PATCH_LABEL)
    set(PATCH_LABEL "external patches")
endif()

find_package(Git QUIET REQUIRED)

get_filename_component(_patch_workdir "." ABSOLUTE)
get_filename_component(_git_ceiling "${_patch_workdir}" DIRECTORY)
set(_git_apply_env GIT_CEILING_DIRECTORIES=${_git_ceiling})

file(GLOB_RECURSE _patches "${PATCH_DIR}/*.patch")
set(_patch_entries)
foreach(PATCH_FILE IN LISTS _patches)
    get_filename_component(_patch_name "${PATCH_FILE}" NAME)
    list(APPEND _patch_entries "${_patch_name}|${PATCH_FILE}")
endforeach()

list(SORT _patch_entries)

# Later patches can edit the context of earlier patches. In that case the
# earlier patch cannot be reverse-checked directly against the final tree.
# Probe an applied prefix on private copies, undoing it in reverse order. This
# leaves the fetched source and its Git index untouched during detection.
set(_applied_prefix 0)
list(LENGTH _patch_entries _patch_count)
if(_patch_count GREATER 1)
    list(GET _patch_entries 0 _first_entry)
    string(REGEX REPLACE "^[^|]*\\|" "" _first_patch "${_first_entry}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env ${_git_apply_env}
            ${GIT_EXECUTABLE} apply --check "${_first_patch}"
        RESULT_VARIABLE _clean_check OUTPUT_QUIET ERROR_QUIET)
    if(NOT _clean_check EQUAL 0)
        set(_probe_paths)
        set(_probe_supported TRUE)
        foreach(_entry IN LISTS _patch_entries)
            string(REGEX REPLACE "^[^|]*\\|" "" _patch "${_entry}")
            execute_process(COMMAND ${GIT_EXECUTABLE} apply --numstat "${_patch}"
                RESULT_VARIABLE _stat_result OUTPUT_VARIABLE _stat)
            if(NOT _stat_result EQUAL 0)
                set(_probe_supported FALSE)
                break()
            endif()
            string(REPLACE "\n" ";" _stat_lines "${_stat}")
            foreach(_line IN LISTS _stat_lines)
                if(_line STREQUAL "")
                    continue()
                endif()
                string(REGEX REPLACE "^[^\t]*\t[^\t]*\t" "" _path "${_line}")
                # Unusual quoted paths/renames use the conservative legacy
                # checks below rather than interpreting Git's path quoting.
                if(NOT _path MATCHES "^[A-Za-z0-9_./-]+$" OR
                   _path MATCHES "(^|/)\\.\\.(/|$)" OR _path MATCHES "^/")
                    set(_probe_supported FALSE)
                    break()
                endif()
                list(APPEND _probe_paths "${_path}")
            endforeach()
        endforeach()
        if(_probe_supported)
            list(REMOVE_DUPLICATES _probe_paths)
            string(RANDOM LENGTH 16 ALPHABET 0123456789abcdef _probe_id)
            set(_probe_root "${_patch_workdir}/.ollama-patch-check-${_probe_id}")
            set(_prefix ${_patch_count})
            while(_prefix GREATER 0)
                file(MAKE_DIRECTORY "${_probe_root}")
                foreach(_path IN LISTS _probe_paths)
                    if(EXISTS "${_patch_workdir}/${_path}")
                        get_filename_component(_parent "${_probe_root}/${_path}" DIRECTORY)
                        file(MAKE_DIRECTORY "${_parent}")
                        configure_file("${_patch_workdir}/${_path}" "${_probe_root}/${_path}" COPYONLY)
                    endif()
                endforeach()
                math(EXPR _last "${_prefix} - 1")
                set(_prefix_ok TRUE)
                foreach(_idx RANGE ${_last} 0 -1)
                    list(GET _patch_entries ${_idx} _entry)
                    string(REGEX REPLACE "^[^|]*\\|" "" _patch "${_entry}")
                    execute_process(COMMAND ${CMAKE_COMMAND} -E env
                        "GIT_CEILING_DIRECTORIES=${_patch_workdir}"
                        ${GIT_EXECUTABLE} apply --reverse "${_patch}"
                        WORKING_DIRECTORY "${_probe_root}"
                        RESULT_VARIABLE _undo OUTPUT_QUIET ERROR_QUIET)
                    if(NOT _undo EQUAL 0)
                        set(_prefix_ok FALSE)
                        break()
                    endif()
                endforeach()
                # The random absolute probe path is constructed only under
                # this source directory and never contains user input.
                file(REMOVE_RECURSE "${_probe_root}")
                if(_prefix_ok)
                    set(_applied_prefix ${_prefix})
                    break()
                endif()
                math(EXPR _prefix "${_prefix} - 1")
            endwhile()
        endif()
    endif()
endif()

set(_patch_index 0)
foreach(_patch_entry IN LISTS _patch_entries)
    math(EXPR _patch_index "${_patch_index} + 1")
    string(REGEX REPLACE "^[^|]*\\|" "" PATCH_FILE "${_patch_entry}")
    file(RELATIVE_PATH _patch_rel "${PATCH_DIR}" "${PATCH_FILE}")

    if(_patch_index LESS_EQUAL _applied_prefix)
        message(STATUS "${PATCH_LABEL}: ${_patch_rel} already applied (verified patch prefix), skipping")
        continue()
    endif()

    # If the patch can be REVERSED cleanly, it's already applied. Skip.
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env ${_git_apply_env}
            ${GIT_EXECUTABLE} apply --reverse --check "${PATCH_FILE}"
        RESULT_VARIABLE _reverse_check
        OUTPUT_QUIET ERROR_QUIET
    )
    if(_reverse_check EQUAL 0)
        message(STATUS "${PATCH_LABEL}: ${_patch_rel} already applied, skipping")
        continue()
    endif()

    # Otherwise, apply forward.
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env ${_git_apply_env}
            ${GIT_EXECUTABLE} apply --whitespace=nowarn "${PATCH_FILE}"
        RESULT_VARIABLE _apply_result
    )
    if(NOT _apply_result EQUAL 0)
        message(FATAL_ERROR
            "${PATCH_LABEL}: failed to apply ${_patch_rel}\n"
            "The pinned source is neither clean nor compatible with this patch. "
            "Remove the retained source directory (${_patch_workdir}), or "
            "regenerate the patch against the pinned source before retrying.")
    endif()

    message(STATUS "${PATCH_LABEL}: applied ${_patch_rel}")
endforeach()
