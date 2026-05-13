# lint_dependencies.cmake — verify DEPENDENCIES.md matches the
# gr4_declare_dependency() registry in the top-level CMakeLists.txt.
#
# Run via:
#     cmake -P cmake/lint_dependencies.cmake
#     cmake -DROOT=/path/to/repo -P cmake/lint_dependencies.cmake
#
# Wired into the build as `cmake --build <build> --target lint-dependencies`
# (see top-level CMakeLists.txt). Exits with FATAL_ERROR on drift.
#
# Source of truth: gr4_declare_dependency(NAME ...) calls in CMakeLists.txt.
# Compared against: markdown link entries [Name](url) inside any table whose
# header row contains "Dependency" in DEPENDENCIES.md.
#
# Pure CMake, no Python or shell-tool dependencies.

cmake_minimum_required(VERSION 3.20)

if(DEFINED ROOT)
    set(_repo_root "${ROOT}")
else()
    get_filename_component(_self "${CMAKE_SCRIPT_MODE_FILE}" REALPATH)
    get_filename_component(_self_dir "${_self}" DIRECTORY)
    get_filename_component(_repo_root "${_self_dir}/.." REALPATH)
endif()

set(_cmake_file "${_repo_root}/CMakeLists.txt")
set(_md_file "${_repo_root}/DEPENDENCIES.md")

if(NOT EXISTS "${_cmake_file}")
    message(FATAL_ERROR "lint_dependencies: missing ${_cmake_file}")
endif()
if(NOT EXISTS "${_md_file}")
    message(FATAL_ERROR "lint_dependencies: missing ${_md_file}")
endif()

# Deps that legitimately appear in DEPENDENCIES.md but are not declared via
# gr4_declare_dependency() — vendored snapshots and system probes that go
# through plain find_package().
set(_exempt_from_declaration magic_enum exprtk Python3 CURL)


# ---- parse gr4_declare_dependency() blocks --------------------------------

file(READ "${_cmake_file}" _cmake_content)

# Escape semicolons so list-foreach over content-derived strings does not
# split on data that happens to contain ';'.
string(REPLACE ";" "\\;" _cmake_content_esc "${_cmake_content}")
# Flatten newlines so REGEX MATCHALL sees each block as one string.
string(REPLACE "\n" " " _cmake_flat "${_cmake_content_esc}")

# The body of each call has no nested parens in this codebase, so '[^)]*' is
# sufficient. If that ever changes the parser will need a balance loop.
string(REGEX MATCHALL "gr4_declare_dependency\\([^)]*\\)" _blocks "${_cmake_flat}")

set(_declared_names "")
set(_declared_tags "") # parallel "name=tag" entries

foreach(_block IN LISTS _blocks)
    set(_name "")
    set(_tag "")
    if(_block MATCHES "NAME[ \t]+([^ \t]+)")
        set(_name "${CMAKE_MATCH_1}")
    endif()
    if(_block MATCHES "FETCH_TAG[ \t]+([^ \t]+)")
        set(_tag "${CMAKE_MATCH_1}")
    endif()
    if(_name)
        list(APPEND _declared_names "${_name}")
        list(APPEND _declared_tags "${_name}=${_tag}")
    endif()
endforeach()

list(REMOVE_DUPLICATES _declared_names)


# ---- parse DEPENDENCIES.md table rows -------------------------------------

file(READ "${_md_file}" _md_content)
string(REPLACE ";" "\\;" _md_content_esc "${_md_content}")
string(REPLACE "\n" ";" _md_lines "${_md_content_esc}")

set(_documented_names "")
set(_documented_lines "") # parallel "name=lineno" entries
set(_in_table FALSE)
set(_line_no 0)

foreach(_line IN LISTS _md_lines)
    math(EXPR _line_no "${_line_no} + 1")
    string(STRIP "${_line}" _stripped)

    # Header row of a dep-bearing table.
    if(_stripped MATCHES "^\\|" AND _stripped MATCHES "Dependency" AND NOT _stripped MATCHES "---")
        set(_in_table TRUE)
        continue()
    endif()
    # End of table when we leave the | ... | column structure.
    if(NOT _stripped MATCHES "^\\|")
        set(_in_table FALSE)
        continue()
    endif()
    # Separator row.
    if(_stripped MATCHES "^\\|[ ]*-")
        continue()
    endif()
    if(NOT _in_table)
        continue()
    endif()

    # First markdown link [...](...) in the row is the dep canonical name.
    if(_stripped MATCHES "\\[([A-Za-z0-9_.+-]+)\\]\\(")
        set(_dep "${CMAKE_MATCH_1}")
        # Map markdown spelling -> canonical CMake NAME.
        if(_dep STREQUAL "Boost.UT")
            set(_dep "ut")
        elseif(_dep STREQUAL "libcurl")
            set(_dep "CURL")
        elseif(_dep STREQUAL "ExprTk")
            set(_dep "exprtk")
        endif()
        list(APPEND _documented_names "${_dep}")
        list(APPEND _documented_lines "${_dep}=${_line_no}")
    endif()
endforeach()

list(REMOVE_DUPLICATES _documented_names)


# ---- compare ---------------------------------------------------------------

set(_drift "")

foreach(_name IN LISTS _declared_names)
    if(NOT _name IN_LIST _documented_names)
        list(APPEND _drift
            "  declared in CMakeLists.txt but not in DEPENDENCIES.md: '${_name}'\n    -> add a row in the appropriate DEPENDENCIES.md table")
    endif()
endforeach()

foreach(_name IN LISTS _documented_names)
    if(NOT _name IN_LIST _declared_names AND NOT _name IN_LIST _exempt_from_declaration)
        set(_lineno "?")
        foreach(_entry IN LISTS _documented_lines)
            if(_entry MATCHES "^${_name}=(.*)$")
                set(_lineno "${CMAKE_MATCH_1}")
                break()
            endif()
        endforeach()
        list(APPEND _drift
            "  documented in DEPENDENCIES.md (line ${_lineno}) but not declared: '${_name}'\n    -> add gr4_declare_dependency(NAME ${_name} ...) in CMakeLists.txt,\n       or add '${_name}' to _exempt_from_declaration in this script")
    endif()
endforeach()

# Stale FETCH_TAG: pinned tag must appear somewhere in DEPENDENCIES.md
# (full hash or its first 8 hex chars for git SHAs).
foreach(_entry IN LISTS _declared_tags)
    if(_entry MATCHES "^([^=]+)=(.+)$")
        set(_n "${CMAKE_MATCH_1}")
        set(_t "${CMAKE_MATCH_2}")
        set(_found_in_md FALSE)
        string(FIND "${_md_content}" "${_t}" _pos)
        if(_pos GREATER -1)
            set(_found_in_md TRUE)
        else()
            string(LENGTH "${_t}" _tlen)
            if(_tlen EQUAL 40 AND "${_t}" MATCHES "^[0-9a-f]+$")
                string(SUBSTRING "${_t}" 0 8 _short)
                string(FIND "${_md_content}" "${_short}" _pos)
                if(_pos GREATER -1)
                    set(_found_in_md TRUE)
                endif()
            endif()
        endif()
        if(NOT _found_in_md)
            list(APPEND _drift
                "  pinned FETCH_TAG for '${_n}' ('${_t}') is not mentioned in DEPENDENCIES.md\n    -> update the dependency table to record the new pinned ref")
        endif()
    endif()
endforeach()


# ---- report ----------------------------------------------------------------

if(_drift)
    message("DEPENDENCIES.md is out of sync with gr4_declare_dependency() registry:")
    foreach(_d IN LISTS _drift)
        message("${_d}")
    endforeach()
    message("")
    message("  declared in CMake: ${_declared_names}")
    message("  documented in MD:  ${_documented_names}")
    message(FATAL_ERROR "drift detected")
endif()

list(LENGTH _declared_names _n_decl)
list(LENGTH _documented_names _n_doc)
list(LENGTH _exempt_from_declaration _n_exempt)
message("OK: ${_n_decl} declared deps all documented (${_n_doc} entries in DEPENDENCIES.md, ${_n_exempt} exempt from CMake declaration).")
