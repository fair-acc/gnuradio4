# Dependencies.cmake — central external dependency policy for GNU Radio 4.0.
#
# Policy:
#   - find_package(...) is the preferred dependency interface.
#   - FetchContent is a controlled acquisition mechanism, not a default model.
#   - Per-dep mode: system | fetch | system-or-fetch (default: system-or-fetch).
#   - Imported targets only — no raw include or library paths.
#   - Vendoring/fetch policy is never leaked into the installed/exported package config.
#
# Public API:
#   gr4_declare_dependency(NAME <name> ...)   register metadata for a dep
#   gr4_resolve_dependency(<name> ...)        find_package and/or FetchContent
#   gr4_dep_effective_mode(<name> <out>)      query effective mode without resolving
#   gr4_dependency_summary()                  end-of-configure feature_summary
#
# Per-dep override: GR4_DEP_<NAME_UPPER>_MODE cache variable, falls back to
# GR4_DEPENDENCY_MODE (default: system-or-fetch) when unset.

include_guard(GLOBAL)
include(FetchContent)
include(FeatureSummary)
include(CMakeParseArguments)

set(GR4_DEPENDENCY_MODE
    "system-or-fetch"
    CACHE STRING "Default dependency mode: system | fetch | system-or-fetch")
set_property(CACHE GR4_DEPENDENCY_MODE PROPERTY STRINGS system fetch system-or-fetch)

set_property(GLOBAL PROPERTY _GR4_DECLARED_DEPS "")

function(_gr4_dep_set name field value)
    set_property(GLOBAL PROPERTY "_GR4_DEP_${name}_${field}" "${value}")
endfunction()

function(_gr4_dep_get name field outVar)
    get_property(_v GLOBAL PROPERTY "_GR4_DEP_${name}_${field}")
    set(${outVar} "${_v}" PARENT_SCOPE)
endfunction()

function(gr4_declare_dependency)
    set(oneValueArgs
        NAME
        FIND_PACKAGE_NAME
        FETCH_GIT
        FETCH_TAG
            FETCH_INCLUDE_IN_ALL
        TYPE
        DESCRIPTION
        PURPOSE
        URL)
    set(multiValueArgs
        FIND_PACKAGE_ARGS
        FIND_PACKAGE_TARGETS
        FETCH_VARIABLES
        FETCH_PATCH_COMMAND
        MODES)
    cmake_parse_arguments(D "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    if(NOT D_NAME)
        message(FATAL_ERROR "gr4_declare_dependency: NAME is required")
    endif()
    if(NOT D_TYPE)
        set(D_TYPE OPTIONAL)
    endif()
    if(NOT D_FIND_PACKAGE_NAME)
        set(D_FIND_PACKAGE_NAME "${D_NAME}")
    endif()
    if(NOT D_MODES)
        set(D_MODES system fetch system-or-fetch)
    endif()

    foreach(field NAME FIND_PACKAGE_NAME FIND_PACKAGE_ARGS FIND_PACKAGE_TARGETS
            FETCH_GIT FETCH_TAG FETCH_PATCH_COMMAND FETCH_VARIABLES FETCH_INCLUDE_IN_ALL
                  TYPE DESCRIPTION PURPOSE URL MODES)
        _gr4_dep_set("${D_NAME}" "${field}" "${D_${field}}")
    endforeach()
    _gr4_dep_set("${D_NAME}" RESOLVED FALSE)
    _gr4_dep_set("${D_NAME}" FOUND FALSE)
    _gr4_dep_set("${D_NAME}" SOURCE "")

    set_package_properties(
        "${D_FIND_PACKAGE_NAME}" PROPERTIES
        TYPE "${D_TYPE}"
        DESCRIPTION "${D_DESCRIPTION}"
        PURPOSE "${D_PURPOSE}"
        URL "${D_URL}")

    set_property(GLOBAL APPEND PROPERTY _GR4_DECLARED_DEPS "${D_NAME}")
endfunction()

function(gr4_dep_effective_mode name outVar)
    string(TOUPPER "${name}" _upper)
    set(_per_dep "GR4_DEP_${_upper}_MODE")
    if(DEFINED ${_per_dep} AND NOT "${${_per_dep}}" STREQUAL "")
        set(_mode "${${_per_dep}}")
    else()
        set(_mode "${GR4_DEPENDENCY_MODE}")
    endif()

    _gr4_dep_get("${name}" MODES _allowed)
    if(NOT _allowed)
        set(_allowed system fetch system-or-fetch)
    endif()
    if(NOT "${_mode}" IN_LIST _allowed)
        list(GET _allowed 0 _mode)
    endif()
    set(${outVar} "${_mode}" PARENT_SCOPE)
endfunction()

function(_gr4_dep_try_find_package name outFound)
    _gr4_dep_get("${name}" FIND_PACKAGE_NAME _pkg)
    _gr4_dep_get("${name}" FIND_PACKAGE_ARGS _args)
    _gr4_dep_get("${name}" FIND_PACKAGE_TARGETS _targets)

    find_package(${_pkg} ${_args} QUIET)
    if(NOT ${_pkg}_FOUND)
        set(${outFound} FALSE PARENT_SCOPE)
        return()
    endif()
    foreach(t IN LISTS _targets)
        if(NOT TARGET "${t}")
            message(STATUS
                "gr4 dep '${name}': find_package(${_pkg}) succeeded but target '${t}' missing — treating as not found")
            set(${outFound} FALSE PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${outFound} TRUE PARENT_SCOPE)
endfunction()

function(_gr4_dep_try_fetch name outFound)
    _gr4_dep_get("${name}" FETCH_GIT _git)
    _gr4_dep_get("${name}" FETCH_TAG _tag)
    _gr4_dep_get("${name}" FETCH_VARIABLES _vars)
    _gr4_dep_get("${name}" FETCH_PATCH_COMMAND _patchCmd)
    _gr4_dep_get("${name}" FETCH_INCLUDE_IN_ALL _includeInAll)

    if(NOT _git)
        set(${outFound} FALSE PARENT_SCOPE)
        return()
    endif()

    foreach(kv IN LISTS _vars)
        if(kv MATCHES "^([^=]+)=(.*)$")
            set("${CMAKE_MATCH_1}" "${CMAKE_MATCH_2}" CACHE BOOL "" FORCE)
        endif()
    endforeach()

    # Default: EXCLUDE_FROM_ALL — fetched targets are vendored / wrapped, not part of
    # the install. Opt-in with FETCH_INCLUDE_IN_ALL TRUE for deps whose own install
    # rules must run (e.g. cpr ships its own *Config.cmake that downstream
    # find_dependency() needs after the fact).
    set(_extraArgs "")
    if (NOT _includeInAll)
        list(APPEND _extraArgs EXCLUDE_FROM_ALL)
    endif ()

    string(TOLOWER "${name}" _lower)
    if(_patchCmd)
        FetchContent_Declare(
            "${_lower}"
            GIT_REPOSITORY "${_git}"
            GIT_TAG "${_tag}"
            PATCH_COMMAND ${_patchCmd}
                ${_extraArgs})
    else()
        FetchContent_Declare(
            "${_lower}"
            GIT_REPOSITORY "${_git}"
            GIT_TAG "${_tag}"
                ${_extraArgs})
    endif()
    FetchContent_MakeAvailable("${_lower}")

    # Mark the package as found so FeatureSummary categorises it correctly even
    # though find_package() either was not called or did not produce the result.
    # FeatureSummary reads PACKAGES_FOUND / PACKAGES_NOT_FOUND, populated by
    # find_package; when fetch wins we move the entry to the FOUND list.
    _gr4_dep_get("${name}" FIND_PACKAGE_NAME _pkg)
    set("${_pkg}_FOUND" TRUE CACHE INTERNAL "set by gr4_resolve_dependency(${name}) — fetched")
    get_property(_not_found GLOBAL PROPERTY PACKAGES_NOT_FOUND)
    if(_pkg IN_LIST _not_found)
        list(REMOVE_ITEM _not_found "${_pkg}")
        set_property(GLOBAL PROPERTY PACKAGES_NOT_FOUND "${_not_found}")
    endif()
    get_property(_found GLOBAL PROPERTY PACKAGES_FOUND)
    if(NOT _pkg IN_LIST _found)
        set_property(GLOBAL APPEND PROPERTY PACKAGES_FOUND "${_pkg}")
    endif()
    set(${outFound} TRUE PARENT_SCOPE)
endfunction()

function(gr4_resolve_dependency name)
    cmake_parse_arguments(R "REQUIRED;QUIET" "" "" ${ARGN})

    get_property(_known GLOBAL PROPERTY _GR4_DECLARED_DEPS)
    if(NOT name IN_LIST _known)
        message(FATAL_ERROR "gr4_resolve_dependency(${name}): not declared. Call gr4_declare_dependency first.")
    endif()

    _gr4_dep_get("${name}" RESOLVED _already)
    if(_already)
        _gr4_dep_get("${name}" FOUND _foundVal)
        set(${name}_FOUND "${_foundVal}" PARENT_SCOPE)
        return()
    endif()

    gr4_dep_effective_mode("${name}" _mode)
    set(_found FALSE)
    set(_source "")

    if(_mode STREQUAL "system" OR _mode STREQUAL "system-or-fetch")
        _gr4_dep_try_find_package("${name}" _found)
        if(_found)
            set(_source "system")
        endif()
    endif()

    if(NOT _found AND (_mode STREQUAL "fetch" OR _mode STREQUAL "system-or-fetch"))
        _gr4_dep_try_fetch("${name}" _found)
        if(_found)
            set(_source "fetch")
        endif()
    endif()

    _gr4_dep_set("${name}" RESOLVED TRUE)
    _gr4_dep_set("${name}" FOUND "${_found}")
    _gr4_dep_set("${name}" SOURCE "${_source}")

    if(_found)
        if(NOT R_QUIET)
            message(STATUS "gr4 dep '${name}': resolved (${_source})")
        endif()
    else()
        if(R_REQUIRED)
            message(FATAL_ERROR "gr4 dep '${name}': required but not available (mode=${_mode})")
        endif()
        if(NOT R_QUIET)
            message(STATUS "gr4 dep '${name}': not available (mode=${_mode})")
        endif()
    endif()

    set(${name}_FOUND "${_found}" PARENT_SCOPE)
endfunction()

function(gr4_dependency_summary)
    feature_summary(WHAT ALL INCLUDE_QUIET_PACKAGES)
endfunction()
