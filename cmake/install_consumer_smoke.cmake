# install_consumer_smoke.cmake — drives the install + downstream find_package
# round-trip. Invoked via `cmake -P` from CTest; expects the parent build to be
# fully built (the install rules will pull in every exported target).
#
# Required input variables (passed via -D before -P):
#   GR4_BUILD_DIR     - parent build directory to install from
#   GR4_CONSUMER_SRC  - path to cmake/install-consumer/
#   GR4_TEST_PREFIX   - throw-away install prefix (will be wiped)
#   GR4_GENERATOR     - CMake generator used for the consumer
#   GR4_BUILD_TYPE    - build type for the consumer (mirrors parent)
#   GR4_CXX_COMPILER  - C++ compiler used for the consumer
#
# Pure CMake, no shell script.

cmake_minimum_required(VERSION 3.27)

foreach (_v GR4_BUILD_DIR GR4_CONSUMER_SRC GR4_TEST_PREFIX GR4_GENERATOR GR4_BUILD_TYPE GR4_CXX_COMPILER)
    if (NOT DEFINED ${_v})
        message(FATAL_ERROR "install_consumer_smoke: ${_v} not set")
    endif ()
endforeach ()
# Optional — sanitizer / libc++ flags forwarded by the parent build so the consumer's tiny executable can link against
# the installed static archives. Empty for default builds.
if (NOT DEFINED GR4_EXTRA_FLAGS)
    set(GR4_EXTRA_FLAGS "")
endif ()

set(_prefix "${GR4_TEST_PREFIX}/prefix")
set(_consumer_build "${GR4_TEST_PREFIX}/consumer-build")

file(REMOVE_RECURSE "${GR4_TEST_PREFIX}")
file(MAKE_DIRECTORY "${_prefix}" "${_consumer_build}")

function(_gr4_run label)
    execute_process(
            COMMAND ${ARGN}
            RESULT_VARIABLE _rv
            OUTPUT_VARIABLE _out
            ERROR_VARIABLE _err)
    if (NOT _rv EQUAL 0)
        message(STATUS "stdout:\n${_out}")
        message(STATUS "stderr:\n${_err}")
        message(FATAL_ERROR "install-consumer-smoke: '${label}' failed (exit ${_rv})")
    endif ()
endfunction()

_gr4_run("install"
        "${CMAKE_COMMAND}" --install "${GR4_BUILD_DIR}" --prefix "${_prefix}")

_gr4_run("configure consumer"
        "${CMAKE_COMMAND}" -S "${GR4_CONSUMER_SRC}" -B "${_consumer_build}"
        -G "${GR4_GENERATOR}"
        "-DCMAKE_PREFIX_PATH=${_prefix}"
        "-DCMAKE_CXX_COMPILER=${GR4_CXX_COMPILER}"
        "-DCMAKE_BUILD_TYPE=${GR4_BUILD_TYPE}"
        "-DCMAKE_CXX_FLAGS_INIT=${GR4_EXTRA_FLAGS}"
        "-DCMAKE_EXE_LINKER_FLAGS_INIT=${GR4_EXTRA_FLAGS}")

_gr4_run("build consumer"
        "${CMAKE_COMMAND}" --build "${_consumer_build}" --config "${GR4_BUILD_TYPE}")

# Locate the produced executable across single- and multi-config generators.
set(_consumer_exe "")
foreach (_candidate
        "${_consumer_build}/consumer"
        "${_consumer_build}/consumer.exe"
        "${_consumer_build}/${GR4_BUILD_TYPE}/consumer"
        "${_consumer_build}/${GR4_BUILD_TYPE}/consumer.exe")
    if (EXISTS "${_candidate}")
        set(_consumer_exe "${_candidate}")
        break()
    endif ()
endforeach ()
if (NOT _consumer_exe)
    message(FATAL_ERROR "install-consumer-smoke: produced consumer executable not found under ${_consumer_build}")
endif ()

_gr4_run("run consumer" "${_consumer_exe}")

message(STATUS "install-consumer-smoke: OK (prefix=${_prefix})")
