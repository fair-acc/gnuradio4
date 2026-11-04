cmake_minimum_required(VERSION 3.28)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

option(ENABLE_SPLIT "Enable splitting of generated files (pass --split to the parser)" OFF)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# The parser tool runs on the BUILD HOST (not the target), so a cross-compile main build (Emscripten/WASM, ARM, …) must
# compile this tool with a native compiler — never CMAKE_CXX_COMPILER, which points at the cross-compiler.
#
# Resolution order for the host compiler: 1. GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPILER   (explicit override) 2.
# GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPLILER  (legacy typo, deprecated) 3. CMAKE_CXX_COMPILER  — only when not
# cross-compiling 4. empty — let the sub-build's own CMake auto-detect from PATH/env
if(NOT DEFINED GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPILER AND DEFINED
                                                                  GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPLILER)
  message(
    DEPRECATION
      "GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPLILER is misspelled; rename to GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPILER."
  )
  set(GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPILER ${GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPLILER})
endif()
if(NOT DEFINED GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPILER)
  if(CMAKE_CROSSCOMPILING)
    set(GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPILER "")
  else()
    set(GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPILER ${CMAKE_CXX_COMPILER})
  endif()
endif()

if(GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPILER)
  message(
    STATUS
      "Using ${GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPILER} to compile the gnuradio_4_0_parse_registrations tool")
else()
  message(STATUS "Cross-compiling: letting the parser-tool sub-build pick a host C++ compiler from PATH "
                 "(override via -DGNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPILER=...)")
endif()

# --------------------------------------------------------------------------
# Manually configure and build the parser tool in a separate directory
# --------------------------------------------------------------------------
set(TOOLS_BUILD_DIR "${PROJECT_BINARY_DIR}/tools_build")

set(_parser_tool_cmake_args
    -S
    ${PROJECT_SOURCE_DIR}/tools
    -B
    ${TOOLS_BUILD_DIR}
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    # Neutralise any cross-toolchain inherited from the parent configure so the sub-build configures for the build host.
    -DCMAKE_TOOLCHAIN_FILE=
    -UCMAKE_SYSROOT
    -UCMAKE_C_COMPILER_TARGET
    -UCMAKE_CXX_COMPILER_TARGET)
if(GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPILER)
  list(APPEND _parser_tool_cmake_args -DCMAKE_CXX_COMPILER=${GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPILER})
endif()

# Strip cross-compile env vars so the sub-build sees a clean host environment, even when the parent was launched from
# inside emcmake / emconfigure / etc.
set(_parser_tool_env_cmd
    ${CMAKE_COMMAND}
    -E
    env
    --unset=CXX
    --unset=CC
    --unset=CXXFLAGS
    --unset=CFLAGS
    --unset=LDFLAGS
    --unset=CPPFLAGS
    --unset=CMAKE_TOOLCHAIN_FILE)

if(NOT EXISTS "${TOOLS_BUILD_DIR}/CMakeCache.txt")
  message(STATUS "Configuring parser tool in ${TOOLS_BUILD_DIR}...")
  execute_process(
    COMMAND ${_parser_tool_env_cmd} ${CMAKE_COMMAND} ${_parser_tool_cmake_args}
    RESULT_VARIABLE config_result
    OUTPUT_VARIABLE config_output
    ERROR_VARIABLE config_error
    OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
  if(NOT
     config_result
     EQUAL
     0)
    message(FATAL_ERROR "Failed to configure parser tool: ${config_error}")
  endif()
endif()

message(STATUS "Building parser tool in ${TOOLS_BUILD_DIR}...")
execute_process(
  COMMAND ${_parser_tool_env_cmd} ${CMAKE_COMMAND} --build ${TOOLS_BUILD_DIR} --target gnuradio_4_0_parse_registrations
  WORKING_DIRECTORY ${TOOLS_BUILD_DIR}
  RESULT_VARIABLE build_result
  OUTPUT_VARIABLE build_output
  ERROR_VARIABLE build_error
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
if(NOT
   build_result
   EQUAL
   0)
  message(FATAL_ERROR "Failed to build gnuradio_4_0_parse_registrations tool: ${build_error} ${build_output}")
else()
  if(WIN32)
    set(PARSER_EXECUTABLE "${TOOLS_BUILD_DIR}/gnuradio_4_0_parse_registrations.exe")
  else()
    set(PARSER_EXECUTABLE "${TOOLS_BUILD_DIR}/gnuradio_4_0_parse_registrations")
  endif()
  message(STATUS "Parser tool built: ${PARSER_EXECUTABLE}")
endif()

# --------------------------------------------------------------------------
# Configure the rest of the project
# --------------------------------------------------------------------------

# add core functionality such as registry
configure_file(GnuRadioBlockLibMacros.cmake.in GnuRadioBlockLibMacros.cmake @ONLY)

include(${PROJECT_BINARY_DIR}/GnuRadioBlockLibMacros.cmake)

set(GR_BLOCKLIB_CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/GnuRadioBlockLib")

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/GnuRadioBlockLibConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/GnuRadioBlockLibConfig.cmake"
  INSTALL_DESTINATION ${GR_BLOCKLIB_CMAKECONFIG_INSTALL_DIR}
  PATH_VARS CMAKE_INSTALL_BINDIR)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/GnuRadioBlockLibConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/GnuRadioBlockLibMacros.cmake"
  DESTINATION "${GR_BLOCKLIB_CMAKECONFIG_INSTALL_DIR}"
  COMPONENT Devel)

install(PROGRAMS ${PARSER_EXECUTABLE} DESTINATION ${CMAKE_INSTALL_BINDIR})
