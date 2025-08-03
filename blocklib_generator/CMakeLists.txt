cmake_minimum_required(VERSION 3.28)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

option(ENABLE_SPLIT "Enable splitting of generated files (pass --split to the parser)" ON)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

if(NOT GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPLILER)
  set(GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPLILER ${CMAKE_CXX_COMPILER})
endif()

message(
  STATUS "Using ${GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPLILER} to compile the gnuradio_4_0_parse_registrations tool"
)

# --------------------------------------------------------------------------
# Manually configure and build the parser tool in a separate directory
# --------------------------------------------------------------------------
set(TOOLS_BUILD_DIR "${PROJECT_BINARY_DIR}/tools_build")
if(NOT EXISTS "${TOOLS_BUILD_DIR}/CMakeCache.txt")
  message(STATUS "Configuring parser tool in ${TOOLS_BUILD_DIR}...")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -S ${PROJECT_SOURCE_DIR}/tools -B ${TOOLS_BUILD_DIR} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            -DCMAKE_CXX_COMPILER=${GNURADIO_PARSE_REGISTRATIONS_TOOL_CXX_COMPLILER}
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
  COMMAND ${CMAKE_COMMAND} --build ${TOOLS_BUILD_DIR} --target gnuradio_4_0_parse_registrations
  WORKING_DIRECTORY ${TOOLS_BUILD_DIR}
  RESULT_VARIABLE build_result
  OUTPUT_VARIABLE build_output
  ERROR_VARIABLE build_error
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
if(NOT
   build_result
   EQUAL
   0)
  message(FATAL_ERROR "Failed to build gnuradio_4_0_parse_registrations tool: ${build_error}")
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
  "${CMAKE_CURRENT_BINARY_DIR}/GnuRadioBlockLibConfig.cmake" INSTALL_DESTINATION ${GR_BLOCKLIB_CMAKECONFIG_INSTALL_DIR})

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/GnuRadioBlockLibConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/GnuRadioBlockLibMacros.cmake"
  DESTINATION "${GR_BLOCKLIB_CMAKECONFIG_INSTALL_DIR}"
  COMPONENT Devel)

install(PROGRAMS ${PARSER_EXECUTABLE} DESTINATION ${CMAKE_INSTALL_BINDIR})
