cmake_minimum_required(VERSION 3.28)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)

option(ENABLE_SPLIT "Enable splitting of generated files (pass --split to the parser)" ON)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# --------------------------------------------------------------------------
# Manually configure and build the parser tool in a separate directory
# --------------------------------------------------------------------------
set(TOOLS_BUILD_DIR "${PROJECT_BINARY_DIR}/tools_build")
if(NOT EXISTS "${TOOLS_BUILD_DIR}/CMakeCache.txt")
    message(STATUS "Configuring parser tool in ${TOOLS_BUILD_DIR}...")
    execute_process(
            COMMAND ${CMAKE_COMMAND} -S ${PROJECT_SOURCE_DIR}/tools -B ${TOOLS_BUILD_DIR} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
            RESULT_VARIABLE config_result
            OUTPUT_VARIABLE config_output
            ERROR_VARIABLE config_error
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
    )
    if(NOT config_result EQUAL 0)
        message(FATAL_ERROR "Failed to configure parser tool: ${config_error}")
    endif()
endif()

message(STATUS "Building parser tool in ${TOOLS_BUILD_DIR}...")
execute_process(
        COMMAND ${CMAKE_COMMAND} --build ${TOOLS_BUILD_DIR} --target parse_registrations
        WORKING_DIRECTORY ${TOOLS_BUILD_DIR}
        RESULT_VARIABLE build_result
        OUTPUT_VARIABLE build_output
        ERROR_VARIABLE build_error
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_STRIP_TRAILING_WHITESPACE
)
if(NOT build_result EQUAL 0)
    message(FATAL_ERROR "Failed to build parse_registrations tool: ${build_error}")
else ()
    set(PARSER_EXECUTABLE "${TOOLS_BUILD_DIR}/parse_registrations")
    message(STATUS "Parser tool built: ${PARSER_EXECUTABLE}")
endif()

# --------------------------------------------------------------------------
# Configure the rest of the project
# --------------------------------------------------------------------------

# add core functionality such as registry
configure_file(BlockLibMacros.cmake.in BlockLibMacros.cmake @ONLY)

include(${PROJECT_BINARY_DIR}/BlockLibMacros.cmake)

function(create_block_library LIB_NAME)
    set(options "")
    set(oneValueArgs "")
    set(multiValueArgs HEADERS SOURCES)
    cmake_parse_arguments(CBL "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    add_library(${LIB_NAME} SHARED
            ${CBL_SOURCES}
    )

    gr_generate_block_instantiations(${LIB_NAME} HEADERS ${CBL_HEADERS})

    target_include_directories(${LIB_NAME} PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:include>
    )

    set_target_properties(${LIB_NAME} PROPERTIES CXX_STANDARD 23)
    install(TARGETS ${LIB_NAME} LIBRARY DESTINATION lib)
endfunction()

# --------------------------------------------------------------------------
# 4. Add subdirectories defining libraries via create_block_library
# --------------------------------------------------------------------------

# Optionally build aggregator library.
# if (TARGET grBasic AND TARGET grMath)
#     add_library(grBlockLib SHARED src/dummy.cpp)
#     target_link_libraries(grBlockLib PUBLIC grBasic grMath)
#     set_target_properties(grBlockLib PROPERTIES CXX_STANDARD 23)
#     install(TARGETS grBlockLib LIBRARY DESTINATION lib)
# endif()
#
set(CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/BlockLib")

configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/BlockLibConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/BlockLibConfig.cmake"
    INSTALL_DESTINATION ${CMAKECONFIG_INSTALL_DIR}
)
#
# install(EXPORT BlockLibTargets DESTINATION ${CMAKECONFIG_INSTALL_DIR} NAMESPACE BlockLib::)

message(WARNING "Current binary dir: ${CMAKE_CURRENT_BINARY_DIR}")

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/BlockLibConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/BlockLibMacros.cmake"
    DESTINATION "${CMAKECONFIG_INSTALL_DIR}"
    COMPONENT Devel
)

install(PROGRAMS ${PARSER_EXECUTABLE} DESTINATION ${CMAKE_INSTALL_BINDIR})
