# libclang is under /usr/lib64 on openSUSE and under /usr/lib/llvm-<version> from apt.llvm.org and Ubuntu (the CI image)
find_library(
  GR_DEVTOOLS_LIBCLANG
  NAMES clang
        clang-22
        clang-21
        clang-20
        libclang
  HINTS /usr/lib/llvm-22/lib
        /usr/lib/llvm-21/lib
        /usr/lib/llvm-20/lib
        REQUIRED)
find_path(
  GR_DEVTOOLS_LIBCLANG_INCLUDE clang-c/Index.h
  HINTS /usr/lib/llvm-22/include
        /usr/lib/llvm-21/include
        /usr/lib/llvm-20/include
        REQUIRED)
foreach(_tool IN ITEMS code-metrics naming-check)
  add_executable(${_tool} ${_tool}.cpp)
  target_include_directories(${_tool} SYSTEM PRIVATE ${GR_DEVTOOLS_LIBCLANG_INCLUDE})
  target_link_libraries(${_tool} PRIVATE gnuradio-options ${GR_DEVTOOLS_LIBCLANG})
endforeach()

find_program(
  GR_DEVTOOLS_CLANG_TIDY
  NAMES clang-tidy-22
        clang-tidy-21
        clang-tidy-20
        clang-tidy)
if(GR_DEVTOOLS_CLANG_TIDY AND CMAKE_CXX_COMPILER_ID MATCHES "Clang") # the fixture is parsed as the Clang build parses
                                                                     # its code
  set(FIXTURE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/test/code-metrics)
  configure_file(test/code-metrics/compile_commands.json.in test/code-metrics/compile_commands.json @ONLY)
  add_test(NAME code-metrics-fixture COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test/code-metrics/check-fixture.sh
                                             $<TARGET_FILE:code-metrics> ${CMAKE_CURRENT_BINARY_DIR}/test/code-metrics)
  set(FIXTURE_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/test/naming-check)
  configure_file(test/naming-check/compile_commands.json.in test/naming-check/compile_commands.json @ONLY)
  add_test(NAME naming-check-fixture COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test/naming-check/check-fixture.sh
                                             $<TARGET_FILE:naming-check> ${CMAKE_CURRENT_BINARY_DIR}/test/naming-check)
endif()

# the tools' pure parts against their ground truths; the fixtures above check them end to end
if(TARGET ut)
  foreach(_test IN ITEMS qa_NamingCheck qa_CodeMetrics)
    add_executable(${_test} test/${_test}.cpp)
    target_include_directories(${_test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${_test} PRIVATE ut gnuradio-options)
    add_test(NAME ${_test} COMMAND ${_test})
  endforeach()
endif()

# the quality bots' reports (CI compiles the same file directly, without CMake)
add_executable(qualitybot qualitybot.cpp)
target_link_libraries(qualitybot PRIVATE gnuradio-options)
if(TARGET ut)
  add_executable(qa_QualityBot test/qa_QualityBot.cpp)
  target_include_directories(qa_QualityBot PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(qa_QualityBot PRIVATE ut gnuradio-options)
  add_test(NAME qa_QualityBot COMMAND qa_QualityBot)
endif()
