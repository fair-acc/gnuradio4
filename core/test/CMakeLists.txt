set(TESTS_BINARY_PATH ${CMAKE_CURRENT_BINARY_DIR})
set(TESTS_SOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

configure_file(build_configure.hpp.in build_configure.hpp @ONLY)

function(setup_test_no_asan TEST_NAME)
  target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_BINARY_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
  target_link_libraries(
    ${TEST_NAME}
    PRIVATE gnuradio-options
            gnuradio-core
            fmt
            fftw
            gr-basic
            gr-fileio
            gr-math
            gr-testing
            gr-basic
            ut)
  add_test(NAME ${TEST_NAME} COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME})
endfunction()

function(setup_test TEST_NAME)
  if(PYTHON_AVAILABLE)
    target_include_directories(${TEST_NAME} PRIVATE ${Python3_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
    target_link_libraries(${TEST_NAME} PRIVATE ${Python3_LIBRARIES})
  endif()

  setup_test_no_asan(${TEST_NAME})
endfunction()

function(add_ut_test TEST_NAME)
  add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
  setup_test(${TEST_NAME})
  set_property(TEST ${TEST_NAME} PROPERTY ENVIRONMENT_MODIFICATION
                                          "GNURADIO4_PLUGIN_DIRECTORIES=set:${CMAKE_CURRENT_BINARY_DIR}/plugins")
endfunction()

function(add_app_test TEST_NAME)
  add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
  setup_test(${TEST_NAME})
  target_link_libraries(${TEST_NAME} PRIVATE gnuradio-plugin)
  add_dependencies(
    ${TEST_NAME}
    good_math_plugin
    good_base_plugin
    bad_plugin)
endfunction()

add_ut_test(qa_buffer)
add_ut_test(qa_AtomicBitset)
add_ut_test(qa_DataSet)
add_ut_test(qa_DynamicBlock)
add_ut_test(qa_DynamicPort)
add_ut_test(qa_HierBlock)
add_ut_test(qa_Block)
add_ut_test(qa_LifeCycle)
add_ut_test(qa_Port)
add_ut_test(qa_Scheduler)
add_ut_test(qa_Graph)
add_ut_test(qa_reader_writer_lock)
add_ut_test(qa_Settings)
add_ut_test(qa_Tags)
add_ut_test(qa_TriggerMatcher)
add_ut_test(qa_Messages)
add_ut_test(qa_thread_affinity)
add_ut_test(qa_thread_pool)
add_ut_test(qa_PerformanceMonitor)
add_ut_test(qa_YamlPmt)
add_ut_test(qa_PmtTypeHelpers)

if(ENABLE_BLOCK_REGISTRY AND ENABLE_BLOCK_PLUGINS)
  add_app_test(qa_grc)
  add_ut_test(qa_GraphMessages)

  add_subdirectory(plugins)
  add_app_test(qa_plugins_test)
endif()
