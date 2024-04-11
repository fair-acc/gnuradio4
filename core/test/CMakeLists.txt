set(TESTS_BINARY_PATH ${CMAKE_CURRENT_BINARY_DIR})
set(TESTS_SOURCE_PATH ${CMAKE_CURRENT_SOURCE_DIR})

configure_file(build_configure.hpp.in build_configure.hpp @ONLY)

function(setup_test_no_asan TEST_NAME)
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_BINARY_DIR}/include ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(${TEST_NAME} PRIVATE gnuradio-options gnuradio-core fmt refl-cpp yaml-cpp::yaml-cpp fftw gr-basic gr-testing ut)
    add_test(NAME ${TEST_NAME} COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME})
endfunction()

function(setup_test TEST_NAME)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU") # limited to gcc due to a Ubuntu packaging bug of libc++, see https://github.com/llvm/llvm-project/issues/59432
        target_compile_options(${TEST_NAME} PRIVATE -fsanitize=address) # for testing consider enabling -D_GLIBCXX_DEBUG and -D_GLIBCXX_SANITIZE_VECTOR
        target_link_options(${TEST_NAME} PRIVATE -fsanitize=address) # for testing consider enabling -D_GLIBCXX_DEBUG and -D_GLIBCXX_SANITIZE_VECTOR
    endif()

    setup_test_no_asan(${TEST_NAME})
endfunction()

function(add_ut_test TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    setup_test(${TEST_NAME})
endfunction()

function(add_app_test TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    setup_test(${TEST_NAME})
    target_link_libraries(${TEST_NAME} PRIVATE gnuradio-plugin)
    add_dependencies(${TEST_NAME} good_math_plugin good_base_plugin bad_plugin)
endfunction()

add_ut_test(qa_buffer)
add_ut_test(qa_DynamicBlock)
add_ut_test(qa_DynamicPort)
add_ut_test(qa_HierBlock)
add_ut_test(qa_Block)
add_ut_test(qa_LifeCycle)
add_ut_test(qa_Scheduler)
add_ut_test(qa_reader_writer_lock)
add_ut_test(qa_Settings)
add_ut_test(qa_Tags)
add_ut_test(qa_Messages)
add_ut_test(qa_thread_affinity)
add_ut_test(qa_thread_pool)

if(NOT EMSCRIPTEN)
    add_subdirectory(plugins)
    add_app_test(qa_grc)
    add_app_test(qa_plugins_test)
else()
    add_ut_test(qa_grc)
endif()
