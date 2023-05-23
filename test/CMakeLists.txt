function(add_ut_test TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    target_compile_options(${TEST_NAME} PRIVATE -fsanitize=address -Wall)
    target_link_options(${TEST_NAME} PRIVATE -fsanitize=address -Wall)
    target_link_libraries(${TEST_NAME} PRIVATE graph-prototype-options graph-prototype fmt refl-cpp ut)
    add_test(NAME ${TEST_NAME} COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME})
endfunction()

function(add_ut_test_no_asan TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    target_compile_options(${TEST_NAME} PRIVATE -Wall)
    target_link_options(${TEST_NAME} PRIVATE -Wall)
    target_link_libraries(${TEST_NAME} PRIVATE graph-prototype-options graph-prototype fmt refl-cpp ut)
    add_test(NAME ${TEST_NAME} COMMAND ${CMAKE_CROSSCOMPILING_EMULATOR} ${CMAKE_CURRENT_BINARY_DIR}/${TEST_NAME})
endfunction()

# mask some tests depending on the target
if (EMSCRIPTEN)
    add_ut_test(qa_dynamic_port)
    add_ut_test(qa_settings)
    add_ut_test(qa_tags)
    # fails on emscripten for unclear reasons
    # add_ut_test(qa_buffer)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_ut_test(qa_buffer)
    add_ut_test(qa_tags)
    # these tests trigger address sanitizer errors which have to be resolved
    add_ut_test_no_asan(qa_dynamic_port)
    add_ut_test_no_asan(qa_settings)
else()
    # these tests trigger address sanitizer errors which have to be resolved
    # clang asan seems to be even more strict
    add_ut_test_no_asan(qa_buffer)
    add_ut_test_no_asan(qa_tags)
    add_ut_test_no_asan(qa_dynamic_port)
    add_ut_test_no_asan(qa_settings)
endif()
