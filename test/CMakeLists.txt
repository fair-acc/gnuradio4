add_executable(qa_buffer qa_buffer.cpp)
target_link_libraries(qa_buffer PRIVATE graph-prototype-options graph-prototype fmt ut)

function(add_ut_test TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    target_compile_options(${TEST_NAME} PRIVATE -fsanitize=address -Wall)
    target_link_options(${TEST_NAME} PRIVATE -fsanitize=address -Wall)
    target_link_libraries(${TEST_NAME} PRIVATE graph-prototype-options graph-prototype fmt refl-cpp ut)
endfunction()

add_ut_test(qa_dynamic_port)
add_ut_test(qa_tags)
