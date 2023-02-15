add_executable(qa_buffer qa_buffer.cpp)
target_link_libraries(qa_buffer PRIVATE graph-prototype fmt ut)
set_project_warnings(qa_buffer)

function(add_ut_test TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    target_compile_options(${TEST_NAME} PRIVATE -fsanitize=address -Wall)
    target_link_options(${TEST_NAME} PRIVATE -fsanitize=address -Wall)
    target_link_libraries(${TEST_NAME} PRIVATE graph-prototype fmt refl-cpp ut)
    set_project_warnings(${TEST_NAME})
endfunction()

add_ut_test(qa_dynamic_port)
