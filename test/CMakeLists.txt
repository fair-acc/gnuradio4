include_directories("../include")

add_executable(qa_buffer qa_buffer.cpp)
target_include_directories(qa_buffer INTERFACE ../include)
target_link_libraries(qa_buffer PRIVATE fmt ut)

function(add_ut_test TEST_NAME)
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp)
    target_include_directories(${TEST_NAME} INTERFACE ../include)
    target_compile_options(${TEST_NAME} PRIVATE -fsanitize=address -Wall)
    target_link_options(${TEST_NAME} PRIVATE -fsanitize=address -Wall)
    target_link_libraries(${TEST_NAME} PRIVATE fmt refl-cpp ut)
endfunction()

add_ut_test(qa_dynamic_port)
