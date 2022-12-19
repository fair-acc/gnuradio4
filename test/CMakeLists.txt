add_executable(qa_buffer qa_buffer.cpp)
include_directories("../include")
target_include_directories(qa_buffer INTERFACE ../include)
target_link_libraries(qa_buffer PRIVATE fmt ut)