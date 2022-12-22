include_directories("../include")

add_executable(qa_buffer qa_buffer.cpp)
target_include_directories(qa_buffer INTERFACE ../include)
target_link_libraries(qa_buffer PRIVATE fmt ut)

add_executable(qa_port qa_port.cpp)
target_include_directories(qa_port INTERFACE ../include)
target_link_libraries(qa_port PRIVATE fmt refl-cpp ut)
