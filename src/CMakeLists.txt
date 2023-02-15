add_executable(main main.cpp)
target_include_directories(main PUBLIC ./include)
target_link_libraries(main PUBLIC graph-prototype fmt::fmt refl-cpp::refl-cpp)
set_project_warnings(main)
