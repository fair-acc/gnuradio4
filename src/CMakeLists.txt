add_executable(main main.cpp)
target_link_libraries(main PUBLIC graph-prototype-options graph-prototype fmt::fmt refl-cpp::refl-cpp)

add_executable(scheduling scheduling.cpp)
target_link_libraries(scheduling PUBLIC graph-prototype-options graph-prototype fmt::fmt)
