include_directories("../include")

add_executable(bm_case1 bm_case1.cpp benchmark.hpp)
target_include_directories(bm_case1 INTERFACE ../include)
target_link_libraries(bm_case1 PRIVATE refl-cpp fmt ut)
