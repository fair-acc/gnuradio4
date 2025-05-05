add_executable(exprtk_example0 exprtk_example0.cpp)
target_link_libraries(exprtk_example0 PRIVATE gnuradio-core exprtk)

add_executable(exprtk_example1 exprtk_example1.cpp)
target_link_libraries(exprtk_example1 PRIVATE gnuradio-core exprtk)

add_executable(exprtk_example2 exprtk_example2.cpp)
target_link_libraries(exprtk_example2 PRIVATE gnuradio-core exprtk)

add_executable(exprtk_example3 exprtk_example3.cpp)
target_link_libraries(exprtk_example3 PRIVATE gnuradio-core exprtk)

add_executable(exprtk_example4 exprtk_example4.cpp)
target_link_libraries(exprtk_example4 PRIVATE gnuradio-core exprtk)
