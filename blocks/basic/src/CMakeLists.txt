add_executable(function_generator_example function_generator_example.cpp)
target_link_libraries(function_generator_example PRIVATE gr-basic gr-testing) #TODO: remove gr-testing after reference to tag printing is removed from clock source
