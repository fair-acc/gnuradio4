if(ENABLE_EXAMPLES)
  add_executable(main main.cpp)
  target_link_libraries(main PUBLIC gnuradio-options gnuradio-core magic_enum)
endif()
