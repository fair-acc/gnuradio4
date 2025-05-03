add_executable(soapy_example soapy_example.cpp)
target_link_libraries(
  soapy_example
  PRIVATE gnuradio-core
          gnuradio-blocklib-core
          gr-basic
          gr-fileio
          gr-testing
          gr-soapy
          ut)
if(WIN32)
  target_link_libraries(soapy_example PRIVATE dl)
endif()
