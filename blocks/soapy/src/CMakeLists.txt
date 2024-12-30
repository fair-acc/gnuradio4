add_executable(soapy_example soapy_example.cpp)
target_link_libraries(
  soapy_example
  PRIVATE gr-basic
          gr-fileio
          gr-testing
          gr-soapy
          ut)
