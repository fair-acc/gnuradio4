add_executable(
  qa_FileIo
  qa_FileIo_main.cpp
  qa_FileIo_integer.cpp
  qa_FileIo_real.cpp
  qa_FileIo_complex.cpp)
setup_test(qa_FileIo)
set_property(TEST qa_FileIo PROPERTY ENVIRONMENT_MODIFICATION
                                    "GNURADIO4_PLUGIN_DIRECTORIES=set:${CMAKE_CURRENT_BINARY_DIR}/plugins")
get_property(_env GLOBAL PROPERTY _GR_TEST_ENV)
if(_env)
  set_tests_properties(qa_FileIo PROPERTIES ENVIRONMENT "${_env}")
endif()
