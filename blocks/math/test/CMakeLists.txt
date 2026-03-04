add_executable(
  qa_Math
  qa_Math_main.cpp
  qa_Math_add.cpp
  qa_Math_subtract.cpp
  qa_Math_multiply.cpp
  qa_Math_divide.cpp
  qa_Math_consts.cpp)
setup_test(qa_Math)
set_property(TEST qa_Math PROPERTY ENVIRONMENT_MODIFICATION
                                  "GNURADIO4_PLUGIN_DIRECTORIES=set:${CMAKE_CURRENT_BINARY_DIR}/plugins")
get_property(_env GLOBAL PROPERTY _GR_TEST_ENV)
if(_env)
  set_tests_properties(qa_Math PROPERTIES ENVIRONMENT "${_env}")
endif()
add_ut_test(qa_ExpressionBlocks)
add_ut_test(qa_Rotator)
