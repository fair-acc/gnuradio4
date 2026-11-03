add_ut_test(qa_Histogram)
target_link_libraries(qa_Histogram PRIVATE gr-math gr-testing)
add_ut_test(qa_Math)
add_ut_test(qa_BooleanMath)
add_ut_test(qa_ByteMath)
add_ut_test(qa_ComplexMath)
add_ut_test(qa_MoreMath)
add_ut_test(qa_ExpressionBlocks)
add_ut_test(qa_Rotator)
add_ut_test(qa_TensorPeakFinder)
target_link_libraries(
  qa_TensorPeakFinder
  PRIVATE gr-math
          gr-filter
          gr-basic
          gr-testing)
