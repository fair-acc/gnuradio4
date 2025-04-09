
find_package(Volk REQUIRED)
find_package(Boost REQUIRED COMPONENTS unit_test_framework)

# Debug VOLK detection
message(STATUS "VOLK_FOUND: ${VOLK_FOUND}")
message(STATUS "VOLK_LIBRARIES: ${VOLK_LIBRARIES}")

add_ut_test(qa_Math)
add_ut_test(qa_ExpressionBlocks)
add_ut_test(qa_Rotator)

# Link VOLK to qa_Math
target_link_libraries(qa_Math PRIVATE ${VOLK_LIBRARIES})