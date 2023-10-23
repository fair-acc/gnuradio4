add_ut_test(qa_formatter)
add_ut_test(qa_traits)
target_link_libraries(qa_formatter PRIVATE gnuradio-meta)
target_link_libraries(qa_traits PRIVATE gnuradio-meta)
