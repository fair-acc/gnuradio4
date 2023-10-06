add_subdirectory(basic)
add_subdirectory(filter)
add_subdirectory(fourier)

if (ENABLE_TESTING)
    add_subdirectory(testing)
endif ()
