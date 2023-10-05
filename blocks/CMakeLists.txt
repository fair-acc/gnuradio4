add_subdirectory(basic)
add_subdirectory(filter)
add_subdirectory(fourier)
add_subdirectory(math)

if (ENABLE_TESTING)
    add_subdirectory(testing)
    add_subdirectory(demo)
endif ()
