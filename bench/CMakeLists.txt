function(append_compiler_flags TARGET_NAME)
    set(FLAGS_COMMON -Wall)
    if (EMSCRIPTEN)
        set(FLAGS_SPECIAL "")
    else ()
        set(FLAGS_SPECIAL -march=native)
    endif ()

    target_compile_options(${TARGET_NAME} PRIVATE ${FLAGS_COMMON} ${FLAGS_SPECIAL})
    target_link_options(${TARGET_NAME} PRIVATE ${FLAGS_COMMON} ${FLAGS_SPECIAL})
endfunction()

function(add_benchmark BM_NAME)
    add_executable(${BM_NAME} ${BM_NAME}.cpp)
    append_compiler_flags(${BM_NAME})
    target_link_libraries(${BM_NAME} PRIVATE gnuradio-options gnuradio-core refl-cpp fmt ut fftw)
endfunction()

add_benchmark(bm_example)