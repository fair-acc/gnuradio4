add_executable(check_name_convention check_name_convention.cpp)
target_link_libraries(check_name_convention PUBLIC yaml-cpp)


if (CMAKE_CXX_COMPILER_ID MATCHES "Clang") # ensure Clang is used
    # equivalent compile command line: clang++-18 check_naming_convention.cpp -o check_naming_convention `llvm-config --cxxflags --ldflags --libs --system-libs` -fexceptions -lclang-cpp
    find_program(LLVM_CONFIG_PATH NAMES llvm-config DOC "Path to llvm-config utility")

    if (NOT LLVM_CONFIG_PATH)
        message(FATAL_ERROR "llvm-config not found")
    else ()
        execute_process(COMMAND ${LLVM_CONFIG_PATH} --cxxflags OUTPUT_VARIABLE LLVM_CXX_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${LLVM_CONFIG_PATH} --ldflags OUTPUT_VARIABLE LLVM_LD_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${LLVM_CONFIG_PATH} --libs OUTPUT_VARIABLE LLVM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
        execute_process(COMMAND ${LLVM_CONFIG_PATH} --system-libs OUTPUT_VARIABLE LLVM_SYSTEM_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif ()

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LLVM_CXX_FLAGS}")

    add_executable(check_name_convention check_name_convention.cpp)
    target_link_libraries(check_name_convention PUBLIC yaml-cpp ${LLVM_LIBS} ${LLVM_SYSTEM_LIBS} ${LLVM_LD_FLAGS})
    target_link_libraries(check_name_convention PUBLIC clang-cpp)
endif ()


