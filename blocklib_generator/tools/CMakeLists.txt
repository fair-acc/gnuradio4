cmake_minimum_required(VERSION 3.16)
project(parseReg LANGUAGES CXX)

# use HOST compiler to build parse_registrations
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_BUILD_TYPE Release)

add_executable(parse_registrations parse_registrations.cpp)

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang") # set default C++ STL to Clang's libc++ when using Clang
    target_compile_options(parse_registrations PRIVATE -stdlib=libc++)
    target_link_options(parse_registrations PRIVATE -stdlib=libc++ -lc++)
endif()
