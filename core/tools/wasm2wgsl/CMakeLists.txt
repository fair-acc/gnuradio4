# a build-host tool: core/CMakeLists.txt configures and builds it with a native compiler, as blocklib_generator does its parser
cmake_minimum_required(VERSION 3.25)
project(wasm2wgsl LANGUAGES CXX)

add_executable(wasm2wgsl wasm2wgsl.cpp)
target_compile_features(wasm2wgsl PRIVATE cxx_std_23)
# the WGSL helpers are `#embed`ded: C++26, which GCC 15 takes silently and Clang as an extension
target_compile_options(wasm2wgsl PRIVATE $<$<CXX_COMPILER_ID:Clang>:-Wno-c23-extensions>)
