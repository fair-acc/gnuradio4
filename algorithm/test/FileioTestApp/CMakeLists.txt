cmake_minimum_required(VERSION 3.27)
project(FileioTestApp CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_executable(FileioTestApp main.cpp)

target_link_libraries(FileioTestApp PRIVATE gnuradio-core gnuradio-algorithm)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -s USE_PTHREADS=1 -s PTHREAD_POOL_SIZE=8")

target_link_options(
  FileioTestApp
  PRIVATE
  "SHELL:-s WASM=1"
  "SHELL:-s EXPORTED_RUNTIME_METHODS=['ccall','cwrap','HEAPU8']"
  "SHELL:-s EXPORTED_FUNCTIONS=['_malloc','_free','_fileio_dialog_on_bytes','_fileio_dialog_on_cancel','_fileio_dialog_on_error','_test_dialog_read','_test_save_file','_test_http_get','_test_http_get_404','_test_http_post','_main']"
  "SHELL:-s ASSERTIONS=2"
  "SHELL:-s EXIT_RUNTIME=0"
  "SHELL:-s ALLOW_MEMORY_GROWTH=1")

add_custom_command(
  TARGET FileioTestApp
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_CURRENT_SOURCE_DIR}/index.html
          $<TARGET_FILE_DIR:FileioTestApp>/index.html)
