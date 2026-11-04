add_library(gr-http INTERFACE)
add_library(gnuradio4::gr-http ALIAS gr-http)
target_link_libraries(gr-http INTERFACE gnuradio4::gnuradio-core gnuradio4::gnuradio-algorithm)
target_include_directories(gr-http INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                             $<INSTALL_INTERFACE:include/>)
install(TARGETS gr-http EXPORT gnuradio4Targets)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/gnuradio-4.0/http"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")

gr_add_block_library(
  GrHttpBlocks
  MAKE_SHARED_LIBRARY
  MAKE_PLUGIN
  HEADERS
  include/gnuradio-4.0/http/HttpBlock.hpp
  LINK_LIBRARIES
  gnuradio4::gr-http)

# the tests use the header-only interface target, not the shared block library, so they must build wherever testing is
# on -- including a SYCL configuration, which is BUILD_SHARED_LIBS=OFF and so has no `...Shared` target. Gating them on
# it meant any device case was only ever compiled where SYCL was absent, and a device case that always skips asserts
# nothing at all.
if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
