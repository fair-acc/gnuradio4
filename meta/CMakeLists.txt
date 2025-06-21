add_library(gnuradio-meta INTERFACE)
target_include_directories(gnuradio-meta INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                   $<INSTALL_INTERFACE:include/>)
target_link_libraries(gnuradio-meta INTERFACE gnuradio-options vir)

set(meta_public_headers
    include/gnuradio-4.0/meta/formatter.hpp
    include/gnuradio-4.0/meta/reflection.hpp
    include/gnuradio-4.0/meta/typelist.hpp
    include/gnuradio-4.0/meta/utils.hpp
    include/gnuradio-4.0/meta/UncertainValue.hpp)

set_target_properties(gnuradio-meta PROPERTIES PUBLIC_HEADER "${meta_public_headers}")
install(
  TARGETS gnuradio-meta
  EXPORT gnuradio4Targets
  PUBLIC_HEADER DESTINATION include/gnuradio-4.0/meta)

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
