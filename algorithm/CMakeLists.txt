add_library(gnuradio-algorithm INTERFACE)
add_library(gnuradio4::gnuradio-algorithm ALIAS gnuradio-algorithm)
target_include_directories(gnuradio-algorithm INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                        $<INSTALL_INTERFACE:include/>)
install(TARGETS gnuradio-algorithm EXPORT gnuradio4Targets)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/gnuradio-4.0/algorithm"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/gnuradio-4.0")

target_link_libraries(
  gnuradio-algorithm
  INTERFACE gnuradio4::gnuradio-options
            gnuradio4::gnuradio-meta
            gnuradio4::vir
            gnuradio4::magic_enum)

# Native HTTP(S) support: cpr + system libcurl. Emscripten uses the browser Fetch API and does not need cpr. cpr is
# patched and ABI-shared (fetch-only); declared centrally in cmake/Dependencies.cmake.
set(GR_HTTP_ENABLED FALSE)
if(EMSCRIPTEN)
  set(GR_HTTP_ENABLED TRUE)
elseif(
  NOT
  GR_ENABLE_HTTP
  STREQUAL
  "OFF")
  find_package(CURL QUIET)
  if(GR_ENABLE_HTTP STREQUAL "ON" AND NOT CURL_FOUND)
    message(FATAL_ERROR "GR_ENABLE_HTTP=ON, but libcurl was not found. "
                        "Install libcurl (e.g. libcurl4-openssl-dev) or set GR_ENABLE_HTTP=OFF/OPTIONAL.")
  endif()

  if(CURL_FOUND)
    # cpr globals must live in a single shared library to avoid ODR violations across GR4's block shared libraries.
    # Force shared just for this fetch.
    set(_saved_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS ON)
    if(GR_ENABLE_HTTP STREQUAL "ON")
      gr4_resolve_dependency(cpr REQUIRED)
    else()
      gr4_resolve_dependency(cpr)
    endif()
    set(BUILD_SHARED_LIBS ${_saved_BUILD_SHARED_LIBS})

    if(cpr_FOUND)
      target_link_libraries(gnuradio-algorithm INTERFACE cpr::cpr)
      set(GR_HTTP_ENABLED TRUE)
      # GR4 builds with hidden visibility by default; restore default visibility on the shared cpr target to keep its
      # symbols linkable from downstream code.
      if(TARGET cpr)
        set_target_properties(cpr PROPERTIES CXX_VISIBILITY_PRESET default VISIBILITY_INLINES_HIDDEN OFF)
      endif()
      set(GR4_PKGCONFIG_EXTRA_LIBS
          "${GR4_PKGCONFIG_EXTRA_LIBS} -lcpr"
          PARENT_SCOPE)
    endif()
  endif()
endif()
target_compile_definitions(gnuradio-algorithm INTERFACE GR_HTTP_ENABLED=$<BOOL:${GR_HTTP_ENABLED}>)

# propagate to top-level so the pkg-config template and gnuradio4Config can see it
set(GR_HTTP_ENABLED
    ${GR_HTTP_ENABLED}
    PARENT_SCOPE)

if(ENABLE_EXAMPLES)
  add_subdirectory(src)
endif()

if(ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
