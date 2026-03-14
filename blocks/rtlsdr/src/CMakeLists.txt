add_executable(ex_rtlsdr_capture ex_rtlsdr_capture.cpp)
target_link_libraries(
  ex_rtlsdr_capture
  PRIVATE gnuradio-options
          gnuradio-core
          gnuradio-blocklib-core
          gr-rtlsdr
          gr-testing)

if(EMSCRIPTEN)
  set(RTLSDR_SHELL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/rtlsdr_shell.html")

  target_compile_options(ex_rtlsdr_capture PRIVATE -Wno-error)
  target_link_options(
    ex_rtlsdr_capture
    PRIVATE
    "SHELL:--shell-file ${RTLSDR_SHELL_FILE}"
    "SHELL:-s ASYNCIFY"
    "SHELL:-s EXPORTED_RUNTIME_METHODS=['ccall','HEAPU8','wasmMemory']"
    "SHELL:-s EXPORTED_FUNCTIONS=['_main','_malloc','_free','_rtlsdr_getQueueBuf','_rtlsdr_getQueueHead','_rtlsdr_getQueueTail','_rtlsdr_getQueueCapacity','_rtlsdr_getQueueMask','_rtlsdrRequestPermission','_rtlsdrConnect','_startRtlSdrCapture','_stopRtlSdrCapture']"
    --bind)
  set_target_properties(ex_rtlsdr_capture PROPERTIES SUFFIX ".html")

  add_custom_target(
    serve_rtlsdr
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "  RTL-SDR capture example served at:"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "    http://localhost:8081/ex_rtlsdr_capture.html"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "  Open in Chrome or Edge [WebUSB required]."
    COMMAND ${CMAKE_COMMAND} -E echo "  Press Ctrl+C to stop the server."
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/serve_coop.py 8081 ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ex_rtlsdr_capture
    COMMENT "Serving RTL-SDR WASM example on http://localhost:8081/"
    VERBATIM USES_TERMINAL)
endif()
