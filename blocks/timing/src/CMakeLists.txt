add_executable(ex0_gps_receiver ex0_gps_receiver.cpp)
target_link_libraries(ex0_gps_receiver PRIVATE gnuradio-core gr-timing)

add_executable(ex1_gps_receiver_gr4 ex1_gps_receiver_gr4.cpp)
target_link_libraries(
  ex1_gps_receiver_gr4
  PRIVATE gnuradio-options
          gnuradio-core
          gnuradio-blocklib-core
          gr-timing
          gr-testing)

if(EMSCRIPTEN)
  # shared shell HTML for both examples
  set(GPS_SHELL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/gps_shell.html")

  # ex0: standalone NMEADevice + NMEAParser example (pthreads for SerialPort IO)
  target_compile_options(ex0_gps_receiver PRIVATE -Wno-error)
  target_link_options(ex0_gps_receiver PRIVATE "SHELL:--shell-file ${GPS_SHELL_FILE}")
  set_target_properties(ex0_gps_receiver PROPERTIES SUFFIX ".html")

  # ex1: GR4 block example (pthreads for scheduler + IO thread)
  target_compile_options(ex1_gps_receiver_gr4 PRIVATE -Wno-error)
  target_link_options(ex1_gps_receiver_gr4 PRIVATE "SHELL:--shell-file ${GPS_SHELL_FILE}")
  set_target_properties(ex1_gps_receiver_gr4 PROPERTIES SUFFIX ".html")

  add_custom_target(
    serve_gps_receiver
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "  GPS receiver examples served at:"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo
            "    http://localhost:8080/ex0_gps_receiver.html  (standalone NMEADevice + NMEAParser)"
    COMMAND ${CMAKE_COMMAND} -E echo "    http://localhost:8080/ex1_gps_receiver_gr4.html  (GR4 GpsSource block)"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "  Open in Chrome or Edge [WebSerial required]."
    COMMAND ${CMAKE_COMMAND} -E echo "  Press Ctrl+C to stop the server."
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/serve_coop.py 8080 ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ex0_gps_receiver ex1_gps_receiver_gr4
    COMMENT "Serving GPS receiver WASM examples on http://localhost:8080/"
    VERBATIM USES_TERMINAL)
endif()
