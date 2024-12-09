# Check for optional SoapySDR local dependency https://github.com/pothosware/SoapySDR
if (ONNX_FOUND)
  add_library(gr-onnx INTERFACE)
  target_link_libraries(gr-onnx INTERFACE gnuradio-options gnuradio-meta gnuradio-core fmt gnuradio-algorithm ${ONNX_LIBRARIES})
  target_include_directories(gr-onnx INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/> $<INSTALL_INTERFACE:include/> ${ONNX_INCLUDE_DIRS})

  add_subdirectory(src)

  if (ENABLE_TESTING)
    add_subdirectory(test)
  endif()
  message(INFO "ONNX development files found:")
  message(INFO "   ONNX_FOUND:        ${ONNX_FOUND}")
  message(INFO "   ONNX_VERSION:      ${ONNX_VERSION}")
  message(INFO "   ONNX_INCLUDE_DIRS: ${ONNX_INCLUDE_DIRS}")
  message(INFO "   ONNX_LIBRARIES:    ${ONNX_LIBRARIES}")
  message(INFO "   ONNX_CXX_FLAGS:    ${ONNX_CXX_FLAGS}")

else()
  message(WARNING "ONNX development files not found: ${ONNX_FOUND} - skipping ONNX support")
endif()