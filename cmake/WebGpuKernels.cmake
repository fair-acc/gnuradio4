# gr_translate_webgpu_kernels(<target> [STRIP_NAMES]): writes <target>.kernels.json beside <target>.wasm,
# the WGSL pipelines DeviceContextWebGpu.js loads at start-up;
# with STRIP_NAMES the name section, which only the translator reads, is removed after it
# (OpenDigitizer's UI: 32.6 of 169.5 MB).
# A no-op without the translator, as in a build without WebGPU.
# Called from any project, so it reads cache variables only: GR_WASM2WGSL_EXECUTABLE, GR_TINT_EXECUTABLE and
# GR_WEBGPU_LINK_DEPENDS, set by GR4's CMake files in a source build,
# and by gnuradio4Config.cmake in an installed one.
function(gr_translate_webgpu_kernels target)
  if(NOT GR_WASM2WGSL_EXECUTABLE)
    return()
  endif()
  cmake_parse_arguments(
    PARSE_ARGV
    1
    arg
    "STRIP_NAMES"
    ""
    "")
  # the runtime is linked in by `--js-library`, which CMake does not track as an input of the link, and a new translator
  # must translate again
  set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS ${GR_WEBGPU_LINK_DEPENDS})
  add_custom_command(
    TARGET ${target} POST_BUILD
    COMMAND ${GR_WASM2WGSL_EXECUTABLE} $<TARGET_FILE_DIR:${target}>/$<TARGET_FILE_BASE_NAME:${target}>.wasm
            $<TARGET_FILE_DIR:${target}>/$<TARGET_FILE_BASE_NAME:${target}>.kernels.json --tint ${GR_TINT_EXECUTABLE}
    COMMENT "Translating the WebGPU kernels of ${target}")
  if(arg_STRIP_NAMES)
    find_program(GR_LLVM_OBJCOPY_EXECUTABLE llvm-objcopy HINTS "${EMSCRIPTEN_ROOT_PATH}/../bin" REQUIRED)
    add_custom_command(
      TARGET ${target} POST_BUILD COMMAND ${GR_LLVM_OBJCOPY_EXECUTABLE} --remove-section=name
                                          $<TARGET_FILE_DIR:${target}>/$<TARGET_FILE_BASE_NAME:${target}>.wasm
      COMMENT "Stripping the name section of ${target}")
  endif()
endfunction()
