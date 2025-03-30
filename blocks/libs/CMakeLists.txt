add_library(GnuRadioBlockLib SHARED src/BlockLib.cpp)

target_include_directories(GnuRadioBlockLib PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
                                                   $<INSTALL_INTERFACE:include/>)

target_link_libraries(
  GnuRadioBlockLib
  PRIVATE GrTestingBlocks
          GrFileIoBlocks
          GrBasicBlocks
          GrElectricalBlocks
          GrHttpBlocks
          GrFilterBlocks)

if(ENABLE_TESTING)
  add_subdirectory(test)
endif()
