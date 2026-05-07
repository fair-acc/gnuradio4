set(CPACK_PACKAGE_VENDOR "GNURadio")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Prototype implementations for a more compile-time efficient flowgraph API")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
set(CPACK_PACKAGE_CONTACT "admin@gnuradio.org")
set(_gr4_package_flavour "generic")
if(EMSCRIPTEN)
  set(_gr4_package_flavour "emscripten")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(_gr4_package_flavour "gcc")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(_gr4_package_flavour "clang")
endif()

if(NOT CPACK_PACKAGE_NAME)
  set(CPACK_PACKAGE_NAME "${PROJECT_NAME}${PROJECT_VERSION_MAJOR}-${_gr4_package_flavour}")
endif()

if(NOT DEFINED CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "/opt/${CPACK_PACKAGE_NAME}")
endif()

if(UNIX)
  if(NOT CPACK_GENERATOR)
    set(CPACK_GENERATOR "DEB")
  endif()

  set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)
  set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
  # When installing to a non-standard prefix (e.g. /opt/gnuradio4-gcc), dpkg-shlibdeps
  # cannot find the package's own private libraries. Point it to the install lib dir.
  set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS_PRIVATE_DIRS "${CPACK_PACKAGING_INSTALL_PREFIX}/lib")
  if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(CPACK_DEBIAN_DEBUGINFO_PACKAGE ON)
  endif()
endif()

include(CPack)
