# FindLibSoundIo.cmake
# Locate the libsoundio shared library and headers.
#
# Upstream libsoundio ships neither a *Config.cmake nor a stable pkg-config file
# on every distribution; this module covers the gap. PkgConfig is queried as a
# hint when available, with a plain find_path/find_library fallback.
#
# Sets:
#   LibSoundIo_FOUND
#   LibSoundIo_INCLUDE_DIR
#   LibSoundIo_LIBRARY
#
# Provides imported target:
#   libsoundio_static — named to match the target produced by the in-tree
#                       FetchContent build, so consumers in blocks/audio/ work
#                       under either acquisition mode without renaming.
#                       (The "static" suffix is historical; the system library
#                       is shared.)

include(FindPackageHandleStandardArgs)

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(_LIBSOUNDIO_PC QUIET libsoundio)
endif()

find_path(
    LibSoundIo_INCLUDE_DIR
    NAMES soundio/soundio.h
    HINTS ${_LIBSOUNDIO_PC_INCLUDE_DIRS})

find_library(
    LibSoundIo_LIBRARY
    NAMES soundio libsoundio
    HINTS ${_LIBSOUNDIO_PC_LIBRARY_DIRS})

find_package_handle_standard_args(
    LibSoundIo
    REQUIRED_VARS LibSoundIo_LIBRARY LibSoundIo_INCLUDE_DIR
    VERSION_VAR _LIBSOUNDIO_PC_VERSION)

if(LibSoundIo_FOUND AND NOT TARGET libsoundio_static)
    add_library(libsoundio_static UNKNOWN IMPORTED)
    set_target_properties(
        libsoundio_static PROPERTIES
        IMPORTED_LOCATION "${LibSoundIo_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${LibSoundIo_INCLUDE_DIR}")
endif()

mark_as_advanced(LibSoundIo_INCLUDE_DIR LibSoundIo_LIBRARY)
