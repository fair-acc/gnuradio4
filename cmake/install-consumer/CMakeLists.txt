# Smallest possible third-party consumer of an installed gnuradio4. Asserts that every package the install tree
# advertises (gnuradio4 + GnuRadioBlockLib) is resolvable via `find_package(... CONFIG REQUIRED)` from a clean build
# dir.
cmake_minimum_required(VERSION 3.27)
project(gr4_install_consumer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(gnuradio4 CONFIG REQUIRED)
find_package(GnuRadioBlockLib CONFIG REQUIRED)

if(NOT TARGET gnuradio4::parse_registrations)
  message(FATAL_ERROR "GnuRadioBlockLib loaded but the parser IMPORTED target is missing")
endif()
get_target_property(_parser_loc gnuradio4::parse_registrations IMPORTED_LOCATION)
if(NOT EXISTS "${_parser_loc}")
  message(FATAL_ERROR "Parser executable does not exist on disk: ${_parser_loc}")
endif()

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE gnuradio4::gnuradio-core gnuradio4::gnuradio-blocklib-core
                                       gnuradio4::gnuradio-algorithm)
