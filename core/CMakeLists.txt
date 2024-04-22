# we need to add an internal target that the pch compilation can rely on, which cannot be the final target since the pch property would introduce a cyclic dependency
add_library(gnuradio-core-internal INTERFACE)
target_include_directories(gnuradio-core-internal INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/> $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/src> $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include> $<INSTALL_INTERFACE:include/>)
target_link_libraries(gnuradio-core-internal INTERFACE gnuradio-options gnuradio-meta refl-cpp magic_enum pmtv vir yaml-cpp fmt)

# generate a precompiled header for all gnuradio-core headers
file(GENERATE OUTPUT empty_pch.cpp CONTENT "/*dummy pch file*/")
add_library(gnuradio-core-pch STATIC empty_pch.cpp)
target_link_libraries(gnuradio-core-pch PRIVATE gnuradio-core-internal)
target_precompile_headers(gnuradio-core-pch PRIVATE
        include/gnuradio-4.0/reader_writer_lock.hpp
        include/gnuradio-4.0/thread/thread_affinity.hpp
        include/gnuradio-4.0/thread/thread_pool.hpp
        include/gnuradio-4.0/WaitStrategy.hpp
        include/gnuradio-4.0/Sequence.hpp
        include/gnuradio-4.0/HistoryBuffer.hpp
        include/gnuradio-4.0/BlockRegistry.hpp
        include/gnuradio-4.0/Buffer.hpp
        include/gnuradio-4.0/BufferSkeleton.hpp
        include/gnuradio-4.0/LifeCycle.hpp
        include/gnuradio-4.0/Transactions.hpp
        # the cmake dependency modeling has to be fixed for pch's to work with plugins
        #include/gnuradio-4.0/plugin.hpp
        # include/gnuradio-4.0/PluginLoader.hpp
        include/gnuradio-4.0/reflection.hpp
        include/gnuradio-4.0/ClaimStrategy.hpp
        include/gnuradio-4.0/Settings.hpp
        include/gnuradio-4.0/Profiler.hpp
        include/gnuradio-4.0/DataSet.hpp
        include/gnuradio-4.0/PortTraits.hpp
        include/gnuradio-4.0/Tag.hpp
        include/gnuradio-4.0/annotated.hpp
        include/gnuradio-4.0/BlockTraits.hpp
        include/gnuradio-4.0/Message.hpp
        include/gnuradio-4.0/Graph.hpp
        include/gnuradio-4.0/Graph_yaml_importer.hpp
        include/gnuradio-4.0/Block.hpp
        include/gnuradio-4.0/CircularBuffer.hpp
        include/gnuradio-4.0/Port.hpp
        include/gnuradio-4.0/Scheduler.hpp
)

# this is the final target so be used by library consumers
add_library(gnuradio-core INTERFACE)
target_precompile_headers(gnuradio-core REUSE_FROM gnuradio-core-pch)
target_link_libraries(gnuradio-core INTERFACE gnuradio-core-internal)

# configure a header file to pass the CMake settings to the source code
configure_file("${PROJECT_SOURCE_DIR}/cmake/config.h.in"  "${CMAKE_CURRENT_BINARY_DIR}/include/gnuradio-4.0/config.h" @ONLY)
# TODO: install configure file... but not really meaningful for header only library, since compile flags are defined by the user...

install(
        TARGETS gnuradio-core
        EXPORT graphTargets
        PUBLIC_HEADER DESTINATION include/
)

add_subdirectory(src)

if (ENABLE_TESTING)
    add_subdirectory(test)
    add_subdirectory(benchmarks)
endif ()
