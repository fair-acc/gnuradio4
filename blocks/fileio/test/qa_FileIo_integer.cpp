#include "qa_FileIo_common.hpp"

const boost::ut::suite<"basic file IO integer tests"> basicFileIOIntegerTests = [] {
    using namespace boost::ut;
    using namespace gr;

    constexpr auto kIntegerTypes = std::tuple<uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t>();

    using enum gr::blocks::fileio::Mode;
    "overwrite mode integers"_test = []<typename T>(const T&) { gr::blocks::fileio::qa::runTest<T>(overwrite); } | kIntegerTypes;
    "append mode integers"_test    = []<typename T>(const T&) { gr::blocks::fileio::qa::runTest<T>(append); } | kIntegerTypes;
    "create new mode integers"_test = []<typename T>(const T&) { gr::blocks::fileio::qa::runTest<T>(multi); } | kIntegerTypes;
};
