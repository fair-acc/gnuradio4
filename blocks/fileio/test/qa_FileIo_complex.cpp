#include "qa_FileIo_common.hpp"

const boost::ut::suite<"basic file IO complex tests"> basicFileIOComplexTests = [] {
    using namespace boost::ut;
    using namespace gr;

    constexpr auto kComplexTypes = std::tuple<std::complex<float>, std::complex<double>>();

    using enum gr::blocks::fileio::Mode;
    "overwrite mode complex"_test = []<typename T>(const T&) { gr::blocks::fileio::qa::runTest<T>(overwrite); } | kComplexTypes;
    "append mode complex"_test    = []<typename T>(const T&) { gr::blocks::fileio::qa::runTest<T>(append); } | kComplexTypes;
    "create new mode complex"_test = []<typename T>(const T&) { gr::blocks::fileio::qa::runTest<T>(multi); } | kComplexTypes;
};
