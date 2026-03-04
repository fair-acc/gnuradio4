#include "qa_FileIo_common.hpp"

const boost::ut::suite<"basic file IO real tests"> basicFileIORealTests = [] {
    using namespace boost::ut;
    using namespace gr;

    constexpr auto kRealTypes = std::tuple<float, double, gr::UncertainValue<float>, gr::UncertainValue<double>>();

    using enum gr::blocks::fileio::Mode;
    "overwrite mode real"_test = []<typename T>(const T&) { gr::blocks::fileio::qa::runTest<T>(overwrite); } | kRealTypes;
    "append mode real"_test    = []<typename T>(const T&) { gr::blocks::fileio::qa::runTest<T>(append); } | kRealTypes;
    "create new mode real"_test = []<typename T>(const T&) { gr::blocks::fileio::qa::runTest<T>(multi); } | kRealTypes;
};
