#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

#include <iostream>

using namespace boost::ut;

using namespace std::string_view_literals;

#include <GrBasicBlocks.hpp>
#include <GrElectricalBlocks.hpp>
#include <GrFileIoBlocks.hpp>
#include <GrFilterBlocks.hpp>
#include <GrFourierBlocks.hpp>
#include <GrHttpBlocks.hpp>
#include <GrTestingBlocks.hpp>

const boost::ut::suite TagTests = [] {
    auto&       registry = gr::globalBlockRegistry();
    std::size_t result   = 0UZ;
    result += gr::blocklib::initGrBasicBlocks(registry);
    result += gr::blocklib::initGrElectricalBlocks(registry);
    result += gr::blocklib::initGrFileIoBlocks(registry);
    result += gr::blocklib::initGrFilterBlocks(registry);
    result += gr::blocklib::initGrFourierBlocks(registry);
    result += gr::blocklib::initGrHttpBlocks(registry);
    result += gr::blocklib::initGrTestingBlocks(registry);
    if (result) {
        std::print("Warning: Failed to init {} blocks\n", result);
    }

    "CheckAvailableBlocks"_test = [&] {
        expect(gt(registry.keys().size(), 20UZ));

        expect(registry.contains("gr::basic::ClockSource"sv));
        expect(registry.contains("gr::testing::Delay<float32>"sv));
        expect(registry.contains("gr::testing::Delay<float64>"sv));
        expect(registry.contains("gr::testing::NullSource<float32>"sv));
        expect(registry.contains("gr::testing::NullSource<complex<float32>>"sv));
        expect(registry.contains("gr::testing::NullSource<gr::Packet<float32>>"sv));
        expect(registry.contains("gr::testing::NullSource<gr::Tensor<float32>>"sv));
        expect(registry.contains("gr::testing::NullSource<gr::DataSet<float32>>"sv));
        expect(registry.contains("gr::testing::ConstantSource<float32>"sv));
        expect(registry.contains("gr::testing::ConstantSource<complex<float32>>"sv));
        expect(registry.contains("gr::testing::ConstantSource<gr::Packet<float32>>"sv));
        expect(registry.contains("gr::testing::ConstantSource<gr::Tensor<float32>>"sv));
        expect(registry.contains("gr::testing::ConstantSource<gr::DataSet<float32>>"sv));
        expect(registry.contains("gr::testing::SlowSource<float32>"sv));
        expect(registry.contains("gr::testing::SlowSource<complex<float32>>"sv));
        expect(registry.contains("gr::testing::SlowSource<gr::Packet<float32>>"sv));
        expect(registry.contains("gr::testing::SlowSource<gr::Tensor<float32>>"sv));
        expect(registry.contains("gr::testing::SlowSource<gr::DataSet<float32>>"sv));
        expect(registry.contains("gr::testing::CountingSource<float32>"sv));
        expect(registry.contains("gr::testing::CountingSource<complex<float32>>"sv));
        expect(registry.contains("gr::testing::Copy<float32>"sv));
        expect(registry.contains("gr::testing::Copy<complex<float32>>"sv));
        expect(registry.contains("gr::testing::Copy<gr::Packet<float32>>"sv));
        expect(registry.contains("gr::testing::Copy<gr::Tensor<float32>>"sv));
        expect(registry.contains("gr::testing::Copy<gr::DataSet<float32>>"sv));
        expect(registry.contains("gr::testing::HeadBlock<float32>"sv));
        expect(registry.contains("gr::testing::HeadBlock<complex<float32>>"sv));
        expect(registry.contains("gr::testing::HeadBlock<gr::Packet<float32>>"sv));
        expect(registry.contains("gr::testing::HeadBlock<gr::Tensor<float32>>"sv));
        expect(registry.contains("gr::testing::HeadBlock<gr::DataSet<float32>>"sv));
        expect(registry.contains("gr::testing::NullSink<float32>"sv));
        expect(registry.contains("gr::testing::NullSink<complex<float32>>"sv));
        expect(registry.contains("gr::testing::NullSink<gr::Packet<float32>>"sv));
        expect(registry.contains("gr::testing::NullSink<gr::Tensor<float32>>"sv));
        expect(registry.contains("gr::testing::NullSink<gr::DataSet<float32>>"sv));
        expect(registry.contains("gr::blocks::fileio::BasicFileSink<float32>"sv));
        expect(registry.contains("gr::blocks::type::converter::Convert<float32, float32>"sv));
        expect(registry.contains("gr::blocks::type::converter::Convert<float32, float64>"sv));
        expect(registry.contains("gr::blocks::type::converter::ScalingConvert<float32, float32>"sv));
        expect(registry.contains("gr::blocks::type::converter::ScalingConvert<float32, float64>"sv));
        expect(registry.contains("gr::basic::DataSink<float32>"sv));
        expect(registry.contains("gr::blocks::basic::SchmittTrigger<float32, (gr::trigger::InterpolationMethod)0>"sv));
#if defined(_WIN32)
        expect(registry.contains("gr::electrical::PowerMetrics<float32, 3ull>"sv));
#else
        expect(registry.contains("gr::electrical::PowerMetrics<float32, 3ul>"sv));
#endif
        expect(registry.contains("gr::http::HttpBlock<float32>"sv));
        expect(registry.contains("gr::filter::fir_filter<float32>"sv));
        expect(registry.contains("gr::blocks::fft::FFT<float32>"sv));
    };

    "CheckBlockInstantiations"_test = [&] {
        expect(registry.create("gr::testing::Delay<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::basic::DataSink<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::basic::ClockSource"sv, {}) != nullptr);
    };
};
int main() { /* not needed for UT */ }
