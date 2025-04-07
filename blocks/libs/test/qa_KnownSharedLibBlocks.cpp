#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

#include <iostream>

using namespace boost::ut;

using namespace std::string_view_literals;

extern "C" {
bool gr_blocklib_init_module_GrFilterBlocks(gr::BlockRegistry& registry);
bool gr_blocklib_init_module_GrFourierBlocks(gr::BlockRegistry& registry);
bool gr_blocklib_init_module_GrElectricalBlocks(gr::BlockRegistry& registry);
bool gr_blocklib_init_module_GrHttpBlocks(gr::BlockRegistry& registry);
bool gr_blocklib_init_module_GrFileIoBlocks(gr::BlockRegistry& registry);
bool gr_blocklib_init_module_GrTestingBlocks(gr::BlockRegistry& registry);
bool gr_blocklib_init_module_GrBasicBlocks(gr::BlockRegistry& registry);
}

const boost::ut::suite TagTests = [] {
    auto&       registry = gr::globalBlockRegistry();
    std::size_t result   = 0UZ;
    result += gr_blocklib_init_module_GrFilterBlocks(registry);
    result += gr_blocklib_init_module_GrElectricalBlocks(registry);
    result += gr_blocklib_init_module_GrHttpBlocks(registry);
    result += gr_blocklib_init_module_GrFileIoBlocks(registry);
    result += gr_blocklib_init_module_GrTestingBlocks(registry);
    result += gr_blocklib_init_module_GrBasicBlocks(registry);
    result += gr_blocklib_init_module_GrFourierBlocks(registry);
    if (result) {
        fmt::print("Warning: Failed to init {} blocks\n", result);
    }

    "CheckKnownBlocks"_test = [&] {
        expect(gt(registry.knownBlocks().size(), 20UZ));
        expect(registry.isBlockKnown("gr::basic::ClockSource"sv));
        expect(registry.isBlockKnown("gr::testing::Delay<float32>"sv));
        expect(registry.isBlockKnown("gr::testing::Delay<float64>"sv));
        expect(registry.isBlockKnown("gr::testing::NullSource<float32>"sv));
        expect(registry.isBlockKnown("gr::testing::NullSource<complex<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::NullSource<gr::Packet<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::NullSource<gr::Tensor<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::NullSource<gr::DataSet<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::ConstantSource<float32>"sv));
        expect(registry.isBlockKnown("gr::testing::ConstantSource<complex<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::ConstantSource<gr::Packet<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::ConstantSource<gr::Tensor<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::ConstantSource<gr::DataSet<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::SlowSource<float32>"sv));
        expect(registry.isBlockKnown("gr::testing::SlowSource<complex<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::SlowSource<gr::Packet<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::SlowSource<gr::Tensor<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::SlowSource<gr::DataSet<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::CountingSource<float32>"sv));
        expect(registry.isBlockKnown("gr::testing::CountingSource<complex<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::Copy<float32>"sv));
        expect(registry.isBlockKnown("gr::testing::Copy<complex<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::Copy<gr::Packet<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::Copy<gr::Tensor<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::Copy<gr::DataSet<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::HeadBlock<float32>"sv));
        expect(registry.isBlockKnown("gr::testing::HeadBlock<complex<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::HeadBlock<gr::Packet<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::HeadBlock<gr::Tensor<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::HeadBlock<gr::DataSet<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::NullSink<float32>"sv));
        expect(registry.isBlockKnown("gr::testing::NullSink<complex<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::NullSink<gr::Packet<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::NullSink<gr::Tensor<float32>>"sv));
        expect(registry.isBlockKnown("gr::testing::NullSink<gr::DataSet<float32>>"sv));
        expect(registry.isBlockKnown("gr::blocks::fileio::BasicFileSink<float32>"sv));
        expect(registry.isBlockKnown("gr::blocks::type::converter::Convert<float32, float32>"sv));
        expect(registry.isBlockKnown("gr::blocks::type::converter::Convert<float32, float64>"sv));
        expect(registry.isBlockKnown("gr::blocks::type::converter::ScalingConvert<float32, float32>"sv));
        expect(registry.isBlockKnown("gr::blocks::type::converter::ScalingConvert<float32, float64>"sv));
        expect(registry.isBlockKnown("gr::basic::DataSink<float32>"sv));
        expect(registry.isBlockKnown("gr::blocks::basic::SchmittTrigger<float32, (gr::trigger::InterpolationMethod)0>"sv));
        expect(registry.isBlockKnown("gr::electrical::PowerMetrics<float32, 3ul>"sv));
        expect(registry.isBlockKnown("gr::http::HttpBlock<float32>"sv));
        expect(registry.isBlockKnown("gr::filter::fir_filter<float32>"sv));
        expect(registry.isBlockKnown("gr::blocks::fft::FFT<float32>"sv));
    };

    "CheckBlockInstantiations"_test = [&] {
        expect(registry.createBlock("gr::testing::Delay<float32>"sv, {}) != nullptr);
        expect(registry.createBlock("gr::basic::DataSink<float32>"sv, {}) != nullptr);
        expect(registry.createBlock("gr::basic::ClockSource"sv, {}) != nullptr);
    };
};
int main() { /* not needed for UT */ }
