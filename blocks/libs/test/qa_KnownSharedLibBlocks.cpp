#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

#include <iostream>

using namespace boost::ut;

using namespace std::string_view_literals;

#include <gnuradio-4.0/GrAnalogBlocks.hpp>
#include <gnuradio-4.0/GrAudioBlocks.hpp>
#include <gnuradio-4.0/GrBasicBlocks.hpp>
#include <gnuradio-4.0/GrDigitalBlocks.hpp>
#include <gnuradio-4.0/GrElectricalBlocks.hpp>
#include <gnuradio-4.0/GrFileIoBlocks.hpp>
#include <gnuradio-4.0/GrFilterBlocks.hpp>
#include <gnuradio-4.0/GrFourierBlocks.hpp>
#include <gnuradio-4.0/GrHttpBlocks.hpp>
#include <gnuradio-4.0/GrTestingBlocks.hpp>
#include <gnuradio-4.0/GrTriggerBlocks.hpp>

const boost::ut::suite TagTests = [] {
    auto&       registry = gr::globalBlockRegistry();
    std::size_t result   = 0UZ;
    result += gr::blocklib::initGrAnalogBlocks(registry);
    result += gr::blocklib::initGrBasicBlocks(registry);
    result += gr::blocklib::initGrDigitalBlocks(registry);
    result += gr::blocklib::initGrAudioBlocks(registry);
    result += gr::blocklib::initGrElectricalBlocks(registry);
    result += gr::blocklib::initGrFileIoBlocks(registry);
    result += gr::blocklib::initGrFilterBlocks(registry);
    result += gr::blocklib::initGrFourierBlocks(registry);
    result += gr::blocklib::initGrHttpBlocks(registry);
    result += gr::blocklib::initGrTestingBlocks(registry);
    result += gr::blocklib::initGrTriggerBlocks(registry);
    if (result) {
        std::print("Warning: Failed to init {} blocks\n", result);
    }

    "CheckAvailableBlocks"_test = [&] {
        expect(gt(registry.keys().size(), 20UZ));

        expect(registry.contains("gr::analog::AMDemodulator<float32>"sv));
        expect(registry.contains("gr::analog::AMDemodulator<float64>"sv));
        expect(registry.contains("gr::analog::AMModulator<float32>"sv));
        expect(registry.contains("gr::analog::AMModulator<float64>"sv));
        expect(registry.contains("gr::analog::AutomaticGainControl<float32>"sv));
        expect(registry.contains("gr::analog::AutomaticGainControl<float64>"sv));
        expect(registry.contains("gr::analog::FMDemodulator<float32>"sv));
        expect(registry.contains("gr::analog::FMDemodulator<float64>"sv));
        expect(registry.contains("gr::analog::FMModulator<float32>"sv));
        expect(registry.contains("gr::analog::FMModulator<float64>"sv));
        expect(registry.contains("gr::analog::FMDeemphasis<float32>"sv));
        expect(registry.contains("gr::analog::FMDeemphasis<float64>"sv));
        expect(registry.contains("gr::analog::FMPreemphasis<float32>"sv));
        expect(registry.contains("gr::analog::FMPreemphasis<float64>"sv));
        expect(registry.contains("gr::analog::FMStereoDecoder<float32>"sv));
        expect(registry.contains("gr::analog::FMStereoDecoder<float64>"sv));
        expect(registry.contains("gr::analog::PhaseLockedLoop<float32>"sv));
        expect(registry.contains("gr::analog::PhaseLockedLoop<float64>"sv));
        expect(registry.contains("gr::analog::PMDemodulator<float32>"sv));
        expect(registry.contains("gr::analog::PMDemodulator<float64>"sv));
        expect(registry.contains("gr::analog::PMModulator<float32>"sv));
        expect(registry.contains("gr::analog::PMModulator<float64>"sv));
        expect(registry.contains("gr::analog::PowerSquelch<float32>"sv));
        expect(registry.contains("gr::analog::PowerSquelch<float64>"sv));
        expect(registry.contains("gr::digital::ConstellationDemapper<float32>"sv));
        expect(registry.contains("gr::digital::ConstellationDemapper<float64>"sv));
        expect(registry.contains("gr::digital::ConstellationModulator<float32>"sv));
        expect(registry.contains("gr::digital::ConstellationModulator<float64>"sv));
        expect(registry.contains("gr::digital::CostasLoop<float32>"sv));
        expect(registry.contains("gr::digital::CostasLoop<float64>"sv));
        expect(registry.contains("gr::digital::DifferentialDecoder"sv));
        expect(registry.contains("gr::digital::DifferentialEncoder"sv));
        expect(registry.contains("gr::digital::ManchesterDecoder<float32>"sv));
        expect(registry.contains("gr::digital::ManchesterDecoder<float64>"sv));
        expect(registry.contains("gr::digital::ManchesterEncoder"sv));
        expect(registry.contains("gr::digital::RDSDecoder"sv));
        expect(registry.contains("gr::digital::RootRaisedCosineFilter<float32>"sv));
        expect(registry.contains("gr::digital::RootRaisedCosineFilter<float64>"sv));
        expect(registry.contains("gr::digital::SymbolSync<float32>"sv));
        expect(registry.contains("gr::digital::SymbolSync<float64>"sv));
        expect(registry.contains("gr::basic::ClockSource"sv));
        expect(registry.contains("gr::testing::Sleep<float32>"sv));
        expect(registry.contains("gr::testing::Sleep<float64>"sv));
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
        expect(registry.contains("gr::blocks::trigger::SchmittTrigger<float32, gr::trigger::InterpolationMethod::NO_INTERPOLATION>"sv));
        expect(registry.contains("gr::electrical::PowerMetrics<float32, 3>"sv)); // NTTP now renders as the enum/value name (was the obsolete (Enum)0 / 3ul form)
        expect(registry.contains("gr::http::HttpSource"sv));
        expect(registry.contains("gr::http::HttpSink"sv));
        expect(registry.contains("gr::blocks::fileio::WavSource<float32>"sv));
        expect(registry.contains("gr::blocks::fileio::WavSink<float32>"sv));
        expect(registry.contains("gr::audio::AudioSink<float32>"sv));
        expect(registry.contains("gr::filter::fir_filter<float32>"sv));
        expect(registry.contains("gr::blocks::fft::FFT<float32>"sv));
    };

    "CheckBlockInstantiations"_test = [&] {
        expect(registry.create("gr::analog::AMDemodulator<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::analog::AMModulator<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::analog::AutomaticGainControl<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::analog::FMDemodulator<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::analog::FMModulator<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::analog::FMDeemphasis<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::analog::FMPreemphasis<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::analog::FMStereoDecoder<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::analog::PhaseLockedLoop<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::analog::PMDemodulator<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::analog::PMModulator<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::analog::PowerSquelch<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::digital::ConstellationDemapper<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::digital::ConstellationModulator<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::digital::CostasLoop<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::digital::DifferentialDecoder"sv, {}) != nullptr);
        expect(registry.create("gr::digital::DifferentialEncoder"sv, {}) != nullptr);
        expect(registry.create("gr::digital::ManchesterDecoder<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::digital::ManchesterEncoder"sv, {}) != nullptr);
        expect(registry.create("gr::digital::RDSDecoder"sv, {}) != nullptr);
        expect(registry.create("gr::digital::RootRaisedCosineFilter<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::digital::SymbolSync<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::testing::Sleep<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::blocks::fileio::WavSource<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::audio::AudioSink<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::basic::DataSink<float32>"sv, {}) != nullptr);
        expect(registry.create("gr::basic::ClockSource"sv, {}) != nullptr);
    };
};
int main() { /* not needed for UT */ }
