#include <boost/ut.hpp>

#include <gnuradio-4.0/basic/CommonBlocks.hpp>
#include <gnuradio-4.0/basic/DataSink.hpp>
#include <gnuradio-4.0/basic/FunctionGenerator.hpp>
#include <gnuradio-4.0/basic/Selector.hpp>
#include <gnuradio-4.0/basic/SignalGenerator.hpp>

#include <gnuradio-4.0/GrBasicBlocks.hpp>

const boost::ut::suite AvailableBlockTests = [] {
    using namespace boost::ut;
    using namespace std::string_literals;

    gr::blocklib::initGrBasicBlocks(gr::globalBlockRegistry());

    "Registered"_test = [] {
        auto known = gr::globalBlockRegistry().keys();
        std::ranges::sort(known);
        std::vector<std::string> desired{
            //
            "builtin_counter<float32>"s,              //
            "builtin_counter<float64>"s,              //
            "builtin_multiply<float32>"s,             //
            "builtin_multiply<float64>"s,             //
            "gr::basic::DataSink<float32>"s,          //
            "gr::basic::DataSink<float64>"s,          //
            "gr::basic::DataSetSink<float32>"s,       //
            "gr::basic::DataSetSink<float64>"s,       //
            "gr::basic::FunctionGenerator<float32>"s, //
            "gr::basic::FunctionGenerator<float64>"s, //
            "gr::basic::Selector<float32>"s,          //
            "gr::basic::Selector<float64>"s,          //
            "gr::basic::SignalGenerator<float32>"s,   //
            "gr::basic::SignalGenerator<float64>"s    //
        };
        std::ranges::sort(desired);

        expect((std::ranges::includes(known, desired)));
    };

    "every converter is created under its own name"_test = [] {
        const std::vector<std::pair<std::string, std::string_view>> converters{
            {"gr::blocks::type::converter::ToRealImag<complex<float32>>"s, "ToRealImag"},
            {"gr::blocks::type::converter::RealImagToComplex<float32>"s, "RealImagToComplex"},
            {"gr::blocks::type::converter::RealImagToComplex<float64>"s, "RealImagToComplex"},
            {"gr::blocks::type::converter::ToMagPhase<complex<float32>>"s, "ToMagPhase"},
            {"gr::blocks::type::converter::ToMagPhase<complex<float64>>"s, "ToMagPhase"},
            {"gr::blocks::type::converter::MagPhaseToComplex<float32>"s, "MagPhaseToComplex"},
            {"gr::blocks::type::converter::MagPhaseToComplex<float64>"s, "MagPhaseToComplex"},
            {"gr::blocks::type::converter::Conjugate<complex<float32>>"s, "Conjugate"},
            {"gr::blocks::type::converter::Conjugate<complex<float64>>"s, "Conjugate"},
            {"gr::blocks::type::converter::SwapEndianness<uint32>"s, "SwapEndianness"},
            {"gr::blocks::type::converter::SwapEndianness<complex<float32>>"s, "SwapEndianness"},
        };
        for (const auto& [name, blockType] : converters) {
            auto block = gr::globalBlockRegistry().create(name, {});
            expect(fatal(block != nullptr)) << name;
            expect(block->typeName().contains(blockType)) << name << " created " << block->typeName();
        }
        const auto conjugates = std::ranges::count_if(gr::globalBlockRegistry().keys(), [](const std::string& key) { return key.starts_with("gr::blocks::type::converter::Conjugate<"); });
        expect(eq(conjugates, 4)) << "Conjugate for std::complex and gr::complex, float and double";
        expect(!gr::globalBlockRegistry().contains("gr::blocks::type::converter::ToRealImag<float32>")) << "a real stream has no imaginary part to split off";
    };
};

int main() { return boost::ut::cfg<boost::ut::override>.run(); }
