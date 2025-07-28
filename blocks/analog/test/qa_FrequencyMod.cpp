#include <boost/ut.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>
#include <gnuradio-4.0/analog/FrequencyMod.hpp>
#include <numbers>
#include <cmath>

using namespace gr;
using namespace gr::testing;
using namespace gr::blocks::analog;
using namespace boost::ut;

suite freqmod_tests = [] {
    "frequency modulator"_test = [] {
        constexpr float    k  = std::numbers::pi_v<float>/4.0f;   // sensitivity
        const float src_data[] = {0.25f,0.5f,0.25f,-0.25f,-0.5f,-0.25f};

        auto sincos = [](float ph){ return std::complex<float>(std::cos(ph),std::sin(ph)); };
        std::vector<std::complex<float>> ref;
        {   float acc = 0.0f;
            for(float x: src_data){ acc += k*x; ref.push_back(sincos(acc)); } }

        Graph g;
        auto& src  = g.emplaceBlock<TagSource<float>>(property_map{{"values",std::vector<float>(std::begin(src_data),std::end(src_data))},
                                                                   {"n_samples_max",std::size(src_data)}});
        auto& mod  = g.emplaceBlock<FrequencyMod<float>>(property_map{{"sensitivity",k}});
        auto& sink = g.emplaceBlock<TagSink<std::complex<float>,ProcessFunction::USE_PROCESS_BULK>>();

        expect(eq(g.connect<"out">(src).to<"in">(mod),ConnectionResult::SUCCESS));
        expect(eq(g.connect<"out">(mod).to<"in">(sink),ConnectionResult::SUCCESS));

        scheduler::Simple sch{std::move(g)};
        expect(bool{sch.runAndWait()});

        expect(sink._samples.size() == ref.size());
        for(std::size_t i=0;i<ref.size();++i)
            expect(std::abs(sink._samples[i]-ref[i]) < 1e-5f) << "i="<<i;
    };
};

int main(){ /* Boost.UT auto‑runs suites */ }
