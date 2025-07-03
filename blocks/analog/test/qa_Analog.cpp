/* blocks/analog/test/qa_Analog.cpp
 * ------------------------------------------------------------
 * Simple test suite for the Analog block collection.
 * Uses standard assertions instead of boost/ut.
 */

#include <iostream>
#include <cmath>
#include <cassert>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <gnuradio-4.0/analog/Analog.hpp>           // the blocks under test

using namespace gr;
using namespace gr::testing;
using namespace gr::blocks::analog;

/* Helper function for float comparison */
bool near(float a, float b, float tol) {
    return std::abs(a - b) <= tol;
}

/* Test 1: SigSource - sine wave */
void test_sigsource_sine() {
    std::cout << "Testing SigSource - sine wave... ";
    
    constexpr std::size_t N     = 8;       // one full period
    constexpr double      freq  = 1.0;   // double, not float
    constexpr double      fs    = 8.0;
    constexpr double      amp   = 1.0;
    constexpr float       tol   = 1e-6f;

    Graph g;
    /* SigSource test -------------------------------------------------- */
    auto& src  = g.emplaceBlock<SigSource<float>>(property_map{
                {"f0", freq}, {"fs", fs}, {"amp", amp},
                {"n_samples_max", N}});
    auto& sink = g.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>();

    assert(g.connect<"out">(src).to<"in">(sink) == ConnectionResult::SUCCESS);

    scheduler::Simple sch{std::move(g)};
    assert(sch.runAndWait().has_value());

    assert(sink._samples.size() == N);

    for (std::size_t k = 0; k < N; ++k) {
        const float expected = static_cast<float>(
            amp * std::sin(2.0 * M_PI * static_cast<double>(k) /
                           static_cast<double>(N)));
        assert(near(sink._samples[k], expected, tol));
    }
    
    std::cout << "PASSED" << std::endl;
}

/* Test 2: NoiseSource - RMS test */
void test_noisesource_rms() {
    std::cout << "Testing NoiseSource - RMS... ";
    
    constexpr std::size_t N   = 1024;
    constexpr float       tol = 0.4f;

    Graph g;
    /* NoiseSource test ------------------------------------------------ */
    auto& src  = g.emplaceBlock<NoiseSource<float>>(property_map{
                {"amp", 1.0}, {"seed", 42u},
                {"n_samples_max", N}});
    auto& sink = g.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_BULK>>();

    assert(g.connect<"out">(src).to<"in">(sink) == ConnectionResult::SUCCESS);

    scheduler::Simple sch{std::move(g)};
    assert(sch.runAndWait().has_value());

    assert(sink._samples.size() == N);

    float sum_sq = 0.0f;
    for (auto v : sink._samples)
        sum_sq += v * v;
    const float rms = std::sqrt(sum_sq / float(N));

    assert(rms > 1.0f - tol && rms < 1.0f + tol);
    
    std::cout << "PASSED (RMS = " << rms << ")" << std::endl;
}

int main() {
    std::cout << "Running Analog block tests..." << std::endl;
    
    test_sigsource_sine();
    test_noisesource_rms();
    
    std::cout << "All tests passed!" << std::endl;
    return 0;
}