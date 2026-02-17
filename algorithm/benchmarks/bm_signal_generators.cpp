#include <benchmark.hpp>

#include <gnuradio-4.0/algorithm/rng/Xoshiro256pp.hpp>
#include <gnuradio-4.0/algorithm/signal/NoiseGenerator.hpp>
#include <gnuradio-4.0/algorithm/signal/SignalGeneratorCore.hpp>
#include <gnuradio-4.0/algorithm/signal/ToneGenerator.hpp>

#include <cstdint>
#include <format>
#include <random>
#include <vector>

template<std::floating_point F, std::size_t NRep = 500UZ>
void benchTones(std::string_view typeName) {
    using namespace benchmark;
    for (const auto& [type, name] : std::vector<std::pair<gr::signal::ToneType, std::string_view>>{
             {gr::signal::ToneType::Sin, "std::sin"},
             {gr::signal::ToneType::FastSin, "phasor sin"},
             {gr::signal::ToneType::Cos, "std::cos"},
             {gr::signal::ToneType::FastCos, "phasor cos"},
         }) {
        for (std::size_t N : {100UZ, 1024UZ, 65536UZ}) {
            gr::signal::ToneGenerator<F> gen;
            gen.configure(type, F(1000), F(48000), F(0), F(1), F(0));
            std::vector<F> buf(N);

            const auto bmName                                         = std::format("{:10} - {:6},N={:5}", name, typeName, N);
            ::benchmark::benchmark<NRep>(std::string_view(bmName), N) = [&] { gen.fill(buf); };
        }
    }
}

template<std::floating_point F, std::size_t NRep = 500UZ>
void benchTonesComplex(std::string_view typeName) {
    using namespace benchmark;
    for (const auto& [type, name] : std::vector<std::pair<gr::signal::ToneType, std::string_view>>{
             {gr::signal::ToneType::Sin, "std::sin"},
             {gr::signal::ToneType::FastSin, "phasor sin"},
             {gr::signal::ToneType::Cos, "std::cos"},
             {gr::signal::ToneType::FastCos, "phasor cos"},
         }) {
        for (std::size_t N : {100UZ, 1024UZ, 65536UZ}) {
            gr::signal::ToneGenerator<F> gen;
            gen.configure(type, F(1000), F(48000), F(0), F(1), F(0));
            std::vector<std::complex<F>> buf(N);

            const auto bmName                                         = std::format("{:10} - {:6},N={:5}", name, typeName, N);
            ::benchmark::benchmark<NRep>(std::string_view(bmName), N) = [&] { gen.fillComplex(buf); };
        }
    }
}

template<std::floating_point F, std::size_t NRep = 500UZ>
void benchNoise(std::string_view typeName) {
    using namespace benchmark;
    for (const auto& [type, name] : std::vector<std::pair<gr::signal::NoiseType, std::string_view>>{
             {gr::signal::NoiseType::Uniform, "std::uniform_real_distribution"},
             {gr::signal::NoiseType::Triangular, "triangular (Irwin-Hall)"},
             {gr::signal::NoiseType::Gaussian, "std::normal_distribution"},
         }) {
        for (std::size_t N : {100UZ, 1024UZ, 65536UZ}) {
            gr::signal::NoiseGenerator<F> gen;
            gen.configure(type, F(1), F(0), 42);
            std::vector<F> buf(N);

            const auto bmName                                         = std::format("{:30} - {:6},N={:5}", name, typeName, N);
            ::benchmark::benchmark<NRep>(std::string_view(bmName), N) = [&] { gen.fill(buf); };
        }
    }
}

template<std::floating_point F, std::size_t NRep = 500UZ>
void benchNoiseComplex(std::string_view typeName) {
    using namespace benchmark;
    for (const auto& [type, name] : std::vector<std::pair<gr::signal::NoiseType, std::string_view>>{
             {gr::signal::NoiseType::Uniform, "std::uniform_real_distribution"},
             {gr::signal::NoiseType::Triangular, "triangular (Irwin-Hall)"},
             {gr::signal::NoiseType::Gaussian, "std::normal_distribution"},
         }) {
        for (std::size_t N : {100UZ, 1024UZ, 65536UZ}) {
            gr::signal::NoiseGenerator<F> gen;
            gen.configure(type, F(1), F(0), 42);
            std::vector<std::complex<F>> buf(N);

            const auto bmName                                         = std::format("{:30} - {:6},N={:5}", name, typeName, N);
            ::benchmark::benchmark<NRep>(std::string_view(bmName), N) = [&] { gen.fillComplex(buf); };
        }
    }
}

template<std::integral T, std::size_t NRep = 500UZ>
void benchIntegerOutput(std::string_view typeName) {
    using namespace benchmark;
    for (const auto& [type, name] : std::vector<std::pair<gr::signal::SignalType, std::string_view>>{
             {gr::signal::SignalType::Sin, "std::sin"},
             {gr::signal::SignalType::FastSin, "phasor sin"},
             {gr::signal::SignalType::Cos, "std::cos"},
             {gr::signal::SignalType::FastCos, "phasor cos"},
             {gr::signal::SignalType::UniformNoise, "std::uniform_real_distribution"},
             {gr::signal::SignalType::TriangularNoise, "triangular (Irwin-Hall)"},
             {gr::signal::SignalType::GaussianNoise, "std::normal_distribution"},
         }) {
        const bool isNoise = static_cast<int>(type) >= static_cast<int>(gr::signal::SignalType::UniformNoise);
        for (std::size_t N : {100UZ, 1024UZ, 65536UZ}) {
            gr::signal::SignalGeneratorCore<T> core;
            core.configure(type, isNoise ? 0.f : 1000.f, 48000.f, 0.f, 30000.f, 0.f, isNoise ? 42ULL : 0ULL);
            std::vector<T> buf(N);

            const auto bmName                                         = std::format("{:30} - {:6},N={:5}", name, typeName, N);
            ::benchmark::benchmark<NRep>(std::string_view(bmName), N) = [&] { core.fill(buf); };
        }
    }
}

void benchmarkPrng() {
    using boost::ut::operator""_test;

    "Xoshiro256pp raw throughput"_test = [] {
        gr::rng::Xoshiro256pp rng(42);
        std::uint64_t         sink                                                                                 = 0;
        ::benchmark::benchmark<1000>(std::string_view("Xoshiro256pp operator()        - uint64,N=65536"), 65536UZ) = [&] {
            std::uint64_t localSink = 0; // local avoids pointer-aliasing with rng._state[]
            for (int i = 0; i < 65536; ++i) {
                localSink += rng();
            }
            sink = localSink;
            benchmark::force_to_memory(sink);
        };

        std::vector<std::uint64_t> buf(65536);
        ::benchmark::benchmark<1000>(std::string_view("Xoshiro256pp fillRaw()         - uint64,N=65536"), 65536UZ) = [&] {
            rng.fillRaw(buf);
            benchmark::force_to_memory(buf[0]);
        };
        ::benchmark::results::add_separator();
    };

    "std::mt19937_64 raw throughput"_test = [] {
        std::mt19937_64 rng(42);
        std::uint64_t   sink                                                                                       = 0;
        ::benchmark::benchmark<1000>(std::string_view("std::mt19937_64 operator()     - uint64,N=65536"), 65536UZ) = [&] {
            std::uint64_t localSink = 0; // same fix for fair comparison
            for (int i = 0; i < 65536; ++i) {
                localSink += rng();
            }
            sink = localSink;
            benchmark::force_to_memory(sink);
        };
        ::benchmark::results::add_separator();
    };
}

template<std::floating_point F, std::size_t NRep = 500UZ>
void benchStlNoise(std::string_view typeName) {
    using namespace benchmark;
    for (std::size_t N : {100UZ, 1024UZ, 65536UZ}) {
        {
            std::mt19937_64                   rng(42);
            std::uniform_real_distribution<F> dist(F(-1), F(1));
            std::vector<F>                    buf(N);
            const auto                        bmName                  = std::format("{:30} - {:6},N={:5}", "STL uniform_real_distribution", typeName, N);
            ::benchmark::benchmark<NRep>(std::string_view(bmName), N) = [&] {
                for (auto& s : buf) {
                    s = dist(rng);
                }
            };
        }
        {
            std::mt19937_64             rng(42);
            std::normal_distribution<F> dist(F(0), F(1));
            std::vector<F>              buf(N);
            const auto                  bmName                        = std::format("{:30} - {:6},N={:5}", "STL normal_distribution", typeName, N);
            ::benchmark::benchmark<NRep>(std::string_view(bmName), N) = [&] {
                for (auto& s : buf) {
                    s = dist(rng);
                }
            };
        }
    }
}

void benchmarkTones() {
    using boost::ut::operator""_test;

    "tones: float"_test = [] {
        benchTones<float>("float");
        ::benchmark::results::add_separator();
    };

    "tones: double"_test = [] {
        benchTones<double>("double");
        ::benchmark::results::add_separator();
    };

    "tones: complex<float>"_test = [] {
        benchTonesComplex<float>("cplx_f");
        ::benchmark::results::add_separator();
    };

    "tones: complex<double>"_test = [] {
        benchTonesComplex<double>("cplx_d");
        ::benchmark::results::add_separator();
    };
}

void benchmarkNoise() {
    using boost::ut::operator""_test;

    "noise: float"_test = [] {
        benchNoise<float>("float");
        ::benchmark::results::add_separator();
    };

    "noise: double"_test = [] {
        benchNoise<double>("double");
        ::benchmark::results::add_separator();
    };

    "noise: complex<float>"_test = [] {
        benchNoiseComplex<float>("cplx_f");
        ::benchmark::results::add_separator();
    };

    "noise: complex<double>"_test = [] {
        benchNoiseComplex<double>("cplx_d");
        ::benchmark::results::add_separator();
    };
}

void benchmarkStlNoise() {
    using boost::ut::operator""_test;

    "STL noise: float"_test = [] {
        benchStlNoise<float>("float");
        ::benchmark::results::add_separator();
    };

    "STL noise: double"_test = [] {
        benchStlNoise<double>("double");
        ::benchmark::results::add_separator();
    };
}

void benchmarkInteger() {
    using boost::ut::operator""_test;

    "integer output: int16"_test = [] {
        benchIntegerOutput<std::int16_t>("int16");
        ::benchmark::results::add_separator();
    };
}

inline const boost::ut::suite<"signal generator benchmarks"> _signal_gen_bm = [] {
    benchmarkPrng();
    benchmarkTones();
    benchmarkNoise();
    benchmarkStlNoise();
    benchmarkInteger();
};

int main() { /* not needed by the UT framework */ }
