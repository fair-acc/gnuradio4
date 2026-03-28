#include <boost/ut.hpp>

#include <memory_resource>
#include <numeric>

#include <gnuradio-4.0/algorithm/BurstTaper.hpp>
#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>

using namespace boost::ut;
using namespace gr::algorithm;

namespace {
void consumeSamples(BurstTaper<float>& taper, std::size_t n) {
    for (std::size_t i = 0UZ; i < n; ++i) {
        expect(ge(taper.processOne(), -1.f));
    }
}
} // namespace

const boost::ut::suite<"BurstTaper"> burstTaperTests = [] {
    using Taper = BurstTaper<float>;

    "coefficient generation"_test = [] {
        "generateEdge produces correct length"_test = [] {
            for (auto type : {TaperType::None, TaperType::Linear, TaperType::RaisedCosine, TaperType::Tukey, TaperType::Gaussian, TaperType::Mushroom, TaperType::MushroomSine}) {
                expect(eq(Taper::generateEdge(type, 64).size(), 64UZ));
            }
        };

        "nSamples=0 returns empty"_test = [] {
            for (auto type : {TaperType::None, TaperType::Linear, TaperType::RaisedCosine, TaperType::Mushroom, TaperType::MushroomSine}) {
                expect(Taper::generateEdge(type, 0).empty());
            }
        };

        "nSamples=1 returns single value"_test = [] { expect(eq(Taper::generateEdge(TaperType::Linear, 1, true).size(), 1UZ)); };

        "rising edge starts at 0 and ends at 1"_test = [] {
            for (auto type : {TaperType::Linear, TaperType::RaisedCosine, TaperType::Gaussian, TaperType::Mushroom, TaperType::MushroomSine}) {
                auto edge = Taper::generateEdge(type, 128, true);
                expect(approx(edge.front(), 0.f, 1e-6f)) << "rise start";
                expect(approx(edge.back(), 1.f, 1e-6f)) << "rise end";
            }
        };

        "falling edge starts at 1 and ends at 0"_test = [] {
            for (auto type : {TaperType::Linear, TaperType::RaisedCosine, TaperType::Gaussian, TaperType::Mushroom, TaperType::MushroomSine}) {
                auto edge = Taper::generateEdge(type, 128, false);
                expect(approx(edge.front(), 1.f, 1e-6f)) << "fall start";
                expect(approx(edge.back(), 0.f, 1e-6f)) << "fall end";
            }
        };

        "rise and reversed fall are symmetric"_test = [] {
            for (auto type : {TaperType::Linear, TaperType::RaisedCosine, TaperType::Mushroom, TaperType::MushroomSine}) {
                auto rise = Taper::generateEdge(type, 128, true);
                auto fall = Taper::generateEdge(type, 128, false);
                std::ranges::reverse(fall);
                for (std::size_t i = 0UZ; i < rise.size(); ++i) {
                    expect(approx(rise[i], fall[i], 1e-6f));
                }
            }
        };

        "None taper is all ones"_test = [] {
            for (auto v : Taper::generateEdge(TaperType::None, 64, true)) {
                expect(approx(v, 1.f, 1e-6f));
            }
        };

        "Linear is monotonically increasing"_test = [] {
            auto edge = Taper::generateEdge(TaperType::Linear, 128, true);
            for (std::size_t i = 1UZ; i < edge.size(); ++i) {
                expect(ge(edge[i], edge[i - 1]));
            }
        };

        "RaisedCosine is monotonically increasing"_test = [] {
            auto edge = Taper::generateEdge(TaperType::RaisedCosine, 128, true);
            for (std::size_t i = 1UZ; i < edge.size(); ++i) {
                expect(ge(edge[i], edge[i - 1]));
            }
        };

        "RaisedCosine power parameter changes curvature"_test = [] {
            auto base = Taper::generateEdge(TaperType::RaisedCosine, 128, true);
            auto slow = Taper::generateEdge(TaperType::RaisedCosine, 128, true, 2.0f);
            auto fast = Taper::generateEdge(TaperType::RaisedCosine, 128, true, 0.5f);
            expect(lt(slow[64], base[64])) << "p=2 rises slower at midpoint";
            expect(gt(fast[64], base[64])) << "p=0.5 rises faster at midpoint";
            expect(approx(slow.front(), 0.f, 1e-6f));
            expect(approx(slow.back(), 1.f, 1e-6f));
            expect(approx(fast.front(), 0.f, 1e-6f));
            expect(approx(fast.back(), 1.f, 1e-6f));
        };

        "Tukey with alpha=2 equals RaisedCosine"_test = [] {
            auto tukey  = Taper::generateEdge(TaperType::Tukey, 128, true, 2.0f);
            auto cosine = Taper::generateEdge(TaperType::RaisedCosine, 128, true);
            for (std::size_t i = 0UZ; i < tukey.size(); ++i) {
                expect(approx(tukey[i], cosine[i], 1e-5f));
            }
        };

        "Gaussian g(0)=0 and g(1)=1"_test = [] {
            auto edge = Taper::generateEdge(TaperType::Gaussian, 128, true);
            expect(approx(edge.front(), 0.f, 1e-6f));
            expect(approx(edge.back(), 1.f, 1e-6f));
        };

        "Mushroom has zero integral"_test = [] {
            constexpr std::size_t n    = 10'000;
            auto                  edge = Taper::generateEdge(TaperType::Mushroom, n, true);
            double                sum  = std::accumulate(edge.begin(), edge.end(), 0.0) - (static_cast<double>(edge.front()) + static_cast<double>(edge.back())) / 2.0;
            sum /= static_cast<double>(n - 1);
            expect(approx(sum, 0.0, 1e-3)) << std::format("integral = {}", sum);
        };

        "MushroomSine has zero integral"_test = [] {
            constexpr std::size_t n    = 10'000;
            auto                  edge = Taper::generateEdge(TaperType::MushroomSine, n, true);
            double                sum  = std::accumulate(edge.begin(), edge.end(), 0.0) - (static_cast<double>(edge.front()) + static_cast<double>(edge.back())) / 2.0;
            sum /= static_cast<double>(n - 1);
            expect(approx(sum, 0.0, 1e-3)) << std::format("integral = {}", sum);
        };

        "Mushroom g(0)=0, g(1)=1 and goes negative"_test = [] {
            auto  edge   = Taper::generateEdge(TaperType::Mushroom, 256, true);
            float minVal = *std::ranges::min_element(edge);
            expect(approx(edge.front(), 0.f, 1e-6f));
            expect(approx(edge.back(), 1.f, 1e-6f));
            expect(lt(minVal, 0.f)) << std::format("min = {}", minVal);
        };

        "MushroomSine g(0)=0, g(1)=1 and goes negative"_test = [] {
            auto  edge   = Taper::generateEdge(TaperType::MushroomSine, 256, true);
            float minVal = *std::ranges::min_element(edge);
            expect(approx(edge.front(), 0.f, 1e-6f));
            expect(approx(edge.back(), 1.f, 1e-6f));
            expect(lt(minVal, 0.f)) << std::format("min = {}", minVal);
        };

        "generateTaper correct length and flat region"_test = [] {
            constexpr std::size_t nRise = 32, nFlat = 100, nFall = 32;
            auto                  taper = Taper::generateTaper(TaperType::RaisedCosine, nRise, nFlat, nFall);
            expect(eq(taper.size(), 164UZ));
            for (std::size_t i = nRise; i < nRise + nFlat; ++i) {
                expect(approx(taper[i], 1.f, 1e-6f));
            }
        };

        "generateTaper with zero edges is all ones"_test = [] {
            auto taper = Taper::generateTaper(TaperType::Mushroom, 0, 50, 0);
            expect(eq(taper.size(), 50UZ));
            for (auto v : taper) {
                expect(approx(v, 1.f, 1e-6f));
            }
        };

        "double precision"_test = [] {
            using TaperD = BurstTaper<double>;
            auto edge    = TaperD::generateEdge(TaperType::MushroomSine, 10'000, true);
            expect(approx(edge.front(), 0.0, 1e-12));
            expect(approx(edge.back(), 1.0, 1e-12));
            double sum = std::accumulate(edge.begin(), edge.end(), 0.0) - (edge.front() + edge.back()) / 2.0;
            sum /= static_cast<double>(edge.size() - 1);
            expect(approx(sum, 0.0, 1e-4));
        };

        "span-based generateEdge"_test = [] {
            std::array<float, 128> buffer{};
            Taper::generateEdge(TaperType::RaisedCosine, buffer);
            expect(approx(buffer.front(), 0.f, 1e-6f));
            expect(approx(buffer.back(), 1.f, 1e-6f));
        };

        "span-based generateTaper"_test = [] {
            std::array<float, 164> buffer{};
            Taper::generateTaper(TaperType::Linear, buffer, 32, 100, 32);
            expect(approx(buffer[0], 0.f, 1e-6f));
            expect(approx(buffer[31], 1.f, 1e-6f));
            expect(approx(buffer[32], 1.f, 1e-6f));
            expect(approx(buffer[131], 1.f, 1e-6f));
            expect(approx(buffer[163], 0.f, 1e-6f));
        };
    };

    "template variants"_test = [] {
        "fixed-size BurstTaper<float, 48>"_test = [] {
            BurstTaper<float, 48> taper;
            expect(eq(taper.rampLength(), 48UZ));
            expect(taper.configure(TaperType::RaisedCosine, 0.001f, 48000.f).has_value());
        };

        "fixed-size rejects mismatched ramp length"_test = [] {
            BurstTaper<float, 48> taper;
            expect(!taper.configure(TaperType::Linear, 0.01f, 1000.f).has_value());
        };

        "fixed-size processOne"_test = [] {
            BurstTaper<float, 4> taper;
            expect(taper.configure(TaperType::Linear, 0.004f, 1000.f).has_value());
            expect(taper.setTarget(true));
            expect(approx(taper.processOne(), 0.f, 1e-6f));
            expect(approx(taper.processOne(), 1.f / 3.f, 1e-5f));
            expect(approx(taper.processOne(), 2.f / 3.f, 1e-5f));
            expect(approx(taper.processOne(), 1.f, 1e-6f));
            expect(taper.isOn());
        };

        "PMR-backed BurstTaper"_test = [] {
            BurstTaper<float, std::dynamic_extent, std::pmr::polymorphic_allocator<float>> taper;
            expect(taper.configure(TaperType::RaisedCosine, 0.001f, 48000.f).has_value());
            expect(eq(taper.rampLength(), 48UZ));
            expect(taper.setTarget(true));
            expect(approx(taper.processOne(), 0.f, 1e-6f));
        };
    };

    "construction"_test = [] {
        "default is Linear, zero-length, Off"_test = [] {
            Taper taper;
            expect(taper.isOff());
            expect(eq(taper.rampLength(), 0UZ));
            expect(taper.type() == TaperType::Linear);
            expect(!taper.targetOn());
        };

        "explicit constructor builds LUT"_test = [] {
            Taper taper(TaperType::RaisedCosine, 0.001f, 48000.f);
            expect(eq(taper.rampLength(), 48UZ));
            expect(taper.type() == TaperType::RaisedCosine);
            expect(taper.isOff());
        };

        "explicit constructor with zero rampTime is instant"_test = [] {
            Taper taper(TaperType::Mushroom, 0.f, 48000.f);
            expect(eq(taper.rampLength(), 0UZ));
            expect(taper.setTarget(true));
            expect(taper.isOn());
        };

        "constructor with invalid params stays at defaults"_test = [] {
            Taper taper(TaperType::RaisedCosine, -1.f, 1000.f); // negative rampTime
            expect(eq(taper.rampLength(), 0UZ));
            expect(taper.type() == TaperType::Linear); // unchanged from default

            Taper taper2(TaperType::Gaussian, 0.01f, 0.f); // zero sampleRate
            expect(eq(taper2.rampLength(), 0UZ));
            expect(taper2.type() == TaperType::Linear);
        };

        "usable as member field with brace init"_test = [] {
            struct MockBlock {
                BurstTaper<float> taper{TaperType::Linear, 0.01f, 1000.f};
            };
            MockBlock block;
            expect(eq(block.taper.rampLength(), 10UZ));
        };

        "setTarget(true) transitions instantly to On"_test = [] {
            Taper taper;
            expect(taper.setTarget(true));
            expect(approx(taper.processOne(), 1.f, 1e-6f));
            expect(taper.isOn());
        };

        "on then off transitions instantly"_test = [] {
            Taper taper;
            expect(taper.setTarget(true));
            expect(taper.isOn());
            expect(taper.setTarget(false));
            expect(taper.isOff());
        };

        "applyInPlace with empty coefficients"_test = [] {
            Taper              taper;
            std::vector<float> data(10, 5.0f);
            taper.applyInPlace(data);
            expect(std::ranges::all_of(data, [](float v) { return v == 0.f; }));

            std::ranges::fill(data, 5.0f);
            expect(taper.setTarget(true));
            taper.applyInPlace(data);
            expect(std::ranges::all_of(data, [](float v) { return v == 5.f; }));
        };
    };

    "configure"_test = [] {
        "sets ramp length from physical units"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::RaisedCosine, 0.001f, 48000.f).has_value());
            expect(eq(taper.rampLength(), 48UZ));
            expect(taper.isOff());
        };

        "very small rampTime produces rampLength=0 (instant transition)"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 1e-6f, 1000.f).has_value());
            expect(eq(taper.rampLength(), 0UZ));
        };

        "rejects negative rampTime"_test = [] {
            Taper taper;
            expect(!taper.configure(TaperType::Linear, -1.f, 1000.f).has_value());
        };

        "rejects zero sampleRate"_test = [] {
            Taper taper;
            expect(!taper.configure(TaperType::Linear, 0.01f, 0.f).has_value());
        };

        "rejects negative sampleRate"_test = [] {
            Taper taper;
            expect(!taper.configure(TaperType::Linear, 0.01f, -1000.f).has_value());
        };

        "rampTime=0 produces instant transition (rampLength=0)"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.f, 48000.f).has_value());
            expect(eq(taper.rampLength(), 0UZ));
            expect(taper.setTarget(true));
            expect(taper.isOn());
            expect(taper.setTarget(false));
            expect(taper.isOff());
        };

        "resets state when called mid-ramp"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.01f, 1000.f).has_value());
            expect(taper.setTarget(true));
            expect(approx(taper.processOne(), 0.f, 1e-6f));
            expect(taper.configure(TaperType::RaisedCosine, 0.002f, 48000.f).has_value());
            expect(taper.isOff());
        };
    };

    "state machine"_test = [] {
        "full on/off cycle"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.01f, 1000.f).has_value());
            auto r1 = taper.setTarget(true);
            expect(r1);
            expect(eq(taper.rampLength(), 10UZ));
            consumeSamples(taper, 10);
            expect(taper.isOn());
            auto r2 = taper.setTarget(false);
            expect(r2);
            consumeSamples(taper, 10);
            expect(taper.isOff());
        };

        "setTarget(true) when On returns false"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.002f, 1000.f).has_value());
            expect(taper.setTarget(true));
            consumeSamples(taper, 2);
            auto r = taper.setTarget(true);
            expect(!r);
        };

        "setTarget(false) when Off returns false"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.001f, 1000.f).has_value());
            auto r = taper.setTarget(false);
            expect(!r);
        };

        "setTarget(true) when RampUp returns false"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.01f, 1000.f).has_value());
            expect(taper.setTarget(true));
            expect(approx(taper.processOne(), 0.f, 1e-6f));
            auto r = taper.setTarget(true);
            expect(!r);
        };

        "setTarget(false) when RampDown returns false"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.002f, 1000.f).has_value());
            expect(taper.setTarget(true));
            consumeSamples(taper, 2);
            expect(taper.setTarget(false));
            expect(approx(taper.processOne(), 1.f, 1e-6f));
            auto r = taper.setTarget(false);
            expect(!r);
        };

        "reset returns to Off"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::RaisedCosine, 0.01f, 48000.f).has_value());
            expect(taper.setTarget(true));
            expect(approx(taper.processOne(), 0.f, 1e-6f));
            taper.reset();
            expect(taper.isOff());
        };

        "remainingSamples decreases mid-ramp"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.01f, 1000.f).has_value());
            expect(eq(taper.remainingSamples(), 0UZ));
            expect(taper.setTarget(true));
            expect(eq(taper.remainingSamples(), 10UZ));
            consumeSamples(taper, 3);
            expect(eq(taper.remainingSamples(), 7UZ));
            consumeSamples(taper, 7);
            expect(taper.isOn());
            expect(eq(taper.remainingSamples(), 0UZ));
        };

        "targetOn is latched immediately on non-force setTarget"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.004f, 1000.f).has_value());
            expect(taper.setTarget(true));
            expect(taper.targetOn());
            expect(approx(taper.processOne(), 0.f, 1e-6f));
            expect(taper.setTarget(false));                // queue deactivate
            expect(!taper.targetOn());                     // target latched immediately
            expect(taper.phase() == Taper::Phase::RampUp); // still ramping
        };
    };

    "processOne envelope values"_test = [] {
        Taper taper;
        expect(taper.configure(TaperType::Linear, 0.004f, 1000.f).has_value());
        expect(taper.setTarget(true));

        expect(approx(taper.processOne(), 0.f, 1e-6f));
        expect(approx(taper.processOne(), 1.f / 3.f, 1e-5f));
        expect(approx(taper.processOne(), 2.f / 3.f, 1e-5f));
        expect(approx(taper.processOne(), 1.f, 1e-6f));
        expect(taper.isOn());
        expect(approx(taper.processOne(), 1.f, 1e-6f));

        expect(taper.setTarget(false));
        expect(approx(taper.processOne(), 1.f, 1e-6f));
        expect(approx(taper.processOne(), 2.f / 3.f, 1e-5f));
        expect(approx(taper.processOne(), 1.f / 3.f, 1e-5f));
        expect(approx(taper.processOne(), 0.f, 1e-6f));
        expect(taper.isOff());
    };

    "bulk processing"_test = [] {
        "applyInPlace ramps and zeros"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.004f, 1000.f).has_value());

            std::vector<float> data(20, 1.0f);
            taper.applyInPlace(data);
            expect(std::ranges::all_of(data, [](float v) { return v == 0.f; }));

            std::ranges::fill(data, 2.0f);
            expect(taper.setTarget(true));
            taper.applyInPlace(std::span(data).subspan(0, 6));
            expect(approx(data[0], 0.f, 1e-6f));
            expect(approx(data[3], 2.f, 1e-6f));
            expect(taper.isOn());
        };

        "applyInPlace mid-chunk ramp-down"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.004f, 1000.f).has_value());
            expect(taper.setTarget(true));
            consumeSamples(taper, 4);
            expect(taper.setTarget(false));
            std::vector<float> data(10, 1.0f);
            taper.applyInPlace(data);
            expect(approx(data[0], 1.f, 1e-6f));
            expect(approx(data[3], 0.f, 1e-6f));
            expect(approx(data[4], 0.f, 1e-6f));
            expect(taper.isOff());
        };

        "applyInPlace ramp split across two calls resumes correctly"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.01f, 1000.f).has_value()); // 10 samples
            expect(taper.setTarget(true));

            std::vector<float> chunk1(4, 1.0f);
            taper.applyInPlace(chunk1);
            expect(approx(chunk1[0], 0.f, 1e-6f));
            expect(taper.isRamping());
            expect(eq(taper.remainingSamples(), 6UZ));

            std::vector<float> chunk2(10, 1.0f);
            taper.applyInPlace(chunk2);
            expect(approx(chunk2[5], 1.f, 1e-6f)); // ramp finished at index 5
            expect(approx(chunk2[6], 1.f, 1e-6f)); // On: pass-through
            expect(taper.isOn());
        };

        "applyTo spans ramp-up through target-change to ramp-down in one call"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.004f, 1000.f).has_value());
            expect(taper.setTarget(true));
            expect(taper.setTarget(false)); // queue deactivate during ramp-up

            const std::vector<float> input(20, 1.0f);
            std::vector<float>       output(20, -1.0f);
            taper.applyTo(input, output);          // should: ramp-up(4) + ramp-down(4) + zeros(12)
            expect(approx(output[0], 0.f, 1e-6f)); // ramp-up start
            expect(approx(output[3], 1.f, 1e-6f)); // ramp-up end
            expect(approx(output[4], 1.f, 1e-6f)); // ramp-down start
            expect(approx(output[7], 0.f, 1e-6f)); // ramp-down end
            expect(approx(output[8], 0.f, 1e-6f)); // zeroed tail
            expect(approx(output[19], 0.f, 1e-6f));
            expect(taper.isOff());
        };

        "applyTo copies with envelope"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.004f, 1000.f).has_value());
            const std::vector<float> input(10, 3.0f);
            std::vector<float>       output(10, -1.0f);

            taper.applyTo(input, output);
            expect(std::ranges::all_of(output, [](float v) { return v == 0.f; }));

            expect(taper.setTarget(true));
            taper.applyTo(input, output);
            expect(approx(output[0], 0.f, 1e-6f));
            expect(approx(output[3], 3.f, 1e-6f));
            expect(taper.isOn());

            std::ranges::fill(output, -1.0f);
            taper.applyTo(input, output);
            expect(approx(output[0], 3.f, 1e-6f));
        };

        "applyTo does not modify input"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.004f, 1000.f).has_value());
            expect(taper.setTarget(true));
            const std::vector<float> input(10, 5.0f);
            std::vector<float>       output(10);
            taper.applyTo(input, output);
            expect(std::ranges::all_of(input, [](float v) { return v == 5.f; }));
        };
    };

    "force transitions"_test = [] {
        "force on during RampDown"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.01f, 1000.f).has_value());
            expect(taper.setTarget(true));
            consumeSamples(taper, 10);
            expect(taper.setTarget(false));
            float lastVal = 0.f;
            for (std::size_t i = 0UZ; i < 3UZ; ++i) {
                lastVal = taper.processOne();
            }
            auto r = taper.setTarget(true, true);
            expect(r);
            expect(taper.phase() == Taper::Phase::RampUp);
            expect(approx(taper.processOne(), lastVal, 0.15f));
        };

        "force off during RampUp"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.01f, 1000.f).has_value());
            expect(taper.setTarget(true));
            float lastVal = 0.f;
            for (std::size_t i = 0UZ; i < 4UZ; ++i) {
                lastVal = taper.processOne();
            }
            auto r = taper.setTarget(false, true);
            expect(r);
            expect(taper.phase() == Taper::Phase::RampDown);
            expect(approx(taper.processOne(), lastVal, 0.15f));
        };
    };

    "target-driven transitions"_test = [] {
        "off during RampUp queues ramp-down"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.004f, 1000.f).has_value());
            expect(taper.setTarget(true));
            expect(approx(taper.processOne(), 0.f, 1e-6f));
            expect(taper.setTarget(false));
            consumeSamples(taper, 3);
            expect(taper.phase() == Taper::Phase::RampDown);
            consumeSamples(taper, 4);
            expect(taper.isOff());
        };

        "on during RampDown queues ramp-up"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.004f, 1000.f).has_value());
            expect(taper.setTarget(true));
            consumeSamples(taper, 4);
            expect(taper.setTarget(false));
            expect(approx(taper.processOne(), 1.f, 1e-6f));
            expect(taper.setTarget(true));
            consumeSamples(taper, 3);
            expect(taper.phase() == Taper::Phase::RampUp);
            consumeSamples(taper, 4);
            expect(taper.isOn());
        };

        "last setTarget determines outcome"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.004f, 1000.f).has_value());
            expect(taper.setTarget(true));
            expect(approx(taper.processOne(), 0.f, 1e-6f));
            expect(taper.setTarget(false));
            consumeSamples(taper, 3);
            expect(taper.phase() == Taper::Phase::RampDown);
            expect(taper.setTarget(true));
            consumeSamples(taper, 4);
            expect(taper.phase() == Taper::Phase::RampUp);
            consumeSamples(taper, 4);
            expect(taper.isOn());
        };

        "rapid toggle settles to last state"_test = [] {
            Taper taper;
            expect(taper.configure(TaperType::Linear, 0.004f, 1000.f).has_value());
            expect(taper.setTarget(true));  // Off → RampUp: true (started)
            expect(taper.setTarget(false)); // RampUp, target=off: true (queued)
            expect(!taper.setTarget(true)); // RampUp, target=on: false (already heading there)
            expect(taper.setTarget(false)); // RampUp, target=off: true (queued again)
            consumeSamples(taper, 4);
            expect(taper.phase() == Taper::Phase::RampDown);
            consumeSamples(taper, 4);
            expect(taper.isOff());
        };
    };
};

const boost::ut::suite<"BurstTaper visual"> burstTaperVisualTests = [] {
    using namespace gr::graphs;

    auto plotTaper = [](TaperType type, std::vector<double> params = {}, double yMin = -1.2, double yMax = 1.2) {
        constexpr std::size_t nEdge  = 500UZ;
        constexpr std::size_t nFlat  = 1000UZ;
        constexpr std::size_t nPad   = 200UZ;
        constexpr std::size_t nBurst = nEdge + nFlat + nEdge;
        constexpr std::size_t nTotal = nPad + nBurst + nPad;

        if (params.empty()) {
            params.push_back(0.0);
        }

        std::vector<double> rectangular(nPad, 0.0);
        rectangular.insert(rectangular.end(), nBurst, 1.0);
        rectangular.resize(nTotal, 0.0);

        std::vector<double> x(nTotal);
        std::iota(x.begin(), x.end(), 0.0);

        auto chart        = ImChart<120, 25>({{-1., static_cast<double>(nTotal)}, {yMin, yMax}});
        chart.axis_name_x = "sample index";
        chart.axis_name_y = "amplitude";
        chart.draw(x, rectangular, "rectangular");

        bool first = true;
        for (double p : params) {
            auto taperedBurst = BurstTaper<double>::generateTaper(type, nEdge, nFlat, nEdge, p);

            std::vector<double> tapered(nPad, 0.0);
            tapered.insert(tapered.end(), taperedBurst.begin(), taperedBurst.end());
            tapered.resize(nTotal, 0.0);

            if (first) {
                chart.draw(x, tapered, (p == 0.0) ? std::format("{}", type) : std::format("{}(p={:g})", type, p));
                first = false;
            } else {
                chart.draw(x, tapered, std::format("(p={:g})", p));
            }
        }
        chart.draw();
    };

    "Linear taper"_test       = [&] { plotTaper(TaperType::Linear); };
    "RaisedCosine taper"_test = [&] { plotTaper(TaperType::RaisedCosine, {0.0, 0.5, 2.0}); };
    "Tukey taper"_test        = [&] { plotTaper(TaperType::Tukey, {0.0, 1.0, 2.0}); };
    "Gaussian taper"_test     = [&] { plotTaper(TaperType::Gaussian, {0.0, 0.2, 0.8}); };
    "Mushroom taper"_test     = [&] { plotTaper(TaperType::Mushroom); };
    "MushroomSine taper"_test = [&] { plotTaper(TaperType::MushroomSine); };

    "all tapers comparison"_test = [] {
        constexpr std::size_t nEdge  = 500UZ;
        constexpr std::size_t nFlat  = 1000UZ;
        constexpr std::size_t nPad   = 200UZ;
        constexpr std::size_t nBurst = nEdge + nFlat + nEdge;
        constexpr std::size_t nTotal = nPad + nBurst + nPad;

        std::vector<double> rectangular(nPad, 0.0);
        rectangular.insert(rectangular.end(), nBurst, 1.0);
        rectangular.resize(nTotal, 0.0);

        std::vector<double> x(nTotal);
        std::iota(x.begin(), x.end(), 0.0);

        auto chart        = ImChart<120, 28>({{-1., static_cast<double>(nTotal)}, {-0.6, 1.4}});
        chart.axis_name_x = "sample index";
        chart.axis_name_y = "amplitude";
        chart.draw(x, rectangular, "rectangular");

        constexpr std::array taperTypes = {TaperType::Linear, TaperType::RaisedCosine, TaperType::Tukey, TaperType::Gaussian, TaperType::Mushroom};

        for (auto type : taperTypes) {
            auto burst = BurstTaper<double>::generateTaper(type, nEdge, nFlat, nEdge);

            std::vector<double> tapered(nPad, 0.0);
            tapered.insert(tapered.end(), burst.begin(), burst.end());
            tapered.resize(nTotal, 0.0);

            chart.draw(x, tapered, std::format("{}", type));
        }
        chart.draw();
    };
};

int main() { /* not needed for UT */ }
