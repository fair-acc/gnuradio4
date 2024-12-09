#include <boost/ut.hpp>

#include <gnuradio-4.0/onnx/OnnxPreprocess.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <print>
#include <vector>

using namespace boost::ut;
using namespace gr::blocks::onnx;

namespace {

std::vector<float> makeRamp(std::size_t n, float start = 1.f) {
    std::vector<float> v(n);
    std::iota(v.begin(), v.end(), start);
    return v;
}

} // namespace

const boost::ut::suite<"OnnxPreprocess LogMAD"> logMadTests = [] {
    "LogMAD produces finite clipped output"_test = [] {
        auto input = makeRamp(256);

        std::vector<float> output(256);
        OnnxPreprocess<float>::normaliseLogMAD(input, output);

        for (std::size_t i = 0; i < output.size(); ++i) {
            expect(std::isfinite(output[i])) << "index" << i;
            expect(ge(output[i], -5.0f));
            expect(le(output[i], 10.0f));
        }
    };

    "LogMAD preserves monotonicity"_test = [] {
        auto input = makeRamp(512);

        std::vector<float> output(512);
        OnnxPreprocess<float>::normaliseLogMAD(input, output);

        for (std::size_t i = 1; i < output.size(); ++i) {
            expect(ge(output[i], output[i - 1])) << "monotonicity broken at index" << i;
        }
    };

    "LogMAD handles constant input"_test = [] {
        std::vector<float> input(128, 42.0f);
        std::vector<float> output(128);
        OnnxPreprocess<float>::normaliseLogMAD(input, output);

        for (const auto v : output) {
            expect(std::isfinite(v));
        }
    };

    "LogMAD custom clip range"_test = [] {
        auto input = makeRamp(64);

        std::vector<float> output(64);
        OnnxPreprocess<float>::normaliseLogMAD(input, output, -2.f, 3.f);

        for (const auto v : output) {
            expect(ge(v, -2.0f));
            expect(le(v, 3.0f));
        }
    };
};

const boost::ut::suite<"OnnxPreprocess MinMax"> minMaxTests = [] {
    "MinMax scales to [0,1]"_test = [] {
        auto input = makeRamp(100);

        std::vector<float> output(100);
        OnnxPreprocess<float>::normaliseMinMax(input, output);

        expect(lt(std::abs(output[0] - 0.f), 1e-6f)) << "min should be 0";
        expect(lt(std::abs(output[99] - 1.f), 1e-6f)) << "max should be 1";

        for (const auto v : output) {
            expect(ge(v, 0.f));
            expect(le(v, 1.f));
        }
    };

    "MinMax constant input gives zeros"_test = [] {
        std::vector<float> input(64, 5.f);
        std::vector<float> output(64);
        OnnxPreprocess<float>::normaliseMinMax(input, output);

        for (const auto v : output) {
            expect(lt(std::abs(v), 1e-6f)) << "constant input should map to 0";
        }
    };

    "MinMax preserves order"_test = [] {
        auto input = makeRamp(128);

        std::vector<float> output(128);
        OnnxPreprocess<float>::normaliseMinMax(input, output);

        for (std::size_t i = 1; i < output.size(); ++i) {
            expect(ge(output[i], output[i - 1]));
        }
    };
};

const boost::ut::suite<"OnnxPreprocess ZScore"> zScoreTests = [] {
    "ZScore centres on zero"_test = [] {
        auto input = makeRamp(256);

        std::vector<float> output(256);
        OnnxPreprocess<float>::normaliseZScore(input, output);

        // mean should be approximately 0
        float sum = 0.f;
        for (const auto v : output) {
            sum += v;
        }
        float mean = sum / static_cast<float>(output.size());
        expect(lt(std::abs(mean), 0.5f)) << "z-scored output mean should be near 0, got " << mean;
    };

    "ZScore clips to range"_test = [] {
        auto input = makeRamp(256);

        std::vector<float> output(256);
        OnnxPreprocess<float>::normaliseZScore(input, output, -3.f, 3.f);

        for (const auto v : output) {
            expect(ge(v, -3.f));
            expect(le(v, 3.f));
        }
    };

    "ZScore handles constant input"_test = [] {
        std::vector<float> input(64, 7.f);
        std::vector<float> output(64);
        OnnxPreprocess<float>::normaliseZScore(input, output);

        // with zero variance, result should be 0
        for (const auto v : output) {
            expect(lt(std::abs(v), 1e-3f));
        }
    };
};

const boost::ut::suite<"OnnxPreprocess Expression"> exprTests = [] {
    "simple pass-through expression"_test = [] {
        OnnxPreprocess<float> pp;
        expect(pp.configure(ResampleMode::None, NormaliseMode::Expression, "vecOut := vecIn").has_value());

        std::vector<float> input = {1.f, 2.f, 3.f, 4.f, 5.f};
        std::vector<float> output(5, 0.f);
        pp.normalise(input, output);

        for (std::size_t i = 0; i < input.size(); ++i) {
            expect(lt(std::abs(output[i] - input[i]), 1e-6f)) << "index" << i;
        }
    };

    "min-max via expression"_test = [] {
        OnnxPreprocess<float> pp;
        expect(pp.configure(ResampleMode::None, NormaliseMode::Expression, "vecOut := (vecIn - min_val) / (max_val - min_val + 1e-10)").has_value());

        auto               input = makeRamp(100);
        std::vector<float> output(100, 0.f);
        pp.normalise(input, output);

        expect(lt(std::abs(output[0] - 0.f), 1e-3f)) << "min should be ~0";
        expect(lt(std::abs(output[99] - 1.f), 1e-3f)) << "max should be ~1";
    };

    "z-score via expression"_test = [] {
        OnnxPreprocess<float> pp;
        expect(pp.configure(ResampleMode::None, NormaliseMode::Expression, "vecOut := (vecIn - mean_val) / (std_val + 1e-10)").has_value());

        auto               input = makeRamp(128);
        std::vector<float> output(128, 0.f);
        pp.normalise(input, output);

        // mean of z-scored data should be ~0
        float sum = 0.f;
        for (const auto v : output) {
            sum += v;
        }
        float mean = sum / static_cast<float>(output.size());
        expect(lt(std::abs(mean), 0.1f)) << "expression z-score mean should be ~0";
    };

    "uses pre-computed statistics"_test = [] {
        OnnxPreprocess<float> pp;
        // expression that uses median and mad
        expect(pp.configure(ResampleMode::None, NormaliseMode::Expression, "vecOut := (vecIn - median) / (1.4826 * mad + 1e-10)").has_value());

        auto               input = makeRamp(64);
        std::vector<float> output(64, 0.f);
        pp.normalise(input, output);

        // output should be finite and centred
        for (const auto v : output) {
            expect(std::isfinite(v));
        }
    };

    "invalid expression returns error"_test = [] {
        OnnxPreprocess<float> pp;
        auto                  result = pp.configure(ResampleMode::None, NormaliseMode::Expression, "vecOut := this_is_not_valid_syntax @#$");
        expect(!result.has_value()) << "invalid expression should return error";
        expect(result.error().message.contains("Error")) << "error message should describe the problem";
    };

    "large input vector"_test = [] {
        OnnxPreprocess<float> pp;
        expect(pp.configure(ResampleMode::None, NormaliseMode::Expression, "vecOut := vecIn / max_val").has_value());

        std::vector<float> input(4096);
        std::iota(input.begin(), input.end(), 1.f);
        std::vector<float> output(4096, 0.f);
        pp.normalise(input, output);

        expect(lt(std::abs(output.back() - 1.f), 1e-3f)) << "last element should be ~1";
        expect(gt(output[0], 0.f));
    };
};

const boost::ut::suite<"OnnxPreprocess dispatch"> dispatchTests = [] {
    "dispatch None mode copies input"_test = [] {
        OnnxPreprocess<float> pp;
        expect(pp.configure(ResampleMode::None, NormaliseMode::None).has_value());

        std::vector<float> input = {1.f, 2.f, 3.f};
        std::vector<float> output(3, 0.f);
        pp.normalise(input, output);

        for (std::size_t i = 0; i < input.size(); ++i) {
            expect(lt(std::abs(output[i] - input[i]), 1e-6f));
        }
    };

    "dispatch LogMAD mode"_test = [] {
        OnnxPreprocess<float> pp;
        expect(pp.configure(ResampleMode::None, NormaliseMode::LogMAD).has_value());

        auto               input = makeRamp(128);
        std::vector<float> output(128, 0.f);
        pp.normalise(input, output);

        for (const auto v : output) {
            expect(ge(v, -5.f));
            expect(le(v, 10.f));
        }
    };

    "dispatch MinMax mode"_test = [] {
        OnnxPreprocess<float> pp;
        expect(pp.configure(ResampleMode::None, NormaliseMode::MinMax).has_value());

        auto               input = makeRamp(64);
        std::vector<float> output(64, 0.f);
        pp.normalise(input, output);

        for (const auto v : output) {
            expect(ge(v, 0.f));
            expect(le(v, 1.f));
        }
    };

    "dispatch ZScore mode"_test = [] {
        OnnxPreprocess<float> pp;
        expect(pp.configure(ResampleMode::None, NormaliseMode::ZScore).has_value());

        auto               input = makeRamp(128);
        std::vector<float> output(128, 0.f);
        pp.normalise(input, output);

        for (const auto v : output) {
            expect(ge(v, -5.f));
            expect(le(v, 10.f));
        }
    };

    "reconfigure between modes"_test = [] {
        OnnxPreprocess<float> pp;
        auto                  input = makeRamp(64);
        std::vector<float>    output(64, 0.f);

        expect(pp.configure(ResampleMode::None, NormaliseMode::MinMax).has_value());
        pp.normalise(input, output);
        expect(lt(std::abs(output.back() - 1.f), 1e-6f)) << "MinMax max should be 1";

        expect(pp.configure(ResampleMode::None, NormaliseMode::None).has_value());
        pp.normalise(input, output);
        expect(lt(std::abs(output[0] - input[0]), 1e-6f)) << "None should copy input";

        expect(pp.configure(ResampleMode::None, NormaliseMode::LogMAD).has_value());
        pp.normalise(input, output);
        expect(le(output.back(), 10.f)) << "LogMAD should clip";
    };
};

const boost::ut::suite<"OnnxPreprocess resample"> resampleTests = [] {
    "identity resample"_test = [] {
        std::vector<float> input = {1.f, 2.f, 3.f, 4.f, 5.f};
        std::vector<float> output(5);
        OnnxPreprocess<float>::resample(input, output);
        for (std::size_t i = 0; i < input.size(); ++i) {
            expect(lt(std::abs(output[i] - input[i]), 1e-6f));
        }
    };

    "upsample preserves endpoints"_test = [] {
        std::vector<float> input = {0.f, 10.f};
        std::vector<float> output(11);
        OnnxPreprocess<float>::resample(input, output);
        expect(lt(std::abs(output[0] - 0.f), 1e-6f));
        expect(lt(std::abs(output[10] - 10.f), 1e-6f));
        expect(lt(std::abs(output[5] - 5.f), 1e-6f));
    };

    "downsample preserves endpoints"_test = [] {
        std::vector<float> input = {0.f, 1.f, 2.f, 3.f, 4.f};
        std::vector<float> output(3);
        OnnxPreprocess<float>::resample(input, output);
        expect(lt(std::abs(output[0] - 0.f), 1e-6f));
        expect(lt(std::abs(output[2] - 4.f), 1e-6f));
    };

    "empty input/output handled"_test = [] {
        std::vector<float> empty;
        std::vector<float> output(5);
        OnnxPreprocess<float>::resample(empty, output);
        // should not crash
    };
};

const boost::ut::suite<"OnnxPreprocess double"> doubleTests = [] {
    "works with double type"_test = [] {
        OnnxPreprocess<double> pp;
        expect(pp.configure(ResampleMode::None, NormaliseMode::LogMAD).has_value());

        std::vector<double> input(64);
        std::iota(input.begin(), input.end(), 1.0);
        std::vector<double> output(64, 0.0);
        pp.normalise(input, output);

        for (const auto v : output) {
            expect(std::isfinite(v));
            expect(ge(v, -5.0));
            expect(le(v, 10.0));
        }
    };
};

int main() { /* boost::ut */ }
