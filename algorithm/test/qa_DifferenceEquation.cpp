#include <boost/ut.hpp>

#include <array>
#include <cmath>
#include <format>
#include <span>
#include <vector>

#include <gnuradio-4.0/algorithm/filter/DifferenceEquation.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

namespace {
using gr::algorithm::filter::Fir;
using gr::algorithm::filter::Iir;
using gr::algorithm::filter::IIRForm;

[[nodiscard]] std::vector<float> ramp(std::size_t n, float scale = 1.f) {
    std::vector<float> v(n);
    for (std::size_t i = 0UZ; i < n; ++i) {
        v[i] = scale * static_cast<float>(i + 1UZ);
    }
    return v;
}

/// the definition, written out: y[n] = Σ b[k] x[n-k] over the window ending at the newest sample
[[nodiscard]] std::vector<float> convolveByDefinition(std::span<const float> input, std::span<const float> taps) {
    const std::size_t nTaps = std::max(std::size_t{1}, taps.size());
    if (input.size() + 1UZ < nTaps) {
        return {};
    }
    std::vector<float> out(input.size() + 1UZ - nTaps, 0.f);
    for (std::size_t n = 0UZ; n < out.size(); ++n) {
        float acc = 0.f;
        for (std::size_t k = 0UZ; k < taps.size(); ++k) {
            acc += taps[k] * input[n + taps.size() - 1UZ - k];
        }
        out[n] = acc;
    }
    return out;
}

/// the difference equation itself, from explicit input and output histories -- independent of anything in
/// DifferenceEquation.hpp, so every form has something real to be checked against. All four forms realise this.
[[nodiscard]] std::vector<float> recurseByDefinition(std::span<const float> input, std::span<const float> b, std::span<const float> a) {
    std::vector<float> x(b.size(), 0.f);                              // newest first
    std::vector<float> y(a.size() > 0UZ ? a.size() - 1UZ : 0UZ, 0.f); // newest first
    std::vector<float> out(input.size());
    for (std::size_t n = 0UZ; n < input.size(); ++n) {
        for (std::size_t k = x.size(); k-- > 1UZ;) {
            x[k] = x[k - 1UZ];
        }
        if (!x.empty()) {
            x[0] = input[n];
        }
        float acc = 0.f;
        for (std::size_t k = 0UZ; k < x.size(); ++k) {
            acc += b[k] * x[k];
        }
        for (std::size_t k = 0UZ; k < y.size(); ++k) {
            acc -= a[k + 1UZ] * y[k];
        }
        for (std::size_t k = y.size(); k-- > 1UZ;) {
            y[k] = y[k - 1UZ];
        }
        if (!y.empty()) {
            y[0] = acc;
        }
        out[n] = acc;
    }
    return out;
}

[[nodiscard]] float worstDeparture(std::span<const float> got, std::span<const float> want) {
    float worst = 0.f;
    for (std::size_t i = 0UZ; i < std::min(got.size(), want.size()); ++i) {
        worst = std::max(worst, std::abs(got[i] - want[i]));
    }
    return worst;
}
} // namespace

const boost::ut::suite<"DifferenceEquation"> _differenceEquation = [] {
    using namespace boost::ut;

    "a convolution over a span answers what the definition answers"_test = [] {
        // 63, 64 and 65 straddle the batch, so the batched body, its boundary and its tail are all exercised
        for (const std::size_t nTaps : {1UZ, 2UZ, 3UZ, 17UZ, 63UZ, 64UZ, 65UZ, 200UZ}) {
            const std::vector<float> taps  = ramp(nTaps, 1.f / static_cast<float>(nTaps));
            const std::vector<float> input = ramp(nTaps + 257UZ);
            const std::vector<float> want  = convolveByDefinition(input, taps);

            std::vector<float> got(want.size(), 0.f);
            Fir<float>::convolve(input, taps, got);

            const float scale = 1.f + static_cast<float>(input.size());
            expect(lt(worstDeparture(got, want), 1e-3f * scale)) << std::format("{} taps: departs by {}", nTaps, worstDeparture(got, want));
        }
    };

    "a convolution given no room emits nothing"_test = [] {
        const std::vector<float> taps  = ramp(8UZ);
        const std::vector<float> input = ramp(4UZ); // fewer samples than taps: no window fits
        std::vector<float>       got(4UZ, -1.f);
        Fir<float>::convolve(input, taps, got);
        expect(eq(got[0], -1.f)) << "nothing may be written when no window fits";
    };

    "one output agrees with the batched body"_test = [] {
        const std::vector<float> taps  = ramp(5UZ, 0.2f);
        const std::vector<float> input = ramp(80UZ);
        std::vector<float>       batched(input.size() + 1UZ - taps.size(), 0.f);
        Fir<float>::convolve(input, taps, batched);

        bool agree = true;
        for (std::size_t n = 0UZ; n < batched.size(); ++n) {
            agree = agree && std::abs(Fir<float>::sampleAt(std::span<const float>{input}.subspan(n, taps.size()), taps) - batched[n]) < 1e-3f;
        }
        expect(agree) << "the single-output form and the batched form must compute the same convolution";
    };

    // a streaming convolution keeps its own lead-in, so a span split anywhere must answer as one pass does
    "a streaming convolution carries its lead-in across calls"_test = [] {
        for (const std::size_t nTaps : {1UZ, 2UZ, 5UZ, 64UZ, 70UZ}) {
            const std::vector<float> taps  = ramp(nTaps, 1.f / static_cast<float>(nTaps));
            const std::vector<float> input = ramp(400UZ, 0.01f);

            std::vector<float> history(nTaps > 0UZ ? nTaps - 1UZ : 0UZ, 0.f);
            std::vector<float> whole(input.size(), 0.f);
            Fir<float>::convolveStreaming(input, taps, history, whole);

            std::ranges::fill(history, 0.f);
            std::vector<float>    split(input.size(), 0.f);
            constexpr std::size_t kCut = 137UZ; // a seam in no particular relation to the taps or the batch
            Fir<float>::convolveStreaming(std::span<const float>{input}.first(kCut), taps, history, std::span<float>{split}.first(kCut));
            Fir<float>::convolveStreaming(std::span<const float>{input}.subspan(kCut), taps, history, std::span<float>{split}.subspan(kCut));

            expect(lt(worstDeparture(split, whole), 1e-4f)) << std::format("{} taps: a seam at {} changed the answer by {}", nTaps, kCut, worstDeparture(split, whole));

            // and, from a zeroed lead-in, it must agree with the definition wherever the window is wholly inside
            const std::vector<float> want  = convolveByDefinition(input, taps);
            bool                     agree = true;
            for (std::size_t n = 0UZ; n < want.size(); ++n) {
                agree = agree && std::abs(whole[n + nTaps - 1UZ] - want[n]) < 1e-3f * (1.f + static_cast<float>(input.size()));
            }
            expect(agree) << std::format("{} taps: the settled part departs from the definition", nTaps);
        }
    };

    "every form realises the same difference equation"_test = [] {
        const std::vector<float> b{0.2929f, 0.5858f, 0.2929f}; // a second-order section, coefficients of that shape
        const std::vector<float> a{1.0f, -0.0000f, 0.1716f};
        const std::vector<float> input = ramp(512UZ, 0.01f);

        const std::vector<float> want = recurseByDefinition(input, b, a);

        const auto check = [&]<IIRForm form>(std::string_view name) {
            std::vector<float> state(Iir<float, form>::stateSize(b.size(), a.size() - 1UZ), 0.f);
            std::vector<float> got(input.size(), 0.f);
            Iir<float, form>::filter(input, b, a, state, got);
            expect(lt(worstDeparture(got, want), 1e-4f)) << std::format("{}: departs by {} from the difference equation", name, worstDeparture(got, want));
        };
        check.template operator()<IIRForm::DF_I>("DF_I");
        check.template operator()<IIRForm::DF_II>("DF_II");
        check.template operator()<IIRForm::DF_I_TRANSPOSED>("DF_I_TRANSPOSED");
        check.template operator()<IIRForm::DF_II_TRANSPOSED>("DF_II_TRANSPOSED");
    };

    "the state carries across calls, so a split span answers as one pass does"_test = [] {
        const std::vector<float> b{0.3f, 0.4f, 0.3f};
        const std::vector<float> a{1.0f, 0.2f, 0.05f};
        const std::vector<float> input = ramp(400UZ, 0.01f);

        std::vector<float> state(Iir<float>::stateSize(b.size(), a.size() - 1UZ), 0.f);
        std::vector<float> whole(input.size(), 0.f);
        Iir<float>::filter(input, b, a, state, whole);

        std::ranges::fill(state, 0.f);
        std::vector<float>    split(input.size(), 0.f);
        constexpr std::size_t kCut = 137UZ; // not a multiple of anything, so the seam is where it hurts
        Iir<float>::filter(std::span<const float>{input}.first(kCut), b, a, state, std::span<float>{split}.first(kCut));
        Iir<float>::filter(std::span<const float>{input}.subspan(kCut), b, a, state, std::span<float>{split}.subspan(kCut));

        expect(lt(worstDeparture(split, whole), 1e-5f)) << std::format("a seam at {} changed the answer by {}", kCut, worstDeparture(split, whole));
    };

    // the pass lifts state into locals only while it fits; past that it must run in the caller's span rather than
    // silently filter with a truncated history
    "state too long for locals is still filtered correctly"_test = [] {
        constexpr std::size_t kOrder = Iir<float>::kMaxLocalState + 5UZ;
        std::vector<float>    b(kOrder + 1UZ, 0.f);
        std::vector<float>    a(kOrder + 1UZ, 0.f);
        b[0]                           = 0.5f;
        b[kOrder]                      = 0.5f; // a comb: the oldest tap must be read, so a truncated history shows up
        a[0]                           = 1.0f;
        const std::vector<float> input = ramp(600UZ, 0.01f);

        expect(gt(Iir<float>::stateSize(b.size(), a.size() - 1UZ) / Iir<float>::kHistories, Iir<float>::kMaxLocalState)) << "the case must actually exceed the local bound";

        const std::vector<float> want = recurseByDefinition(input, b, a);
        std::vector<float>       state(Iir<float>::stateSize(b.size(), a.size() - 1UZ), 0.f);
        std::vector<float>       got(input.size(), 0.f);
        Iir<float>::filter(input, b, a, state, got);
        expect(lt(worstDeparture(got, want), 1e-4f)) << std::format("the in-place path departs by {}", worstDeparture(got, want));
    };

    "a bare gain has no state to keep"_test = [] {
        const std::vector<float> b{2.f};
        const std::vector<float> a{1.f};
        const std::vector<float> input = ramp(32UZ);
        std::vector<float>       state(Iir<float>::stateSize(b.size(), a.size() - 1UZ), 0.f);
        std::vector<float>       got(input.size(), 0.f);
        Iir<float>::filter(input, b, a, state, got);
        expect(lt(std::abs(got[10] - 2.f * input[10]), 1e-5f)) << "one feed-forward coefficient is a gain";
    };
};

int main() { return boost::ut::cfg<boost::ut::override>.run({.report_errors = true}); }
