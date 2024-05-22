#ifndef GNURADIO_BM_TEST_HELPER_HPP
#define GNURADIO_BM_TEST_HELPER_HPP

#include <utility>
#include <variant>

#include <benchmark.hpp>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Tag.hpp>

inline constexpr std::size_t N_MAX = std::numeric_limits<std::size_t>::max();

namespace bm::test {

inline static std::size_t n_samples_produced = 0UZ;

template<typename T, std::size_t min = 0UZ, std::size_t count = N_MAX, bool use_bulk_operation = true>
struct source : public gr::Block<source<T, min, count>> {
    gr::PortOut<T> out;

    gr::Size_t n_samples_max;

    friend constexpr std::size_t
    available_samples(const source &self) noexcept {
        return self.n_samples_max - n_samples_produced;
    }

    [[nodiscard]] constexpr auto
    processOne_simd(auto N) const noexcept -> gr::meta::simdize<T, decltype(N)::value> {
        n_samples_produced += N;
        gr::meta::simdize<T, N> x{};
        benchmark::force_to_memory(x);
        return x;
    }

    [[nodiscard]] constexpr T
    processOne() const noexcept {
        n_samples_produced++;
        T x{};
        benchmark::force_to_memory(x);
        return x;
    }
};

inline static std::size_t n_samples_consumed = 0UZ;

template<typename T, std::size_t N_MIN = 1UZ, std::size_t N_MAX = N_MAX>
struct sink : public gr::Block<sink<T, N_MIN, N_MAX>> {
    gr::PortIn<T, gr::RequiredSamples<N_MIN, N_MAX>> in;
    uint64_t                                         should_receive_n_samples = 0;
    int64_t                                          _last_tag_position       = -1;

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    processOne(V a) noexcept {
        // optional user-level tag processing
        if (this->input_tags_present()) {
            if (this->input_tags_present() && this->mergedInputTag().map.contains("N_SAMPLES_MAX")) {
                const auto value = this->mergedInputTag().map.at("N_SAMPLES_MAX");
                if (std::holds_alternative<uint64_t>(value)) { // should be std::size_t but emscripten/pmtv seem to have issues with it
                    should_receive_n_samples = std::get<uint64_t>(value);
                    _last_tag_position       = in.streamReader().position();
                }
            }
        }

        if constexpr (gr::meta::any_simd<V>) {
            n_samples_consumed += V::size();
        } else {
            n_samples_consumed++;
        }
        benchmark::force_store(a);
    }
};

template<std::size_t N, typename base, typename aggregate>
constexpr auto
cascade(aggregate &&src, std::function<base()> generator = [] { return base(); }) {
    if constexpr (N <= 1) {
        return src;
    } else {
        return cascade<N - 1, base>(gr::mergeByIndex<0, 0>(std::forward<aggregate>(src), generator()), generator);
    }
}

} // namespace bm::test

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, std::size_t min, std::size_t count, bool use_bulk_operation), (bm::test::source<T, min, count, use_bulk_operation>), out, n_samples_max);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, std::size_t N_MIN, std::size_t N_MAX), (bm::test::sink<T, N_MIN, N_MAX>), in, should_receive_n_samples);

#endif // GNURADIO_BM_TEST_HELPER_HPP
