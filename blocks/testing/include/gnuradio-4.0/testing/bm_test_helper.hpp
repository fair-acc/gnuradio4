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

    GR_MAKE_REFLECTABLE(source, out, n_samples_max);

    friend constexpr std::size_t available_samples(const source& self) noexcept { return self.n_samples_max - n_samples_produced; }

    [[nodiscard]] constexpr auto processOne_simd(auto N) const noexcept -> vir::simdize<T, decltype(N)::value> {
        n_samples_produced += N;
        vir::simdize<T, N> x{};
        benchmark::force_to_memory(x);
        return x;
    }

    [[nodiscard]] constexpr T processOne() const noexcept {
        n_samples_produced++;
        T x{};
        benchmark::force_to_memory(x);
        return x;
    }
}; // namespace bm::test

inline static std::size_t n_samples_consumed = 0UZ;

template<typename T, std::size_t N_MIN = 1UZ, std::size_t N_MAX = N_MAX>
struct sink : public gr::Block<sink<T, N_MIN, N_MAX>> {
    gr::PortIn<T, gr::RequiredSamples<N_MIN, N_MAX>> in;
    uint64_t                                         should_receive_n_samples = 0;
    std::optional<std::size_t>                       _last_tag_position;

    GR_MAKE_REFLECTABLE(sink, in, should_receive_n_samples);

    [[nodiscard]] constexpr auto processOne(T a) noexcept {
        // optional user-level tag processing
        if (this->inputTagsPresent()) {
            if (this->inputTagsPresent() && this->mergedInputTag().map.contains("N_SAMPLES_MAX")) {
                const auto value = this->mergedInputTag().map.at("N_SAMPLES_MAX");
                if (value.template holds<uint64_t>()) { // should be std::size_t but emscripten/pmtv seem to have issues with it
                    if (auto ptr = value.template get_if<std::uint64_t>()) {
                        should_receive_n_samples = *ptr;
                        _last_tag_position       = in.streamReader().position();
                    }
                }
            }
        }

        n_samples_consumed++;
        benchmark::force_store(a);
    }
};

template<std::size_t N, typename Base, typename Aggregate>
struct CascadeTypeHelper;

template<typename Base, typename Aggregate>
struct CascadeTypeHelper<0, Base, Aggregate> {
    using type = Base;
};

template<std::size_t N, typename Base, typename Aggregate>
struct CascadeTypeHelper {
    using type = gr::MergeByIndex<typename CascadeTypeHelper<N - 1, Base, Aggregate>::type, 0, Base, 0>;
};

template<std::size_t N, typename Base>
using CascadeType = typename CascadeTypeHelper<N, Base, Base>::type;

} // namespace bm::test

#endif // GNURADIO_BM_TEST_HELPER_HPP
