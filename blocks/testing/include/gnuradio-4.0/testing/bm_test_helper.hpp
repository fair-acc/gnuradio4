#ifndef GNURADIO_BM_TEST_HELPER_HPP
#define GNURADIO_BM_TEST_HELPER_HPP

#include <utility>
#include <variant>

#include <benchmark.hpp>

#include <gnuradio-4.0/BlockMerging.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Tag.hpp>

inline constexpr std::size_t N_MAX = std::numeric_limits<std::size_t>::max();

namespace bm::test {

inline static std::size_t n_samples_produced = 0UZ;

template<typename T, std::size_t min = 0UZ, std::size_t count = N_MAX, bool use_bulk_operation = true>
struct source : public gr::Block<source<T, min, count>> {
    gr::PortOut<T> out;

    gr::Size_t n_samples_max = 0U;

    GR_MAKE_REFLECTABLE(source, out, n_samples_max);

    void reset() { n_samples_produced = 0UZ; }

    [[nodiscard]] constexpr T processOne() noexcept {
        n_samples_produced++;
        if (n_samples_max > 0 && n_samples_produced >= n_samples_max) {
            this->requestStop();
        }
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
    std::optional<std::size_t>                       _last_tag_position;

    GR_MAKE_REFLECTABLE(sink, in, should_receive_n_samples);

    void reset() { n_samples_consumed = 0UZ; }

    [[nodiscard]] constexpr auto processOne(T a) noexcept {
        if (this->inputTagsPresent()) {
            const auto& tag = this->mergedInputTag();
            if (auto it = tag.map.find("N_SAMPLES_MAX"); it != tag.map.end()) {
                if (auto ptr = it->second.template get_if<std::uint64_t>()) {
                    should_receive_n_samples = *ptr;
                    _last_tag_position       = in.streamReader().position();
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
