#ifndef GRAPH_PROTOTYPE_BM_TEST_HELPER_HPP
#define GRAPH_PROTOTYPE_BM_TEST_HELPER_HPP

#include <graph.hpp>

inline constexpr std::size_t N_MAX = std::numeric_limits<std::size_t>::max();

namespace test {

namespace fg = fair::graph;
using namespace fair::literals;

inline static std::size_t n_samples_produced = 0_UZ;

template<typename T, std::size_t min = 0_UZ, std::size_t count = N_MAX, bool use_bulk_operation = true>
class source : public fg::node<source<T, min, count>> {
public:
    std::size_t _n_samples_max;
    fg::OUT<T> out;

    source() = delete;

    source(std::size_t n_samples) : _n_samples_max(n_samples) {}

    friend constexpr std::size_t
    available_samples(const source &self) noexcept {
        return self._n_samples_max - n_samples_produced;
    }

    [[nodiscard]] constexpr auto
    process_one_simd(auto N) const noexcept -> fair::meta::simdize<T, decltype(N)::value> {
        n_samples_produced += N;
        fair::meta::simdize<T, N> x {};
        benchmark::force_to_memory(x);
        return x;
    }

    [[nodiscard]] constexpr T
    process_one() const noexcept {
        n_samples_produced++;
        T x{};
        benchmark::force_to_memory(x);
        return x;
    }

    fair::graph::work_return_t
    work() {
        const std::size_t n_to_publish = _n_samples_max - n_samples_produced;
        if (n_to_publish > 0) {
            auto &port   = out;
            auto &writer = port.streamWriter();

            if constexpr (use_bulk_operation) {
                std::size_t n_write = std::clamp(n_to_publish, 0UL, std::min(writer.available(), port.max_buffer_size()));
                if (n_write == 0_UZ) {
                    return fair::graph::work_return_t::INSUFFICIENT_INPUT_ITEMS;
                }

                writer.publish( //
                        [this](std::span<T> output) {
                            for (auto &val : output) {
                                val = process_one();
                            }
                        },
                        n_write);
            } else {
                auto [data, token] = writer.get(1);
                if (data.size() == 0_UZ) {
                    return fair::graph::work_return_t::ERROR;
                }
                data[0] = process_one();
                writer.publish(token, 1);
            }
            return fair::graph::work_return_t::OK;
        } else {
            return fair::graph::work_return_t::DONE;
        }
    }
};

inline static std::size_t n_samples_consumed = 0_UZ;

template<typename T, std::size_t N_MIN = 0_UZ, std::size_t N_MAX = N_MAX>
class sink : public fg::node<sink<T, N_MIN, N_MAX>> {
public:
    fg::IN<T, N_MIN, N_MAX> in;

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(V a) const noexcept {
        if constexpr (fair::meta::any_simd<V>) {
            n_samples_consumed += V::size();
        } else {
            n_samples_consumed++;
        }
        benchmark::force_store(a);
    }
};

template<std::size_t N, typename base, typename aggregate>
constexpr auto
cascade(
        aggregate &&src, std::function<base()> generator = [] { return base(); }) {
    if constexpr (N <= 1) {
        return src;
    } else {
        return cascade<N - 1, base>(fair::graph::merge_by_index<0, 0>(std::forward<aggregate>(src), generator()), generator);
    }
}

} // namespace test

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, std::size_t min, std::size_t count, bool use_bulk_operation), (test::source<T, min, count, use_bulk_operation>), out);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, std::size_t N_MIN, std::size_t N_MAX), (test::sink<T, N_MIN, N_MAX>), in);

#endif // GRAPH_PROTOTYPE_BM_TEST_HELPER_HPP
