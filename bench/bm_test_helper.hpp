#ifndef GRAPH_PROTOTYPE_BM_TEST_HELPER_HPP
#define GRAPH_PROTOTYPE_BM_TEST_HELPER_HPP

#include <graph.hpp>

inline constexpr std::size_t N_MAX = std::numeric_limits<std::size_t>::max();

namespace test {
namespace fg                                 = fair::graph;
inline static std::size_t n_samples_produced = 0LU;

template<typename T, std::size_t min = 0, std::size_t count = N_MAX, bool use_bulk_operation = true>
class source : public fg::node<source<T, min, count>, fg::OUT<T, "out">, fg::limits<min, count>> {
    std::size_t _n_samples_max;

public:
    source() = delete;

    source(std::size_t n_samples) : _n_samples_max(n_samples) {}

    [[nodiscard]] constexpr T
    process_one() const noexcept {
        n_samples_produced++;
        return T{};
    }

    fair::graph::work_result
    work() {
        const std::size_t n_to_publish = _n_samples_max - n_samples_produced;
        if (n_to_publish > 0) {
            auto &port   = output_port<"out">(this);
            auto &writer = port.writer();

            if constexpr (use_bulk_operation) {
                std::size_t n_write = std::clamp(n_to_publish, 0UL, std::min(writer.available(), port.max_buffer_size()));
                if (n_write == 0) {
                    return fair::graph::work_result::has_unprocessed_data;
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
                if (data.size() == 0) {
                    return fair::graph::work_result::error;
                }
                data[0] = process_one();
                writer.publish(token, 1);
            }
            return fair::graph::work_result::success;
        } else {
            return fair::graph::work_result::inputs_empty;
        }
    }
};

inline static std::size_t n_samples_consumed = 0LU;

template<typename T, std::size_t N_MIN = 0, std::size_t N_MAX = N_MAX>
class sink : public fg::node<sink<T, N_MIN, N_MAX>, fg::IN<T, "in", N_MIN, N_MAX>, fg::limits<N_MIN, N_MAX>> {
public:
    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(V a) const noexcept {
        n_samples_consumed++;
        benchmark::do_not_optimize(a);
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

#endif // GRAPH_PROTOTYPE_BM_TEST_HELPER_HPP
