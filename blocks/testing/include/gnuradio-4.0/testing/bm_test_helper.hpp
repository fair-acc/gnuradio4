#ifndef GNURADIO_BM_TEST_HELPER_HPP
#define GNURADIO_BM_TEST_HELPER_HPP

#include <utility>
#include <variant>

#include <benchmark.hpp>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Tag.hpp>

inline constexpr std::size_t N_MAX = std::numeric_limits<std::size_t>::max();

namespace test {

inline static std::size_t n_samples_produced = 0UZ;

template<typename T, std::size_t min = 0UZ, std::size_t count = N_MAX, bool use_bulk_operation = true>
class source : public gr::Block<source<T, min, count>> {
public:
    uint64_t                   _n_samples_max;
    gr::Tag::signed_index_type _n_tag_offset;
    gr::PortOut<T>             out;

    source() = delete;

    source(std::size_t n_samples, gr::Tag::signed_index_type n_tag_offset = 100) : _n_samples_max(n_samples), _n_tag_offset(n_tag_offset) {}

    friend constexpr std::size_t
    available_samples(const source &self) noexcept {
        return self._n_samples_max - n_samples_produced;
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

    gr::work::Result
    work(std::size_t requested_work) {
        const std::size_t n_to_publish = _n_samples_max - n_samples_produced;
        if (n_to_publish > 0) {
            auto &port   = out;
            auto &writer = port.streamWriter();
            if (n_samples_produced == 0) {
                port.publishTag({ { "N_SAMPLES_MAX", _n_samples_max } }, static_cast<gr::Tag::signed_index_type>(_n_tag_offset)); // shorter version
            }

            if constexpr (use_bulk_operation) {
                std::size_t n_write = std::clamp(n_to_publish, 0UL, std::min(writer.available(), port.max_buffer_size()));
                if (n_write == 0UZ) {
                    return { requested_work, 0UZ, gr::work::Status::INSUFFICIENT_INPUT_ITEMS };
                }

                writer.publish( //
                        [this](std::span<T> output) {
                            for (auto &val : output) {
                                val = processOne();
                            }
                        },
                        n_write);
            } else {
                auto [data, token] = writer.get(1);
                if (data.size() == 0UZ) {
                    return { requested_work, 0UZ, gr::work::Status::ERROR };
                }
                data[0] = processOne();
                writer.publish(token, 1);
            }
            return { requested_work, 1UZ, gr::work::Status::OK };
        } else {
            return { requested_work, 0UZ, gr::work::Status::DONE };
        }
    }
};

inline static std::size_t n_samples_consumed = 0UZ;

template<typename T, std::size_t N_MIN = 1UZ, std::size_t N_MAX = N_MAX>
struct sink : public gr::Block<sink<T, N_MIN, N_MAX>> {
    gr::PortIn<T, gr::RequiredSamples<N_MIN, N_MAX>> in;
    std::size_t                                      should_receive_n_samples = 0;
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
cascade(
        aggregate &&src, std::function<base()> generator = [] { return base(); }) {
    if constexpr (N <= 1) {
        return src;
    } else {
        return cascade<N - 1, base>(gr::mergeByIndex<0, 0>(std::forward<aggregate>(src), generator()), generator);
    }
}

} // namespace test

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, std::size_t min, std::size_t count, bool use_bulk_operation), (test::source<T, min, count, use_bulk_operation>), out);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, std::size_t N_MIN, std::size_t N_MAX), (test::sink<T, N_MIN, N_MAX>), in);

#endif // GNURADIO_BM_TEST_HELPER_HPP
