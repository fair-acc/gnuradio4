#ifndef GNURADIO_BOOLEANMATH_H
#define GNURADIO_BOOLEANMATH_H

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::blocks::math {

GR_REGISTER_BLOCK("gr::blocks::math::AndConst", gr::blocks::math::AndConst, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t ])

template<typename T>
struct AndConst : Block<AndConst<T>> {
    using Description = Doc<"(@brief applies bitwise boolean AND of a constant with the input data stream.">;
    // ports
    PortIn<T> in{};
    PortOut<T> out{};
    T constant = static_cast<T>(1);

    GR_MAKE_REFLECTABLE(AndConst, in, out, constant);

    template<meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V processOne(const V &a) const noexcept {
        return static_cast<V>(a & constant);
    }
};

GR_REGISTER_BLOCK("gr::blocks::math::And", gr::blocks::math::And, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t ])

template<typename T>
struct And : Block<And<T>> {
    using Description = Doc<"(@brief applies bitwise boolean AND across multiple input streams.">;
    // ports
    std::vector<PortIn<T>> in;
    PortOut<T> out{};
    
    GR_MAKE_REFLECTABLE(And, in, out);

    gr::work::Status processBulk(std::span<const std::span<const T>> in_spans, std::span<T> out_span) {
        if (in_spans.empty() || out_span.empty()) {
            return gr::work::Status::OK;
        }
        size_t n_items = in_spans.size();
        if (n_items == 0) {
            return gr::work::Status::OK;
        }

        // Mathematical calculation
        size_t num_streams = in_spans[0].size();
        for (size_t i = 0; i < n_items; ++i) {
            // Initialization
            T result = in_spans[i][0];
            for (size_t j = 1; j < num_streams; ++j) {
                result &= in_spans[i][j];
            }
            // Store result in output stream
            out_span[i] = result;
        }

        return gr::work::Status::OK;
    }
};

GR_REGISTER_BLOCK("gr::blocks::math::Not", gr::blocks::math::Not, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t ])

template<typename T>
struct Not : Block<Not<T>> {
    using Description = Doc<"(@brief applies bitwise boolean NOT of an input data stream.">;
    // ports
    PortIn<T> in{};
    PortOut<T> out{};

    GR_MAKE_REFLECTABLE(Not, in, out);

    template<meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V processOne(const V &a) const noexcept {
        return ~a;
    }
};

GR_REGISTER_BLOCK("gr::blocks::math::Or", gr::blocks::math::Or, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t ])

template<typename T>
struct Or : Block<Or<T>> {
    using Description = Doc<"(@brief applies bitwise boolean OR across multiple input streams">;
    // ports
    std::vector<PortIn<T>> in;
    PortOut<T> out{};
    
    GR_MAKE_REFLECTABLE(Or, in, out);

    gr::work::Status processBulk(std::span<const std::span<const T>> in_spans, std::span<T> out_span) {
        if (in_spans.empty() || out_span.empty()) {
            return gr::work::Status::OK;
        }
        size_t n_items = in_spans.size();
        if (n_items == 0) {
            return gr::work::Status::OK;
        }

        // Mathematical calculation
        size_t num_streams = in_spans[0].size();
        for (size_t i = 0; i < n_items; ++i) {
            // Initialization
            T result = in_spans[i][0];
            for (size_t j = 1; j < num_streams; ++j) {
                result |= in_spans[i][j];
            }
            // Store result in output stream
            out_span[i] = result;
        }

        return gr::work::Status::OK;
    }
};

GR_REGISTER_BLOCK("gr::blocks::math::Xor", gr::blocks::math::Xor, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t ])

template<typename T>
struct Xor : Block<Xor<T>> {
    using Description = Doc<"(@brief applies bitwise boolean XOR across multiple input streams">;
    // ports
    std::vector<PortIn<T>> in;
    PortOut<T> out{};
    
    GR_MAKE_REFLECTABLE(Xor, in, out);

    gr::work::Status processBulk(std::span<const std::span<const T>> in_spans, std::span<T> out_span) {
        if (in_spans.empty() || out_span.empty()) {
            return gr::work::Status::OK;
        }
        size_t n_items = in_spans.size();
        if (n_items == 0) {
            return gr::work::Status::OK;
        }

        // Mathematical calculation
        size_t num_streams = in_spans[0].size();
        for (size_t i = 0; i < n_items; ++i) {
            // Initialization
            T result = in_spans[i][0];
            for (size_t j = 1; j < num_streams; ++j) {
                result ^= in_spans[i][j];
            }
            // Store result in output stream
            out_span[i] = result;
        }

        return gr::work::Status::OK;
    }
};

} // namespace gr::blocks::math

#endif // GNURADIO_BOOLEANMATH_H
