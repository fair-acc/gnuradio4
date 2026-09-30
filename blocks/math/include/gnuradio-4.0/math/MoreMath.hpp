#ifndef GNURADIO_MOREMATH_H
#define GNURADIO_MOREMATH_H

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::blocks::math {

GR_REGISTER_BLOCK("gr::blocks::math::Argmax", gr::blocks::math::Argmax, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])

template<typename T>
struct Argmax : Block<Argmax<T>> {
    using Description = Doc<"(@brief outputs the input stream number and the corresponding index, where an overall maximum value occurred (first).">;
    // ports
    std::vector<PortIn<T>> in;
    PortOut<Size_t>        out;

    GR_MAKE_REFLECTABLE(Argmax, in, out);

    gr::work::Status processBulk(std::span<const std::span<const T>> in_spans, std::span<Size_t> out_span) {
        if (in_spans.empty() || out_span.empty()) {
            return gr::work::Status::OK;
        }
        size_t n_items = in_spans.size();
        if (n_items == 0) {
            return gr::work::Status::OK;
        }

        // Mathematical calculation
        size_t num_streams = in_spans[0].size();
        // Initialization
        T max_val = in_spans[0][0];
        Size_t index = 0;
        Size_t stream = 0;
        // Search for first encounter of maximum value for all input streams
        for (Size_t i = 0; i < n_items; ++i) {
            for (Size_t j = 0; j < num_streams; ++j) {
                if (in_spans[i][j] > max_val) {
                    max_val = std::max(max_val, in_spans[i][j]);
                    index = i;
                    stream = j;
                }
            }
        }
        // Store result in output stream
        out_span[0] = index;
        out_span[1] = stream;

        return gr::work::Status::OK;
    }
};

GR_REGISTER_BLOCK("gr::blocks::math::Max", gr::blocks::math::Max, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])

template<typename T>
struct Max : Block<Max<T>> {
    using Description = Doc<"(@brief calculate maximum envelope across all input streams.">;
    // ports
    std::vector<PortIn<T>> in;
    PortOut<T>             out;

    GR_MAKE_REFLECTABLE(Max, in, out);

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
            // First value of i-th sample
            T max_val = in_spans[i][0];
            // Search for maximum value across all input streams
            for (size_t j = 1; j < num_streams; ++j) {
                max_val = std::max(max_val, in_spans[i][j]);
            }
            // Store result in output stream
            out_span[i] = max_val;
        }

        return gr::work::Status::OK;
    }
};

GR_REGISTER_BLOCK("gr::blocks::math::Min", gr::blocks::math::Min, [T], [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])

template<typename T>
struct Min : Block<Min<T>> {
    using Description = Doc<"(@brief calculate minimum envelope across all input streams.">;
    // ports
    std::vector<PortIn<T>> in;
    PortOut<T>             out;

    GR_MAKE_REFLECTABLE(Min, in, out);

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
            // First value of i-th sample
            T min_val = in_spans[i][0];
            // Search for maximum value across all input streams
            for (size_t j = 1; j < num_streams; ++j) {
                min_val = std::min(min_val, in_spans[i][j]);
            }
            // Store result in output stream
            out_span[i] = min_val;
        }

        return gr::work::Status::OK;
    }
};

GR_REGISTER_BLOCK("gr::blocks::math::Log10", gr::blocks::math::Log10, [T], [ float ])

template<typename T>
struct Log10 : Block<Log10<T>> {
    using Description = Doc<"(@brief calculate n * log10(a) + k for an input stream.">;
    PortIn<T> in;
    PortOut<T> out;
    T n = static_cast<T>(1);
    T k = static_cast<T>(0);

    GR_MAKE_REFLECTABLE(Log10, in, out, n, k);

    gr::work::Status processBulk(std::span<const T> in_span, std::span<T> out_span) {
        if (in_span.empty() || out_span.empty()) {
            return gr::work::Status::OK;
        }
        size_t n_items = in_span.size();
        if (n_items == 0) {
            return gr::work::Status::OK;
        }

        // Mathematical calculation
        for (size_t i = 0; i < n_items; ++i) {
            // Store result in output stream
            out_span[i] = n * std::log10(in_span[i]) + k;
        }

        return gr::work::Status::OK;
    }
};

GR_REGISTER_BLOCK("gr::blocks::math::RMS", gr::blocks::math::RMS, [T], [ float, double ])

template<typename T>
struct RMS : Block<RMS<T>> {
    using Description = Doc<"(@brief calculate the root mean square value of an input stream.">;
    PortIn<T> in;
    PortOut<T> out;

    GR_MAKE_REFLECTABLE(RMS, in, out);

    gr::work::Status processBulk(std::span<const T> in_span, std::span<T> out_span) {
        if (in_span.empty() || out_span.empty()) {
            return gr::work::Status::OK;
        }
        size_t n_items = in_span.size();
        if (n_items == 0) {
            return gr::work::Status::OK;
        }

        // Mathematical calculation
        T sum_of_squares = std::ranges::fold_left(
            in_span,
            0.0,
            [](T acc, T val) { return acc + (val * val); }
        );
        
        // Store result in output stream
        out_span[0] = std::sqrt(sum_of_squares / static_cast<T>(n_items));

        return gr::work::Status::OK;
    }
};

} // namespace gr::blocks::math

#endif // GNURADIO_MOREMATH_H
