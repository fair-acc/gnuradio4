#ifndef GNURADIO_BYTEMATH_H
#define GNURADIO_BYTEMATH_H

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::blocks::math {

namespace detail {
template<typename T>
constexpr T bit_width_v = sizeof(T) * CHAR_BIT;
} // namespace detail

GR_REGISTER_BLOCK("gr::blocks::math::PackKBits", gr::blocks::math::PackKBits, [T], [ uint8_t, uint16_t, uint32_t, uint64_t ])

template<typename T>
struct PackKBits : Block<PackKBits<T>> {
    using Description = Doc<"(@brief convert an input data stream of low bits into an output data stream of k packed bits.">;
    PortIn<T> in;
    PortOut<T> out;
    Size_t k = 1;

    GR_MAKE_REFLECTABLE(PackKBits, in, out, k);

    gr::work::Status processBulk(std::span<const T> in_span, std::span<T> out_span) {
        if (in_span.empty() || out_span.empty()) {
            return gr::work::Status::OK;
        }
        auto n_items = static_cast<Size_t>(in_span.size());
        if (n_items == 0) {
            return gr::work::Status::OK;
        }

        Size_t no_of_bits = k;
        Size_t rest = n_items % no_of_bits;
        if (rest > static_cast<Size_t>(0)) {
            // n_items is not an integral multiple
            // of bit_width_v<T>
            return gr::work::Status::OK;
        }

        // Mathematical calculation
        Size_t count = 0;
        Size_t bit_count = 1;
        T shift = 0;
        T part = 0;
        T result = 0;
        for (Size_t i = 0; i < n_items; ++i) {
            shift = static_cast<T>(no_of_bits - bit_count);
            part = static_cast<T>((in_span[i] & 1) << shift);
            result |= part;
            ++bit_count;
            if ((i + 1) % no_of_bits == 0) {
                out_span[count] = result;
                ++count;
                result = 0;
                bit_count = 1;
            }
        }

        return gr::work::Status::OK;
    }
};

GR_REGISTER_BLOCK("gr::blocks::math::PackedToUnpacked", gr::blocks::math::PackedToUnpacked, [T], [ uint8_t, uint16_t, uint32_t, uint64_t ])

template<typename T>
struct PackedToUnpacked : Block<PackedToUnpacked<T>> {
    using Description = Doc<"(@brief convert an input data stream of packed bits into an output data stream of low bits.">;
    PortIn<T> in;
    PortOut<T> out;

    GR_MAKE_REFLECTABLE(PackedToUnpacked, in, out);

    gr::work::Status processBulk(std::span<const T> in_span, std::span<T> out_span) {
        if (in_span.empty() || out_span.empty()) {
            return gr::work::Status::OK;
        }
        auto n_items = static_cast<Size_t>(in_span.size());
        if (n_items == 0) {
            return gr::work::Status::OK;
        }

        constexpr Size_t no_of_bits = detail::bit_width_v<T>;
        auto m_items = static_cast<Size_t>(out_span.size());
        if (m_items != n_items * no_of_bits) {
            // m_items has an improper size
            return gr::work::Status::OK;
        }

        // Mathematical calculation
        Size_t count = 0;
        for (Size_t i = 0; i < n_items; ++i) {
            std::bitset<no_of_bits> in_value{in_span[i]};
            for (Size_t j = 0; j < no_of_bits; ++j) {
                Size_t idx1 = no_of_bits - 1 - j;
                T result = in_value.test(idx1) ? 1 : 0;
                Size_t idx2 = count * no_of_bits + j;
                out_span[idx2] = result;
            }
            ++count;
        }

        return gr::work::Status::OK;
    }
};

GR_REGISTER_BLOCK("gr::blocks::math::RepackBits", gr::blocks::math::RepackBits, [T], [ uint8_t, uint16_t, uint32_t, uint64_t ])

template<typename T>
struct RepackBits : Block<RepackBits<T>> {
    using Description = Doc<"(@brief repack l bits from the input stream with k bits onto bits of the output stream.">;
    PortIn<T> in{};
    PortOut<T> out{};
    Size_t k = 8;
    Size_t l = 8;
    std::string len_tag_key;
    bool align_output = false;
    std::endian endianness = std::endian::little;

    GR_MAKE_REFLECTABLE(RepackBits, in, out, k, l, len_tag_key, align_output, endianness);

    gr::work::Status processBulk(std::span<const T> in_span, std::span<T> out_span) {
        if (in_span.empty() || out_span.empty()) {
            return gr::work::Status::OK;
        }
        if (k < 1 || k > 8 || l < 1 || l > 8) {
            return gr::work::Status::OK;
        }
        bool packet_mode(!len_tag_key.empty());
        auto n_items = static_cast<Size_t>(in_span.size());
        auto m_items = static_cast<Size_t>(out_span.size());
        const Size_t total_bits = n_items * k;
        Size_t required_bytes = total_bits / l;
        if ((total_bits % l != 0 ) && (!packet_mode || (packet_mode && !align_output))) {
            ++required_bytes;
        }
        if (required_bytes != m_items) {
            return gr::work::Status::OK;
        }

        // Mathematical calculation
        Size_t n_read = 0;
        Size_t n_written = 0;
        Size_t in_index = 0;
        Size_t out_index = 0;
        T shift1 = 0;
        T shift2 = 0;
        switch (endianness) {
        case std::endian::little:
            while (n_written < m_items && n_read < n_items) {
                if (out_index == 0) { // Starting a fresh byte
                    out_span[n_written] = 0;
                }
                shift1 = in_span[n_read] >> in_index;
                shift2 = static_cast<Size_t>(shift1 & 0x01) << out_index;
                out_span[n_written] |= shift2;

                in_index = (in_index + 1) % k;
                out_index = (out_index + 1) % l;
                if (in_index == 0) {
                    n_read++;
                    in_index = 0;
                }
                if (out_index == 0) {
                    n_written++;
                    out_index = 0;
                }
            }
            if (packet_mode) {
                if (out_index) {
                    n_written++;
                    out_index = 0;
                }
            }
            break;
        case std::endian::big:
            while (n_written < m_items && n_read < n_items) {
                if (out_index == 0) { // Starting a fresh byte
                    out_span[n_written] = 0;
                }
                shift1 = in_span[n_read] >> (k - 1 - in_index);
                shift2 = static_cast<Size_t>(shift1 & 0x01) << (l - 1 - out_index);
                out_span[n_written] |= shift2;

                in_index = (in_index + 1) % k;
                out_index = (out_index + 1) % l;
                if (in_index == 0) {
                    n_read++;
                    in_index = 0;
                }
                if (out_index == 0) {
                    n_written++;
                    out_index = 0;
                }
            }
            if (packet_mode) {
                if (out_index) {
                    n_written++;
                    out_index = 0;
                }
            }
            break;
        default:
            return gr::work::Status::OK;
        }

        return gr::work::Status::OK;
    }
};

GR_REGISTER_BLOCK("gr::blocks::math::UnpackKBits", gr::blocks::math::UnpackKBits, [T], [ uint8_t, uint16_t, uint32_t, uint64_t ])

template<typename T>
struct UnpackKBits : Block<UnpackKBits<T>> {
    using Description = Doc<"(@brief convert an input data stream of k packed bits into an output data stream of low bits.">;
    PortIn<T> in;
    PortOut<T> out;
    Size_t k = 1;

    GR_MAKE_REFLECTABLE(UnpackKBits, in, out, k);

    gr::work::Status processBulk(std::span<const T> in_span, std::span<T> out_span) {
        if (in_span.empty() || out_span.empty()) {
            return gr::work::Status::OK;
        }
        auto n_items = static_cast<Size_t>(in_span.size());
        if (n_items == 0) {
            return gr::work::Status::OK;
        }

        constexpr Size_t no_of_bits = detail::bit_width_v<T>;
        auto m_items = static_cast<Size_t>(out_span.size());
        if (m_items != n_items * k) {
            // m_items has an improper size
            return gr::work::Status::OK;
        }

        // Mathematical calculation
        Size_t count = 0;
        Size_t offset = no_of_bits - k;
        for (Size_t i = 0; i < n_items; ++i) {
            std::bitset<no_of_bits> in_value{in_span[i]};
            for (Size_t j = offset; j < no_of_bits; ++j) {
                Size_t idx1 = no_of_bits - 1 - j;
                T result = in_value.test(idx1) ? 1 : 0;
                Size_t idx2 = count * k + j - offset;
                out_span[idx2] = result;
            }
            ++count;
        }

        return gr::work::Status::OK;
    }
};

GR_REGISTER_BLOCK("gr::blocks::math::UnpackedToPacked", gr::blocks::math::UnpackedToPacked, [T], [ uint8_t, uint16_t, uint32_t, uint64_t ])

template<typename T>
struct UnpackedToPacked : Block<UnpackedToPacked<T>> {
    using Description = Doc<"(@brief convert an input data stream of low bits into an output data stream of packed bits.">;
    PortIn<T> in;
    PortOut<T> out;

    GR_MAKE_REFLECTABLE(UnpackedToPacked, in, out);

    gr::work::Status processBulk(std::span<const T> in_span, std::span<T> out_span) {
        if (in_span.empty() || out_span.empty()) {
            return gr::work::Status::OK;
        }
        auto n_items = static_cast<Size_t>(in_span.size());
        if (n_items == 0) {
            return gr::work::Status::OK;
        }

        constexpr size_t no_of_bits = detail::bit_width_v<T>;
        Size_t rest = n_items % no_of_bits;
        if (rest > static_cast<Size_t>(0)) {
            // n_items is not an integral multiple
            // of bit_width_v<T>
            return gr::work::Status::OK;
        }

        // Mathematical calculation
        size_t count = 0;
        Size_t bit_count = 1;
        T shift = 0;
        T part = 0;
        T result = 0;
        for (Size_t i = 0; i < n_items; ++i) {
            shift = static_cast<T>(no_of_bits - bit_count);
            part = static_cast<T>((in_span[i] & 1) << shift);
            result |= part;
            ++bit_count;
            if ((i + 1) % no_of_bits == 0) {
                out_span[count] = result;
                ++count;
                result = 0;
                bit_count = 1;
            }
        }

        return gr::work::Status::OK;
    }
};

} // namespace gr::blocks::math

#endif // GNURADIO_BYTEMATH_H
