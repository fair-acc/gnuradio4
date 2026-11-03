#ifndef GNURADIO_BASIC_KEEP_M_IN_N_HPP
#define GNURADIO_BASIC_KEEP_M_IN_N_HPP

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <format>
#include <limits>
#include <string>
#include <string_view>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/basic/PortCollection.hpp>

namespace gr::basic {

[[nodiscard]] constexpr std::size_t countKeptBefore(std::size_t inputIndex, std::size_t m, std::size_t n, std::size_t offset) noexcept {
    const std::size_t inChunk   = inputIndex % n;
    const std::size_t wrapped   = m > n - offset ? m - (n - offset) : 0UZ;
    const std::size_t pastFirst = inChunk > offset ? inChunk - offset : 0UZ;
    return (inputIndex / n) * m + std::min(inChunk, wrapped) + std::min(pastFirst, m - wrapped);
}

GR_REGISTER_BLOCK(gr::basic::KeepMInN, [T], [ uint8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double> ])

template<typename T>
requires std::is_arithmetic_v<T> || meta::complex_like<T>
struct KeepMInN : Block<KeepMInN<T>, NoTagPropagation, Resampling<1UZ, 1UZ, false>> {
    using Description = Doc<R""(keeps m of every n samples: sample i when (i - offset) mod n < m; the output rate is m/n of the input rate.

A tag on a dropped sample moves to the next kept one; a new m, n or offset applies from the next chunk of n samples; an incomplete last chunk is dropped.)"">;

    static constexpr std::string_view kSampleRateKey = gr::tag::kSampleRate.key();

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<gr::Size_t, "m", Doc<"samples kept per chunk">, Visible, Limits<1U, std::numeric_limits<gr::Size_t>::max()>>      m           = 1U;
    Annotated<gr::Size_t, "n", Doc<"chunk size">, Unit<"samples">, Visible, Limits<1U, std::numeric_limits<gr::Size_t>::max()>> n           = 2U;
    Annotated<gr::Size_t, "offset", Doc<"first kept sample of each chunk, below n">, Visible>                                   offset      = 0U;
    gr::SampleRate                                                                                                              sample_rate = 0.f;

    gr::Size_t _settings_epoch = 0U; // to keep processBulk const for GPU

    GR_MAKE_REFLECTABLE(KeepMInN, in, out, m, n, offset, sample_rate, _settings_epoch);

    mutable gr::Size_t _announcedEpoch = std::numeric_limits<gr::Size_t>::max();

    void start() {
        this->input_chunk_size  = n.value;
        this->output_chunk_size = m.value;
        _announcedEpoch         = std::numeric_limits<gr::Size_t>::max();
    }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) const {
        const std::size_t nChunks = std::min(input.size() / n.value, output.size() / m.value);
        if (nChunks == 0UZ) {
            std::ignore = input.consume(0UZ);
            output.publish(0UZ);
            return input.size() < n.value ? gr::work::Status::INSUFFICIENT_INPUT_ITEMS : gr::work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }

        const std::size_t wrapped = m.value > n.value - offset.value ? m.value - (n.value - offset.value) : 0UZ;
        for (std::size_t chunk = 0UZ; chunk < nChunks; chunk++) {
            const auto chunkStart = input.begin() + static_cast<std::ptrdiff_t>(chunk * n.value);
            const auto wrappedEnd = std::copy_n(chunkStart, wrapped, output.begin() + static_cast<std::ptrdiff_t>(chunk * m.value));
            std::copy_n(chunkStart + static_cast<std::ptrdiff_t>(offset.value), m.value - wrapped, wrappedEnd);
        }

        const float ratio       = static_cast<float>(m.value) / static_cast<float>(n.value);
        const auto  publishRate = [&output](float rate, std::size_t at) {
            alignas(gr::pmt::kBlobAlignment) std::array<std::byte, gr::kDeviceEventSlotBytes> blob{};
            gr::property_map_view                                                             tag = gr::formatEvent(blob, 1U);
            std::ignore                                                                           = tag.try_emplace(kSampleRateKey, rate);
            output.publishTag(tag, at);
        };
        const auto publishRescaled = [&](const gr::property_map_view& tagMap, std::size_t at) {
            const float* inputRate = tagMap.template get_if<float>(kSampleRateKey);
            const auto   source    = tagMap.blob();
            if (inputRate == nullptr || source.size() > gr::kDeviceEventSlotBytes) {
                output.publishTag(tagMap, at);
                if (inputRate != nullptr) {
                    publishRate(*inputRate * ratio, at);
                }
                return;
            }

            alignas(gr::pmt::kBlobAlignment) std::array<std::byte, gr::kDeviceEventSlotBytes> copy{};
            std::ranges::copy(source, copy.begin());
            gr::property_map_view rescaled = gr::pmt::ValueMapView::viewOf(std::span<const std::byte>(copy).first(source.size()));
            std::ignore                    = rescaled.insert_or_assign(kSampleRateKey, *inputRate * ratio);
            output.publishTag(rescaled, at);
        };

        const auto carriesRateAtStart = [&](const gr::Tag& tag) {
            const bool atStart = tag.index == input.streamIndex;
            return atStart && tag.map.template get_if<float>(kSampleRateKey) != nullptr;
        };
        const bool rateTagAtStart = std::ranges::any_of(input.rawTags(), carriesRateAtStart);
        if (_announcedEpoch != _settings_epoch) {
            if (sample_rate.value > 0.f && !rateTagAtStart) {
                publishRate(sample_rate.value * ratio, 0UZ);
            }
            _announcedEpoch = _settings_epoch;
        }

        const std::size_t consumed = nChunks * n.value;
        for (const gr::Tag& tag : input.rawTags()) {
            if (tag.index < input.streamIndex) { // already published by the previous call
                continue;
            }
            const std::size_t position = tag.index - input.streamIndex;
            if (position >= consumed) {
                break;
            }
            publishRescaled(tag.map, countKeptBefore(position, m.value, n.value, offset.value));
        }

        std::ignore = input.consume(consumed);
        output.publish(nChunks * m.value);
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        if (!newSettings.contains("m") && !newSettings.contains("n") && !newSettings.contains("offset")) {
            return;
        }
        const auto oldM = oldSettings.value_or<gr::Size_t>("m", 1U);
        const auto oldN = oldSettings.value_or<gr::Size_t>("n", 2U);
        if (m.value > n.value || offset.value >= n.value) [[unlikely]] {
            const std::string refusal = std::format("m = {} and offset = {} do not fit n = {}: keeps the previous selection", m.value, offset.value, n.value);
            this->emitErrorMessage("settingsChanged()", refusal);
            m      = oldM;
            n      = oldN;
            offset = oldSettings.value_or<gr::Size_t>("offset", 0U);
            return;
        }
        if (!detail::chunkFitsBuffer(*this, in, n.value, std::format("a chunk of n = {} samples", n.value))) {
            return;
        }
        this->input_chunk_size  = n.value;
        this->output_chunk_size = m.value;
        if (static_cast<float>(m.value) / static_cast<float>(n.value) != static_cast<float>(oldM) / static_cast<float>(oldN)) {
            _settings_epoch++;
        }
    }
};

} // namespace gr::basic

#endif // GNURADIO_BASIC_KEEP_M_IN_N_HPP
