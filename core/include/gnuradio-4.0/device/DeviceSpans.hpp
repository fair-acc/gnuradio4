#ifndef GNURADIO_DEVICE_SPANS_HPP
#define GNURADIO_DEVICE_SPANS_HPP

#include <cstddef>
#include <span>

#include <concepts>

#include <gnuradio-4.0/Tag.hpp>

namespace gr::device {

/**
 * @brief What a kernel-side span records for the host to apply once the kernel has finished.
 *
 * A device span cannot advance a host ring cursor, so `consume()`/`publish()` write here instead and the host
 * replays them onto the real port spans afterwards — from there `blockManagedIO` finalises the work() exactly as
 * it does for a `processBulk_sycl` hatch. Lives in device-visible memory and is read by the host after the barrier.
 */
struct DeviceSpanAccounting {
    std::size_t consumed            = 0UZ;
    std::size_t published           = 0UZ;
    std::size_t tagsConsumed        = 0UZ;
    std::size_t tagsPublished       = 0UZ;
    bool        consumeRequested    = false;
    bool        publishRequested    = false;
    bool        tagPublishAttempted = false; // an OWNING payload was passed: only a view can be built in a kernel
    bool        tagSlotsExhausted   = false;
    bool        inputTagsTruncated  = false; // more input tags, or a larger payload, than the staging slots hold
};

/**
 * @brief A kernel-side `InputSpanLike`: the samples, the input tags, and somewhere to record the consume.
 *
 * `std::span` already serves a body constrained to `InputViewLike`. This exists for the *span* concepts, which a
 * variable-rate body needs — a decimator or an IIR cannot express itself without `consume()`.
 */
template<typename T>
struct DeviceInputSpan {
    using value_type = T;

    const T*              _data       = nullptr;
    std::size_t           _size       = 0UZ;
    const gr::Tag*        _tags       = nullptr;
    std::size_t           _tagCount   = 0UZ;
    DeviceSpanAccounting* _acct       = nullptr;
    std::size_t           streamIndex = 0UZ; // absolute position of the first sample, as the host reader sees it
    bool                  isConnected = true;
    bool                  isSync      = true;

    [[nodiscard]] constexpr const T*    begin() const noexcept { return _data; }
    [[nodiscard]] constexpr const T*    end() const noexcept { return _data + _size; }
    [[nodiscard]] constexpr const T*    data() const noexcept { return _data; }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return _size; }
    [[nodiscard]] constexpr const T&    operator[](std::size_t index) const noexcept { return _data[index]; }
    constexpr                           operator std::span<const T>() const noexcept { return {_data, _size}; }

    [[nodiscard]] constexpr bool empty() const noexcept { return _size == 0UZ; }

    [[nodiscard]] constexpr std::span<const T> first(std::size_t nSamples) const noexcept { return std::span<const T>(_data, nSamples); }
    [[nodiscard]] constexpr std::span<const T> last(std::size_t nSamples) const noexcept { return std::span<const T>(_data + (_size - nSamples), nSamples); }
    [[nodiscard]] constexpr std::span<const T> subspan(std::size_t offset, std::size_t count) const noexcept { return std::span<const T>(_data + offset, count); }

    [[nodiscard]] static constexpr std::ptrdiff_t relIndex(std::size_t absolute, std::size_t base) noexcept {
        return absolute >= base ? static_cast<std::ptrdiff_t>(absolute - base) : -static_cast<std::ptrdiff_t>(base - absolute);
    }

    /// `bool`, not `void`: real reader spans return one and block bodies write `std::ignore = input.consume(n)`
    [[nodiscard]] constexpr bool consume(std::size_t nSamples) const noexcept {
        _acct->consumed         = nSamples;
        _acct->consumeRequested = true;
        return true;
    }

    [[nodiscard]] constexpr bool tryConsume(std::size_t nSamples) const noexcept { return nSamples <= _size && consume(nSamples); }

    [[nodiscard]] constexpr bool isConsumeRequested() const noexcept { return _acct->consumeRequested; }

    [[nodiscard]] constexpr std::span<const gr::Tag> rawTags() const noexcept { return {_tags, _tagCount}; }
    [[nodiscard]] constexpr std::span<const gr::Tag> tags() const noexcept { return rawTags(); }
    [[nodiscard]] constexpr std::span<const gr::Tag> tags(std::size_t) const noexcept { return rawTags(); }
    constexpr void               consumeTags(std::size_t nTags) const noexcept { _acct->tagsConsumed = nTags; }
    [[nodiscard]] constexpr bool consumeRawTags(std::size_t nTags) const noexcept {
        _acct->tagsConsumed = nTags;
        return true;
    }
};

/// the nested `tags` a kernel-side `OutputSpanLike` must expose; the slots are pre-reserved by the host
template<typename TTag = gr::Tag>
struct DeviceTagWriterSpan {
    using value_type = TTag;

    TTag*                 _data = nullptr;
    std::size_t           _size = 0UZ;
    DeviceSpanAccounting* _acct = nullptr;

    [[nodiscard]] constexpr TTag*       begin() const noexcept { return _data; }
    [[nodiscard]] constexpr TTag*       end() const noexcept { return _data + _size; }
    [[nodiscard]] constexpr TTag*       data() const noexcept { return _data; }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return _size; }
    [[nodiscard]] constexpr TTag&       operator[](std::size_t index) const noexcept { return _data[index]; }
    constexpr                           operator std::span<TTag>() const noexcept { return {_data, _size}; }

    constexpr void publish(std::size_t) const noexcept { /* host reserves and publishes; see DeviceSpans §tags */ }
};

/// @brief A kernel-side `OutputSpanLike`: the samples, the tag slots, and somewhere to record the publish.
template<typename T>
struct DeviceOutputSpan {
    using value_type = T;

    T*                    _data       = nullptr;
    std::size_t           _size       = 0UZ;
    DeviceSpanAccounting* _acct       = nullptr;
    bool                  isConnected = true;
    bool                  isSync      = true;

    DeviceTagWriterSpan<gr::Tag> tags{};

    // pre-reserved by the host: one fixed-size, kBlobAlignment-aligned slot per tag the kernel may publish
    std::byte*   _tagSlots     = nullptr;
    std::size_t* _tagOffsets   = nullptr;
    std::size_t  _tagSlotCount = 0UZ;
    std::size_t  _tagSlotBytes = 0UZ;

    [[nodiscard]] constexpr T*          begin() const noexcept { return _data; }
    [[nodiscard]] constexpr T*          end() const noexcept { return _data + _size; }
    [[nodiscard]] constexpr T*          data() const noexcept { return _data; }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return _size; }
    [[nodiscard]] constexpr T&          operator[](std::size_t index) const noexcept { return _data[index]; }
    constexpr                           operator std::span<T>() const noexcept { return {_data, _size}; }

    [[nodiscard]] constexpr bool          empty() const noexcept { return _size == 0UZ; }
    [[nodiscard]] constexpr std::span<T>  first(std::size_t nSamples) const noexcept { return std::span<T>(_data, nSamples); }
    [[nodiscard]] constexpr std::span<T>  last(std::size_t nSamples) const noexcept { return std::span<T>(_data + (_size - nSamples), nSamples); }
    [[nodiscard]] constexpr std::span<T>  subspan(std::size_t offset, std::size_t count) const noexcept { return std::span<T>(_data + offset, count); }

    constexpr void publish(std::size_t nSamples) const noexcept {
        _acct->published        = nSamples;
        _acct->publishRequested = true;
    }

    [[nodiscard]] constexpr bool isPublishRequested() const noexcept { return _acct->publishRequested; }

    /// A `ValueMapView` aliases bytes a kernel can build (`ValueMapView::formatAt` + `try_emplace` are
    /// device-callable), so it is copied into a pre-reserved slot and replayed by the host through the ordinary
    /// `publishTag`. An owning `property_map` cannot be constructed in a kernel at all, so that form only records
    /// the attempt — the dispatcher's canary sees it before any kernel launches and takes the host path.
    ///
    /// A template on purpose: the body is instantiated only when a kernel actually calls it.
    template<typename TPropertyMap>
    constexpr void publishTag(TPropertyMap&& tagData, std::size_t tagOffset = 0UZ) const noexcept {
        if constexpr (std::same_as<std::remove_cvref_t<TPropertyMap>, gr::pmt::ValueMapView>) {
            const std::size_t slot = _acct->tagsPublished;
            if (_tagSlots == nullptr || slot >= _tagSlotCount) {
                _acct->tagSlotsExhausted = true;
                return;
            }
            const std::span<const std::byte> blob = tagData.blob();
            if (blob.size() > _tagSlotBytes) {
                _acct->tagSlotsExhausted = true;
                return;
            }
            std::byte* destination = _tagSlots + slot * _tagSlotBytes;
            for (std::size_t byteIndex = 0UZ; byteIndex < blob.size(); ++byteIndex) {
                destination[byteIndex] = blob[byteIndex];
            }
            _tagOffsets[slot]     = tagOffset;
            _acct->tagsPublished  = slot + 1UZ;
        } else {
            _acct->tagPublishAttempted = true;
        }
    }
};

} // namespace gr::device

#endif // GNURADIO_DEVICE_SPANS_HPP
