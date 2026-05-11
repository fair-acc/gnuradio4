#ifndef GNURADIO_WIRESTREAM_HPP
#define GNURADIO_WIRESTREAM_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <vector>

#include <gnuradio-4.0/CRC.hpp>
#include <gnuradio-4.0/WireFormat.hpp>

namespace gr::wire {

/// Streaming transport layer for the V1 wire-format — stream header (once per stream/file)
/// + frame envelope (per chunk) wrapping a concat of self-delimited per-element records.
/// Opt-in for file/network IO; in-memory transports (PMR / USM ring) skip these layers.
/// Full spec in docs/CORE_WireFormat.md (Streaming transport section).

inline constexpr std::array<std::byte, 4> kStreamMagic     = {std::byte{'G'}, std::byte{'R'}, std::byte{'4'}, std::byte{'W'}};
inline constexpr std::uint16_t            kStreamVersionV1 = 0x0001U;

inline constexpr std::uint16_t kStreamFlagFrameCRC = 0x0001U; // per-frame CRC32C present; receiver verifies

inline constexpr std::size_t kStreamHeaderBytes  = 8UZ;
inline constexpr std::size_t kFrameEnvelopeBytes = 16UZ;
inline constexpr std::size_t kFrameCrcOffset     = 12UZ; // wire offset of crc32c within the envelope

struct StreamHeader {
    std::array<std::byte, 4> magic   = kStreamMagic;
    std::uint16_t            version = kStreamVersionV1;
    std::uint16_t            flags   = 0U;
};

struct FrameEnvelope {
    std::uint32_t frame_size = 0U; // payload bytes only (envelope NOT included)
    std::uint64_t sequence   = 0U; // per-frame monotonic; resets on new stream header
    std::uint32_t crc32c     = 0U; // CRC32C(envelope-with-this-field-zeroed || payload)
};

/// Serialise stream header into `out` (must be ≥ kStreamHeaderBytes; debug-asserted).
inline void writeStreamHeader(std::span<std::byte> out, const StreamHeader& h) noexcept {
    assert(out.size() >= kStreamHeaderBytes);
    std::memcpy(out.data() + 0, h.magic.data(), 4);
    std::memcpy(out.data() + 4, &h.version, sizeof(h.version));
    std::memcpy(out.data() + 6, &h.flags, sizeof(h.flags));
}

/// Parse a stream header; returns nullopt on magic / version mismatch.
[[nodiscard]] inline std::optional<StreamHeader> readStreamHeader(std::span<const std::byte> in) noexcept {
    if (in.size() < kStreamHeaderBytes) {
        return std::nullopt;
    }
    StreamHeader h;
    std::memcpy(h.magic.data(), in.data(), 4);
    if (h.magic != kStreamMagic) {
        return std::nullopt;
    }
    std::memcpy(&h.version, in.data() + 4, sizeof(h.version));
    std::memcpy(&h.flags, in.data() + 6, sizeof(h.flags));
    if (h.version != kStreamVersionV1) {
        return std::nullopt; // forward-compat: unknown major; caller may special-case
    }
    return h;
}

/// Serialise frame envelope (with CRC pre-computed, or 0 if not used).
/// `out` must be ≥ kFrameEnvelopeBytes; debug-asserted.
inline void writeFrameEnvelope(std::span<std::byte> out, const FrameEnvelope& e) noexcept {
    assert(out.size() >= kFrameEnvelopeBytes);
    std::memcpy(out.data() + 0, &e.frame_size, sizeof(e.frame_size));
    std::memcpy(out.data() + 4, &e.sequence, sizeof(e.sequence));
    std::memcpy(out.data() + kFrameCrcOffset, &e.crc32c, sizeof(e.crc32c));
}

[[nodiscard]] inline std::optional<FrameEnvelope> readFrameEnvelope(std::span<const std::byte> in) noexcept {
    if (in.size() < kFrameEnvelopeBytes) {
        return std::nullopt;
    }
    FrameEnvelope e;
    std::memcpy(&e.frame_size, in.data() + 0, sizeof(e.frame_size));
    std::memcpy(&e.sequence, in.data() + 4, sizeof(e.sequence));
    std::memcpy(&e.crc32c, in.data() + kFrameCrcOffset, sizeof(e.crc32c));
    return e;
}

/// Computes CRC32C over the envelope-with-CRC-field-zeroed concatenated with the payload.
/// Standard zero-out trick: caller writes envelope with crc32c=0, computes via this helper,
/// back-patches crc32c into the envelope before sending.
[[nodiscard]] inline std::uint32_t computeFrameCrc(std::span<const std::byte> envelopeWithZeroedCrc, std::span<const std::byte> payload) noexcept {
    auto state = gr::crc::Traits<gr::crc::Flavour::CRC32C_CASTAGNOLI>::kInit;
    state      = gr::crc::updateSimd<gr::crc::Flavour::CRC32C_CASTAGNOLI>(state, envelopeWithZeroedCrc);
    state      = gr::crc::updateSimd<gr::crc::Flavour::CRC32C_CASTAGNOLI>(state, payload);
    state ^= gr::crc::Traits<gr::crc::Flavour::CRC32C_CASTAGNOLI>::kXorOut;
    return state;
}

/// Builder that appends a stream header + frames to a byte buffer. Each frame wraps a
/// concatenation of records previously written (or directly supplied as bytes).
/// Not thread-safe: single-instance-per-thread. Independent instances are independent.
class WireStreamWriter {
public:
    explicit WireStreamWriter(std::uint16_t streamFlags = 0U) : _streamFlags(streamFlags) {}

    /// Emit the stream header. May throw `std::bad_alloc` from `out.resize()` on allocation failure.
    void writeHeader(std::vector<std::byte>& out) {
        const auto pos = out.size();
        out.resize(pos + kStreamHeaderBytes);
        writeStreamHeader(std::span<std::byte>(out.data() + pos, kStreamHeaderBytes), StreamHeader{.flags = _streamFlags});
        _sequence      = 0U;
        _headerWritten = true;
    }

    /// Emit one frame wrapping the given payload. Auto-emits the stream header
    /// on first call if `writeHeader()` was not invoked explicitly. May throw
    /// `std::bad_alloc` from `out.resize()` on allocation failure.
    void writeFrame(std::vector<std::byte>& out, std::span<const std::byte> payload) {
        if (!_headerWritten) {
            writeHeader(out);
        }
        const auto envPos     = out.size();
        const auto payloadPos = envPos + kFrameEnvelopeBytes;
        out.resize(payloadPos + payload.size());

        FrameEnvelope env{.frame_size = static_cast<std::uint32_t>(payload.size()), .sequence = _sequence++, .crc32c = 0U};
        writeFrameEnvelope(std::span<std::byte>(out.data() + envPos, kFrameEnvelopeBytes), env);
        std::memcpy(out.data() + payloadPos, payload.data(), payload.size());

        if ((_streamFlags & kStreamFlagFrameCRC) != 0U) {
            env.crc32c = computeFrameCrc(std::span<const std::byte>(out.data() + envPos, kFrameEnvelopeBytes), payload);
            writeFrameEnvelope(std::span<std::byte>(out.data() + envPos, kFrameEnvelopeBytes), env);
        }
    }

    [[nodiscard]] std::uint64_t nextSequence() const noexcept { return _sequence; }

private:
    std::uint16_t _streamFlags;
    std::uint64_t _sequence      = 0U;
    bool          _headerWritten = false;
};

enum class FrameStatus : std::uint8_t { Ok, Truncated, BadCrc, BadSequence };

/// `payload` aliases the buffer passed to WireStreamReader::begin(); do not retain past
/// that buffer's lifetime.
struct FrameView {
    FrameEnvelope              envelope{};
    std::span<const std::byte> payload{};
    FrameStatus                status = FrameStatus::Ok;
};

/// Iterator-style reader over a byte buffer that begins with a stream header.
/// Not thread-safe: single-instance-per-thread.
class WireStreamReader {
public:
    /// Parse the stream header at the start of `bytes`. Sets the internal cursor past it.
    /// Returns the parsed StreamHeader (or nullopt on bad magic / unknown version).
    [[nodiscard]] std::optional<StreamHeader> begin(std::span<const std::byte> bytes) noexcept {
        _bytes       = bytes;
        _cursor      = 0UZ;
        const auto h = readStreamHeader(_bytes);
        if (!h) {
            return std::nullopt;
        }
        _cursor       = kStreamHeaderBytes;
        _streamFlags  = h->flags;
        _nextExpected = 0U;
        return h;
    }

    /// Advance one frame; returns nullopt at end-of-stream.
    /// `status` field reflects truncation / CRC failure; payload may still be usable on BadCrc
    /// (caller's discretion). Sequence is checked monotonic against `_nextExpected`.
    [[nodiscard]] std::optional<FrameView> next() noexcept {
        if (_cursor >= _bytes.size()) {
            return std::nullopt;
        }
        FrameView  fv{};
        const auto envSpan = _bytes.subspan(_cursor);
        const auto e       = readFrameEnvelope(envSpan);
        if (!e) {
            _cursor   = _bytes.size(); // make subsequent next() return nullopt
            fv.status = FrameStatus::Truncated;
            return fv;
        }
        fv.envelope       = *e;
        const auto envEnd = _cursor + kFrameEnvelopeBytes;
        // Overflow-safe bound: the additive `envEnd + frame_size` can wrap on 32-bit size_t
        // (WASM/Emscripten), letting a corrupt frame_size slip past a `> size()` check and yield
        // an out-of-range subspan. envEnd <= _bytes.size() holds here (readFrameEnvelope verified
        // the envelope fits), so the subtraction never underflows.
        if (fv.envelope.frame_size > _bytes.size() - envEnd) {
            _cursor   = _bytes.size(); // make subsequent next() return nullopt
            fv.status = FrameStatus::Truncated;
            return fv;
        }
        const auto frameEnd = envEnd + fv.envelope.frame_size;
        fv.payload          = _bytes.subspan(envEnd, fv.envelope.frame_size);

        if (fv.envelope.sequence != _nextExpected) {
            fv.status = FrameStatus::BadSequence;
        }
        if ((_streamFlags & kStreamFlagFrameCRC) != 0U) {
            // recompute CRC over envelope-with-zeroed-CRC || payload
            std::array<std::byte, kFrameEnvelopeBytes> envCopy{};
            std::memcpy(envCopy.data(), _bytes.data() + _cursor, kFrameEnvelopeBytes);
            std::memset(envCopy.data() + kFrameCrcOffset, 0, sizeof(std::uint32_t));
            const auto actual = computeFrameCrc(envCopy, fv.payload);
            if (actual != fv.envelope.crc32c) {
                fv.status = FrameStatus::BadCrc; // CRC failure takes precedence over BadSequence
            }
        }
        _cursor = frameEnd;
        _nextExpected++;
        return fv;
    }

    [[nodiscard]] std::uint16_t streamFlags() const noexcept { return _streamFlags; }

private:
    std::span<const std::byte> _bytes{};
    std::size_t                _cursor       = 0UZ;
    std::uint16_t              _streamFlags  = 0U;
    std::uint64_t              _nextExpected = 0U;
};

} // namespace gr::wire

#endif // GNURADIO_WIRESTREAM_HPP
