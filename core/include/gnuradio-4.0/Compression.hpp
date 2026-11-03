#ifndef GNURADIO_COMPRESSION_HPP
#define GNURADIO_COMPRESSION_HPP

#include <gnuradio-4.0/CRC.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <limits>
#include <memory_resource>
#include <new>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace gr::compression {

/**
 * @brief DEFLATE/zlib/gzip codec usable during constant evaluation and at run time.
 *
 * The format implementation follows RFC 1950 (zlib), RFC 1951 (DEFLATE), and
 * RFC 1952 (gzip):
 * - https://www.rfc-editor.org/rfc/rfc1950
 * - https://www.rfc-editor.org/rfc/rfc1951
 * - https://www.rfc-editor.org/rfc/rfc1952
 *
 * Encoder and decoder share one `constexpr` implementation, so a compile-time
 * resource and a run-time call exercise identical code. Every `gzip<>()`
 * resource is proven to round-trip by a `static_assert` at the point of
 * definition.
 *
 * Text resources:
 * ```cpp
 * using namespace gr::compression::literals;
 * static constexpr auto help = "embedded help text"_gzip;
 * static constexpr auto compactHelp = gr::compression::makeCompressedText<
 *     "embedded help text", gr::compression::CompressionLevel::best>();
 * std::string_view view = help;
 * ```
 *
 * Binary resources:
 * ```cpp
 * inline constexpr std::array source{std::byte{0x01}, std::byte{0x02}};
 * inline constexpr auto packed = gr::compression::gzip<source, gr::compression::CompressionLevel::best>();
 * ```
 *
 * Run-time use:
 * ```cpp
 * const auto packed  = gr::compression::compress(payload, gr::compression::Format::gzip);
 * const auto plain   = gr::compression::decompress(*packed, gr::compression::Format::gzip);
 * const auto pooled  = gr::compression::decompress(*packed, gr::compression::Format::gzip, &arena);
 * ```
 *
 * Streaming use:
 * ```cpp
 * gr::compression::StreamDecompressor inflate{gr::compression::Format::gzip};
 * while (!inflate.finished()) {
 *     const auto state = inflate.process(pending, output, atEndOfSource ? Flush::finish : Flush::none);
 *     pending = pending.subspan(state->consumed); // keep what was not consumed
 *     consume(output.first(state->produced));
 *     if (state->status == gr::compression::StreamStatus::needInput) { pending = refill(pending); }
 * }
 * ```
 * Contract for both stream types:
 * - `input` and `output` are borrowed for the duration of one `process()` call only.
 * - `consumed` may be **less** than `input.size()` even on `needInput`: the decoder rewinds to a symbol
 *   boundary when it runs out of bits, so the caller must offer the unconsumed tail again.
 * - `status` says why the call returned; act on it rather than inferring from `consumed`/`produced`.
 * - errors are sticky: once `process()` fails it keeps failing and produces nothing until `reset()`.
 * - on error the produced count is not reported, so partial output from a broken stream is discarded.
 *
 * `fast`, `balanced`, and `best` increase match-search effort; `best` also uses
 * lazy matching. The `_gzip` literal and APIs without a level use `balanced`.
 *
 * Compile-time resources up to ~64 KiB fit GCC's default `-fconstexpr-ops-limit`
 * and ~16 KiB fit Clang's default `-fconstexpr-steps`; larger payloads need the
 * respective flag raised.
 */

enum class Format : std::uint8_t { rawDeflate, zlib, gzip };

enum class CompressionLevel : std::uint8_t { fast, balanced, best };

enum class Error : std::uint8_t { truncatedInput, outputTooSmall, invalidHeader, unsupportedCompressionMethod, reservedFlags, presetDictionaryRequired, invalidBlockType, invalidStoredBlockLength, invalidHuffmanTree, invalidCode, invalidLength, invalidDistance, invalidBackReference, checksumMismatch, sizeMismatch, trailingData, integerOverflow, allocationFailed, sizeLimitExceeded };

enum class Flush : std::uint8_t { none, finish };

/// why `process()` returned: the caller feeds more input, drains the output, or stops
enum class StreamStatus : std::uint8_t { needInput, needOutput, done };

struct StreamResult {
    std::size_t  consumed{};
    std::size_t  produced{};
    StreamStatus status{StreamStatus::needInput};

    [[nodiscard]] constexpr bool finished() const noexcept { return status == StreamStatus::done; }
};

/// default cap for allocating decode overloads; enforced while counting before allocation
inline constexpr std::size_t kDefaultMaxDecompressedSize = 256UZ << 20U;

namespace detail {
/// 5552 is the longest run that cannot overflow the 32-bit accumulators, so the modulo runs once per run
[[nodiscard]] constexpr std::pair<std::uint32_t, std::uint32_t> updateAdler(std::uint32_t a, std::uint32_t b, std::span<const std::byte> input) noexcept {
    constexpr std::uint32_t kModAdler = 65521U;
    constexpr std::size_t   kMaxRun   = 5552UZ;
    while (!input.empty()) {
        const auto run = input.first(std::min(kMaxRun, input.size()));
        for (const std::byte byte : run) {
            a += std::to_integer<std::uint8_t>(byte);
            b += a;
        }
        a %= kModAdler;
        b %= kModAdler;
        input = input.subspan(run.size());
    }
    return {a, b};
}
} // namespace detail

[[nodiscard]] constexpr std::uint32_t adler32(std::span<const std::byte> input) noexcept {
    const auto [a, b] = detail::updateAdler(1U, 0U, input);
    return (b << 16U) | a;
}

namespace detail {

inline constexpr std::array<std::uint16_t, 29> kLengthBase{3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};
inline constexpr std::array<std::uint8_t, 29>  kLengthExtra{0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0};
inline constexpr std::array<std::uint16_t, 30> kDistanceBase{1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577};
inline constexpr std::array<std::uint8_t, 30>  kDistanceExtra{0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
inline constexpr std::array<std::uint8_t, 19>  kCodeLengthOrder{16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15};

inline constexpr std::size_t kMaxStoredBlock = 65535UZ;
inline constexpr std::size_t kMaxMatchLength = 258UZ;
inline constexpr std::size_t kWindowSize     = 32768UZ;

[[nodiscard]] constexpr std::size_t storedBlockCount(std::size_t size) noexcept { return size == 0UZ ? 1UZ : (size + kMaxStoredBlock - 1UZ) / kMaxStoredBlock; }

[[nodiscard]] constexpr std::uint16_t reverseBits(std::uint16_t value, unsigned count) noexcept {
    std::uint16_t result = 0;
    for (unsigned i = 0; i < count; ++i) {
        result = static_cast<std::uint16_t>((static_cast<unsigned>(result) << 1U) | (static_cast<unsigned>(value) & 1U));
        value >>= 1U;
    }
    return result;
}

struct EncodedSymbol {
    unsigned symbol{};
    unsigned extraValue{};
    unsigned extraBits{};
};

[[nodiscard]] constexpr EncodedSymbol encodeLength(std::size_t length) noexcept {
    const auto index = static_cast<std::size_t>(std::ranges::upper_bound(kLengthBase, static_cast<std::uint16_t>(length)) - kLengthBase.begin()) - 1UZ;
    return {.symbol = static_cast<unsigned>(257UZ + index), .extraValue = static_cast<unsigned>(length - kLengthBase[index]), .extraBits = kLengthExtra[index]};
}

[[nodiscard]] constexpr EncodedSymbol encodeDistance(std::size_t distance) noexcept {
    const auto index = static_cast<std::size_t>(std::ranges::upper_bound(kDistanceBase, static_cast<std::uint16_t>(distance)) - kDistanceBase.begin()) - 1UZ;
    return {.symbol = static_cast<unsigned>(index), .extraValue = static_cast<unsigned>(distance - kDistanceBase[index]), .extraBits = kDistanceExtra[index]};
}

/// fills a buffer whose exact size a BitCounter pass determined beforehand; keeps counting past the end so
/// a mismatch between the measuring and writing pass surfaces as `overflowed()` instead of a stray write
struct ByteSink {
    std::span<std::byte> bytes;
    std::size_t          pos{};

    constexpr void put(std::byte value) {
        if (pos < bytes.size()) [[likely]] {
            bytes[pos] = value;
        }
        ++pos;
    }

    constexpr void insert(std::span<const std::byte> values) {
        const auto writePos = std::min(pos, bytes.size());
        if (values.size() <= bytes.size() - writePos) [[likely]] {
            std::ranges::copy(values, bytes.begin() + static_cast<std::ptrdiff_t>(writePos));
        }
        pos += values.size();
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept { return pos; }
    [[nodiscard]] constexpr bool        overflowed() const noexcept { return pos > bytes.size(); }
};

/// appends to a vector that grows as needed; used by the streaming encoder, whose block size is not known up front
struct GrowingSink {
    std::vector<std::byte>& bytes;

    void                      put(std::byte value) { bytes.push_back(value); }
    void                      insert(std::span<const std::byte> values) { bytes.insert(bytes.end(), values.begin(), values.end()); }
    [[nodiscard]] std::size_t size() const noexcept { return bytes.size(); }
};

template<typename Sink>
struct BitWriter {
    Sink&         sink;
    std::uint64_t bits{};
    unsigned      count{};

    constexpr void writeBits(std::uint32_t value, unsigned width) {
        bits |= static_cast<std::uint64_t>(value) << count;
        count += width;
        while (count >= 8U) {
            sink.put(static_cast<std::byte>(bits & 0xffU));
            bits >>= 8U;
            count -= 8U;
        }
    }

    constexpr void alignByte() {
        if (count != 0U) {
            sink.put(static_cast<std::byte>(bits & 0xffU));
            bits  = 0;
            count = 0;
        }
    }

    constexpr void writeBytes(std::span<const std::byte> values) { sink.insert(values); }

    [[nodiscard]] constexpr std::size_t size() const noexcept { return sink.size(); }
};

/// measures an encoding through the very same emit path that writes it, so size and output cannot drift apart
struct BitCounter {
    std::size_t bits{};

    constexpr void                      writeBits(std::uint32_t, unsigned width) noexcept { bits += width; }
    constexpr void                      alignByte() noexcept { bits = (bits + 7UZ) & ~7UZ; }
    constexpr void                      writeBytes(std::span<const std::byte> values) noexcept { bits += 8UZ * values.size(); }
    [[nodiscard]] constexpr std::size_t size() const noexcept { return bits / 8UZ; }
};

template<typename Writer>
constexpr void putLittle16(Writer& writer, std::uint16_t value) {
    writer.writeBits(value, 16);
}

template<typename Writer>
constexpr void putLittle32(Writer& writer, std::uint32_t value) {
    writer.writeBits(value, 32);
}

template<typename Writer>
constexpr void putBig32(Writer& writer, std::uint32_t value) {
    for (unsigned shift = 32U; shift > 0U; shift -= 8U) {
        writer.writeBits((value >> (shift - 8U)) & 0xffU, 8);
    }
}

struct HuffmanCode {
    std::uint16_t bits{};
    std::uint8_t  width{};
};

[[nodiscard]] constexpr HuffmanCode fixedLiteralCode(unsigned symbol) noexcept {
    if (symbol <= 143U) {
        return {.bits = reverseBits(static_cast<std::uint16_t>(0x30U + symbol), 8), .width = 8};
    }
    if (symbol <= 255U) {
        return {.bits = reverseBits(static_cast<std::uint16_t>(0x190U + symbol - 144U), 9), .width = 9};
    }
    if (symbol <= 279U) {
        return {.bits = reverseBits(static_cast<std::uint16_t>(symbol - 256U), 7), .width = 7};
    }
    return {.bits = reverseBits(static_cast<std::uint16_t>(0xc0U + symbol - 280U), 8), .width = 8};
}

[[nodiscard]] constexpr HuffmanCode fixedDistanceCode(unsigned symbol) noexcept { return {.bits = reverseBits(static_cast<std::uint16_t>(symbol), 5), .width = 5}; }

struct Token {
    std::uint16_t value{};
    std::uint16_t distance{};

    [[nodiscard]] constexpr bool isLiteral() const noexcept { return distance == 0U; }
};

struct Match {
    std::size_t length{};
    std::size_t distance{};
};

struct MatchPolicy {
    std::size_t maxCandidates{};
    bool        lazyMatching{};
};

[[nodiscard]] constexpr MatchPolicy matchPolicy(CompressionLevel level) noexcept {
    switch (level) {
    case CompressionLevel::fast: return {.maxCandidates = 4UZ, .lazyMatching = false};
    case CompressionLevel::balanced: return {.maxCandidates = 16UZ, .lazyMatching = false};
    case CompressionLevel::best: return {.maxCandidates = 128UZ, .lazyMatching = true};
    }
    std::unreachable();
}

[[nodiscard]] constexpr std::size_t historyBound(std::size_t size) noexcept { return std::max(1UZ, std::min(size, kWindowSize)); }

inline constexpr std::size_t kHashSize = 65536UZ;

template<std::size_t N>
struct TokenBuffer {
    std::array<Token, N> value{};
    std::size_t          size{};

    [[nodiscard]] constexpr std::span<const Token> view() const noexcept { return std::span<const Token>(value).first(size); }
};

/// scratch tables owned by the caller: `std::array` during constant evaluation, `std::vector` at run time
struct EncoderScratch {
    std::span<Token>       tokens;
    std::span<std::size_t> head;
    std::span<std::size_t> previous;
};

/// hot loops index raw pointers: `std::span` and even `std::array` subscripting cost several times more per access during constant evaluation
///
/// Chains hold absolute stream positions, so a streaming encoder can keep them across blocks without
/// clearing: entries that have fallen out of the 32 KiB window are rejected by the distance guard.
struct Matcher {
    const std::byte* data{};
    std::size_t      dataSize{};
    std::size_t      basePosition{};
    std::size_t      maxCandidates{};
    std::size_t*     head{};
    std::size_t*     previous{};
    std::size_t      chainSize{};

    [[nodiscard]] constexpr std::size_t hash(std::size_t pos) const noexcept {
        const auto a     = std::to_integer<std::uint32_t>(data[pos]);
        const auto b     = std::to_integer<std::uint32_t>(data[pos + 1UZ]);
        const auto c     = std::to_integer<std::uint32_t>(data[pos + 2UZ]);
        const auto value = (a << 16U) | (b << 8U) | c;
        return (value * 2654435761U) >> 16U;
    }

    constexpr void insert(std::size_t pos) noexcept {
        if (pos + 2UZ >= dataSize) {
            return;
        }
        const auto key                 = hash(pos);
        const auto absolute            = basePosition + pos;
        previous[absolute % chainSize] = head[key];
        head[key]                      = absolute + 1UZ;
    }

    [[nodiscard]] constexpr Match find(std::size_t pos) const noexcept {
        if (pos + 2UZ >= dataSize) {
            return {};
        }

        const auto  limit    = std::min(kMaxMatchLength, dataSize - pos);
        const auto  absolute = basePosition + pos;
        Match       best{};
        const auto  oldest    = std::max(basePosition, absolute > kWindowSize ? absolute - kWindowSize : 0UZ);
        std::size_t candidate = head[hash(pos)];
        std::size_t checked   = 0;
        while (candidate > oldest && candidate <= absolute && checked < maxCandidates) {
            --candidate;
            ++checked;
            const auto  offset = candidate - basePosition;
            std::size_t length = 0;
            while (length < limit && data[offset + length] == data[pos + length]) {
                ++length;
            }
            if (length >= 3UZ && length > best.length) {
                best = {.length = length, .distance = absolute - candidate};
                if (length == kMaxMatchLength) {
                    break;
                }
            }
            candidate = previous[candidate % chainSize];
        }
        return best;
    }
};

/// describes where a block sits in a longer stream; all-zero for a one-shot buffer
struct WindowLayout {
    std::size_t basePosition{}; // absolute stream position of input[0]
    std::size_t indexedFrom{};  // offset from which the hash chains still need filling
    std::size_t tokenFrom{};    // offset at which token emission starts
};

[[nodiscard]] constexpr std::span<const Token> tokenize(std::span<const std::byte> input, WindowLayout layout, CompressionLevel level, EncoderScratch scratch) {
    const auto       policy = matchPolicy(level);
    const std::byte* data   = input.data();
    const auto       size   = input.size();
    Token*           output = scratch.tokens.data();
    Matcher          matcher{.data = data, .dataSize = size, .basePosition = layout.basePosition, .maxCandidates = policy.maxCandidates, .head = scratch.head.data(), .previous = scratch.previous.data(), .chainSize = scratch.previous.size()};

    std::size_t count = 0;
    std::size_t pos   = layout.indexedFrom;
    while (pos < layout.tokenFrom) {
        matcher.insert(pos);
        ++pos;
    }
    while (pos < size) {
        const auto  match       = matcher.find(pos);
        std::size_t firstInsert = 0;
        if (policy.lazyMatching && match.length >= 3UZ) {
            matcher.insert(pos);
            firstInsert = 1;
            if (matcher.find(pos + 1UZ).length > match.length) {
                output[count++] = {.value = std::to_integer<std::uint8_t>(data[pos]), .distance = 0};
                ++pos;
                continue;
            }
        }
        if (match.length >= 3UZ) {
            output[count++] = {.value = static_cast<std::uint16_t>(match.length), .distance = static_cast<std::uint16_t>(match.distance)};
            for (std::size_t i = firstInsert; i < match.length; ++i) {
                matcher.insert(pos + i);
            }
            pos += match.length;
        } else {
            output[count++] = {.value = std::to_integer<std::uint8_t>(data[pos]), .distance = 0};
            matcher.insert(pos);
            ++pos;
        }
    }
    return scratch.tokens.first(count);
}

template<std::size_t N>
struct CanonicalCodes {
    std::array<std::uint8_t, N>  lengths{};
    std::array<std::uint16_t, N> bits{};
};

template<std::size_t N>
[[nodiscard]] constexpr CanonicalCodes<N> makeCanonicalCodes(std::array<std::uint8_t, N> lengths) noexcept {
    CanonicalCodes<N>             result{.lengths = lengths};
    std::array<std::uint16_t, 16> counts{};
    std::array<std::uint16_t, 16> next{};
    for (const auto length : lengths) {
        if (length != 0U) {
            ++counts[length];
        }
    }
    std::uint16_t code = 0;
    for (std::size_t width = 1; width < counts.size(); ++width) {
        code        = static_cast<std::uint16_t>((code + counts[width - 1UZ]) << 1U);
        next[width] = code;
    }
    for (std::size_t symbol = 0; symbol < N; ++symbol) {
        const auto width = lengths[symbol];
        if (width != 0U) {
            result.bits[symbol] = reverseBits(next[width]++, width);
        }
    }
    return result;
}

template<std::size_t N, unsigned MaxBits>
[[nodiscard]] constexpr std::array<std::uint8_t, N> makeCodeLengths(std::array<std::uint32_t, N> frequencies) noexcept {
    constexpr std::size_t kNone  = std::numeric_limits<std::size_t>::max();
    std::size_t           active = static_cast<std::size_t>(std::ranges::count_if(frequencies, [](std::uint32_t f) { return f != 0U; }));
    for (std::size_t symbol = 0; active < 2UZ && symbol < N; ++symbol) {
        if (frequencies[symbol] == 0U) {
            frequencies[symbol] = 1U;
            ++active;
        }
    }

    std::array<std::uint64_t, 2UZ * N> nodeFrequency{};
    std::array<std::size_t, 2UZ * N>   parent{};
    parent.fill(kNone);
    std::ranges::copy(frequencies, nodeFrequency.begin());

    // std::ranges heap operations want a "greater" comparator to build a min-heap; index breaks ties for reproducible output
    const auto weaker = [&nodeFrequency](std::size_t left, std::size_t right) constexpr { return nodeFrequency[left] > nodeFrequency[right] || (nodeFrequency[left] == nodeFrequency[right] && left > right); };

    std::array<std::size_t, 2UZ * N> heap{};
    std::size_t                      heapSize = 0;
    const auto                       push     = [&heap, &heapSize, &weaker](std::size_t node) constexpr {
        heap[heapSize++] = node;
        std::ranges::push_heap(heap.begin(), heap.begin() + static_cast<std::ptrdiff_t>(heapSize), weaker);
    };
    const auto pop = [&heap, &heapSize, &weaker]() constexpr {
        std::ranges::pop_heap(heap.begin(), heap.begin() + static_cast<std::ptrdiff_t>(heapSize), weaker);
        return heap[--heapSize];
    };

    for (std::size_t symbol = 0; symbol < N; ++symbol) {
        if (frequencies[symbol] != 0U) {
            push(symbol);
        }
    }
    std::size_t nodeCount = N;
    while (heapSize > 1UZ) {
        const auto first         = pop();
        const auto second        = pop();
        parent[first]            = nodeCount;
        parent[second]           = nodeCount;
        nodeFrequency[nodeCount] = nodeFrequency[first] + nodeFrequency[second];
        push(nodeCount++);
    }

    std::array<unsigned, MaxBits + 1UZ> lengthCounts{};
    std::array<unsigned, 2UZ * N>       depths{};
    for (std::size_t node = nodeCount; node-- > 0UZ;) {
        if (parent[node] != kNone) {
            depths[node] = depths[parent[node]] + 1U;
        }
    }
    for (std::size_t symbol = 0; symbol < N; ++symbol) {
        if (frequencies[symbol] != 0U) {
            ++lengthCounts[std::min(depths[symbol], MaxBits)];
        }
    }

    std::size_t usedCodeSpace = 0;
    for (unsigned width = 1; width <= MaxBits; ++width) {
        usedCodeSpace += static_cast<std::size_t>(lengthCounts[width]) << (MaxBits - width);
    }
    const auto availableCodeSpace = std::size_t{1} << MaxBits;
    while (usedCodeSpace > availableCodeSpace) {
        unsigned width = MaxBits - 1U;
        while (width > 0U && lengthCounts[width] == 0U) {
            --width;
        }
        --lengthCounts[width];
        lengthCounts[width + 1U] += 2U;
        --lengthCounts[MaxBits];
        --usedCodeSpace;
    }

    std::array<std::size_t, N> ordered{};
    std::size_t                orderedSize = 0;
    for (std::size_t symbol = 0; symbol < N; ++symbol) {
        if (frequencies[symbol] != 0U) {
            ordered[orderedSize++] = symbol;
        }
    }
    std::ranges::sort(ordered.begin(), ordered.begin() + static_cast<std::ptrdiff_t>(orderedSize), [&frequencies](std::size_t left, std::size_t right) constexpr { return frequencies[left] < frequencies[right] || (frequencies[left] == frequencies[right] && left < right); });

    std::array<std::uint8_t, N> lengths{};
    std::size_t                 orderedIndex = 0;
    for (unsigned width = MaxBits; width > 0U; --width) {
        for (unsigned i = 0; i < lengthCounts[width]; ++i) {
            lengths[ordered[orderedIndex++]] = static_cast<std::uint8_t>(width);
        }
    }
    return lengths;
}

struct CodeLengthToken {
    std::uint8_t symbol{};
    std::uint8_t extraValue{};
    std::uint8_t extraBits{};
};

/// RFC 1951 caps the combined literal and distance alphabets at 316 entries, so the run encoding cannot exceed 632 tokens
struct CodeLengthTokens {
    std::array<CodeLengthToken, 632> value{};
    std::size_t                      size{};

    constexpr void push(CodeLengthToken token) noexcept { value[size++] = token; }

    [[nodiscard]] constexpr std::span<const CodeLengthToken> tokens() const noexcept { return std::span<const CodeLengthToken>(value).first(size); }
};

[[nodiscard]] constexpr CodeLengthTokens encodeCodeLengths(std::span<const std::uint8_t> lengths) noexcept {
    CodeLengthTokens result{};
    std::size_t      pos = 0;
    while (pos < lengths.size()) {
        const auto  length = lengths[pos];
        std::size_t run    = 1;
        while (pos + run < lengths.size() && lengths[pos + run] == length) {
            ++run;
        }
        pos += run;
        if (length == 0U) {
            while (run >= 11UZ) {
                const auto count = std::min(run, 138UZ);
                result.push({.symbol = 18, .extraValue = static_cast<std::uint8_t>(count - 11UZ), .extraBits = 7});
                run -= count;
            }
            if (run >= 3UZ) {
                result.push({.symbol = 17, .extraValue = static_cast<std::uint8_t>(run - 3UZ), .extraBits = 3});
                run = 0;
            }
        } else {
            result.push({.symbol = length});
            --run;
            while (run >= 3UZ) {
                const auto count = std::min(run, 6UZ);
                result.push({.symbol = 16, .extraValue = static_cast<std::uint8_t>(count - 3UZ), .extraBits = 2});
                run -= count;
            }
        }
        for (std::size_t i = 0; i < run; ++i) {
            result.push({.symbol = length});
        }
    }
    return result;
}

struct DynamicCoding {
    CanonicalCodes<286> literals{};
    CanonicalCodes<30>  distances{};
    CanonicalCodes<19>  codeLengths{};
    CodeLengthTokens    encodedLengths{};
    std::uint16_t       literalCount{};
    std::uint8_t        distanceCount{};
    std::uint8_t        codeLengthCount{};
};

[[nodiscard]] constexpr DynamicCoding makeDynamicCoding(std::span<const Token> tokens) noexcept {
    std::array<std::uint32_t, 286> literalFrequencies{};
    std::array<std::uint32_t, 30>  distanceFrequencies{};
    for (const auto token : tokens) {
        if (token.isLiteral()) {
            ++literalFrequencies[token.value];
        } else {
            ++literalFrequencies[encodeLength(token.value).symbol];
            ++distanceFrequencies[encodeDistance(token.distance).symbol];
        }
    }
    ++literalFrequencies[256];

    DynamicCoding result{};
    result.literals  = makeCanonicalCodes(makeCodeLengths<286, 15>(literalFrequencies));
    result.distances = makeCanonicalCodes(makeCodeLengths<30, 15>(distanceFrequencies));

    std::size_t literalCount = 286;
    while (literalCount > 257UZ && result.literals.lengths[literalCount - 1UZ] == 0U) {
        --literalCount;
    }
    std::size_t distanceCount = 30;
    while (distanceCount > 1UZ && result.distances.lengths[distanceCount - 1UZ] == 0U) {
        --distanceCount;
    }
    result.literalCount  = static_cast<std::uint16_t>(literalCount);
    result.distanceCount = static_cast<std::uint8_t>(distanceCount);

    std::array<std::uint8_t, 316> combined{};
    std::ranges::copy_n(result.literals.lengths.begin(), static_cast<std::ptrdiff_t>(literalCount), combined.begin());
    std::ranges::copy_n(result.distances.lengths.begin(), static_cast<std::ptrdiff_t>(distanceCount), combined.begin() + static_cast<std::ptrdiff_t>(literalCount));
    result.encodedLengths = encodeCodeLengths(std::span<const std::uint8_t>(combined).first(literalCount + distanceCount));

    std::array<std::uint32_t, 19> codeLengthFrequencies{};
    for (const auto token : result.encodedLengths.tokens()) {
        ++codeLengthFrequencies[token.symbol];
    }
    result.codeLengths = makeCanonicalCodes(makeCodeLengths<19, 7>(codeLengthFrequencies));

    std::size_t codeLengthCount = 19;
    while (codeLengthCount > 4UZ && result.codeLengths.lengths[kCodeLengthOrder[codeLengthCount - 1UZ]] == 0U) {
        --codeLengthCount;
    }
    result.codeLengthCount = static_cast<std::uint8_t>(codeLengthCount);
    return result;
}

template<typename Writer, std::size_t N>
constexpr void writeCode(Writer& writer, const CanonicalCodes<N>& codes, unsigned symbol) {
    writer.writeBits(codes.bits[symbol], codes.lengths[symbol]);
}

template<typename Writer>
constexpr void emitTokensFixed(Writer& writer, std::span<const Token> tokens) {
    for (const auto token : tokens) {
        if (token.isLiteral()) {
            const auto code = fixedLiteralCode(token.value);
            writer.writeBits(code.bits, code.width);
            continue;
        }
        const auto length     = encodeLength(token.value);
        const auto lengthCode = fixedLiteralCode(length.symbol);
        writer.writeBits(lengthCode.bits, lengthCode.width);
        writer.writeBits(length.extraValue, length.extraBits);
        const auto distance     = encodeDistance(token.distance);
        const auto distanceCode = fixedDistanceCode(distance.symbol);
        writer.writeBits(distanceCode.bits, distanceCode.width);
        writer.writeBits(distance.extraValue, distance.extraBits);
    }
    const auto end = fixedLiteralCode(256);
    writer.writeBits(end.bits, end.width);
}

template<typename Writer>
constexpr void emitTokensDynamic(Writer& writer, std::span<const Token> tokens, const DynamicCoding& coding) {
    for (const auto token : tokens) {
        if (token.isLiteral()) {
            writeCode(writer, coding.literals, token.value);
            continue;
        }
        const auto length = encodeLength(token.value);
        writeCode(writer, coding.literals, length.symbol);
        writer.writeBits(length.extraValue, length.extraBits);
        const auto distance = encodeDistance(token.distance);
        writeCode(writer, coding.distances, distance.symbol);
        writer.writeBits(distance.extraValue, distance.extraBits);
    }
    writeCode(writer, coding.literals, 256);
}

template<typename Writer>
constexpr void emitStoredDeflate(Writer& writer, std::span<const std::byte> data, bool final = true) {
    std::size_t pos = 0;
    do {
        const auto size       = std::min(data.size() - pos, kMaxStoredBlock);
        const auto finalChunk = pos + size == data.size();
        writer.writeBits(final && finalChunk ? 1U : 0U, 1);
        writer.writeBits(0U, 2);
        writer.alignByte();
        putLittle16(writer, static_cast<std::uint16_t>(size));
        putLittle16(writer, static_cast<std::uint16_t>(~static_cast<std::uint16_t>(size)));
        writer.writeBytes(data.subspan(pos, size));
        pos += size;
    } while (pos < data.size());
}

template<typename Writer>
constexpr void emitFixedDeflate(Writer& writer, std::span<const Token> tokens, bool final = true) {
    writer.writeBits(final ? 1U : 0U, 1);
    writer.writeBits(1U, 2);
    emitTokensFixed(writer, tokens);
    if (final) {
        writer.alignByte();
    }
}

template<typename Writer>
constexpr void emitDynamicDeflate(Writer& writer, std::span<const Token> tokens, const DynamicCoding& coding, bool final = true) {
    writer.writeBits(final ? 1U : 0U, 1);
    writer.writeBits(2U, 2);
    writer.writeBits(coding.literalCount - 257U, 5);
    writer.writeBits(coding.distanceCount - 1U, 5);
    writer.writeBits(coding.codeLengthCount - 4U, 4);
    for (std::size_t i = 0; i < coding.codeLengthCount; ++i) {
        writer.writeBits(coding.codeLengths.lengths[kCodeLengthOrder[i]], 3);
    }
    for (const auto token : coding.encodedLengths.tokens()) {
        writeCode(writer, coding.codeLengths, token.symbol);
        writer.writeBits(token.extraValue, token.extraBits);
    }
    emitTokensDynamic(writer, tokens, coding);
    if (final) {
        writer.alignByte();
    }
}

enum class Encoding : std::uint8_t { stored, fixed, dynamic };

/// measures every block type through the same emit path that would write it, so size and output cannot drift apart
[[nodiscard]] constexpr Encoding selectEncoding(std::size_t dataSize, std::span<const Token> tokens, const DynamicCoding& coding, bool final = true) noexcept {
    BitCounter fixedSize{};
    emitFixedDeflate(fixedSize, tokens, final);
    BitCounter dynamicSize{};
    emitDynamicDeflate(dynamicSize, tokens, coding, final);
    const auto storedSize = dataSize + 5UZ * storedBlockCount(dataSize);

    if (storedSize < std::min(fixedSize.size(), dynamicSize.size())) {
        return Encoding::stored;
    }
    return dynamicSize.size() < fixedSize.size() ? Encoding::dynamic : Encoding::fixed;
}

template<typename Writer>
constexpr void emitDeflate(Writer& writer, std::span<const std::byte> data, std::span<const Token> tokens, const DynamicCoding& coding, Encoding encoding, bool final = true) {
    switch (encoding) {
    case Encoding::stored: emitStoredDeflate(writer, data, final); return;
    case Encoding::fixed: emitFixedDeflate(writer, tokens, final); return;
    case Encoding::dynamic: emitDynamicDeflate(writer, tokens, coding, final); return;
    }
    std::unreachable();
}

[[nodiscard]] constexpr std::size_t compressBound(std::size_t size, Format format) noexcept {
    const auto deflateBound = size + 5UZ * storedBlockCount(size);
    switch (format) {
    case Format::rawDeflate: return deflateBound;
    case Format::zlib: return deflateBound + 6UZ;
    case Format::gzip: return deflateBound + 18UZ;
    }
    std::unreachable();
}

[[nodiscard]] constexpr std::uint32_t computeCrc32(std::span<const std::byte> input) noexcept {
    if consteval {
        return gr::crc::computeScalar<gr::crc::Flavour::CRC32_IEEE>(input);
    } else {
        return gr::crc::compute<gr::crc::Flavour::CRC32_IEEE>(input);
    }
}

template<typename Writer>
constexpr void emitWrapped(Writer& writer, std::span<const std::byte> data, std::span<const Token> tokens, const DynamicCoding& coding, Encoding encoding, Format format, CompressionLevel level) {
    if (format == Format::zlib) {
        constexpr std::uint16_t kCmf   = 0x78U; // deflate, 32 KiB window
        const std::uint16_t     flevel = level == CompressionLevel::fast ? 0U : level == CompressionLevel::best ? 3U : 2U;
        std::uint16_t           header = static_cast<std::uint16_t>((kCmf << 8U) | (flevel << 6U));
        header                         = static_cast<std::uint16_t>(header + (31U - (header % 31U)) % 31U);
        writer.writeBits(header >> 8U, 8);
        writer.writeBits(header & 0xffU, 8);
        emitDeflate(writer, data, tokens, coding, encoding);
        putBig32(writer, adler32(data));
        return;
    }
    if (format == Format::gzip) {
        const auto xfl = level == CompressionLevel::fast ? 0x04U : level == CompressionLevel::best ? 0x02U : 0x00U;
        for (const auto byte : std::array<std::uint32_t, 10>{0x1fU, 0x8bU, 0x08U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, xfl, 0xffU}) {
            writer.writeBits(byte, 8);
        }
        emitDeflate(writer, data, tokens, coding, encoding);
        putLittle32(writer, computeCrc32(data));
        putLittle32(writer, static_cast<std::uint32_t>(data.size()));
        return;
    }
    emitDeflate(writer, data, tokens, coding, encoding);
}

template<typename Writer>
constexpr void compressInto(Writer& writer, std::span<const std::byte> data, Format format, CompressionLevel level, EncoderScratch scratch) {
    const auto tokens = tokenize(data, {}, level, scratch);
    const auto coding = makeDynamicCoding(tokens);
    emitWrapped(writer, data, tokens, coding, selectEncoding(data.size(), tokens, coding), format, level);
}

template<gr::meta::fixed_string S>
[[nodiscard]] consteval auto staticBytesFromString() {
    std::array<std::byte, S.size()> bytes{};
    for (std::size_t i = 0; i < S.size(); ++i) {
        bytes[i] = static_cast<std::byte>(static_cast<unsigned char>(S[i]));
    }
    return bytes;
}

struct BitReader {
    std::span<const std::byte> input;
    std::size_t                pos{};
    std::uint64_t              bits{};
    unsigned                   count{};

    [[nodiscard]] constexpr bool fill(unsigned width) noexcept {
        while (count < width) {
            if (pos == input.size()) {
                return false;
            }
            bits |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(input[pos++])) << count;
            count += 8U;
        }
        return true;
    }

    /// consumes one buffered bit; callers must `fill(1)` first
    [[nodiscard]] constexpr std::uint32_t takeBit() noexcept {
        const auto result = static_cast<std::uint32_t>(bits & 1U);
        bits >>= 1U;
        --count;
        return result;
    }

    [[nodiscard]] constexpr std::expected<std::uint32_t, Error> readBits(unsigned width) noexcept {
        if (width > 32U) {
            return std::unexpected(Error::invalidLength);
        }
        if (!fill(width)) {
            return std::unexpected(Error::truncatedInput);
        }
        const auto mask   = width == 32U ? 0xffff'ffffULL : ((std::uint64_t{1} << width) - 1U);
        const auto result = static_cast<std::uint32_t>(bits & mask);
        bits >>= width;
        count -= width;
        return result;
    }

    constexpr void alignByte() noexcept {
        const auto discard = count & 7U;
        bits >>= discard;
        count -= discard;
    }
};

template<std::size_t N, unsigned MaxBits>
struct HuffmanDecoder {
    std::array<std::uint16_t, MaxBits + 1UZ> counts{};
    std::array<std::uint16_t, N>             symbols{};
    std::uint8_t                             maxWidth{};
};

struct DecoderPolicy {
    bool allowEmpty{};
    bool allowIncompleteSingleBitCode = true;
};

/// mirrors zlib's inflate_table: incomplete literal/distance trees are allowed only for one one-bit code
template<std::size_t N, unsigned MaxBits>
[[nodiscard]] constexpr std::expected<HuffmanDecoder<N, MaxBits>, Error> makeDecoder(const std::array<std::uint8_t, N>& lengths, std::size_t symbolCount = N, DecoderPolicy policy = {}) noexcept {
    HuffmanDecoder<N, MaxBits> result{};
    std::size_t                used = 0;
    for (std::size_t symbol = 0; symbol < symbolCount; ++symbol) {
        const auto length = lengths[symbol];
        if (length > MaxBits) {
            return std::unexpected(Error::invalidHuffmanTree);
        }
        if (length != 0U) {
            ++result.counts[length];
            result.maxWidth = std::max(result.maxWidth, length);
            ++used;
        }
    }
    if (used == 0UZ) {
        return policy.allowEmpty ? std::expected<HuffmanDecoder<N, MaxBits>, Error>{result} : std::unexpected(Error::invalidHuffmanTree);
    }

    int remaining = 1;
    for (unsigned width = 1; width <= MaxBits; ++width) {
        remaining = (remaining << 1) - result.counts[width];
        if (remaining < 0) {
            return std::unexpected(Error::invalidHuffmanTree);
        }
    }
    if (remaining > 0 && (!policy.allowIncompleteSingleBitCode || result.maxWidth != 1U)) {
        return std::unexpected(Error::invalidHuffmanTree);
    }

    std::array<std::uint16_t, MaxBits + 1UZ> nextIndex{};
    std::uint16_t                            running = 0;
    for (unsigned width = 1; width <= MaxBits; ++width) {
        nextIndex[width] = running;
        running          = static_cast<std::uint16_t>(running + result.counts[width]);
    }
    for (std::size_t symbol = 0; symbol < symbolCount; ++symbol) {
        if (const auto width = lengths[symbol]; width != 0U) {
            result.symbols[nextIndex[width]++] = static_cast<std::uint16_t>(symbol);
        }
    }
    return result;
}

template<std::size_t N, unsigned MaxBits>
[[nodiscard]] constexpr std::expected<unsigned, Error> decodeSymbol(BitReader& reader, const HuffmanDecoder<N, MaxBits>& decoder) noexcept {
    std::uint32_t code  = 0;
    std::uint32_t first = 0;
    std::size_t   index = 0;
    for (unsigned width = 1; width <= decoder.maxWidth; ++width) {
        if (!reader.fill(1)) {
            return std::unexpected(Error::truncatedInput);
        }
        code             = (code << 1U) | reader.takeBit();
        const auto count = decoder.counts[width];
        if (code >= first && code - first < count) {
            return decoder.symbols[index + code - first];
        }
        index += count;
        first = (first + count) << 1U;
    }
    return std::unexpected(Error::invalidCode);
}

[[nodiscard]] consteval auto makeFixedLiteralDecoder() {
    std::array<std::uint8_t, 288> lengths{};
    std::ranges::fill(std::span{lengths}.subspan(0UZ, 144UZ), std::uint8_t{8});
    std::ranges::fill(std::span{lengths}.subspan(144UZ, 112UZ), std::uint8_t{9});
    std::ranges::fill(std::span{lengths}.subspan(256UZ, 24UZ), std::uint8_t{7});
    std::ranges::fill(std::span{lengths}.subspan(280UZ), std::uint8_t{8});
    return *makeDecoder<288, 15>(lengths);
}

[[nodiscard]] consteval auto makeFixedDistanceDecoder() {
    std::array<std::uint8_t, 32> lengths{};
    lengths.fill(5);
    return *makeDecoder<32, 15>(lengths);
}

inline constexpr auto kFixedLiteralDecoder  = makeFixedLiteralDecoder();
inline constexpr auto kFixedDistanceDecoder = makeFixedDistanceDecoder();

struct CountingOutput {
    std::size_t count{};
    std::size_t historyBegin{};
    std::size_t limit = std::numeric_limits<std::size_t>::max();

    [[nodiscard]] constexpr std::size_t size() const noexcept { return count; }
    constexpr void                      resetHistory() noexcept { historyBegin = count; }

    [[nodiscard]] constexpr std::expected<void, Error> append(std::byte) noexcept { return grow(1UZ); }

    [[nodiscard]] constexpr std::expected<void, Error> copy(std::size_t distance, std::size_t length) noexcept {
        if (distance == 0UZ || distance > count - historyBegin) {
            return std::unexpected(Error::invalidBackReference);
        }
        return grow(length);
    }

    [[nodiscard]] constexpr std::expected<void, Error> grow(std::size_t n) noexcept {
        if (n > std::numeric_limits<std::size_t>::max() - count) {
            return std::unexpected(Error::integerOverflow);
        }
        if (count + n > limit) {
            return std::unexpected(Error::sizeLimitExceeded);
        }
        count += n;
        return {};
    }
};

/// writes through a raw pointer: `std::span` subscripting costs several times more per byte during constant evaluation
struct SpanOutput {
    std::byte*  output{};
    std::size_t capacity{};
    std::size_t pos{};
    std::size_t historyBegin{};

    constexpr explicit SpanOutput(std::span<std::byte> target) noexcept : output(target.data()), capacity(target.size()) {}

    [[nodiscard]] constexpr std::size_t size() const noexcept { return pos; }
    constexpr void                      resetHistory() noexcept { historyBegin = pos; }

    [[nodiscard]] constexpr std::expected<void, Error> append(std::byte value) noexcept {
        if (pos == capacity) {
            return std::unexpected(Error::outputTooSmall);
        }
        output[pos++] = value;
        return {};
    }

    [[nodiscard]] constexpr std::expected<void, Error> copy(std::size_t distance, std::size_t length) noexcept {
        if (distance == 0UZ || distance > pos - historyBegin) {
            return std::unexpected(Error::invalidBackReference);
        }
        if (length > capacity - pos) {
            return std::unexpected(Error::outputTooSmall);
        }
        std::byte*       target = output + pos;
        const std::byte* source = target - distance;
        for (std::size_t i = 0; i < length; ++i) {
            target[i] = source[i];
        }
        pos += length;
        return {};
    }

    [[nodiscard]] constexpr std::span<const std::byte> range(std::size_t begin) const noexcept {
        const auto size = pos - begin;
        return size == 0UZ ? std::span<const std::byte>{} : std::span<const std::byte>{output + begin, size};
    }
};

template<typename Output, std::size_t LiteralSymbols, std::size_t DistanceSymbols>
[[nodiscard]] constexpr std::expected<void, Error> decodeCompressedBlock(BitReader& reader, Output& output, const HuffmanDecoder<LiteralSymbols, 15>& literals, const HuffmanDecoder<DistanceSymbols, 15>& distances) noexcept {
    while (true) {
        const auto symbol = decodeSymbol(reader, literals);
        if (!symbol) {
            return std::unexpected(symbol.error());
        }
        if (*symbol < 256U) {
            if (const auto result = output.append(static_cast<std::byte>(*symbol)); !result) {
                return result;
            }
            continue;
        }
        if (*symbol == 256U) {
            return {};
        }
        if (*symbol > 285U) {
            return std::unexpected(Error::invalidLength);
        }

        const auto  lengthIndex = *symbol - 257U;
        std::size_t length      = kLengthBase[lengthIndex];
        if (const auto extra = kLengthExtra[lengthIndex]; extra != 0U) {
            const auto value = reader.readBits(extra);
            if (!value) {
                return std::unexpected(value.error());
            }
            length += *value;
        }

        const auto distanceSymbol = decodeSymbol(reader, distances);
        if (!distanceSymbol) {
            return std::unexpected(distanceSymbol.error());
        }
        if (*distanceSymbol >= kDistanceBase.size()) {
            return std::unexpected(Error::invalidDistance);
        }
        std::size_t distance = kDistanceBase[*distanceSymbol];
        if (const auto extra = kDistanceExtra[*distanceSymbol]; extra != 0U) {
            const auto value = reader.readBits(extra);
            if (!value) {
                return std::unexpected(value.error());
            }
            distance += *value;
        }
        if (const auto result = output.copy(distance, length); !result) {
            return result;
        }
    }
}

template<typename Output>
[[nodiscard]] constexpr std::expected<void, Error> decodeDynamicBlock(BitReader& reader, Output& output) noexcept {
    const auto literalBits  = reader.readBits(5);
    const auto distanceBits = reader.readBits(5);
    const auto codeBits     = reader.readBits(4);
    if (!literalBits || !distanceBits || !codeBits) {
        return std::unexpected(Error::truncatedInput);
    }
    const auto literalCount    = static_cast<std::size_t>(*literalBits + 257U);
    const auto distanceCount   = static_cast<std::size_t>(*distanceBits + 1U);
    const auto codeLengthCount = static_cast<std::size_t>(*codeBits + 4U);
    if (literalCount > 286UZ || distanceCount > 30UZ) {
        return std::unexpected(Error::invalidHuffmanTree);
    }

    std::array<std::uint8_t, 19> codeLengthLengths{};
    for (std::size_t i = 0; i < codeLengthCount; ++i) {
        const auto length = reader.readBits(3);
        if (!length) {
            return std::unexpected(length.error());
        }
        codeLengthLengths[kCodeLengthOrder[i]] = static_cast<std::uint8_t>(*length);
    }
    const auto codeLengthDecoder = makeDecoder<19, 7>(codeLengthLengths, codeLengthLengths.size(), {.allowEmpty = false, .allowIncompleteSingleBitCode = false});
    if (!codeLengthDecoder) {
        return std::unexpected(codeLengthDecoder.error());
    }

    std::array<std::uint8_t, 320> lengths{};
    const auto                    total = literalCount + distanceCount;
    std::size_t                   index = 0;
    while (index < total) {
        const auto symbol = decodeSymbol(reader, *codeLengthDecoder);
        if (!symbol) {
            return std::unexpected(symbol.error());
        }
        if (*symbol <= 15U) {
            lengths[index++] = static_cast<std::uint8_t>(*symbol);
            continue;
        }

        std::uint8_t value = 0;
        unsigned     repeatBase{};
        unsigned     repeatBits{};
        if (*symbol == 16U) {
            if (index == 0UZ) {
                return std::unexpected(Error::invalidHuffmanTree);
            }
            value      = lengths[index - 1UZ];
            repeatBase = 3;
            repeatBits = 2;
        } else if (*symbol == 17U) {
            repeatBase = 3;
            repeatBits = 3;
        } else if (*symbol == 18U) {
            repeatBase = 11;
            repeatBits = 7;
        } else {
            return std::unexpected(Error::invalidCode);
        }
        const auto extra = reader.readBits(repeatBits);
        if (!extra) {
            return std::unexpected(extra.error());
        }
        const auto repeat = static_cast<std::size_t>(repeatBase + *extra);
        if (repeat > total - index) {
            return std::unexpected(Error::invalidHuffmanTree);
        }
        for (std::size_t i = 0; i < repeat; ++i) {
            lengths[index++] = value;
        }
    }

    std::array<std::uint8_t, 288> literalLengths{};
    std::array<std::uint8_t, 32>  distanceLengths{};
    std::ranges::copy_n(lengths.begin(), static_cast<std::ptrdiff_t>(literalCount), literalLengths.begin());
    std::ranges::copy_n(lengths.begin() + static_cast<std::ptrdiff_t>(literalCount), static_cast<std::ptrdiff_t>(distanceCount), distanceLengths.begin());
    if (literalLengths[256] == 0U) {
        return std::unexpected(Error::invalidHuffmanTree);
    }
    const auto literalDecoder = makeDecoder<288, 15>(literalLengths, literalCount);
    if (!literalDecoder) {
        return std::unexpected(literalDecoder.error());
    }
    const auto distanceDecoder = makeDecoder<32, 15>(distanceLengths, distanceCount, {.allowEmpty = true});
    if (!distanceDecoder) {
        return std::unexpected(distanceDecoder.error());
    }
    return decodeCompressedBlock(reader, output, *literalDecoder, *distanceDecoder);
}

/// whole-buffer inflate: `constexpr`, so it also backs the compile-time round-trip proof, and several times
/// faster than the resumable StreamInflate below, which exists for chunked input. Both share the tables,
/// `makeDecoder` and its DecoderPolicy, so their acceptance rules cannot drift apart.
template<typename Output>
[[nodiscard]] constexpr std::expected<std::size_t, Error> decodeDeflate(std::span<const std::byte> input, Output& output) noexcept {
    output.resetHistory();
    BitReader reader{.input = input};
    bool      final = false;
    while (!final) {
        const auto finalBit = reader.readBits(1);
        const auto block    = reader.readBits(2);
        if (!finalBit || !block) {
            return std::unexpected(Error::truncatedInput);
        }
        final = *finalBit != 0U;
        switch (*block) {
        case 0: {
            reader.alignByte();
            const auto lengthValue     = reader.readBits(16);
            const auto complementValue = reader.readBits(16);
            if (!lengthValue || !complementValue) {
                return std::unexpected(Error::truncatedInput);
            }
            const auto length = static_cast<std::uint16_t>(*lengthValue);
            if (static_cast<std::uint16_t>(~length) != static_cast<std::uint16_t>(*complementValue)) {
                return std::unexpected(Error::invalidStoredBlockLength);
            }
            for (std::size_t i = 0; i < length; ++i) {
                const auto value = reader.readBits(8);
                if (!value) {
                    return std::unexpected(value.error());
                }
                if (const auto result = output.append(static_cast<std::byte>(*value)); !result) {
                    return std::unexpected(result.error());
                }
            }
            break;
        }
        case 1:
            if (const auto result = decodeCompressedBlock(reader, output, kFixedLiteralDecoder, kFixedDistanceDecoder); !result) {
                return std::unexpected(result.error());
            }
            break;
        case 2:
            if (const auto result = decodeDynamicBlock(reader, output); !result) {
                return std::unexpected(result.error());
            }
            break;
        default: return std::unexpected(Error::invalidBlockType);
        }
    }
    reader.alignByte();
    return reader.pos;
}

[[nodiscard]] constexpr std::uint16_t readLittle16(std::span<const std::byte> input, std::size_t pos) noexcept { return static_cast<std::uint16_t>(std::to_integer<std::uint16_t>(input[pos]) | (std::to_integer<std::uint16_t>(input[pos + 1UZ]) << 8U)); }
[[nodiscard]] constexpr std::uint32_t readLittle32(std::span<const std::byte> input, std::size_t pos) noexcept { return std::to_integer<std::uint32_t>(input[pos]) | (std::to_integer<std::uint32_t>(input[pos + 1UZ]) << 8U) | (std::to_integer<std::uint32_t>(input[pos + 2UZ]) << 16U) | (std::to_integer<std::uint32_t>(input[pos + 3UZ]) << 24U); }
[[nodiscard]] constexpr std::uint32_t readBig32(std::span<const std::byte> input, std::size_t pos) noexcept { return (std::to_integer<std::uint32_t>(input[pos]) << 24U) | (std::to_integer<std::uint32_t>(input[pos + 1UZ]) << 16U) | (std::to_integer<std::uint32_t>(input[pos + 2UZ]) << 8U) | std::to_integer<std::uint32_t>(input[pos + 3UZ]); }

template<typename Output>
[[nodiscard]] constexpr std::expected<std::size_t, Error> decodeRaw(std::span<const std::byte> input, Output& output) noexcept {
    const auto consumed = decodeDeflate(input, output);
    if (!consumed) {
        return std::unexpected(consumed.error());
    }
    if (*consumed != input.size()) {
        return std::unexpected(Error::trailingData);
    }
    return output.size();
}

template<typename Output>
[[nodiscard]] constexpr std::expected<std::size_t, Error> decodeZlib(std::span<const std::byte> input, Output& output) noexcept {
    if (input.size() < 2UZ) {
        return std::unexpected(Error::truncatedInput);
    }
    const auto cmf = std::to_integer<std::uint8_t>(input[0]);
    const auto flg = std::to_integer<std::uint8_t>(input[1]);
    if ((cmf & 0x0fU) != 8U || (cmf >> 4U) > 7U || ((static_cast<unsigned>(cmf) << 8U) + flg) % 31U != 0U) {
        return std::unexpected(Error::invalidHeader);
    }
    if ((flg & 0x20U) != 0U) {
        if (input.size() < 6UZ) {
            return std::unexpected(Error::truncatedInput);
        }
        return std::unexpected(Error::presetDictionaryRequired);
    }

    const auto outputBegin = output.size();
    const auto consumed    = decodeDeflate(input.subspan(2), output);
    if (!consumed) {
        return std::unexpected(consumed.error());
    }
    const auto trailer = 2UZ + *consumed;
    if (input.size() < trailer + 4UZ) {
        return std::unexpected(Error::truncatedInput);
    }
    if (input.size() != trailer + 4UZ) {
        return std::unexpected(Error::trailingData);
    }
    if constexpr (requires { output.range(outputBegin); }) {
        if (adler32(output.range(outputBegin)) != readBig32(input, trailer)) {
            return std::unexpected(Error::checksumMismatch);
        }
    }
    return output.size();
}

template<typename Output>
[[nodiscard]] constexpr std::expected<std::size_t, Error> decodeGzip(std::span<const std::byte> input, Output& output) noexcept {
    std::size_t pos = 0;
    if (input.empty()) {
        return std::unexpected(Error::truncatedInput);
    }
    while (pos < input.size()) {
        const auto memberStart = pos;
        if (input.size() - pos < 2UZ) {
            return memberStart == 0UZ || input[pos] == std::byte{0x1f} ? std::unexpected(Error::truncatedInput) : std::unexpected(Error::trailingData);
        }
        if (input[pos] != std::byte{0x1f} || input[pos + 1UZ] != std::byte{0x8b}) {
            return memberStart == 0UZ ? std::unexpected(Error::invalidHeader) : std::unexpected(Error::trailingData);
        }
        if (input.size() - pos < 10UZ) {
            return std::unexpected(Error::truncatedInput);
        }
        if (input[pos + 2UZ] != std::byte{0x08}) {
            return std::unexpected(Error::unsupportedCompressionMethod);
        }
        const auto flags = std::to_integer<std::uint8_t>(input[pos + 3UZ]);
        if ((flags & 0xe0U) != 0U) {
            return std::unexpected(Error::reservedFlags);
        }
        pos += 10UZ;

        if ((flags & 0x04U) != 0U) {
            if (input.size() - pos < 2UZ) {
                return std::unexpected(Error::truncatedInput);
            }
            const auto extraLength = readLittle16(input, pos);
            pos += 2UZ;
            if (input.size() - pos < extraLength) {
                return std::unexpected(Error::truncatedInput);
            }
            pos += extraLength;
        }
        for (const auto flag : {0x08U, 0x10U}) {
            if ((flags & flag) == 0U) {
                continue;
            }
            const auto terminator = std::ranges::find(input.subspan(pos), std::byte{0x00});
            if (terminator == input.end()) {
                return std::unexpected(Error::truncatedInput);
            }
            pos = static_cast<std::size_t>(terminator - input.begin()) + 1UZ;
        }
        if ((flags & 0x02U) != 0U) {
            if (input.size() - pos < 2UZ) {
                return std::unexpected(Error::truncatedInput);
            }
            const auto expected = readLittle16(input, pos);
            const auto actual   = static_cast<std::uint16_t>(computeCrc32(input.subspan(memberStart, pos - memberStart)) & 0xffffU);
            if (expected != actual) {
                return std::unexpected(Error::checksumMismatch);
            }
            pos += 2UZ;
        }

        const auto outputBegin = output.size();
        const auto consumed    = decodeDeflate(input.subspan(pos), output);
        if (!consumed) {
            return std::unexpected(consumed.error());
        }
        pos += *consumed;
        if (input.size() - pos < 8UZ) {
            return std::unexpected(Error::truncatedInput);
        }
        if constexpr (requires { output.range(outputBegin); }) {
            if (computeCrc32(output.range(outputBegin)) != readLittle32(input, pos)) {
                return std::unexpected(Error::checksumMismatch);
            }
        }
        if (static_cast<std::uint32_t>(output.size() - outputBegin) != readLittle32(input, pos + 4UZ)) {
            return std::unexpected(Error::sizeMismatch);
        }
        pos += 8UZ;
    }
    return output.size();
}

template<typename Output>
[[nodiscard]] constexpr std::expected<std::size_t, Error> decompressInto(std::span<const std::byte> input, Output& output, Format format) noexcept {
    switch (format) {
    case Format::rawDeflate: return decodeRaw(input, output);
    case Format::zlib: return decodeZlib(input, output);
    case Format::gzip: return decodeGzip(input, output);
    }
    std::unreachable();
}

enum class StreamProgress : std::uint8_t { advanced, needInput, needOutput, finished };

struct StreamBitReader {
    std::span<const std::byte> input{};
    std::size_t                pos{};
    std::uint64_t              bits{};
    unsigned                   count{};

    void setInput(std::span<const std::byte> source) noexcept {
        input = source;
        pos   = 0UZ;
    }

    void clearInput() noexcept { input = {}; }

    [[nodiscard]] bool fill(unsigned width) noexcept {
        while (count < width) {
            if (pos == input.size()) {
                return false;
            }
            bits |= static_cast<std::uint64_t>(std::to_integer<std::uint8_t>(input[pos++])) << count;
            count += 8U;
        }
        return true;
    }

    /// the three scalars are the whole resumable state; saving them is cheaper than copying the reader
    struct Mark {
        std::size_t   pos{};
        std::uint64_t bits{};
        unsigned      count{};
    };

    [[nodiscard]] Mark mark() const noexcept { return {.pos = pos, .bits = bits, .count = count}; }

    void restore(Mark saved) noexcept {
        pos   = saved.pos;
        bits  = saved.bits;
        count = saved.count;
    }

    [[nodiscard]] std::expected<std::optional<std::uint32_t>, Error> readBits(unsigned width, bool finish) noexcept {
        if (width > 32U) {
            return std::unexpected(Error::invalidLength);
        }
        const auto saved = mark();
        if (!fill(width)) {
            restore(saved);
            if (finish) {
                return std::unexpected(Error::truncatedInput);
            }
            return std::optional<std::uint32_t>{};
        }
        const auto mask   = width == 32U ? 0xffff'ffffULL : ((std::uint64_t{1} << width) - 1U);
        const auto result = static_cast<std::uint32_t>(bits & mask);
        bits >>= width;
        count -= width;
        return result;
    }

    [[nodiscard]] std::uint32_t takeBit() noexcept {
        const auto result = static_cast<std::uint32_t>(bits & 1U);
        bits >>= 1U;
        --count;
        return result;
    }

    void alignByte() noexcept {
        const auto discard = count & 7U;
        bits >>= discard;
        count -= discard;
    }
};

template<std::size_t N, unsigned MaxBits>
[[nodiscard]] std::expected<std::optional<unsigned>, Error> decodeSymbol(StreamBitReader& reader, const HuffmanDecoder<N, MaxBits>& decoder, bool finish) noexcept {
    const auto    saved = reader.mark();
    std::uint32_t code  = 0;
    std::uint32_t first = 0;
    std::size_t   index = 0;
    for (unsigned width = 1; width <= decoder.maxWidth; ++width) {
        if (!reader.fill(1)) {
            reader.restore(saved);
            if (finish) {
                return std::unexpected(Error::truncatedInput);
            }
            return std::optional<unsigned>{};
        }
        code             = (code << 1U) | reader.takeBit();
        const auto count = decoder.counts[width];
        if (code >= first && code - first < count) {
            return static_cast<unsigned>(decoder.symbols[index + code - first]);
        }
        index += count;
        first = (first + count) << 1U;
    }
    reader.restore(saved);
    return std::unexpected(Error::invalidCode);
}

struct StreamOutput {
    static constexpr std::size_t kWindowMask = kWindowSize - 1UZ; // kWindowSize is a power of two

    std::array<std::byte, kWindowSize> history{};
    std::span<std::byte>               target{};
    std::size_t                        produced{};
    std::size_t                        checksumPos{};
    std::size_t                        historyPos{};
    std::size_t                        historySize{};
    std::size_t                        total{};
    std::size_t                        limit = std::numeric_limits<std::size_t>::max();
    std::uint32_t                      crcState{};
    std::uint32_t                      adlerA = 1U;
    std::uint32_t                      adlerB{};
    std::uint32_t                      memberSize{};

    void reset(std::size_t maxOutputSize) noexcept {
        target      = {};
        produced    = 0UZ;
        checksumPos = 0UZ;
        total       = 0UZ;
        limit       = maxOutputSize;
        adlerA      = 1U;
        adlerB      = 0U;
        memberSize  = 0U;
        crcState    = 0xffff'ffffU;
        historyPos  = 0UZ;
        historySize = 0UZ;
    }

    void setTarget(std::span<std::byte> output) noexcept {
        target      = output;
        produced    = 0UZ;
        checksumPos = 0UZ;
    }

    /// checksums are folded in whole spans rather than per byte, so this must run before the target goes away
    void flushChecksums() noexcept {
        if (checksumPos == produced) {
            return;
        }
        const auto fresh         = target.subspan(checksumPos, produced - checksumPos);
        crcState                 = gr::crc::updateScalar<gr::crc::Flavour::CRC32_IEEE>(crcState, fresh);
        std::tie(adlerA, adlerB) = updateAdler(adlerA, adlerB, fresh);
        memberSize += static_cast<std::uint32_t>(fresh.size());
        checksumPos = produced;
    }

    void clearTarget() noexcept {
        flushChecksums();
        target = {};
    }

    void resetHistory() noexcept {
        historyPos  = 0UZ;
        historySize = 0UZ;
    }

    void resetGzipMember() noexcept {
        flushChecksums();
        resetHistory();
        crcState   = 0xffff'ffffU;
        memberSize = 0U;
    }

    [[nodiscard]] bool full() const noexcept { return produced == target.size(); }

    [[nodiscard]] std::uint32_t crc32() noexcept {
        flushChecksums();
        return crcState ^ 0xffff'ffffU;
    }

    [[nodiscard]] std::uint32_t adler32Value() noexcept {
        flushChecksums();
        return (adlerB << 16U) | adlerA;
    }

    [[nodiscard]] std::uint32_t memberBytes() noexcept {
        flushChecksums();
        return memberSize;
    }

    [[nodiscard]] std::expected<bool, Error> append(std::byte byte) noexcept {
        if (full()) {
            return false;
        }
        if (total == limit) {
            return std::unexpected(Error::sizeLimitExceeded);
        }
        target[produced++]  = byte;
        history[historyPos] = byte;
        historyPos          = (historyPos + 1UZ) & kWindowMask;
        historySize         = std::min(historySize + 1UZ, kWindowSize);
        ++total;
        return true;
    }

    [[nodiscard]] std::expected<bool, Error> copy(std::size_t distance, std::size_t& length) noexcept {
        if (distance == 0UZ || distance > historySize) {
            return std::unexpected(Error::invalidBackReference);
        }
        while (length != 0UZ) {
            if (full()) {
                return false;
            }
            const auto value = history[(historyPos + kWindowSize - distance) & kWindowMask];
            if (const auto appended = append(value); !appended) {
                return std::unexpected(appended.error());
            }
            --length;
        }
        return true;
    }
};

struct StreamInflate {
    enum class Mode : std::uint8_t { header, storedLength, storedData, dynamicHeader, dynamicCodeLengths, dynamicLengths, compressed, done };

    Mode                          mode = Mode::header;
    bool                          finalBlock{};
    std::uint16_t                 storedRemaining{};
    HuffmanDecoder<288, 15>       literalDecoder{};
    HuffmanDecoder<32, 15>        distanceDecoder{};
    HuffmanDecoder<19, 7>         codeLengthDecoder{};
    std::array<std::uint8_t, 19>  codeLengthLengths{};
    std::array<std::uint8_t, 320> dynamicLengths{};
    std::size_t                   literalCount{};
    std::size_t                   distanceCount{};
    std::size_t                   codeLengthCount{};
    std::size_t                   dynamicIndex{};
    std::size_t                   dynamicCodeLengthIndex{};
    std::size_t                   pendingDistance{};
    std::size_t                   pendingLength{};

    void reset() noexcept {
        mode                   = Mode::header;
        finalBlock             = false;
        storedRemaining        = 0U;
        literalDecoder         = {};
        distanceDecoder        = {};
        codeLengthDecoder      = {};
        codeLengthLengths      = {};
        dynamicLengths         = {};
        literalCount           = 0UZ;
        distanceCount          = 0UZ;
        codeLengthCount        = 0UZ;
        dynamicIndex           = 0UZ;
        dynamicCodeLengthIndex = 0UZ;
        pendingDistance        = 0UZ;
        pendingLength          = 0UZ;
    }

    [[nodiscard]] std::expected<StreamProgress, Error> process(StreamBitReader& reader, StreamOutput& output, bool finish) noexcept {
        while (true) {
            if (pendingLength != 0UZ) {
                const auto copied = output.copy(pendingDistance, pendingLength);
                if (!copied) {
                    return std::unexpected(copied.error());
                }
                if (!*copied) {
                    return StreamProgress::needOutput;
                }
                continue;
            }

            switch (mode) {
            case Mode::done: return StreamProgress::finished;
            case Mode::header: {
                const auto bits = reader.readBits(3, finish);
                if (!bits) {
                    return std::unexpected(bits.error());
                }
                if (!*bits) {
                    return StreamProgress::needInput;
                }
                finalBlock = (**bits & 1U) != 0U;
                switch (**bits >> 1U) {
                case 0U:
                    reader.alignByte();
                    mode = Mode::storedLength;
                    break;
                case 1U:
                    literalDecoder  = kFixedLiteralDecoder;
                    distanceDecoder = kFixedDistanceDecoder;
                    mode            = Mode::compressed;
                    break;
                case 2U: mode = Mode::dynamicHeader; break;
                default: return std::unexpected(Error::invalidBlockType);
                }
                break;
            }
            case Mode::storedLength: {
                const auto value = reader.readBits(32, finish);
                if (!value) {
                    return std::unexpected(value.error());
                }
                if (!*value) {
                    return StreamProgress::needInput;
                }
                storedRemaining       = static_cast<std::uint16_t>(**value);
                const auto complement = static_cast<std::uint16_t>(**value >> 16U);
                if (static_cast<std::uint16_t>(~storedRemaining) != complement) {
                    return std::unexpected(Error::invalidStoredBlockLength);
                }
                mode = Mode::storedData;
                break;
            }
            case Mode::storedData:
                if (storedRemaining == 0U) {
                    if (finalBlock) {
                        mode = Mode::done;
                        return StreamProgress::finished;
                    }
                    mode = Mode::header;
                    break;
                }
                if (output.full()) {
                    return StreamProgress::needOutput;
                }
                if (const auto value = reader.readBits(8, finish); !value) {
                    return std::unexpected(value.error());
                } else if (!*value) {
                    return StreamProgress::needInput;
                } else if (const auto appended = output.append(static_cast<std::byte>(**value)); !appended) {
                    return std::unexpected(appended.error());
                } else {
                    --storedRemaining;
                }
                break;
            case Mode::dynamicHeader:
                if (const auto result = readDynamicHeader(reader, finish); !result || *result != StreamProgress::advanced) {
                    return result;
                }
                break;
            case Mode::dynamicCodeLengths:
                if (const auto result = readDynamicCodeLengths(reader, finish); !result || *result != StreamProgress::advanced) {
                    return result;
                }
                break;
            case Mode::dynamicLengths:
                if (const auto result = readDynamicLengths(reader, finish); !result || *result != StreamProgress::advanced) {
                    return result;
                }
                break;
            case Mode::compressed:
                if (const auto result = readCompressedSymbol(reader, output, finish); !result || *result != StreamProgress::advanced) {
                    return result;
                }
                break;
            }
        }
    }

private:
    [[nodiscard]] std::expected<StreamProgress, Error> readDynamicHeader(StreamBitReader& reader, bool finish) noexcept {
        const auto bits = reader.readBits(14, finish);
        if (!bits) {
            return std::unexpected(bits.error());
        }
        if (!*bits) {
            return StreamProgress::needInput;
        }
        literalCount    = static_cast<std::size_t>((**bits & 0x1fU) + 257U);
        distanceCount   = static_cast<std::size_t>(((**bits >> 5U) & 0x1fU) + 1U);
        codeLengthCount = static_cast<std::size_t>(((**bits >> 10U) & 0x0fU) + 4U);
        if (literalCount > 286UZ || distanceCount > 30UZ) {
            return std::unexpected(Error::invalidHuffmanTree);
        }
        codeLengthLengths      = {};
        dynamicLengths         = {};
        dynamicIndex           = 0UZ;
        dynamicCodeLengthIndex = 0UZ;
        mode                   = Mode::dynamicCodeLengths;
        return StreamProgress::advanced;
    }

    [[nodiscard]] std::expected<StreamProgress, Error> readDynamicCodeLengths(StreamBitReader& reader, bool finish) noexcept {
        while (dynamicCodeLengthIndex < codeLengthCount) {
            const auto length = reader.readBits(3, finish);
            if (!length) {
                return std::unexpected(length.error());
            }
            if (!*length) {
                return StreamProgress::needInput;
            }
            codeLengthLengths[kCodeLengthOrder[dynamicCodeLengthIndex++]] = static_cast<std::uint8_t>(**length);
        }
        const auto decoder = makeDecoder<19, 7>(codeLengthLengths, codeLengthLengths.size(), {.allowEmpty = false, .allowIncompleteSingleBitCode = false});
        if (!decoder) {
            return std::unexpected(decoder.error());
        }
        codeLengthDecoder = *decoder;
        mode              = Mode::dynamicLengths;
        return StreamProgress::advanced;
    }

    [[nodiscard]] std::expected<StreamProgress, Error> readDynamicLengths(StreamBitReader& reader, bool finish) noexcept {
        const auto total = literalCount + distanceCount;
        while (dynamicIndex < total) {
            const auto before = reader;
            const auto symbol = decodeSymbol(reader, codeLengthDecoder, finish);
            if (!symbol) {
                return std::unexpected(symbol.error());
            }
            if (!*symbol) {
                return StreamProgress::needInput;
            }
            if (**symbol <= 15U) {
                dynamicLengths[dynamicIndex++] = static_cast<std::uint8_t>(**symbol);
                continue;
            }

            std::uint8_t value{};
            unsigned     repeatBase{};
            unsigned     repeatBits{};
            if (**symbol == 16U) {
                if (dynamicIndex == 0UZ) {
                    return std::unexpected(Error::invalidHuffmanTree);
                }
                value      = dynamicLengths[dynamicIndex - 1UZ];
                repeatBase = 3U;
                repeatBits = 2U;
            } else if (**symbol == 17U) {
                repeatBase = 3U;
                repeatBits = 3U;
            } else if (**symbol == 18U) {
                repeatBase = 11U;
                repeatBits = 7U;
            } else {
                return std::unexpected(Error::invalidCode);
            }

            const auto extra = reader.readBits(repeatBits, finish);
            if (!extra) {
                return std::unexpected(extra.error());
            }
            if (!*extra) {
                reader = before;
                return StreamProgress::needInput;
            }
            const auto repeat = static_cast<std::size_t>(repeatBase + **extra);
            if (repeat > total - dynamicIndex) {
                return std::unexpected(Error::invalidHuffmanTree);
            }
            std::ranges::fill(std::span(dynamicLengths).subspan(dynamicIndex, repeat), value);
            dynamicIndex += repeat;
        }

        std::array<std::uint8_t, 288> literalLengths{};
        std::array<std::uint8_t, 32>  distanceLengths{};
        std::ranges::copy_n(dynamicLengths.begin(), static_cast<std::ptrdiff_t>(literalCount), literalLengths.begin());
        std::ranges::copy_n(dynamicLengths.begin() + static_cast<std::ptrdiff_t>(literalCount), static_cast<std::ptrdiff_t>(distanceCount), distanceLengths.begin());
        if (literalLengths[256] == 0U) {
            return std::unexpected(Error::invalidHuffmanTree);
        }
        const auto literals = makeDecoder<288, 15>(literalLengths, literalCount);
        if (!literals) {
            return std::unexpected(literals.error());
        }
        const auto distances = makeDecoder<32, 15>(distanceLengths, distanceCount, {.allowEmpty = true});
        if (!distances) {
            return std::unexpected(distances.error());
        }
        literalDecoder  = *literals;
        distanceDecoder = *distances;
        mode            = Mode::compressed;
        return StreamProgress::advanced;
    }

    [[nodiscard]] std::expected<StreamProgress, Error> readCompressedSymbol(StreamBitReader& reader, StreamOutput& output, bool finish) noexcept {
        const auto before = reader;
        const auto symbol = decodeSymbol(reader, literalDecoder, finish);
        if (!symbol) {
            return std::unexpected(symbol.error());
        }
        if (!*symbol) {
            return StreamProgress::needInput;
        }
        if (**symbol < 256U) {
            if (output.full()) {
                reader = before;
                return StreamProgress::needOutput;
            }
            if (const auto appended = output.append(static_cast<std::byte>(**symbol)); !appended) {
                return std::unexpected(appended.error());
            }
            return StreamProgress::advanced;
        }
        if (**symbol == 256U) {
            if (finalBlock) {
                reader.alignByte();
                mode = Mode::done;
                return StreamProgress::finished;
            }
            mode = Mode::header;
            return StreamProgress::advanced;
        }
        if (**symbol > 285U) {
            return std::unexpected(Error::invalidLength);
        }
        if (output.full()) {
            reader = before;
            return StreamProgress::needOutput;
        }

        std::size_t length      = kLengthBase[**symbol - 257U];
        const auto  lengthExtra = kLengthExtra[**symbol - 257U];
        if (lengthExtra != 0U) {
            const auto value = reader.readBits(lengthExtra, finish);
            if (!value) {
                return std::unexpected(value.error());
            }
            if (!*value) {
                reader = before;
                return StreamProgress::needInput;
            }
            length += **value;
        }

        const auto distanceSymbol = decodeSymbol(reader, distanceDecoder, finish);
        if (!distanceSymbol) {
            return std::unexpected(distanceSymbol.error());
        }
        if (!*distanceSymbol) {
            reader = before;
            return StreamProgress::needInput;
        }
        if (**distanceSymbol >= kDistanceBase.size()) {
            return std::unexpected(Error::invalidDistance);
        }
        std::size_t distance      = kDistanceBase[**distanceSymbol];
        const auto  distanceExtra = kDistanceExtra[**distanceSymbol];
        if (distanceExtra != 0U) {
            const auto value = reader.readBits(distanceExtra, finish);
            if (!value) {
                return std::unexpected(value.error());
            }
            if (!*value) {
                reader = before;
                return StreamProgress::needInput;
            }
            distance += **value;
        }
        pendingDistance = distance;
        pendingLength   = length;
        return StreamProgress::advanced;
    }
};

} // namespace detail

[[nodiscard]] constexpr std::expected<std::size_t, Error> decompress(std::span<const std::byte> input, std::span<std::byte> output, Format format) noexcept {
    detail::SpanOutput writer{output};
    return detail::decompressInto(input, writer, format);
}

/// returns the exact decoded size after validating the stream
[[nodiscard]] constexpr std::expected<std::size_t, Error> decompressedSize(std::span<const std::byte> input, Format format, std::size_t maxOutputSize = std::numeric_limits<std::size_t>::max()) noexcept {
    detail::CountingOutput counter{};
    counter.limit = maxOutputSize;
    return detail::decompressInto(input, counter, format);
}

namespace detail {

template<typename Vector>
[[nodiscard]] std::expected<Vector, Error> decompressToVector(std::span<const std::byte> input, Format format, std::size_t maxOutputSize, typename Vector::allocator_type allocator) noexcept {
    CountingOutput counter{};
    counter.limit      = maxOutputSize;
    const auto counted = decompressInto(input, counter, format);
    if (!counted) {
        return std::unexpected(counted.error());
    }
#if __cpp_exceptions
    try {
#endif
        Vector     output(*counted, allocator);
        const auto written = gr::compression::decompress(input, std::span<std::byte>(output), format);
        if (!written) {
            return std::unexpected(written.error());
        }
        if (*written != output.size()) {
            return std::unexpected(Error::sizeMismatch);
        }
        return output;
#if __cpp_exceptions
    } catch (const std::bad_alloc&) {
        return std::unexpected(Error::allocationFailed);
    } catch (const std::length_error&) {
        return std::unexpected(Error::allocationFailed);
    }
#endif // -fno-exceptions: an allocation failure faults hard instead of reporting Error::allocationFailed
}

} // namespace detail

[[nodiscard]] inline std::expected<std::vector<std::byte>, Error> decompress(std::span<const std::byte> input, Format format, std::size_t maxOutputSize = kDefaultMaxDecompressedSize) noexcept { return detail::decompressToVector<std::vector<std::byte>>(input, format, maxOutputSize, {}); }

[[nodiscard]] inline std::expected<std::pmr::vector<std::byte>, Error> decompress(std::span<const std::byte> input, Format format, std::pmr::memory_resource* resource, std::size_t maxOutputSize = kDefaultMaxDecompressedSize) noexcept { return detail::decompressToVector<std::pmr::vector<std::byte>>(input, format, maxOutputSize, resource != nullptr ? resource : std::pmr::get_default_resource()); }

namespace detail {

template<typename Vector>
[[nodiscard]] std::expected<Vector, Error> compressToVector(std::span<const std::byte> input, Format format, CompressionLevel level, typename Vector::allocator_type allocator) noexcept {
#if __cpp_exceptions
    try {
#endif
        std::vector<Token>       tokens(input.size());
        std::vector<std::size_t> head(kHashSize);
        std::vector<std::size_t> previous(historyBound(input.size()));

        Vector              output(compressBound(input.size(), format), allocator);
        ByteSink            sink{output};
        BitWriter<ByteSink> writer{sink};
        compressInto(writer, input, format, level, {.tokens = tokens, .head = head, .previous = previous});
        if (sink.overflowed()) {
            return std::unexpected(Error::outputTooSmall);
        }
        output.resize(writer.size());
        return output;
#if __cpp_exceptions
    } catch (const std::bad_alloc&) {
        return std::unexpected(Error::allocationFailed);
    } catch (const std::length_error&) {
        return std::unexpected(Error::allocationFailed);
    }
#endif // -fno-exceptions: an allocation failure faults hard instead of reporting Error::allocationFailed
}

} // namespace detail

[[nodiscard]] inline std::expected<std::vector<std::byte>, Error> compress(std::span<const std::byte> input, Format format = Format::gzip, CompressionLevel level = CompressionLevel::balanced) noexcept { return detail::compressToVector<std::vector<std::byte>>(input, format, level, {}); }

/// scratch tables use the default resource; only the returned bytes live in `resource`
[[nodiscard]] inline std::expected<std::pmr::vector<std::byte>, Error> compress(std::span<const std::byte> input, std::pmr::memory_resource* resource, Format format = Format::gzip, CompressionLevel level = CompressionLevel::balanced) noexcept { return detail::compressToVector<std::pmr::vector<std::byte>>(input, format, level, resource != nullptr ? resource : std::pmr::get_default_resource()); }

class StreamDecompressor {
public:
    explicit StreamDecompressor(Format format = Format::gzip, std::size_t maxOutputSize = kDefaultMaxDecompressedSize) noexcept { reset(format, maxOutputSize); }

    void reset(Format format = Format::gzip, std::size_t maxOutputSize = kDefaultMaxDecompressedSize) noexcept {
        _format = format;
        _reader = {};
        _output.reset(maxOutputSize);
        _inflate.reset();
        _state              = format == Format::rawDeflate ? State::rawDeflate : format == Format::zlib ? State::zlibHeader : State::gzipHeader;
        _zlibHeader         = {};
        _zlibHeaderPos      = 0UZ;
        _trailer            = {};
        _trailerPos         = 0UZ;
        _gzipHeader         = {};
        _gzipHeaderPos      = 0UZ;
        _gzipFlags          = 0U;
        _gzipExtraRemaining = 0UZ;
        _gzipHeaderCrcState = 0xffff'ffffU;
        _gzipMemberCount    = 0UZ;
    }

    [[nodiscard]] bool finished() const noexcept { return _state == State::done; }

    [[nodiscard]] std::expected<StreamResult, Error> process(std::span<const std::byte> input, std::span<std::byte> output, Flush flush = Flush::none) noexcept {
        _reader.setInput(input);
        _output.setTarget(output);
        const auto finish = flush == Flush::finish;

        const auto result = [this](detail::StreamProgress progress) noexcept {
            const auto   status = _state == State::done ? StreamStatus::done : progress == detail::StreamProgress::needOutput ? StreamStatus::needOutput : StreamStatus::needInput;
            StreamResult value{.consumed = _reader.pos, .produced = _output.produced, .status = status};
            _reader.clearInput();
            _output.clearTarget();
            return value;
        };
        const auto fail = [this](Error error) noexcept -> std::expected<StreamResult, Error> {
            _reader.clearInput();
            _output.clearTarget();
            return std::unexpected(error);
        };

        while (true) {
            switch (_state) {
            case State::done:
                if (_reader.pos != input.size()) {
                    return fail(Error::trailingData);
                }
                return result(detail::StreamProgress::finished);
            case State::rawDeflate: {
                const auto status = _inflate.process(_reader, _output, finish);
                if (!status) {
                    return fail(status.error());
                }
                if (*status == detail::StreamProgress::needInput || *status == detail::StreamProgress::needOutput) {
                    return result(*status);
                }
                _state = State::done;
                break;
            }
            case State::zlibHeader:
                if (const auto status = readZlibHeader(finish); !status) {
                    return fail(status.error());
                } else if (*status != detail::StreamProgress::advanced) {
                    return result(*status);
                }
                break;
            case State::zlibDictionary:
                if (const auto status = readUnsupportedDictionaryId(finish); !status) {
                    return fail(status.error());
                } else if (*status != detail::StreamProgress::advanced) {
                    return result(*status);
                }
                return fail(Error::presetDictionaryRequired);
            case State::zlibDeflate: {
                const auto status = _inflate.process(_reader, _output, finish);
                if (!status) {
                    return fail(status.error());
                }
                if (*status == detail::StreamProgress::needInput || *status == detail::StreamProgress::needOutput) {
                    return result(*status);
                }
                _state      = State::zlibTrailer;
                _trailerPos = 0UZ;
                break;
            }
            case State::zlibTrailer:
                if (const auto status = readZlibTrailer(finish); !status) {
                    return fail(status.error());
                } else if (*status != detail::StreamProgress::advanced) {
                    return result(*status);
                }
                break;
            case State::gzipHeader:
                if (const auto status = readGzipHeader(finish); !status) {
                    return fail(status.error());
                } else if (*status != detail::StreamProgress::advanced) {
                    return result(*status);
                }
                break;
            case State::gzipExtraLength:
                if (const auto status = readGzipExtraLength(finish); !status) {
                    return fail(status.error());
                } else if (*status != detail::StreamProgress::advanced) {
                    return result(*status);
                }
                break;
            case State::gzipExtra:
                if (const auto status = readGzipExtra(finish); !status) {
                    return fail(status.error());
                } else if (*status != detail::StreamProgress::advanced) {
                    return result(*status);
                }
                break;
            case State::gzipName:
                if (const auto status = readGzipString(OptionalField::comment, finish); !status) {
                    return fail(status.error());
                } else if (*status != detail::StreamProgress::advanced) {
                    return result(*status);
                }
                break;
            case State::gzipComment:
                if (const auto status = readGzipString(OptionalField::headerCrc, finish); !status) {
                    return fail(status.error());
                } else if (*status != detail::StreamProgress::advanced) {
                    return result(*status);
                }
                break;
            case State::gzipHeaderCrc:
                if (const auto status = readGzipHeaderCrc(finish); !status) {
                    return fail(status.error());
                } else if (*status != detail::StreamProgress::advanced) {
                    return result(*status);
                }
                break;
            case State::gzipDeflate: {
                const auto status = _inflate.process(_reader, _output, finish);
                if (!status) {
                    return fail(status.error());
                }
                if (*status == detail::StreamProgress::needInput || *status == detail::StreamProgress::needOutput) {
                    return result(*status);
                }
                _state      = State::gzipTrailer;
                _trailerPos = 0UZ;
                break;
            }
            case State::gzipTrailer:
                if (const auto status = readGzipTrailer(finish); !status) {
                    return fail(status.error());
                } else if (*status != detail::StreamProgress::advanced) {
                    return result(*status);
                }
                break;
            case State::gzipBetweenMembers:
                if (_reader.pos == input.size()) {
                    if (finish) {
                        _state = State::done;
                        break;
                    }
                    return result(detail::StreamProgress::needInput);
                }
                beginGzipHeader();
                break;
            }
        }
    }

private:
    enum class State : std::uint8_t { rawDeflate, zlibHeader, zlibDictionary, zlibDeflate, zlibTrailer, gzipHeader, gzipExtraLength, gzipExtra, gzipName, gzipComment, gzipHeaderCrc, gzipDeflate, gzipTrailer, gzipBetweenMembers, done };

    Format                    _format = Format::gzip;
    State                     _state  = State::gzipHeader;
    detail::StreamBitReader   _reader{};
    detail::StreamOutput      _output{};
    detail::StreamInflate     _inflate{};
    std::array<std::byte, 2>  _zlibHeader{};
    std::size_t               _zlibHeaderPos{};
    std::array<std::byte, 8>  _trailer{};
    std::size_t               _trailerPos{};
    std::array<std::byte, 10> _gzipHeader{};
    std::size_t               _gzipHeaderPos{};
    std::uint8_t              _gzipFlags{};
    std::size_t               _gzipExtraRemaining{};
    std::uint32_t             _gzipHeaderCrcState = 0xffff'ffffU;
    std::size_t               _gzipMemberCount{};

    [[nodiscard]] std::expected<std::optional<std::byte>, Error> readByte(bool finish) noexcept {
        const auto value = _reader.readBits(8, finish);
        if (!value) {
            return std::unexpected(value.error());
        }
        if (!*value) {
            return std::optional<std::byte>{};
        }
        return static_cast<std::byte>(**value);
    }

    [[nodiscard]] std::expected<std::optional<std::byte>, Error> readGzipHeaderByte(bool finish) noexcept {
        const auto byte = readByte(finish);
        if (!byte) {
            return std::unexpected(byte.error());
        }
        if (*byte) {
            const auto value    = **byte;
            _gzipHeaderCrcState = gr::crc::updateScalar<gr::crc::Flavour::CRC32_IEEE>(_gzipHeaderCrcState, std::span<const std::byte>(&value, 1UZ));
        }
        return *byte;
    }

    [[nodiscard]] std::expected<detail::StreamProgress, Error> readZlibHeader(bool finish) noexcept {
        while (_zlibHeaderPos < _zlibHeader.size()) {
            const auto byte = readByte(finish);
            if (!byte) {
                return std::unexpected(byte.error());
            }
            if (!*byte) {
                return detail::StreamProgress::needInput;
            }
            _zlibHeader[_zlibHeaderPos++] = **byte;
        }
        const auto cmf = std::to_integer<std::uint8_t>(_zlibHeader[0]);
        const auto flg = std::to_integer<std::uint8_t>(_zlibHeader[1]);
        if ((cmf & 0x0fU) != 8U || (cmf >> 4U) > 7U || ((static_cast<unsigned>(cmf) << 8U) + flg) % 31U != 0U) {
            return std::unexpected(Error::invalidHeader);
        }
        if ((flg & 0x20U) != 0U) {
            _state      = State::zlibDictionary;
            _trailerPos = 0UZ;
            return detail::StreamProgress::advanced;
        }
        _inflate.reset();
        _output.resetHistory();
        _state = State::zlibDeflate;
        return detail::StreamProgress::advanced;
    }

    [[nodiscard]] std::expected<detail::StreamProgress, Error> readUnsupportedDictionaryId(bool finish) noexcept {
        while (_trailerPos < 4UZ) {
            const auto byte = readByte(finish);
            if (!byte) {
                return std::unexpected(byte.error());
            }
            if (!*byte) {
                return detail::StreamProgress::needInput;
            }
            _trailer[_trailerPos++] = **byte;
        }
        return detail::StreamProgress::advanced;
    }

    [[nodiscard]] std::expected<detail::StreamProgress, Error> readZlibTrailer(bool finish) noexcept {
        while (_trailerPos < 4UZ) {
            const auto byte = readByte(finish);
            if (!byte) {
                return std::unexpected(byte.error());
            }
            if (!*byte) {
                return detail::StreamProgress::needInput;
            }
            _trailer[_trailerPos++] = **byte;
        }
        if (_output.adler32Value() != detail::readBig32(std::span<const std::byte>(_trailer).first(4UZ), 0UZ)) {
            return std::unexpected(Error::checksumMismatch);
        }
        _state = State::done;
        return detail::StreamProgress::advanced;
    }

    void beginGzipHeader() noexcept {
        _state              = State::gzipHeader;
        _gzipHeader         = {};
        _gzipHeaderPos      = 0UZ;
        _gzipFlags          = 0U;
        _gzipExtraRemaining = 0UZ;
        _gzipHeaderCrcState = 0xffff'ffffU;
        _trailer            = {};
        _trailerPos         = 0UZ;
    }

    void beginGzipDeflate() noexcept {
        _output.resetGzipMember();
        _inflate.reset();
        _state = State::gzipDeflate;
    }

    /// RFC 1952 orders the optional fields FEXTRA, FNAME, FCOMMENT, FHCRC; `from` is the first one still to consider
    enum class OptionalField : std::uint8_t { extra, name, comment, headerCrc };

    void nextGzipOptionalState(OptionalField from) noexcept {
        const auto pending = [this, from](OptionalField field, std::uint8_t flag) noexcept { return from <= field && (_gzipFlags & flag) != 0U; };
        if (pending(OptionalField::extra, 0x04U)) {
            _state = State::gzipExtraLength;
            return;
        }
        if (pending(OptionalField::name, 0x08U)) {
            _state = State::gzipName;
            return;
        }
        if (pending(OptionalField::comment, 0x10U)) {
            _state = State::gzipComment;
            return;
        }
        if (pending(OptionalField::headerCrc, 0x02U)) {
            _state      = State::gzipHeaderCrc;
            _trailerPos = 0UZ;
            return;
        }
        beginGzipDeflate();
    }

    [[nodiscard]] std::expected<detail::StreamProgress, Error> readGzipHeader(bool finish) noexcept {
        while (_gzipHeaderPos < _gzipHeader.size()) {
            const auto byte = readGzipHeaderByte(finish);
            if (!byte) {
                return std::unexpected(byte.error());
            }
            if (!*byte) {
                return detail::StreamProgress::needInput;
            }
            if (_gzipHeaderPos == 0UZ && **byte != std::byte{0x1f}) {
                return std::unexpected(_gzipMemberCount == 0UZ ? Error::invalidHeader : Error::trailingData);
            }
            if (_gzipHeaderPos == 1UZ && **byte != std::byte{0x8b}) {
                return std::unexpected(_gzipMemberCount == 0UZ ? Error::invalidHeader : Error::trailingData);
            }
            _gzipHeader[_gzipHeaderPos++] = **byte;
        }
        if (_gzipHeader[2] != std::byte{0x08}) {
            return std::unexpected(Error::unsupportedCompressionMethod);
        }
        _gzipFlags = std::to_integer<std::uint8_t>(_gzipHeader[3]);
        if ((_gzipFlags & 0xe0U) != 0U) {
            return std::unexpected(Error::reservedFlags);
        }
        nextGzipOptionalState(OptionalField::extra);
        return detail::StreamProgress::advanced;
    }

    [[nodiscard]] std::expected<detail::StreamProgress, Error> readGzipExtraLength(bool finish) noexcept {
        while (_trailerPos < 2UZ) {
            const auto byte = readGzipHeaderByte(finish);
            if (!byte) {
                return std::unexpected(byte.error());
            }
            if (!*byte) {
                return detail::StreamProgress::needInput;
            }
            _trailer[_trailerPos++] = **byte;
        }
        _gzipExtraRemaining = detail::readLittle16(std::span<const std::byte>(_trailer).first(2UZ), 0UZ);
        _trailerPos         = 0UZ;
        _state              = State::gzipExtra;
        return detail::StreamProgress::advanced;
    }

    [[nodiscard]] std::expected<detail::StreamProgress, Error> readGzipExtra(bool finish) noexcept {
        while (_gzipExtraRemaining != 0UZ) {
            const auto byte = readGzipHeaderByte(finish);
            if (!byte) {
                return std::unexpected(byte.error());
            }
            if (!*byte) {
                return detail::StreamProgress::needInput;
            }
            --_gzipExtraRemaining;
        }
        nextGzipOptionalState(OptionalField::name);
        return detail::StreamProgress::advanced;
    }

    [[nodiscard]] std::expected<detail::StreamProgress, Error> readGzipString(OptionalField next, bool finish) noexcept {
        while (true) {
            const auto byte = readGzipHeaderByte(finish);
            if (!byte) {
                return std::unexpected(byte.error());
            }
            if (!*byte) {
                return detail::StreamProgress::needInput;
            }
            if (**byte == std::byte{0}) {
                nextGzipOptionalState(next);
                return detail::StreamProgress::advanced;
            }
        }
    }

    [[nodiscard]] std::expected<detail::StreamProgress, Error> readGzipHeaderCrc(bool finish) noexcept {
        while (_trailerPos < 2UZ) {
            const auto byte = readByte(finish);
            if (!byte) {
                return std::unexpected(byte.error());
            }
            if (!*byte) {
                return detail::StreamProgress::needInput;
            }
            _trailer[_trailerPos++] = **byte;
        }
        const auto expected = detail::readLittle16(std::span<const std::byte>(_trailer).first(2UZ), 0UZ);
        const auto actual   = static_cast<std::uint16_t>((_gzipHeaderCrcState ^ 0xffff'ffffU) & 0xffffU);
        if (expected != actual) {
            return std::unexpected(Error::checksumMismatch);
        }
        beginGzipDeflate();
        return detail::StreamProgress::advanced;
    }

    [[nodiscard]] std::expected<detail::StreamProgress, Error> readGzipTrailer(bool finish) noexcept {
        while (_trailerPos < _trailer.size()) {
            const auto byte = readByte(finish);
            if (!byte) {
                return std::unexpected(byte.error());
            }
            if (!*byte) {
                return detail::StreamProgress::needInput;
            }
            _trailer[_trailerPos++] = **byte;
        }
        const auto trailer = std::span<const std::byte>(_trailer);
        if (_output.crc32() != detail::readLittle32(trailer, 0UZ)) {
            return std::unexpected(Error::checksumMismatch);
        }
        if (_output.memberBytes() != detail::readLittle32(trailer, 4UZ)) {
            return std::unexpected(Error::sizeMismatch);
        }
        ++_gzipMemberCount;
        _state = State::gzipBetweenMembers;
        return detail::StreamProgress::advanced;
    }
};

class StreamCompressor {
public:
    explicit StreamCompressor(Format format = Format::gzip, CompressionLevel level = CompressionLevel::balanced, std::size_t blockBytes = detail::kMaxStoredBlock) { reset(format, level, blockBytes); }

    void reset(Format format = Format::gzip, CompressionLevel level = CompressionLevel::balanced, std::size_t blockBytes = detail::kMaxStoredBlock) {
        _format     = format;
        _level      = level;
        _blockBytes = std::clamp(blockBytes, 1UZ, detail::kMaxStoredBlock);
        _window.clear();
        _window.reserve(detail::kWindowSize + _blockBytes);
        _historyBytes = 0UZ;
        _basePosition = 0UZ;
        _indexedEnd   = 0UZ;
        _tokens.assign(detail::kWindowSize + _blockBytes, detail::Token{});
        _head.assign(detail::kHashSize, 0UZ);
        _previous.assign(detail::kWindowSize + _blockBytes, 0UZ);
        _pendingOutput    = {};
        _pendingOutputPos = 0UZ;
        _bits             = 0U;
        _bitCount         = 0U;
        _started          = false;
        _finalQueued      = false;
        _finished         = false;
        _crcState         = 0xffff'ffffU;
        _adlerA           = 1U;
        _adlerB           = 0U;
        _inputSize        = 0U;
    }

    [[nodiscard]] bool finished() const noexcept { return _finished; }

    [[nodiscard]] std::expected<StreamResult, Error> process(std::span<const std::byte> input, std::span<std::byte> output, Flush flush = Flush::none) noexcept {
#if __cpp_exceptions
        try {
#endif
            StreamResult result{};
            if (_finished) {
                if (!input.empty()) {
                    return std::unexpected(Error::trailingData);
                }
                result.status = StreamStatus::done;
                return result;
            }
            if (!_started) {
                queueHeader();
                _started = true;
            }

            const auto finish = flush == Flush::finish;
            while (true) {
                result.produced += drain(output.subspan(result.produced));
                if (result.produced == output.size()) {
                    break;
                }
                if (_finalQueued) {
                    if (_pendingOutput.empty()) {
                        _finished = true;
                    }
                    break;
                }
                if (pendingBytes() == _blockBytes) {
                    const auto finalBlock = finish && result.consumed == input.size();
                    queueDeflateBlock(finalBlock);
                    if (finalBlock) {
                        queueTrailer();
                        _finalQueued = true;
                    }
                    continue;
                }
                if (result.consumed < input.size()) {
                    const auto count = std::min(input.size() - result.consumed, _blockBytes - pendingBytes());
                    const auto chunk = input.subspan(result.consumed, count);
                    noteInput(chunk);
                    _window.insert(_window.end(), chunk.begin(), chunk.end());
                    result.consumed += count;
                    continue;
                }
                if (finish) {
                    queueDeflateBlock(true);
                    queueTrailer();
                    _finalQueued = true;
                    continue;
                }
                break;
            }
            result.status = _finished ? StreamStatus::done : (_pendingOutput.size() > _pendingOutputPos ? StreamStatus::needOutput : StreamStatus::needInput);
            return result;
#if __cpp_exceptions
        } catch (const std::bad_alloc&) {
            return std::unexpected(Error::allocationFailed);
        } catch (const std::length_error&) {
            return std::unexpected(Error::allocationFailed);
        }
#endif // -fno-exceptions: an allocation failure faults hard instead of reporting Error::allocationFailed
    }

private:
    Format                     _format     = Format::gzip;
    CompressionLevel           _level      = CompressionLevel::balanced;
    std::size_t                _blockBytes = detail::kMaxStoredBlock;
    std::vector<std::byte>     _window{};
    std::size_t                _historyBytes{};
    std::size_t                _basePosition{};
    std::size_t                _indexedEnd{};
    std::vector<detail::Token> _tokens{};
    std::vector<std::size_t>   _head{};
    std::vector<std::size_t>   _previous{};
    std::vector<std::byte>     _pendingOutput{};
    std::size_t                _pendingOutputPos{};
    std::uint64_t              _bits{};
    unsigned                   _bitCount{};
    bool                       _started{};
    bool                       _finalQueued{};
    bool                       _finished{};
    std::uint32_t              _crcState = 0xffff'ffffU;
    std::uint32_t              _adlerA   = 1U;
    std::uint32_t              _adlerB{};
    std::uint32_t              _inputSize{};

    [[nodiscard]] std::size_t pendingBytes() const noexcept { return _window.size() - _historyBytes; }

    static void appendByte(std::vector<std::byte>& output, std::uint32_t value) { output.push_back(static_cast<std::byte>(value & 0xffU)); }

    static void appendLittle32(std::vector<std::byte>& output, std::uint32_t value) {
        appendByte(output, value);
        appendByte(output, value >> 8U);
        appendByte(output, value >> 16U);
        appendByte(output, value >> 24U);
    }

    static void appendBig32(std::vector<std::byte>& output, std::uint32_t value) {
        appendByte(output, value >> 24U);
        appendByte(output, value >> 16U);
        appendByte(output, value >> 8U);
        appendByte(output, value);
    }

    void queueHeader() {
        if (_format == Format::zlib) {
            constexpr std::uint16_t kCmf   = 0x78U;
            const std::uint16_t     flevel = _level == CompressionLevel::fast ? 0U : _level == CompressionLevel::best ? 3U : 2U;
            std::uint16_t           header = static_cast<std::uint16_t>((kCmf << 8U) | (flevel << 6U));
            header                         = static_cast<std::uint16_t>(header + (31U - (header % 31U)) % 31U);
            appendByte(_pendingOutput, header >> 8U);
            appendByte(_pendingOutput, header);
            return;
        }
        if (_format == Format::gzip) {
            const auto xfl = _level == CompressionLevel::fast ? 0x04U : _level == CompressionLevel::best ? 0x02U : 0x00U;
            for (const auto byte : std::array<std::uint32_t, 10>{0x1fU, 0x8bU, 0x08U, 0x00U, 0x00U, 0x00U, 0x00U, 0x00U, xfl, 0xffU}) {
                appendByte(_pendingOutput, byte);
            }
        }
    }

    void noteInput(std::span<const std::byte> input) noexcept {
        _crcState                  = gr::crc::updateScalar<gr::crc::Flavour::CRC32_IEEE>(_crcState, input);
        std::tie(_adlerA, _adlerB) = detail::updateAdler(_adlerA, _adlerB, input);
        _inputSize += static_cast<std::uint32_t>(input.size());
    }

    void queueDeflateBlock(bool final) {
        const auto pending = std::span<const std::byte>(_window).subspan(_historyBytes);
        assert(_previous.size() >= _window.size());

        const detail::WindowLayout layout{.basePosition = _basePosition, .indexedFrom = _indexedEnd - _basePosition, .tokenFrom = _historyBytes};
        const auto                 tokens   = detail::tokenize(_window, layout, _level, {.tokens = _tokens, .head = _head, .previous = _previous});
        const auto                 coding   = detail::makeDynamicCoding(tokens);
        const auto                 encoding = detail::selectEncoding(pending.size(), tokens, coding, final);

        detail::GrowingSink                    sink{_pendingOutput};
        detail::BitWriter<detail::GrowingSink> writer{sink};
        writer.bits  = _bits;
        writer.count = _bitCount;
        detail::emitDeflate(writer, pending, tokens, coding, encoding, final);
        _bits     = writer.bits;
        _bitCount = writer.count;

        // tokenize() indexes every position it visits except the final two, which the next block picks up
        _indexedEnd = _basePosition + (_window.size() >= 2UZ ? _window.size() - 2UZ : 0UZ);
        slideWindow();
    }

    /// keeps the last 32 KiB of emitted input so the next block can still match into it
    void slideWindow() {
        _historyBytes = _window.size();
        if (_historyBytes > detail::kWindowSize) {
            const auto drop = _historyBytes - detail::kWindowSize;
            _window.erase(_window.begin(), _window.begin() + static_cast<std::ptrdiff_t>(drop));
            _historyBytes = detail::kWindowSize;
            _basePosition += drop;
        }
    }

    void queueTrailer() {
        if (_format == Format::zlib) {
            appendBig32(_pendingOutput, (_adlerB << 16U) | _adlerA);
        } else if (_format == Format::gzip) {
            appendLittle32(_pendingOutput, _crcState ^ 0xffff'ffffU);
            appendLittle32(_pendingOutput, _inputSize);
        }
    }

    std::size_t drain(std::span<std::byte> output) noexcept {
        const auto count = std::min(output.size(), _pendingOutput.size() - _pendingOutputPos);
        std::ranges::copy_n(_pendingOutput.begin() + static_cast<std::ptrdiff_t>(_pendingOutputPos), static_cast<std::ptrdiff_t>(count), output.begin());
        _pendingOutputPos += count;
        if (_pendingOutputPos == _pendingOutput.size()) {
            _pendingOutput.clear();
            _pendingOutputPos = 0UZ;
        }
        return count;
    }
};

namespace detail {

template<std::size_t Bound>
struct EncodedResource {
    std::array<std::byte, Bound> buffer{};
    std::size_t                  size{};
};

template<auto Data, CompressionLevel Level>
[[nodiscard]] consteval auto tokenizeConstant() {
    std::array<std::size_t, kHashSize>                 head{};
    std::array<std::size_t, historyBound(Data.size())> previous{};

    TokenBuffer<Data.size()> result{};
    result.size = tokenize(std::span<const std::byte>(Data), {}, Level, {.tokens = result.value, .head = head, .previous = previous}).size();
    return result;
}

/// each stage is a namespace-scope variable template, hence its own top-level constant evaluation with a fresh
/// -fconstexpr-steps / -fconstexpr-ops-limit budget; locals inside one `consteval` call would share a single budget
template<auto Data, CompressionLevel Level>
inline constexpr auto kTokens = tokenizeConstant<Data, Level>();

template<auto Data, CompressionLevel Level>
inline constexpr DynamicCoding kCoding = makeDynamicCoding(kTokens<Data, Level>.view());

template<auto Data, CompressionLevel Level>
inline constexpr Encoding kEncoding = selectEncoding(Data.size(), kTokens<Data, Level>.view(), kCoding<Data, Level>);

template<auto Data, Format Wrapper, CompressionLevel Level>
[[nodiscard]] consteval auto encode() {
    EncodedResource<compressBound(Data.size(), Wrapper)> result{};
    ByteSink                                             sink{result.buffer};
    BitWriter<ByteSink>                                  writer{sink};
    emitWrapped(writer, std::span<const std::byte>(Data), kTokens<Data, Level>.view(), kCoding<Data, Level>, kEncoding<Data, Level>, Wrapper, Level);
    result.size = writer.size();
    return result;
}

template<auto Data, Format Wrapper, CompressionLevel Level>
[[nodiscard]] consteval auto encodeTrimmed() {
    constexpr auto encoded = encode<Data, Wrapper, Level>();
    static_assert(encoded.size <= encoded.buffer.size(), "emitted stream exceeds compressBound()");

    std::array<std::byte, encoded.size> packed{};
    std::ranges::copy_n(encoded.buffer.begin(), static_cast<std::ptrdiff_t>(encoded.size), packed.begin());
    return packed;
}

template<auto Packed, auto Original, Format Wrapper>
[[nodiscard]] consteval bool roundTrips() {
    std::array<std::byte, Original.size()> decoded{};
    const auto                             written = gr::compression::decompress(std::span<const std::byte>(Packed), decoded, Wrapper);
    if (!written || *written != Original.size()) {
        return false;
    }
    return std::ranges::equal(decoded, Original);
}

} // namespace detail

template<auto Data, CompressionLevel Level = CompressionLevel::balanced>
[[nodiscard]] consteval auto gzip() {
    static_assert(Data.size() <= std::numeric_limits<std::uint32_t>::max(), "gzip ISIZE is limited to 32 bits for compile-time resources");
    constexpr auto packed = detail::encodeTrimmed<Data, Format::gzip, Level>();
    static_assert(detail::roundTrips<packed, Data, Format::gzip>(), "compile-time gzip resource does not decode back to its source");
    return packed;
}

template<auto Packed, std::size_t OriginalSize>
struct CompressedBlob {
    [[nodiscard]] static constexpr std::size_t compressedSize() noexcept { return Packed.size(); }
    [[nodiscard]] static constexpr std::size_t uncompressedSize() noexcept { return OriginalSize; }
    [[nodiscard]] static constexpr const auto& compressed() noexcept { return Packed; }

    [[nodiscard]] static const std::vector<std::byte>& decompress() {
        static const std::vector<std::byte> cache = [] {
            std::vector<std::byte> output(OriginalSize);
            const auto             result = gr::compression::decompress(Packed, output, Format::gzip);
            assert(result.has_value() && *result == OriginalSize);
            if (!result || *result != OriginalSize) {
                output.clear();
            }
            return output;
        }();
        return cache;
    }

    [[nodiscard]] static std::span<const std::byte> bytes() { return decompress(); }
};

template<auto Packed, std::size_t OriginalSize>
struct CompressedText {
    [[nodiscard]] static constexpr std::size_t compressedSize() noexcept { return Packed.size(); }
    [[nodiscard]] static constexpr std::size_t uncompressedSize() noexcept { return OriginalSize; }
    [[nodiscard]] static constexpr const auto& compressed() noexcept { return Packed; }

    [[nodiscard]] static const std::string& str() {
        static const std::string cache = [] {
            std::string output(OriginalSize, '\0');
            const auto  result = gr::compression::decompress(Packed, std::as_writable_bytes(std::span(output)), Format::gzip);
            assert(result.has_value() && *result == OriginalSize);
            if (!result || *result != OriginalSize) {
                output.clear();
            }
            return output;
        }();
        return cache;
    }

    [[nodiscard]] static std::string_view view() { return str(); }
    [[nodiscard]] static const char*      c_str() { return str().c_str(); }

    [[nodiscard]] operator std::string_view() const { return view(); }
};

template<auto Data, CompressionLevel Level = CompressionLevel::balanced>
[[nodiscard]] consteval auto makeCompressedBlob() {
    constexpr auto packed = gzip<Data, Level>();
    return CompressedBlob<packed, Data.size()>{};
}

template<gr::meta::fixed_string S, CompressionLevel Level = CompressionLevel::balanced>
[[nodiscard]] consteval auto makeCompressedText() {
    constexpr auto packed = gzip<detail::staticBytesFromString<S>(), Level>();
    return CompressedText<packed, S.size()>{};
}

namespace literals {
template<gr::meta::fixed_string S>
[[nodiscard]] consteval auto operator""_gzip() {
    return makeCompressedText<S>();
}
} // namespace literals

} // namespace gr::compression

#endif // GNURADIO_COMPRESSION_HPP
