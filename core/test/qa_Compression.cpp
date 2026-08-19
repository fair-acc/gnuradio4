#include <boost/ut.hpp>

#include <gnuradio-4.0/Compression.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <memory_resource>
#include <print>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

using namespace boost::ut;
using namespace gr::compression::literals;

namespace {

inline constexpr auto kResource = gr::meta::fixed_string{R"(GNU Radio 4.0 embedded documentation.
This sentence is deliberately repetitive. This sentence is deliberately repetitive.
This sentence is deliberately repetitive. This sentence is deliberately repetitive.
This sentence is deliberately repetitive. This sentence is deliberately repetitive.
The source exists during constant evaluation; the resulting object contains gzip bytes.)"};

inline constexpr std::string_view kResourceText   = kResource;
static constexpr auto             kCompressedText = gr::compression::makeCompressedText<kResource>();

consteval bool constexprGzipRoundTrip() {
    std::array<std::byte, kResourceText.size()> output{};
    const auto                                  result = gr::compression::decompress(kCompressedText.compressed(), output, gr::compression::Format::gzip);
    if (!result || *result != output.size()) {
        return false;
    }
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (output[i] != static_cast<std::byte>(static_cast<unsigned char>(kResourceText[i]))) {
            return false;
        }
    }
    return true;
}

static_assert(constexprGzipRoundTrip());

inline constexpr std::string_view kSkewedPattern = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbccccccccccccddddddddddeeeeeeffffgghijklmnopqrstuvwxyz0123456789|";

template<std::size_t Repetitions>
consteval auto makeSkewedInput() {
    std::array<std::byte, kSkewedPattern.size() * Repetitions> result{};
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = static_cast<std::byte>(kSkewedPattern[i % kSkewedPattern.size()]);
    }
    return result;
}

template<std::size_t N>
consteval auto makeNoise() {
    std::array<std::byte, N> result{};
    std::uint32_t            state = 0x9e37'79b9U;
    for (auto& value : result) {
        state ^= state << 13U;
        state ^= state >> 17U;
        state ^= state << 5U;
        value = static_cast<std::byte>(state & 0xffU);
    }
    return result;
}

template<std::size_t Distractors>
consteval auto makeSearchInput() {
    constexpr std::string_view                                                         phrase     = "xabcdefghijklmnopqrstuvwxyz0123456789";
    constexpr std::string_view                                                         distractor = "xab!";
    std::array<std::byte, 2UZ * phrase.size() + Distractors * distractor.size() + 1UZ> result{};
    std::size_t                                                                        pos    = 0;
    const auto                                                                         append = [&result, &pos](std::string_view text) {
        for (const char value : text) {
            result[pos++] = static_cast<std::byte>(value);
        }
    };
    append(phrase);
    for (std::size_t i = 0; i < Distractors; ++i) {
        append(distractor);
    }
    append("#");
    append(phrase);
    return result;
}

consteval auto makeAllBytes() {
    std::array<std::byte, 256> result{};
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = static_cast<std::byte>(i);
    }
    return result;
}

template<std::size_t N>
consteval auto makeFibonacciFrequencies() {
    std::array<std::uint32_t, N> frequencies{};
    frequencies[0] = 1;
    frequencies[1] = 1;
    for (std::size_t i = 2; i < N; ++i) {
        frequencies[i] = frequencies[i - 1UZ] + frequencies[i - 2UZ];
    }
    return frequencies;
}

template<unsigned MaxBits, std::size_t N>
consteval bool isCompleteTree(const std::array<std::uint8_t, N>& lengths) {
    unsigned units = 0;
    for (const auto length : lengths) {
        if (length == 0U || length > MaxBits) {
            return false;
        }
        units += 1U << (MaxBits - length);
    }
    return units == (1U << MaxBits);
}

inline constexpr auto kLimited15BitLengths = gr::compression::detail::makeCodeLengths<30, 15>(makeFibonacciFrequencies<30>());
inline constexpr auto kLimited7BitLengths  = gr::compression::detail::makeCodeLengths<19, 7>(makeFibonacciFrequencies<19>());
static_assert(isCompleteTree<15>(kLimited15BitLengths));
static_assert(isCompleteTree<7>(kLimited7BitLengths));

inline constexpr std::array<std::byte, 0> kEmptyInput{};
inline constexpr std::array               kOneByteInput{std::byte{'x'}};
inline constexpr auto                     kAllBytesInput   = makeAllBytes();
inline constexpr auto                     kDynamicInput    = makeSkewedInput<30>();
inline constexpr auto                     kNoiseInput      = makeNoise<4096>();
inline constexpr auto                     kFastSearchInput = makeSearchInput<8>();
inline constexpr auto                     kBestSearchInput = makeSearchInput<24>();
inline constexpr auto                     kLazyResource    = gr::meta::fixed_string{"abcdefghijklmnopqrstuvwxyz0123456789|xaby#xabcdefghijklmnopqrstuvwxyz0123456789"};
inline constexpr auto                     kLazyInput       = gr::compression::detail::staticBytesFromString<kLazyResource>();

inline constexpr auto kEmptyGzip          = gr::compression::gzip<kEmptyInput>();
inline constexpr auto kOneByteGzip        = gr::compression::gzip<kOneByteInput>();
inline constexpr auto kAllBytesGzip       = gr::compression::gzip<kAllBytesInput>();
inline constexpr auto kDynamicGzip        = gr::compression::gzip<kDynamicInput>();
inline constexpr auto kNoiseGzip          = gr::compression::gzip<kNoiseInput>();
inline constexpr auto kCompressedBlob     = gr::compression::makeCompressedBlob<kAllBytesInput>();
inline constexpr auto kFastLevelGzip      = gr::compression::gzip<kLazyInput, gr::compression::CompressionLevel::fast>();
inline constexpr auto kBalancedLevelGzip  = gr::compression::gzip<kLazyInput, gr::compression::CompressionLevel::balanced>();
inline constexpr auto kBestLevelGzip      = gr::compression::gzip<kLazyInput, gr::compression::CompressionLevel::best>();
inline constexpr auto kFastSearchGzip     = gr::compression::gzip<kFastSearchInput, gr::compression::CompressionLevel::fast>();
inline constexpr auto kBalancedSearchGzip = gr::compression::gzip<kFastSearchInput, gr::compression::CompressionLevel::balanced>();
inline constexpr auto kDeepBalancedGzip   = gr::compression::gzip<kBestSearchInput, gr::compression::CompressionLevel::balanced>();
inline constexpr auto kBestSearchGzip     = gr::compression::gzip<kBestSearchInput, gr::compression::CompressionLevel::best>();

template<auto Input, gr::compression::CompressionLevel Level>
consteval std::size_t tokenCount() {
    return gr::compression::detail::tokenizeConstant<Input, Level>().size;
}

template<auto Input, gr::compression::CompressionLevel Level>
consteval bool lastMatchIsLiteral() {
    constexpr auto tokens = gr::compression::detail::tokenizeConstant<Input, Level>();
    return tokens.view()[tokens.size - 2UZ].isLiteral();
}

static_assert(kEmptyGzip[0] == std::byte{0x1f});
static_assert(kEmptyGzip[1] == std::byte{0x8b});
static_assert(kEmptyGzip[2] == std::byte{0x08});
static_assert((std::to_integer<unsigned>(kEmptyGzip[10]) & 7U) == 3U);
static_assert((std::to_integer<unsigned>(kDynamicGzip[10]) & 7U) == 5U);
static_assert((std::to_integer<unsigned>(kNoiseGzip[10]) & 7U) == 1U);
static_assert(gr::compression::gzip<kLazyInput>() == kBalancedLevelGzip);
static_assert(tokenCount<kFastSearchInput, gr::compression::CompressionLevel::balanced>() < tokenCount<kFastSearchInput, gr::compression::CompressionLevel::fast>());
static_assert(tokenCount<kBestSearchInput, gr::compression::CompressionLevel::best>() < tokenCount<kBestSearchInput, gr::compression::CompressionLevel::balanced>());
static_assert(!lastMatchIsLiteral<kLazyInput, gr::compression::CompressionLevel::balanced>());
static_assert(lastMatchIsLiteral<kLazyInput, gr::compression::CompressionLevel::best>());
static_assert(kBalancedSearchGzip.size() < kFastSearchGzip.size());
static_assert(kBestSearchGzip.size() < kDeepBalancedGzip.size());

template<std::size_t N>
[[nodiscard]] constexpr std::span<const std::byte> bytesOf(const std::array<std::byte, N>& input) noexcept {
    return input;
}

[[nodiscard]] std::span<const std::byte> asBytes(std::string_view input) noexcept { return {reinterpret_cast<const std::byte*>(input.data()), input.size()}; }

[[nodiscard]] std::vector<std::byte> bytesFromHex(std::string_view text) {
    const auto digit = [](char value) -> unsigned {
        if (value >= '0' && value <= '9') {
            return static_cast<unsigned>(value - '0');
        }
        return static_cast<unsigned>(value - 'a' + 10);
    };
    std::vector<std::byte> result(text.size() / 2UZ);
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] = static_cast<std::byte>((digit(text[2UZ * i]) << 4U) | digit(text[2UZ * i + 1UZ]));
    }
    return result;
}

[[nodiscard]] std::vector<std::byte> copyBytes(std::span<const std::byte> input) { return {input.begin(), input.end()}; }

[[nodiscard]] std::string repeat(std::string_view text, std::size_t count) {
    std::string result;
    result.reserve(text.size() * count);
    for (std::size_t i = 0; i < count; ++i) {
        result += text;
    }
    return result;
}

void expectBytes(std::span<const std::byte> actual, std::span<const std::byte> expected) {
    expect(eq(actual.size(), expected.size()));
    expect(std::ranges::equal(actual, expected));
}

void expectDecoded(std::span<const std::byte> compressed, gr::compression::Format format, std::span<const std::byte> expected) {
    const auto allocated = gr::compression::decompress(compressed, format);
    expect(allocated.has_value());
    if (!allocated) {
        return;
    }
    expectBytes(*allocated, expected);

    std::vector<std::byte> output(expected.size() + 7UZ, std::byte{0xa5});
    const auto             written = gr::compression::decompress(compressed, output, format);
    expect(written.has_value());
    if (!written) {
        return;
    }
    expect(eq(*written, expected.size()));
    expectBytes(std::span<const std::byte>(output).first(*written), expected);
    expect(std::ranges::all_of(std::span<const std::byte>(output).subspan(*written), [](std::byte value) { return value == std::byte{0xa5}; }));
}

[[nodiscard]] std::expected<std::vector<std::byte>, gr::compression::Error> streamDecompress(std::span<const std::byte> compressed, gr::compression::Format format, std::size_t inputStep, std::size_t outputStep, std::size_t maxOutputSize = gr::compression::kDefaultMaxDecompressedSize) {
    gr::compression::StreamDecompressor decoder{format, maxOutputSize};
    std::vector<std::byte>              result;
    std::vector<std::byte>              chunk(outputStep);
    std::size_t                         consumed  = 0UZ;
    std::size_t                         available = std::min(inputStep, compressed.size());

    while (!decoder.finished()) {
        const auto finish = available == compressed.size();
        const auto state  = decoder.process(compressed.subspan(consumed, available - consumed), chunk, finish ? gr::compression::Flush::finish : gr::compression::Flush::none);
        if (!state) {
            return std::unexpected(state.error());
        }
        consumed += state->consumed;
        result.insert(result.end(), chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(state->produced));
        if (state->finished()) {
            break;
        }
        if (state->status == gr::compression::StreamStatus::needInput) {
            if (finish) {
                return std::unexpected(gr::compression::Error::truncatedInput);
            }
            available = std::min(compressed.size(), available + inputStep);
        }
    }
    if (consumed != compressed.size()) {
        return std::unexpected(gr::compression::Error::trailingData);
    }
    return result;
}

[[nodiscard]] std::expected<std::vector<std::byte>, gr::compression::Error> streamCompress(std::span<const std::byte> plain, gr::compression::Format format, gr::compression::CompressionLevel level, std::size_t inputStep, std::size_t outputStep, std::size_t blockBytes = 97UZ) {
    gr::compression::StreamCompressor encoder{format, level, blockBytes};
    std::vector<std::byte>            result;
    std::vector<std::byte>            chunk(outputStep);
    std::size_t                       consumed = 0UZ;

    while (!encoder.finished()) {
        const auto count  = consumed == plain.size() ? 0UZ : std::min(inputStep, plain.size() - consumed);
        const auto finish = consumed + count == plain.size();
        const auto input  = plain.subspan(consumed, count);
        const auto state  = encoder.process(input, chunk, finish ? gr::compression::Flush::finish : gr::compression::Flush::none);
        if (!state) {
            return std::unexpected(state.error());
        }
        consumed += state->consumed;
        result.insert(result.end(), chunk.begin(), chunk.begin() + static_cast<std::ptrdiff_t>(state->produced));
        if (state->finished()) {
            break;
        }
        if (state->status == gr::compression::StreamStatus::needInput && consumed == plain.size() && count == 0UZ) {
            return std::unexpected(gr::compression::Error::truncatedInput);
        }
    }
    if (consumed != plain.size()) {
        return std::unexpected(gr::compression::Error::trailingData);
    }
    return result;
}

template<std::size_t N>
void expectGeneratedRoundTrip(const std::array<std::byte, N>& packed, std::span<const std::byte> expected) {
    expectDecoded(packed, gr::compression::Format::gzip, expected);
}

void appendLittle16(std::vector<std::byte>& output, std::uint16_t value) {
    output.push_back(static_cast<std::byte>(value & 0xffU));
    output.push_back(static_cast<std::byte>(value >> 8U));
}

void appendLittle32(std::vector<std::byte>& output, std::uint32_t value) {
    for (unsigned shift = 0; shift < 32U; shift += 8U) {
        output.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
    }
}

[[nodiscard]] std::vector<std::byte> makeStoredDeflate(std::span<const std::byte> input) {
    std::vector<std::byte> result;
    std::size_t            pos = 0;
    do {
        const auto size  = std::min<std::size_t>(input.size() - pos, 65535UZ);
        const auto final = pos + size == input.size();
        result.push_back(final ? std::byte{0x01} : std::byte{0x00});
        appendLittle16(result, static_cast<std::uint16_t>(size));
        appendLittle16(result, static_cast<std::uint16_t>(~static_cast<std::uint16_t>(size)));
        result.insert(result.end(), input.begin() + static_cast<std::ptrdiff_t>(pos), input.begin() + static_cast<std::ptrdiff_t>(pos + size));
        pos += size;
    } while (pos < input.size());
    return result;
}

[[nodiscard]] std::vector<std::byte> makeGzipWithOptionalFields(std::span<const std::byte> deflate, std::span<const std::byte> plain) {
    std::vector<std::byte> result{std::byte{0x1f}, std::byte{0x8b}, std::byte{0x08}, std::byte{0x1e}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0x00}, std::byte{0xff}};
    appendLittle16(result, 4);
    result.insert(result.end(), {std::byte{'G'}, std::byte{'R'}, std::byte{'4'}, std::byte{0x00}});
    for (const char value : std::string_view{"resource.bin"}) {
        result.push_back(static_cast<std::byte>(value));
    }
    result.push_back(std::byte{0x00});
    for (const char value : std::string_view{"RFC 1952 optional fields"}) {
        result.push_back(static_cast<std::byte>(value));
    }
    result.push_back(std::byte{0x00});
    appendLittle16(result, static_cast<std::uint16_t>(gr::crc::compute<gr::crc::Flavour::CRC32_IEEE>(result) & 0xffffU));
    result.insert(result.end(), deflate.begin(), deflate.end());
    appendLittle32(result, gr::crc::compute<gr::crc::Flavour::CRC32_IEEE>(plain));
    appendLittle32(result, static_cast<std::uint32_t>(plain.size()));
    return result;
}

[[nodiscard]] std::vector<std::byte> makeRuntimeNoise(std::size_t size) {
    std::vector<std::byte> result(size);
    std::uint32_t          state = 0x243f'6a88U;
    for (auto& value : result) {
        state ^= state << 13U;
        state ^= state >> 17U;
        state ^= state << 5U;
        value = static_cast<std::byte>(state & 0xffU);
    }
    return result;
}

#ifndef __EMSCRIPTEN__
[[nodiscard]] bool writeBytes(const std::filesystem::path& path, std::span<const std::byte> bytes) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return output.good();
}

[[nodiscard]] std::vector<std::byte> readBytes(const std::filesystem::path& path) {
    std::error_code error;
    const auto      size = std::filesystem::file_size(path, error);
    if (error || size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
        return {};
    }

    std::vector<std::byte> contents(static_cast<std::size_t>(size));
    if (contents.empty()) {
        return contents;
    }

    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(contents.data()), static_cast<std::streamsize>(contents.size()));
    if (!input) {
        return {};
    }
    return contents;
}

[[nodiscard]] std::string shellQuote(const std::filesystem::path& path) {
    std::string quoted = "'";
    for (const char value : path.string()) {
        quoted += value == '\'' ? "'\\''" : std::string(1, value);
    }
    quoted += "'";
    return quoted;
}

struct TemporaryFiles {
    std::vector<std::filesystem::path> paths;

    ~TemporaryFiles() {
        for (const auto& path : paths) {
            std::error_code error;
            std::filesystem::remove(path, error);
        }
    }

    [[nodiscard]] std::filesystem::path add(std::string_view suffix) {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        auto       path  = std::filesystem::temp_directory_path() / ("qa_Compression_" + std::to_string(stamp) + std::string(suffix));
        paths.push_back(path);
        return path;
    }
};

[[nodiscard]] bool gzipAvailable() { return std::system("gzip --version >/dev/null 2>&1") == 0; }
#endif

// Captured from zlib's level-9 raw/zlib encoders; they are independent decoder fixtures.
inline constexpr std::string_view kFixedRawHex      = "0bc94855282ccd4cce56482aca2fcf5348cbaf50c82acd2d2856c82f4b2d5228014ae72456552aa4e4a7eb29848c2a1e553caa98da8a01";
inline constexpr std::string_view kDynamicRawHex    = "edce4b1681500000d02d2584e53cd2c7a7108a63f19c061d93268def5dc10d619aed88dd9f74b0ef653f795e9487e3e95cd597ebadb93f9e6df77a47b378be5826abf5e6137c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c7c463e5f";
inline constexpr std::string_view kMultiBlockRawHex = "4acc29c84854484a2d4954484fcccd4d54481cac0200000000ffff3330343236313533b7b01c4c2c00";

// hand-built dynamic-block headers exercising limits real encoders never emit
inline constexpr std::string_view kOversizedSymbolCountsHex = "fdff81040000000000fcff0f01";   // HLIT=31, HDIST=31 -> 320 code lengths
inline constexpr std::string_view kEmptyDistanceTreeHex     = "05c081080000000020b6fda54e";   // all-literal block, no distance codes
inline constexpr std::string_view kIncompleteLiteralTreeHex = "05c08108000000c030b6df1fea23"; // under-subscribed literal tree

} // namespace

const boost::ut::suite<"Compression"> _compression_tests = [] {
    if (std::getenv("DISABLE_SENSITIVE_TESTS") == nullptr) {
        boost::ext::ut::cfg<override> = {.tag = {"visual"}};
    }

    "consteval encoder selects fixed, dynamic, and stored blocks"_test = [] {
        expect(eq(std::to_integer<unsigned>(kEmptyGzip[10]) & 7U, 3U));
        expect(eq(std::to_integer<unsigned>(kDynamicGzip[10]) & 7U, 5U));
        expect(eq(std::to_integer<unsigned>(kNoiseGzip[10]) & 7U, 1U));
        expect(lt(kDynamicGzip.size(), kDynamicInput.size()));
        expect(eq(kNoiseGzip.size(), kNoiseInput.size() + 23UZ));
    };

    "consteval gzip round-trips empty, text, binary, dynamic, and stored resources"_test = [] {
        expectGeneratedRoundTrip(kEmptyGzip, bytesOf(kEmptyInput));
        expectGeneratedRoundTrip(kOneByteGzip, bytesOf(kOneByteInput));
        expectGeneratedRoundTrip(kAllBytesGzip, bytesOf(kAllBytesInput));
        expectGeneratedRoundTrip(kDynamicGzip, bytesOf(kDynamicInput));
        expectGeneratedRoundTrip(kNoiseGzip, bytesOf(kNoiseInput));
        expectGeneratedRoundTrip(kCompressedText.compressed(), asBytes(kResourceText));
    };

    "compression levels control match effort and preserve compatible output"_test = [] {
        expect(kFastLevelGzip[8] == std::byte{0x04});
        expect(kBalancedLevelGzip[8] == std::byte{0x00});
        expect(kBestLevelGzip[8] == std::byte{0x02});
        expectGeneratedRoundTrip(kFastLevelGzip, bytesOf(kLazyInput));
        expectGeneratedRoundTrip(kBalancedLevelGzip, bytesOf(kLazyInput));
        expectGeneratedRoundTrip(kBestLevelGzip, bytesOf(kLazyInput));
        expectGeneratedRoundTrip(kFastSearchGzip, bytesOf(kFastSearchInput));
        expectGeneratedRoundTrip(kBalancedSearchGzip, bytesOf(kFastSearchInput));
        expectGeneratedRoundTrip(kDeepBalancedGzip, bytesOf(kBestSearchInput));
        expectGeneratedRoundTrip(kBestSearchGzip, bytesOf(kBestSearchInput));

        static constexpr auto fastBlob = gr::compression::makeCompressedBlob<kLazyInput, gr::compression::CompressionLevel::fast>();
        static constexpr auto bestText = gr::compression::makeCompressedText<kLazyResource, gr::compression::CompressionLevel::best>();
        expectBytes(fastBlob.bytes(), bytesOf(kLazyInput));
        expect(eq(bestText.view(), kLazyResource.view()));
    };

    "gzip text literal creates a lazy compressed resource"_test = [] {
        static constexpr auto literal  = "GNU Radio gzip literal"_gzip;
        static constexpr auto balanced = gr::compression::makeCompressedText<"GNU Radio gzip literal", gr::compression::CompressionLevel::balanced>();
        static_assert(literal.uncompressedSize() == std::string_view{"GNU Radio gzip literal"}.size());
        static_assert(literal.compressed()[0] == std::byte{0x1f});
        static_assert(literal.compressed()[1] == std::byte{0x8b});
        static_assert(literal.compressed() == balanced.compressed());
        expect(eq(literal.view(), std::string_view{"GNU Radio gzip literal"}));
    };

    "raw DEFLATE decoder handles stored blocks larger than 64 KiB"_test = [] {
        const auto plain      = makeRuntimeNoise(70'000UZ);
        const auto compressed = makeStoredDeflate(plain);
        expectDecoded(compressed, gr::compression::Format::rawDeflate, plain);
        expect(eq(std::to_integer<unsigned>(compressed[0]) & 1U, 0U));
        expect(eq(std::to_integer<unsigned>(compressed[65'540UZ]) & 1U, 1U));
    };

    "raw DEFLATE decoder handles external fixed, dynamic, and mixed blocks"_test = [] {
        const auto phrase = repeat("The quick brown fox jumps over the lazy dog. ", 20);
        expectDecoded(bytesFromHex(kFixedRawHex), gr::compression::Format::rawDeflate, asBytes(phrase));

        const auto skewed = repeat(kSkewedPattern, 80);
        expectDecoded(bytesFromHex(kDynamicRawHex), gr::compression::Format::rawDeflate, asBytes(skewed));

        const auto mixed = repeat("alpha beta gamma ", 10) + repeat("0123456789", 15);
        expectDecoded(bytesFromHex(kMultiBlockRawHex), gr::compression::Format::rawDeflate, asBytes(mixed));
    };

    "raw DEFLATE decoder handles maximum match length and distance"_test = [] {
        const auto maximumLength = bytesFromHex("731c0500");
        const auto repeated      = repeat("A", 259);
        expectDecoded(maximumLength, gr::compression::Format::rawDeflate, asBytes(repeated));

        auto history               = makeRuntimeNoise(32768UZ);
        auto compressed            = makeStoredDeflate(history);
        compressed[0]              = std::byte{0x00};
        const auto maximumDistance = bytesFromHex("03deff0f00");
        compressed.insert(compressed.end(), maximumDistance.begin(), maximumDistance.end());
        auto expected = history;
        expected.insert(expected.end(), history.begin(), history.begin() + 3);
        expectDecoded(compressed, gr::compression::Format::rawDeflate, expected);
    };

    "zlib decoder validates dynamic streams and Adler-32"_test = [] {
        const auto plain      = repeat(kSkewedPattern, 80);
        auto       compressed = bytesFromHex("78da");
        const auto deflate    = bytesFromHex(kDynamicRawHex);
        compressed.insert(compressed.end(), deflate.begin(), deflate.end());
        const auto trailer = bytesFromHex("3e361440");
        compressed.insert(compressed.end(), trailer.begin(), trailer.end());
        expectDecoded(compressed, gr::compression::Format::zlib, asBytes(plain));

        compressed.back() ^= std::byte{0x01};
        const auto invalid = gr::compression::decompress(compressed, gr::compression::Format::zlib);
        expect(!invalid);
        expect(invalid.error() == gr::compression::Error::checksumMismatch);

        const auto dictionary = bytesFromHex("782000000000");
        const auto required   = gr::compression::decompress(dictionary, gr::compression::Format::zlib);
        expect(!required);
        expect(required.error() == gr::compression::Error::presetDictionaryRequired);
    };

    "gzip decoder handles optional fields and concatenated members"_test = [] {
        const auto phrase      = repeat("The quick brown fox jumps over the lazy dog. ", 20);
        const auto fixedRaw    = bytesFromHex(kFixedRawHex);
        const auto firstMember = makeGzipWithOptionalFields(fixedRaw, asBytes(phrase));
        expectDecoded(firstMember, gr::compression::Format::gzip, asBytes(phrase));

        auto concatenated = firstMember;
        concatenated.insert(concatenated.end(), kOneByteGzip.begin(), kOneByteGzip.end());
        const auto expected = phrase + "x";
        expectDecoded(concatenated, gr::compression::Format::gzip, asBytes(expected));

        auto invalidHeaderCrc = firstMember;
        invalidHeaderCrc[12] ^= std::byte{0x01};
        const auto invalid = gr::compression::decompress(invalidHeaderCrc, gr::compression::Format::gzip);
        expect(!invalid);
        expect(invalid.error() == gr::compression::Error::checksumMismatch);

        auto       invalidHistory = copyBytes(kOneByteGzip);
        const auto secondMember   = makeGzipWithOptionalFields(bytesFromHex("030200"), asBytes("AAA"));
        invalidHistory.insert(invalidHistory.end(), secondMember.begin(), secondMember.end());
        const auto historyError = gr::compression::decompress(invalidHistory, gr::compression::Format::gzip);
        expect(!historyError);
        expect(historyError.error() == gr::compression::Error::invalidBackReference);
    };

    "caller-provided output reports insufficient capacity"_test = [] {
        std::span<std::byte> emptyOutput{};
        const auto           empty = gr::compression::decompress(kEmptyGzip, emptyOutput, gr::compression::Format::gzip);
        expect(empty.has_value());
        expect(eq(*empty, 0UZ));

        std::array<std::byte, 3> output{};
        const auto               result = gr::compression::decompress(kDynamicGzip, output, gr::compression::Format::gzip);
        expect(!result);
        expect(result.error() == gr::compression::Error::outputTooSmall);
    };

    "malformed DEFLATE streams fail diagnostically"_test = [] {
        const auto invalidType = gr::compression::decompress(bytesFromHex("07"), gr::compression::Format::rawDeflate);
        expect(!invalidType);
        expect(invalidType.error() == gr::compression::Error::invalidBlockType);

        const auto invalidStored = gr::compression::decompress(bytesFromHex("010100ffff00"), gr::compression::Format::rawDeflate);
        expect(!invalidStored);
        expect(invalidStored.error() == gr::compression::Error::invalidStoredBlockLength);

        const auto invalidTree = gr::compression::decompress(bytesFromHex("05000000"), gr::compression::Format::rawDeflate);
        expect(!invalidTree);
        expect(invalidTree.error() == gr::compression::Error::invalidHuffmanTree);

        const auto invalidReference = gr::compression::decompress(bytesFromHex("030200"), gr::compression::Format::rawDeflate);
        expect(!invalidReference);
        expect(invalidReference.error() == gr::compression::Error::invalidBackReference);

        auto trailing = bytesFromHex(kFixedRawHex);
        trailing.push_back(std::byte{0x00});
        const auto trailingResult = gr::compression::decompress(trailing, gr::compression::Format::rawDeflate);
        expect(!trailingResult);
        expect(trailingResult.error() == gr::compression::Error::trailingData);
    };

    "truncated dynamic streams fail for every strict prefix"_test = [] {
        const auto compressed = bytesFromHex(kDynamicRawHex);
        for (std::size_t size = 0; size < compressed.size(); ++size) {
            const auto result = gr::compression::decompress(std::span<const std::byte>(compressed).first(size), gr::compression::Format::rawDeflate);
            expect(!result) << "prefix size " << size;
        }
    };

    "gzip headers and trailers reject invalid metadata"_test = [] {
        auto invalidMethod     = copyBytes(kOneByteGzip);
        invalidMethod[2]       = std::byte{0x09};
        const auto methodError = gr::compression::decompress(invalidMethod, gr::compression::Format::gzip);
        expect(!methodError);
        expect(methodError.error() == gr::compression::Error::unsupportedCompressionMethod);

        auto reservedFlags    = copyBytes(kOneByteGzip);
        reservedFlags[3]      = std::byte{0x20};
        const auto flagsError = gr::compression::decompress(reservedFlags, gr::compression::Format::gzip);
        expect(!flagsError);
        expect(flagsError.error() == gr::compression::Error::reservedFlags);

        auto invalidChecksum = copyBytes(kOneByteGzip);
        invalidChecksum[invalidChecksum.size() - 8UZ] ^= std::byte{0x01};
        const auto checksumError = gr::compression::decompress(invalidChecksum, gr::compression::Format::gzip);
        expect(!checksumError);
        expect(checksumError.error() == gr::compression::Error::checksumMismatch);

        auto invalidSize = copyBytes(kOneByteGzip);
        invalidSize.back() ^= std::byte{0x01};
        const auto sizeError = gr::compression::decompress(invalidSize, gr::compression::Format::gzip);
        expect(!sizeError);
        expect(sizeError.error() == gr::compression::Error::sizeMismatch);

        auto trailing = copyBytes(kOneByteGzip);
        trailing.push_back(std::byte{0x00});
        const auto trailingError = gr::compression::decompress(trailing, gr::compression::Format::gzip);
        expect(!trailingError);
        expect(trailingError.error() == gr::compression::Error::trailingData);

        const auto truncatedName = bytesFromHex("1f8b08080000000000ff6e616d65");
        const auto nameError     = gr::compression::decompress(truncatedName, gr::compression::Format::gzip);
        expect(!nameError);
        expect(nameError.error() == gr::compression::Error::truncatedInput);
    };

    "CompressedText lazily exposes persistent text"_test = [] {
        const std::string_view view = kCompressedText;
        expect(eq(view, kResourceText));
        expect(eq(kCompressedText.view(), kResourceText));
        expect(eq(kCompressedText.str(), kResourceText));
        expect(eq(std::string_view{kCompressedText.c_str()}, kResourceText));
        expect(kCompressedText.str().data() == kCompressedText.view().data());
    };

    "CompressedBlob lazily exposes persistent binary bytes"_test = [] {
        expect(eq(kCompressedBlob.compressedSize(), kAllBytesGzip.size()));
        expect(eq(kCompressedBlob.uncompressedSize(), kAllBytesInput.size()));
        expectBytes(kCompressedBlob.bytes(), bytesOf(kAllBytesInput));
        expect(kCompressedBlob.bytes().data() == kCompressedBlob.decompress().data());
    };

    "CompressedText formatter preserves string_view formatting"_test = [] {
        const std::string_view view = kResourceText;
        expect(eq(std::format("{}", kCompressedText), std::format("{}", view)));
        expect(eq(std::format("{:>320}", kCompressedText), std::format("{:>320}", view)));
        expect(eq(std::format("{:*<320.80}", kCompressedText), std::format("{:*<320.80}", view)));
        expect(eq(std::format("{:^{}}", kCompressedText, 320), std::format("{:^{}}", view, 320)));
        expect(eq(std::format("{:.{}}", kCompressedText, 37), std::format("{:.{}}", view, 37)));
        expect(eq(std::format("[{}]", gr::compression::makeCompressedText<"">()), std::string{"[]"}));
    };

    tag("visual") / "compressed text demo"_test = [] {
        constexpr auto gzipWrapperSize = 18UZ;
        constexpr auto deflateSize     = kCompressedText.compressedSize() - gzipWrapperSize;
        const auto     decoded         = kCompressedText.view();

        expect(eq(decoded, kResourceText));
        expect(eq(decoded.size(), kCompressedText.uncompressedSize()));
        std::println("\nGNU Radio compile-time gzip demo");
        std::println("raw source:      {:>4} bytes", kResourceText.size());
        std::println("gzip member:     {:>4} bytes", kCompressedText.compressedSize());
        std::println("DEFLATE payload: {:>4} bytes", deflateSize);
        std::println("inflated output: {:>4} bytes", decoded.size());
        std::println("\n{}\n", kCompressedText);
    };

    "dynamic block headers reject out-of-range symbol counts"_test = [] {
        // HLIT/HDIST encode more symbols than RFC 1951 defines; an unchecked count overruns the code-length buffer
        const auto oversized = gr::compression::decompress(bytesFromHex(kOversizedSymbolCountsHex), gr::compression::Format::rawDeflate);
        expect(!oversized);
        expect(oversized.error() == gr::compression::Error::invalidHuffmanTree);

        std::array<std::byte, 64> output{};
        const auto                intoSpan = gr::compression::decompress(bytesFromHex(kOversizedSymbolCountsHex), output, gr::compression::Format::rawDeflate);
        expect(!intoSpan);
        expect(intoSpan.error() == gr::compression::Error::invalidHuffmanTree);
    };

    "Huffman tree completeness follows zlib"_test = [] {
        expectDecoded(bytesFromHex(kEmptyDistanceTreeHex), gr::compression::Format::rawDeflate, asBytes("A"));

        const auto incomplete = gr::compression::decompress(bytesFromHex(kIncompleteLiteralTreeHex), gr::compression::Format::rawDeflate);
        expect(!incomplete);
        expect(incomplete.error() == gr::compression::Error::invalidHuffmanTree);

        std::array<std::uint8_t, 288> oneBitLiteral{};
        oneBitLiteral[256] = 1;
        expect(gr::compression::detail::makeDecoder<288, 15>(oneBitLiteral, 257).has_value());

        std::array<std::uint8_t, 288> widerLiteral{};
        widerLiteral[256] = 2;
        expect(!gr::compression::detail::makeDecoder<288, 15>(widerLiteral, 257));

        std::array<std::uint8_t, 19> codeLengthLengths{};
        codeLengthLengths[0]      = 1;
        const auto codeLengthTree = gr::compression::detail::makeDecoder<19, 7>(codeLengthLengths, codeLengthLengths.size(), {.allowEmpty = false, .allowIncompleteSingleBitCode = false});
        expect(!codeLengthTree);
    };

    "decoded size can be queried and capped before allocating"_test = [] {
        const auto empty = gr::compression::decompress(kEmptyGzip, gr::compression::Format::gzip, 0UZ);
        expect(empty.has_value());
        if (empty) {
            expect(empty->empty());
        }

        const auto size = gr::compression::decompressedSize(kDynamicGzip, gr::compression::Format::gzip);
        expect(size.has_value());
        expect(eq(*size, kDynamicInput.size()));

        const auto cappedSize = gr::compression::decompressedSize(kDynamicGzip, gr::compression::Format::gzip, kDynamicInput.size() - 1UZ);
        expect(!cappedSize);
        expect(cappedSize.error() == gr::compression::Error::sizeLimitExceeded);

        const auto capped = gr::compression::decompress(kDynamicGzip, gr::compression::Format::gzip, kDynamicInput.size() - 1UZ);
        expect(!capped);
        expect(capped.error() == gr::compression::Error::sizeLimitExceeded);

        const auto allowed = gr::compression::decompress(kDynamicGzip, gr::compression::Format::gzip, kDynamicInput.size());
        expect(allowed.has_value());

        auto truncatedAfterPayload = copyBytes(kDynamicGzip);
        truncatedAfterPayload.pop_back();
        const auto cappedBeforeTrailer = gr::compression::decompress(truncatedAfterPayload, gr::compression::Format::gzip, kDynamicInput.size() - 1UZ);
        expect(!cappedBeforeTrailer);
        expect(cappedBeforeTrailer.error() == gr::compression::Error::sizeLimitExceeded);

        const auto uncappedTrailerError = gr::compression::decompress(truncatedAfterPayload, gr::compression::Format::gzip, kDynamicInput.size());
        expect(!uncappedTrailerError);
        expect(uncappedTrailerError.error() == gr::compression::Error::truncatedInput);

        const auto invalid = gr::compression::decompressedSize(bytesFromHex("07"), gr::compression::Format::rawDeflate);
        expect(!invalid);
        expect(invalid.error() == gr::compression::Error::invalidBlockType);
    };

    "runtime encoder round-trips every format and level"_test = [] {
        const auto text  = repeat("runtime deflate payload with repeated words. ", 200);
        const auto noise = makeRuntimeNoise(4096UZ);
        for (const auto plain : {asBytes(text), std::span<const std::byte>(noise), bytesOf(kEmptyInput)}) {
            for (const auto format : {gr::compression::Format::rawDeflate, gr::compression::Format::zlib, gr::compression::Format::gzip}) {
                for (const auto level : {gr::compression::CompressionLevel::fast, gr::compression::CompressionLevel::balanced, gr::compression::CompressionLevel::best}) {
                    const auto packed = gr::compression::compress(plain, format, level);
                    expect(packed.has_value());
                    if (!packed) {
                        continue;
                    }
                    expectDecoded(*packed, format, plain);
                }
            }
        }
    };

    "streaming decoder handles narrow input and output spans"_test = [] {
        const auto phrase      = repeat("The quick brown fox jumps over the lazy dog. ", 20);
        const auto firstMember = makeGzipWithOptionalFields(bytesFromHex(kFixedRawHex), asBytes(phrase));
        auto       gzip        = firstMember;
        gzip.insert(gzip.end(), kOneByteGzip.begin(), kOneByteGzip.end());

        auto expectedGzip = copyBytes(asBytes(phrase));
        expectedGzip.push_back(std::byte{'x'});
        const auto decodedGzip = streamDecompress(gzip, gr::compression::Format::gzip, 1UZ, 3UZ);
        expect(decodedGzip.has_value());
        if (decodedGzip) {
            expectBytes(*decodedGzip, expectedGzip);
        }

        const auto mixed      = repeat("alpha beta gamma ", 10) + repeat("0123456789", 15);
        const auto decodedRaw = streamDecompress(bytesFromHex(kMultiBlockRawHex), gr::compression::Format::rawDeflate, 1UZ, 5UZ);
        expect(decodedRaw.has_value());
        if (decodedRaw) {
            expectBytes(*decodedRaw, asBytes(mixed));
        }

        const auto packedZlib = gr::compression::compress(bytesOf(kAllBytesInput), gr::compression::Format::zlib, gr::compression::CompressionLevel::best);
        expect(packedZlib.has_value());
        if (packedZlib) {
            const auto decodedZlib = streamDecompress(*packedZlib, gr::compression::Format::zlib, 2UZ, 7UZ);
            expect(decodedZlib.has_value());
            if (decodedZlib) {
                expectBytes(*decodedZlib, bytesOf(kAllBytesInput));
            }
        }
    };

    "streaming decoder waits for final flush before reporting truncation"_test = [] {
        auto truncated = copyBytes(kDynamicGzip);
        truncated.pop_back();

        gr::compression::StreamDecompressor decoder{gr::compression::Format::gzip};
        std::vector<std::byte>              output(kDynamicInput.size() + 16UZ);
        const auto                          partial = decoder.process(truncated, output, gr::compression::Flush::none);
        expect(partial.has_value());
        if (!partial) {
            return;
        }
        expect(!partial->finished());
        expect(eq(partial->consumed, truncated.size()));

        const auto strict = decoder.process({}, output, gr::compression::Flush::finish);
        expect(!strict);
        expect(strict.error() == gr::compression::Error::truncatedInput);
    };

    "streaming decoder rewinds unconsumed input and callers must re-offer it"_test = [] {
        const auto text   = repeat("chunk boundary payload with repetition ", 300);
        const auto packed = gr::compression::compress(asBytes(text), gr::compression::Format::gzip, gr::compression::CompressionLevel::best);
        expect(packed.has_value());
        if (!packed) {
            return;
        }

        // feeding one byte at a time forces the decoder to rewind mid-symbol on almost every call
        for (const std::size_t sourceChunk : {1UZ, 2UZ, 3UZ, 5UZ}) {
            gr::compression::StreamDecompressor decoder{gr::compression::Format::gzip};
            std::vector<std::byte>              pending;
            std::vector<std::byte>              decoded;
            std::vector<std::byte>              buffer(64UZ);
            std::size_t                         offset  = 0UZ;
            bool                                rewound = false;

            while (!decoder.finished()) {
                const auto take = std::min(sourceChunk, packed->size() - offset);
                pending.insert(pending.end(), packed->begin() + static_cast<std::ptrdiff_t>(offset), packed->begin() + static_cast<std::ptrdiff_t>(offset + take));
                offset += take;
                const auto finish = offset == packed->size();

                const auto state = decoder.process(pending, buffer, finish ? gr::compression::Flush::finish : gr::compression::Flush::none);
                expect(state.has_value());
                if (!state) {
                    break;
                }
                rewound = rewound || state->consumed < pending.size();
                decoded.insert(decoded.end(), buffer.begin(), buffer.begin() + static_cast<std::ptrdiff_t>(state->produced));
                pending.erase(pending.begin(), pending.begin() + static_cast<std::ptrdiff_t>(state->consumed));
                if (state->finished()) {
                    break;
                }
            }
            expect(rewound) << "chunk " << sourceChunk;
            expectBytes(decoded, asBytes(text));
        }
    };

    "streaming decoder enforces the decoded size cap across calls"_test = [] {
        const auto capped = streamDecompress(kDynamicGzip, gr::compression::Format::gzip, 3UZ, 11UZ, kDynamicInput.size() - 1UZ);
        expect(!capped);
        expect(capped.error() == gr::compression::Error::sizeLimitExceeded);
    };

    "streaming encoder produces decodable raw, zlib, and gzip streams"_test = [] {
        const auto text  = repeat("streaming compression payload with repeated words and block boundaries. ", 40);
        const auto plain = copyBytes(asBytes(text));
        for (const auto format : {gr::compression::Format::rawDeflate, gr::compression::Format::zlib, gr::compression::Format::gzip}) {
            for (const auto level : {gr::compression::CompressionLevel::fast, gr::compression::CompressionLevel::balanced, gr::compression::CompressionLevel::best}) {
                const auto packed = streamCompress(plain, format, level, 13UZ, 2UZ, 31UZ);
                expect(packed.has_value());
                if (!packed) {
                    continue;
                }
                expectDecoded(*packed, format, plain);

                const auto decoded = streamDecompress(*packed, format, 3UZ, 5UZ);
                expect(decoded.has_value());
                if (decoded) {
                    expectBytes(*decoded, plain);
                }
            }
        }
    };

    "stored blocks fill the size bound exactly at every block boundary"_test = [] {
        for (const std::size_t size : {0UZ, 1UZ, 65534UZ, 65535UZ, 65536UZ, 131070UZ}) {
            const auto plain = makeRuntimeNoise(size);
            for (const auto format : {gr::compression::Format::rawDeflate, gr::compression::Format::zlib, gr::compression::Format::gzip}) {
                const auto packed = gr::compression::compress(plain, format, gr::compression::CompressionLevel::best);
                expect(packed.has_value());
                if (!packed) {
                    continue;
                }
                expect(le(packed->size(), gr::compression::detail::compressBound(size, format))) << "size " << size;
                expectDecoded(*packed, format, plain);
            }
        }
    };

    "runtime and consteval encoders agree byte for byte"_test = [] {
        const auto runtime = gr::compression::compress(bytesOf(kDynamicInput), gr::compression::Format::gzip, gr::compression::CompressionLevel::balanced);
        expect(runtime.has_value());
        if (runtime) {
            expectBytes(*runtime, kDynamicGzip);
        }
    };

    "allocating overloads honour a polymorphic memory resource"_test = [] {
        std::array<std::byte, 4096>         storage{};
        std::pmr::monotonic_buffer_resource arena{storage.data(), storage.size(), std::pmr::null_memory_resource()};

        const auto decoded = gr::compression::decompress(kAllBytesGzip, gr::compression::Format::gzip, &arena);
        expect(decoded.has_value());
        if (decoded) {
            expectBytes(*decoded, bytesOf(kAllBytesInput));
            expect(decoded->get_allocator().resource() == &arena);
        }

        const auto packed = gr::compression::compress(bytesOf(kAllBytesInput), &arena, gr::compression::Format::gzip);
        expect(packed.has_value());
        if (packed) {
            expect(packed->get_allocator().resource() == &arena);
            expectDecoded(*packed, gr::compression::Format::gzip, bytesOf(kAllBytesInput));
        }

        const auto capped = gr::compression::decompress(kAllBytesGzip, gr::compression::Format::gzip, &arena, 8UZ);
        expect(!capped);
        expect(capped.error() == gr::compression::Error::sizeLimitExceeded);
    };

    "mutated streams never decode incorrectly and never crash"_test = [] {
        const std::vector<std::vector<std::byte>> seeds{copyBytes(kDynamicGzip), copyBytes(kAllBytesGzip), copyBytes(kOneByteGzip), bytesFromHex(kFixedRawHex), bytesFromHex(kDynamicRawHex), bytesFromHex(kMultiBlockRawHex)};

        std::uint64_t state      = 0x1234'5678'9abc'def0ULL;
        const auto    nextRandom = [&state] {
            state ^= state << 13U;
            state ^= state >> 7U;
            state ^= state << 17U;
            return state;
        };
        std::vector<std::byte> output(1UZ << 16U);
        std::size_t            decoded = 0;
        for (std::size_t iteration = 0; iteration < 200'000UZ; ++iteration) {
            auto stream = seeds[nextRandom() % seeds.size()];
            for (std::size_t mutation = 0; mutation < 1UZ + nextRandom() % 4UZ; ++mutation) {
                stream[nextRandom() % stream.size()] ^= static_cast<std::byte>(1U << (nextRandom() % 8U));
            }
            const auto format = std::array{gr::compression::Format::rawDeflate, gr::compression::Format::zlib, gr::compression::Format::gzip}[nextRandom() % 3UZ];
            if (gr::compression::decompress(stream, output, format)) {
                ++decoded;
            }
        }
        expect(gt(decoded, 0UZ)) << "mutations never reached the decoder";
    };

#ifndef __EMSCRIPTEN__
    "GNU gzip accepts generated streams and produces decodable streams"_test = [] {
        expect(gzipAvailable());
        if (!gzipAvailable()) {
            return;
        }

        TemporaryFiles files;
        const auto     sourcePath   = files.add("_source.bin");
        const auto     externalPath = files.add("_external.gz");

        const auto validateGenerated = [&files]<std::size_t PackedSize>(std::string_view name, const std::array<std::byte, PackedSize>& packed, std::span<const std::byte> plain) {
            const auto generatedPath = files.add("_" + std::string(name) + ".gz");
            const auto decodedPath   = files.add("_" + std::string(name) + ".out");
            expect(writeBytes(generatedPath, packed));
            expect(eq(0, std::system(("gzip -t " + shellQuote(generatedPath)).c_str()))) << name;
            expect(eq(0, std::system(("gzip -dc " + shellQuote(generatedPath) + " > " + shellQuote(decodedPath)).c_str()))) << name;
            expectBytes(readBytes(decodedPath), plain);
        };

        validateGenerated("empty", kEmptyGzip, bytesOf(kEmptyInput));
        validateGenerated("fixed", kOneByteGzip, bytesOf(kOneByteInput));
        validateGenerated("binary", kAllBytesGzip, bytesOf(kAllBytesInput));
        validateGenerated("dynamic", kDynamicGzip, bytesOf(kDynamicInput));
        validateGenerated("stored", kNoiseGzip, bytesOf(kNoiseInput));
        validateGenerated("level_fast", kFastLevelGzip, bytesOf(kLazyInput));
        validateGenerated("level_balanced", kBalancedLevelGzip, bytesOf(kLazyInput));
        validateGenerated("level_best", kBestLevelGzip, bytesOf(kLazyInput));

        const auto runtimePacked = gr::compression::compress(bytesOf(kDynamicInput), gr::compression::Format::gzip, gr::compression::CompressionLevel::best);
        expect(runtimePacked.has_value());
        if (runtimePacked) {
            const auto runtimePath = files.add("_runtime.gz");
            expect(writeBytes(runtimePath, *runtimePacked));
            expect(eq(0, std::system(("gzip -t " + shellQuote(runtimePath)).c_str())));
        }

        expect(writeBytes(sourcePath, kDynamicInput));
        expect(eq(0, std::system(("gzip -9 -c " + shellQuote(sourcePath) + " > " + shellQuote(externalPath)).c_str())));
        const auto external = readBytes(externalPath);
        expectDecoded(external, gr::compression::Format::gzip, bytesOf(kDynamicInput));
    };
#endif
};

int main() { return 0; }
