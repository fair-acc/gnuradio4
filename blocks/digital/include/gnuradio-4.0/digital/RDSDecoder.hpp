#ifndef GNURADIO_DIGITAL_RDSDECODER_HPP
#define GNURADIO_DIGITAL_RDSDECODER_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::digital {

namespace rds {

enum class Offset : std::uint8_t { A, B, C, CPrime, D };

inline constexpr std::size_t                    kBlockBits           = 26UZ;
inline constexpr std::size_t                    kCheckwordBits       = 10UZ;
inline constexpr std::uint32_t                  kBlockMask           = (1U << kBlockBits) - 1U;
inline constexpr std::uint32_t                  kCheckwordMask       = (1U << kCheckwordBits) - 1U;
inline constexpr std::uint32_t                  kGeneratorPolynomial = 0b101'1011'1001U;
inline constexpr std::array<std::uint32_t, 5UZ> kOffsetWords{0x0FCU, 0x198U, 0x168U, 0x350U, 0x1B4U};

[[nodiscard]] constexpr std::uint32_t cyclicRedundancy(std::uint16_t information) noexcept {
    std::uint32_t remainder = static_cast<std::uint32_t>(information) << kCheckwordBits;
    for (std::size_t bit = kBlockBits - 1UZ; bit >= kCheckwordBits; --bit) {
        if (((remainder >> bit) & 1U) != 0U) {
            remainder ^= kGeneratorPolynomial << (bit - kCheckwordBits);
        }
    }
    return remainder;
}

[[nodiscard]] constexpr std::uint32_t encodeBlock(std::uint16_t information, Offset offset) noexcept { return (static_cast<std::uint32_t>(information) << kCheckwordBits) | (cyclicRedundancy(information) ^ kOffsetWords[std::to_underlying(offset)]); }

[[nodiscard]] constexpr std::uint16_t informationOf(std::uint32_t block) noexcept { return static_cast<std::uint16_t>((block & kBlockMask) >> kCheckwordBits); }

[[nodiscard]] constexpr std::optional<Offset> offsetOf(std::uint32_t block) noexcept {
    const std::uint32_t offsetWord = (block ^ cyclicRedundancy(informationOf(block))) & kCheckwordMask;
    const auto          match      = std::ranges::find(kOffsetWords, offsetWord);
    if (match == kOffsetWords.end()) {
        return std::nullopt;
    }
    return static_cast<Offset>(std::distance(kOffsetWords.begin(), match));
}

[[nodiscard]] constexpr std::size_t groupPosition(Offset offset) noexcept { return offset == Offset::D ? 3UZ : std::min<std::size_t>(std::to_underlying(offset), 2UZ); }

} // namespace rds

GR_REGISTER_BLOCK(gr::digital::RDSDecoder)

struct RDSDecoder : Block<RDSDecoder> {
    using Description = Doc<R""(@brief Radio Data System (RDS) decoder: reads the station data from the RDS bits, passes the bits on unchanged and publishes the data as a tag.

The input is the differentially decoded RDS bit stream.
It synchronises on two consecutive 26-bit blocks with valid checkwords and searches again after 8 invalid blocks in a row; it reads only groups of four valid blocks and corrects no bit errors.
When a group changes a value, a tag on its last bit holds all values known so far: rds_programme_identification (uint16), rds_programme_type (uint8) and, once complete, rds_programme_service_name and rds_radio_text (strings).
Characters outside printable ASCII become '?'.
A new programme identification or a reset forgets the previous station.

 * IEC 62106-1:2018 and IEC 62106-2:2021, "Radio data system (RDS) - VHF/FM sound broadcasting in the frequency range from 64,0 MHz to 108,0 MHz", parts 1 and 2.)"">;

    struct Station {
        std::uint16_t programmeIdentification{};
        std::uint8_t  programmeType{};
        std::string   programmeServiceName;
        std::string   radioText;

        bool operator==(const Station&) const = default;
    };

    static constexpr std::size_t      kInvalidBlocksToLoseSync    = 8UZ;
    static constexpr std::size_t      kProgrammeServiceNameLength = 8UZ;
    static constexpr std::size_t      kRadioTextLength            = 64UZ;
    static constexpr std::size_t      kRadioTextSegments          = 16UZ;
    static constexpr std::uint8_t     kEndOfText                  = 0x0DU;
    static constexpr std::string_view kProgrammeIdentificationKey = "rds_programme_identification";
    static constexpr std::string_view kProgrammeTypeKey           = "rds_programme_type";
    static constexpr std::string_view kProgrammeServiceNameKey    = "rds_programme_service_name";
    static constexpr std::string_view kRadioTextKey               = "rds_radio_text";

    PortIn<std::uint8_t>  in;
    PortOut<std::uint8_t> out;

    GR_MAKE_REFLECTABLE(RDSDecoder, in, out);

    std::uint64_t                                         _receivedBits{};
    std::size_t                                           _bitsSinceBlock{};
    std::size_t                                           _nextPosition{};
    std::size_t                                           _invalidBlocks{};
    bool                                                  _synchronised = false;
    std::array<std::uint16_t, 4UZ>                        _group{};
    std::uint32_t                                         _validBlocks{};
    bool                                                  _thirdBlockIsCPrime = false;
    std::array<std::uint8_t, kProgrammeServiceNameLength> _programmeServiceNameCodes{};
    std::uint32_t                                         _receivedNameSegments{};
    std::array<std::uint8_t, kRadioTextLength>            _radioTextCodes{};
    std::uint32_t                                         _receivedTextSegments{};
    bool                                                  _radioTextFlag     = false;
    bool                                                  _radioTextVersionB = false;
    Station                                               _station;
    std::optional<Station>                                _publishedStation;

    void reset() {
        _receivedBits       = 0U;
        _bitsSinceBlock     = 0UZ;
        _nextPosition       = 0UZ;
        _invalidBlocks      = 0UZ;
        _synchronised       = false;
        _validBlocks        = 0U;
        _thirdBlockIsCPrime = false;
        forgetStation();
        _publishedStation.reset();
    }

    [[nodiscard]] std::uint8_t processOne(std::uint8_t bit) {
        _receivedBits = (_receivedBits << 1U) | (bit != 0U ? 1U : 0U);
        ++_bitsSinceBlock;
        if (!_synchronised) {
            searchBlockPair();
        } else if (_bitsSinceBlock == rds::kBlockBits) {
            checkExpectedBlock();
        }
        return bit;
    }

    void searchBlockPair() {
        const std::uint32_t              block          = receivedBlock(0UZ);
        const std::uint32_t              previousBlock  = receivedBlock(1UZ);
        const std::optional<rds::Offset> offset         = rds::offsetOf(block);
        const std::optional<rds::Offset> previousOffset = rds::offsetOf(previousBlock);
        if (!offset || !previousOffset || rds::groupPosition(*offset) != (rds::groupPosition(*previousOffset) + 1UZ) % 4UZ) {
            return;
        }
        _synchronised  = true;
        _invalidBlocks = 0UZ;
        _validBlocks   = 0U;
        storeBlock(previousBlock, *previousOffset);
        storeBlock(block, *offset);
    }

    void checkExpectedBlock() {
        const std::uint32_t              block  = receivedBlock(0UZ);
        const std::optional<rds::Offset> offset = rds::offsetOf(block);
        if (offset && rds::groupPosition(*offset) == _nextPosition) {
            _invalidBlocks = 0UZ;
            storeBlock(block, *offset);
            return;
        }
        ++_invalidBlocks;
        _synchronised = _invalidBlocks < kInvalidBlocksToLoseSync;
        finishBlock(_nextPosition);
    }

    void storeBlock(std::uint32_t block, rds::Offset offset) {
        const std::size_t position = rds::groupPosition(offset);
        _group[position]           = rds::informationOf(block);
        _validBlocks |= 1U << position;
        if (position == 2UZ) {
            _thirdBlockIsCPrime = offset == rds::Offset::CPrime;
        }
        finishBlock(position);
    }

    void finishBlock(std::size_t position) {
        if (position == 3UZ) {
            if (_validBlocks == 0xFU) {
                decodeGroup();
            }
            _validBlocks = 0U;
        }
        _nextPosition   = (position + 1UZ) % 4UZ;
        _bitsSinceBlock = 0UZ;
    }

    void decodeGroup() {
        const std::uint32_t blockB   = _group[1];
        const bool          versionB = ((blockB >> 11U) & 1U) != 0U;
        if (versionB != _thirdBlockIsCPrime) {
            return;
        }
        if (_group[0] != _station.programmeIdentification) {
            forgetStation();
        }
        _station.programmeIdentification = _group[0];
        _station.programmeType           = static_cast<std::uint8_t>((blockB >> 5U) & 0x1FU);
        const std::uint32_t groupType    = blockB >> 12U;
        if (groupType == 0U) {
            decodeProgrammeServiceName(blockB);
        } else if (groupType == 2U) {
            decodeRadioText(blockB, versionB);
        }
        if (_publishedStation != _station) {
            this->publishTag(stationTag(_station), 0UZ);
            _publishedStation = _station;
        }
    }

    void forgetStation() {
        _programmeServiceNameCodes.fill(0U);
        _receivedNameSegments = 0U;
        _radioTextCodes.fill(0U);
        _receivedTextSegments = 0U;
        _radioTextFlag        = false;
        _radioTextVersionB    = false;
        _station              = {};
    }

    void decodeProgrammeServiceName(std::uint32_t blockB) {
        const std::size_t segment = blockB & 0x3U;
        storeCharacters(std::span(_programmeServiceNameCodes).subspan(2UZ * segment, 2UZ), _group[3]);
        _receivedNameSegments |= 1U << segment;
        if (_receivedNameSegments == 0xFU) {
            _station.programmeServiceName = printableText(_programmeServiceNameCodes);
            _receivedNameSegments         = 0U;
        }
    }

    void decodeRadioText(std::uint32_t blockB, bool versionB) {
        const bool        textFlag = ((blockB >> 4U) & 1U) != 0U;
        const std::size_t segment  = blockB & 0xFU;
        if (textFlag != _radioTextFlag || versionB != _radioTextVersionB) {
            _radioTextCodes.fill(0U);
            _receivedTextSegments = 0U;
            _radioTextFlag        = textFlag;
            _radioTextVersionB    = versionB;
        }
        const std::size_t             charactersPerSegment = versionB ? 2UZ : 4UZ;
        const std::span<std::uint8_t> text                 = std::span(_radioTextCodes).first(kRadioTextSegments * charactersPerSegment);
        if (versionB) {
            storeCharacters(text.subspan(2UZ * segment, 2UZ), _group[3]);
        } else {
            storeCharacters(text.subspan(4UZ * segment, 2UZ), _group[2]);
            storeCharacters(text.subspan(4UZ * segment + 2UZ, 2UZ), _group[3]);
        }
        _receivedTextSegments |= 1U << segment;
        const auto          textEnd        = std::ranges::find(text, kEndOfText);
        const auto          length         = static_cast<std::size_t>(std::distance(text.begin(), textEnd));
        const std::size_t   neededSegments = textEnd == text.end() ? kRadioTextSegments : length / charactersPerSegment + 1UZ;
        const std::uint32_t neededMask     = (1U << neededSegments) - 1U;
        if ((_receivedTextSegments & neededMask) == neededMask) {
            _station.radioText    = printableText(text.first(length));
            _receivedTextSegments = 0U;
        }
    }

    [[nodiscard]] std::uint32_t receivedBlock(std::size_t blocksAgo) const noexcept { return static_cast<std::uint32_t>(_receivedBits >> (blocksAgo * rds::kBlockBits)) & rds::kBlockMask; }

    static void storeCharacters(std::span<std::uint8_t> characters, std::uint32_t block) noexcept {
        characters[0] = static_cast<std::uint8_t>(block >> 8U);
        characters[1] = static_cast<std::uint8_t>(block & 0xFFU);
    }

    [[nodiscard]] static std::string printableText(std::span<const std::uint8_t> codes) {
        std::string text(codes.size(), ' ');
        std::ranges::transform(codes, text.begin(), [](std::uint8_t code) { return code >= 0x20U && code <= 0x7EU ? static_cast<char>(code) : '?'; });
        text.erase(text.find_last_not_of(' ') + 1UZ);
        return text;
    }

    [[nodiscard]] static property_map stationTag(const Station& station) {
        property_map stationData{{kProgrammeIdentificationKey, station.programmeIdentification}, {kProgrammeTypeKey, station.programmeType}};
        if (!station.programmeServiceName.empty()) {
            stationData.insert_or_assign(kProgrammeServiceNameKey, station.programmeServiceName);
        }
        if (!station.radioText.empty()) {
            stationData.insert_or_assign(kRadioTextKey, station.radioText);
        }
        return stationData;
    }
};

} // namespace gr::digital

#endif // GNURADIO_DIGITAL_RDSDECODER_HPP
