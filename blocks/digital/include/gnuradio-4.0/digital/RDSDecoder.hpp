#ifndef GNURADIO_DIGITAL_RDSDECODER_HPP
#define GNURADIO_DIGITAL_RDSDECODER_HPP

#include <algorithm>
#include <array>
#include <bitset>
#include <cstddef>
#include <cstdint>
#include <format>
#include <iterator>
#include <map>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

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
inline constexpr std::uint8_t                   kLineFeed         = 0x0AU;
inline constexpr std::uint8_t                   kEndOfText        = 0x0DU;
inline constexpr std::uint8_t                   kFirstCharacter   = 0x20U;
inline constexpr char                           kMissingCharacter = '_';

// clang-format off
inline constexpr std::array<std::string_view, 224UZ> kCharacters{
    " ", "!", "\"", "#", "¤", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?",
    "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "―", "_",
    "‖", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o",
    "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "¯", "",
    "á", "à", "é", "è", "í", "ì", "ó", "ò", "ú", "ù", "Ñ", "Ç", "Ş", "ß", "¡", "Ĳ",
    "â", "ä", "ê", "ë", "î", "ï", "ô", "ö", "û", "ü", "ñ", "ç", "ş", "ğ", "ı", "ĳ",
    "ª", "α", "©", "‰", "Ğ", "ě", "ň", "ő", "π", "€", "£", "$", "←", "↑", "→", "↓",
    "º", "¹", "²", "³", "±", "İ", "ń", "ű", "µ", "¿", "÷", "°", "¼", "½", "¾", "§",
    "Á", "À", "É", "È", "Í", "Ì", "Ó", "Ò", "Ú", "Ù", "Ř", "Č", "Š", "Ž", "Đ", "Ŀ",
    "Â", "Ä", "Ê", "Ë", "Î", "Ï", "Ô", "Ö", "Û", "Ü", "ř", "č", "š", "ž", "đ", "ŀ",
    "Ã", "Å", "Æ", "Œ", "ŷ", "Ý", "Õ", "Ø", "Þ", "Ŋ", "Ŕ", "Ć", "Ś", "Ź", "Ŧ", "ð",
    "ã", "å", "æ", "œ", "ŵ", "ý", "õ", "ø", "þ", "ŋ", "ŕ", "ć", "ś", "ź", "ŧ", "",
};
// clang-format on

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

[[nodiscard]] constexpr std::optional<std::uint32_t> correctBurst(std::uint32_t block, Offset expected) noexcept {
    for (std::size_t bit = 0UZ; bit < kBlockBits; ++bit) {
        for (const std::uint32_t burst : {0b1U, 0b11U}) {
            const std::uint32_t candidate = block ^ ((burst << bit) & kBlockMask);
            if (offsetOf(candidate) == expected) {
                return candidate;
            }
        }
    }
    return std::nullopt;
}

[[nodiscard]] constexpr std::string_view characterOf(std::uint8_t code) noexcept {
    if (code >= kFirstCharacter) {
        return kCharacters[static_cast<std::size_t>(code - kFirstCharacter)];
    }
    return code == kLineFeed ? " " : "";
}

[[nodiscard]] constexpr std::optional<float> frequencyOf(std::uint32_t code) noexcept { return code >= 1U && code <= 204U ? std::optional(static_cast<float>(875U + code) / 10.f) : std::nullopt; }

template<std::size_t N>
struct Text {
    std::array<std::uint8_t, N> codes{};
    std::bitset<N>              received;

    void store(std::size_t first, std::uint16_t pair) noexcept {
        if (first + 1UZ >= N) {
            return;
        }
        codes[first]       = static_cast<std::uint8_t>(pair >> 8U);
        codes[first + 1UZ] = static_cast<std::uint8_t>(pair & 0xFFU);
        received.set(first);
        received.set(first + 1UZ);
    }

    [[nodiscard]] std::string text(std::size_t first = 0UZ, std::size_t count = N) const {
        std::string result;
        std::size_t kept = 0UZ;
        for (std::size_t i = 0UZ; i < std::min(N, first + count); ++i) {
            if (received[i] && codes[i] == kEndOfText) {
                break;
            }
            if (i < first) {
                continue;
            }
            if (!received[i]) {
                result += kMissingCharacter;
                continue;
            }
            result += characterOf(codes[i]);
            if (codes[i] != ' ') {
                kept = result.size();
            }
        }
        result.resize(kept);
        return result;
    }
};

} // namespace rds

GR_REGISTER_BLOCK(gr::digital::RDSDecoder)

struct RDSDecoder : Block<RDSDecoder> {
    using Description = Doc<R""(@brief Radio Data System (RDS) decoder: reads the station data from the RDS bits, passes the bits on unchanged and publishes the data as tags.

 * IEC 62106-1:2018 and IEC 62106-2:2021, "Radio data system (RDS) - VHF/FM sound broadcasting in the frequency range from 64,0 MHz to 108,0 MHz", parts 1 and 2.
 * ISO 14819-1:2013, "Intelligent transport systems - Traffic and travel information messages via traffic message coding - Part 1: Coding protocol for Radio Data System - Traffic Message Channel (RDS-TMC) using ALERT-C".
 * RDS Forum, "RadioText Plus (RT+)", open data application 4BD7.)"">;

    struct OtherNetwork {
        std::string                 programmeServiceName;
        std::vector<float>          alternativeFrequencies;
        bool                        trafficProgramme{};
        std::optional<bool>         trafficAnnouncement;
        std::optional<std::uint8_t> programmeType;

        bool operator==(const OtherNetwork&) const = default;
    };

    struct Station {
        std::uint16_t                         programmeIdentification{};
        std::uint8_t                          programmeType{};
        std::string                           programmeServiceName;
        std::string                           radioText;
        bool                                  trafficProgramme{};
        std::optional<bool>                   trafficAnnouncement;
        std::optional<bool>                   music;
        std::optional<std::int64_t>           clockTime;
        std::int32_t                          localTimeOffset{};
        std::string                           programmeTypeName;
        std::optional<bool>                   stereo;
        std::optional<bool>                   dynamicProgrammeType;
        std::uint8_t                          extendedCountryCode{};
        std::vector<float>                    alternativeFrequencies;
        std::string                           itemArtist;
        std::string                           itemTitle;
        std::map<std::uint16_t, OtherNetwork> otherNetworks;

        bool operator==(const Station&) const = default;
    };

    struct TrafficMessage {
        std::uint16_t               event{};
        std::uint16_t               location{};
        std::uint8_t                extent{};
        bool                        direction{};
        std::optional<std::uint8_t> duration;
    };

    struct ItemTag {
        std::size_t start{};
        std::size_t length{};
    };

    struct FrequencyList {
        std::uint32_t              announced{};
        std::uint32_t              received{};
        std::uint32_t              tuningCode{};
        bool                       methodB = false;
        std::vector<std::uint32_t> codes{};
    };

    struct PartialStation {
        rds::Text<8UZ>                          programmeServiceName;
        FrequencyList                           frequencyList;
        rds::Text<64UZ>                         radioText;
        bool                                    radioTextFlag     = false;
        bool                                    radioTextVersionB = false;
        rds::Text<8UZ>                          programmeTypeName;
        bool                                    programmeTypeNameFlag = false;
        std::optional<std::uint32_t>            radioTextPlusGroup;
        bool                                    itemToggle  = false;
        bool                                    itemRunning = false;
        std::optional<ItemTag>                  artistTag;
        std::optional<ItemTag>                  titleTag;
        std::map<std::uint16_t, rds::Text<8UZ>> otherNetworkNames;
        std::vector<std::uint16_t>              otherNetworkCandidates;
        bool                                    trafficGroupTaken = false;
    };

    static constexpr std::size_t                    kInvalidBlocksToLoseSync   = 8UZ;
    static constexpr std::size_t                    kGroupsBetweenRefreshes    = 45UZ;
    static constexpr std::size_t                    kMaxAlternativeFrequencies = 25UZ;
    static constexpr std::size_t                    kMaxOtherNetworks          = 16UZ;
    static constexpr std::size_t                    kErrorFreeBlocks           = 0UZ;
    static constexpr std::size_t                    kCorrectedBlocks           = 1UZ;
    static constexpr std::size_t                    kLostBlocks                = 2UZ;
    static constexpr std::uint32_t                  kNoAlternativeFrequencies  = 224U;
    static constexpr std::uint32_t                  kLowFrequencyFollows       = 250U;
    static constexpr std::uint32_t                  kFiller                    = 205U;
    static constexpr std::uint32_t                  kNotCarried                = 0U;
    static constexpr std::uint32_t                  kTemporaryDataFault        = 0x1FU;
    static constexpr std::uint16_t                  kRadioTextPlusApplication  = 0x4BD7U;
    static constexpr std::uint32_t                  kTrafficMessageGroup       = 8U << 1U;
    static constexpr std::array<std::uint16_t, 2UZ> kTrafficMessageApplications{0xCD46U, 0xCD47U};
    static constexpr std::uint32_t                  kDynamicProgrammeTypeSegment = 0U;
    static constexpr std::uint32_t                  kStereoSegment               = 3U;
    static constexpr std::uint32_t                  kItemTitle                   = 1U;
    static constexpr std::uint32_t                  kItemArtist                  = 4U;
    static constexpr std::int64_t                   kUnixEpochModifiedJulianDay  = 40'587;
    static constexpr char                           kMissingCharacter            = rds::kMissingCharacter;
    static constexpr std::string_view               kProgrammeIdentificationKey  = "rds_programme_identification";
    static constexpr std::string_view               kProgrammeTypeKey            = "rds_programme_type";
    static constexpr std::string_view               kProgrammeServiceNameKey     = "rds_programme_service_name";
    static constexpr std::string_view               kRadioTextKey                = "rds_radio_text";
    static constexpr std::string_view               kTrafficProgrammeKey         = "rds_traffic_programme";
    static constexpr std::string_view               kTrafficAnnouncementKey      = "rds_traffic_announcement";
    static constexpr std::string_view               kMusicKey                    = "rds_music";
    static constexpr std::string_view               kClockTimeKey                = "rds_clock_time";
    static constexpr std::string_view               kLocalTimeOffsetKey          = "rds_local_time_offset";
    static constexpr std::string_view               kProgrammeTypeNameKey        = "rds_programme_type_name";
    static constexpr std::string_view               kStereoKey                   = "rds_stereo";
    static constexpr std::string_view               kDynamicProgrammeTypeKey     = "rds_dynamic_programme_type";
    static constexpr std::string_view               kSynchronisedKey             = "rds_synchronised";
    static constexpr std::string_view               kExtendedCountryCodeKey      = "rds_extended_country_code";
    static constexpr std::string_view               kAlternativeFrequenciesKey   = "rds_alternative_frequencies";
    static constexpr std::string_view               kItemArtistKey               = "rds_item_artist";
    static constexpr std::string_view               kItemTitleKey                = "rds_item_title";
    static constexpr std::string_view               kOtherNetworksKey            = "rds_other_networks";
    static constexpr std::string_view               kGroupCountsKey              = "rds_group_counts";
    static constexpr std::string_view               kBlockCountsKey              = "rds_block_counts";
    static constexpr std::string_view               kTrafficEventKey             = "rds_tmc_event";
    static constexpr std::string_view               kTrafficLocationKey          = "rds_tmc_location";
    static constexpr std::string_view               kTrafficExtentKey            = "rds_tmc_extent";
    static constexpr std::string_view               kTrafficDirectionKey         = "rds_tmc_direction";
    static constexpr std::string_view               kTrafficDurationKey          = "rds_tmc_duration";

    PortIn<std::uint8_t>  in;
    PortOut<std::uint8_t> out;

    GR_MAKE_REFLECTABLE(RDSDecoder, in, out);

    std::uint64_t                   _receivedBits{};
    std::size_t                     _bitsSinceBlock{};
    std::size_t                     _nextPosition{};
    std::size_t                     _invalidBlocks{};
    bool                            _synchronised = false;
    std::array<std::uint16_t, 4UZ>  _group{};
    std::uint32_t                   _validBlocks{};
    bool                            _thirdBlockIsCPrime = false;
    std::uint16_t                   _lastReceivedProgrammeIdentification{};
    std::array<std::uint32_t, 32UZ> _groupCounts{};
    std::array<std::uint32_t, 3UZ>  _blockCounts{};
    std::size_t                     _groupsSincePublish{};
    PartialStation                  _partial;
    Station                         _station;
    std::optional<Station>          _publishedStation;

    void reset() {
        _receivedBits                        = 0U;
        _bitsSinceBlock                      = 0UZ;
        _nextPosition                        = 0UZ;
        _invalidBlocks                       = 0UZ;
        _synchronised                        = false;
        _validBlocks                         = 0U;
        _thirdBlockIsCPrime                  = false;
        _lastReceivedProgrammeIdentification = 0U;
        _blockCounts.fill(0U);
        _groupsSincePublish = 0UZ;
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
        _blockCounts[kErrorFreeBlocks] += 2U;
        storeBlock(previousBlock, *previousOffset);
        storeBlock(block, *offset);
    }

    void checkExpectedBlock() {
        const std::uint32_t              block  = receivedBlock(0UZ);
        const std::optional<rds::Offset> offset = rds::offsetOf(block);
        if (offset && rds::groupPosition(*offset) == _nextPosition) {
            _invalidBlocks = 0UZ;
            ++_blockCounts[kErrorFreeBlocks];
            storeBlock(block, *offset);
            return;
        }
        constexpr std::array kOffsets{rds::Offset::A, rds::Offset::B, rds::Offset::C, rds::Offset::D};
        const bool           versionB = _nextPosition == 2UZ && isBitSet(_group[1], 11U);
        const rds::Offset    expected = versionB ? rds::Offset::CPrime : kOffsets[_nextPosition];
        if (const std::optional<std::uint32_t> corrected = rds::correctBurst(block, expected)) {
            _invalidBlocks = 0UZ;
            ++_blockCounts[kCorrectedBlocks];
            storeBlock(*corrected, expected);
            return;
        }
        ++_blockCounts[kLostBlocks];
        ++_invalidBlocks;
        if (_invalidBlocks == kInvalidBlocksToLoseSync) {
            _synchronised = false;
            _publishedStation.reset();
            this->publishTag(property_map{{kSynchronisedKey, false}}, 0UZ);
        }
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
            if ((_validBlocks & 0x2U) != 0U) {
                decodeGroup();
            }
            _validBlocks = 0U;
        }
        _nextPosition   = (position + 1UZ) % 4UZ;
        _bitsSinceBlock = 0UZ;
    }

    void decodeGroup() {
        const std::uint32_t blockB   = _group[1];
        const bool          versionB = isBitSet(blockB, 11U);
        const bool          versionA = !versionB;
        const bool          hasA     = (_validBlocks & 0x1U) != 0U;
        const bool          hasC     = (_validBlocks & 0x4U) != 0U;
        const bool          hasD     = (_validBlocks & 0x8U) != 0U;
        if (hasC && versionB != _thirdBlockIsCPrime) {
            return;
        }
        if (!hasA && !(versionB && hasC)) {
            return;
        }
        const std::uint16_t identification   = hasA ? _group[0] : _group[2];
        const bool          repeated         = identification == _lastReceivedProgrammeIdentification;
        _lastReceivedProgrammeIdentification = identification;
        if (identification != _station.programmeIdentification) {
            if (!repeated) {
                return;
            }
            forgetStation();
            _station.programmeIdentification = identification;
        }
        if (_station.programmeIdentification == 0U) {
            return;
        }
        const std::uint32_t groupCode        = blockB >> 11U;
        bool                trafficPublished = false;
        ++_groupCounts[groupCode];
        ++_groupsSincePublish;
        _station.programmeType    = static_cast<std::uint8_t>((blockB >> 5U) & 0x1FU);
        _station.trafficProgramme = isBitSet(blockB, 10U);
        if (_partial.radioTextPlusGroup == groupCode) {
            if (hasC && hasD) {
                decodeRadioTextPlus(blockB);
            }
        } else {
            switch (blockB >> 12U) {
            case 0U:
                decodeSwitching(blockB);
                decodeBasicTuning(blockB, versionA && hasC, hasD);
                break;
            case 1U:
                if (versionA && hasC) {
                    decodeCountry();
                }
                break;
            case 2U: decodeRadioText(blockB, versionB, versionA && hasC, hasD); break;
            case 3U:
                if (versionA && hasD) {
                    decodeApplication(blockB);
                }
                break;
            case 4U:
                if (versionA && hasC && hasD) {
                    decodeClockTime(blockB);
                }
                break;
            case 8U:
                if (versionA && hasC && hasD && !_partial.trafficGroupTaken) {
                    trafficPublished = decodeTrafficMessage(blockB);
                }
                break;
            case 10U:
                if (versionA) {
                    decodeProgrammeTypeName(blockB, hasC, hasD);
                }
                break;
            case 14U:
                if (hasD) {
                    decodeOtherNetwork(blockB, versionB, versionA && hasC);
                }
                break;
            case 15U:
                if (versionB) {
                    decodeSwitching(blockB);
                }
                break;
            default: break;
            }
        }
        updateItemName(_station.itemArtist, _partial.artistTag);
        updateItemName(_station.itemTitle, _partial.titleTag);
        if (!trafficPublished && (_publishedStation != _station || _groupsSincePublish >= kGroupsBetweenRefreshes)) {
            this->publishTag(stationTag(), 0UZ);
            _publishedStation   = _station;
            _groupsSincePublish = 0UZ;
        }
    }

    void forgetStation() {
        _partial = {};
        _station = {};
        _groupCounts.fill(0U);
    }

    void decodeSwitching(std::uint32_t blockB) {
        const std::uint32_t segment    = blockB & 0x3U;
        const bool          decoderBit = isBitSet(blockB, 2U);
        _station.trafficAnnouncement   = isBitSet(blockB, 4U);
        _station.music                 = isBitSet(blockB, 3U);
        if (segment == kDynamicProgrammeTypeSegment) {
            _station.dynamicProgrammeType = decoderBit;
        } else if (segment == kStereoSegment) {
            _station.stereo = decoderBit;
        }
    }

    void decodeBasicTuning(std::uint32_t blockB, bool hasFrequencies, bool hasD) {
        const std::size_t segment = blockB & 0x3U;
        if (hasFrequencies) {
            decodeAlternativeFrequencies(_group[2]);
        }
        if (hasD) {
            _partial.programmeServiceName.store(2UZ * segment, _group[3]);
            if (_partial.programmeServiceName.received.all()) {
                _station.programmeServiceName = _partial.programmeServiceName.text();
                _partial.programmeServiceName.received.reset();
            }
        }
    }

    void decodeAlternativeFrequencies(std::uint32_t block) {
        const std::uint32_t first  = (block >> 8U) & 0xFFU;
        const std::uint32_t second = block & 0xFFU;
        FrequencyList&      list   = _partial.frequencyList;
        const auto          count  = [&list](std::uint32_t code) {
            if (code != kFiller) {
                ++list.received;
                list.codes.push_back(code);
            }
        };
        if (first == kNoAlternativeFrequencies) {
            list = {};
            _station.alternativeFrequencies.clear();
            return;
        }
        if (first > kNoAlternativeFrequencies && first < kLowFrequencyFollows) {
            list = FrequencyList{.announced = first - kNoAlternativeFrequencies, .tuningCode = second};
            count(second);
        } else if (list.announced == 0U) {
            return;
        } else if (first == kLowFrequencyFollows) {
            ++list.received;
        } else {
            list.methodB = list.methodB || first == list.tuningCode || second == list.tuningCode;
            count(first);
            count(second);
        }
        if (list.received < list.announced) {
            return;
        }
        if (!list.methodB) {
            _station.alternativeFrequencies.clear();
        }
        for (const std::uint32_t code : list.codes) {
            addFrequency(_station.alternativeFrequencies, code);
        }
        list = {};
    }

    void decodeCountry() {
        const std::uint32_t blockC = _group[2];
        if (((blockC >> 12U) & 0x7U) == 0U) {
            _station.extendedCountryCode = static_cast<std::uint8_t>(blockC & 0xFFU);
        }
    }

    void decodeRadioText(std::uint32_t blockB, bool versionB, bool hasC, bool hasD) {
        const bool        textFlag = isBitSet(blockB, 4U);
        const std::size_t segment  = blockB & 0xFU;
        if (textFlag != _partial.radioTextFlag || versionB != _partial.radioTextVersionB) {
            _partial.radioText         = {};
            _partial.radioTextFlag     = textFlag;
            _partial.radioTextVersionB = versionB;
            _partial.artistTag.reset();
            _partial.titleTag.reset();
        }
        if (versionB && hasD) {
            _partial.radioText.store(2UZ * segment, _group[3]);
        }
        if (!versionB && hasC) {
            _partial.radioText.store(4UZ * segment, _group[2]);
        }
        if (!versionB && hasD) {
            _partial.radioText.store(4UZ * segment + 2UZ, _group[3]);
        }
        _station.radioText = _partial.radioText.text();
    }

    void decodeApplication(std::uint32_t blockB) {
        const std::uint32_t applicationGroup = blockB & 0x1FU;
        if (_group[3] == kRadioTextPlusApplication) {
            _partial.radioTextPlusGroup = applicationGroup == kNotCarried || applicationGroup == kTemporaryDataFault ? std::nullopt : std::optional(applicationGroup);
        } else if (_partial.radioTextPlusGroup == applicationGroup) {
            _partial.radioTextPlusGroup.reset();
        }
        if (applicationGroup == kTrafficMessageGroup) {
            _partial.trafficGroupTaken = !std::ranges::contains(kTrafficMessageApplications, _group[3]);
        }
    }

    void decodeClockTime(std::uint32_t blockB) {
        const std::uint32_t blockC            = _group[2];
        const std::uint32_t blockD            = _group[3];
        const std::uint32_t modifiedJulianDay = ((blockB & 0x3U) << 15U) | (blockC >> 1U);
        const std::uint32_t hour              = ((blockC & 0x1U) << 4U) | (blockD >> 12U);
        const std::uint32_t minute            = (blockD >> 6U) & 0x3FU;
        const auto          halfHours         = static_cast<std::int32_t>(blockD & 0x1FU);
        if (modifiedJulianDay < kUnixEpochModifiedJulianDay || hour > 23U || minute > 59U) {
            return;
        }
        _station.clockTime       = (static_cast<std::int64_t>(modifiedJulianDay) - kUnixEpochModifiedJulianDay) * 86'400 + static_cast<std::int64_t>(hour) * 3'600 + static_cast<std::int64_t>(minute) * 60;
        _station.localTimeOffset = (isBitSet(blockD, 5U) ? -30 : 30) * halfHours;
    }

    [[nodiscard]] bool decodeTrafficMessage(std::uint32_t blockB) {
        const std::uint32_t blockC      = _group[2];
        const bool          singleGroup = isBitSet(blockB, 3U);
        if (isBitSet(blockB, 4U) || (!singleGroup && !isBitSet(blockC, 15U))) {
            return false;
        }
        const TrafficMessage message{
            .event     = static_cast<std::uint16_t>(blockC & 0x7FFU),
            .location  = _group[3],
            .extent    = static_cast<std::uint8_t>((blockC >> 11U) & 0x7U),
            .direction = isBitSet(blockC, 14U),
            .duration  = singleGroup ? std::optional(static_cast<std::uint8_t>(blockB & 0x7U)) : std::nullopt,
        };
        property_map tag{{kProgrammeIdentificationKey, _station.programmeIdentification}, {kTrafficEventKey, message.event}, {kTrafficLocationKey, message.location}, {kTrafficExtentKey, message.extent}, {kTrafficDirectionKey, message.direction}};
        if (message.duration) {
            tag.insert_or_assign(kTrafficDurationKey, *message.duration);
        }
        this->publishTag(std::move(tag), 0UZ);
        return true;
    }

    void decodeProgrammeTypeName(std::uint32_t blockB, bool hasC, bool hasD) {
        const bool        flag    = isBitSet(blockB, 4U);
        const std::size_t segment = blockB & 0x1U;
        if (flag != _partial.programmeTypeNameFlag) {
            _partial.programmeTypeName     = {};
            _partial.programmeTypeNameFlag = flag;
        }
        if (hasC) {
            _partial.programmeTypeName.store(4UZ * segment, _group[2]);
        }
        if (hasD) {
            _partial.programmeTypeName.store(4UZ * segment + 2UZ, _group[3]);
        }
        _station.programmeTypeName = _partial.programmeTypeName.text();
    }

    void decodeOtherNetwork(std::uint32_t blockB, bool versionB, bool hasC) {
        const std::uint16_t programmeIdentification = _group[3];
        if (programmeIdentification == 0U || programmeIdentification == _station.programmeIdentification) {
            return;
        }
        if (!_station.otherNetworks.contains(programmeIdentification)) {
            std::vector<std::uint16_t>& candidates = _partial.otherNetworkCandidates;
            if (!std::ranges::contains(candidates, programmeIdentification)) {
                if (candidates.size() == kMaxOtherNetworks) {
                    candidates.erase(candidates.begin());
                }
                candidates.push_back(programmeIdentification);
                return;
            }
            if (_station.otherNetworks.size() >= kMaxOtherNetworks) {
                return;
            }
        }
        OtherNetwork& network    = _station.otherNetworks[programmeIdentification];
        network.trafficProgramme = isBitSet(blockB, 4U);
        if (versionB) {
            network.trafficAnnouncement = isBitSet(blockB, 3U);
            return;
        }
        if (!hasC) {
            return;
        }
        const std::uint32_t variant = blockB & 0xFU;
        const std::uint32_t blockC  = _group[2];
        if (variant <= 3U) {
            rds::Text<8UZ>& networkName = _partial.otherNetworkNames[programmeIdentification];
            networkName.store(2UZ * variant, static_cast<std::uint16_t>(blockC));
            network.programmeServiceName = networkName.text();
        } else if (variant == 4U) {
            addAlternativeFrequencies(network.alternativeFrequencies, blockC);
        } else if (variant <= 8U) {
            addFrequency(network.alternativeFrequencies, blockC & 0xFFU);
        } else if (variant == 13U) {
            network.programmeType       = static_cast<std::uint8_t>(blockC >> 11U);
            network.trafficAnnouncement = isBitSet(blockC, 0U);
        }
    }

    void decodeRadioTextPlus(std::uint32_t blockB) {
        const bool toggle    = isBitSet(blockB, 4U);
        _partial.itemRunning = isBitSet(blockB, 3U);
        if (toggle != _partial.itemToggle || !_partial.itemRunning) {
            _partial.itemToggle = toggle;
            _partial.artistTag.reset();
            _partial.titleTag.reset();
            _station.itemArtist.clear();
            _station.itemTitle.clear();
        }
        const std::uint32_t blockC = _group[2];
        const std::uint32_t blockD = _group[3];
        storeItemTag(((blockB & 0x7U) << 3U) | (blockC >> 13U), (blockC >> 7U) & 0x3FU, (blockC >> 1U) & 0x3FU);
        storeItemTag(((blockC & 0x1U) << 5U) | (blockD >> 11U), (blockD >> 5U) & 0x3FU, blockD & 0x1FU);
    }

    void storeItemTag(std::uint32_t contentType, std::uint32_t start, std::uint32_t lengthMarker) {
        const ItemTag tag{.start = start, .length = lengthMarker + 1UZ};
        if (contentType == kItemTitle) {
            _partial.titleTag = tag;
        } else if (contentType == kItemArtist) {
            _partial.artistTag = tag;
        }
    }

    void updateItemName(std::string& itemName, const std::optional<ItemTag>& tag) {
        const rds::Text<64UZ>& text = _partial.radioText;
        if (!tag || !_partial.itemRunning || tag->start + tag->length > text.codes.size()) {
            return;
        }
        if (std::ranges::all_of(std::views::iota(tag->start, tag->start + tag->length), [&text](std::size_t i) { return text.received[i]; })) {
            itemName = text.text(tag->start, tag->length);
        }
    }

    static void addAlternativeFrequencies(std::vector<float>& frequencies, std::uint32_t block) {
        const std::uint32_t first = (block >> 8U) & 0xFFU;
        if (first == kNoAlternativeFrequencies) {
            frequencies.clear();
            return;
        }
        if (first == kLowFrequencyFollows) {
            return;
        }
        addFrequency(frequencies, first);
        addFrequency(frequencies, block & 0xFFU);
    }

    static void addFrequency(std::vector<float>& frequencies, std::uint32_t code) {
        const std::optional<float> frequency = rds::frequencyOf(code);
        if (!frequency || frequencies.size() >= kMaxAlternativeFrequencies) {
            return;
        }
        const auto position = std::ranges::lower_bound(frequencies, *frequency);
        if (position == frequencies.end() || *position != *frequency) {
            frequencies.insert(position, *frequency);
        }
    }

    [[nodiscard]] static constexpr bool isBitSet(std::uint32_t word, std::size_t bit) noexcept { return ((word >> bit) & 1U) != 0U; }

    [[nodiscard]] std::uint32_t receivedBlock(std::size_t blocksAgo) const noexcept { return static_cast<std::uint32_t>(_receivedBits >> (blocksAgo * rds::kBlockBits)) & rds::kBlockMask; }

    [[nodiscard]] property_map stationTag() const {
        const Station& station = _station;
        property_map   tag{
              {kSynchronisedKey, true},
              {kProgrammeIdentificationKey, station.programmeIdentification},
              {kProgrammeTypeKey, station.programmeType},
              {kTrafficProgrammeKey, station.trafficProgramme},
        };
        const auto putText = [](property_map& map, std::string_view key, const std::string& text) {
            if (!text.empty()) {
                map.insert_or_assign(key, text);
            }
        };
        const auto putKnown = [](property_map& map, std::string_view key, const auto& value) {
            if (value) {
                map.insert_or_assign(key, *value);
            }
        };
        putKnown(tag, kTrafficAnnouncementKey, station.trafficAnnouncement);
        putKnown(tag, kMusicKey, station.music);
        putKnown(tag, kStereoKey, station.stereo);
        putKnown(tag, kDynamicProgrammeTypeKey, station.dynamicProgrammeType);
        putText(tag, kProgrammeServiceNameKey, station.programmeServiceName);
        putText(tag, kRadioTextKey, station.radioText);
        putText(tag, kProgrammeTypeNameKey, station.programmeTypeName);
        putText(tag, kItemArtistKey, station.itemArtist);
        putText(tag, kItemTitleKey, station.itemTitle);
        if (station.clockTime) {
            tag.insert_or_assign(kClockTimeKey, *station.clockTime);
            tag.insert_or_assign(kLocalTimeOffsetKey, station.localTimeOffset);
        }
        if (station.extendedCountryCode != 0U) {
            tag.insert_or_assign(kExtendedCountryCodeKey, station.extendedCountryCode);
        }
        if (!station.alternativeFrequencies.empty()) {
            tag.insert_or_assign(kAlternativeFrequenciesKey, Tensor<float>(station.alternativeFrequencies.begin(), station.alternativeFrequencies.end()));
        }
        if (!station.otherNetworks.empty()) {
            property_map networks;
            for (const auto& [programmeIdentification, network] : station.otherNetworks) {
                property_map entry{{kTrafficProgrammeKey, network.trafficProgramme}};
                putKnown(entry, kTrafficAnnouncementKey, network.trafficAnnouncement);
                putKnown(entry, kProgrammeTypeKey, network.programmeType);
                putText(entry, kProgrammeServiceNameKey, network.programmeServiceName);
                if (!network.alternativeFrequencies.empty()) {
                    entry.insert_or_assign(kAlternativeFrequenciesKey, Tensor<float>(network.alternativeFrequencies.begin(), network.alternativeFrequencies.end()));
                }
                const std::string key = std::format("{:04X}", programmeIdentification);
                networks.insert_or_assign(std::string_view(key), std::move(entry));
            }
            tag.insert_or_assign(kOtherNetworksKey, std::move(networks));
        }
        tag.insert_or_assign(kGroupCountsKey, Tensor<std::uint32_t>(_groupCounts.begin(), _groupCounts.end()));
        tag.insert_or_assign(kBlockCountsKey, Tensor<std::uint32_t>(_blockCounts.begin(), _blockCounts.end()));
        return tag;
    }
};

} // namespace gr::digital

#endif // GNURADIO_DIGITAL_RDSDECODER_HPP
