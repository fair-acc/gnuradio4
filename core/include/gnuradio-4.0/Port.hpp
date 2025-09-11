#ifndef GNURADIO_PORT_HPP
#define GNURADIO_PORT_HPP

#include <algorithm>
#include <any>
#include <complex>
#include <set>
#include <span>
#include <variant>

#include <gnuradio-4.0/meta/utils.hpp>

#include "CircularBuffer.hpp"
#include "DataSet.hpp"
#include "Message.hpp"
#include "Tag.hpp"
#include "annotated.hpp"

namespace gr {

using gr::meta::fixed_string;

enum class PortDirection { INPUT, OUTPUT };

enum class ConnectionResult { SUCCESS, FAILED };

enum class PortType {
    STREAM,  /*!< used for single-producer-only ond usually synchronous one-to-one or one-to-many communications */
    MESSAGE, /*!< used for multiple-producer one-to-one, one-to-many, many-to-one, or many-to-many communications */
    ANY      // 'ANY' only for querying and not to be used for port declarations
};

enum class PortSync { SYNCHRONOUS, ASYNCHRONOUS };

namespace port {
enum class BitMask : uint8_t {
    None        = 0U,
    Input       = 1U << 0U,
    Stream      = 1U << 1U,
    Synchronous = 1U << 2U,
    Optional    = 1U << 3U,
    Connected   = 1U << 4U,
};

constexpr BitMask operator|(BitMask a, BitMask b) { return static_cast<BitMask>(static_cast<uint8_t>(a) | static_cast<uint8_t>(b)); }
constexpr BitMask operator&(BitMask a, BitMask b) { return static_cast<BitMask>(static_cast<uint8_t>(a) & static_cast<uint8_t>(b)); }
constexpr bool    any(BitMask mask, BitMask test) { return static_cast<uint8_t>(mask & test) != 0; }

[[nodiscard]] inline constexpr BitMask encodeMask(PortDirection dir, PortType type, bool synchronous, bool optional, bool connected) noexcept {
    assert(type != PortType::ANY && "ANY is not encodable");

    using enum BitMask;
    BitMask mask = None;
    if (dir == PortDirection::INPUT) {
        mask = mask | Input;
    }
    if (type == PortType::STREAM) {
        mask = mask | Stream;
    }
    if (synchronous) {
        mask = mask | Synchronous;
    }
    if (optional) {
        mask = mask | Optional;
    }
    if (connected) {
        mask = mask | Connected;
    }
    return mask;
}

struct BitPattern {
    std::uint8_t mask;
    std::uint8_t value;

    constexpr BitPattern(BitMask m, BitMask v) : mask(static_cast<std::uint8_t>(m)), value(static_cast<std::uint8_t>(v)) {}
    constexpr BitPattern(BitMask m, std::uint8_t v) : mask(static_cast<std::uint8_t>(m)), value(v) {}
    constexpr BitPattern(std::uint8_t m, std::uint8_t v) : mask(m), value(v) {}

    [[nodiscard]] constexpr bool matches(std::uint8_t bits) const { return (bits & mask) == value; }
    [[nodiscard]] constexpr bool matches(BitMask b) const { return matches(static_cast<std::uint8_t>(b)); }
    constexpr BitPattern         operator|(const BitPattern& other) const { return BitPattern(static_cast<uint8_t>(mask | other.mask), static_cast<uint8_t>(value | other.value)); }
    static constexpr BitPattern  Any() { return BitPattern{0U, 0U}; }
};

[[nodiscard]] inline constexpr bool isConnected(BitMask m) { return any(m, BitMask::Connected); }
[[nodiscard]] inline constexpr bool isInput(BitMask m) { return any(m, BitMask::Input); }
[[nodiscard]] inline constexpr bool isStream(BitMask m) { return any(m, BitMask::Stream); }
[[nodiscard]] inline constexpr bool isSynchronous(BitMask m) { return any(m, BitMask::Synchronous); }

[[nodiscard]] inline constexpr PortDirection decodeDirection(BitMask m) { return isInput(m) ? PortDirection::INPUT : PortDirection::OUTPUT; }
[[nodiscard]] inline constexpr PortType      decodePortType(BitMask m) { return isStream(m) ? PortType::STREAM : PortType::MESSAGE; }

// comparison operators
[[nodiscard]] inline constexpr bool operator==(BitMask mask, PortDirection dir) { return decodeDirection(mask) == dir; }
[[nodiscard]] inline constexpr bool operator!=(BitMask mask, PortDirection dir) { return !(mask == dir); }
[[nodiscard]] inline constexpr bool operator==(BitMask mask, PortType type) { return decodePortType(mask) == type; }
[[nodiscard]] inline constexpr bool operator!=(BitMask mask, PortType type) { return !(mask == type); }
[[nodiscard]] inline constexpr bool operator==(PortDirection dir, BitMask mask) { return mask == dir; }
[[nodiscard]] inline constexpr bool operator!=(PortDirection dir, BitMask mask) { return !(mask == dir); }
[[nodiscard]] inline constexpr bool operator==(PortType type, BitMask mask) { return mask == type; }
[[nodiscard]] inline constexpr bool operator!=(PortType type, BitMask mask) { return !(mask == type); }

[[nodiscard]] inline constexpr BitPattern matchBits(PortDirection d) {
    using enum BitMask;
    switch (d) {
    case PortDirection::INPUT: return {Input, Input};
    case PortDirection::OUTPUT: return {Input, 0UZ};
    default: return BitPattern::Any();
    }
}

[[nodiscard]] inline constexpr BitPattern matchBits(PortType t) {
    using enum BitMask;
    switch (t) {
    case PortType::STREAM: return {Stream, Stream};
    case PortType::MESSAGE: return {Stream, 0UZ};
    case PortType::ANY: return BitPattern::Any();
    }
    return BitPattern::Any();
}

[[nodiscard]] inline constexpr BitPattern matchBits(PortSync s) {
    using enum BitMask;
    switch (s) {
    case PortSync::SYNCHRONOUS: return {Synchronous, Synchronous};
    case PortSync::ASYNCHRONOUS: return {Synchronous, 0UZ};
    }
    return BitPattern::Any();
}

template<auto... Enums>
[[nodiscard]] consteval BitPattern pattern() {
    BitPattern combined = BitPattern::Any();
    ((combined = combined | matchBits(Enums)), ...);
    return combined;
}
} // namespace port

/**
 * @brief optional port annotation argument to described whether the port can be handled within the same scheduling domain.
 *
 * @tparam PortDomainName the unique name of the domain, name shouldn't clash with other existing definitions (e.g. 'CPU' and 'GPU')
 */
template<fixed_string PortDomainName>
struct PortDomain {
    static constexpr fixed_string Name = PortDomainName;
};

template<typename T>
concept PortDomainLike = requires { T::Name; } && std::is_base_of_v<PortDomain<T::Name>, T>;

template<typename T>
using is_port_domain = std::bool_constant<PortDomainLike<T>>;

struct CPU : PortDomain<"CPU"> {};

struct GPU : PortDomain<"GPU"> {};

static_assert(is_port_domain<CPU>::value);
static_assert(is_port_domain<GPU>::value);
static_assert(!is_port_domain<int>::value);

struct PortInfo { // maybe/should be replaced by gr::port::BitMask
    PortType         portType                  = PortType::ANY;
    PortDirection    portDirection             = PortDirection::INPUT;
    std::string_view portDomain                = "unknown";
    ConnectionResult portConnectionResult      = ConnectionResult::FAILED;
    std::string      valueTypeName             = "uninitialised type";
    bool             isValueTypeArithmeticLike = false;
    std::size_t      valueTypeSize             = 0UZ;
    std::size_t      bufferSize                = 0UZ;
    std::size_t      availableBufferSize       = 0UZ;
};

struct PortMetaInfo {
    using description = Doc<R"*(@brief Port meta-information for increased type and physical-unit safety. Uses ISO 80000-1:2022 conventions.

**Some example usages:**
  * prevents to accidentally connect ports with incompatible sampling rates, quantity- and unit-types.
  * used to condition graphs/charts (notably the min/max range),
  * detect saturation/LNA non-linearities,
  * detect computation errors
  * ...

Follows the ISO 80000-1:2022 Quantities and Units conventions:
  * https://www.iso.org/standard/76921.html
  * https://en.wikipedia.org/wiki/ISO/IEC_80000
  * https://blog.ansi.org/iso-80000-1-2022-quantities-and-units/
)*">; // long-term goal: enable compile-time checks based on https://github.com/mpusz/mp-units (N.B. will become part of C++26)

    Annotated<std::string, "data type name", Visible, Doc<"portable port data type name">>                           data_type = "<unknown>";
    Annotated<std::string, "port name", Visible, Doc<"port name">>                                                   name;
    Annotated<float, "sample rate", Visible, Doc<"sampling rate in samples per second (Hz)">>                        sample_rate = 1.f;
    Annotated<std::string, "signal name", Doc<"name of the signal">>                                                 signal_name = "<unnamed>";
    Annotated<std::string, "signal quantity", Doc<"physical quantity (e.g., 'voltage'). Follows ISO 80000-1:2022.">> signal_quantity{};
    Annotated<std::string, "signal unit", Doc<"unit of measurement (e.g., '[V]', '[m]'). Follows ISO 80000-1:2022">> signal_unit{};
    Annotated<float, "signal min", Doc<"minimum expected signal value">>                                             signal_min = std::numeric_limits<float>::lowest();
    Annotated<float, "signal max", Doc<"maximum expected signal value">>                                             signal_max = std::numeric_limits<float>::max();

    GR_MAKE_REFLECTABLE(PortMetaInfo, data_type, name, sample_rate, signal_name, signal_quantity, signal_unit, signal_min, signal_max);

    // controls automatic (if set) or manual update of above parameters
    std::set<std::string, std::less<>> auto_update{gr::tag::kDefaultTags.begin(), gr::tag::kDefaultTags.end()};

    constexpr PortMetaInfo() noexcept = default;
    explicit PortMetaInfo(std::string_view dataTypeName) noexcept : data_type(dataTypeName) {};
    explicit PortMetaInfo(std::initializer_list<std::pair<const std::string, pmtv::pmt>> initMetaInfo) noexcept(false) //
        : PortMetaInfo(property_map{initMetaInfo.begin(), initMetaInfo.end()}) {}
    explicit PortMetaInfo(const property_map& metaInfo) noexcept(false) {
        if (auto res = update(metaInfo); !res) {
            throw gr::exception(res.error().message, res.error().sourceLocation);
        }
    }

    void reset() { auto_update = {gr::tag::kDefaultTags.begin(), gr::tag::kDefaultTags.end()}; }

    [[nodiscard]] std::expected<void, Error> update(const property_map& metaInfo, const std::source_location location = std::source_location::current()) noexcept {
        for (const auto& [key, value] : metaInfo) {
            if (!auto_update.contains(key)) {
                continue;
            }
            std::expected<void, Error> maybeError = {};
            refl::for_each_data_member_index<PortMetaInfo>([&](auto kIdx) {
                using MemberType = refl::data_member_type<PortMetaInfo, kIdx>;
                using Type       = unwrap_if_wrapped_t<std::remove_cvref_t<MemberType>>;

                const auto fieldName = refl::data_member_name<PortMetaInfo, kIdx>.view();
                if (fieldName == key) {
                    if (std::holds_alternative<Type>(value)) {
                        auto& member = refl::data_member<kIdx>(*this);
                        std::ignore  = member.validate_and_set(std::get<Type>(value));
                    } else {
                        maybeError = std::unexpected(Error{std::format("PortMetaInfo invalid-argument: incorrect type for key {} (expected:{}, got:{}, value:{})", key, gr::meta::type_name<Type>(), gr::meta::type_name<std::decay_t<decltype(value)>>(), value), location});
                    }
                }
            });
            if (!maybeError) {
                return maybeError;
            }
        }

        return {};
    }

    [[nodiscard]] property_map get() const noexcept {
        property_map metaInfo;
        refl::for_each_data_member_index<PortMetaInfo>([&](auto kIdx) { //
            metaInfo.insert_or_assign(std::string(refl::data_member_name<PortMetaInfo, kIdx>.view()), pmtv::pmt(refl::data_member<kIdx>(*this)));
        });

        return metaInfo;
    }
};

template<class T>
concept PortLike = requires(T t, const std::size_t n_items, const std::any& newDefault) { // dynamic definitions
    typename T::value_type;
    { t.defaultValue() } -> std::same_as<std::any>;
    { t.setDefaultValue(newDefault) } -> std::same_as<bool>;
    { t.name } -> std::convertible_to<std::string_view>;
    { t.priority } -> std::convertible_to<std::int32_t>;
    { t.min_samples } -> std::convertible_to<std::size_t>;
    { t.max_samples } -> std::convertible_to<std::size_t>;
    { t.metaInfo } -> std::convertible_to<gr::PortMetaInfo>;
    { t.type() } -> std::same_as<PortType>;
    { t.direction() } -> std::same_as<PortDirection>;
    { t.domain() } -> std::same_as<std::string_view>;
    { t.resizeBuffer(n_items) } -> std::same_as<ConnectionResult>;
    { t.isConnected() } -> std::same_as<bool>;
    { t.disconnect() } -> std::same_as<ConnectionResult>;
    { t.isSynchronous() } -> std::same_as<bool>;
    { t.isOptional() } -> std::same_as<bool>;
};

/**
 * @brief internal port buffer handler
 *
 * N.B. void* needed for type-erasure/Python compatibility/wrapping
 */
struct InternalPortBuffers {
    void* streamHandler;
    void* tagHandler;
};

/**
 * @brief optional port annotation argument to describe the min/max number of samples required from this port before invoking the blocks work function.
 *
 * @tparam minSamples (>0) specifies the minimum number of samples the port/block requires for processing in one scheduler iteration
 * @tparam maxSamples specifies the maximum number of samples the port/block can process in one scheduler iteration
 * @tparam isConst specifies if the range is constant or can be modified during run-time.
 */
template<std::size_t minSamples = std::dynamic_extent, std::size_t maxSamples = std::dynamic_extent, bool isConst = false>
struct RequiredSamples {
    static_assert(minSamples > 0, "Port<T, ..., RequiredSamples::MIN_SAMPLES, ...>, ..> must be >= 0");
    static constexpr std::size_t kMinSamples = minSamples == std::dynamic_extent ? 1UZ : minSamples;
    static constexpr std::size_t kMaxSamples = maxSamples == std::dynamic_extent ? std::numeric_limits<std::size_t>::max() : maxSamples;
    static constexpr bool        kIsConst    = isConst;
};

template<typename T>
concept IsRequiredSamples = requires {
    T::kMinSamples;
    T::kMaxSamples;
    T::kIsConst;
} && std::is_base_of_v<RequiredSamples<T::kMinSamples, T::kMaxSamples, T::kIsConst>, T>;

template<typename T>
using is_required_samples = std::bool_constant<IsRequiredSamples<T>>;

static_assert(is_required_samples<RequiredSamples<1, 1024>>::value);
static_assert(!is_required_samples<int>::value);

/**
 * @brief optional port annotation argument informing the graph/scheduler that a given port does not require to be connected
 */
struct Optional {};

/**
 * @brief optional port annotation argument to define the buffer implementation to be used for streaming data
 *
 * @tparam BufferType user-extendable buffer implementation for the streaming data
 */
template<gr::BufferLike BufferType>
struct StreamBufferType {
    using type = BufferType;
};

/**
 * @brief optional port annotation argument to define the buffer implementation to be used for tag data
 *
 * @tparam BufferType user-extendable buffer implementation for the tag data
 */
template<gr::BufferLike BufferType>
struct TagBufferType {
    using type = BufferType;
};

template<typename T>
concept IsStreamBufferAttribute = requires { typename T::type; } && gr::BufferLike<typename T::type> && std::is_base_of_v<StreamBufferType<typename T::type>, T>;

template<typename T>
concept IsTagBufferAttribute = requires { typename T::type; } && gr::BufferLike<typename T::type> && std::is_base_of_v<TagBufferType<typename T::type>, T>;

template<typename T>
using is_stream_buffer_attribute = std::bool_constant<IsStreamBufferAttribute<T>>;

template<typename T>
using is_tag_buffer_attribute = std::bool_constant<IsTagBufferAttribute<T>>;

template<typename T>
struct DefaultStreamBuffer : StreamBufferType<gr::CircularBuffer<T>> {};

struct DefaultMessageBuffer : StreamBufferType<gr::CircularBuffer<Message, std::dynamic_extent, gr::ProducerType::Multi>> {};

struct DefaultTagBuffer : TagBufferType<gr::CircularBuffer<Tag>> {};

static_assert(is_stream_buffer_attribute<DefaultStreamBuffer<int>>::value);
static_assert(is_stream_buffer_attribute<DefaultMessageBuffer>::value);
static_assert(!is_stream_buffer_attribute<DefaultTagBuffer>::value);
static_assert(!is_tag_buffer_attribute<DefaultStreamBuffer<int>>::value);
static_assert(is_tag_buffer_attribute<DefaultTagBuffer>::value);

} // namespace gr

namespace gr {

/**
 * @brief Annotation for making a port asynchronous in a signal flow-graph block.
 *
 * In a standard block, the processing function is invoked based on the least common number of samples
 * available across all input and output ports. When a port is annotated with `Async`, it is excluded from this
 * least common number calculation.
 *
 * Applying `Async` as an optional template argument of the Port class essentially marks the port as "optional" for the
 * synchronization mechanism. The block's processing function will be invoked regardless of the number of samples
 * available at this specific port, relying solely on the state of other ports that are not marked as asynchronous.
 *
 * Use this annotation to create ports that do not constrain the block's ability to process data, making it
 * asynchronous relative to the other ports in the block.
 */
struct Async {};

/**
 * @brief Provides an interface for accessing input samples and their associated tags within the `processBulk` function.
 *
 * The `InputSpanLike` concept is used as the type for input parameters in the `processBulk` function:`work::Status processBulk(InputSpanLike auto& in, OutputSpan auto& out);`
 * **Features:**
 * - Access to Samples: Allows direct access to samples via array indexing (via conversion to `std::span`): `auto value = in[0];  // Access the first sample`
 * - Consuming Samples: Provides a `consume(nSamplesToConsume)` method to indicate how many samples to consume.
 *   - Default Behavior:
 *     - For `Synch` ports, all samples are published by default.
 *     - For `Async` ports, no samples are published by default.
 * - Access to Tags:
 *   - Using `tags()`: Returns a `range::view` of all input tags. Indices are relative to the first sample in the span and can be negative for unconsumed tags.
 *   - Using `tags(untilLocalIndex)`: Returns a `range::view` of input tags up to `untilLocalIndex` (exclusively). Indices are relative to the first sample in the span and can be negative for unconsumed tags.
 *   - Using `rawTags`: Provides direct access to the underlying `ReaderSpan<Tag>` for advanced manipulation.
 * - Consuming Tags: By default, tags associated with samples up to and including the first sample are consumed. One can manually consume tags up to a specific sample index using `consumeTags(streamSampleIndex)`.
 */
template<typename T>
concept InputSpanLike = std::ranges::contiguous_range<T> && ConstSpanLike<T> && requires(T& span, std::size_t n) {
    { span.consume(0) };
    { span.isConnected } -> std::convertible_to<bool>;
    { span.isSync } -> std::convertible_to<bool>;
    { span.rawTags };
    requires ReaderSpanLike<std::remove_cvref_t<decltype(span.rawTags)>> && std::same_as<gr::Tag, std::ranges::range_value_t<decltype(span.rawTags)>>;
    { span.tags() } -> std::ranges::range;
    { span.tags(n) } -> std::ranges::range;
    { span.consumeTags(n) };
};

/**
 * @brief Provides an interface for writing output samples and publishing tags within the `processBulk` function.
 *
 * The `OutputSpan` concept is used as the type for output parameters in the `processBulk` function: `work::Status processBulk(InputSpanLike auto& in, OutputSpan auto& out);`
 * **Features:**
 * - Writing Output Samples: Allows writing to output samples via array indexing (via conversion to `std::span`): `out[0] = 4.2;  // Write a value to the first output sample`
 * - Publishing Samples: Provides a `publish(nSamplesToPublish)` method to indicate how many samples are to be published.
 *   - Default Behavior:
 *     - For `Synch` ports, all samples are published by default.
 *     - For `Async` ports, no samples are published by default.
 * - Publishing Tags: Use `publishTag(tagData, tagOffset)` to publish tags. `tagOffset` is relative to the first sample.
 */
template<typename T>
concept OutputSpanLike = std::ranges::contiguous_range<T> && std::ranges::output_range<T, std::remove_cvref_t<typename T::value_type>> && SpanLike<T> && requires(T& span, property_map& tagData, std::size_t tagOffset) {
    span.publish(0UZ);
    { span.isConnected } -> std::convertible_to<bool>;
    { span.isSync } -> std::convertible_to<bool>;
    requires WriterSpanLike<std::remove_cvref_t<decltype(span.tags)>>;
    { *span.tags.begin() } -> std::same_as<gr::Tag&>;
    { span.publishTag(tagData, tagOffset) } -> std::same_as<void>;
};

namespace detail {
enum PortOrCollectionKind { SinglePort, DynamicPortCollection, StaticPortCollection, TupleOfPorts };

// int KindExtraData -- tuple index for ports in tuples, array size for arrays of ports, unused otherwise
template<typename T, meta::fixed_string portName, PortType portType, PortDirection portDirection, PortOrCollectionKind Kind, std::size_t KindExtraData, std::size_t MemberIdx, typename... Attributes>
struct PortDescriptor {
    static_assert(not portName.empty());
    static_assert(MemberIdx != size_t(-1));

    // descriptor for a std::vector<Port> (or similar dynamically sized container)
    static constexpr bool kIsDynamicCollection = Kind == DynamicPortCollection;
    static constexpr bool kIsStaticCollection  = Kind == StaticPortCollection;

    // tuple-like
    static constexpr bool kPartOfTuple = Kind == TupleOfPorts;

    static constexpr PortDirection kDirection                 = portDirection;
    static constexpr PortType      kPortType                  = portType;
    static constexpr bool          kIsInput                   = portDirection == PortDirection::INPUT;
    static constexpr bool          kIsOutput                  = portDirection == PortDirection::OUTPUT;
    static constexpr bool          kIsArithmeticLikeValueType = gr::arithmetic_or_complex_like<T> || gr::UncertainValueLike<T>;

    using Required = meta::typelist<Attributes...>::template find_or_default<is_required_samples, RequiredSamples<std::dynamic_extent, std::dynamic_extent>>;

    // directly used as the return type of processOne (if there are multiple PortDescriptors a tuple of all value_types)
    using value_type =                                                            //
        std::conditional_t<kIsDynamicCollection, std::vector<T>,                  //
            std::conditional_t<kIsStaticCollection, std::array<T, KindExtraData>, //
                T>>;

    template<typename TBlock>
    requires std::same_as<std::remove_cvref_t<TBlock>, typename std::remove_cvref_t<TBlock>::derived_t>
    static constexpr decltype(auto) getPortObject(TBlock&& obj) {
        if constexpr (kPartOfTuple) {
            return std::get<KindExtraData>(refl::data_member<MemberIdx>(obj));
        } else {
            return refl::data_member<MemberIdx>(obj);
        }
    }

    using NameT = meta::constexpr_string<portName>;

    static constexpr NameT Name{};

    PortDescriptor()  = delete;
    ~PortDescriptor() = delete;
};

template<typename T>
concept PortDescription = requires {
    typename T::value_type;
    typename T::NameT;
    typename T::Required;
    { auto(T::kIsDynamicCollection) } -> std::same_as<bool>;
    { auto(T::kIsStaticCollection) } -> std::same_as<bool>;
    { auto(T::kPartOfTuple) } -> std::same_as<bool>;
    { auto(T::kIsInput) } -> std::same_as<bool>;
    { auto(T::kIsOutput) } -> std::same_as<bool>;
};
} // namespace detail

/**
 * @brief 'ports' are interfaces that allows data to flow between blocks in a graph, similar to RF connectors.
 * Each block can have zero or more input/output ports. When connecting ports, either a single-step or a two-step
 * connection method can be used. Ports belong to a computing domain, such as CPU, GPU, or FPGA, and transitions
 * between domains require explicit data conversion.
 * Each port consists of a synchronous performance-optimised streaming and asynchronous tag communication component:
 *                                                                                      ┌───────────────────────
 *         ───────────────────┐                                       ┌─────────────────┤  <node/block definition>
 *             output-port    │                                       │    input-port   │  ...
 *          stream-buffer<T>  │>───────┬─────────────────┬───────────>│                 │
 *          tag-buffer<Tag>   │      tag#0             tag#1          │                 │
 *                            │                                       │                 │
 *         ───────────────────┘                                       └─────────────────┤
 *
 * Tags contain the index ID of the sending/receiving stream sample <T> they are attached to. Block implementations
 * may choose to chunk the data based on the MIN_SAMPLES/MAX_SAMPLES criteria only, or in addition break-up the stream
 * so that there is only one tag per scheduler iteration. Multiple tags on the same sample shall be merged to one.
 *
 * @tparam T the data type of the port. It can be any copyable preferably cache-aligned (i.e. 64 byte-sized) type.
 * @tparam portType STREAM  or MESSAGE
 * @tparam portDirection either input or output
 * @tparam Attributes optional: default to 'DefaultStreamBuffer' and DefaultTagBuffer' based on 'gr::circular_buffer', and CPU domain
 */
template<typename T, PortType portType, PortDirection portDirection, typename... Attributes>
struct Port {
    template<meta::fixed_string newName, detail::PortOrCollectionKind Kind, std::size_t KindExtraData, size_t MemberIdx>
    using make_port_descriptor = detail::PortDescriptor<T, newName, portType, portDirection, Kind, KindExtraData, MemberIdx, Attributes...>;

    static_assert(portType != PortType::ANY, "ANY reserved for queries and not port type declarations");
    static_assert(portType == PortType::STREAM || std::is_same_v<T, gr::Message>, "If a port type is MESSAGE, the value type needs to be gr::Message");

    using value_type        = T;
    using AttributeTypeList = gr::meta::typelist<Attributes...>;
    using Domain            = AttributeTypeList::template find_or_default<is_port_domain, CPU>;
    using Required          = AttributeTypeList::template find_or_default<is_required_samples, RequiredSamples<std::dynamic_extent, std::dynamic_extent>>;
    using BufferType        = AttributeTypeList::template find_or_default<is_stream_buffer_attribute, DefaultStreamBuffer<T>>::type;
    using TagBufferType     = AttributeTypeList::template find_or_default<is_tag_buffer_attribute, DefaultTagBuffer>::type;

    static constexpr bool        kIsArithmeticLikeValueType = gr::arithmetic_or_complex_like<T> && sizeof(T) <= 16UZ;
    static constexpr std::size_t kDefaultBufferSize         = 4096UZ; // TODO: limit initial max buffer size based on kIsArithmeticLikeValueType

    // constexpr members:
    static constexpr PortDirection kDirection = portDirection;
    static constexpr PortType      kPortType  = portType;
    static constexpr bool          kIsInput   = portDirection == PortDirection::INPUT;
    static constexpr bool          kIsOutput  = portDirection == PortDirection::OUTPUT;

    // dependent types
    using ReaderType        = decltype(std::declval<BufferType>().new_reader());
    using WriterType        = decltype(std::declval<BufferType>().new_writer());
    using IoType            = std::conditional_t<kIsInput, ReaderType, WriterType>;
    using TagReaderType     = decltype(std::declval<TagBufferType>().new_reader());
    using TagWriterType     = decltype(std::declval<TagBufferType>().new_writer());
    using TagIoType         = std::conditional_t<kIsInput, TagReaderType, TagWriterType>;
    using TagReaderSpanType = decltype(std::declval<TagReaderType>().get());
    using TagWriterSpanType = decltype(std::declval<TagWriterType>().reserve(0UZ));

    // public properties
    // kIsSynch:
    //   true  -> port participates in synchronous scheduling with other sync ports
    //   false -> port is asynchronous (does not gate scheduling)
    // Rule: an input marked Optional is implicitly also async (i.e. otherwise it would block the sync block data processing);
    //       outputs stay synchronous unless explicitly annotated with Async.
    constexpr static bool kIsSynch    = !(std::disjunction_v<std::is_same<Async, Attributes>...> || (kIsInput && std::disjunction_v<std::is_same<Optional, Attributes>...>));
    constexpr static bool kIsOptional = std::disjunction_v<std::is_same<Optional, Attributes>...>; // port may be left unconnected

    std::string_view name;

    std::int16_t priority      = 0; // → dependents of a higher-prio port should be scheduled first (Q: make this by order of ports?)
    T            default_value = T{};

    //
    std::conditional_t<Required::kIsConst, const std::size_t, std::size_t> min_samples = Required::kMinSamples;
    std::conditional_t<Required::kIsConst, const std::size_t, std::size_t> max_samples = Required::kMaxSamples;

    // Port meta-information for increased type and physical-unit safety. Uses ISO 80000-1:2022 conventions.
    PortMetaInfo metaInfo{gr::meta::type_name<T>()};

    GR_MAKE_REFLECTABLE(Port, kDirection, kPortType, kIsInput, kIsOutput, kIsSynch, kIsOptional, name, priority, min_samples, max_samples, metaInfo);

    template<SpanReleasePolicy spanReleasePolicy>
    using ReaderSpanType = decltype(std::declval<ReaderType>().template get<spanReleasePolicy>());

    template<SpanReleasePolicy spanReleasePolicy, bool consumeOnlyFirstTag = false>
    struct InputSpan : public ReaderSpanType<spanReleasePolicy> {
        TagReaderSpanType rawTags;
        std::size_t       streamIndex;
        bool              isConnected = true; // true if Port is connected
        bool              isSync      = true; // true if  Port is Sync

        InputSpan(std::size_t nSamples_, ReaderType& reader, TagReaderType& tagReader, bool connected, bool sync) //
            : ReaderSpanType<spanReleasePolicy>(reader.template get<spanReleasePolicy>(nSamples_)),               //
              rawTags(getTagsInRange(nSamples_, tagReader, reader.position())),                                   //
              streamIndex{reader.position()}, isConnected(connected), isSync(sync) {}

        InputSpan(const InputSpan&)            = default;
        InputSpan& operator=(const InputSpan&) = default;
        // InputSpan(InputSpan&&) noexcept            = delete;
        // InputSpan& operator=(InputSpan&&) noexcept = delete;

        ~InputSpan() override {
            if (ReaderSpanType<spanReleasePolicy>::instanceCount() == 1UZ) { // has to be one, because the parent destructor which decrements it to zero is only called afterward
                if (rawTags.isConsumeRequested()) {                          // the user has already manually consumed tags
                    return;
                }

                if (this->empty() //
                    || (ReaderSpanType<spanReleasePolicy>::isConsumeRequested() && ReaderSpanType<spanReleasePolicy>::nRequestedSamplesToConsume() == 0)) {
                    return;
                }

                if constexpr (consumeOnlyFirstTag) {
                    consumeTags(1UZ);
                } else {
                    if (ReaderSpanType<spanReleasePolicy>::isConsumeRequested()) {
                        consumeTags(ReaderSpanType<spanReleasePolicy>::nRequestedSamplesToConsume());
                    } else {
                        if (ReaderSpanType<spanReleasePolicy>::spanReleasePolicy() == SpanReleasePolicy::ProcessAll) {
                            consumeTags(ReaderSpanType<spanReleasePolicy>::size());
                        }
                    }
                }
            }
        }

        [[nodiscard]] auto tags() {
            using PairRefType = std::pair<std::ptrdiff_t, std::reference_wrapper<const property_map>>;
            return std::views::transform(rawTags, [this](const Tag& tag) -> PairRefType { return PairRefType{relIndex(tag.index, streamIndex), std::cref(tag.map)}; });
        }

        [[nodiscard]] auto tags(std::size_t untilLocalIndex) {
            using PairRefType            = std::pair<std::ptrdiff_t, std::reference_wrapper<const property_map>>;
            const std::size_t untilIndex = streamIndex + untilLocalIndex;
            return rawTags | std::views::take_while([untilIndex](const Tag& t) { return t.index < untilIndex; }) | std::views::transform([this](const Tag& tag) -> PairRefType { //
                return PairRefType{relIndex(tag.index, streamIndex), std::cref(tag.map)};
            });
        }

        void consumeTags(std::size_t untilLocalIndex) {
            std::size_t tagsToConsume = static_cast<std::size_t>(std::ranges::count_if(rawTags | std::views::take_while([untilLocalIndex, this](auto& t) { return t.index < streamIndex + untilLocalIndex; }), [](auto /*v*/) { return true; }));
            std::ignore               = rawTags.tryConsume(tagsToConsume);
        }

    private:
        [[nodiscard]] static constexpr std::ptrdiff_t relIndex(std::size_t abs, std::size_t base) noexcept { return abs >= base ? static_cast<std::ptrdiff_t>(abs - base) : -static_cast<std::ptrdiff_t>(base - abs); }

        auto getTagsInRange(std::size_t nSamples, TagReaderType& reader, std::size_t currentStreamOffset) {
            const auto tags = reader.get(reader.available());
            const auto it   = std::ranges::find_if_not(tags, [nSamples, currentStreamOffset](const auto& tag) { return tag.index < currentStreamOffset + nSamples; });
            const auto n    = static_cast<std::size_t>(std::distance(tags.begin(), it));
            return reader.get(n);
        }
    }; // end of InputSpan
    static_assert(ReaderSpanLike<InputSpan<gr::SpanReleasePolicy::ProcessAll, false>>);
    static_assert(InputSpanLike<InputSpan<gr::SpanReleasePolicy::ProcessAll, false>>);

    template<SpanReleasePolicy spanReleasePolicy>
    using WriterSpanType = decltype(std::declval<WriterType>().template reserve<spanReleasePolicy>(1UZ));

    template<SpanReleasePolicy spanReleasePolicy, WriterSpanReservePolicy spanReservePolicy>
    struct OutputSpan : public WriterSpanType<spanReleasePolicy> {
        TagWriterSpanType tags;
        std::size_t       streamIndex;
        std::size_t       tagsPublished{0UZ};
        bool              isConnected = true; // true if Port is connected
        bool              isSync      = true; // true if  Port is Sync

        constexpr OutputSpan(std::size_t nSamples, WriterType& streamWriter, TagWriterType& tagsWriter, std::size_t streamOffset, bool connected, bool sync) noexcept //
        requires(spanReservePolicy == WriterSpanReservePolicy::Reserve)
            : WriterSpanType<spanReleasePolicy>(streamWriter.template reserve<spanReleasePolicy>(nSamples)), //
              tags(tagsWriter.template reserve<SpanReleasePolicy::ProcessNone>(tagsWriter.available())),     //
              streamIndex{streamOffset}, isConnected(connected), isSync(sync) {}

        constexpr OutputSpan(std::size_t nSamples_, WriterType& streamWriter, TagWriterType& tagsWriter, std::size_t streamOffset, bool connected, bool sync) noexcept //
        requires(spanReservePolicy == WriterSpanReservePolicy::TryReserve)
            : WriterSpanType<spanReleasePolicy>(streamWriter.template tryReserve<spanReleasePolicy>(nSamples_)), //
              tags(tagsWriter.template tryReserve<SpanReleasePolicy::ProcessNone>(tagsWriter.available())),      //
              streamIndex{streamOffset}, isConnected(connected), isSync(sync) {}

        OutputSpan(const OutputSpan&)            = default;
        OutputSpan& operator=(const OutputSpan&) = default;
        // OutputSpan(OutputSpan&&) noexcept            = delete;
        // OutputSpan& operator=(OutputSpan&&) noexcept = delete;

        ~OutputSpan() {
            if (WriterSpanType<spanReleasePolicy>::instanceCount() == 1UZ) { // has to be one, because the parent destructor which decrements it to zero is only called afterward
                tags.publish(tagsPublished);
            }
        }

        template<PropertyMapType TPropertyMap>
        inline constexpr void publishTag(TPropertyMap&& tagData, std::size_t tagOffset = 0UZ) noexcept {
            // Do not publish tags if port is not connected, as it can lead to a tag buffer overflow.
            if (!isConnected) {
                return;
            }

            if (tagsPublished > tags.size()) {
                // TODO(error handling): Decide how to surface failures.
                // Option A: throw an exception, but this function is marked noexcept—either remove noexcept or avoid throwing.
                // Option B: return an error (or set a port-status flag) that the Scheduler can observe and handle accordingly.
                // std::println(stderr, "Tags buffer is full (published:{}, size:{}), can not process tag publishing", tagsPublished, tags.size());
                return;
            }
            const auto index = streamIndex + tagOffset;

#ifndef NDEBUG
            if (tagsPublished > 0) {
                auto& lastTag = tags[tagsPublished - 1];
                if (lastTag.index > index) { // check the order of published Tags.index
                    std::println(stderr, "Tag indices are not in the correct order, tagsPublished:{}, lastTag.index:{}, index:{}", tagsPublished, lastTag.index, index);
                    std::abort();
                }
            }
#endif
            tags[tagsPublished++] = {index, std::forward<TPropertyMap>(tagData)};
        }
    }; // end of PortOutputSpan
    static_assert(WriterSpanLike<OutputSpan<gr::SpanReleasePolicy::ProcessAll, WriterSpanReservePolicy::Reserve>>);
    static_assert(OutputSpanLike<OutputSpan<gr::SpanReleasePolicy::ProcessAll, WriterSpanReservePolicy::Reserve>>);

private:
    IoType    _ioHandler    = newIoHandler();
    TagIoType _tagIoHandler = newTagIoHandler();
    Tag       _cachedTag{}; // todo: for now this is only used in the output ports

    [[nodiscard]] constexpr auto newIoHandler(std::size_t bufferSize = kDefaultBufferSize) const noexcept {
        if constexpr (kIsInput) {
            return BufferType(bufferSize).new_reader();
        } else {
            return BufferType(bufferSize).new_writer();
        }
    }

    [[nodiscard]] constexpr auto newTagIoHandler(std::size_t bufferSize = kDefaultBufferSize) const noexcept {
        if constexpr (kIsInput) {
            return TagBufferType(bufferSize).new_reader();
        } else {
            return TagBufferType(bufferSize).new_writer();
        }
    }

public:
    constexpr Port() noexcept = default;
    explicit Port(std::int16_t priority_, std::size_t min_samples_ = 0UZ, std::size_t max_samples_ = SIZE_MAX) noexcept : priority{priority_}, min_samples(min_samples_), max_samples(max_samples_), _ioHandler{newIoHandler()}, _tagIoHandler{newTagIoHandler()} {}
    constexpr Port(Port&& other) noexcept : name(other.name), priority{other.priority}, min_samples(other.min_samples), max_samples(other.max_samples), metaInfo(std::move(other.metaInfo)), _ioHandler(std::move(other._ioHandler)), _tagIoHandler(std::move(other._tagIoHandler)) {}
    Port(const Port&)                       = delete;
    auto            operator=(const Port&)  = delete;
    constexpr Port& operator=(Port&& other) = delete;
    ~Port()                                 = default;

    [[nodiscard]] constexpr bool initBuffer(std::size_t nSamples = 0) noexcept {
        if constexpr (kIsOutput) {
            // write one default value into output -- needed for cyclic graph initialisation
            return _ioHandler.try_publish([val = default_value](std::span<T>& out) { std::ranges::fill(out, val); }, nSamples);
        }
        return true;
    }

    [[nodiscard]] InternalPortBuffers writerHandlerInternal() noexcept
    requires(kIsOutput)
    {
        return {static_cast<void*>(std::addressof(_ioHandler)), static_cast<void*>(std::addressof(_tagIoHandler))};
    }

    [[nodiscard]] bool updateReaderInternal(InternalPortBuffers buffer_writer_handler_other) noexcept
    requires(kIsInput)
    {
        if (buffer_writer_handler_other.streamHandler == nullptr) {
            return false;
        }
        if (buffer_writer_handler_other.tagHandler == nullptr) {
            return false;
        }

        // TODO: If we want to allow ports with different buffer types to be mixed
        //       this will fail. We need to add a check that two ports that
        //       connect to each other use the same buffer type
        //       (std::any could be a viable approach)
        auto typed_buffer_writer     = static_cast<WriterType*>(buffer_writer_handler_other.streamHandler);
        auto typed_tag_buffer_writer = static_cast<TagWriterType*>(buffer_writer_handler_other.tagHandler);
        setBuffer(typed_buffer_writer->buffer(), typed_tag_buffer_writer->buffer());
        return true;
    }

    [[nodiscard]] constexpr bool isConnected() const noexcept {
        if constexpr (kIsInput) {
            return _ioHandler.buffer().n_writers() > 0;
        } else {
            return _ioHandler.buffer().n_readers() > 0;
        }
    }

    [[nodiscard]] constexpr static PortType type() noexcept { return portType; }

    [[nodiscard]] constexpr static PortDirection direction() noexcept { return portDirection; }

    [[nodiscard]] constexpr static std::string_view domain() noexcept { return std::string_view(Domain::Name); }

    [[nodiscard]] constexpr static bool isSynchronous() noexcept { return kIsSynch; }

    [[nodiscard]] constexpr static bool isOptional() noexcept { return kIsOptional; }

    [[nodiscard]] constexpr std::size_t nReaders() const noexcept {
        if constexpr (kIsInput) {
            return -1UZ;
        } else {
            return _ioHandler.buffer().n_readers();
        }
    }

    [[nodiscard]] constexpr std::size_t nWriters() const noexcept {
        if constexpr (kIsInput) {
            return _ioHandler.buffer().n_writers();
        } else {
            return -1UZ;
        }
    }

    [[nodiscard]] constexpr std::size_t bufferSize() const noexcept { return _ioHandler.buffer().size(); }

    [[nodiscard]] std::any defaultValue() const noexcept { return default_value; }

    [[nodiscard]] bool setDefaultValue(const std::any& newDefault) {
        if (newDefault.type() == typeid(T)) {
            default_value = std::any_cast<T>(newDefault);
            return true;
        }
        return false;
    }

    [[nodiscard]] constexpr std::size_t available() const noexcept {
        if constexpr (kIsInput) {
            return streamReader().available();
        } else {
            return streamWriter().available();
        }
    }

    [[nodiscard]] constexpr std::size_t min_buffer_size() const noexcept {
        if constexpr (Required::kIsConst) {
            return Required::kMinSamples;
        } else {
            return min_samples;
        }
    }

    [[nodiscard]] constexpr std::size_t max_buffer_size() const noexcept {
        if constexpr (Required::kIsConst) {
            return Required::kMaxSamples;
        } else {
            return max_samples;
        }
    }

    [[nodiscard]] constexpr ConnectionResult resizeBuffer(std::size_t min_size) noexcept {
        using enum gr::ConnectionResult;
        if constexpr (kIsInput) {
            return SUCCESS;
        } else {
            try {
                _ioHandler    = BufferType(min_size).new_writer();
                _tagIoHandler = TagBufferType(min_size).new_writer();
            } catch (...) {
                return FAILED;
            }
        }
        return SUCCESS;
    }

    [[nodiscard]] auto buffer() {
        struct port_buffers {
            BufferType    streamBuffer;
            TagBufferType tagBuffer;
        };

        return port_buffers{_ioHandler.buffer(), _tagIoHandler.buffer()};
    }

    void setBuffer(gr::BufferLike auto streamBuffer, gr::BufferLike auto tagBuffer) noexcept {
        if constexpr (kIsInput) {
            _ioHandler    = streamBuffer.new_reader();
            _tagIoHandler = tagBuffer.new_reader();
        } else {
            _ioHandler    = streamBuffer.new_writer();
            _tagIoHandler = tagBuffer.new_writer();
        }
    }

    [[nodiscard]] constexpr const ReaderType& streamReader() const noexcept {
        static_assert(!kIsOutput, "streamReader() not applicable for outputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] constexpr ReaderType& streamReader() noexcept {
        static_assert(!kIsOutput, "streamReader() not applicable for outputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] constexpr const WriterType& streamWriter() const noexcept {
        static_assert(!kIsInput, "streamWriter() not applicable for inputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] constexpr WriterType& streamWriter() noexcept {
        static_assert(!kIsInput, "streamWriter() not applicable for inputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] constexpr const TagReaderType& tagReader() const noexcept {
        static_assert(!kIsOutput, "tagReader() not applicable for outputs (yet)");
        return _tagIoHandler;
    }

    [[nodiscard]] constexpr TagReaderType& tagReader() noexcept {
        static_assert(!kIsOutput, "tagReader() not applicable for outputs (yet)");
        return _tagIoHandler;
    }

    [[nodiscard]] constexpr const TagWriterType& tagWriter() const noexcept {
        static_assert(!kIsInput, "tagWriter() not applicable for inputs (yet)");
        return _tagIoHandler;
    }

    [[nodiscard]] constexpr TagWriterType& tagWriter() noexcept {
        static_assert(!kIsInput, "tagWriter() not applicable for inputs (yet)");
        return _tagIoHandler;
    }

    [[nodiscard]] ConnectionResult disconnect() noexcept {
        if (isConnected() == false) {
            return ConnectionResult::FAILED;
        }
        _ioHandler    = newIoHandler();
        _tagIoHandler = newTagIoHandler();
        return ConnectionResult::SUCCESS;
    }

    template<typename Other>
    [[nodiscard]] ConnectionResult connect(Other&& other) {
        static_assert(kIsOutput && std::remove_cvref_t<Other>::kIsInput);
        static_assert(std::is_same_v<value_type, typename std::remove_cvref_t<Other>::value_type>);
        auto src_buffer = writerHandlerInternal();
        return std::forward<Other>(other).updateReaderInternal(src_buffer) ? ConnectionResult::SUCCESS : ConnectionResult::FAILED;
    }

    template<SpanReleasePolicy spanReleasePolicy, bool consumeOnlyFirstTag = false>
    InputSpan<spanReleasePolicy, consumeOnlyFirstTag> get(std::size_t nSamples)
    requires(kIsInput)
    {
        return InputSpan<spanReleasePolicy, consumeOnlyFirstTag>(nSamples, streamReader(), tagReader(), this->isConnected(), this->isSynchronous());
    }

    template<SpanReleasePolicy spanReleasePolicy>
    auto reserve(std::size_t nSamples)
    requires(kIsOutput)
    {
        return OutputSpan<spanReleasePolicy, WriterSpanReservePolicy::Reserve>(nSamples, streamWriter(), tagWriter(), streamWriter().position(), this->isConnected(), this->isSynchronous());
    }

    template<SpanReleasePolicy spanReleasePolicy>
    auto tryReserve(std::size_t nSamples)
    requires(kIsOutput)
    {
        return OutputSpan<spanReleasePolicy, WriterSpanReservePolicy::TryReserve>(nSamples, streamWriter(), tagWriter(), streamWriter().position(), this->isConnected(), this->isSynchronous());
    }

    template<PropertyMapType TPropertyMap>
    inline constexpr void publishTag(TPropertyMap&& tagData, std::size_t tagOffset = 0UZ) noexcept
    requires(kIsOutput)
    {
        if (isConnected()) {
            WriterSpanLike auto outTags = tagWriter().tryReserve(1UZ);
            if (!outTags.empty()) {
                outTags[0].index = streamWriter().position() + tagOffset;
                outTags[0].map   = std::forward<TPropertyMap>(tagData);
                outTags.publish(1UZ);
            } else {
                // TODO(error handling): Decide how to surface failures. Function is noexcept now
            }
        }
    }

private:
    friend class DynamicPort;
};

template<typename T, typename... Attributes>
using PortIn = Port<T, PortType::STREAM, PortDirection::INPUT, Attributes...>;
template<typename T, typename... Attributes>
using PortOut = Port<T, PortType::STREAM, PortDirection::OUTPUT, Attributes...>;

using MsgPortIn  = Port<Message, PortType::MESSAGE, PortDirection::INPUT, DefaultMessageBuffer>;
using MsgPortOut = Port<Message, PortType::MESSAGE, PortDirection::OUTPUT, DefaultMessageBuffer>;

struct BuiltinTag {};
using MsgPortInBuiltin  = Port<Message, PortType::MESSAGE, PortDirection::INPUT, DefaultMessageBuffer, BuiltinTag>;
using MsgPortOutBuiltin = Port<Message, PortType::MESSAGE, PortDirection::OUTPUT, DefaultMessageBuffer, BuiltinTag>;

struct FromChildrenTag {};
using MsgPortInFromChildren = Port<Message, PortType::MESSAGE, PortDirection::INPUT, DefaultMessageBuffer, FromChildrenTag>;

struct ForChildrenTag {};
using MsgPortOutForChildren = Port<Message, PortType::MESSAGE, PortDirection::OUTPUT, DefaultMessageBuffer, ForChildrenTag>;

static_assert(PortLike<PortIn<float>>);
static_assert(PortLike<decltype(PortIn<float>())>);
static_assert(PortLike<PortOut<float>>);
static_assert(PortLike<MsgPortIn>);
static_assert(PortLike<MsgPortOut>);

static_assert(std::is_same_v<MsgPortIn::BufferType, gr::CircularBuffer<Message, std::dynamic_extent, gr::ProducerType::Multi>>);

static_assert(PortIn<float, RequiredSamples<1, 2>>::Required::kMinSamples == 1);
static_assert(PortIn<float, RequiredSamples<1, 2>>::Required::kMaxSamples == 2);
static_assert(std::same_as<PortIn<float, RequiredSamples<1, 2>>::Domain, CPU>);
static_assert(std::same_as<PortIn<float, RequiredSamples<1, 2>, GPU>::Domain, GPU>);

static_assert(PortIn<float>::kPortType == PortType::STREAM);
static_assert(PortIn<Message>::kPortType == PortType::STREAM);
static_assert(MsgPortIn::kPortType == PortType::MESSAGE);

static_assert(std::is_default_constructible_v<PortIn<float>>);
static_assert(std::is_default_constructible_v<PortOut<float>>);

/**
 *  Runtime capable wrapper to be used within a block. It's primary purpose is to allow the runtime
 *  initialisation/connections between blocks that are not in the same compilation unit.
 *  Ownership is defined by if the strongly-typed port P is either passed
 *  a) as an lvalue (i.e. P& -> keep reference), or
 *  b) as an rvalue (P&& -> being moved into dyn_port).
 *
 *  N.B. the intended use is within the node/block interface where there is -- once initialised --
 *  always a strong-reference between the strongly-typed port and it's dyn_port wrapper. I.e no ports
 *  are added or removed after the initialisation and the port life-time is coupled to that of it's
 *  parent block/node.
 */
class DynamicPort {
public:
    std::string  name;
    std::int16_t priority; // → dependents of a higher-prio port should be scheduled first (Q: make this by order of ports?)
    std::size_t  min_samples;
    std::size_t  max_samples;
    PortMetaInfo metaInfo;

private:
    struct model { // intentionally class-private definition to limit interface exposure and enhance composition
        virtual ~model() = default;

        [[nodiscard]] virtual DynamicPort weakRef() const noexcept = 0;

        [[nodiscard]] virtual std::intptr_t    internalId() const noexcept                   = 0;
        [[nodiscard]] virtual std::any         defaultValue() const noexcept                 = 0;
        [[nodiscard]] virtual bool             setDefaultValue(const std::any& val) noexcept = 0;
        [[nodiscard]] virtual PortType         type() const noexcept                         = 0;
        [[nodiscard]] virtual PortDirection    direction() const noexcept                    = 0;
        [[nodiscard]] virtual std::string_view domain() const noexcept                       = 0;
        [[nodiscard]] virtual bool             isSynchronous() noexcept                      = 0;
        [[nodiscard]] virtual bool             isOptional() noexcept                         = 0;

        [[nodiscard]] virtual ConnectionResult resizeBuffer(std::size_t min_size) noexcept = 0;
        [[nodiscard]] virtual bool             isConnected() const noexcept                = 0;
        [[nodiscard]] virtual ConnectionResult disconnect() noexcept                       = 0;
        [[nodiscard]] virtual ConnectionResult connect(DynamicPort& dst_port)              = 0;

        // internal runtime polymorphism access
        [[nodiscard]] virtual bool updateReaderInternal(InternalPortBuffers buffer_other) noexcept = 0;

        [[nodiscard]] virtual std::size_t nReaders() const   = 0;
        [[nodiscard]] virtual std::size_t nWriters() const   = 0;
        [[nodiscard]] virtual std::size_t bufferSize() const = 0;

        [[nodiscard]] virtual std::string typeName() const = 0;

        [[nodiscard]] virtual std::string_view portName() noexcept       = 0; // TODO: rename to 'name()' and eliminate local 'name' field (moved to metaInfo()), and use string&
        [[nodiscard]] virtual std::string_view portName() const noexcept = 0;

        [[nodiscard]] virtual PortInfo            portInfo() const              = 0; // TODO: rename to type() and remove existing type(), direction(), domain(), ... API
        [[nodiscard]] virtual PortMetaInfo const& portMetaInfo() const noexcept = 0;
        [[nodiscard]] virtual PortMetaInfo&       portMetaInfo() noexcept       = 0;
        [[nodiscard]] virtual port::BitMask       portMaskInfo() const noexcept = 0;
    };

    std::unique_ptr<model> _accessor;

    template<PortLike T, bool owning>
    class PortWrapper final : public model {
        using TPortType = std::decay_t<T>;
        std::conditional_t<owning, TPortType, TPortType&> _value;

        [[nodiscard]] InternalPortBuffers writerHandlerInternal() noexcept { return _value.writerHandlerInternal(); };

        [[nodiscard]] bool updateReaderInternal(InternalPortBuffers buffer_other) noexcept override {
            if constexpr (T::kIsInput) {
                return _value.updateReaderInternal(buffer_other);
            } else {
                assert(false && "This works only on input ports");
                return false;
            }
        }

    public:
        PortWrapper() = delete;

        PortWrapper(const PortWrapper&) = delete;
        PortWrapper(PortWrapper&&)      = delete;

        auto& operator=(const PortWrapper&) = delete;
        auto& operator=(PortWrapper&&)      = delete;

        explicit constexpr PortWrapper(T& arg) noexcept : _value{arg} {
            if constexpr (T::kIsInput) {
                static_assert(requires { arg.updateReaderInternal(std::declval<InternalPortBuffers>()); }, "'private bool updateReaderInternal(void* buffer)' not implemented");
            } else {
                static_assert(requires { arg.writerHandlerInternal(); }, "'private void* writerHandlerInternal()' not implemented");
            }
        }

        explicit constexpr PortWrapper(T&& arg) noexcept : _value{std::move(arg)} {
            if constexpr (T::kIsInput) {
                static_assert(requires { arg.updateReaderInternal(std::declval<InternalPortBuffers>()); }, "'private bool updateReaderInternal(void* buffer)' not implemented");
            } else {
                static_assert(requires { arg.writerHandlerInternal(); }, "'private void* writerHandlerInternal()' not implemented");
            }
        }

        ~PortWrapper() override = default;

        [[nodiscard]] DynamicPort weakRef() const noexcept override;

        [[nodiscard]] std::intptr_t internalId() const noexcept override { return reinterpret_cast<std::intptr_t>(std::addressof(_value)); }

        [[nodiscard]] std::any defaultValue() const noexcept override { return _value.defaultValue(); }
        [[nodiscard]] bool     setDefaultValue(const std::any& val) noexcept override { return _value.setDefaultValue(val); }

        [[nodiscard]] constexpr PortType         type() const noexcept override { return _value.type(); }
        [[nodiscard]] constexpr PortDirection    direction() const noexcept override { return _value.direction(); }
        [[nodiscard]] constexpr std::string_view domain() const noexcept override { return _value.domain(); }
        [[nodiscard]] bool                       isSynchronous() noexcept override { return _value.isSynchronous(); }
        [[nodiscard]] bool                       isOptional() noexcept override { return _value.isOptional(); }

        [[nodiscard]] ConnectionResult resizeBuffer(std::size_t min_size) noexcept override { return _value.resizeBuffer(min_size); }
        [[nodiscard]] std::size_t      nReaders() const override { return _value.nReaders(); }
        [[nodiscard]] std::size_t      nWriters() const override { return _value.nWriters(); }
        [[nodiscard]] std::size_t      bufferSize() const override { return _value.bufferSize(); }
        [[nodiscard]] bool             isConnected() const noexcept override { return _value.isConnected(); }
        [[nodiscard]] ConnectionResult disconnect() noexcept override { return _value.disconnect(); }

        [[nodiscard]] ConnectionResult connect(DynamicPort& dst_port) override { // TODO: return signature: refactor to non-throwing std::expected<ConnectionResult, Error> return -> follow-up PR
            using enum gr::ConnectionResult;
            port::BitMask thisMask = portMaskInfo();
            port::BitMask other    = dst_port.portMaskInfo();
            if (port::decodePortType(thisMask) != port::decodePortType(other)) {
#ifdef DEBUG
                throw std::runtime_error(std::format("port type mismatch: {}::{} != {}::{}", portName(), port::decodePortType(thisMask), dst_port.portName(), port::decodePortType(other)));
#endif
                return FAILED;
            }
            if (portMetaInfo().data_type != dst_port.portMetaInfo().data_type) {
#ifdef DEBUG
                throw std::runtime_error(std::format("port data type mismatch: {}::{} != {}::{}", portName(), _value.metaInfo.data_type, dst_port.portName(), dst_port.metaInfo.data_type));
#endif
                return FAILED;
            }
            if constexpr (T::kIsOutput) {
                auto src_buffer = _value.writerHandlerInternal();
                return dst_port.updateReaderInternal(src_buffer) ? SUCCESS : FAILED;
            } else {
#ifdef DEBUG
                throw std::runtime_error("This works only on input ports");
#endif
                return FAILED;
            }
        }

        [[nodiscard]] std::string typeName() const override { return meta::type_name<typename T::value_type>(); }

        [[nodiscard]] std::string_view portName() noexcept override { return _value.name; } // TODO: '_value.name' -> '_value.metaInfo.name' and use string&
        [[nodiscard]] std::string_view portName() const noexcept override { return _value.name; }

        [[nodiscard]] PortInfo portInfo() const override {
            return {// snapshot
                .portType                  = T::kPortType,
                .portDirection             = T::kDirection,
                .portDomain                = T::Domain::Name,
                .portConnectionResult      = _value.isConnected() ? ConnectionResult::SUCCESS : ConnectionResult::FAILED,
                .valueTypeName             = meta::type_name<typename T::value_type>(),
                .isValueTypeArithmeticLike = T::kIsArithmeticLikeValueType,
                .valueTypeSize             = sizeof(typename T::value_type),
                .bufferSize                = _value.bufferSize(),
                .availableBufferSize       = _value.available()};
        }

        [[nodiscard]] PortMetaInfo const& portMetaInfo() const noexcept override { return _value.metaInfo; }
        [[nodiscard]] PortMetaInfo&       portMetaInfo() noexcept override { return _value.metaInfo; }
        [[nodiscard]] port::BitMask       portMaskInfo() const noexcept override { return port::encodeMask(T::kDirection, T::kPortType, T::kIsSynch, T::kIsOptional, _value.isConnected()); }
    };

    bool updateReaderInternal(InternalPortBuffers buffer_other) noexcept { return _accessor->updateReaderInternal(buffer_other); }

public:
    using value_type = void; // a sterile port

    struct owned_value_tag {};

    struct non_owned_reference_tag {};

    constexpr DynamicPort() = delete;

    DynamicPort(const DynamicPort& arg)            = delete;
    DynamicPort& operator=(const DynamicPort& arg) = delete;

    DynamicPort(DynamicPort&& other) noexcept : name(other.name), priority(other.priority), min_samples(other.min_samples), max_samples(other.max_samples), _accessor(std::move(other._accessor)) {}
    auto& operator=(DynamicPort&& other) noexcept {
        auto tmp = std::move(other);
        std::swap(_accessor, tmp._accessor);
        std::swap(name, tmp.name);
        std::swap(priority, tmp.priority);
        std::swap(min_samples, tmp.min_samples);
        std::swap(max_samples, tmp.max_samples);
        return *this;
    }

    template<class T>
    explicit constexpr DynamicPort(const T& arg, non_owned_reference_tag) noexcept                            // TODO: remove const-cast (super dangerous, and only a temporary fix) -> Ivan volunteerd to fix in follor-up PR
    requires PortLike<std::remove_const_t<T>>                                                                 //
        : name(arg.name), priority(arg.priority), min_samples(arg.min_samples), max_samples(arg.max_samples), //
          _accessor{std::make_unique<PortWrapper<std::remove_const_t<T>, false>>(const_cast<std::remove_const_t<T>&>(arg))} {}

    bool operator==(const DynamicPort& other) const noexcept { return _accessor->internalId() == other._accessor->internalId(); }
    bool operator!=(const DynamicPort& other) const noexcept { return _accessor->internalId() != other._accessor->internalId(); }

    // TODO: The lifetime of ports is a problem here, if we keep a reference to the port in DynamicPort, the port object/ can not be reallocated
    template<PortLike T>
    explicit constexpr DynamicPort(T& arg, non_owned_reference_tag) noexcept : name(arg.name), priority(arg.priority), min_samples(arg.min_samples), max_samples(arg.max_samples), _accessor{std::make_unique<PortWrapper<T, false>>(arg)} {}

    template<PortLike T>
    explicit constexpr DynamicPort(T&& arg, owned_value_tag) noexcept : name(arg.name), priority(arg.priority), min_samples(arg.min_samples), max_samples(arg.max_samples), _accessor{std::make_unique<PortWrapper<T, true>>(std::forward<T>(arg))} {}

    [[nodiscard]] DynamicPort weakRef() const noexcept { return _accessor->weakRef(); }
    [[nodiscard]] std::any    defaultValue() const noexcept { return _accessor->defaultValue(); }

    [[nodiscard]] bool             setDefaultValue(const std::any& val) noexcept { return _accessor->setDefaultValue(val); }
    [[nodiscard]] PortType         type() const noexcept { return _accessor->type(); }
    [[nodiscard]] PortDirection    direction() const noexcept { return _accessor->direction(); }
    [[nodiscard]] std::string_view domain() const noexcept { return _accessor->domain(); }
    [[nodiscard]] std::string      typeName() const noexcept { return _accessor->typeName(); }
    [[nodiscard]] std::string_view portName() noexcept { return _accessor->portName(); }
    [[nodiscard]] std::string_view portName() const noexcept { return _accessor->portName(); }
    [[nodiscard]] PortInfo         portInfo() const noexcept { return _accessor->portInfo(); }
    [[nodiscard]] PortMetaInfo     portMetaInfo() const noexcept { return _accessor->portMetaInfo(); }
    [[nodiscard]] port::BitMask    portMaskInfo() const noexcept { return _accessor->portMaskInfo(); }

    [[nodiscard]] bool isSynchronous() noexcept { return _accessor->isSynchronous(); }

    [[nodiscard]] bool isOptional() noexcept { return _accessor->isOptional(); }

    [[nodiscard]] ConnectionResult resizeBuffer(std::size_t min_size) {
        if (direction() == PortDirection::OUTPUT) {
            return _accessor->resizeBuffer(min_size);
        }
        return ConnectionResult::FAILED;
    }

    [[nodiscard]] bool isConnected() const noexcept { return _accessor->isConnected(); }

    [[nodiscard]] std::size_t nReaders() const { return _accessor->nReaders(); }
    [[nodiscard]] std::size_t nWriters() const { return _accessor->nWriters(); }
    [[nodiscard]] std::size_t bufferSize() const { return _accessor->bufferSize(); }

    [[nodiscard]] ConnectionResult disconnect() noexcept { return _accessor->disconnect(); }

    [[nodiscard]] ConnectionResult connect(DynamicPort& dst_port) { return _accessor->connect(dst_port); }
};

template<PortLike T, bool owning>
[[nodiscard]] DynamicPort DynamicPort::PortWrapper<T, owning>::weakRef() const noexcept {
    return DynamicPort(_value, DynamicPort::non_owned_reference_tag{});
}

static_assert(PortLike<DynamicPort>);

namespace detail {
template<typename T>
concept TagPredicate = requires(const T& t, const Tag& tag, std::size_t readPosition) {
    { t(tag, readPosition) } -> std::convertible_to<bool>;
};
inline constexpr TagPredicate auto defaultTagMatcher    = [](const Tag& tag, std::size_t readPosition) noexcept { return tag.index >= readPosition; };
inline constexpr TagPredicate auto defaultEOSTagMatcher = [](const Tag& tag, std::size_t readPosition) noexcept {
    if (tag.index < readPosition) {
        return false;
    }
    auto eosTagIter = tag.map.find(gr::tag::END_OF_STREAM);
    return eosTagIter != tag.map.end() && eosTagIter->second == true;
};
} // namespace detail

inline constexpr std::optional<std::size_t> nSamplesToNextTagConditional(PortLike auto& port, detail::TagPredicate auto& predicate, std::size_t readOffset) {
    ReaderSpanLike auto tagData = port.tagReader().get();
    if (!port.isConnected() || tagData.empty()) [[likely]] {
        return std::nullopt; // default: no tags in sight
    }
    const std::size_t readPosition = port.streamReader().position();

    // at least one tag is present -> if tag is not on the first tag position read up to the tag position
    const auto firstMatchingTag = std::ranges::find_if(tagData, [&](const auto& tag) { return predicate(tag, readPosition + readOffset); });
    std::ignore                 = tagData.consume(0UZ);
    if (firstMatchingTag != tagData.end()) {
        return static_cast<std::size_t>(std::max(firstMatchingTag->index - readPosition, std::size_t(0))); // Tags in the past will have a negative distance -> deliberately map them to '0'
    } else {
        return std::nullopt;
    }
}

inline constexpr std::optional<std::size_t> nSamplesUntilNextTag(PortLike auto& port, std::size_t offset = 0) { return nSamplesToNextTagConditional(port, detail::defaultTagMatcher, offset); }

inline constexpr std::optional<std::size_t> samples_to_eos_tag(PortLike auto& port, std::size_t offset = 0) { return nSamplesToNextTagConditional(port, detail::defaultEOSTagMatcher, offset); }

} // namespace gr

#endif // GNURADIO_PORT_HPP
