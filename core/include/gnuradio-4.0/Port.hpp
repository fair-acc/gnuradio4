#ifndef GNURADIO_PORT_HPP
#define GNURADIO_PORT_HPP

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

enum class PortDirection { INPUT, OUTPUT, ANY }; // 'ANY' only for query and not to be used for port declarations

enum class ConnectionResult { SUCCESS, FAILED };

enum class PortType {
    STREAM,  /*!< used for single-producer-only ond usually synchronous one-to-one or one-to-many communications */
    MESSAGE, /*!< used for multiple-producer one-to-one, one-to-many, many-to-one, or many-to-many communications */
    ANY      // 'ANY' only for querying and not to be used for port declarations
};

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

struct CPU : public PortDomain<"CPU"> {};

struct GPU : public PortDomain<"GPU"> {};

static_assert(is_port_domain<CPU>::value);
static_assert(is_port_domain<GPU>::value);
static_assert(!is_port_domain<int>::value);

template<class T>
concept PortLike = requires(T t, const std::size_t n_items, const std::any& newDefault) { // dynamic definitions
    typename T::value_type;
    { t.defaultValue() } -> std::same_as<std::any>;
    { t.setDefaultValue(newDefault) } -> std::same_as<bool>;
    { t.name } -> std::convertible_to<std::string_view>;
    { t.priority } -> std::convertible_to<std::int32_t>;
    { t.min_samples } -> std::convertible_to<std::size_t>;
    { t.max_samples } -> std::convertible_to<std::size_t>;
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

    Annotated<float, "sample rate", Visible, Doc<"sampling rate in samples per second (Hz)">>                        sample_rate = 1.f;
    Annotated<std::string, "signal name", Doc<"name of the signal">>                                                 signal_name = "<unnamed>";
    Annotated<std::string, "signal quantity", Doc<"physical quantity (e.g., 'voltage'). Follows ISO 80000-1:2022.">> signal_quantity{};
    Annotated<std::string, "signal unit", Doc<"unit of measurement (e.g., '[V]', '[m]'). Follows ISO 80000-1:2022">> signal_unit{};
    Annotated<float, "signal min,", Doc<"minimum expected signal value">>                                            signal_min = std::numeric_limits<float>::lowest();
    Annotated<float, "signal max,", Doc<"maximum expected signal value">>                                            signal_max = std::numeric_limits<float>::max();

    // controls automatic (if set) or manual update of above parameters
    std::set<std::string, std::less<>> auto_update{"sample_rate", "signal_name", "signal_quantity", "signal_unit", "signal_min", "signal_max"};

    constexpr PortMetaInfo() noexcept = default;
    explicit PortMetaInfo(std::initializer_list<std::pair<const std::string, pmtv::pmt>> initMetaInfo) noexcept(true) : PortMetaInfo(property_map{initMetaInfo.begin(), initMetaInfo.end()}) {}
    explicit PortMetaInfo(const property_map& metaInfo) noexcept(true) { update<true>(metaInfo); }

    void reset() { auto_update = {"sample_rate", "signal_name", "signal_quantity", "signal_unit", "signal_min", "signal_max"}; }

    template<bool isNoexcept = false>
    void update(const property_map& metaInfo) noexcept(isNoexcept) {
        if (metaInfo.empty()) {
            return;
        }

        auto updateValue = [&metaInfo](const std::string& key, auto& member) {
            if (!metaInfo.contains(key)) {
                return;
            }
            const auto& value = metaInfo.at(key);
            using T           = std::decay_t<decltype(member.value)>;
            if (std::holds_alternative<T>(value)) {
                member = std::get<T>(value);
            } else {
                throw gr::exception("invalid-argument: incorrect type for " + key);
            }
        };

        for (const auto& key : auto_update) {
            if (key == "sample_rate") {
                updateValue(key, sample_rate);
            } else if (key == "signal_name") {
                updateValue(key, signal_name);
            } else if (key == "signal_quantity") {
                updateValue(key, signal_quantity);
            } else if (key == "signal_unit") {
                updateValue(key, signal_unit);
            } else if (key == "signal_min") {
                updateValue(key, signal_min);
            } else if (key == "signal_max") {
                updateValue(key, signal_max);
            }
        }
    }

    [[nodiscard]] property_map get() const noexcept {
        property_map metaInfo;
        metaInfo["sample_rate"]     = sample_rate;
        metaInfo["signal_name"]     = signal_name;
        metaInfo["signal_quantity"] = signal_quantity;
        metaInfo["signal_unit"]     = signal_unit;
        metaInfo["signal_min"]      = signal_min;
        metaInfo["signal_max"]      = signal_max;

        return metaInfo;
    }
};

} // namespace gr
ENABLE_REFLECTION(gr::PortMetaInfo, sample_rate, signal_name, signal_quantity, signal_unit, signal_min, signal_max)

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
 * @brief API for access to the input samples and tags from the process_bulk function.
 *
 * This concept is used for the input parameters of the process_bulk function to allow:
 * - access to the samples (via conversion to std::span)
 * - consumption of samples. By default all available samples get consumed.
 * - access to tags
 *   - via a range::view returned by `tag()` which returns the index relative to first sample in this span. The index can be negative for tags that were not consumed before.
 *   - via rawTags, which gives access to the bare ConsumableSpan<Tag>
 * - consumption of tags. By default the tags belonging to up to and including the first sample get consumed. This can be manually changed by calling consumeTags with the index of a stream sample.
 * - get a merged tag which contains the data of all tags belonging to up to and including the first sample. Optionally this can be changed to merge all tags until a supplied local stream index.
 */
template<typename T>
concept InputSpan = requires(T span, gr::Tag::signed_index_type n) {
    { span } -> std::ranges::contiguous_range;
    { span.consume(0) };
    { span.rawTags };
    requires ConsumableSpan<std::remove_cvref_t<decltype(span.rawTags)>> && std::same_as<gr::Tag, std::ranges::range_value_t<decltype(span.rawTags)>>;
    { span.tags() } -> std::ranges::range;
    { span.consumeTags(n) };
    { span.getMergedTag(n) } -> std::same_as<gr::Tag>;
};

template<typename T>
concept PublishablePortSpan = PublishableSpan<T> && requires(T span, property_map& tagData, Tag::signed_index_type index) {
    requires PublishableSpan<std::remove_cvref_t<decltype(span.tags)>>;
    { *span.tags.begin() } -> std::same_as<gr::Tag&>;
    { span.publishTag(tagData, index) } -> std::same_as<void>;
};

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
 * @tparam portName a string to identify the port, notably to be used in an UI- and hand-written explicit code context.
 * @tparam portType STREAM  or MESSAGE
 * @tparam portDirection either input or output
 * @tparam Attributes optional: default to 'DefaultStreamBuffer' and DefaultTagBuffer' based on 'gr::circular_buffer', and CPU domain
 */
template<typename T, fixed_string portName, PortType portType, PortDirection portDirection, typename... Attributes>
struct Port {
    template<fixed_string newName, typename ReflDescriptor>
    using with_name_and_descriptor = Port<T, newName, portType, portDirection, ReflDescriptor, Attributes...>;

    static_assert(portDirection != PortDirection::ANY, "ANY reserved for queries and not port direction declarations");
    static_assert(portType != PortType::ANY, "ANY reserved for queries and not port type declarations");
    static_assert(portType == PortType::STREAM || std::is_same_v<T, gr::Message>, "If a port type is MESSAGE, the value type needs to be gr::Message");

    using value_type        = T;
    using AttributeTypeList = typename gr::meta::typelist<Attributes...>;
    using Domain            = AttributeTypeList::template find_or_default<is_port_domain, CPU>;
    using Required          = AttributeTypeList::template find_or_default<is_required_samples, RequiredSamples<std::dynamic_extent, std::dynamic_extent>>;
    using BufferType        = AttributeTypeList::template find_or_default<is_stream_buffer_attribute, DefaultStreamBuffer<T>>::type;
    using TagBufferType     = AttributeTypeList::template find_or_default<is_tag_buffer_attribute, DefaultTagBuffer>::type;
    using ReflDescriptor    = AttributeTypeList::template find_or_default<refl::trait::is_descriptor, std::false_type>;

    // constexpr members:
    static constexpr PortDirection kDirection = portDirection;
    static constexpr PortType      kPortType  = portType;
    static constexpr bool          kIsInput   = portDirection == PortDirection::INPUT;
    static constexpr bool          kIsOutput  = portDirection == PortDirection::OUTPUT;
    static constexpr fixed_string  Name       = portName;

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
    constexpr static bool kIsSynch      = !std::disjunction_v<std::is_same<Async, Attributes>...>;
    constexpr static bool kIsOptional   = std::disjunction_v<std::is_same<Optional, Attributes>...>;
    std::string           name          = static_cast<std::string>(portName);
    std::int16_t          priority      = 0; // → dependents of a higher-prio port should be scheduled first (Q: make this by order of ports?)
    T                     default_value = T{};

    //
    std::conditional_t<Required::kIsConst, const std::size_t, std::size_t> min_samples = Required::kMinSamples;
    std::conditional_t<Required::kIsConst, const std::size_t, std::size_t> max_samples = Required::kMaxSamples;

    // Port meta-information for increased type and physical-unit safety. Uses ISO 80000-1:2022 conventions.
    PortMetaInfo metaInfo{};

    template<SpanReleasePolicy spanReleasePolicy>
    using ReaderSpanType = decltype(std::declval<ReaderType>().template get<spanReleasePolicy>());

    template<SpanReleasePolicy spanReleasePolicy>
    struct PortInputSpan : public ReaderSpanType<spanReleasePolicy> {
        TagReaderSpanType      rawTags;
        Tag::signed_index_type streamIndex;

        PortInputSpan(std::size_t nSamples, ReaderType& reader, TagReaderType& tagReader) : ReaderSpanType<spanReleasePolicy>(reader.template get<spanReleasePolicy>(nSamples)), rawTags(getTags(static_cast<gr::Tag::signed_index_type>(nSamples), tagReader, reader.position())), streamIndex{reader.position()} {};

        ~PortInputSpan() override {
            if (ReaderSpanType<spanReleasePolicy>::instanceCount() == 1UZ) { // has to be one, because the parent destructor which decrements it to zero is only called afterward
                if (rawTags.isConsumeRequested()) {                             // the user has already manually consumed tags
                    return;
                }
                if ((ReaderSpanType<spanReleasePolicy>::isConsumeRequested() && ReaderSpanType<spanReleasePolicy>::nRequestedSamplesToConsume() == 0) || this->empty()) {
                    return; // no samples to be consumed -> do not consume any tags
                }
                consumeTags(0); // consume all tags including the one on the first sample
            }
        }

        [[nodiscard]] auto tags() {
            return std::views::transform(rawTags, [this](auto &tag) { return std::make_pair(std::max(tag.index, 0l) - streamIndex, std::ref(tag.map)); });
        }

        void consumeTags(gr::Tag::signed_index_type untilLocalIndex) {
            std::size_t tagsToConsume = static_cast<std::size_t>(std::ranges::count_if(
                    rawTags | std::views::take_while([untilLocalIndex, this](auto& t) { return t.index <= streamIndex + untilLocalIndex; }),
                    [](auto /*v*/) {return true;} ));
            std::ignore = rawTags.tryConsume(tagsToConsume);
        }

        [[nodiscard]] inline Tag getMergedTag(gr::Tag::signed_index_type untilLocalIndex = 0) const {
            auto mergeSrcMapInto = [](const property_map& sourceMap, property_map& destinationMap) {
                assert(&sourceMap != &destinationMap);
                for (const auto& [key, value] : sourceMap) {
                    destinationMap.insert_or_assign(key, value);
                }
            };
            Tag  result{-1, {}};
            std::ranges::for_each(rawTags | std::views::take_while([untilLocalIndex, this](auto& t) { return t.index <= streamIndex + untilLocalIndex; }),
                                  [&mergeSrcMapInto, &result](const Tag& tag) { mergeSrcMapInto(tag.map, result.map); });
            return result;
        }

    private:
        auto getTags(gr::Tag::signed_index_type nSamples, TagReaderType& reader, gr::Tag::signed_index_type _currentStreamOffset) {
            std::size_t nTagsProcessed    = 0UZ;
            bool        properTagDistance = false;
            for (const Tag& tag : reader.get(reader.available())) {
                const bool tagIsWithinRange = (tag.index != -1) && tag.index < _currentStreamOffset + nSamples;
                if ((!properTagDistance && tag.index < 0) || tagIsWithinRange) { // 'index == -1' wildcard Tag index -> process unconditionally
                    nTagsProcessed++;
                    if (tagIsWithinRange) { // detected regular Tag position, ignore and stop at further wildcard Tags
                        properTagDistance = true;
                    }
                } else {
                    break; // Tag is wildcard (index == -1) after a regular or newer than the present reading position (+ offset)
                }
            }
            return reader.get(nTagsProcessed);
        }
    }; // end of ConsumablePortInputRange
    static_assert(ConsumableSpan<PortInputSpan<gr::SpanReleasePolicy::ProcessAll>>);
    static_assert(InputSpan<PortInputSpan<gr::SpanReleasePolicy::ProcessAll>>);

    template<SpanReleasePolicy spanReleasePolicy>
    using WriterSpanType = decltype(std::declval<WriterType>().template reserve<spanReleasePolicy>(1UZ));

    template<SpanReleasePolicy spanReleasePolicy, PublishableSpanReservePolicy spanReservePolicy>
    struct PublishablePortOutputRange : public WriterSpanType<spanReleasePolicy> {
        TagWriterSpanType      tags;
        Tag::signed_index_type streamIndex;
        std::size_t            tagsPublished{0UZ};

        constexpr PublishablePortOutputRange(std::size_t nSamples, WriterType& streamWriter, TagWriterType& tagsWriter, Tag::signed_index_type streamOffset) noexcept //
        requires(spanReservePolicy == PublishableSpanReservePolicy::Reserve)
            : WriterSpanType<spanReleasePolicy>(streamWriter.template reserve<spanReleasePolicy>(nSamples)), tags(tagsWriter.template reserve<SpanReleasePolicy::ProcessNone>(tagsWriter.available())), streamIndex{streamOffset} {};

        constexpr PublishablePortOutputRange(std::size_t nSamples, WriterType& streamWriter, TagWriterType& tagsWriter, Tag::signed_index_type streamOffset) noexcept //
        requires(spanReservePolicy == PublishableSpanReservePolicy::TryReserve)
            : WriterSpanType<spanReleasePolicy>(streamWriter.template tryReserve<spanReleasePolicy>(nSamples)), tags(tagsWriter.template tryReserve<SpanReleasePolicy::ProcessNone>(tagsWriter.available())), streamIndex{streamOffset} {};

        ~PublishablePortOutputRange() {
            if (WriterSpanType<spanReleasePolicy>::instanceCount() == 1UZ) { // has to be one, because the parent destructor which decrements it to zero is only called afterward
                tags.publish(tagsPublished);
            }
        }

        inline constexpr void publishTag(property_map&& tagData, Tag::signed_index_type tagOffset = -1) noexcept { processPublishTag(std::move(tagData), tagOffset); }

        inline constexpr void publishTag(const property_map& tagData, Tag::signed_index_type tagOffset = -1) noexcept { processPublishTag(tagData, tagOffset); }

    private:
        template<PropertyMapType PropertyMap>
        inline constexpr void processPublishTag(PropertyMap&& tagData, Tag::signed_index_type tagOffset) noexcept {
            const auto index = streamIndex + static_cast<gr::Tag::signed_index_type>(tagOffset);

            if (tagsPublished > 0) {
                auto& lastTag = tags[tagsPublished - 1];
#ifndef NDEBUG

                if (lastTag.index > index) { // check the order of published Tags.index
                    fmt::println(stderr, "Tag indices are not in the correct order, tagsPublished:{}, lastTag.index:{}, index:{}", tagsPublished, lastTag.index, index);
                    // std::abort();
                }
#endif
                if (lastTag.index == index) { // -> merge tags with the same index
                    for (auto&& [key, value] : tagData) {
                        lastTag.map.insert_or_assign(std::forward<decltype(key)>(key), std::forward<decltype(value)>(value));
                    }
                } else {
                    tags[tagsPublished++] = {index, std::forward<PropertyMap>(tagData)};
                }
            } else {
                tags[tagsPublished++] = {index, std::forward<PropertyMap>(tagData)};
            }
        }
    }; // end of PublishablePortOutputRange

    static_assert(PublishablePortSpan<PublishablePortOutputRange<gr::SpanReleasePolicy::ProcessAll, PublishableSpanReservePolicy::Reserve>>);

private:
    IoType    _ioHandler    = newIoHandler();
    TagIoType _tagIoHandler = newTagIoHandler();
    Tag       _cachedTag{}; // todo: for now this is only used in the output ports

    [[nodiscard]] constexpr auto newIoHandler(std::size_t buffer_size = 65536) const noexcept {
        if constexpr (kIsInput) {
            return BufferType(buffer_size).new_reader();
        } else {
            return BufferType(buffer_size).new_writer();
        }
    }

    [[nodiscard]] constexpr auto newTagIoHandler(std::size_t buffer_size = 65536) const noexcept {
        if constexpr (kIsInput) {
            return TagBufferType(buffer_size).new_reader();
        } else {
            return TagBufferType(buffer_size).new_writer();
        }
    }

public:
    constexpr Port() noexcept = default;
    Port(std::string port_name, std::int16_t priority_ = 0, std::size_t min_samples_ = 0UZ, std::size_t max_samples_ = SIZE_MAX) noexcept : name(std::move(port_name)), priority{priority_}, min_samples(min_samples_), max_samples(max_samples_), _ioHandler{newIoHandler()}, _tagIoHandler{newTagIoHandler()} { static_assert(portName.empty(), "port name must be exclusively declared via NTTP or constructor parameter"); }
    constexpr Port(Port&& other) noexcept : name(std::move(other.name)), priority{other.priority}, min_samples(other.min_samples), max_samples(other.max_samples), _ioHandler(std::move(other._ioHandler)), _tagIoHandler(std::move(other._tagIoHandler)) {}
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

    [[nodiscard]] InternalPortBuffers writerHandlerInternal() noexcept {
        static_assert(kIsOutput, "only to be used with output ports");
        return {static_cast<void*>(std::addressof(_ioHandler)), static_cast<void*>(std::addressof(_tagIoHandler))};
    }

    [[nodiscard]] bool updateReaderInternal(InternalPortBuffers buffer_writer_handler_other) noexcept {
        static_assert(kIsInput, "only to be used with input ports");

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

    [[nodiscard]] constexpr static decltype(portName) static_name() noexcept
    requires(!portName.empty())
    {
        return portName;
    }

    [[nodiscard]] std::any defaultValue() const noexcept { return default_value; }

    [[nodiscard]] bool setDefaultValue(const std::any& newDefault) {
        if (newDefault.type() == typeid(T)) {
            default_value = std::any_cast<T>(newDefault);
            return true;
        }
        return false;
    }

    [[nodiscard]] constexpr static std::size_t available() noexcept { return 0; } //  ↔ maps to Buffer::Buffer[Reader, Writer].available()

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

    template<SpanReleasePolicy spanReleasePolicy>
    PortInputSpan<spanReleasePolicy> get(std::size_t nSamples)
    requires(kIsInput)
    {
        return PortInputSpan<spanReleasePolicy>(nSamples, streamReader(), tagReader());
    }

    template<SpanReleasePolicy spanReleasePolicy>
    auto reserve(std::size_t nSamples)
    requires(kIsOutput)
    {
        return PublishablePortOutputRange<spanReleasePolicy, PublishableSpanReservePolicy::Reserve>(nSamples, streamWriter(), tagWriter(), streamWriter().position());
    }

    template<SpanReleasePolicy spanReleasePolicy>
    auto tryReserve(std::size_t nSamples)
    requires(kIsOutput)
    {
        return PublishablePortOutputRange<spanReleasePolicy, PublishableSpanReservePolicy::TryReserve>(nSamples, streamWriter(), tagWriter(), streamWriter().position());
    }

    inline constexpr void publishTag(property_map&& tag_data, Tag::signed_index_type tagOffset = -1) noexcept
    requires(kIsOutput)
    {
        processPublishTag(std::move(tag_data), tagOffset);
    }

    inline constexpr void publishTag(const property_map& tag_data, Tag::signed_index_type tagOffset = -1) noexcept
    requires(kIsOutput)
    {
        processPublishTag(tag_data, tagOffset);
    }

    [[maybe_unused]] inline constexpr bool publishPendingTags() noexcept
    requires(kIsOutput)
    {
        if (_cachedTag.map.empty() /*|| streamWriter().buffer().n_readers() == 0UZ*/) {
            return false;
        }
        {
            PublishableSpan auto outTags = tagWriter().tryReserve(1UZ);
            if (!outTags.empty()) {
                outTags[0].index = _cachedTag.index;
                outTags[0].map   = _cachedTag.map;
                outTags.publish(1UZ);
            } else {
                return false;
            }
        }

        _cachedTag.reset();
        return true;
    }

private:
    template<PropertyMapType PropertyMap>
    inline constexpr void processPublishTag(PropertyMap&& tag_data, Tag::signed_index_type tagOffset) noexcept
    requires(kIsOutput)
    {
        const auto newTagIndex = tagOffset < 0 ? tagOffset : streamWriter().position() + tagOffset;

        if (isConnected() && tagOffset >= 0 && (_cachedTag.index != newTagIndex && _cachedTag.index != -1)) { // do not cache tags that have an explicit index
            publishPendingTags();
        }
        _cachedTag.index = newTagIndex;
        if constexpr (std::is_rvalue_reference_v<PropertyMap&&>) { // -> move semantics
            for (auto& [key, value] : tag_data) {
                _cachedTag.map.insert_or_assign(std::move(key), std::move(value));
            }
        } else { // -> copy semantics
            for (const auto& [key, value] : tag_data) {
                _cachedTag.map.insert_or_assign(key, value);
            }
        }
        if (isConnected() && (tagOffset != -1L || _cachedTag.map.contains(gr::tag::END_OF_STREAM))) { // force tag publishing for explicitly published tags or EOS
            publishPendingTags();
        }
    }

    friend class DynamicPort;
};

namespace detail {
template<typename T, auto>
using just_t = T;

template<typename T, fixed_string baseName, PortType portType, PortDirection portDirection, typename... Attributes, std::size_t... Is>
consteval gr::meta::typelist<just_t<Port<T, baseName + meta::make_fixed_string<Is>(), portType, portDirection, Attributes...>, Is>...> repeated_ports_impl(std::index_sequence<Is...>) {
    return {};
}
} // namespace detail

template<std::size_t count, typename T, fixed_string baseName, PortType portType, PortDirection portDirection, typename... Attributes>
using repeated_ports = decltype(detail::repeated_ports_impl<T, baseName, portType, portDirection, Attributes...>(std::make_index_sequence<count>()));

static_assert(repeated_ports<3, float, "out", PortType::STREAM, PortDirection::OUTPUT, Optional>::at<0>::Name == fixed_string("out0"));
static_assert(repeated_ports<3, float, "out", PortType::STREAM, PortDirection::OUTPUT, Optional>::at<1>::Name == fixed_string("out1"));
static_assert(repeated_ports<3, float, "out", PortType::STREAM, PortDirection::OUTPUT, Optional>::at<2>::Name == fixed_string("out2"));

template<typename T, typename... Attributes>
using PortIn = Port<T, "", PortType::STREAM, PortDirection::INPUT, Attributes...>;
template<typename T, typename... Attributes>
using PortOut = Port<T, "", PortType::STREAM, PortDirection::OUTPUT, Attributes...>;
template<typename T, fixed_string PortName, typename... Attributes>
using PortInNamed = Port<T, PortName, PortType::STREAM, PortDirection::INPUT, Attributes...>;
template<typename T, fixed_string PortName, typename... Attributes>
using PortOutNamed = Port<T, PortName, PortType::STREAM, PortDirection::OUTPUT, Attributes...>;

using MsgPortIn  = Port<Message, "", PortType::MESSAGE, PortDirection::INPUT, DefaultMessageBuffer>;
using MsgPortOut = Port<Message, "", PortType::MESSAGE, PortDirection::OUTPUT, DefaultMessageBuffer>;
template<fixed_string PortName, typename... Attributes>
using MsgPortInNamed = Port<Message, PortName, PortType::MESSAGE, PortDirection::INPUT, DefaultMessageBuffer, Attributes...>;
template<fixed_string PortName, typename... Attributes>
using MsgPortOutNamed = Port<Message, PortName, PortType::MESSAGE, PortDirection::OUTPUT, DefaultMessageBuffer, Attributes...>;

static_assert(PortLike<PortIn<float>>);
static_assert(PortLike<decltype(PortIn<float>())>);
static_assert(PortLike<PortOut<float>>);
static_assert(PortLike<MsgPortIn>);
static_assert(PortLike<MsgPortOut>);
static_assert(PortLike<PortInNamed<float, "test">>);
static_assert(PortLike<PortOutNamed<float, "test">>);

static_assert(std::is_same_v<MsgPortIn::BufferType, gr::CircularBuffer<Message, std::dynamic_extent, gr::ProducerType::Multi>>);

static_assert(PortIn<float, RequiredSamples<1, 2>>::Required::kMinSamples == 1);
static_assert(PortIn<float, RequiredSamples<1, 2>>::Required::kMaxSamples == 2);
static_assert(std::same_as<PortIn<float, RequiredSamples<1, 2>>::Domain, CPU>);
static_assert(std::same_as<PortIn<float, RequiredSamples<1, 2>, GPU>::Domain, GPU>);

static_assert(MsgPortOutNamed<"out_msg">::static_name() == fixed_string("out_msg"));
static_assert(!(MsgPortOutNamed<"out_msg">::with_name_and_descriptor<"out_message", std::false_type>::static_name() == fixed_string("out_msg")));
static_assert(MsgPortOutNamed<"out_msg">::with_name_and_descriptor<"out_message", std::false_type>::static_name() == fixed_string("out_message"));

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
    const std::string& name;
    std::int16_t&      priority; // → dependents of a higher-prio port should be scheduled first (Q: make this by order of ports?)
    std::size_t&       min_samples;
    std::size_t&       max_samples;

private:
    struct model { // intentionally class-private definition to limit interface exposure and enhance composition
        virtual ~model() = default;

        [[nodiscard]] virtual std::any defaultValue() const noexcept = 0;

        [[nodiscard]] virtual bool setDefaultValue(const std::any& val) noexcept = 0;

        [[nodiscard]] virtual PortType type() const noexcept = 0;

        [[nodiscard]] virtual PortDirection direction() const noexcept = 0;

        [[nodiscard]] virtual std::string_view domain() const noexcept = 0;

        [[nodiscard]] virtual bool isSynchronous() noexcept = 0;

        [[nodiscard]] virtual bool isOptional() noexcept = 0;

        [[nodiscard]] virtual ConnectionResult resizeBuffer(std::size_t min_size) noexcept = 0;

        [[nodiscard]] virtual bool isConnected() const noexcept = 0;

        [[nodiscard]] virtual ConnectionResult disconnect() noexcept = 0;

        [[nodiscard]] virtual ConnectionResult connect(DynamicPort& dst_port) = 0;

        // internal runtime polymorphism access
        [[nodiscard]] virtual bool updateReaderInternal(InternalPortBuffers buffer_other) noexcept = 0;
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
                static_assert(requires { arg.writerHandlerInternal(); }, "'private void* writerHandlerInternal()' not implemented");
            } else {
                static_assert(requires { arg.updateReaderInternal(std::declval<InternalPortBuffers>()); }, "'private bool updateReaderInternal(void* buffer)' not implemented");
            }
        }

        explicit constexpr PortWrapper(T&& arg) noexcept : _value{std::move(arg)} {
            if constexpr (T::kIsInput) {
                static_assert(requires { arg.writerHandlerInternal(); }, "'private void* writerHandlerInternal()' not implemented");
            } else {
                static_assert(requires { arg.updateReaderInternal(std::declval<InternalPortBuffers>()); }, "'private bool updateReaderInternal(void* buffer)' not implemented");
            }
        }

        ~PortWrapper() override = default;

        [[nodiscard]] std::any defaultValue() const noexcept override { return _value.defaultValue(); }

        [[nodiscard]] bool setDefaultValue(const std::any& val) noexcept override { return _value.setDefaultValue(val); }

        [[nodiscard]] constexpr PortType type() const noexcept override { return _value.type(); }

        [[nodiscard]] constexpr PortDirection direction() const noexcept override { return _value.direction(); }

        [[nodiscard]] constexpr std::string_view domain() const noexcept override { return _value.domain(); }

        [[nodiscard]] bool isSynchronous() noexcept override { return _value.isSynchronous(); }

        [[nodiscard]] bool isOptional() noexcept override { return _value.isOptional(); }

        [[nodiscard]] ConnectionResult resizeBuffer(std::size_t min_size) noexcept override { return _value.resizeBuffer(min_size); }

        [[nodiscard]] bool isConnected() const noexcept override { return _value.isConnected(); }

        [[nodiscard]] ConnectionResult disconnect() noexcept override { return _value.disconnect(); }

        [[nodiscard]] ConnectionResult connect(DynamicPort& dst_port) override {
            using enum gr::ConnectionResult;
            if constexpr (T::kIsOutput) {
                auto src_buffer = _value.writerHandlerInternal();
                return dst_port.updateReaderInternal(src_buffer) ? SUCCESS : FAILED;
            } else {
                assert(false && "This works only on input ports");
                return FAILED;
            }
        }
    };

    bool updateReaderInternal(InternalPortBuffers buffer_other) noexcept { return _accessor->updateReaderInternal(buffer_other); }

public:
    using value_type = void; // a sterile port

    struct owned_value_tag {};

    struct non_owned_reference_tag {};

    constexpr DynamicPort() = delete;

    DynamicPort(const DynamicPort& arg)            = delete;
    DynamicPort& operator=(const DynamicPort& arg) = delete;

    DynamicPort(DynamicPort&& arg)            = default;
    DynamicPort& operator=(DynamicPort&& arg) = delete;

    // TODO: The lifetime of ports is a problem here, if we keep a reference to the port in DynamicPort, the port object/ can not be reallocated
    template<PortLike T>
    explicit constexpr DynamicPort(T& arg, non_owned_reference_tag) noexcept : name(arg.name), priority(arg.priority), min_samples(arg.min_samples), max_samples(arg.max_samples), _accessor{std::make_unique<PortWrapper<T, false>>(arg)} {}

    template<PortLike T>
    explicit constexpr DynamicPort(T&& arg, owned_value_tag) noexcept : name(arg.name), priority(arg.priority), min_samples(arg.min_samples), max_samples(arg.max_samples), _accessor{std::make_unique<PortWrapper<T, true>>(std::forward<T>(arg))} {}

    [[nodiscard]] std::any defaultValue() const noexcept { return _accessor->defaultValue(); }

    [[nodiscard]] bool setDefaultValue(const std::any& val) noexcept { return _accessor->setDefaultValue(val); }

    [[nodiscard]] PortType type() const noexcept { return _accessor->type(); }

    [[nodiscard]] PortDirection direction() const noexcept { return _accessor->direction(); }

    [[nodiscard]] std::string_view domain() const noexcept { return _accessor->domain(); }

    [[nodiscard]] bool isSynchronous() noexcept { return _accessor->isSynchronous(); }

    [[nodiscard]] bool isOptional() noexcept { return _accessor->isOptional(); }

    [[nodiscard]] ConnectionResult resizeBuffer(std::size_t min_size) {
        if (direction() == PortDirection::OUTPUT) {
            return _accessor->resizeBuffer(min_size);
        }
        return ConnectionResult::FAILED;
    }

    [[nodiscard]] bool isConnected() const noexcept { return _accessor->isConnected(); }

    [[nodiscard]] ConnectionResult disconnect() noexcept { return _accessor->disconnect(); }

    [[nodiscard]] ConnectionResult connect(DynamicPort& dst_port) { return _accessor->connect(dst_port); }
};

static_assert(PortLike<DynamicPort>);

namespace detail {
template<typename T>
concept TagPredicate = requires(const T& t, const Tag& tag, Tag::signed_index_type readPosition) {
    { t(tag, readPosition) } -> std::convertible_to<bool>;
};
inline constexpr TagPredicate auto defaultTagMatcher    = [](const Tag& tag, Tag::signed_index_type readPosition) noexcept { return tag.index >= readPosition; };
inline constexpr TagPredicate auto defaultEOSTagMatcher = [](const Tag& tag, Tag::signed_index_type readPosition) noexcept {
    auto eosTagIter = tag.map.find(gr::tag::END_OF_STREAM);
    if (eosTagIter != tag.map.end() && eosTagIter->second == true) {
        if (tag.index >= readPosition || tag.index < 0) {
            return true;
        }
    }
    return false;
};
} // namespace detail

inline constexpr std::optional<std::size_t> nSamplesToNextTagConditional(const PortLike auto& port, detail::TagPredicate auto& predicate, Tag::signed_index_type readOffset) {
    const gr::ConsumableSpan auto tagData = port.tagReader().get();
    if (!port.isConnected() || tagData.empty()) [[likely]] {
        return std::nullopt; // default: no tags in sight
    }
    const Tag::signed_index_type readPosition = port.streamReader().position();

    // at least one tag is present -> if tag is not on the first tag position read up to the tag position, or if the tag has a special 'index = -1'
    const auto firstMatchingTag = std::ranges::find_if(tagData, [&](const auto& tag) { return predicate(tag, readPosition + readOffset); });
    std::ignore                 = tagData.consume(0UZ);
    if (firstMatchingTag != tagData.end()) {
        return static_cast<std::size_t>(std::max(firstMatchingTag->index - readPosition, Tag::signed_index_type(0))); // Tags in the past will have a negative distance -> deliberately map them to '0'
    } else {
        return std::nullopt;
    }
}

inline constexpr std::optional<std::size_t> nSamplesUntilNextTag(const PortLike auto& port, Tag::signed_index_type offset = 0) { return nSamplesToNextTagConditional(port, detail::defaultTagMatcher, offset); }

inline constexpr std::optional<std::size_t> samples_to_eos_tag(const PortLike auto& port, Tag::signed_index_type offset = 0) { return nSamplesToNextTagConditional(port, detail::defaultEOSTagMatcher, offset); }

} // namespace gr
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, gr::fixed_string portName, gr::PortType portType, gr::PortDirection portDirection, typename... Attributes), (gr::Port<T, portName, portType, portDirection, Attributes...>), kDirection, kPortType, kIsInput, kIsOutput, kIsSynch, kIsOptional, name, priority, min_samples, max_samples, metaInfo)

#endif // GNURADIO_PORT_HPP
