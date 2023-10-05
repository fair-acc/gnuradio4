#ifndef GNURADIO_PORT_HPP
#define GNURADIO_PORT_HPP

#include <complex>
#include <span>
#include <variant>

#include "dataset.hpp"
#include "node.hpp"
#include "annotated.hpp"
#include "circular_buffer.hpp"
#include "tag.hpp"
#include "utils.hpp"

namespace fair::graph {

using fair::meta::fixed_string;
using namespace fair::literals;

#ifndef PMT_SUPPORTED_TYPE // // #### default supported types -- TODO: to be replaced by pmt::pmtv declaration
#define PMT_SUPPORTED_TYPE
// Only DataSet<double> and DataSet<float> are added => consider to support more Dataset<T>
using supported_type = std::variant<uint8_t, uint32_t, int8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double>, DataSet<float>, DataSet<double> /*, ...*/>;
#endif

enum class port_direction_t { INPUT, OUTPUT, ANY }; // 'ANY' only for query and not to be used for port declarations

enum class connection_result_t { SUCCESS, FAILED };

enum class port_type_t {
    STREAM, /*!< used for single-producer-only ond usually synchronous one-to-one or one-to-many communications */
    MESSAGE /*!< used for multiple-producer one-to-one, one-to-many, many-to-one, or many-to-many communications */
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
concept PortDomainType = requires { T::Name; } && std::is_base_of_v<PortDomain<T::Name>, T>;

template<typename T>
using is_port_domain = std::bool_constant<PortDomainType<T>>;

struct CPU : public PortDomain<"CPU"> {};

struct GPU : public PortDomain<"GPU"> {};

static_assert(is_port_domain<CPU>::value);
static_assert(is_port_domain<GPU>::value);
static_assert(!is_port_domain<int>::value);

template<class T>
concept PortType = requires(T t, const std::size_t n_items, const supported_type &newDefault) { // dynamic definitions
    typename T::value_type;
    { t.defaultValue() } -> std::same_as<supported_type>;
    { t.setDefaultValue(newDefault) } -> std::same_as<bool>;
    { t.name } -> std::convertible_to<std::string_view>;
    { t.priority } -> std::convertible_to<std::int32_t>;
    { t.min_samples } -> std::convertible_to<std::size_t>;
    { t.max_samples } -> std::convertible_to<std::size_t>;
    { t.type() } -> std::same_as<port_type_t>;
    { t.direction() } -> std::same_as<port_direction_t>;
    { t.domain() } -> std::same_as<std::string_view>;
    { t.resize_buffer(n_items) } -> std::same_as<connection_result_t>;
    { t.disconnect() } -> std::same_as<connection_result_t>;
    { t.isSynchronous() } -> std::same_as<bool>;
    { t.isOptional() } -> std::same_as<bool>;
};

/**
 * @brief internal port buffer handler
 *
 * N.B. void* needed for type-erasure/Python compatibility/wrapping
 */
struct internal_port_buffers {
    void *streamHandler;
    void *tagHandler;
};

/**
 * @brief optional port annotation argument to describe the min/max number of samples required from this port before invoking the blocks work function.
 *
 * @tparam MIN_SAMPLES (>0) specifies the minimum number of samples the port/block requires for processing in one scheduler iteration
 * @tparam MAX_SAMPLES specifies the maximum number of samples the port/block can process in one scheduler iteration
 */
template<std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent>
struct RequiredSamples {
    static_assert(MIN_SAMPLES > 0, "Port<T, ..., RequiredSamples::MIN_SAMPLES, ...>, ..> must be >= 0");
    static constexpr std::size_t MinSamples = MIN_SAMPLES;
    static constexpr std::size_t MaxSamples = MAX_SAMPLES;
};

template<typename T>
concept IsRequiredSamples = requires {
    T::MinSamples;
    T::MaxSamples;
} && std::is_base_of_v<RequiredSamples<T::MinSamples, T::MaxSamples>, T>;

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
template<gr::Buffer BufferType>
struct StreamBufferType {
    using type = BufferType;
};

/**
 * @brief optional port annotation argument to define the buffer implementation to be used for tag data
 *
 * @tparam BufferType user-extendable buffer implementation for the tag data
 */
template<gr::Buffer BufferType>
struct TagBufferType {
    using type = BufferType;
};

template<typename T>
concept IsStreamBufferAttribute = requires { typename T::type; } && gr::Buffer<typename T::type> && std::is_base_of_v<StreamBufferType<typename T::type>, T>;
;

template<typename T>
concept IsTagBufferAttribute = requires { typename T::type; } && gr::Buffer<typename T::type> && std::is_base_of_v<TagBufferType<typename T::type>, T>;

template<typename T>
using is_stream_buffer_attribute = std::bool_constant<IsStreamBufferAttribute<T>>;

template<typename T>
using is_tag_buffer_attribute = std::bool_constant<IsTagBufferAttribute<T>>;

template<typename T>
struct DefaultStreamBuffer : StreamBufferType<gr::circular_buffer<T>> {};

struct DefaultTagBuffer : TagBufferType<gr::circular_buffer<tag_t>> {};

static_assert(is_stream_buffer_attribute<DefaultStreamBuffer<int>>::value);
static_assert(!is_stream_buffer_attribute<DefaultTagBuffer>::value);
static_assert(!is_tag_buffer_attribute<DefaultStreamBuffer<int>>::value);
static_assert(is_tag_buffer_attribute<DefaultTagBuffer>::value);

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
 * @brief 'ports' are interfaces that allows data to flow between blocks in a graph, similar to RF connectors.
 * Each block can have zero or more input/output ports. When connecting ports, either a single-step or a two-step
 * connection method can be used. Ports belong to a computing domain, such as CPU, GPU, or FPGA, and transitions
 * between domains require explicit data conversion.
 * Each port consists of a synchronous performance-optimised streaming and asynchronous tag communication component:
 *                                                                                      ┌───────────────────────
 *         ───────────────────┐                                       ┌─────────────────┤  <node/block definition>
 *             output-port    │                                       │    input-port   │  ...
 *          stream-buffer<T>  │>───────┬─────────────────┬───────────>│                 │
 *          tag-buffer<tag_t> │      tag#0             tag#1          │                 │
 *                            │                                       │                 │
 *         ───────────────────┘                                       └─────────────────┤
 *
 * Tags contain the index ID of the sending/receiving stream sample <T> they are attached to. Node implementations
 * may choose to chunk the data based on the MIN_SAMPLES/MAX_SAMPLES criteria only, or in addition break-up the stream
 * so that there is only one tag per scheduler iteration. Multiple tags on the same sample shall be merged to one.
 *
 * @tparam T the data type of the port. It can be any copyable preferably cache-aligned (i.e. 64 byte-sized) type.
 * @tparam PortName a string to identify the port, notably to be used in an UI- and hand-written explicit code context.
 * @tparam PortType STREAM  or MESSAGE
 * @tparam PortDirection either input or output
 * @tparam Arguments optional: default to 'DefaultStreamBuffer' and DefaultTagBuffer' based on 'gr::circular_buffer', and CPU domain
 */
template<typename T, fixed_string PortName, port_type_t PortType, port_direction_t PortDirection, typename... Arguments>
struct Port {
    template<fixed_string NewName>
    using with_name = Port<T, NewName, PortType, PortDirection, Arguments...>;

    static_assert(PortDirection != port_direction_t::ANY, "ANY reserved for queries and not port direction declarations");

    using value_type                            = T;
    using ArgumentsTypeList                     = typename fair::meta::typelist<Arguments...>;
    using Domain                                = ArgumentsTypeList::template find_or_default<is_port_domain, CPU>;
    using Required                              = ArgumentsTypeList::template find_or_default<is_required_samples, RequiredSamples<std::dynamic_extent, std::dynamic_extent>>;
    using BufferType                            = ArgumentsTypeList::template find_or_default<is_stream_buffer_attribute, DefaultStreamBuffer<T>>::type;
    using TagBufferType                         = ArgumentsTypeList::template find_or_default<is_tag_buffer_attribute, DefaultTagBuffer>::type;
    static constexpr port_direction_t Direction = PortDirection;
    static constexpr bool             IS_INPUT  = PortDirection == port_direction_t::INPUT;
    static constexpr bool             IS_OUTPUT = PortDirection == port_direction_t::OUTPUT;
    static constexpr fixed_string     Name      = PortName;

    using ReaderType                            = decltype(std::declval<BufferType>().new_reader());
    using WriterType                            = decltype(std::declval<BufferType>().new_writer());
    using IoType                                = std::conditional_t<IS_INPUT, ReaderType, WriterType>;
    using TagReaderType                         = decltype(std::declval<TagBufferType>().new_reader());
    using TagWriterType                         = decltype(std::declval<TagBufferType>().new_writer());
    using TagIoType                             = std::conditional_t<IS_INPUT, TagReaderType, TagWriterType>;

    // public properties
    constexpr static bool synchronous   = !std::disjunction_v<std::is_same<Async, Arguments>...>;
    constexpr static bool optional      = std::disjunction_v<std::is_same<Optional, Arguments>...>;
    std::string           name          = static_cast<std::string>(PortName);
    std::int16_t          priority      = 0; // → dependents of a higher-prio port should be scheduled first (Q: make this by order of ports?)
    std::size_t           min_samples   = (Required::MinSamples == std::dynamic_extent ? 1 : Required::MinSamples);
    std::size_t           max_samples   = Required::MaxSamples;
    T                     default_value = T{};

private:
    bool      _connected    = false;
    IoType    _ioHandler    = new_io_handler();
    TagIoType _tagIoHandler = new_tag_io_handler();

public:
    [[nodiscard]] constexpr bool
    initBuffer(std::size_t nSamples = 0) noexcept {
        if constexpr (IS_OUTPUT) {
            // write one default value into output -- needed for cyclic graph initialisation
            return _ioHandler.try_publish([val = default_value](std::span<T> &out) { std::ranges::fill(out, val); }, nSamples);
        }
        return true;
    }

    [[nodiscard]] constexpr auto
    new_io_handler(std::size_t buffer_size = 65536) const noexcept {
        if constexpr (IS_INPUT) {
            return BufferType(buffer_size).new_reader();
        } else {
            return BufferType(buffer_size).new_writer();
        }
    }

    [[nodiscard]] constexpr auto
    new_tag_io_handler(std::size_t buffer_size = 65536) const noexcept {
        if constexpr (IS_INPUT) {
            return TagBufferType(buffer_size).new_reader();
        } else {
            return TagBufferType(buffer_size).new_writer();
        }
    }

    [[nodiscard]] internal_port_buffers
    writer_handler_internal() noexcept {
        static_assert(IS_OUTPUT, "only to be used with output ports");
        return { static_cast<void *>(std::addressof(_ioHandler)), static_cast<void *>(std::addressof(_tagIoHandler)) };
    }

    [[nodiscard]] bool
    update_reader_internal(internal_port_buffers buffer_writer_handler_other) noexcept {
        static_assert(IS_INPUT, "only to be used with input ports");

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
        auto typed_buffer_writer     = static_cast<WriterType *>(buffer_writer_handler_other.streamHandler);
        auto typed_tag_buffer_writer = static_cast<TagWriterType *>(buffer_writer_handler_other.tagHandler);
        setBuffer(typed_buffer_writer->buffer(), typed_tag_buffer_writer->buffer());
        return true;
    }

    constexpr Port()   = default;
    Port(const Port &) = delete;
    auto
    operator=(const Port &)
            = delete;

    Port(std::string port_name, std::int16_t priority_ = 0, std::size_t min_samples_ = 0_UZ, std::size_t max_samples_ = SIZE_MAX) noexcept
        : name(std::move(port_name)), priority{ priority_ }, min_samples(min_samples_), max_samples(max_samples_) {
        static_assert(PortName.empty(), "port name must be exclusively declared via NTTP or constructor parameter");
    }

    constexpr Port(Port &&other) noexcept : name(std::move(other.name)), priority{ other.priority }, min_samples(other.min_samples), max_samples(other.max_samples) {}

    constexpr Port &
    operator=(Port &&other) noexcept {
        Port tmp(std::move(other));
        std::swap(name, tmp._name);
        std::swap(min_samples, tmp._min_samples);
        std::swap(max_samples, tmp._max_samples);
        std::swap(priority, tmp._priority);

        std::swap(_connected, tmp._connected);
        std::swap(_ioHandler, tmp._ioHandler);
        std::swap(_tagIoHandler, tmp._tagIoHandler);
        return *this;
    }

    ~Port() = default;

    [[nodiscard]] constexpr static port_type_t
    type() noexcept {
        return PortType;
    }

    [[nodiscard]] constexpr static port_direction_t
    direction() noexcept {
        return PortDirection;
    }

    [[nodiscard]] constexpr static std::string_view
    domain() noexcept {
        return std::string_view(Domain::Name);
    }

    [[nodiscard]] constexpr static bool
    isSynchronous() noexcept {
        return synchronous;
    }

    [[nodiscard]] constexpr static bool
    isOptional() noexcept {
        return optional;
    }

    [[nodiscard]] constexpr static decltype(PortName)
    static_name() noexcept
        requires(!PortName.empty())
    {
        return PortName;
    }

    // TODO revisit: constexpr was removed because emscripten does not support constexpr function for non literal type, like DataSet<T>
#if defined(__EMSCRIPTEN__)
    [[nodiscard]] supported_type
#else
    [[nodiscard]] constexpr supported_type
#endif
    defaultValue() const noexcept {
        return default_value;
    }

    bool
    setDefaultValue(const supported_type &newDefault) noexcept {
        if (std::holds_alternative<T>(newDefault)) {
            default_value = std::get<T>(newDefault);
            return true;
        }
        return false;
    }

    [[nodiscard]] constexpr static std::size_t
    available() noexcept {
        return 0;
    } //  ↔ maps to Buffer::Buffer[Reader, Writer].available()

    [[nodiscard]] constexpr std::size_t
    min_buffer_size() const noexcept {
        if constexpr (Required::MinSamples == std::dynamic_extent) {
            return min_samples;
        } else {
            return Required::MinSamples;
        }
    }

    [[nodiscard]] constexpr std::size_t
    max_buffer_size() const noexcept {
        if constexpr (Required::MaxSamples == std::dynamic_extent) {
            return max_samples;
        } else {
            return Required::MaxSamples;
        }
    }

    [[nodiscard]] constexpr connection_result_t
    resize_buffer(std::size_t min_size) noexcept {
        using enum fair::graph::connection_result_t;
        if constexpr (IS_INPUT) {
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

    [[nodiscard]] auto
    buffer() {
        struct port_buffers {
            BufferType    streamBuffer;
            TagBufferType tagBufferType;
        };

        return port_buffers{ _ioHandler.buffer(), _tagIoHandler.buffer() };
    }

    void
    setBuffer(gr::Buffer auto streamBuffer, gr::Buffer auto tagBuffer) noexcept {
        if constexpr (IS_INPUT) {
            _ioHandler    = streamBuffer.new_reader();
            _tagIoHandler = tagBuffer.new_reader();
            _connected    = true;
        } else {
            _ioHandler    = streamBuffer.new_writer();
            _tagIoHandler = tagBuffer.new_reader();
        }
    }

    [[nodiscard]] constexpr const ReaderType &
    streamReader() const noexcept {
        static_assert(!IS_OUTPUT, "streamReader() not applicable for outputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] constexpr ReaderType &
    streamReader() noexcept {
        static_assert(!IS_OUTPUT, "streamReader() not applicable for outputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] constexpr const WriterType &
    streamWriter() const noexcept {
        static_assert(!IS_INPUT, "streamWriter() not applicable for inputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] constexpr WriterType &
    streamWriter() noexcept {
        static_assert(!IS_INPUT, "streamWriter() not applicable for inputs (yet)");
        return _ioHandler;
    }

    [[nodiscard]] constexpr const TagReaderType &
    tagReader() const noexcept {
        static_assert(!IS_OUTPUT, "tagReader() not applicable for outputs (yet)");
        return _tagIoHandler;
    }

    [[nodiscard]] constexpr TagReaderType &
    tagReader() noexcept {
        static_assert(!IS_OUTPUT, "tagReader() not applicable for outputs (yet)");
        return _tagIoHandler;
    }

    [[nodiscard]] constexpr const TagWriterType &
    tagWriter() const noexcept {
        static_assert(!IS_INPUT, "tagWriter() not applicable for inputs (yet)");
        return _tagIoHandler;
    }

    [[nodiscard]] constexpr TagWriterType &
    tagWriter() noexcept {
        static_assert(!IS_INPUT, "tagWriter() not applicable for inputs (yet)");
        return _tagIoHandler;
    }

    [[nodiscard]] connection_result_t
    disconnect() noexcept {
        if (_connected == false) {
            return connection_result_t::FAILED;
        }
        _ioHandler    = new_io_handler();
        _tagIoHandler = new_tag_io_handler();
        _connected    = false;
        return connection_result_t::SUCCESS;
    }

    template<typename Other>
    [[nodiscard]] connection_result_t
    connect(Other &&other) {
        static_assert(IS_OUTPUT && std::remove_cvref_t<Other>::IS_INPUT);
        auto src_buffer = writer_handler_internal();
        return std::forward<Other>(other).update_reader_internal(src_buffer) ? connection_result_t::SUCCESS : connection_result_t::FAILED;
    }

    friend class dynamic_port;
};

namespace detail {
template<typename T, auto>
using just_t = T;

template<typename T, fixed_string BaseName, port_type_t PortType, port_direction_t PortDirection, typename... Arguments, std::size_t... Is>
consteval fair::meta::typelist<just_t<Port<T, BaseName + meta::make_fixed_string<Is>(), PortType, PortDirection, Arguments...>, Is>...>
repeated_ports_impl(std::index_sequence<Is...>) {
    return {};
}
} // namespace detail

template<std::size_t Count, typename T, fixed_string BaseName, port_type_t PortType, port_direction_t PortDirection, typename... Arguments>
using repeated_ports = decltype(detail::repeated_ports_impl<T, BaseName, PortType, PortDirection, Arguments...>(std::make_index_sequence<Count>()));

static_assert(repeated_ports<3, float, "out", port_type_t::STREAM, port_direction_t::OUTPUT, Optional>::at<0>::Name == fixed_string("out0"));
static_assert(repeated_ports<3, float, "out", port_type_t::STREAM, port_direction_t::OUTPUT, Optional>::at<1>::Name == fixed_string("out1"));
static_assert(repeated_ports<3, float, "out", port_type_t::STREAM, port_direction_t::OUTPUT, Optional>::at<2>::Name == fixed_string("out2"));

template<typename T, typename... Arguments>
using PortIn = Port<T, "", port_type_t::STREAM, port_direction_t::INPUT, Arguments...>;
template<typename T, typename... Arguments>
using PortOut = Port<T, "", port_type_t::STREAM, port_direction_t::OUTPUT, Arguments...>;
template<typename... Arguments>
using MsgPortIn = Port<property_map, "", port_type_t::MESSAGE, port_direction_t::INPUT, Arguments...>;
template<typename... Arguments>
using MsgPortOut = Port<property_map, "", port_type_t::MESSAGE, port_direction_t::OUTPUT, Arguments...>;

template<typename T, fixed_string PortName, typename... Arguments>
using PortInNamed = Port<T, PortName, port_type_t::STREAM, port_direction_t::INPUT, Arguments...>;
template<typename T, fixed_string PortName, typename... Arguments>
using PortOutNamed = Port<T, PortName, port_type_t::STREAM, port_direction_t::OUTPUT, Arguments...>;
template<fixed_string PortName, typename... Arguments>
using MsgPortInNamed = Port<property_map, PortName, port_type_t::STREAM, port_direction_t::INPUT, Arguments...>;
template<fixed_string PortName, typename... Arguments>
using MsgPortOutNamed = Port<property_map, PortName, port_type_t::STREAM, port_direction_t::OUTPUT, Arguments...>;

static_assert(PortType<PortIn<float>>);
static_assert(PortType<decltype(PortIn<float>())>);
static_assert(PortType<PortOut<float>>);
static_assert(PortType<MsgPortIn<float>>);
static_assert(PortType<MsgPortOut<float>>);

static_assert(PortIn<float, RequiredSamples<1, 2>>::Required::MinSamples == 1);
static_assert(PortIn<float, RequiredSamples<1, 2>>::Required::MaxSamples == 2);
static_assert(std::same_as<PortIn<float, RequiredSamples<1, 2>>::Domain, CPU>);
static_assert(std::same_as<PortIn<float, RequiredSamples<1, 2>, GPU>::Domain, GPU>);

static_assert(MsgPortOutNamed<"out_msg">::static_name() == fixed_string("out_msg"));
static_assert(!(MsgPortOutNamed<"out_msg">::with_name<"out_message">::static_name() == fixed_string("out_msg")));
static_assert(MsgPortOutNamed<"out_msg">::with_name<"out_message">::static_name() == fixed_string("out_message"));

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
class dynamic_port {
public:
    const std::string &name;
    std::int16_t      &priority; // → dependents of a higher-prio port should be scheduled first (Q: make this by order of ports?)
    std::size_t       &min_samples;
    std::size_t       &max_samples;

private:
    struct model { // intentionally class-private definition to limit interface exposure and enhance composition
        virtual ~model() = default;

        [[nodiscard]] virtual supported_type
        defaultValue() const noexcept
                = 0;

        [[nodiscard]] virtual bool
        setDefaultValue(const supported_type &val) noexcept
                = 0;

        [[nodiscard]] virtual port_type_t
        type() const noexcept
                = 0;

        [[nodiscard]] virtual port_direction_t
        direction() const noexcept
                = 0;

        [[nodiscard]] virtual std::string_view
        domain() const noexcept
                = 0;

        [[nodiscard]] virtual bool
        isSynchronous() noexcept
                = 0;

        [[nodiscard]] virtual bool
        isOptional() noexcept
                = 0;

        [[nodiscard]] virtual connection_result_t
        resize_buffer(std::size_t min_size) noexcept
                = 0;

        [[nodiscard]] virtual connection_result_t
        disconnect() noexcept
                = 0;

        [[nodiscard]] virtual connection_result_t
        connect(dynamic_port &dst_port)
                = 0;

        // internal runtime polymorphism access
        [[nodiscard]] virtual bool
        update_reader_internal(internal_port_buffers buffer_other) noexcept
                = 0;
    };

    std::unique_ptr<model> _accessor;

    template<PortType T, bool owning>
    class wrapper final : public model {
        using PortType = std::decay_t<T>;
        std::conditional_t<owning, PortType, PortType &> _value;

        [[nodiscard]] internal_port_buffers
        writer_handler_internal() noexcept {
            return _value.writer_handler_internal();
        };

        [[nodiscard]] bool
        update_reader_internal(internal_port_buffers buffer_other) noexcept override {
            if constexpr (T::IS_INPUT) {
                return _value.update_reader_internal(buffer_other);
            } else {
                assert(false && "This works only on input ports");
                return false;
            }
        }

    public:
        wrapper()                = delete;

        wrapper(const wrapper &) = delete;

        auto &
        operator=(const wrapper &)
                = delete;

        auto &
        operator=(wrapper &&)
                = delete;

        explicit constexpr wrapper(T &arg) noexcept : _value{ arg } {
            if constexpr (T::IS_INPUT) {
                static_assert(
                        requires { arg.writer_handler_internal(); }, "'private void* writer_handler_internal()' not implemented");
            } else {
                static_assert(
                        requires { arg.update_reader_internal(std::declval<internal_port_buffers>()); }, "'private bool update_reader_internal(void* buffer)' not implemented");
            }
        }

        explicit constexpr wrapper(T &&arg) noexcept : _value{ std::move(arg) } {
            if constexpr (T::IS_INPUT) {
                static_assert(
                        requires { arg.writer_handler_internal(); }, "'private void* writer_handler_internal()' not implemented");
            } else {
                static_assert(
                        requires { arg.update_reader_internal(std::declval<internal_port_buffers>()); }, "'private bool update_reader_internal(void* buffer)' not implemented");
            }
        }

        ~wrapper() override = default;

        // TODO revisit: constexpr was removed because emscripten does not support constexpr function for non literal type, like DataSet<T>
#if defined(__EMSCRIPTEN__)
        [[nodiscard]] supported_type
#else
        [[nodiscard]] constexpr supported_type
#endif
        defaultValue() const noexcept override {
            return _value.defaultValue();
        }

        [[nodiscard]] bool
        setDefaultValue(const supported_type &val) noexcept override {
            return _value.setDefaultValue(val);
        }

        [[nodiscard]] constexpr port_type_t
        type() const noexcept override {
            return _value.type();
        }

        [[nodiscard]] constexpr port_direction_t
        direction() const noexcept override {
            return _value.direction();
        }

        [[nodiscard]] constexpr std::string_view
        domain() const noexcept override {
            return _value.domain();
        }

        [[nodiscard]] bool
        isSynchronous() noexcept override {
            return _value.isSynchronous();
        }

        [[nodiscard]] bool
        isOptional() noexcept override {
            return _value.isOptional();
        }

        [[nodiscard]] connection_result_t
        resize_buffer(std::size_t min_size) noexcept override {
            return _value.resize_buffer(min_size);
        }

        [[nodiscard]] connection_result_t
        disconnect() noexcept override {
            return _value.disconnect();
        }

        [[nodiscard]] connection_result_t
        connect(dynamic_port &dst_port) override {
            using enum fair::graph::connection_result_t;
            if constexpr (T::IS_OUTPUT) {
                auto src_buffer = _value.writer_handler_internal();
                return dst_port.update_reader_internal(src_buffer) ? SUCCESS : FAILED;
            } else {
                assert(false && "This works only on input ports");
                return FAILED;
            }
        }
    };

    bool
    update_reader_internal(internal_port_buffers buffer_other) noexcept {
        return _accessor->update_reader_internal(buffer_other);
    }

public:
    using value_type                      = void; // a sterile port

    struct owned_value_tag {};
    struct non_owned_reference_tag {};

    constexpr dynamic_port()              = delete;

    dynamic_port(const dynamic_port &arg) = delete;
    dynamic_port &
    operator=(const dynamic_port &arg)
            = delete;

    dynamic_port(dynamic_port &&arg) = default;
    dynamic_port &
    operator=(dynamic_port &&arg)
            = delete;

    // TODO: The lifetime of ports is a problem here, if we keep
    // a reference to the port in dynamic_port, the port object
    // can not be reallocated
    template<PortType T>
    explicit constexpr dynamic_port(T &arg, non_owned_reference_tag) noexcept
        : name(arg.name), priority(arg.priority), min_samples(arg.min_samples), max_samples(arg.max_samples), _accessor{ std::make_unique<wrapper<T, false>>(arg) } {}

    template<PortType T>
    explicit constexpr dynamic_port(T &&arg, owned_value_tag) noexcept
        : name(arg.name), priority(arg.priority), min_samples(arg.min_samples), max_samples(arg.max_samples), _accessor{ std::make_unique<wrapper<T, true>>(std::forward<T>(arg)) } {}

    [[nodiscard]] supported_type
    defaultValue() const noexcept {
        return _accessor->defaultValue();
    }

    [[nodiscard]] bool
    setDefaultValue(const supported_type &val) noexcept {
        return _accessor->setDefaultValue(val);
    }

    [[nodiscard]] port_type_t
    type() const noexcept {
        return _accessor->type();
    }

    [[nodiscard]] port_direction_t
    direction() const noexcept {
        return _accessor->direction();
    }

    [[nodiscard]] std::string_view
    domain() const noexcept {
        return _accessor->domain();
    }

    [[nodiscard]] bool
    isSynchronous() noexcept {
        return _accessor->isSynchronous();
    }

    [[nodiscard]] bool
    isOptional() noexcept {
        return _accessor->isOptional();
    }

    [[nodiscard]] connection_result_t
    resize_buffer(std::size_t min_size) {
        if (direction() == port_direction_t::OUTPUT) {
            return _accessor->resize_buffer(min_size);
        }
        return connection_result_t::FAILED;
    }

    [[nodiscard]] connection_result_t
    disconnect() noexcept {
        return _accessor->disconnect();
    }

    [[nodiscard]] connection_result_t
    connect(dynamic_port &dst_port) {
        return _accessor->connect(dst_port);
    }
};

static_assert(PortType<dynamic_port>);

constexpr void
publish_tag(PortType auto &port, property_map &&tag_data, std::size_t tag_offset = 0) noexcept {
    port.tagWriter().publish(
            [&port, data = std::move(tag_data), &tag_offset](std::span<fair::graph::tag_t> tag_output) {
                tag_output[0].index = port.streamWriter().position() + std::make_signed_t<std::size_t>(tag_offset);
                tag_output[0].map   = std::move(data);
            },
            1_UZ);
}

constexpr void
publish_tag(PortType auto &port, const property_map &tag_data, std::size_t tag_offset = 0) noexcept {
    port.tagWriter().publish(
            [&port, &tag_data, &tag_offset](std::span<fair::graph::tag_t> tag_output) {
                tag_output[0].index = port.streamWriter().position() + tag_offset;
                tag_output[0].map   = tag_data;
            },
            1_UZ);
}

constexpr std::size_t
samples_to_next_tag(const PortType auto &port) {
    if (port.tagReader().available() == 0) [[likely]] {
        return std::numeric_limits<std::size_t>::max(); // default: no tags in sight
    }

    // at least one tag is present -> if tag is not on the first tag position read up to the tag position
    const auto &tagData           = port.tagReader().get();
    const auto &readPosition      = port.streamReader().position();
    const auto  future_tags_begin = std::ranges::find_if(tagData, [&readPosition](const auto &tag) noexcept { return tag.index > readPosition + 1; });

    if (future_tags_begin == tagData.begin()) {
        const auto        first_future_tag_index   = static_cast<std::size_t>(future_tags_begin->index);
        const std::size_t n_samples_until_next_tag = readPosition == -1 ? first_future_tag_index : (first_future_tag_index - static_cast<std::size_t>(readPosition) - 1_UZ);
        return n_samples_until_next_tag;
    } else {
        return 0;
    }
}

} // namespace fair::graph

#endif // include guard
