#ifndef GNURADIO_PORT_HPP
#define GNURADIO_PORT_HPP

#include <complex>
#include <span>
#include <variant>

#include "circular_buffer.hpp"
#include "tag.hpp"
#include "utils.hpp"

namespace fair::graph {

using fair::meta::fixed_string;
using namespace fair::literals;

// #### default supported types -- TODO: to be replaced by pmt::pmtv declaration
using supported_type = std::variant<uint8_t, uint32_t, int8_t, int16_t, int32_t, float, double, std::complex<float>, std::complex<double> /*, ...*/>;

enum class port_direction_t { INPUT, OUTPUT, ANY }; // 'ANY' only for query and not to be used for port declarations
enum class connection_result_t { SUCCESS, FAILED };
enum class port_type_t {
    STREAM, /*!< used for single-producer-only ond usually synchronous one-to-one or one-to-many communications */
    MESSAGE /*!< used for multiple-producer one-to-one, one-to-many, many-to-one, or many-to-many communications */
};
enum class port_domain_t { CPU, GPU, NET, FPGA, DSP, MLU };

template<class T>
concept Port = requires(T t, const std::size_t n_items) { // dynamic definitions
    typename T::value_type;
    { t.pmt_type() } -> std::same_as<supported_type>;
    { t.name } -> std::convertible_to<std::string_view>;
    { t.priority } -> std::convertible_to<std::int32_t>;
    { t.min_samples } -> std::convertible_to<std::size_t>;
    { t.max_samples } -> std::convertible_to<std::size_t>;
    { t.type() } -> std::same_as<port_type_t>;
    { t.direction() } -> std::same_as<port_direction_t>;
    { t.resize_buffer(n_items) } -> std::same_as<connection_result_t>;
    { t.disconnect() } -> std::same_as<connection_result_t>;
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
 * @tparam MIN_SAMPLES specifies the minimum number of samples the port/block requires for processing in one scheduler iteration
 * @tparam MAX_SAMPLES specifies the maximum number of samples the port/block can process in one scheduler iteration
 * @tparam BufferType user-extendable buffer implementation for the streaming data
 * @tparam TagBufferType user-extendable buffer implementation for the tag data
 */
template<typename T, fixed_string PortName, port_type_t PortType, port_direction_t PortDirection, // TODO: sort default arguments
         std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent, gr::Buffer BufferType = gr::circular_buffer<T>,
         gr::Buffer TagBufferType = gr::circular_buffer<tag_t>>
class port {
public:
    static_assert(PortDirection != port_direction_t::ANY, "ANY reserved for queries and not port direction declarations");

    using value_type                = T;

    static constexpr bool IS_INPUT  = PortDirection == port_direction_t::INPUT;
    static constexpr bool IS_OUTPUT = PortDirection == port_direction_t::OUTPUT;

    template<fixed_string NewName>
    using with_name     = port<T, NewName, PortType, PortDirection, MIN_SAMPLES, MAX_SAMPLES, BufferType>;

    using ReaderType    = decltype(std::declval<BufferType>().new_reader());
    using WriterType    = decltype(std::declval<BufferType>().new_writer());
    using IoType        = std::conditional_t<IS_INPUT, ReaderType, WriterType>;
    using TagReaderType = decltype(std::declval<TagBufferType>().new_reader());
    using TagWriterType = decltype(std::declval<TagBufferType>().new_writer());
    using TagIoType     = std::conditional_t<IS_INPUT, TagReaderType, TagWriterType>;

    // public properties
    const std::string name        = static_cast<std::string>(PortName);
    std::int16_t      priority    = 0; // → dependents of a higher-prio port should be scheduled first (Q: make this by order of ports?)
    std::size_t       min_samples = (MIN_SAMPLES == std::dynamic_extent ? 1 : MIN_SAMPLES);
    std::size_t       max_samples = MAX_SAMPLES;

private:
    bool      _connected    = false;
    IoType    _ioHandler    = new_io_handler();
    TagIoType _tagIoHandler = new_tag_io_handler();

public:
    [[nodiscard]] constexpr auto
    new_io_handler() const noexcept {
        if constexpr (IS_INPUT) {
            return BufferType(65536).new_reader();
        } else {
            return BufferType(65536).new_writer();
        }
    }

    [[nodiscard]] constexpr auto
    new_tag_io_handler() const noexcept {
        if constexpr (IS_INPUT) {
            return TagBufferType(65536).new_reader();
        } else {
            return TagBufferType(65536).new_writer();
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

public:
    port()             = default;
    port(const port &) = delete;
    auto
    operator=(const port &)
            = delete;

    port(std::string port_name, std::int16_t priority_ = 0, std::size_t min_samples_ = 0_UZ, std::size_t max_samples_ = SIZE_MAX) noexcept
        : name(std::move(port_name)), priority{ priority_ }, min_samples(min_samples_), max_samples(max_samples_) {
        static_assert(PortName.empty(), "port name must be exclusively declared via NTTP or constructor parameter");
    }

    constexpr port(port &&other) noexcept : name(std::move(other.name)), priority{ other.priority }, min_samples(other.min_samples), max_samples(other.max_samples) {}

    constexpr port &
    operator=(port &&other) {
        port tmp(std::move(other));
        std::swap(name, tmp._name);
        std::swap(min_samples, tmp._min_samples);
        std::swap(max_samples, tmp._max_samples);
        std::swap(priority, tmp._priority);

        std::swap(_connected, tmp._connected);
        std::swap(_ioHandler, tmp._ioHandler);
        std::swap(_tagIoHandler, tmp._tagIoHandler);
        return *this;
    }

    [[nodiscard]] constexpr static port_type_t
    type() noexcept {
        return PortType;
    }

    [[nodiscard]] constexpr static port_direction_t
    direction() noexcept {
        return PortDirection;
    }

    [[nodiscard]] constexpr static decltype(PortName)
    static_name() noexcept
        requires(!PortName.empty())
    {
        return PortName;
    }

    [[nodiscard]] constexpr supported_type
    pmt_type() const noexcept {
        return T();
    }

    [[nodiscard]] constexpr static std::size_t
    available() noexcept {
        return 0;
    } //  ↔ maps to Buffer::Buffer[Reader, Writer].available()

    [[nodiscard]] constexpr std::size_t
    min_buffer_size() const noexcept {
        if constexpr (MIN_SAMPLES == std::dynamic_extent) {
            return min_samples;
        } else {
            return MIN_SAMPLES;
        }
    }

    [[nodiscard]] constexpr std::size_t
    max_buffer_size() const noexcept {
        if constexpr (MAX_SAMPLES == std::dynamic_extent) {
            return max_samples;
        } else {
            return MAX_SAMPLES;
        }
    }

    [[nodiscard]] constexpr connection_result_t
    resize_buffer(std::size_t min_size) noexcept {
        if constexpr (IS_INPUT) {
            return connection_result_t::SUCCESS;
        } else {
            try {
                _ioHandler    = BufferType(min_size).new_writer();
                _tagIoHandler = TagBufferType(min_size).new_writer();
            } catch (...) {
                return connection_result_t::FAILED;
            }
        }
        return connection_result_t::SUCCESS;
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

template<typename T, std::size_t... Is>
consteval fair::meta::typelist<just_t<T, Is>...>
repeated_ports_impl(std::index_sequence<Is...>) {
    return {};
}
} // namespace detail

// TODO: Add port index to BaseName
template<std::size_t Count, typename T, fixed_string BaseName, port_type_t PortType, port_direction_t PortDirection, std::size_t MIN_SAMPLES = std::dynamic_extent,
         std::size_t MAX_SAMPLES = std::dynamic_extent>
using repeated_ports = decltype(detail::repeated_ports_impl<port<T, BaseName, PortType, PortDirection, MIN_SAMPLES, MAX_SAMPLES>>(std::make_index_sequence<Count>()));

template<typename T, std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent, fixed_string PortName = "">
using IN = port<T, PortName, port_type_t::STREAM, port_direction_t::INPUT, MIN_SAMPLES, MAX_SAMPLES>;
template<typename T, std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent, fixed_string PortName = "">
using OUT = port<T, PortName, port_type_t::STREAM, port_direction_t::OUTPUT, MIN_SAMPLES, MAX_SAMPLES>;
template<typename T, std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent, fixed_string PortName = "">
using IN_MSG = port<T, PortName, port_type_t::MESSAGE, port_direction_t::INPUT, MIN_SAMPLES, MAX_SAMPLES>;
template<typename T, std::size_t MIN_SAMPLES = std::dynamic_extent, std::size_t MAX_SAMPLES = std::dynamic_extent, fixed_string PortName = "">
using OUT_MSG = port<T, PortName, port_type_t::MESSAGE, port_direction_t::OUTPUT, MIN_SAMPLES, MAX_SAMPLES>;

static_assert(Port<IN<float>>);
static_assert(Port<decltype(IN<float>())>);
static_assert(Port<OUT<float>>);
static_assert(Port<IN_MSG<float>>);
static_assert(Port<OUT_MSG<float>>);

static_assert(IN<float, 0, 0, "in">::static_name() == fixed_string("in"));
static_assert(requires { IN<float>("in").name; });

static_assert(OUT_MSG<float, 0, 0, "out_msg">::static_name() == fixed_string("out_msg"));
static_assert(!(OUT_MSG<float, 0, 0, "out_msg">::with_name<"out_message">::static_name() == fixed_string("out_msg")));
static_assert(OUT_MSG<float, 0, 0, "out_msg">::with_name<"out_message">::static_name() == fixed_string("out_message"));

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
        pmt_type() const noexcept
                = 0;

        [[nodiscard]] virtual port_type_t
        type() const noexcept
                = 0;

        [[nodiscard]] virtual port_direction_t
        direction() const noexcept
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

    template<Port T, bool owning>
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

        [[nodiscard]] constexpr supported_type
        pmt_type() const noexcept override {
            return _value.pmt_type();
        }

        [[nodiscard]] constexpr port_type_t
        type() const noexcept override {
            return _value.type();
        }

        [[nodiscard]] constexpr port_direction_t
        direction() const noexcept override {
            return _value.direction();
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
            if constexpr (T::IS_OUTPUT) {
                auto src_buffer = _value.writer_handler_internal();
                return dst_port.update_reader_internal(src_buffer) ? connection_result_t::SUCCESS : connection_result_t::FAILED;
            } else {
                assert(false && "This works only on input ports");
                return connection_result_t::FAILED;
            }
        }
    };

    bool
    update_reader_internal(internal_port_buffers buffer_other) noexcept {
        return _accessor->update_reader_internal(buffer_other);
    }

public:
    using value_type                      = void; // a sterile port

    constexpr dynamic_port()              = delete;

    dynamic_port(const dynamic_port &arg) = delete;
    dynamic_port &
    operator=(const dynamic_port &arg)
            = delete;

    dynamic_port(dynamic_port &&arg) = default;
    dynamic_port &
    operator=(dynamic_port &&arg)
            = delete;

    // TODO: Make owning versus non-owning API more explicit
    template<Port T>
    explicit constexpr dynamic_port(T &arg) noexcept
        : name(arg.name), priority(arg.priority), min_samples(arg.min_samples), max_samples(arg.max_samples), _accessor{ std::make_unique<wrapper<T, false>>(arg) } {}

    template<Port T>
    explicit constexpr dynamic_port(T &&arg) noexcept
        : name(arg.name), priority(arg.priority), min_samples(arg.min_samples), max_samples(arg.max_samples), _accessor{ std::make_unique<wrapper<T, true>>(std::forward<T>(arg)) } {}

    [[nodiscard]] supported_type
    pmt_type() const noexcept {
        return _accessor->pmt_type();
    }

    [[nodiscard]] port_type_t
    type() const noexcept {
        return _accessor->type();
    }

    [[nodiscard]] port_direction_t
    direction() const noexcept {
        return _accessor->direction();
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

static_assert(Port<dynamic_port>);

constexpr void
publish_tag(Port auto &port, property_map &&tag_data, std::size_t tag_offset = 0) noexcept {
    port.tagWriter().publish(
            [&port, data = std::move(tag_data), &tag_offset](std::span<fair::graph::tag_t> tag_output) {
                tag_output[0].index = port.streamWriter().position() + std::make_signed_t<std::size_t>(tag_offset);
                tag_output[0].map   = std::move(data);
            },
            1_UZ);
}

constexpr void
publish_tag(Port auto &port, const property_map &tag_data, std::size_t tag_offset = 0) noexcept {
    port.tagWriter().publish(
            [&port, &tag_data, &tag_offset](std::span<fair::graph::tag_t> tag_output) {
                tag_output[0].index = port.streamWriter().position() + tag_offset;
                tag_output[0].map   = tag_data;
            },
            1_UZ);
}

} // namespace fair::graph

#endif // include guard
