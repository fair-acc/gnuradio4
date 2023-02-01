#ifndef GNURADIO_GRAPH_HPP
#define GNURADIO_GRAPH_HPP

#include <circular_buffer.hpp> // localinclude
#include <buffer.hpp> // localinclude
#include <utils.hpp> // localinclude
#include <typelist.hpp> // localinclude
#include <port.hpp> // localinclude
#include <node.hpp> // localinclude
#include <vir/simd.h> // localinclude

#include <algorithm>
#include <complex>
#include <iostream>
#include <map>
#include <ranges>
#include <tuple>
#include <variant>

#if !__has_include(<source_location>)
#define HAVE_SOURCE_LOCATION 0
#else

#include <source_location>

#if defined __cpp_lib_source_location && __cpp_lib_source_location >= 201907L
#define HAVE_SOURCE_LOCATION 1
#else
#define HAVE_SOURCE_LOCATION 0
#endif
#endif

namespace fair::graph {

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

        [[nodiscard]] virtual std::string_view
        name() const noexcept
                = 0;

        [[nodiscard]] virtual connection_result_t
        resize_buffer(std::size_t min_size) noexcept
                = 0;

        [[nodiscard]] virtual connection_result_t
        disconnect() noexcept
                = 0;

        [[nodiscard]] virtual connection_result_t
        connect(dynamic_port &dst_port) = 0;

        // internal runtime polymorphism access
        [[nodiscard]] virtual bool
        update_reader_internal(void *buffer_other) noexcept
                = 0;
    };

    std::unique_ptr<model> _accessor;

    template<Port T, bool owning>
    class wrapper final : public model {
        using PortType = std::decay_t<T>;
        std::conditional_t<owning, PortType, PortType &> _value;

        [[nodiscard]] void *
        writer_handler_internal() noexcept {
            return _value.writer_handler_internal();
        };

        [[nodiscard]] bool
        update_reader_internal(void *buffer_other) noexcept override {
            if constexpr (T::IS_INPUT) {
                return _value.update_reader_internal(buffer_other);
            } else {
                assert(!"This works only on input ports");
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
                        requires { arg.update_reader_internal(std::declval<void *>()); }, "'private bool update_reader_internal(void* buffer)' not implemented");
            }
        }

        explicit constexpr wrapper(T &&arg) noexcept : _value{ std::move(arg) } {
            if constexpr (T::IS_INPUT) {
                static_assert(
                        requires { arg.writer_handler_internal(); }, "'private void* writer_handler_internal()' not implemented");
            } else {
                static_assert(
                        requires { arg.update_reader_internal(std::declval<void *>()); }, "'private bool update_reader_internal(void* buffer)' not implemented");
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

        [[nodiscard]] constexpr std::string_view
        name() const noexcept override {
            return _value.name();
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
                return dst_port.update_reader_internal(src_buffer) ? connection_result_t::SUCCESS
                                                                   : connection_result_t::FAILED;
            } else {
                assert(!"This works only on input ports");
                return connection_result_t::FAILED;
            }
        }
    };

    bool
    update_reader_internal(void *buffer_other) noexcept {
        return _accessor->update_reader_internal(buffer_other);
    }

public:
    using value_type         = void; // a sterile port

    constexpr dynamic_port() = delete;

    template<Port T>
    constexpr dynamic_port(const T &arg) = delete;

    template<Port T>
    explicit constexpr dynamic_port(T &arg) noexcept : _accessor{ std::make_unique<wrapper<T, false>>(arg) } {}

    template<Port T>
    explicit constexpr dynamic_port(T &&arg) noexcept : _accessor{ std::make_unique<wrapper<T, true>>(std::forward<T>(arg)) } {}

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

    [[nodiscard]] std::string_view
    name() const noexcept {
        return _accessor->name();
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

#define ENABLE_PYTHON_INTEGRATION
#ifdef ENABLE_PYTHON_INTEGRATION

// TODO: Not yet implemented
class dynamic_node {
private:
    // TODO: replace the following with array<2, vector<dynamic_port>>
    using dynamic_ports = std::vector<dynamic_port>;
    dynamic_ports                                         _dynamic_input_ports;
    dynamic_ports                                         _dynamic_output_ports;

    std::function<void(dynamic_ports &, dynamic_ports &)> _process;

public:
    void
    work() {
        _process(_dynamic_input_ports, _dynamic_output_ports);
    }

    template<typename T>
    void
    add_port(T &&port) {
        switch (port.direction()) {
        case port_direction_t::INPUT:
            if (auto portID = port_index<port_direction_t::INPUT>(port.name()); portID.has_value()) {
                throw std::invalid_argument(fmt::format("port already has a defined input port named '{}' at ID {}", port.name(), portID.value()));
            }
            _dynamic_input_ports.emplace_back(std::forward<T>(port));
            break;

        case port_direction_t::OUTPUT:
            if (auto portID = port_index<port_direction_t::OUTPUT>(port.name()); portID.has_value()) {
                throw std::invalid_argument(fmt::format("port already has a defined output port named '{}' at ID {}", port.name(), portID.value()));
            }
            _dynamic_output_ports.emplace_back(std::forward<T>(port));
            break;

        default: assert(false && "cannot add port with ANY designation");
        }
    }

    [[nodiscard]] std::optional<dynamic_port *>
    dynamic_input_port(std::size_t index) {
        return index < _dynamic_input_ports.size() ? std::optional{ &_dynamic_input_ports[index] } : std::nullopt;
    }

    [[nodiscard]] std::optional<std::size_t>
    dynamic_input_port_index(std::string_view name) const {
        auto       portNameMatches = [name](const auto &port) { return port.name() == name; };
        const auto it              = std::find_if(_dynamic_input_ports.cbegin(), _dynamic_input_ports.cend(), portNameMatches);
        return it != _dynamic_input_ports.cend() ? std::optional{ std::distance(_dynamic_input_ports.cbegin(), it) } : std::nullopt;
    }

    [[nodiscard]] std::optional<dynamic_port *>
    dynamic_input_port(std::string_view name) {
        if (const auto index = dynamic_input_port_index(name); index.has_value()) {
            return &_dynamic_input_ports[*index];
        }
        return std::nullopt;
    }

    [[nodiscard]] std::optional<dynamic_port *>
    dynamic_output_port(std::size_t index) {
        return index < _dynamic_output_ports.size() ? std::optional{ &_dynamic_output_ports[index] } : std::nullopt;
    }

    [[nodiscard]] std::optional<std::size_t>
    dynamic_output_port_index(std::string_view name) const {
        auto       portNameMatches = [name](const auto &port) { return port.name() == name; };
        const auto it              = std::find_if(_dynamic_output_ports.cbegin(), _dynamic_output_ports.cend(), portNameMatches);
        return it != _dynamic_output_ports.cend() ? std::optional{ std::distance(_dynamic_output_ports.cbegin(), it) } : std::nullopt;
    }

    [[nodiscard]] std::optional<dynamic_port *>
    dynamic_output_port(std::string_view name) {
        if (const auto index = dynamic_output_port_index(name); index.has_value()) {
            return &_dynamic_output_ports[*index];
        }
        return std::nullopt;
    }

    [[nodiscard]] std::span<const dynamic_port>
    dynamic_input_ports() const noexcept {
        return _dynamic_input_ports;
    }

    [[nodiscard]] std::span<const dynamic_port>
    dynamic_output_ports() const noexcept {
        return _dynamic_output_ports;
    }
};

#endif


class graph {
private:
    class node_model {
    public:
        virtual ~node_model() = default;

        virtual std::string_view
        name() const
                = 0;

        virtual node_ports_state
        work() = 0;

        virtual void *
        raw() const
                = 0;
    };

    template<typename T>
    class reference_node_wrapper final : public node_model {
    private:
        T *_node;

        auto &
        data() {
            return *_node;
        }

        const auto &
        data() const {
            return *_node;
        }

    public:
        reference_node_wrapper(const reference_node_wrapper &other) = delete;

        reference_node_wrapper &
        operator=(const reference_node_wrapper &other)
                = delete;

        reference_node_wrapper(reference_node_wrapper &&other) : _node(std::exchange(other._node, nullptr)) {}

        reference_node_wrapper &
        operator=(reference_node_wrapper &&other) {
            auto tmp = std::move(other);
            std::swap(_node, tmp._node);
            return *this;
        }

        ~reference_node_wrapper() override = default;

        template<typename In>
        reference_node_wrapper(In &&node) : _node(std::forward<In>(node)) {}

        node_ports_state
        work() override {
            return data().work();
        }

        std::string_view
        name() const override {
            return data().name();
        }

        void *
        raw() const override {
            return _node;
        }
    };

    class edge {
    public:
        using port_direction_t::INPUT;
        using port_direction_t::OUTPUT;
        std::unique_ptr<node_model> _src_node;
        std::unique_ptr<node_model> _dst_node;
        std::size_t                 _src_port_index;
        std::size_t                 _dst_port_index;
        int32_t                     _weight;
        std::string                 _name; // custom edge name
        bool                        _connected;

    public:
        edge()             = delete;

        edge(const edge &) = delete;

        edge &
        operator=(const edge &)
                = delete;

        edge(edge &&) noexcept = default;

        edge &
        operator=(edge &&) noexcept
                = default;

        edge(std::unique_ptr<node_model> src_node, std::size_t src_port_index, std::unique_ptr<node_model> dst_node, std::size_t dst_port_index, int32_t weight, std::string_view name)
            : _src_node(std::move(src_node))
            , _dst_node(std::move(dst_node))
            , _src_port_index(src_port_index)
            , _dst_port_index(dst_port_index)
            , _weight(weight)
            , _name(name) {
            // if (!_src_node->port<OUTPUT>(_src_port_index)) {
            //     throw fmt::format("source node '{}' has not output port id {}", std::string() /* _src_node->name() */, _src_port_index);
            // }
            // if (!_dst_node->port<INPUT>(_dst_port_index)) {
            //     throw fmt::format("destination node '{}' has not output port id {}", std::string() /*_dst_node->name()*/, _dst_port_index);
            // }
            // const dynamic_port& src_port = *_src_node->port<OUTPUT>(_src_port_index).value();
            // const dynamic_port& dst_port = *_dst_node->port<INPUT>(_dst_port_index).value();
            // if (src_port.pmt_type().index() != dst_port.pmt_type().index()) {
            //     throw fmt::format("edge({}::{}<{}> -> {}::{}<{}>, weight: {}, name:\"{}\") incompatible to type id='{}'",
            //         std::string() /*_src_node->name()*/, std::string() /*src_port.name()*/, src_port.pmt_type().index(),
            //         std::string() /*_dst_node->name()*/, std::string() /*dst_port.name()*/, dst_port.pmt_type().index(),
            //         _weight, _name, dst_port.pmt_type().index());
            // }
        }

        // edge(std::shared_ptr<node_model> src_node, std::string_view src_port_name, std::shared_ptr<node_model> dst_node, std::string_view dst_port_name, int32_t weight, std::string_view name) :
        //         _src_node(src_node), _dst_node(dst_node), _weight(weight), _name(name) {
        //     const auto src_id = _src_node->port_index<OUTPUT>(src_port_name);
        //     const auto dst_id = _dst_node->port_index<INPUT>(dst_port_name);
        //     if (!src_id) {
        //         throw std::invalid_argument(fmt::format("source node '{}' has not output port '{}'", std::string() /*_src_node->name()*/, src_port_name));
        //     }
        //     if (!dst_id) {
        //         throw fmt::format("destination node '{}' has not output port '{}'", std::string() /*_dst_node->name()*/, dst_port_name);
        //     }
        //     _src_port_index = src_id.value();
        //     _dst_port_index = dst_id.value();
        //     const dynamic_port& src_port = *src_node->port<OUTPUT>(_src_port_index).value();
        //     const dynamic_port& dst_port = *dst_node->port<INPUT>(_dst_port_index).value();
        //     if (src_port.pmt_type().index() != dst_port.pmt_type().index()) {
        //         throw fmt::format("edge({}::{}<{}> -> {}::{}<{}>, weight: {}, name:\"{}\") incompatible to type id='{}'",
        //                           std::string() /*_src_node->name()*/, src_port.name(), src_port.pmt_type().index(),
        //                           std::string() /*_dst_node->name()*/, dst_port.name(), dst_port.pmt_type().index(),
        //                           _weight, _name, dst_port.pmt_type().index());
        //     }
        // }

        [[nodiscard]] constexpr int32_t
        weight() const noexcept {
            return _weight;
        }

        [[nodiscard]] constexpr std::string_view
        name() const noexcept {
            return _name;
        }

        [[nodiscard]] constexpr bool
        connected() const noexcept {
            return _connected;
        }

        [[nodiscard]] connection_result_t
        connect() noexcept {
            return connection_result_t::FAILED;
        }

        [[nodiscard]] connection_result_t
        disconnect() noexcept { /* return _dst_node->port<INPUT>(_dst_port_index).value()->disconnect(); */
            return connection_result_t::FAILED;
        }
    };

    std::vector<edge>                        _edges;
    std::vector<std::unique_ptr<node_model>> _nodes;

    template<std::size_t src_port_index, std::size_t dst_port_index, typename Source_, typename Destination_>
    [[nodiscard]] connection_result_t
    connect_impl(Source_ &src_node_raw, Destination_ &dst_node_raw, int32_t weight = 0,
            std::string_view name = "unnamed edge") {
        using Source = std::remove_cvref_t<Source_>;
        using Destination = std::remove_cvref_t<Destination_>;
        static_assert(
                std::is_same_v<typename Source::output_port_types::template at<src_port_index>, typename Destination::input_port_types::template at<dst_port_index>>,
                "The source port type needs to match the sink port type");

        OutPort auto &source_port = output_port<src_port_index>(&src_node_raw);
        InPort auto &destination_port = input_port<dst_port_index>(&dst_node_raw);

        if (!std::any_of(_nodes.begin(), _nodes.end(), [&](const auto &registered_node) {
            return registered_node->raw() == std::addressof(src_node_raw);
        })
            || !std::any_of(_nodes.begin(), _nodes.end(), [&](const auto &registered_node) {
            return registered_node->raw() == std::addressof(dst_node_raw);
        })) {
            throw std::runtime_error(fmt::format("Can not connect nodes that are not registered first:\n {}:{} -> {}:{}\n", src_node_raw.name(), src_port_index, dst_node_raw.name(), dst_port_index));
        }

        auto result = source_port.connect(destination_port);
        if (result == connection_result_t::SUCCESS) {
            std::unique_ptr<node_model> src_node = std::make_unique<reference_node_wrapper<Source>>(std::addressof(src_node_raw));
            std::unique_ptr<node_model> dst_node = std::make_unique<reference_node_wrapper<Destination>>(std::addressof(dst_node_raw));
            _edges.emplace_back(std::move(src_node), src_port_index, std::move(dst_node), src_port_index, weight, name);
        }

        return result;
    }

    // Just a dummy class that stores the graph and the source node and port
    // to be able to split the connection into two separate calls
    // connect(source) and .to(destination)
    template <std::size_t src_port_index, typename Source>
    struct source_connector {
        graph& self;
        Source& source;

        source_connector(graph& _self, Source& _source) : self(_self), source(_source) {}

        template <std::size_t dst_port_index, typename Destination>
        [[nodiscard]] auto to(Destination& destination) {
            return self.connect_impl<src_port_index, dst_port_index>(source, destination);
        }

        template <fixed_string dst_port_name, typename Destination>
        [[nodiscard]] auto to(Destination& destination) {
            return self.connect_impl<src_port_index, meta::indexForName<dst_port_name, typename Destination::input_ports>()>(source, destination);
        }

        source_connector(const source_connector&) = delete;
        source_connector(source_connector&&) = delete;
        source_connector& operator=(const source_connector&) = delete;
        source_connector& operator=(source_connector&&) = delete;
    };

public:
    template<std::size_t src_port_index, typename Source>
    [[nodiscard]] auto connect(Source& source) {
        return source_connector<src_port_index, Source>(*this, source);
    }

    template<fixed_string src_port_name, typename Source>
    [[nodiscard]] auto connect(Source& source) {
        return connect<meta::indexForName<src_port_name, typename Source::output_ports>(), Source>(source);
    }

    auto
    edges_count() const {
        return _edges.size();
    }

    template<typename Node>
    void
    register_node(Node &node) {
        static_assert(std::is_same_v<Node, std::remove_reference_t<Node>>);
        _nodes.push_back(std::make_unique<reference_node_wrapper<Node>>(std::addressof(node)));
    }

    node_ports_state
    work() {
        bool run = true;
        while (run) {
            bool something_happened = false;
            for (auto &node : _nodes) {
                auto result = node->work();
                if (result == node_ports_state::error) {
                    return node_ports_state::error;
                } else if (result == node_ports_state::has_unprocessed_data) {
                    // nothing
                } else if (result == node_ports_state::inputs_empty) {
                    // nothing
                } else if (result == node_ports_state::success) {
                    something_happened = true;
                } else if (result == node_ports_state::writers_not_available) {
                    something_happened = true;
                }
            }

            run = something_happened;
        }

        return node_ports_state::inputs_empty;
    }
};

// TODO: add nicer enum formatter
inline std::ostream &
operator<<(std::ostream &os, const connection_result_t &value) {
    return os << static_cast<int>(value);
}

inline std::ostream &
operator<<(std::ostream &os, const port_type_t &value) {
    return os << static_cast<int>(value);
}

inline std::ostream &
operator<<(std::ostream &os, const port_direction_t &value) {
    return os << static_cast<int>(value);
}

inline std::ostream &
operator<<(std::ostream &os, const port_domain_t &value) {
    return os << static_cast<int>(value);
}

#ifndef __EMSCRIPTEN__
auto
this_source_location(std::source_location l = std::source_location::current()) {
    return fmt::format("{}:{},{}", l.file_name(), l.line(), l.column());
}
#else
auto
this_source_location() {
    return "not yet implemented";
}
#endif // __EMSCRIPTEN__

} // namespace fair::graph

#endif // include guard
