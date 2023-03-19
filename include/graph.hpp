#ifndef GNURADIO_GRAPH_HPP
#define GNURADIO_GRAPH_HPP

#include "circular_buffer.hpp"
#include "buffer.hpp"
#include "typelist.hpp"
#include "port.hpp"
#include "node.hpp"

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

using namespace fair::literals;

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
    update_reader_internal(internal_port_buffers buffer_other) noexcept {
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

        virtual work_return_t
        work() = 0;

        virtual void *
        raw()
                = 0;
    };

    template<typename T>
    class node_wrapper final : public node_model {
    private:
        static_assert(std::is_same_v<T, std::remove_reference_t<T>>);
        T _node;

    public:
        node_wrapper(const node_wrapper &other) = delete;

        node_wrapper &
        operator=(const node_wrapper &other)
                = delete;

        node_wrapper(node_wrapper &&other) : _node(std::exchange(other._node, nullptr)) {}

        node_wrapper &
        operator=(node_wrapper &&other) {
            auto tmp = std::move(other);
            std::swap(_node, tmp._node);
            return *this;
        }

        ~node_wrapper() override = default;

        node_wrapper() {}

        template<typename Arg>
            requires (!std::is_same_v<std::remove_cvref_t<Arg>, T>)
        node_wrapper(Arg&& arg) : _node(std::forward<Arg>(arg)) {}

        template<typename ...Args>
            requires (sizeof...(Args) > 1)
        node_wrapper(Args&&... args) : _node{std::forward<Args>(args)...} {}

        constexpr work_return_t
        work() override {
            return _node.work();
        }

        std::string_view
        name() const override {
            return _node.name();
        }

        void *
        raw() override {
            return std::addressof(_node);
        }
    };

    class edge {
    public:
        using port_direction_t::INPUT;
        using port_direction_t::OUTPUT;
        node_model* _src_node;
        node_model* _dst_node;
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

        edge(node_model* src_node, std::size_t src_port_index, node_model* dst_node, std::size_t dst_port_index, int32_t weight, std::string_view name)
            : _src_node(src_node)
            , _dst_node(dst_node)
            , _src_port_index(src_port_index)
            , _dst_port_index(dst_port_index)
            , _weight(weight)
            , _name(name) {
        }

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

    template<std::size_t src_port_index, std::size_t dst_port_index, typename Source, typename SourcePort, typename Destination, typename DestinationPort>
    [[nodiscard]] connection_result_t
    connect_impl(Source &src_node_raw, SourcePort& source_port, Destination &dst_node_raw, DestinationPort& destination_port,
            int32_t weight = 0, std::string_view name = "unnamed edge") {
        static_assert(
                std::is_same_v<typename SourcePort::value_type, typename DestinationPort::value_type>,
                "The source port type needs to match the sink port type");

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
            auto find_wrapper = [this] (auto* node) {
                auto it = std::find_if(_nodes.begin(), _nodes.end(), [node] (auto& wrapper) {
                        return wrapper->raw() == node;
                    });
                if (it == _nodes.end()) {
                    throw fmt::format("This node {} does not belong to this graph\n", node->name());
                }
                return it->get();
            };
            auto* src_node = find_wrapper(&src_node_raw);
            auto* dst_node = find_wrapper(&dst_node_raw);
            _edges.emplace_back(src_node, src_port_index, dst_node, src_port_index, weight, name);
        }

        return result;
    }

    std::vector<std::function<connection_result_t()>> _connection_definitions;

    // Just a dummy class that stores the graph and the source node and port
    // to be able to split the connection into two separate calls
    // connect(source) and .to(destination)
    template <typename Source, typename Port, std::size_t src_port_index = 1_UZ>
    struct source_connector {
        graph& self;
        Source& source;
        Port& port;

        source_connector(graph& _self, Source& _source, Port& _port) : self(_self), source(_source), port(_port) {}

    private:
        template <typename Destination, typename DestinationPort, std::size_t dst_port_index = meta::invalid_index>
        [[nodiscard]] constexpr auto to(Destination& destination, DestinationPort& destination_port) {
            // Not overly efficient as the node doesn't know the graph it belongs to,
            // but this is not a frequent operation and the check is important.
            auto is_node_known = [this] (const auto& query_node) {
                return std::any_of(self._nodes.cbegin(), self._nodes.cend(), [&query_node] (const auto& known_node) {
                    return known_node->raw() == std::addressof(query_node);
                        });

            };
            if (!is_node_known(source) || !is_node_known(destination)) {
                throw fmt::format("Source {} and/or destination {} do not belong to this graph\n", source.name(), destination.name());
            }
            self._connection_definitions.push_back([self = &self, source = &source, source_port = &port, destination = &destination, destination_port = &destination_port] () {
                return self->connect_impl<src_port_index, dst_port_index>(*source, *source_port, *destination, *destination_port);
            });
            return connection_result_t::SUCCESS;
        }

    public:
        template <typename Destination, typename DestinationPort, std::size_t dst_port_index = meta::invalid_index>
        [[nodiscard]] constexpr auto to(Destination& destination, DestinationPort Destination::* member_ptr) {
            return to<Destination, DestinationPort, dst_port_index>(destination, std::invoke(member_ptr, destination));
        }

        template <std::size_t dst_port_index, typename Destination>
        [[nodiscard]] constexpr auto to(Destination& destination) {
            auto &destination_port = input_port<dst_port_index>(&destination);
            return to<Destination, std::remove_cvref_t<decltype(destination_port)>, dst_port_index>(destination, destination_port);
        }

        template <fixed_string dst_port_name, typename Destination>
        [[nodiscard]] constexpr auto to(Destination& destination) {
            using destination_input_ports = typename traits::node::input_ports<Destination>;
            constexpr std::size_t dst_port_index = meta::indexForName<dst_port_name, destination_input_ports>();
            if constexpr (dst_port_index == meta::invalid_index) {
                meta::print_types<
                    meta::message_type<"There is no input port with the specified name in this destination node">,
                    Destination,
                    meta::message_type<dst_port_name>,
                    meta::message_type<"These are the known names:">,
                    traits::node::input_port_names<Destination>,
                    meta::message_type<"Full ports info:">,
                    destination_input_ports
                        > port_not_found_error{};
            }
            return to<dst_port_index, Destination>(destination);
        }

        source_connector(const source_connector&) = delete;
        source_connector(source_connector&&) = delete;
        source_connector& operator=(const source_connector&) = delete;
        source_connector& operator=(source_connector&&) = delete;
    };

    struct init_proof {
        init_proof(bool _success) : success(_success) {}
        bool success = true;

        operator bool() const { return success; }
    };

    template<std::size_t src_port_index, typename Source>
    friend
    auto connect(Source& source);

    template<fixed_string src_port_name, typename Source>
    friend
    auto connect(Source& source);

    template<typename Source, typename Port>
    friend
    auto connect(Source& source, Port Source::* member_ptr);

public:
    auto
    edges_count() const {
        return _edges.size();
    }

    template<typename Node, typename... Args>
    auto&
    make_node(Args&&... args) {
        static_assert(std::is_same_v<Node, std::remove_reference_t<Node>>);
        auto& new_node_ref = _nodes.emplace_back(std::make_unique<node_wrapper<Node>>(std::forward<Args>(args)...));
        return *static_cast<Node*>(new_node_ref->raw());
    }

    template<std::size_t src_port_index, typename Source>
    [[nodiscard]] auto connect(Source& source) {
        auto &port = output_port<src_port_index>(&source);
        return graph::source_connector<Source, std::remove_cvref_t<decltype(port)>, src_port_index>(*this, source, port);
    }

    template<fixed_string src_port_name, typename Source>
    [[nodiscard]] auto connect(Source& source) {
        using source_output_ports = typename traits::node::output_ports<Source>;
        constexpr std::size_t src_port_index = meta::indexForName<src_port_name, source_output_ports>();
        if constexpr (src_port_index == meta::invalid_index) {
            meta::print_types<
                meta::message_type<"There is no output port with the specified name in this source node">,
                Source,
                meta::message_type<src_port_name>,
                meta::message_type<"These are the known names:">,
                traits::node::output_port_names<Source>,
                meta::message_type<"Full ports info:">,
                source_output_ports
                    > port_not_found_error{};
        }
        return connect<src_port_index, Source>(source);
    }

    template<typename Source, typename Port>
    [[nodiscard]] auto connect(Source& source, Port Source::* member_ptr) {
        return graph::source_connector<Source, Port>(*this, source, std::invoke(member_ptr, source));
    }

    init_proof init() {
        auto result = init_proof(
            std::all_of(_connection_definitions.begin(), _connection_definitions.end(), [] (auto& connection_definition) {
                return connection_definition() == connection_result_t::SUCCESS;
            }));
        _connection_definitions.clear();
        return result;
    }

    work_return_t
    work(init_proof& init) {
        if (!init) {
            return work_return_t::ERROR;
        }
        bool run = true;
        while (run) {
            bool something_happened = false;
            for (auto &node : _nodes) {
                auto result = node->work();
                if (result == work_return_t::ERROR) {
                    return work_return_t::ERROR;
                } else if (result == work_return_t::INSUFFICIENT_INPUT_ITEMS) {
                    // nothing
                } else if (result == work_return_t::DONE) {
                    // nothing
                } else if (result == work_return_t::OK) {
                    something_happened = true;
                } else if (result == work_return_t::INSUFFICIENT_OUTPUT_ITEMS) {
                    something_happened = true;
                }
            }

            run = something_happened;
        }

        return work_return_t::DONE;
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
