#ifndef GNURADIO_NODE_HPP
#define GNURADIO_NODE_HPP

#include <map>

#include <annotated.hpp>
#include <node_traits.hpp>
#include <port.hpp>
#include <tag.hpp>
#include <typelist.hpp>
#include <utils.hpp>

#include <fmt/format.h>
#include <refl.hpp>
#include <reflection.hpp>
#include <settings.hpp>

namespace fair::graph {

using namespace fair::literals;

namespace stdx = vir::stdx;
using fair::meta::fixed_string;

template<typename F>
constexpr void
simd_epilogue(auto width, F &&fun) {
    static_assert(std::has_single_bit(+width));
    auto w2 = std::integral_constant<std::size_t, width / 2>{};
    if constexpr (w2 > 0) {
        fun(w2);
        simd_epilogue(w2, std::forward<F>(fun));
    }
}

template<std::ranges::contiguous_range... Ts, typename Flag = stdx::element_aligned_tag>
constexpr auto
simdize_tuple_load_and_apply(auto width, const std::tuple<Ts...> &rngs, auto offset, auto &&fun, Flag f = {}) {
    using Tup = meta::simdize<std::tuple<std::ranges::range_value_t<Ts>...>, width>;
    return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        return fun(std::tuple_element_t<Is, Tup>(std::ranges::data(std::get<Is>(rngs)) + offset, f)...);
    }(std::make_index_sequence<sizeof...(Ts)>());
}

enum class work_return_t {
    ERROR                     = -100, /// error occurred in the work function
    INSUFFICIENT_OUTPUT_ITEMS = -3,   /// work requires a larger output buffer to produce output
    INSUFFICIENT_INPUT_ITEMS  = -2,   /// work requires a larger input buffer to produce output
    DONE                      = -1,   /// this block has completed its processing and the flowgraph should be done
    OK                        = 0,    /// work call was successful and return values in i/o structs are valid
    CALLBACK_INITIATED        = 1,    /// rather than blocking in the work function, the block will call back to the
                                      /// parent interface when it is ready to be called again
};

template<std::size_t Index, typename Self>
[[nodiscard]] constexpr auto &
input_port(Self *self) noexcept {
    using requested_port_type = typename traits::node::input_ports<Self>::template at<Index>;
    if constexpr (traits::node::node_defines_ports_as_member_variables<Self>) {
        using member_descriptor = traits::node::get_port_member_descriptor<Self, requested_port_type>;
        return member_descriptor()(*self);
    } else {
        return std::get<requested_port_type>(*self);
    }
}

template<std::size_t Index, typename Self>
[[nodiscard]] constexpr auto &
output_port(Self *self) noexcept {
    using requested_port_type = typename traits::node::output_ports<Self>::template at<Index>;
    if constexpr (traits::node::node_defines_ports_as_member_variables<Self>) {
        using member_descriptor = traits::node::get_port_member_descriptor<Self, requested_port_type>;
        return member_descriptor()(*self);
    } else {
        return std::get<requested_port_type>(*self);
    }
}

template<fixed_string Name, typename Self>
[[nodiscard]] constexpr auto &
input_port(Self *self) noexcept {
    constexpr int Index = meta::indexForName<Name, traits::node::input_ports<Self>>();
    return input_port<Index, Self>(self);
}

template<fixed_string Name, typename Self>
[[nodiscard]] constexpr auto &
output_port(Self *self) noexcept {
    constexpr int Index = meta::indexForName<Name, traits::node::output_ports<Self>>();
    return output_port<Index, Self>(self);
}

template<typename Self>
[[nodiscard]] constexpr auto
input_ports(Self *self) noexcept {
    return [self]<std::size_t... Idx>(std::index_sequence<Idx...>) { return std::tie(input_port<Idx>(self)...); }(std::make_index_sequence<traits::node::input_ports<Self>::size>());
}

template<typename Self>
[[nodiscard]] constexpr auto
output_ports(Self *self) noexcept {
    return [self]<std::size_t... Idx>(std::index_sequence<Idx...>) { return std::tie(output_port<Idx>(self)...); }(std::make_index_sequence<traits::node::output_ports<Self>::size>());
}

/**
 * @brief The 'node<Derived>' is a base class for blocks that perform specific signal processing operations. It stores
 * references to its input and output 'ports' that can be zero, one, or many, depending on the use case.
 * As the base class for all user-defined nodes, it implements common convenience functions and a default public API
 * through the Curiously-Recurring-Template-Pattern (CRTP). For example:
 * @code
 * struct user_defined_block : node<user_defined_block> {
 *   IN<float> in;
 *   OUT<float> out;
 *   // implement one of the possible work or abstracted functions
 * };
 * ENABLE_REFLECTION(user_defined_block, in, out);
 * @endcode
 * The macro `ENABLE_REFLECTION` since it relies on a template specialisation needs to be declared on the global scope.
 *
 * As an alternative definition that does not require the 'ENABLE_REFLECTION' macro and that also supports arbitrary
 * types for input 'T' and for the return 'R':
 * @code
 * template<typename T, typename R>
 * struct user_defined_block : node<user_defined_block, IN<T, 0, N_MAX, "in">, OUT<R, 0, N_MAX, "out">> {
 *   // implement one of the possible work or abstracted functions
 * };
 * @endcode
 * This implementation provides efficient compile-time static polymorphism (i.e. access to the ports, settings, etc. does
 * not require virtual functions or inheritance, which can have performance penalties in high-performance computing contexts).
 * Note: The template parameter '<Derived>' can be dropped once C++23's 'deducing this' is widely supported by compilers.
 *
 * The 'node<Derived>' implementation provides simple defaults for users who want to focus on generic signal-processing
 * algorithms and don't need full flexibility (and complexity) of using the generic `work_return_t work() {...}`.
 * The following defaults are defined for one of the two 'user_defined_block' block definitions (WIP):
 * <ul>
 * <li> <b>case 1a</b> - non-decimating N-in->N-out mechanic and automatic handling of streaming tags and settings changes:
 * @code
 *  fg::IN<T> in;
 *  fg::OUT<R> out;
 *  T _factor = T{1.0};
 *
 *  [[nodiscard]] constexpr auto process_one(T a) const noexcept {
 *      return static_cast<R>(a * _factor);
 *  }
 * @endcode
 * The number, type, and ordering of input and arguments of `process_one(..)` are defined by the port definitions.
 * <li> <b>case 1b</b> - non-decimating N-in->N-out mechanic providing bulk access to the input/output data and automatic
 * handling of streaming tags and settings changes:
 * @code
 *  [[nodiscard]] constexpr auto process_bulk(std::span<const T> input, std::span<R> output) const noexcept {
 *      std::ranges::copy(input, output | std::views::transform([a = this->_factor](T x) { return static_cast<R>(x * a); }));
 *  }
 * @endcode
 * <li> <b>case 2a</b>: N-in->M-out -> process_bulk(<ins...>, <outs...>) N,M fixed -> aka. interpolator (M>N) or decimator (M<N) (to-be-done)
 * <li> <b>case 2b</b>: N-in->M-out -> process_bulk(<{ins,tag-IO}...>, <{outs,tag-IO}...>) user-level tag handling (to-be-done)
 * <li> <b>case 3</b> -- generic `work()` providing full access/logic capable of handling any N-in->M-out tag-handling case:
 * @code
 * [[nodiscard]] constexpr work_return_t work() const noexcept {
 *     auto &out_port = output_port<"out">(this);
 *     auto &in_port = input_port<"in">(this);
 *
 *     auto &reader = in_port.streamReader();
 *     auto &writer = out_port.streamWriter();
 *     const auto n_readable = std::min(reader.available(), in_port.max_buffer_size());
 *     const auto n_writable = std::min(writer.available(), out_port.max_buffer_size());
 *     if (n_readable == 0) {
 *         return fair::graph::work_return_t::INSUFFICIENT_INPUT_ITEMS;
 *     } else if (n_writable == 0) {
 *         return fair::graph::work_return_t::INSUFFICIENT_OUTPUT_ITEMS;
 *     }
 *     const std::size_t n_to_publish = std::min(n_readable, n_writable); // N.B. here enforcing N_input == N_output
 *
 *     writer.publish([&reader, n_to_publish, this](std::span<T> output) {
 *         const auto input = reader.get(n_to_publish);
 *         for (; i < n_to_publish; i++) {
 *             output[i] = input[i] * value;
 *         }
 *     }, n_to_publish);
 *
 *     if (!reader.consume(n_to_publish)) {
 *         return fair::graph::work_return_t::ERROR;
 *     }
 *     return fair::graph::work_return_t::OK;
 * }
 * @endcode
 * <li> <b>case 4</b>:  Python -> map to cases 1-3 and/or dedicated callback (to-be-implemented)
 * <li> <b>special cases<b>: (to-be-implemented)
 *     * case sources: HW triggered vs. generating data per invocation (generators via Port::MIN)
 *     * case sinks: HW triggered vs. fixed-size consumer (may block/never finish for insufficient input data and fixed Port::MIN>0)
 * <ul>
 * @tparam Derived the user-defined block CRTP: https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern
 * @tparam Arguments NTTP list containing the compile-time defined port instances, setting structs, or other constraints.
 */
template<typename Derived, typename... Arguments>
class node : protected std::tuple<Arguments...> {
public:
    using derived_t                                      = Derived;
    using node_template_parameters                       = meta::typelist<Arguments...>;
    constexpr static tag_propagation_policy_t tag_policy = tag_propagation_policy_t::TPP_ALL_TO_ALL;

protected:
    using setting_map = std::map<std::string, int, std::less<>>;
    std::string                  _name{ std::string(fair::meta::type_name<Derived>()) };
    bool                         _input_tags_present  = false;
    bool                         _output_tags_changed = false;
    std::vector<tag_t::map_type> _tags_at_input;
    std::vector<tag_t::map_type> _tags_at_output;

    // intermediate non-real-time<->real-time setting states
    std::unique_ptr<settings_base<Derived>> _settings = std::make_unique<basic_settings<Derived>>(self());

    [[nodiscard]] constexpr auto &
    self() noexcept {
        return *static_cast<Derived *>(this);
    }

    [[nodiscard]] constexpr const auto &
    self() const noexcept {
        return *static_cast<const Derived *>(this);
    }

    constexpr bool
    enough_samples_for_output_ports(std::size_t n) {
        return std::apply([n](const auto &...port) noexcept { return ((n >= port.min_buffer_size()) && ... && true); }, output_ports(&self()));
    }

    constexpr bool
    space_available_on_output_ports(std::size_t n) {
        return std::apply([n](const auto &...port) noexcept { return ((n <= port.streamWriter().available()) && ... && true); }, output_ports(&self()));
    }

public:
    node() noexcept : node({}) {}

    node(std::initializer_list<std::pair<const std::string, pmtv::pmt>> init_parameter) noexcept
        : _tags_at_input(traits::node::input_port_types<Derived>::size())
        , _tags_at_output(traits::node::output_port_types<Derived>::size())
        , _settings(std::make_unique<basic_settings<Derived>>(*static_cast<Derived *>(this))) { // N.B. safe delegated use of this (i.e. not used during construction)
        if (init_parameter.size() != 0) {
            std::ignore = settings().set(init_parameter);
        }
    }

    node(node &&other) noexcept
        : std::tuple<Arguments...>(std::move(other)), _tags_at_input(std::move(other._tags_at_input)), _tags_at_output(std::move(other._tags_at_output)), _settings(std::move(other._settings)) {}

    [[nodiscard]] std::string_view
    name() const noexcept {
        return _name;
    }

    void
    set_name(std::string name) noexcept {
        _name = std::move(name);
    }

    [[nodiscard]] constexpr std::string_view
    description() const noexcept {
        using Description = typename fair::meta::find_type_or_default<is_doc, EmptyDoc, Arguments...>::type;
        return std::string_view(Description::value);
    }

    [[nodiscard]] constexpr bool
    input_tags_present() const noexcept {
        return _input_tags_present;
    };

    constexpr bool
    acknowledge_input_tags() noexcept {
        if (_input_tags_present) {
            _input_tags_present = false;
            return true;
        }
        return false;
    };

    [[nodiscard]] constexpr std::span<const tag_t::map_type>
    input_tags() const noexcept {
        return { _tags_at_input.data(), _tags_at_input.size() };
    }

    [[nodiscard]] constexpr std::span<const tag_t::map_type>
    output_tags() const noexcept {
        return { _tags_at_output.data(), _tags_at_output.size() };
    }

    [[nodiscard]] constexpr std::span<tag_t::map_type>
    output_tags() noexcept {
        _output_tags_changed = true;
        return { _tags_at_output.data(), _tags_at_output.size() };
    }

    constexpr settings_base<Derived> &
    settings() const noexcept {
        return *_settings;
    }

    constexpr settings_base<Derived> &
    settings() noexcept {
        return *_settings;
    }

    template<std::size_t Index, typename Self>
    friend constexpr auto &
    input_port(Self *self) noexcept;

    template<std::size_t Index, typename Self>
    friend constexpr auto &
    output_port(Self *self) noexcept;

    template<fixed_string Name, typename Self>
    friend constexpr auto &
    input_port(Self *self) noexcept;

    template<fixed_string Name, typename Self>
    friend constexpr auto &
    output_port(Self *self) noexcept;

    // This function is a template and static to provide easier
    // transition to C++23's deducing this later
    template<typename Self>
    [[nodiscard]] constexpr auto static inputs_status(Self &self) noexcept {
        static_assert(traits::node::input_ports<Derived>::size > 0, "A source node has no inputs, therefore no inputs status.");
        bool       at_least_one_input_has_data = false;
        const auto availableForPort            = [&at_least_one_input_has_data]<typename Port>(Port &port) noexcept -> std::pair<std::size_t, std::size_t> {
            std::size_t       availableSamples = port.streamReader().available();
            const std::size_t availableTags    = port.tagReader().available();
            if (availableTags > 0) {
                // at least one tag is present -> if tag is not on the first tag position read up to the tag position
                auto tagData                  = port.tagReader().get();
                auto tag_stream_head_distance = tagData[0].index - port.streamReader().position();
                assert(tag_stream_head_distance >= 0 && "negative tag vs. stream distance");

                if (tag_stream_head_distance > 0 && availableSamples > tag_stream_head_distance) {
                    // limit number of samples to read up to the next tag <-> forces processing from tag to tag|MAX_SIZE
                    // N.B. new tags are thus always on the first readable sample
                    availableSamples = std::min(availableSamples, static_cast<std::size_t>(tag_stream_head_distance));
                    // TODO: handle corner case where the distance to the next tag is less than the ports MIN_SIZE
                }
            }
            if (availableSamples > 0_UZ) at_least_one_input_has_data = true;
            if (availableSamples < port.min_buffer_size()) {
                return { 0_UZ, availableTags };
            } else {
                return { std::min(availableSamples, port.max_buffer_size()), availableTags };
            }
        };

        const std::pair<std::size_t, std::size_t> available_values_and_tag_count = std::apply([&availableForPort](auto &...input_port) { return meta::safe_min(availableForPort(input_port)...); },
                                                                                              input_ports(&self));

        struct result {
            bool        at_least_one_input_has_data;
            std::size_t available_values_count;
            std::size_t available_tags_count;
        };

        return result{ .at_least_one_input_has_data = at_least_one_input_has_data,
                       .available_values_count      = available_values_and_tag_count.first,
                       .available_tags_count        = available_values_and_tag_count.second };
    }

    // This function is a template and static to provide easier
    // transition to C++23's deducing this later
    auto
    write_to_outputs(std::size_t available_values_count, auto &writers_tuple) noexcept {
        if constexpr (traits::node::output_ports<Derived>::size > 0) {
            meta::tuple_for_each([available_values_count](auto &output_range) { output_range.publish(available_values_count); }, writers_tuple);
        }
    }

    // This function is a template and static to provide easier
    // transition to C++23's deducing this later
    template<typename Self>
    bool
    consume_readers(Self &self, std::size_t available_values_count) {
        bool success = true;
        if constexpr (traits::node::input_ports<Derived>::size > 0) {
            std::apply([available_values_count, &success](auto &...input_port) { ((success = success && input_port.streamReader().consume(available_values_count)), ...); }, input_ports(&self));
        }
        return success;
    }

    template<typename... Ts>
    constexpr auto
    invoke_process_one(Ts &&...inputs) {
        if constexpr (traits::node::output_ports<Derived>::size == 0) {
            self().process_one(std::forward<Ts>(inputs)...);
            return std::tuple{};
        } else if constexpr (traits::node::output_ports<Derived>::size == 1) {
            return std::tuple{ self().process_one(std::forward<Ts>(inputs)...) };
        } else {
            return self().process_one(std::forward<Ts>(inputs)...);
        }
    }

    template<typename... Ts>
    constexpr auto
    invoke_process_one_simd(auto width, Ts &&...input_simds) {
        if constexpr (sizeof...(Ts) == 0) {
            if constexpr (traits::node::output_ports<Derived>::size == 0) {
                self().process_one_simd(width);
                return std::tuple{};
            } else if constexpr (traits::node::output_ports<Derived>::size == 1) {
                return std::tuple{ self().process_one_simd(width) };
            } else {
                return self().process_one_simd(width);
            }
        } else {
            return invoke_process_one(std::forward<Ts>(input_simds)...);
        }
    }

public:
    work_return_t
    work() noexcept {
        using input_types                 = traits::node::input_port_types<Derived>;
        using output_types                = traits::node::output_port_types<Derived>;

        constexpr bool is_source_node     = input_types::size == 0;
        constexpr bool is_sink_node       = output_types::size == 0;

        std::size_t    samples_to_process = 0;
        std::size_t    tags_to_process    = 0;
        if constexpr (is_source_node) {
            if constexpr (requires(const Derived &d) {
                              { self().available_samples(d) } -> std::same_as<std::int64_t>;
                          }) {
                // the (source) node wants to determine the number of samples to process
                std::size_t max_buffer = std::numeric_limits<std::size_t>::max();
                meta::tuple_for_each([&max_buffer](auto &&out) { max_buffer = std::min(max_buffer, out.streamWriter().available()); }, output_ports(&self()));
                const std::int64_t available_samples = self().available_samples(self());
                if (available_samples < 0 && max_buffer > 0) {
                    return work_return_t::DONE;
                }
                if (available_samples == 0) {
                    return work_return_t::OK;
                }
                samples_to_process = std::max(0UL, std::min(static_cast<std::size_t>(available_samples), max_buffer));
                if (not enough_samples_for_output_ports(samples_to_process)) {
                    return work_return_t::INSUFFICIENT_INPUT_ITEMS;
                }
                if (samples_to_process == 0) {
                    return work_return_t::INSUFFICIENT_OUTPUT_ITEMS;
                }
            } else if constexpr (requires(const Derived &d) {
                                     { available_samples(d) } -> std::same_as<std::size_t>;
                                 }) {
                // the (source) node wants to determine the number of samples to process
                samples_to_process = available_samples(self());
                if (samples_to_process == 0) {
                    return work_return_t::OK;
                }
                if (not enough_samples_for_output_ports(samples_to_process)) {
                    return work_return_t::INSUFFICIENT_INPUT_ITEMS;
                }
                if (not space_available_on_output_ports(samples_to_process)) {
                    return work_return_t::INSUFFICIENT_OUTPUT_ITEMS;
                }
            } else if constexpr (is_sink_node) {
                // no input or output buffers, derive from internal "buffer sizes" (i.e. what the
                // buffer size would be if the node were not merged)
                constexpr std::size_t chunk_size = Derived::merged_work_chunk_size();
                static_assert(chunk_size != std::dynamic_extent && chunk_size > 0, "At least one internal port must define a maximum number of samples or the non-member/hidden "
                                                                                   "friend function `available_samples(const NodeType&)` must be defined.");
                samples_to_process = chunk_size;
            } else {
                // derive value from output buffer size
                samples_to_process = std::apply([&](const auto &...ports) { return std::min({ ports.streamWriter().available()..., ports.max_buffer_size()... }); }, output_ports(&self()));
                if (not enough_samples_for_output_ports(samples_to_process)) {
                    return work_return_t::INSUFFICIENT_OUTPUT_ITEMS;
                }
                // space_available_on_output_ports is true by construction of samples_to_process
            }
        } else {
            // Capturing structured bindings does not work in Clang...
            const auto [at_least_one_input_has_data, available_values_count, available_tags_count] = self().inputs_status(self());
            if (available_values_count == 0) {
                return at_least_one_input_has_data ? work_return_t::INSUFFICIENT_INPUT_ITEMS : work_return_t::DONE;
            }
            samples_to_process = available_values_count;
            tags_to_process    = available_tags_count;
            if (not enough_samples_for_output_ports(samples_to_process)) {
                return work_return_t::INSUFFICIENT_INPUT_ITEMS;
            }
            if (not space_available_on_output_ports(samples_to_process)) {
                return work_return_t::INSUFFICIENT_OUTPUT_ITEMS;
            }
        }

        const auto input_spans   = meta::tuple_transform([samples_to_process](auto &input_port) noexcept { return input_port.streamReader().get(samples_to_process); }, input_ports(&self()));

        auto       writers_tuple = meta::tuple_transform([samples_to_process](auto &output_port) noexcept { return output_port.streamWriter().reserve_output_range(samples_to_process); },
                                                   output_ports(&self()));

        _input_tags_present      = false;
        _output_tags_changed     = false;
        bool auto_change         = false;
        if (tags_to_process) {
            tag_t::map_type merged_tag_map;
            _input_tags_present    = true;
            std::size_t port_index = 0; // TODO absorb this as optional tuple_for_each argument
            meta::tuple_for_each(
                    [&merged_tag_map, &port_index, this](auto &input_port) noexcept {
                        auto tags = input_port.tagReader().get(1_UZ);
                        if (tags.size() > 0 && (tags[0].index == input_port.streamReader().position() || tags[0].index == -1)) {
                            _tags_at_input[port_index].clear();
                            for (const auto &[index, map] : tags) {
                                _tags_at_input[port_index].insert(map.begin(), map.end());
                                merged_tag_map.insert(map.begin(), map.end());
                            }
                            std::ignore = input_port.tagReader().consume(1_UZ);
                            port_index++;
                        }
                    },
                    input_ports(&self()));

            if (_input_tags_present) { // apply tags as new settings if matching
                if (!merged_tag_map.empty()) {
                    settings().auto_update(merged_tag_map);
                    auto_change = true;
                }
            }

            if constexpr (tag_policy == tag_propagation_policy_t::TPP_ALL_TO_ALL) {
                // N.B. ranges omitted because of missing Clang/Emscripten support
                std::for_each(_tags_at_output.begin(), _tags_at_output.end(), [&merged_tag_map](tag_t::map_type &tag) { tag = merged_tag_map; });
                _output_tags_changed = true;
            }
        }

        const auto forward_tags = [this]() noexcept {
            if (!_output_tags_changed) {
                return;
            }
            std::size_t port_id = 0; // TODO absorb this as optional tuple_for_each argument
            // TODO: following function does not call the lvalue but erroneously the lvalue version of publish_tag(...) ?!?!
            // meta::tuple_for_each([&port_id, this](auto &output_port) noexcept { publish_tag2(output_port, _tags_at_output[port_id++]); }, output_ports(&self()));
            meta::tuple_for_each(
                    [&port_id, this](auto &output_port) noexcept {
                        if (_tags_at_output[port_id].empty()) {
                            port_id++;
                            return;
                        }
                        auto data           = output_port.tagWriter().reserve_output_range(1);
                        data[port_id].index = output_port.streamWriter().position();
                        data[port_id].map   = _tags_at_output[port_id];
                        data.publish(1);
                        port_id++;
                    },
                    output_ports(&self()));
            // clear input/output tags after processing,  N.B. ranges omitted because of missing Clang/Emscripten support
            _input_tags_present  = false;
            _output_tags_changed = false;
            std::for_each(_tags_at_input.begin(), _tags_at_input.end(), [](tag_t::map_type &tag) { tag.clear(); });
            std::for_each(_tags_at_output.begin(), _tags_at_output.end(), [](tag_t::map_type &tag) { tag.clear(); });
        };

        if (settings().changed()) {
            const auto forward_parameters = settings().apply_staged_parameters();
            if (!forward_parameters.empty()) {
                std::for_each(_tags_at_output.begin(), _tags_at_output.end(), [&forward_parameters](tag_t::map_type &tag) { tag.insert(forward_parameters.cbegin(), forward_parameters.cend()); });
                _output_tags_changed = true;
            }
            settings()._changed.store(false);
        }

        // TODO: check here whether a process_one(...) or a bulk access process has been defined, cases:
        // case 1a: N-in->N-out -> process_one(...) -> auto-handling of streaming tags
        // case 1b: N-in->N-out -> process_bulk(<ins...>, <outs...>) -> auto-handling of streaming tags
        // case 2a: N-in->M-out -> process_bulk(<ins...>, <outs...>) N,M fixed -> aka. interpolator (M>N) or decimator (M<N)
        // case 2b: N-in->M-out -> process_bulk(<{ins,tag-IO}...>, <{outs,tag-IO}...>) user-level tag handling
        // case 3:  N-in->M-out -> work() N,M arbitrary -> used need to handle the full logic (e.g. PLL algo)
        // case 4:  Python -> map to cases 1-3 and/or dedicated callback
        // special cases:
        // case sources: HW triggered vs. generating data per invocation (generators via Port::MIN)
        // case sinks: HW triggered vs. fixed-size consumer (may block/never finish for insufficient input data and fixed Port::MIN>0)

        if constexpr (requires { &Derived::process_bulk; }) {
            const work_return_t ret = std::apply([this](auto... args) { return static_cast<Derived *>(this)->process_bulk(args...); },
                                                 std::tuple_cat(input_spans, meta::tuple_transform([](auto &output_range) { return std::span(output_range); }, writers_tuple)));

            write_to_outputs(samples_to_process, writers_tuple);
            const bool success = consume_readers(self(), samples_to_process);
            forward_tags();
            return success ? ret : work_return_t::ERROR;
        }

        using input_simd_types  = meta::simdize<typename input_types::template apply<std::tuple>>;
        using output_simd_types = meta::simdize<typename output_types::template apply<std::tuple>>;

        std::integral_constant<std::size_t, (meta::simdize_size_v<input_simd_types> == 0 ? std::size_t(stdx::simd_abi::max_fixed_size<double>)
                                                                                         : std::min(std::size_t(stdx::simd_abi::max_fixed_size<double>), meta::simdize_size_v<input_simd_types> * 4))>
                width{};

        if constexpr ((is_sink_node or meta::simdize_size_v<output_simd_types> != 0) and ((is_source_node and requires(Derived &d) {
                                                                                              { d.process_one_simd(width) };
                                                                                          }) or (meta::simdize_size_v<input_simd_types> != 0 and traits::node::can_process_simd<Derived>))) {
            // SIMD loop
            std::size_t i = 0;
            for (; i + width <= samples_to_process; i += width) {
                const auto &results = simdize_tuple_load_and_apply(width, input_spans, i, [&](const auto &...input_simds) { return invoke_process_one_simd(width, input_simds...); });
                meta::tuple_for_each([i](auto &output_range, const auto &result) { result.copy_to(output_range.data() + i, stdx::element_aligned); }, writers_tuple, results);
            }
            simd_epilogue(width, [&](auto w) {
                if (i + w <= samples_to_process) {
                    const auto results = simdize_tuple_load_and_apply(w, input_spans, i, [&](auto &&...input_simds) { return invoke_process_one_simd(w, input_simds...); });
                    meta::tuple_for_each([i](auto &output_range, auto &result) { result.copy_to(output_range.data() + i, stdx::element_aligned); }, writers_tuple, results);
                    i += w;
                }
            });
        } else {
            // Non-SIMD loop
            for (std::size_t i = 0; i < samples_to_process; ++i) {
                const auto results = std::apply([this, i](auto &...inputs) { return invoke_process_one(inputs[i]...); }, input_spans);
                meta::tuple_for_each([i](auto &output_range, auto &result) { output_range[i] = std::move(result); }, writers_tuple, results);
            }
        }

        write_to_outputs(samples_to_process, writers_tuple);

        const bool success = consume_readers(self(), samples_to_process);

#ifdef _DEBUG
        if (!success) {
            fmt::print("Node {} failed to consume {} values from inputs\n", self().name(), samples_to_process);
        }
#endif
        forward_tags();
        return success ? work_return_t::OK : work_return_t::ERROR;
    } // end: work_return_t work() noexcept { ..}
};

template<typename Node>
concept source_node = requires(Node &node, typename traits::node::input_port_types<Node>::tuple_type const &inputs) {
    {
        [](Node &n, auto &inputs) {
            constexpr std::size_t port_count = traits::node::input_port_types<Node>::size;
            if constexpr (port_count > 0) {
                return []<std::size_t... Is>(Node &n_inside, auto const &tup, std::index_sequence<Is...>) -> decltype(n_inside.process_one(std::get<Is>(tup)...)) {
                    return {};
                }(n, inputs, std::make_index_sequence<port_count>());
            } else {
                return n.process_one();
            }
        }(node, inputs)
    } -> std::same_as<typename traits::node::return_type<Node>>;
};

template<typename Node>
concept sink_node = requires(Node &node, typename traits::node::input_port_types<Node>::tuple_type const &inputs) {
    {
        [](Node &n, auto &inputs) {
            constexpr std::size_t port_count = traits::node::output_port_types<Node>::size;
            []<std::size_t... Is>(Node &n_inside, auto const &tup, std::index_sequence<Is...>) {
                if constexpr (port_count > 0) {
                    std::ignore = n_inside.process_one(std::get<Is>(tup)...);
                } else {
                    n_inside.process_one(std::get<Is>(tup)...);
                }
            }(n, inputs, std::make_index_sequence<traits::node::input_port_types<Node>::size>());
        }(node, inputs)
    };
};

template<source_node Left, sink_node Right, std::size_t OutId, std::size_t InId>
class merged_node : public node<merged_node<Left, Right, OutId, InId>, meta::concat<typename traits::node::input_ports<Left>, meta::remove_at<InId, typename traits::node::input_ports<Right>>>,
                                meta::concat<meta::remove_at<OutId, typename traits::node::output_ports<Left>>, typename traits::node::output_ports<Right>>> {
private:
    // copy-paste from above, keep in sync
    using base = node<merged_node<Left, Right, OutId, InId>, meta::concat<typename traits::node::input_ports<Left>, meta::remove_at<InId, typename traits::node::input_ports<Right>>>,
                      meta::concat<meta::remove_at<OutId, typename traits::node::output_ports<Left>>, typename traits::node::output_ports<Right>>>;

    Left  left;
    Right right;

    // merged_work_chunk_size, that's what friends are for
    friend base;
    template<source_node, sink_node, std::size_t, std::size_t>
    friend class merged_node;

    // returns the minimum of all internal max_samples port template parameters
    static constexpr std::size_t
    merged_work_chunk_size() noexcept {
        constexpr std::size_t left_size = []() {
            if constexpr (requires {
                              { Left::merged_work_chunk_size() } -> std::same_as<std::size_t>;
                          }) {
                return Left::merged_work_chunk_size();
            } else {
                return std::dynamic_extent;
            }
        }();
        constexpr std::size_t right_size = []() {
            if constexpr (requires {
                              { Right::merged_work_chunk_size() } -> std::same_as<std::size_t>;
                          }) {
                return Right::merged_work_chunk_size();
            } else {
                return std::dynamic_extent;
            }
        }();
        return std::min({ traits::node::input_ports<Right>::template apply<traits::port::max_samples>::value, traits::node::output_ports<Left>::template apply<traits::port::max_samples>::value,
                          left_size, right_size });
    }

    template<std::size_t I>
    constexpr auto
    apply_left(auto &&input_tuple) noexcept {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) { return left.process_one(std::get<Is>(std::forward<decltype(input_tuple)>(input_tuple))...); }(std::make_index_sequence<I>());
    }

    template<std::size_t I, std::size_t J>
    constexpr auto
    apply_right(auto &&input_tuple, auto &&tmp) noexcept {
        return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
            constexpr std::size_t first_offset  = traits::node::input_port_types<Left>::size;
            constexpr std::size_t second_offset = traits::node::input_port_types<Left>::size + sizeof...(Is);
            static_assert(second_offset + sizeof...(Js) == std::tuple_size_v<std::remove_cvref_t<decltype(input_tuple)>>);
            return right.process_one(std::get<first_offset + Is>(std::forward<decltype(input_tuple)>(input_tuple))..., std::forward<decltype(tmp)>(tmp), std::get<second_offset + Js>(input_tuple)...);
        }(std::make_index_sequence<I>(), std::make_index_sequence<J>());
    }

public:
    using input_port_types  = typename traits::node::input_port_types<base>;
    using output_port_types = typename traits::node::output_port_types<base>;
    using return_type       = typename traits::node::return_type<base>;

    constexpr merged_node(Left l, Right r) : left(std::move(l)), right(std::move(r)) {}

    // if the left node (source) implements available_samples (a customization point), then pass the call through
    friend constexpr std::size_t
    available_samples(const merged_node &self) noexcept
        requires requires(const Left &l) {
            { available_samples(l) } -> std::same_as<std::size_t>;
        }
    {
        return available_samples(self.left);
    }

    template<meta::any_simd... Ts>
        requires traits::node::can_process_simd<Left> && traits::node::can_process_simd<Right>
    constexpr meta::simdize<return_type, meta::simdize_size_v<std::tuple<Ts...>>>
    process_one(const Ts &...inputs) {
        static_assert(traits::node::output_port_types<Left>::size == 1, "TODO: SIMD for multiple output ports not implemented yet");
        return apply_right<InId, traits::node::input_port_types<Right>::size() - InId - 1>(std::tie(inputs...), apply_left<traits::node::input_port_types<Left>::size()>(std::tie(inputs...)));
    }

    constexpr auto
    process_one_simd(auto N)
        requires traits::node::can_process_simd<Right>
    {
        if constexpr (requires(Left &l) {
                          { l.process_one_simd(N) };
                      }) {
            return right.process_one(left.process_one_simd(N));
        } else {
            using LeftResult = typename traits::node::return_type<Left>;
            using V          = meta::simdize<LeftResult, N>;
            alignas(stdx::memory_alignment_v<V>) LeftResult tmp[V::size()];
            for (std::size_t i = 0; i < V::size(); ++i) {
                tmp[i] = left.process_one();
            }
            return right.process_one(V(tmp, stdx::vector_aligned));
        }
    }

    template<typename... Ts>
    // Nicer error messages for the following would be good, but not at the expense of breaking
    // can_process_simd.
        requires(input_port_types::template are_equal<std::remove_cvref_t<Ts>...>)
    constexpr return_type
    process_one(Ts &&...inputs) {
        // if (sizeof...(Ts) == 0) we could call `return process_one_simd(integral_constant<size_t, width>)`. But if
        // the caller expects to process *one* sample (no inputs for the caller to explicitly
        // request simd), and we process more, we risk inconsistencies.
        if constexpr (traits::node::output_port_types<Left>::size == 1) {
            // only the result from the right node needs to be returned
            return apply_right<InId, traits::node::input_port_types<Right>::size() - InId - 1>(std::forward_as_tuple(std::forward<Ts>(inputs)...),
                                                                                               apply_left<traits::node::input_port_types<Left>::size()>(
                                                                                                       std::forward_as_tuple(std::forward<Ts>(inputs)...)));

        } else {
            // left produces a tuple
            auto left_out  = apply_left<traits::node::input_port_types<Left>::size()>(std::forward_as_tuple(std::forward<Ts>(inputs)...));
            auto right_out = apply_right<InId, traits::node::input_port_types<Right>::size() - InId - 1>(std::forward_as_tuple(std::forward<Ts>(inputs)...), std::move(std::get<OutId>(left_out)));

            if constexpr (traits::node::output_port_types<Left>::size == 2 && traits::node::output_port_types<Right>::size == 1) {
                return std::make_tuple(std::move(std::get<OutId ^ 1>(left_out)), std::move(right_out));

            } else if constexpr (traits::node::output_port_types<Left>::size == 2) {
                return std::tuple_cat(std::make_tuple(std::move(std::get<OutId ^ 1>(left_out))), std::move(right_out));

            } else if constexpr (traits::node::output_port_types<Right>::size == 1) {
                return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
                    return std::make_tuple(std::move(std::get<Is>(left_out))..., std::move(std::get<OutId + 1 + Js>(left_out))..., std::move(right_out));
                }(std::make_index_sequence<OutId>(), std::make_index_sequence<traits::node::output_port_types<Left>::size - OutId - 1>());

            } else {
                return [&]<std::size_t... Is, std::size_t... Js, std::size_t... Ks>(std::index_sequence<Is...>, std::index_sequence<Js...>, std::index_sequence<Ks...>) {
                    return std::make_tuple(std::move(std::get<Is>(left_out))..., std::move(std::get<OutId + 1 + Js>(left_out))..., std::move(std::get<Ks>(right_out)...));
                }(std::make_index_sequence<OutId>(), std::make_index_sequence<traits::node::output_port_types<Left>::size - OutId - 1>(), std::make_index_sequence<Right::output_port_types::size>());
            }
        }
    } // end:: process_one

    work_return_t
    work() noexcept {
        return base::work();
    }
};

/**
 * This methods can merge simple blocks that are defined via a single `auto process_one(..)` producing a
 * new `merged` node, bypassing the dynamic run-time buffers.
 * Since the merged node can be highly optimised during compile-time, it's execution performance is usually orders
 * of magnitude more efficient than executing a cascade of the same constituent blocks. See the benchmarks for details.
 * This function uses the connect-by-port-ID API.
 *
 * Example:
 * @code
 * // declare flow-graph: 2 x in -> adder -> scale-by-2 -> scale-by-minus1 -> output
 * auto merged = merge_by_index<0, 0>(scale<int, -1>(), merge_by_index<0, 0>(scale<int, 2>(), adder<int>()));
 *
 * // execute graph
 * std::array<int, 4> a = { 1, 2, 3, 4 };
 * std::array<int, 4> b = { 10, 10, 10, 10 };
 *
 * int                r = 0;
 * for (std::size_t i = 0; i < 4; ++i) {
 *     r += merged.process_one(a[i], b[i]);
 * }
 * @endcode
 */
template<std::size_t OutId, std::size_t InId, source_node A, sink_node B>
constexpr auto
merge_by_index(A &&a, B &&b) -> merged_node<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId> {
    if constexpr (!std::is_same_v<typename traits::node::output_port_types<std::remove_cvref_t<A>>::template at<OutId>,
                                  typename traits::node::input_port_types<std::remove_cvref_t<B>>::template at<InId>>) {
        fair::meta::print_types<fair::meta::message_type<"OUTPUT_PORTS_ARE:">, typename traits::node::output_port_types<std::remove_cvref_t<A>>, std::integral_constant<int, OutId>,
                                typename traits::node::output_port_types<std::remove_cvref_t<A>>::template at<OutId>,

                                fair::meta::message_type<"INPUT_PORTS_ARE:">, typename traits::node::input_port_types<std::remove_cvref_t<A>>, std::integral_constant<int, InId>,
                                typename traits::node::input_port_types<std::remove_cvref_t<A>>::template at<InId>>{};
    }
    return { std::forward<A>(a), std::forward<B>(b) };
}

/**
 * This methods can merge simple blocks that are defined via a single `auto process_one(..)` producing a
 * new `merged` node, bypassing the dynamic run-time buffers.
 * Since the merged node can be highly optimised during compile-time, it's execution performance is usually orders
 * of magnitude more efficient than executing a cascade of the same constituent blocks. See the benchmarks for details.
 * This function uses the connect-by-port-name API.
 *
 * Example:
 * @code
 * // declare flow-graph: 2 x in -> adder -> scale-by-2 -> output
 * auto merged = merge<"scaled", "addend1">(scale<int, 2>(), adder<int>());
 *
 * // execute graph
 * std::array<int, 4> a = { 1, 2, 3, 4 };
 * std::array<int, 4> b = { 10, 10, 10, 10 };
 *
 * int                r = 0;
 * for (std::size_t i = 0; i < 4; ++i) {
 *     r += merged.process_one(a[i], b[i]);
 * }
 * @endcode
 */
template<fixed_string OutName, fixed_string InName, source_node A, sink_node B>
constexpr auto
merge(A &&a, B &&b) {
    constexpr std::size_t OutId = meta::indexForName<OutName, typename traits::node::output_ports<A>>();
    constexpr std::size_t InId  = meta::indexForName<InName, typename traits::node::input_ports<B>>();
    static_assert(OutId != -1);
    static_assert(InId != -1);
    static_assert(std::same_as<typename traits::node::output_port_types<std::remove_cvref_t<A>>::template at<OutId>, typename traits::node::input_port_types<std::remove_cvref_t<B>>::template at<InId>>,
                  "Port types do not match");
    return merged_node<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId>{ std::forward<A>(a), std::forward<B>(b) };
}

#if !DISABLE_SIMD
namespace test {
struct copy : public node<copy, IN<float, 0, -1_UZ, "in">, OUT<float, 0, -1_UZ, "out">> {
public:
    template<meta::t_or_simd<float> V>
    constexpr V
    process_one(const V &a) const noexcept {
        return a;
    }
};

static_assert(traits::node::input_port_types<copy>::size() == 1);
static_assert(std::same_as<traits::node::return_type<copy>, float>);
static_assert(traits::node::can_process_simd<copy>);
static_assert(traits::node::can_process_simd<decltype(merge_by_index<0, 0>(copy(), copy()))>);
} // namespace test
#endif

} // namespace fair::graph

#endif // include guard
