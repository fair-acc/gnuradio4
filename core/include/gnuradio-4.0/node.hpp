#ifndef GNURADIO_NODE_HPP
#define GNURADIO_NODE_HPP

#include <fmt/format.h>
#include <map>

#include "node_traits.hpp"
#include "port.hpp"
#include "sequence.hpp"
#include "tag.hpp"
#include "thread/thread_pool.hpp"
#include "typelist.hpp"
#include "utils.hpp"

#include "annotated.hpp" // This needs to be included after fmt/format.h, as it defines formatters only if FMT_FORMAT_H_ is defined
#include "reflection.hpp"
#include "settings.hpp"

#ifdef FMT_FORMAT_H_
#include <fmt/core.h>
#include <limits>
#endif

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

template<typename T, typename... Us>
auto
invoke_process_one_with_or_without_offset(T &node, std::size_t offset, const Us &...inputs) {
    if constexpr (traits::node::can_process_one_with_offset<T>) return node.process_one(offset, inputs...);
    else
        return node.process_one(inputs...);
}

enum class work_return_status_t {
    ERROR                     = -100, /// error occurred in the work function
    INSUFFICIENT_OUTPUT_ITEMS = -3,   /// work requires a larger output buffer to produce output
    INSUFFICIENT_INPUT_ITEMS  = -2,   /// work requires a larger input buffer to produce output
    DONE                      = -1,   /// this block has completed its processing and the flowgraph should be done
    OK                        = 0,    /// work call was successful and return values in i/o structs are valid
};

struct work_return_t {
    std::size_t          requested_work = std::numeric_limits<std::size_t>::max();
    std::size_t          performed_work = 0;
    work_return_status_t status         = work_return_status_t::OK;
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

template<typename T>
concept NodeType = requires(T t, std::size_t requested_work) {
    { t.unique_name } -> std::same_as<const std::string &>;
    { unwrap_if_wrapped_t<decltype(t.name)>{} } -> std::same_as<std::string>;
    { unwrap_if_wrapped_t<decltype(t.meta_information)>{} } -> std::same_as<property_map>;
    { t.description } noexcept -> std::same_as<const std::string_view &>;

    { t.is_blocking() } noexcept -> std::same_as<bool>;

    { t.settings() } -> std::same_as<settings_base &>;
    { t.work(requested_work) } -> std::same_as<work_return_t>;

    // N.B. TODO discuss these requirements
    requires !std::is_copy_constructible_v<T>;
    requires !std::is_copy_assignable_v<T>;
    // requires !std::is_move_constructible_v<T>;
    // requires !std::is_move_assignable_v<T>;
};

namespace detail {
class WorkCounter {
    std::atomic_uint64_t encodedCounter{ static_cast<uint64_t>(std::numeric_limits<std::uint32_t>::max()) << 32 };

public:
    void
    increment(std::size_t workRequestedInc, std::size_t workDoneInc) {
        uint64_t oldCounter;
        uint64_t newCounter;
        do {
            oldCounter         = encodedCounter;
            auto workRequested = static_cast<std::uint32_t>(oldCounter >> 32);
            auto workDone      = static_cast<std::uint32_t>(oldCounter & 0xFFFFFFFF);
            if (workRequested != std::numeric_limits<std::uint32_t>::max()) {
                workRequested = static_cast<uint32_t>(std::min(static_cast<std::uint64_t>(workRequested) + workRequestedInc, static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())));
            }
            workDone += static_cast<std::uint32_t>(workDoneInc);
            newCounter = (static_cast<uint64_t>(workRequested) << 32) | workDone;
        } while (!encodedCounter.compare_exchange_weak(oldCounter, newCounter));
    }

    std::pair<std::size_t, std::size_t>
    getAndReset() {
        uint64_t oldCounter    = encodedCounter.exchange(0);
        auto     workRequested = static_cast<std::uint32_t>(oldCounter >> 32);
        auto     workDone      = static_cast<std::uint32_t>(oldCounter & 0xFFFFFFFF);
        if (workRequested == std::numeric_limits<std::uint32_t>::max()) {
            return { std::numeric_limits<std::size_t>::max(), static_cast<std::size_t>(workDone) };
        }
        return { static_cast<std::size_t>(workRequested), static_cast<std::size_t>(workDone) };
    }

    std::pair<std::size_t, std::size_t>
    get() {
        uint64_t oldCounter    = encodedCounter.load();
        auto     workRequested = static_cast<std::uint32_t>(oldCounter >> 32);
        auto     workDone      = static_cast<std::uint32_t>(oldCounter & 0xFFFFFFFF);
        if (workRequested == std::numeric_limits<std::uint32_t>::max()) {
            return { std::numeric_limits<std::size_t>::max(), static_cast<std::size_t>(workDone) };
        }
        return { static_cast<std::size_t>(workRequested), static_cast<std::size_t>(workDone) };
    }
};
} // namespace detail

template<typename Derived>
concept HasProcessOneFunction = traits::node::can_process_one<Derived>;

template<typename Derived>
concept HasProcessBulkFunction = traits::node::can_process_bulk<Derived>;

template<typename Derived>
concept HasRequiredProcessFunction = (HasProcessBulkFunction<Derived> or HasProcessOneFunction<Derived>) and(HasProcessOneFunction<Derived> + HasProcessBulkFunction<Derived>) == 1;

template<typename T>
concept ConsumableSpan = std::ranges::contiguous_range<T> and std::convertible_to<T, std::span<const std::remove_cvref_t<typename T::value_type>>> and requires(T &s) { s.consume(0); };

static_assert(ConsumableSpan<traits::node::detail::dummy_input_span<float>>);

template<typename T>
concept PublishableSpan = std::ranges::contiguous_range<T> and std::ranges::output_range<T, std::remove_cvref_t<typename T::value_type>>
                      and std::convertible_to<T, std::span<std::remove_cvref_t<typename T::value_type>>> and requires(T &s) { s.publish(0_UZ); };

static_assert(PublishableSpan<traits::node::detail::dummy_output_span<float>>);

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
 *         return { 0, fair::graph::work_return_status_t::INSUFFICIENT_INPUT_ITEMS };
 *     } else if (n_writable == 0) {
 *         return { 0, fair::graph::work_return_status_t::INSUFFICIENT_OUTPUT_ITEMS };
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
 *         return { n_to_publish, fair::graph::work_return_status_t::ERROR };
 *     }
 *     return { n_to_publish, fair::graph::work_return_status_t::OK };
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
struct node : protected std::tuple<Arguments...> {
    static std::atomic_size_t _unique_id_counter;
    template<typename T, fair::meta::fixed_string description = "", typename... Args>
    using A                          = Annotated<T, description, Args...>;

    using base_t                     = node<Derived, Arguments...>;
    using derived_t                  = Derived;
    using node_template_parameters   = meta::typelist<Arguments...>;
    using Description                = typename node_template_parameters::template find_or_default<is_doc, EmptyDoc>;
    constexpr static bool blockingIO = std::disjunction_v<std::is_same<BlockingIO<true>, Arguments>...> || std::disjunction_v<std::is_same<BlockingIO<false>, Arguments>...>;

    alignas(hardware_destructive_interference_size) std::atomic_uint32_t ioThreadRunning{ 0_UZ };
    alignas(hardware_destructive_interference_size) std::atomic_bool ioThreadShallRun{ false };
    alignas(hardware_destructive_interference_size) std::atomic<std::size_t> ioRequestedWork{ std::numeric_limits<std::size_t>::max() };
    alignas(hardware_destructive_interference_size) detail::WorkCounter ioWorkDone{};
    alignas(hardware_destructive_interference_size) std::atomic<work_return_status_t> ioLastWorkStatus{ work_return_status_t::OK };
    alignas(hardware_destructive_interference_size) std::shared_ptr<gr::Sequence> progress                           = std::make_shared<gr::Sequence>();
    alignas(hardware_destructive_interference_size) std::shared_ptr<fair::thread_pool::BasicThreadPool> ioThreadPool = std::make_shared<fair::thread_pool::BasicThreadPool>(
            "node_thread_pool", fair::thread_pool::TaskType::IO_BOUND, 2_UZ, std::numeric_limits<uint32_t>::max());

    constexpr static tag_propagation_policy_t tag_policy = tag_propagation_policy_t::TPP_ALL_TO_ALL;
    A<std::size_t, "numerator", Doc<"top number of input-to-output sample ratio: < 1 decimation, >1 interpolation, 1_ no effect">, Limits<1_UZ, std::size_t(-1)>>      numerator      = 1_UZ;
    A<std::size_t, "denominator", Doc<"bottom number of input-to-output sample ratio: < 1 decimation, >1 interpolation, 1_ no effect">, Limits<1_UZ, std::size_t(-1)>> denominator    = 1_UZ;
    A<std::size_t, "stride", Doc<"samples between data processing. <N for overlap, >N for skip, =0 for back-to-back.">>                                                stride         = 0_UZ;
    std::size_t                                                                                                                                                        stride_counter = 0_UZ;
    const std::size_t                                                                                                                                                  unique_id = _unique_id_counter++;
    const std::string                                                                                              unique_name = fmt::format("{}#{}", fair::meta::type_name<Derived>(), unique_id);
    A<std::string, "user-defined name", Doc<"N.B. may not be unique -> ::unique_name">>                            name{ std::string(fair::meta::type_name<Derived>()) };
    A<property_map, "meta-information", Doc<"store non-graph-processing information like UI block position etc.">> meta_information;
    constexpr static std::string_view                                                                              description = static_cast<std::string_view>(Description::value);

    struct ports_status_t {
        std::size_t in_min_samples{ std::numeric_limits<std::size_t>::min() };         // max of `port.min_buffer_size()` of all input ports
        std::size_t in_max_samples{ std::numeric_limits<std::size_t>::max() };         // min of `port.max_buffer_size()` of all input ports
        std::size_t in_available{ std::numeric_limits<std::size_t>::max() };           // min of `port.streamReader().available()` of all input ports
        std::size_t in_samples_to_next_tag{ std::numeric_limits<std::size_t>::max() }; // min of `port.samples_to_next_tag` of all input ports

        std::size_t out_min_samples{ std::numeric_limits<std::size_t>::min() }; // max of `port.min_buffer_size()` of all output ports
        std::size_t out_max_samples{ std::numeric_limits<std::size_t>::max() }; // min of `port.max_buffer_size()` of all output ports
        std::size_t out_available{ std::numeric_limits<std::size_t>::max() };   // min of `port.streamWriter().available()` of all input ports

        std::size_t in_samples{ 0 };  // number of input samples to process
        std::size_t out_samples{ 0 }; // number of output samples, calculated based on `numerator` and `denominator`

        bool        in_at_least_one_port_has_data{ false }; // at least one port has data
        bool        in_at_least_one_tag_available{ false }; // at least one port has a tag

        bool        has_sync_input_ports{ false };  // if all ports are async, status is not important
        bool        has_sync_output_ports{ false }; // if all ports are async, status is not important

        constexpr bool
        enough_samples_for_output_ports(std::size_t n) {
            return !has_sync_output_ports || n >= out_min_samples;
        }

        constexpr bool
        space_available_on_output_ports(std::size_t n) {
            return !has_sync_output_ports || n <= out_available;
        }
    };

    ports_status_t ports_status{};

protected:
    bool               _input_tags_present  = false;
    bool               _output_tags_changed = false;
    std::vector<tag_t> _tags_at_input;
    std::vector<tag_t> _tags_at_output;

    // intermediate non-real-time<->real-time setting states
    std::unique_ptr<settings_base> _settings = std::make_unique<basic_settings<Derived>>(self());

    [[nodiscard]] constexpr auto &
    self() noexcept {
        return *static_cast<Derived *>(this);
    }

    [[nodiscard]] constexpr const auto &
    self() const noexcept {
        return *static_cast<const Derived *>(this);
    }

    void
    update_ports_status() {
        ports_status               = ports_status_t();

        auto adjust_for_input_port = [&ps = ports_status]<PortType Port>(Port &port) {
            if constexpr (std::remove_cvref_t<Port>::synchronous) {
                ps.has_sync_input_ports          = true;
                ps.in_min_samples                = std::max(ps.in_min_samples, port.min_buffer_size());
                ps.in_max_samples                = std::min(ps.in_max_samples, port.max_buffer_size());
                ps.in_available                  = std::min(ps.in_available, port.streamReader().available());
                ps.in_samples_to_next_tag        = std::min(ps.in_samples_to_next_tag, samples_to_next_tag(port));
                ps.in_at_least_one_port_has_data = ps.in_at_least_one_port_has_data | (port.streamReader().available() > 0);
                ps.in_at_least_one_tag_available = ps.in_at_least_one_port_has_data | (port.tagReader().available() > 0);
            }
        };
        meta::tuple_for_each(
                [&adjust_for_input_port]<typename Port>(Port &port_or_collection) {
                    if constexpr (traits::port::is_port_v<Port>) {
                        adjust_for_input_port(port_or_collection);
                    } else {
                        for (auto &port : port_or_collection) {
                            adjust_for_input_port(port);
                        }
                    }
                },
                input_ports(&self()));

        auto adjust_for_output_port = [&ps = ports_status]<PortType Port>(Port &port) {
            if constexpr (std::remove_cvref_t<Port>::synchronous) {
                ps.has_sync_output_ports = true;
                ps.out_min_samples       = std::max(ps.out_min_samples, port.min_buffer_size());
                ps.out_max_samples       = std::min(ps.out_max_samples, port.max_buffer_size());
                ps.out_available         = std::min(ps.out_available, port.streamWriter().available());
            }
        };
        meta::tuple_for_each(
                [&adjust_for_output_port]<typename Port>(Port &port_or_collection) {
                    if constexpr (traits::port::is_port_v<Port>) {
                        adjust_for_output_port(port_or_collection);
                    } else {
                        for (auto &port : port_or_collection) {
                            adjust_for_output_port(port);
                        }
                    }
                },
                output_ports(&self()));

        ports_status.in_samples = ports_status.in_available;
        if (ports_status.in_samples < ports_status.in_min_samples) ports_status.in_samples = 0;
        if (ports_status.in_samples > ports_status.in_max_samples) ports_status.in_samples = ports_status.in_max_samples;

        // By default N-in == N-out
        // TODO: adjust `samples_to_proceed` to output limits?
        ports_status.out_samples = ports_status.in_samples;

        if (ports_status.has_sync_input_ports && ports_status.in_min_samples > ports_status.in_max_samples)
            throw std::runtime_error(fmt::format("Min samples for input ports ({}) is larger then max samples for input ports ({})", ports_status.in_min_samples, ports_status.in_max_samples));
        if (ports_status.has_sync_output_ports && ports_status.out_min_samples > ports_status.out_max_samples)
            throw std::runtime_error(fmt::format("Min samples for output ports ({}) is larger then max samples for output ports ({})", ports_status.out_min_samples, ports_status.out_max_samples));

        if (!ports_status.has_sync_input_ports) {
            ports_status.in_samples   = 0;
            ports_status.in_available = 0;
        }
        if (!ports_status.has_sync_output_ports) {
            ports_status.out_samples   = 0;
            ports_status.out_available = 0;
        }
    }

public:
    node() noexcept : node({}) {}

    node(std::initializer_list<std::pair<const std::string, pmtv::pmt>> init_parameter)
        : _tags_at_input(traits::node::input_port_types<Derived>::size())
        , _tags_at_output(traits::node::output_port_types<Derived>::size())
        , _settings(std::make_unique<basic_settings<Derived>>(*static_cast<Derived *>(this))) { // N.B. safe delegated use of this (i.e. not used during construction)
        if (init_parameter.size() != 0) {
            const auto failed = settings().set(init_parameter);
            if (!failed.empty()) {
                throw std::invalid_argument("Settings not applied successfully");
            }
        }
    }

    node(node &&other) noexcept
        : std::tuple<Arguments...>(std::move(other)), _tags_at_input(std::move(other._tags_at_input)), _tags_at_output(std::move(other._tags_at_output)), _settings(std::move(other._settings)) {}

    ~node() { // NOSONAR -- need to request the (potentially) running ioThread to stop
        stop();
    }

    void
    init(std::shared_ptr<gr::Sequence> progress_, std::shared_ptr<fair::thread_pool::BasicThreadPool> ioThreadPool_) {
        progress     = std::move(progress_);
        ioThreadPool = std::move(ioThreadPool_);
        if (const auto forward_parameters = settings().apply_staged_parameters(); !forward_parameters.empty()) {
            std::for_each(_tags_at_output.begin(), _tags_at_output.end(), [&forward_parameters](tag_t &tag) {
                for (const auto &[key, value] : forward_parameters) {
                    tag.map.insert_or_assign(key, value);
                }
            });
            _output_tags_changed = true;
        }

        // store default settings -> can be recovered with 'reset_defaults()'
        settings().store_defaults();
    }

    void
    stop() {
        std::atomic_store_explicit(&ioThreadShallRun, false, std::memory_order_release);
        ioThreadShallRun.notify_all();
        // wait for done
        for (auto running = ioThreadRunning.load(); running > 0; running = ioThreadRunning.load()) {
            ioThreadRunning.wait(running);
        }
    }

    template<fair::meta::array_or_vector_type Container>
    [[nodiscard]] constexpr std::size_t
    available_input_samples(Container &data) const noexcept {
        if constexpr (fair::meta::vector_type<Container>) {
            data.resize(traits::node::input_port_types<Derived>::size);
        } else if constexpr (fair::meta::array_type<Container>) {
            static_assert(std::tuple_size<Container>::value >= traits::node::input_port_types<Derived>::size);
        } else {
            static_assert(fair::meta::always_false<Container>, "type not supported");
        }
        meta::tuple_for_each_enumerate(
                [&data]<typename Port>(auto index, Port &input_port) {
                    if constexpr (traits::port::is_port_v<Port>) {
                        data[index] = input_port.streamReader().available();
                    } else {
                        data[index] = 0;
                        for (auto &port : input_port) {
                            data[index] += port.streamReader().available();
                        }
                    }
                },
                input_ports(&self()));
        return traits::node::input_port_types<Derived>::size;
    }

    template<fair::meta::array_or_vector_type Container>
    [[nodiscard]] constexpr std::size_t
    available_output_samples(Container &data) const noexcept {
        if constexpr (fair::meta::vector_type<Container>) {
            data.resize(traits::node::output_port_types<Derived>::size);
        } else if constexpr (fair::meta::array_type<Container>) {
            static_assert(std::tuple_size<Container>::value >= traits::node::output_port_types<Derived>::size);
        } else {
            static_assert(fair::meta::always_false<Container>, "type not supported");
        }
        meta::tuple_for_each_enumerate(
                [&data]<typename Port>(auto index, Port &output_port) {
                    if constexpr (traits::port::is_port_v<Port>) {
                        data[index] = output_port.streamWriter().available();
                    } else {
                        data[index] = 0;
                        for (auto &port : output_port) {
                            data[index] += port.streamWriter().available();
                        }
                    }
                },
                output_ports(&self()));
        return traits::node::output_port_types<Derived>::size;
    }

    [[nodiscard]] constexpr bool
    is_blocking() const noexcept {
        return blockingIO;
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

    [[nodiscard]] constexpr std::span<const tag_t>
    input_tags() const noexcept {
        return { _tags_at_input.data(), _tags_at_input.size() };
    }

    [[nodiscard]] constexpr std::span<const tag_t>
    output_tags() const noexcept {
        return { _tags_at_output.data(), _tags_at_output.size() };
    }

    [[nodiscard]] constexpr std::span<tag_t>
    output_tags() noexcept {
        _output_tags_changed = true;
        return { _tags_at_output.data(), _tags_at_output.size() };
    }

    [[nodiscard]] constexpr settings_base &
    settings() const noexcept {
        return *_settings;
    }

    [[nodiscard]] constexpr settings_base &
    settings() noexcept {
        return *_settings;
    }

    template<typename T>
    void
    setSettings(std::unique_ptr<T> &settings) {
        _settings = std::move(settings);
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

    void
    write_to_outputs(std::size_t available_values_count, auto &writers_tuple) noexcept {
        if constexpr (traits::node::output_ports<Derived>::size > 0) {
            meta::tuple_for_each_enumerate(
                    [available_values_count]<typename OutputRange>(auto i, OutputRange &output_range) {
                        if constexpr (traits::node::can_process_one<Derived> or traits::node::process_bulk_requires_ith_output_as_span<Derived, i>) {
                            auto process_out = [available_values_count]<typename Out>(Out &out) {
                                // This will be a pointer if the port was async
                                // TODO: Make this check more specific
                                if constexpr (not std::is_pointer_v<std::remove_cvref_t<Out>>) {
                                    out.publish(available_values_count);
                                }
                            };
                            if (available_values_count) {
                                if constexpr (refl::trait::is_instance_of_v<std::vector, std::remove_cvref_t<OutputRange>>) {
                                    for (auto &out : output_range) {
                                        process_out(out);
                                    }
                                } else {
                                    process_out(output_range);
                                }
                            }
                        } else {
                            if constexpr (requires { output_range.is_published(); }) {
                                if (not output_range.is_published()) {
                                    fmt::print(stderr, "process_bulk failed to publish one of its outputs. Use a std::span argument if you do not want to publish manually.\n");
                                    std::abort();
                                }
                            }
                        }
                    },
                    writers_tuple);
        }
    }

    // This function is a template and static to provide easier
    // transition to C++23's deducing this later
    template<typename Self>
    bool
    consume_readers(Self &self, std::size_t available_values_count) {
        bool success = true;
        if constexpr (traits::node::input_ports<Derived>::size > 0) {
            std::apply(
                    [available_values_count, &success](auto &...input_port) {
                        auto consume_port = [&]<typename Port>(Port &port_or_collection) {
                            if constexpr (traits::port::is_port_v<Port>) {
                                success = success && port_or_collection.streamReader().consume(available_values_count);
                            } else {
                                for (auto &port : port_or_collection) {
                                    success = success && port.streamReader().consume(available_values_count);
                                }
                            }
                        };
                        (consume_port(input_port), ...);
                    },
                    input_ports(&self));
        }
        return success;
    }

    template<typename... Ts>
    constexpr auto
    invoke_process_one(std::size_t offset, Ts &&...inputs) {
        if constexpr (traits::node::output_ports<Derived>::size == 0) {
            invoke_process_one_with_or_without_offset(self(), offset, std::forward<Ts>(inputs)...);
            return std::tuple{};
        } else if constexpr (traits::node::output_ports<Derived>::size == 1) {
            return std::tuple{ invoke_process_one_with_or_without_offset(self(), offset, std::forward<Ts>(inputs)...) };
        } else {
            return invoke_process_one_with_or_without_offset(self(), offset, std::forward<Ts>(inputs)...);
        }
    }

    template<typename... Ts>
    constexpr auto
    invoke_process_one_simd(std::size_t offset, auto width, Ts &&...input_simds) {
        if constexpr (sizeof...(Ts) == 0) {
            if constexpr (traits::node::output_ports<Derived>::size == 0) {
                self().process_one_simd(offset, width);
                return std::tuple{};
            } else if constexpr (traits::node::output_ports<Derived>::size == 1) {
                return std::tuple{ self().process_one_simd(offset, width) };
            } else {
                return self().process_one_simd(offset, width);
            }
        } else {
            return invoke_process_one(offset, std::forward<Ts>(input_simds)...);
        }
    }

    constexpr void
    forward_tags() noexcept {
        if (!(_output_tags_changed || _input_tags_present)) {
            return;
        }
        std::size_t port_id = 0; // TODO absorb this as optional tuple_for_each argument
        // TODO: following function does not call the lvalue but erroneously the lvalue version of publish_tag(...) ?!?!
        // meta::tuple_for_each([&port_id, this](auto &output_port) noexcept { publish_tag2(output_port, _tags_at_output[port_id++]); }, output_ports(&self()));
        meta::tuple_for_each(
                [&port_id, this]<typename Port>(Port &output_port) noexcept {
                    if constexpr (!traits::port::is_port_v<Port>) {
                        // TODO Add tag support to port collections?
                        return;
                    } else {
                        if (_tags_at_output[port_id].map.empty()) {
                            port_id++;
                            return;
                        }
                        auto data                 = output_port.tagWriter().reserve_output_range(1);
                        auto stream_writer_offset = std::max(static_cast<decltype(output_port.streamWriter().position())>(0), output_port.streamWriter().position() + 1);
                        data[0].index             = stream_writer_offset + _tags_at_output[port_id].index;
                        data[0].map               = _tags_at_output[port_id].map;
                        data.publish(1);
                        port_id++;
                    }
                },
                output_ports(&self()));
        // clear input/output tags after processing,  N.B. ranges omitted because of missing Clang/Emscripten support
        _input_tags_present  = false;
        _output_tags_changed = false;
        std::for_each(_tags_at_input.begin(), _tags_at_input.end(), [](tag_t &tag) { tag.reset(); });
        std::for_each(_tags_at_output.begin(), _tags_at_output.end(), [](tag_t &tag) { tag.reset(); });
    }

protected:
    /**
     * @brief
     * @return struct { std::size_t produced_work, work_return_t}
     */
    work_return_t
    work_internal(std::size_t requested_work) {
        using fair::graph::work_return_status_t;
        using input_types             = traits::node::input_port_types<Derived>;
        using output_types            = traits::node::output_port_types<Derived>;

        constexpr bool is_source_node = input_types::size == 0;
        constexpr bool is_sink_node   = output_types::size == 0;

        // TODO: these checks can be moved to setting changed
        if constexpr (node_template_parameters::template contains<PerformDecimationInterpolation>) {
            static_assert(!is_sink_node, "Decimation/interpolation is not available for sink blocks. Remove 'PerformDecimationInterpolation' from the block definition.");
            static_assert(!is_source_node, "Decimation/interpolation is not available for source blocks. Remove 'PerformDecimationInterpolation' from the block definition.");
            static_assert(HasProcessBulkFunction<Derived>,
                          "Blocks which allow decimation/interpolation must implement process_bulk(...) method. Remove 'PerformDecimationInterpolation' from the block definition.");
        } else {
            if (numerator != 1_UZ || denominator != 1_UZ) {
                throw std::runtime_error(
                        fmt::format("Block is not defined as `PerformDecimationInterpolation`, but numerator = {}, denominator = {}, they both must equal to 1.", numerator, denominator));
            }
        }

        if constexpr (node_template_parameters::template contains<PerformStride>) {
            static_assert(!is_source_node, "Stride is not available for source blocks. Remove 'PerformStride' from the block definition.");
        } else {
            if (stride != 0_UZ) {
                throw std::runtime_error(fmt::format("Block is not defined as `PerformStride`, but stride = {}, it must equal to 0.", stride));
            }
        }

        update_ports_status();

        if constexpr (is_source_node) {
            ports_status.in_samples_to_next_tag = std::numeric_limits<std::size_t>::max(); // no tags to processed for source node
            if constexpr (requires(const Derived &d) {
                              { self().available_samples(d) } -> std::same_as<std::make_signed_t<std::size_t>>;
                          }) {
                // the (source) node wants to determine the number of samples to process
                std::size_t                           max_buffer        = ports_status.out_available;
                const std::make_signed_t<std::size_t> available_samples = self().available_samples(self());
                if (available_samples < 0 && max_buffer > 0) {
                    return { requested_work, 0_UZ, work_return_status_t::DONE };
                }
                if (available_samples == 0) {
                    return { requested_work, 0_UZ, work_return_status_t::OK };
                }
                std::size_t samples_to_process = std::max(0UL, std::min(static_cast<std::size_t>(available_samples), max_buffer));
                if (not ports_status.enough_samples_for_output_ports(samples_to_process)) {
                    return { requested_work, 0_UZ, work_return_status_t::INSUFFICIENT_INPUT_ITEMS };
                }
                if (samples_to_process == 0) {
                    return { requested_work, 0_UZ, work_return_status_t::INSUFFICIENT_OUTPUT_ITEMS };
                }
                ports_status.in_samples  = std::min(samples_to_process, requested_work);
                ports_status.out_samples = ports_status.in_samples;

            } else if constexpr (requires(const Derived &d) {
                                     { available_samples(d) } -> std::same_as<std::size_t>;
                                 }) {
                // the (source) node wants to determine the number of samples to process
                std::size_t samples_to_process = available_samples(self());
                if (samples_to_process == 0) {
                    return { requested_work, 0_UZ, work_return_status_t::OK };
                }
                if (not ports_status.enough_samples_for_output_ports(samples_to_process)) {
                    return { requested_work, 0_UZ, work_return_status_t::INSUFFICIENT_INPUT_ITEMS };
                }
                if (not ports_status.space_available_on_output_ports(samples_to_process)) {
                    return { requested_work, 0_UZ, work_return_status_t::INSUFFICIENT_OUTPUT_ITEMS };
                }
                ports_status.in_samples  = std::min(samples_to_process, requested_work);
                ports_status.out_samples = ports_status.in_samples;

            } else if constexpr (is_sink_node) {
                // no input or output buffers, derive from internal "buffer sizes" (i.e. what the
                // buffer size would be if the node were not merged)
                constexpr std::size_t chunk_size = Derived::merged_work_chunk_size();
                static_assert(chunk_size != std::dynamic_extent && chunk_size > 0, "At least one internal port must define a maximum number of samples or the non-member/hidden "
                                                                                   "friend function `available_samples(const NodeType&)` must be defined.");
                ports_status.in_samples  = std::min(chunk_size, requested_work);
                ports_status.out_samples = ports_status.in_samples;

            } else {
                // derive value from output buffer size
                std::size_t samples_to_process = std::min(ports_status.out_available, ports_status.out_max_samples);
                if (not ports_status.enough_samples_for_output_ports(samples_to_process)) {
                    return { requested_work, 0_UZ, work_return_status_t::INSUFFICIENT_OUTPUT_ITEMS };
                }
                ports_status.in_samples  = std::min(samples_to_process, requested_work);
                ports_status.out_samples = ports_status.in_samples;
                // space_available_on_output_ports is true by construction of samples_to_process
            }

        } else {
            ports_status.in_samples  = std::min(ports_status.in_samples, requested_work);
            ports_status.out_samples = ports_status.in_samples;

            if (ports_status.has_sync_input_ports && ports_status.in_available == 0) {
                return { requested_work, 0_UZ, ports_status.in_at_least_one_port_has_data ? work_return_status_t::INSUFFICIENT_INPUT_ITEMS : work_return_status_t::DONE };
            }

            if constexpr (node_template_parameters::template contains<PerformDecimationInterpolation>) {
                if (numerator != 1_UZ || denominator != 1_UZ) {
                    // TODO: this ill-defined checks can be done only once after parameters were changed
                    const double ratio  = static_cast<double>(numerator) / static_cast<double>(denominator);
                    bool is_ill_defined = (denominator > ports_status.in_max_samples) || (static_cast<double>(ports_status.in_min_samples) * ratio > static_cast<double>(ports_status.out_max_samples))
                                       || (static_cast<double>(ports_status.in_max_samples) * ratio < static_cast<double>(ports_status.out_min_samples));
                    assert(!is_ill_defined);
                    if (is_ill_defined) {
                        return { requested_work, 0_UZ, work_return_status_t::ERROR };
                    }

                    ports_status.in_samples          = static_cast<std::size_t>(ports_status.in_samples / denominator) * denominator; // remove reminder

                    const std::size_t out_min_limit  = ports_status.out_min_samples;
                    const std::size_t out_max_limit  = std::min(ports_status.out_available, ports_status.out_max_samples);

                    std::size_t       in_min_samples = static_cast<std::size_t>(static_cast<double>(out_min_limit) / ratio);
                    if (in_min_samples % denominator != 0) in_min_samples += denominator;
                    std::size_t       in_min_wo_reminder = (in_min_samples / denominator) * denominator;

                    const std::size_t in_max_samples     = static_cast<std::size_t>(static_cast<double>(out_max_limit) / ratio);
                    std::size_t       in_max_wo_reminder = (in_max_samples / denominator) * denominator;

                    if (ports_status.in_samples < in_min_wo_reminder) return { requested_work, 0_UZ, work_return_status_t::INSUFFICIENT_INPUT_ITEMS };
                    ports_status.in_samples  = std::clamp(ports_status.in_samples, in_min_wo_reminder, in_max_wo_reminder);
                    ports_status.out_samples = numerator * (ports_status.in_samples / denominator);
                }
            }

            // TODO: special case for ports_status.in_samples == 0 ?

            if (not ports_status.enough_samples_for_output_ports(ports_status.out_samples)) {
                return { requested_work, 0_UZ, work_return_status_t::INSUFFICIENT_INPUT_ITEMS };
            }
            if (not ports_status.space_available_on_output_ports(ports_status.out_samples)) {
                return { requested_work, 0_UZ, work_return_status_t::INSUFFICIENT_OUTPUT_ITEMS };
            }
        }

        if (ports_status.in_samples_to_next_tag == 0) {
            if constexpr (HasProcessOneFunction<Derived>) {
                ports_status.in_samples  = 1; // N.B. limit to one so that only one process_on(...) invocation receives the tag
                ports_status.out_samples = 1;
            }
            property_map merged_tag_map;
            _input_tags_present    = true;
            std::size_t port_index = 0; // TODO absorb this as optional tuple_for_each argument
            meta::tuple_for_each(
                    [&merged_tag_map, &port_index, this]<typename Port>(Port &input_port) noexcept {
                        // TODO: Do we want to support tags for non-compile-time ports? [ivan][port_group][move_to_policy?]
                        if constexpr (traits::port::is_port_v<Port>) {
                            auto &tag_at_present_input = _tags_at_input[port_index++];
                            tag_at_present_input.reset();
                            if (!input_port.tagReader().available()) {
                                return;
                            }
                            const auto tags           = input_port.tagReader().get(1_UZ);
                            const auto readPos        = input_port.streamReader().position();
                            const auto tag_stream_pos = tags[0].index - 1 - readPos;
                            if ((readPos == -1 && tags[0].index <= 0) // first tag on initialised stream
                                || tag_stream_pos <= 0) {
                                for (const auto &[index, map] : tags) {
                                    for (const auto &[key, value] : map) {
                                        tag_at_present_input.map.insert_or_assign(key, value);
                                        merged_tag_map.insert_or_assign(key, value);
                                    }
                                }
                                std::ignore = input_port.tagReader().consume(1_UZ);
                            }
                        }
                    },
                    input_ports(&self()));

            if (_input_tags_present && !merged_tag_map.empty()) { // apply tags as new settings if matching
                settings().auto_update(merged_tag_map);
            }

            if constexpr (Derived::tag_policy == tag_propagation_policy_t::TPP_ALL_TO_ALL) {
                // N.B. ranges omitted because of missing Clang/Emscripten support
                std::for_each(_tags_at_output.begin(), _tags_at_output.end(), [&merged_tag_map](tag_t &tag) { tag.map = merged_tag_map; });
                _output_tags_changed = true;
            }
        }

        if (settings().changed() || _input_tags_present || _output_tags_changed) {
            if (const auto forward_parameters = settings().apply_staged_parameters(); !forward_parameters.empty()) {
                std::for_each(_tags_at_output.begin(), _tags_at_output.end(), [&forward_parameters](tag_t &tag) {
                    for (const auto &[key, value] : forward_parameters) {
                        tag.map.insert_or_assign(key, value);
                    }
                });
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

        std::size_t n_samples_to_consume = ports_status.in_samples; // default stride == 0
        if constexpr (node_template_parameters::template contains<PerformStride>) {
            if (stride != 0_UZ) {
                const bool first_time_stride = stride_counter == 0;
                if (first_time_stride) {
                    // sample processing are done as usual, ports_status.in_samples samples will be processed
                    if (stride.value > stride_counter + ports_status.in_available) { // stride can not be consumed at once -> start stride_counter
                        stride_counter += ports_status.in_available;
                        n_samples_to_consume = ports_status.in_available;
                    } else { // if the stride can be consumed at once -> no stride_counter is needed
                        stride_counter       = 0;
                        n_samples_to_consume = stride.value;
                    }
                } else {
                    // |====================|...|====================|==============----| -> ====== is the stride
                    //   ^first                    ^we are here (1)  or ^here (2)
                    // if it is not the "first time" stride -> just consume (1) all samples or (2) missing rest of the samples
                    // forward tags but no additional sample processing are done ->return
                    if (stride.value > stride_counter + ports_status.in_available) {
                        stride_counter += ports_status.in_available;
                        n_samples_to_consume = ports_status.in_available;
                    } else { // stride is at the end -> reset stride_counter
                        n_samples_to_consume = stride.value - stride_counter;
                        stride_counter       = 0;
                    }
                    const bool success = consume_readers(self(), n_samples_to_consume);
                    forward_tags();
                    return { requested_work, n_samples_to_consume, success ? work_return_status_t::OK : work_return_status_t::ERROR };
                }
            }
        }

        const auto input_spans = meta::tuple_transform(
                [&self = self(), sync_in_samples = self().ports_status.in_samples]<typename PortOrCollection>( PortOrCollection &input_port_or_collection) noexcept {
                    auto in_samples          = sync_in_samples;

                    auto process_single_port = [&in_samples]<typename Port>(Port &&port) {
                        if constexpr (std::remove_cvref_t<Port>::synchronous) {
                            return std::forward<Port>(port).streamReader().get(in_samples);
                        } else {
                            return std::addressof(std::forward<Port>(port).streamReader());
                        }
                    };
                    if constexpr (traits::port::is_port_v<PortOrCollection>) {
                        return process_single_port(input_port_or_collection);
                    } else {
                        using value_span = decltype(process_single_port(std::declval<typename PortOrCollection::value_type>()));
                        std::vector<value_span> result;
                        std::transform(input_port_or_collection.begin(), input_port_or_collection.end(), std::back_inserter(result), process_single_port);
                        return result;
                    }
                },
                input_ports(&self()));
        auto writers_tuple = meta::tuple_transform(
                [&self = self(), sync_out_samples = ports_status.out_samples]<typename PortOrCollection>( PortOrCollection &output_port_or_collection) noexcept {
                    auto out_samples         = sync_out_samples;

                    auto process_single_port = [&out_samples]<typename Port>(Port &&port) {
                        if constexpr (std::remove_cvref_t<Port>::synchronous) {
                            return std::forward<Port>(port).streamWriter().reserve_output_range(out_samples);
                        } else {
                            return std::addressof(std::forward<Port>(port).streamWriter());
                        }
                    };
                    if constexpr (traits::port::is_port_v<PortOrCollection>) {
                        return process_single_port(output_port_or_collection);
                    } else {
                        using value_span = decltype(process_single_port(std::declval<typename PortOrCollection::value_type>()));
                        std::vector<value_span> result;
                        std::transform(output_port_or_collection.begin(), output_port_or_collection.end(), std::back_inserter(result), process_single_port);
                        return result;
                    }
                },
                output_ports(&self()));

        if constexpr (HasProcessBulkFunction<Derived>) {
            // cannot use std::apply because it requires tuple_cat(input_spans, writers_tuple). The latter doesn't work because writers_tuple isn't copyable.
            const work_return_status_t ret = [&]<std::size_t... InIdx, std::size_t... OutIdx>(std::index_sequence<InIdx...>, std::index_sequence<OutIdx...>) {
                return self().process_bulk(std::get<InIdx>(input_spans)..., std::get<OutIdx>(writers_tuple)...);
            }(std::make_index_sequence<traits::node::input_ports<Derived>::size>(), std::make_index_sequence<traits::node::output_ports<Derived>::size>());

            write_to_outputs(ports_status.out_samples, writers_tuple);
            const bool success = consume_readers(self(), n_samples_to_consume);
            forward_tags();
            return { requested_work, ports_status.in_samples, success ? ret : work_return_status_t::ERROR };

        } else if constexpr (HasProcessOneFunction<Derived>) {
            if (ports_status.in_samples != ports_status.out_samples)
                throw std::runtime_error(fmt::format("N input samples ({}) does not equal to N output samples ({}) for process_one() method.", ports_status.in_samples, ports_status.out_samples));
            // handle process_one(...)
            using input_simd_types  = meta::simdize<typename input_types::template apply<std::tuple>>;
            using output_simd_types = meta::simdize<typename output_types::template apply<std::tuple>>;

            std::integral_constant<std::size_t, (meta::simdize_size_v<input_simd_types> == 0 ? std::size_t(stdx::simd_abi::max_fixed_size<double>)
                                                                                             : std::min(std::size_t(stdx::simd_abi::max_fixed_size<double>), meta::simdize_size_v<input_simd_types> * 4))>
                    width{};

            if constexpr ((is_sink_node or meta::simdize_size_v<output_simd_types> != 0) and ((is_source_node and requires(Derived &d) {
                                                                                                  { d.process_one_simd(width) };
                                                                                              }) or (meta::simdize_size_v<input_simd_types> != 0 and traits::node::can_process_one_simd<Derived>))) {
                // SIMD loop
                std::size_t i = 0;
                for (; i + width <= ports_status.in_samples; i += width) {
                    const auto &results = simdize_tuple_load_and_apply(width, input_spans, i, [&](const auto &...input_simds) { return invoke_process_one_simd(i, width, input_simds...); });
                    meta::tuple_for_each([i](auto &output_range, const auto &result) { result.copy_to(output_range.data() + i, stdx::element_aligned); }, writers_tuple, results);
                }
                simd_epilogue(width, [&](auto w) {
                    if (i + w <= ports_status.in_samples) {
                        const auto results = simdize_tuple_load_and_apply(w, input_spans, i, [&](auto &&...input_simds) { return invoke_process_one_simd(i, w, input_simds...); });
                        meta::tuple_for_each([i](auto &output_range, auto &result) { result.copy_to(output_range.data() + i, stdx::element_aligned); }, writers_tuple, results);
                        i += w;
                    }
                });
            } else {
                // Non-SIMD loop
                for (std::size_t i = 0; i < ports_status.in_samples; ++i) {
                    const auto results = std::apply([this, i](auto &...inputs) { return this->invoke_process_one(i, inputs[i]...); }, input_spans);
                    meta::tuple_for_each([i](auto &output_range, auto &result) { output_range[i] = std::move(result); }, writers_tuple, results);
                }
            }

            write_to_outputs(ports_status.out_samples, writers_tuple);

            const bool success = consume_readers(self(), n_samples_to_consume);

#ifdef _DEBUG
            if (!success) {
                fmt::print("Node {} failed to consume {} values from inputs\n", self().name(), samples_to_process);
            }
#endif
            forward_tags();
            return { requested_work, ports_status.in_samples, success ? work_return_status_t::OK : work_return_status_t::ERROR };
        } // process_one(...) handling
        //        else {
        //            static_assert(fair::meta::always_false<Derived>, "neither process_bulk(...) nor process_one(...) implemented");
        //        }
        return { requested_work, 0_UZ, work_return_status_t::ERROR };
    } // end: work_return_t work_internal() noexcept { ..}

public:
    work_return_status_t
    invokeWork()
        requires(blockingIO)
    {
        auto [work_requested, work_done, last_status] = work_internal(ioRequestedWork.load(std::memory_order_relaxed));
        ioWorkDone.increment(work_requested, work_done);
        if (auto [incWorkRequested, incWorkDone] = ioWorkDone.get(); last_status == work_return_status_t::DONE && incWorkDone > 0) {
            // finished local iteration but need to report more work to be done until
            // external scheduler loop acknowledged all samples being processed
            // via the 'ioWorkDone.getAndReset()' call
            last_status = work_return_status_t::OK;
        }
        ioLastWorkStatus.exchange(last_status, std::memory_order_relaxed);

        std::ignore = progress->incrementAndGet();
        progress->notify_all();
        return last_status;
    }

    /**
     * @brief Process as many samples as available and compatible with the internal boundary requirements or limited by 'requested_work`
     *
     * @param requested_work: usually the processed number of input samples, but could be any other metric as long as
     * requested_work limit as an affine relation with the returned performed_work.
     * @return { requested_work, performed_work, status}
     */
    work_return_t
    work(std::size_t requested_work = std::numeric_limits<std::size_t>::max()) noexcept {
        if constexpr (blockingIO) {
            constexpr bool useIoThread = std::disjunction_v<std::is_same<BlockingIO<true>, Arguments>...>;
            std::atomic_store_explicit(&ioRequestedWork, requested_work, std::memory_order_release);
            if (bool expectedThreadState = false; ioThreadShallRun.compare_exchange_strong(expectedThreadState, true, std::memory_order_acq_rel)) {
                if constexpr (useIoThread) { // use graph-provided ioThreadPool
                    ioThreadPool->execute([this]() {
#ifdef _DEBUG
                        fmt::print("starting thread for {} count {}\n", name, ioThreadRunning.fetch_add(1));
#else
                        ioThreadRunning.fetch_add(1);
#endif
                        for (int retryCount = 2; ioThreadShallRun && retryCount > 0; retryCount--) {
                            while (ioThreadShallRun && retryCount) {
                                if (invokeWork() == work_return_status_t::DONE) {
                                    break;
                                } else {
                                    // processed data before shutting down wait (see below) and retry (here: once)
                                    retryCount = 2;
                                }
                            }
                            // delayed shut-down in case there are more tasks to be processed
                            std::this_thread::sleep_for(std::chrono::milliseconds(10));
                        }
                        std::atomic_store_explicit(&ioThreadShallRun, false, std::memory_order_release);
                        ioThreadShallRun.notify_all();
#ifdef _DEBUG
                        fmt::print("shutting down thread for {} count {}\n", name, ioThreadRunning.fetch_sub(1));
#else
                        ioThreadRunning.fetch_sub(1);
#endif
                        ioThreadRunning.notify_all();
                    });
                } else {
                    // let user call '' explicitly
                }
            }
            const work_return_status_t lastStatus                 = ioLastWorkStatus.exchange(work_return_status_t::OK, std::memory_order_relaxed);
            const auto &[accumulatedRequestedWork, performedWork] = ioWorkDone.getAndReset();
            // TODO: this is just "working" solution for deadlock with emscripten, need to be investigated further
#if defined(__EMSCRIPTEN__)
            std::this_thread::sleep_for(std::chrono::nanoseconds(1));
#endif
            return { accumulatedRequestedWork, performedWork, performedWork > 0 ? work_return_status_t::OK : lastStatus };
        } else {
            return work_internal(requested_work);
        }
    }
};

template<typename Derived, typename... Arguments>
inline std::atomic_size_t node<Derived, Arguments...>::_unique_id_counter{ 0_UZ };
} // namespace fair::graph

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, typename... Arguments), (fair::graph::node<T, Arguments...>), numerator, denominator, stride, unique_name, name, meta_information);

namespace fair::graph {

/**
 * @brief a short human-readable/markdown description of the node -- content is not contractual and subject to change
 */
template<NodeType Node>
[[nodiscard]] /*constexpr*/ std::string
node_description() noexcept {
    using DerivedNode          = typename Node::derived_t;
    using ArgumentList         = typename Node::node_template_parameters;
    using Description          = typename ArgumentList::template find_or_default<is_doc, EmptyDoc>;
    using SupportedTypes       = typename ArgumentList::template find_or_default<is_supported_types, DefaultSupportedTypes>;
    constexpr bool is_blocking = ArgumentList::template contains<BlockingIO<true>> || ArgumentList::template contains<BlockingIO<false>>;

    // re-enable once string and constexpr static is supported by all compilers
    /*constexpr*/ std::string ret = fmt::format("# {}\n{}\n{}\n**supported data types:**", //
                                                fair::meta::type_name<DerivedNode>(), Description::value._data,
                                                is_blocking ? "**BlockingIO**\n_i.e. potentially non-deterministic/non-real-time behaviour_\n" : "");
    fair::meta::typelist<SupportedTypes>::template apply_func([&](std::size_t index, auto &&t) {
        std::string type_name = fair::meta::type_name<decltype(t)>();
        ret += fmt::format("{}:{} ", index, type_name);
    });
    ret += fmt::format("\n**Parameters:**\n");
    if constexpr (refl::is_reflectable<DerivedNode>()) {
        for_each(refl::reflect<DerivedNode>().members, [&](auto member) {
            using RawType = std::remove_cvref_t<typename decltype(member)::value_type>;
            using Type    = unwrap_if_wrapped_t<RawType>;

            if constexpr (is_readable(member) && (std::integral<Type> || std::floating_point<Type> || std::is_same_v<Type, std::string>) ) {
                if constexpr (is_annotated<RawType>()) {
                    const std::string type_name   = refl::detail::get_type_name<Type>().str();
                    const std::string member_name = get_display_name_const(member).str();
                    ret += fmt::format("{}{:10} {:<20} - annotated info: {} unit: [{}] documentation: {}{}\n",
                                       RawType::visible() ? "" : "_", //
                                       type_name,
                                       member_name, //
                                       RawType::description(), RawType::unit(),
                                       RawType::documentation(), //
                                       RawType::visible() ? "" : "_");
                } else {
                    const std::string type_name   = refl::detail::get_type_name<Type>().str();
                    const std::string member_name = get_display_name_const(member).str();
                    ret += fmt::format("_{:10} {}_\n", type_name, member_name);
                }
            }
        });
    }
    ret += fmt::format("\n~~Ports:~~\ntbd.");
    return ret;
}

template<typename Node>
concept source_node = traits::node::can_process_one<Node> and traits::node::template output_port_types<Node>::size > 0;

static_assert(not source_node<int>);

template<typename Node>
concept sink_node = traits::node::can_process_one<Node> and traits::node::template input_port_types<Node>::size > 0;

static_assert(not sink_node<int>);

template<source_node Left, sink_node Right, std::size_t OutId, std::size_t InId>
class merged_node : public node<merged_node<Left, Right, OutId, InId>, meta::concat<typename traits::node::input_ports<Left>, meta::remove_at<InId, typename traits::node::input_ports<Right>>>,
                                meta::concat<meta::remove_at<OutId, typename traits::node::output_ports<Left>>, typename traits::node::output_ports<Right>>> {
    static std::atomic_size_t _unique_id_counter;

public:
    const std::size_t unique_id   = _unique_id_counter++;
    const std::string unique_name = fmt::format("merged_node<{}:{},{}:{}>#{}", fair::meta::type_name<Left>(), OutId, fair::meta::type_name<Right>(), InId, unique_id);

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
    apply_left(std::size_t offset, auto &&input_tuple) noexcept {
        return [&]<std::size_t... Is>(std::index_sequence<Is...>) {
            return invoke_process_one_with_or_without_offset(left, offset, std::get<Is>(std::forward<decltype(input_tuple)>(input_tuple))...);
        }(std::make_index_sequence<I>());
    }

    template<std::size_t I, std::size_t J>
    constexpr auto
    apply_right(std::size_t offset, auto &&input_tuple, auto &&tmp) noexcept {
        return [&]<std::size_t... Is, std::size_t... Js>(std::index_sequence<Is...>, std::index_sequence<Js...>) {
            constexpr std::size_t first_offset  = traits::node::input_port_types<Left>::size;
            constexpr std::size_t second_offset = traits::node::input_port_types<Left>::size + sizeof...(Is);
            static_assert(second_offset + sizeof...(Js) == std::tuple_size_v<std::remove_cvref_t<decltype(input_tuple)>>);
            return invoke_process_one_with_or_without_offset(right, offset, std::get<first_offset + Is>(std::forward<decltype(input_tuple)>(input_tuple))..., std::forward<decltype(tmp)>(tmp),
                                                             std::get<second_offset + Js>(input_tuple)...);
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
        requires traits::node::can_process_one_simd<Left> and traits::node::can_process_one_simd<Right>
    constexpr meta::simdize<return_type, meta::simdize_size_v<std::tuple<Ts...>>>
    process_one(std::size_t offset, const Ts &...inputs) {
        static_assert(traits::node::output_port_types<Left>::size == 1, "TODO: SIMD for multiple output ports not implemented yet");
        return apply_right<InId, traits::node::input_port_types<Right>::size() - InId - 1>(offset, std::tie(inputs...),
                                                                                           apply_left<traits::node::input_port_types<Left>::size()>(offset, std::tie(inputs...)));
    }

    constexpr auto
    process_one_simd(std::size_t offset, auto N)
        requires traits::node::can_process_one_simd<Right>
    {
        if constexpr (requires(Left &l) {
                          { l.process_one_simd(offset, N) };
                      }) {
            return invoke_process_one_with_or_without_offset(right, offset, left.process_one_simd(offset, N));
        } else if constexpr (requires(Left &l) {
                                 { l.process_one_simd(N) };
                             }) {
            return invoke_process_one_with_or_without_offset(right, offset, left.process_one_simd(N));
        } else {
            using LeftResult = typename traits::node::return_type<Left>;
            using V          = meta::simdize<LeftResult, N>;
            alignas(stdx::memory_alignment_v<V>) LeftResult tmp[V::size()];
            for (std::size_t i = 0; i < V::size(); ++i) {
                tmp[i] = invoke_process_one_with_or_without_offset(left, offset + i);
            }
            return invoke_process_one_with_or_without_offset(right, offset, V(tmp, stdx::vector_aligned));
        }
    }

    template<typename... Ts>
    // Nicer error messages for the following would be good, but not at the expense of breaking can_process_one_simd.
        requires(input_port_types::template are_equal<std::remove_cvref_t<Ts>...>)
    constexpr return_type
    process_one(std::size_t offset, Ts &&...inputs) {
        // if (sizeof...(Ts) == 0) we could call `return process_one_simd(integral_constant<size_t, width>)`. But if
        // the caller expects to process *one* sample (no inputs for the caller to explicitly
        // request simd), and we process more, we risk inconsistencies.
        if constexpr (traits::node::output_port_types<Left>::size == 1) {
            // only the result from the right node needs to be returned
            return apply_right<InId, traits::node::input_port_types<Right>::size() - InId - 1>(offset, std::forward_as_tuple(std::forward<Ts>(inputs)...),
                                                                                               apply_left<traits::node::input_port_types<Left>::size()>(offset, std::forward_as_tuple(
                                                                                                                                                                        std::forward<Ts>(inputs)...)));

        } else {
            // left produces a tuple
            auto left_out  = apply_left<traits::node::input_port_types<Left>::size()>(offset, std::forward_as_tuple(std::forward<Ts>(inputs)...));
            auto right_out = apply_right<InId, traits::node::input_port_types<Right>::size() - InId - 1>(offset, std::forward_as_tuple(std::forward<Ts>(inputs)...),
                                                                                                         std::move(std::get<OutId>(left_out)));

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
    work(std::size_t requested_work) noexcept {
        return base::work(requested_work);
    }
};

template<source_node Left, sink_node Right, std::size_t OutId, std::size_t InId>
inline std::atomic_size_t merged_node<Left, Right, OutId, InId>::_unique_id_counter{ 0_UZ };

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
    constexpr int OutIdUnchecked = meta::indexForName<OutName, typename traits::node::output_ports<A>>();
    constexpr int InIdUnchecked  = meta::indexForName<InName, typename traits::node::input_ports<B>>();
    static_assert(OutIdUnchecked != -1);
    static_assert(InIdUnchecked != -1);
    constexpr auto OutId = static_cast<std::size_t>(OutIdUnchecked);
    constexpr auto InId  = static_cast<std::size_t>(InIdUnchecked);
    static_assert(std::same_as<typename traits::node::output_port_types<std::remove_cvref_t<A>>::template at<OutId>, typename traits::node::input_port_types<std::remove_cvref_t<B>>::template at<InId>>,
                  "Port types do not match");
    return merged_node<std::remove_cvref_t<A>, std::remove_cvref_t<B>, OutId, InId>{ std::forward<A>(a), std::forward<B>(b) };
}

#if !DISABLE_SIMD
namespace test { // TODO: move to dedicated tests

struct copy : public node<copy> {
    PortIn<float>  in;
    PortOut<float> out;

public:
    template<meta::t_or_simd<float> V>
    [[nodiscard]] constexpr V
    process_one(const V &a) const noexcept {
        return a;
    }
};
} // namespace test
#endif
} // namespace fair::graph

#if !DISABLE_SIMD
ENABLE_REFLECTION(fair::graph::test::copy, in, out);
#endif

namespace fair::graph {

#if !DISABLE_SIMD
namespace test {
static_assert(traits::node::input_port_types<copy>::size() == 1);
static_assert(std::same_as<traits::node::return_type<copy>, float>);
static_assert(traits::node::can_process_one_scalar<copy>);
static_assert(traits::node::can_process_one_simd<copy>);
static_assert(traits::node::can_process_one_scalar_with_offset<decltype(merge_by_index<0, 0>(copy(), copy()))>);
static_assert(traits::node::can_process_one_simd_with_offset<decltype(merge_by_index<0, 0>(copy(), copy()))>);
static_assert(source_node<copy>);
static_assert(sink_node<copy>);
static_assert(source_node<decltype(merge_by_index<0, 0>(copy(), copy()))>);
static_assert(sink_node<decltype(merge_by_index<0, 0>(copy(), copy()))>);
} // namespace test
#endif

namespace detail {
template<typename... Types>
struct node_parameters {
    template<template<typename...> typename NodeTemplate, typename RegisterInstance>
    void
    register_on(RegisterInstance *plugin_instance, std::string node_type) const {
        plugin_instance->template add_node_type<NodeTemplate, Types...>(node_type);
    }
};

template<template<typename...> typename NodeTemplate, typename... NodeParameters>
struct register_node {
    template<typename RegisterInstance>
    register_node(RegisterInstance *plugin_instance, std::string node_type) {
        auto add_node_type = [&]<typename Type> {
            if constexpr (meta::is_instantiation_of<Type, node_parameters>) {
                Type().template register_on<NodeTemplate>(plugin_instance, node_type);
            } else {
                plugin_instance->template add_node_type<NodeTemplate, Type>(node_type);
            }
        };
        ((add_node_type.template operator()<NodeParameters>()), ...);
    }
};
} // namespace detail

} // namespace fair::graph

#ifdef FMT_FORMAT_H_

template<>
struct fmt::formatter<fair::graph::work_return_status_t> {
    static constexpr auto
    parse(const format_parse_context &ctx) {
        const auto it = ctx.begin();
        if (it != ctx.end() && *it != '}') throw format_error("invalid format");
        return it;
    }

    template<typename FormatContext>
    auto
    format(const fair::graph::work_return_status_t &status, FormatContext &ctx) {
        using enum fair::graph::work_return_status_t;
        switch (status) {
        case ERROR: return fmt::format_to(ctx.out(), "ERROR");
        case INSUFFICIENT_OUTPUT_ITEMS: return fmt::format_to(ctx.out(), "INSUFFICIENT_OUTPUT_ITEMS");
        case INSUFFICIENT_INPUT_ITEMS: return fmt::format_to(ctx.out(), "INSUFFICIENT_INPUT_ITEMS");
        case DONE: return fmt::format_to(ctx.out(), "DONE");
        case OK: return fmt::format_to(ctx.out(), "OK");
        default: return fmt::format_to(ctx.out(), "UNKNOWN");
        }
        return fmt::format_to(ctx.out(), "UNKNOWN");
    }
};

template<>
struct fmt::formatter<fair::graph::work_return_t> {
    static constexpr auto
    parse(const format_parse_context &ctx) {
        const auto it = ctx.begin();
        if (it != ctx.end() && *it != '}') throw format_error("invalid format");
        return it;
    }

    template<typename FormatContext>
    auto
    format(const fair::graph::work_return_t &work_return, FormatContext &ctx) {
        return fmt::format_to(ctx.out(), "requested_work: {}, performed_work: {}, status: {}", work_return.requested_work, work_return.performed_work, fmt::format("{}", work_return.status));
    }
};

#endif // FMT_FORMAT_H_

#endif // include guard
