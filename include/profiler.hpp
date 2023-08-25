#ifndef GNURADIO_PROFILER_H
#define GNURADIO_PROFILER_H

#include "circular_buffer.hpp"

#include <fmt/format.h>

#include <chrono>
#include <fstream>
#include <string>
#include <string_view>
#include <thread>
#include <variant>

#include <unistd.h>

namespace fair::graph::profiling {

using arg_value = std::pair<std::string, std::variant<std::string, int, double>>;

namespace detail {
using clock      = std::chrono::high_resolution_clock;
using time_point = clock::time_point;

enum class EventType {
    DurationBegin = 'B', // Duration Event (begin).
    DurationEnd   = 'E', // Duration Event (end).
    Complete      = 'X', // Complete Event.
    Instant       = 'I', // Instant Event.
    Counter       = 'C', // Counter Event.
    AsyncStart    = 'b', // Async Event (start).
    AsyncStep     = 'n', // Async Event (step).
    AsyncEnd      = 'e', // Async Event (end).
    FlowStart     = 's', // Flow Event (start).
    FlowStep      = 't', // Flow Event (step).
    FlowEnd       = 'f'  // Flow Event (end).
};

inline std::string
format_args(const std::vector<arg_value> &args) {
    if (args.empty()) {
        return "{}";
    }
    std::string r = "{";
    for (std::size_t i = 0; i < args.size(); ++i) {
        if (i > 0) {
            r += ",";
        }
        if (std::holds_alternative<std::string>(args[i].second)) {
            r += fmt::format("\"{}\":\"{}\"", args[i].first, std::get<std::string>(args[i].second));
        } else if (std::holds_alternative<double>(args[i].second)) {
            r += fmt::format("\"{}\":{}", args[i].first, std::get<double>(args[i].second));
        } else {
            r += fmt::format("\"{}\":{}", args[i].first, std::get<int>(args[i].second));
        }
    }
    r += "}";
    return r;
}

struct TraceEvent {
    std::thread::id           thread_id;
    std::string               name; // Event name.
    EventType                 type; // Event type.
    std::chrono::microseconds ts;   // Timestamp
    std::chrono::microseconds dur;  // Duration of the event, for 'X' type.
    std::string               id;   // ID for matching async or flow events.
    std::string               cat;  // Event categories.
    std::vector<arg_value>    args; // Event arguments.
    char filler[104]; // fills up to power of 2 size, see static_assert below

    // Function to format a TraceEvent into JSON format.
    std::string
    toJSON(int pid, int tid) const {
        using enum EventType;
        switch (type) {
        case DurationBegin:
        case DurationEnd:
        case Counter:
        case Instant:
            return fmt::format(R"({{"name": "{}", "ph": "{}", "ts": {}, "pid": {}, "tid": {}, "cat": "{}", "args": {}}})", name, static_cast<char>(type), ts.count(), pid, tid, cat, format_args(args));
        case Complete:
            return fmt::format(R"({{"name": "{}", "ph": "X", "ts": {}, "pid": {}, "tid": {}, "dur": {}, "cat": "{}", "args": {}}})", name, ts.count(), pid, tid, dur.count(), cat, format_args(args));
        case AsyncStart:
        case AsyncStep:
        case AsyncEnd:
            return fmt::format(R"({{"name": "{}", "ph": "{}", "ts": {}, "pid": {}, "tid": {}, "id": "{}", "cat": "{}", "args": {}}})", name, static_cast<char>(type), ts.count(), pid, tid, id, cat,
                               format_args(args));
        default: // TODO
            return fmt::format(R"({{"name": "{}", "ph": "{}", "ts": {}, "pid": {}, "tid": {}, "cat": "{}", "args": {}}})", name, static_cast<char>(type), ts.count(), pid, tid, cat,
                               format_args(args));
        }

        return {};
    }

    TraceEvent()                       = default;
    ~TraceEvent()                      = default;
    TraceEvent(TraceEvent &&) noexcept = default;
    TraceEvent &
    operator=(TraceEvent &&) noexcept
            = default;

    // the default implementations slow down publish() even if never called at runtime (branch prediction going wrong for the non-mmap case?)
    // seen with -O3 on gcc 12.3.0
    TraceEvent(const TraceEvent &) { throw std::logic_error("unexpected copy"); }

    TraceEvent &
    operator=(const TraceEvent &) {
        throw std::logic_error("unexpected assignment");
    }
};

// size of TraceEvent must be power of 2, otherwise circular_buffer doesn't work correctly
static_assert(std::has_single_bit(sizeof(TraceEvent)));

} // namespace detail

template<typename T>
concept SimpleEvent = requires(T e) {
    { e.finish() } -> std::same_as<void>;
};

template<typename T>
concept StepEvent = requires(T e) {
    { e.finish() } -> std::same_as<void>;
    { e.step() } -> std::same_as<void>;
};

template<typename T>
concept ProfilerHandler = requires(T h, std::string_view name, std::string_view categories, std::initializer_list<arg_value> args) {
    { h.start_complete_event(name, categories, args) } -> SimpleEvent;
    { h.start_async_event(name, categories, args) } -> StepEvent;
    { h.instant_event(name, categories, args) } -> std::same_as<void>;
    { h.counter_event(name, categories, args) } -> std::same_as<void>;
};

template<typename T>
concept Profiler = requires(T p) {
    { p.reset() } -> std::same_as<void>;
    { p.for_this_thread() } -> ProfilerHandler;
};

enum class output_mode_t { StdOut, File };

struct options {
    std::string   output_file;
    output_mode_t output_mode = output_mode_t::File;
};

namespace null {
class step_event {
public:
    constexpr void
    step() const noexcept {}

    constexpr void
    finish() const noexcept {}
};

class simple_event {
public:
    constexpr void
    finish() const noexcept {}
};

class handler {
public:
    constexpr void
    instant_event(std::string_view name, std::string_view categories = {}, std::initializer_list<arg_value> args = {}) const noexcept {
        std::ignore = name;
        std::ignore = categories;
        std::ignore = args;
    }

    constexpr void
    counter_event(std::string_view name, std::string_view categories = {}, std::initializer_list<arg_value> args = {}) const noexcept {
        std::ignore = name;
        std::ignore = categories;
        std::ignore = args;
    }

    [[nodiscard]] constexpr simple_event
    start_complete_event(std::string_view name, std::string_view categories = {}, std::initializer_list<arg_value> args = {}) const noexcept {
        std::ignore = name;
        std::ignore = categories;
        std::ignore = args;
        return simple_event{};
    }

    [[nodiscard]] constexpr step_event
    start_async_event(std::string_view name, std::string_view categories = {}, std::initializer_list<arg_value> args = {}) const noexcept {
        std::ignore = name;
        std::ignore = categories;
        std::ignore = args;
        return step_event{};
    }
};

class profiler {
    handler _handler;

public:
    constexpr explicit profiler(const options & = {}) {}

    constexpr void
    reset() const {}

    constexpr handler &
    for_this_thread() {
        return _handler;
    }
};

} // namespace null

template<typename Handler>
class complete_event {
    Handler               &_handler;
    std::string            _name;
    std::string            _categories;
    std::vector<arg_value> _args;
    bool                   _finished = false;
    detail::time_point     _start    = detail::clock::now();

public:
    explicit complete_event(Handler &handler, std::string_view name, std::string_view categories, std::initializer_list<arg_value> args)
        : _handler{ handler }, _name{ name }, _categories{ categories }, _args{ args } {}

    ~complete_event() { finish(); }

    complete_event(const complete_event &) = delete;
    complete_event &
    operator=(const complete_event &)
            = delete;
    complete_event(complete_event &&) noexcept = default;
    complete_event &
    operator=(complete_event &&) noexcept
            = default;

    void
    finish() noexcept {
        if (_finished) {
            return;
        }
        const auto elapsed = detail::clock::now() - _start;
        auto       r       = _handler.reserve_event();
        r[0].name          = _name;
        r[0].type          = detail::EventType::Complete;
        r[0].dur           = std::chrono::duration_cast<std::chrono::microseconds>(elapsed);
        r[0].cat           = _categories;
        r[0].args          = _args;
        r.publish(1);
        _finished = true;
    }
};

template<typename Handler>
class async_event {
    Handler    &_handler;
    bool        _finished = false;
    std::string _id;
    std::string _name;

public:
    explicit async_event(Handler &handler, std::string_view name, std::string_view categories = {}, std::initializer_list<arg_value> args = {}) : _handler(handler), _name{ name } {
        // TODO generate ID or pass as argument?
        post_event(detail::EventType::AsyncStart, categories, args);
    }

    ~async_event() { finish(); }

    async_event(const async_event &) = delete;
    async_event &
    operator=(const async_event &)
            = delete;
    async_event(async_event &&) noexcept = default;
    async_event &
    operator=(async_event &&) noexcept
            = default;

    void
    step() noexcept {
        post_event(detail::EventType::AsyncStep);
    }

    void
    finish() noexcept {
        if (_finished) {
            return;
        }
        post_event(detail::EventType::AsyncEnd);
        _finished = true;
    }

private:
    void
    post_event(detail::EventType type, std::string_view categories = {}, std::initializer_list<arg_value> args = {}) noexcept {
        auto r    = _handler.reserve_event();
        r[0].name = _name;
        r[0].type = type;
        r[0].id   = _id;
        r[0].cat  = std::string{ categories };
        r[0].args = args;
        r.publish(1);
    }
};

template<typename Profiler, typename WriterType>
class handler {
    using this_t = handler<Profiler, WriterType>;
    Profiler  &_profiler;
    WriterType _writer;

public:
    explicit handler(Profiler &profiler, WriterType &&writer) : _profiler(profiler), _writer{ std::move(writer) } {}

    handler(const this_t &) = delete;
    this_t &
    operator=(const this_t &)
            = delete;
    handler(this_t &&) noexcept = delete;
    this_t &
    operator=(this_t &&) noexcept
            = delete;

    auto
    reserve_event() noexcept {
        const auto elapsed = detail::clock::now() - _profiler.start();
        auto       r       = _writer.reserve_output_range(1);
        r[0].thread_id     = std::this_thread::get_id();
        r[0].ts            = std::chrono::duration_cast<std::chrono::microseconds>(elapsed);
        return r;
    }

    void
    instant_event(std::string_view name, std::string_view categories = {}, std::initializer_list<arg_value> args = {}) noexcept {
        auto r    = reserve_event();
        r[0].name = std::string{ name };
        r[0].type = detail::EventType::Instant;
        r[0].cat  = std::string{ categories };
        r[0].args = args;
        r.publish(1);
    }

    void
    counter_event(std::string_view name, std::string_view categories, std::initializer_list<arg_value> args = {}) noexcept {
        auto r    = reserve_event();
        r[0].name = std::string{ name };
        r[0].type = detail::EventType::Counter;
        r[0].cat  = std::string{ categories };
        r[0].args = args;
        r.publish(1);
    }

    [[nodiscard]] complete_event<this_t>
    start_complete_event(std::string_view name, std::string_view categories = {}, std::initializer_list<arg_value> args = {}) noexcept {
        return complete_event<this_t>{ *this, name, categories, args };
    }

    [[nodiscard]] async_event<this_t>
    start_async_event(std::string_view name, std::string_view categories = {}, std::initializer_list<arg_value> args = {}) noexcept {
        return async_event<this_t>{ *this, name, categories, args };
    }
};

class profiler {
    gr::circular_buffer<detail::TraceEvent> _buffer;
    using WriterType  = decltype(_buffer.new_writer());
    using HandlerType = handler<profiler, WriterType>;
    std::mutex                             _handlers_lock;
    std::map<std::thread::id, HandlerType> _handlers;
    std::atomic<bool>                      _finished = false;
    decltype(_buffer.new_reader())         _reader   = _buffer.new_reader();
    std::thread                            _event_handler;
    detail::time_point                     _start = detail::clock::now();

public:
    explicit profiler(const options &options = {}) : _buffer(500000) {
        _event_handler = std::thread([options, &reader = _reader, &finished = _finished]() {
            auto          file_name = options.output_file;
            std::ofstream out_file;
            if (file_name.empty() && options.output_mode == output_mode_t::File) {
                static std::atomic<int> counter = 0;
                file_name                       = fmt::format("profile.{}.{}.trace", getpid(), counter++);
                out_file                        = std::ofstream(file_name, std::ios::out | std::ios::binary);
            }

            std::ostream &out_stream = options.output_mode == output_mode_t::File ? out_file : std::cout;
            const int     pid        = getpid();

            // assign numerical values to threads as we see them
            std::unordered_map<std::thread::id, int> mapped_threads;
            bool                                     seen_finished = false;

            fmt::print(out_stream, "[\n");
            bool is_first = true;
            while (!seen_finished) {
                seen_finished = finished;
                while (reader.available() > 0) {
                    auto event = reader.get(1);
                    auto it    = mapped_threads.find(event[0].thread_id);
                    if (it == mapped_threads.end()) {
                        it = mapped_threads.emplace(event[0].thread_id, static_cast<int>(mapped_threads.size())).first;
                    }
                    if (!is_first) {
                        fmt::print(out_stream, ",\n{}", event[0].toJSON(pid, it->second));
                    } else {
                        fmt::print(out_stream, "{}", event[0].toJSON(pid, it->second));
                    }
                    is_first    = false;
                    std::ignore = reader.consume(1);
                }
            }
            fmt::print(out_stream, "\n]\n");
        });

        reset();
    }

    ~profiler() {
        _finished = true;
        _event_handler.join();
    }

    detail::time_point
    start() const {
        return _start;
    }

    void
    reset() noexcept {
        _start = detail::clock::now();
    }

    handler<profiler, WriterType> &
    for_this_thread() {
        const auto            this_id = std::this_thread::get_id();
        const std::lock_guard lock{ _handlers_lock };
        auto                  it = _handlers.find(this_id);
        if (it == _handlers.end()) {
            auto [new_it, _] = _handlers.try_emplace(this_id, *this, _buffer.new_writer());
            it               = new_it;
        }

        return it->second;
    }
};

} // namespace fair::graph::profiling

#endif // include guard
