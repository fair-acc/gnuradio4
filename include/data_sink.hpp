#ifndef GNURADIO_DATA_SINK_HPP
#define GNURADIO_DATA_SINK_HPP

#include "circular_buffer.hpp"
#include "dataset.hpp"
#include "history_buffer.hpp"
#include "node.hpp"
#include "tag.hpp"

#include <any>
#include <chrono>
#include <limits>

namespace fair::graph {

enum class blocking_mode { NonBlocking, Blocking };

enum class trigger_match_result {
    Matching,    ///< Start a new dataset
    NotMatching, ///< Finish dataset
    Ignore       ///< Ignore tag
};

template<typename T>
class data_sink;

// Until clang-format can handle concepts
// clang-format off

template<typename T, typename V>
concept DataSetCallback = std::invocable<T, DataSet<V>>;

/**
 * Stream callback functions receive the span of data, with optional tags and reference to the sink.
 */
template<typename T, typename V>
concept StreamCallback = std::invocable<T, std::span<const V>> || std::invocable<T, std::span<const V>, std::span<const tag_t>> || std::invocable<T, std::span<const V>, std::span<const tag_t>, const data_sink<V>&>;

/**
 * Used for testing whether a tag should trigger data acquisition.
 *
 * For the 'Triggered' (data window) and 'Snapshot' (single sample) acquisition modes:
 * Stateless predicate to check whether a tag matches the trigger criteria.
 *
 * @code
 * auto matcher = [](const auto &tag) {
 *     const auto is_trigger = ...check if tag is trigger...;
 *     return is_trigger ? trigger_match_result::Matching : trigger_match_result::Ignore;
 * };
 * @endcode
 *
 * For the 'Multiplexed' acquisition mode: Possibly stateful object checking all incoming tags to control which data should be sent
 * to the listener.
 *
 * A new dataset is started when the matcher returns @c Start or @c StopAndStart.
 * A dataset is closed and sent when @c Stop or @StopAndStart is returned.
 *
 * For the multiplexed case, the matcher might be stateful and can rely on being called with each incoming tag exactly once, in the order they arrive.
 *
 * Example:
 *
 * @code
 * // matcher observing three possible tag values, "green", "yellow", "red".
 * // starting a dataset when seeing "green", stopping on "red", starting a new dataset on "yellow"
 * struct color_matcher {
 *     matcher_result operator()(const tag_t &tag) {
 *         if (tag == green || tag == yellow) {
 *             return trigger_match_result::Matching;
 *         }
 *         if (tag == red) {
 *             return trigger_match_result::NotMatching;
 *         }
 *
 *         return trigger_match_result::Ignore;
 *     }
 * };
 * @endcode
 *
 * @see trigger_match_result
 */
template<typename T>
concept TriggerMatcher = requires(T matcher, tag_t tag) {
    { matcher(tag) } -> std::convertible_to<trigger_match_result>;
};

// clang-format on

struct data_sink_query {
    std::optional<std::string> _sink_name;
    std::optional<std::string> _signal_name;

    static data_sink_query
    signal_name(std::string_view name) {
        return { {}, std::string{ name } };
    }

    static data_sink_query
    sink_name(std::string_view name) {
        return { std::string{ name }, {} };
    }
};

class data_sink_registry {
    std::mutex            _mutex;
    std::vector<std::any> _sinks;

public:
    // TODO this shouldn't be a singleton but associated with the flow graph (?)
    // TODO reconsider mutex usage when moving to the graph
    static data_sink_registry &
    instance() {
        static data_sink_registry s_instance;
        return s_instance;
    }

    template<typename T>
    void
    register_sink(data_sink<T> *sink) {
        std::lock_guard lg{ _mutex };
        _sinks.push_back(sink);
    }

    template<typename T>
    void
    unregister_sink(data_sink<T> *sink) {
        std::lock_guard lg{ _mutex };
        std::erase_if(_sinks, [sink](const std::any &v) {
            try {
                return std::any_cast<data_sink<T> *>(v) == sink;
            } catch (...) {
                return false;
            }
        });
    }

    template<typename T>
    std::shared_ptr<typename data_sink<T>::poller>
    get_streaming_poller(const data_sink_query &query, blocking_mode block = blocking_mode::Blocking) {
        std::lock_guard lg{ _mutex };
        auto            sink = find_sink<T>(query);
        return sink ? sink->get_streaming_poller(block) : nullptr;
    }

    template<typename T, TriggerMatcher M>
    std::shared_ptr<typename data_sink<T>::dataset_poller>
    get_trigger_poller(const data_sink_query &query, M matcher, std::size_t pre_samples, std::size_t post_samples, blocking_mode block = blocking_mode::Blocking) {
        std::lock_guard lg{ _mutex };
        auto            sink = find_sink<T>(query);
        return sink ? sink->get_trigger_poller(std::forward<M>(matcher), pre_samples, post_samples, block) : nullptr;
    }

    template<typename T, TriggerMatcher M>
    std::shared_ptr<typename data_sink<T>::dataset_poller>
    get_multiplexed_poller(const data_sink_query &query, M matcher, std::size_t maximum_window_size, blocking_mode block = blocking_mode::Blocking) {
        std::lock_guard lg{ _mutex };
        auto            sink = find_sink<T>(query);
        return sink ? sink->get_multiplexed_poller(std::forward<M>(matcher), maximum_window_size, block) : nullptr;
    }

    template<typename T, TriggerMatcher M>
    std::shared_ptr<typename data_sink<T>::dataset_poller>
    get_snapshot_poller(const data_sink_query &query, M matcher, std::chrono::nanoseconds delay, blocking_mode block = blocking_mode::Blocking) {
        std::lock_guard lg{ _mutex };
        auto            sink = find_sink<T>(query);
        return sink ? sink->get_snapshot_poller(std::forward<M>(matcher), delay, block) : nullptr;
    }

    template<typename T, StreamCallback<T> Callback>
    bool
    register_streaming_callback(const data_sink_query &query, std::size_t max_chunk_size, Callback callback) {
        std::lock_guard lg{ _mutex };
        auto            sink = find_sink<T>(query);
        if (!sink) {
            return false;
        }

        sink->register_streaming_callback(max_chunk_size, std::forward<Callback>(callback));
        return true;
    }

    template<typename T, DataSetCallback<T> Callback, TriggerMatcher M>
    bool
    register_trigger_callback(const data_sink_query &query, M matcher, std::size_t pre_samples, std::size_t post_samples, Callback callback) {
        std::lock_guard lg{ _mutex };
        auto            sink = find_sink<T>(query);
        if (!sink) {
            return false;
        }

        sink->register_trigger_callback(std::forward<M>(matcher), pre_samples, post_samples, std::forward<Callback>(callback));
        return true;
    }

    template<typename T, DataSetCallback<T> Callback, TriggerMatcher M>
    bool
    register_multiplexed_callback(const data_sink_query &query, M matcher, std::size_t maximum_window_size, Callback callback) {
        std::lock_guard lg{ _mutex };
        auto            sink = find_sink<T>(query);
        if (!sink) {
            return false;
        }

        sink->register_multiplexed_callback(std::forward<M>(matcher), maximum_window_size, std::forward<Callback>(callback));
        return true;
    }

    template<typename T, DataSetCallback<T> Callback, TriggerMatcher M>
    bool
    register_snapshot_callback(const data_sink_query &query, M matcher, std::chrono::nanoseconds delay, Callback callback) {
        std::lock_guard lg{ _mutex };
        auto            sink = find_sink<T>(query);
        if (!sink) {
            return false;
        }

        sink->register_snapshot_callback(std::forward<M>(matcher), delay, std::forward<Callback>(callback));
        return true;
    }

private:
    template<typename T>
    data_sink<T> *
    find_sink(const data_sink_query &query) {
        auto matches = [&query](const std::any &v) {
            try {
                auto       sink                = std::any_cast<data_sink<T> *>(v);
                const auto sink_name_matches   = !query._sink_name || *query._sink_name == sink->name();
                const auto signal_name_matches = !query._signal_name || *query._signal_name == sink->signal_name;
                return sink_name_matches && signal_name_matches;
            } catch (...) {
                return false;
            }
        };

        const auto it = std::find_if(_sinks.begin(), _sinks.end(), matches);
        if (it == _sinks.end()) {
            return nullptr;
        }

        return std::any_cast<data_sink<T> *>(*it);
    }
};

namespace detail {
template<typename T>
inline bool
copy_span(std::span<const T> src, std::span<T> dst) {
    assert(src.size() <= dst.size());
    if (src.size() > dst.size()) {
        return false;
    }
    std::copy(src.begin(), src.end(), dst.begin());
    return true;
}

template<typename T>
inline std::optional<T>
get(const property_map &m, const std::string_view &key) {
    const auto it = m.find(key);
    if (it == m.end()) {
        return {};
    }

    return std::get<T>(it->second);
}

} // namespace detail

/**
 * @brief generic data sink for exporting arbitrary-typed streams to non-GR C++ APIs.
 *
 * Each sink registers with a (user-defined/exchangeable) global registry that can be
 * queried by the non-GR caller to find the sink responsible for a given signal name, etc.
 * and either retrieve a poller handler that allows asynchronous data from a different thread,
 * or register a callback that is invoked by the sink if the user-conditions are met.
 *
 * <pre>
 * @code
 *         ╔═══════════════╗
 *    in0 ━╢   data sink   ║                      ┌──── caller ────┐
 * (err0) ━╢ (opt. error)  ║                      │                │
 *         ║               ║  retrieve poller or  │ (custom non-GR │
 *         ║ :signal_name  ║←--------------------→│  user code...) │
 *         ║ :signal_unit  ║  register            │                │
 *         ║ :...          ║  callback function   └───┬────────────┘
 *         ╚═ GR block ═╤══╝                          │
 *                      │                             │
 *                      │                             │
 *                      │      ╭─registry─╮           │
 *            register/ │      ╞══════════╡           │ queries for specific
 *          deregister  ╰─────→│ [sinks]  │←──────────╯ signal_info_t list/criteria
 *                             ╞══════════╡
 *                             ╰──────────╯
 *
 * </pre>
 * Pollers can be configured to be blocking, i.e. blocks the flow-graph
 * if data is not being retrieved in time, or non-blocking, i.e. data being dropped when
 * the user-defined buffer size is full.
 * N.B. due to the nature of the GR scheduler, signals from the same sink are notified
 * synchronuously (/asynchronuously) if handled by the same (/different) sink block.
 *
 * @tparam T input sample type
 */
template<typename T>
class data_sink : public node<data_sink<T>> {
    struct abstract_listener;

    static constexpr std::size_t                   _listener_buffer_size = 65536;
    std::deque<std::unique_ptr<abstract_listener>> _listeners;
    std::mutex                                     _listener_mutex;
    gr::history_buffer<T>                          _history                       = gr::history_buffer<T>(1);
    bool                                           _has_signal_info_from_settings = false;

public:
    Annotated<float, "sample rate", Doc<"signal sample rate">, Unit<"Hz">>           sample_rate = 1.f;
    Annotated<std::string, "signal name", Visible>                                   signal_name = std::string("unknown signal");
    Annotated<std::string, "signal unit", Visible, Doc<"signal's physical SI unit">> signal_unit = std::string("a.u.");
    Annotated<T, "signal min", Doc<"signal physical min. (e.g. DAQ) limit">>         signal_min  = std::numeric_limits<T>::lowest();
    Annotated<T, "signal max", Doc<"signal physical max. (e.g. DAQ) limit">>         signal_max  = std::numeric_limits<T>::max();

    IN<T, std::dynamic_extent, _listener_buffer_size>                                in;

    struct poller {
        // TODO consider whether reusing port<T> here makes sense
        gr::circular_buffer<T>            buffer       = gr::circular_buffer<T>(_listener_buffer_size);
        decltype(buffer.new_reader())     reader       = buffer.new_reader();
        decltype(buffer.new_writer())     writer       = buffer.new_writer();
        gr::circular_buffer<tag_t>        tag_buffer   = gr::circular_buffer<tag_t>(1024);
        decltype(tag_buffer.new_reader()) tag_reader   = tag_buffer.new_reader();
        decltype(tag_buffer.new_writer()) tag_writer   = tag_buffer.new_writer();
        std::size_t                       samples_read = 0; // reader thread
        std::atomic<bool>                 finished     = false;
        std::atomic<std::size_t>          drop_count   = 0;

        template<typename Handler>
        [[nodiscard]] bool
        process(Handler fnc) {
            const auto available = reader.available();
            if (available == 0) {
                return false;
            }

            const auto read_data = reader.get(available);
            if constexpr (requires { fnc(std::span<const T>(), std::span<const tag_t>()); }) {
                const auto tags          = tag_reader.get();
                const auto it            = std::find_if_not(tags.begin(), tags.end(), [until = static_cast<int64_t>(samples_read + available)](const auto &tag) { return tag.index < until; });
                auto       relevant_tags = std::vector<tag_t>(tags.begin(), it);
                for (auto &t : relevant_tags) {
                    t.index -= static_cast<int64_t>(samples_read);
                }
                fnc(read_data, std::span<const tag_t>(relevant_tags));
                std::ignore = tag_reader.consume(relevant_tags.size());
            } else {
                std::ignore = tag_reader.consume(tag_reader.available());
                fnc(read_data);
            }

            std::ignore = reader.consume(available);
            samples_read += available;
            return true;
        }
    };

    struct dataset_poller {
        gr::circular_buffer<DataSet<T>> buffer     = gr::circular_buffer<DataSet<T>>(_listener_buffer_size);
        decltype(buffer.new_reader())   reader     = buffer.new_reader();
        decltype(buffer.new_writer())   writer     = buffer.new_writer();

        std::atomic<bool>               finished   = false;
        std::atomic<std::size_t>        drop_count = 0;

        [[nodiscard]] bool
        process(std::invocable<std::span<DataSet<T>>> auto fnc) {
            const auto available = reader.available();
            if (available == 0) {
                return false;
            }

            const auto read_data = reader.get(available);
            fnc(read_data);
            std::ignore = reader.consume(available);
            return true;
        }
    };

    data_sink() { data_sink_registry::instance().register_sink(this); }

    ~data_sink() {
        stop();
        data_sink_registry::instance().unregister_sink(this);
    }

    void
    init(const property_map & /*old_settings*/, const property_map &new_settings) {
        if (apply_signal_info(new_settings)) {
            _has_signal_info_from_settings = true;
        }
    }

    std::shared_ptr<poller>
    get_streaming_poller(blocking_mode block_mode = blocking_mode::Blocking) {
        std::lock_guard lg(_listener_mutex);
        const auto      block   = block_mode == blocking_mode::Blocking;
        auto            handler = std::make_shared<poller>();
        add_listener(std::make_unique<continuous_listener<fair::meta::null_type>>(handler, block, *this), block);
        return handler;
    }

    template<TriggerMatcher M>
    std::shared_ptr<dataset_poller>
    get_trigger_poller(M matcher, std::size_t pre_samples, std::size_t post_samples, blocking_mode block_mode = blocking_mode::Blocking) {
        const auto      block   = block_mode == blocking_mode::Blocking;
        auto            handler = std::make_shared<dataset_poller>();
        std::lock_guard lg(_listener_mutex);
        add_listener(std::make_unique<trigger_listener<fair::meta::null_type, M>>(std::move(matcher), handler, pre_samples, post_samples, block), block);
        ensure_history_size(pre_samples);
        return handler;
    }

    template<TriggerMatcher M>
    std::shared_ptr<dataset_poller>
    get_multiplexed_poller(M matcher, std::size_t maximum_window_size, blocking_mode block_mode = blocking_mode::Blocking) {
        std::lock_guard lg(_listener_mutex);
        const auto      block   = block_mode == blocking_mode::Blocking;
        auto            handler = std::make_shared<dataset_poller>();
        add_listener(std::make_unique<multiplexed_listener<fair::meta::null_type, M>>(std::move(matcher), maximum_window_size, handler, block), block);
        return handler;
    }

    template<TriggerMatcher M>
    std::shared_ptr<dataset_poller>
    get_snapshot_poller(M matcher, std::chrono::nanoseconds delay, blocking_mode block_mode = blocking_mode::Blocking) {
        const auto      block   = block_mode == blocking_mode::Blocking;
        auto            handler = std::make_shared<dataset_poller>();
        std::lock_guard lg(_listener_mutex);
        add_listener(std::make_unique<snapshot_listener<fair::meta::null_type, M>>(std::move(matcher), delay, handler, block), block);
        return handler;
    }

    template<StreamCallback<T> Callback>
    void
    register_streaming_callback(std::size_t max_chunk_size, Callback callback) {
        add_listener(std::make_unique<continuous_listener<Callback>>(max_chunk_size, std::move(callback), *this), false);
    }

    template<TriggerMatcher M, DataSetCallback<T> Callback>
    void
    register_trigger_callback(M matcher, std::size_t pre_samples, std::size_t post_samples, Callback callback) {
        add_listener(std::make_unique<trigger_listener<Callback, M>>(std::move(matcher), pre_samples, post_samples, std::move(callback)), false);
        ensure_history_size(pre_samples);
    }

    template<TriggerMatcher M, DataSetCallback<T> Callback>
    void
    register_multiplexed_callback(M matcher, std::size_t maximum_window_size, Callback callback) {
        std::lock_guard lg(_listener_mutex);
        add_listener(std::make_unique<multiplexed_listener<Callback, M>>(std::move(matcher), maximum_window_size, std::move(callback)), false);
    }

    template<TriggerMatcher M, DataSetCallback<T> Callback>
    void
    register_snapshot_callback(M matcher, std::chrono::nanoseconds delay, Callback callback) {
        std::lock_guard lg(_listener_mutex);
        add_listener(std::make_unique<snapshot_listener<Callback, M>>(std::move(matcher), delay, std::move(callback)), false);
    }

    // TODO this code should be called at the end of graph processing
    void
    stop() noexcept {
        std::lock_guard lg(_listener_mutex);
        for (auto &listener : _listeners) {
            listener->stop();
        }
    }

    [[nodiscard]] work_return_t
    process_bulk(std::span<const T> in_data) noexcept {
        std::optional<property_map> tagData;
        if (this->input_tags_present()) {
            assert(this->input_tags()[0].index == 0);
            tagData = this->input_tags()[0].map;
            // signal info from settings overrides info from tags
            if (!_has_signal_info_from_settings) {
                apply_signal_info(this->input_tags()[0].map);
            }
        }

        {
            std::lock_guard lg(_listener_mutex); // TODO review/profile if a lock-free data structure should be used here
            const auto      history_view = _history.get_span(0);
            std::erase_if(_listeners, [](const auto &l) { return l->expired; });
            for (auto &listener : _listeners) {
                listener->process(history_view, in_data, tagData);
            }

            // store potential pre-samples for triggers at the beginning of the next chunk
            const auto to_write = std::min(in_data.size(), _history.capacity());
            _history.push_back_bulk(in_data.last(to_write));
        }

        return work_return_t::OK;
    }

private:
    bool
    apply_signal_info(const property_map &properties) {
        try {
            const auto srate = detail::get<float>(properties, tag::SAMPLE_RATE.key());
            const auto name  = detail::get<std::string>(properties, tag::SIGNAL_NAME.key());
            const auto unit  = detail::get<std::string>(properties, tag::SIGNAL_UNIT.key());
            const auto min   = detail::get<T>(properties, tag::SIGNAL_MIN.key());
            const auto max   = detail::get<T>(properties, tag::SIGNAL_MAX.key());

            // commit
            if (srate) {
                sample_rate = *srate;
            }
            if (name) {
                signal_name = *name;
            }
            if (unit) {
                signal_unit = *unit;
            }
            if (min) {
                signal_min = *min;
            }
            if (max) {
                signal_max = *max;
            }

            // forward to listeners
            if (srate || name || unit || min || max) {
                const auto      dstempl = make_dataset_template();

                std::lock_guard lg{ _listener_mutex };
                for (auto &l : _listeners) {
                    if (srate) {
                        l->apply_sample_rate(sample_rate);
                    }
                    if (name || unit || min || max) {
                        l->set_dataset_template(dstempl);
                    }
                }
            }
            return name || unit || min || max;
        } catch (const std::bad_variant_access &) {
            // TODO log?
            return false;
        }
    }

    DataSet<T>
    make_dataset_template() const {
        DataSet<T> dstempl;
        dstempl.signal_names  = { signal_name };
        dstempl.signal_units  = { signal_unit };
        dstempl.signal_ranges = { { signal_min, signal_max } };
        return dstempl;
    }

    void
    ensure_history_size(std::size_t new_size) {
        if (new_size <= _history.capacity()) {
            return;
        }
        // TODO Important!
        //  - History size must be limited to avoid users causing OOM
        //  - History should shrink again

        // transitional, do not reallocate/copy, but create a shared buffer with size N,
        // and a per-listener history buffer where more than N samples is needed.
        auto new_history = gr::history_buffer<T>(std::max(new_size, _history.capacity()));
        new_history.push_back_bulk(_history.begin(), _history.end());
        std::swap(_history, new_history);
    }

    void
    add_listener(std::unique_ptr<abstract_listener> &&l, bool block) {
        l->set_dataset_template(make_dataset_template());
        l->apply_sample_rate(sample_rate);
        if (block) {
            _listeners.push_back(std::move(l));
        } else {
            _listeners.push_front(std::move(l));
        }
    }

    struct abstract_listener {
        bool expired                 = false;

        virtual ~abstract_listener() = default;

        void
        set_expired() {
            expired = true;
        }

        virtual void
        apply_sample_rate(float /*sample_rate*/) {}

        virtual void
        set_dataset_template(DataSet<T>) {}

        virtual void
        process(std::span<const T> history, std::span<const T> data, std::optional<property_map> tag_data0)
                = 0;
        virtual void
        stop() = 0;
    };

    template<typename Callback>
    struct continuous_listener : public abstract_listener {
        static constexpr auto has_callback        = !std::is_same_v<Callback, fair::meta::null_type>;
        static constexpr auto callback_takes_tags = std::is_invocable_v<Callback, std::span<const T>, std::span<const tag_t>>
                                                 || std::is_invocable_v<Callback, std::span<const T>, std::span<const tag_t>, const data_sink<T> &>;

        const data_sink<T> &parent_sink;
        bool                block           = false;
        std::size_t         samples_written = 0;

        // callback-only
        std::size_t        buffer_fill = 0;
        std::vector<T>     buffer;
        std::vector<tag_t> tag_buffer;

        // polling-only
        std::weak_ptr<poller> polling_handler = {};

        Callback              callback;

        explicit continuous_listener(std::size_t max_chunk_size, Callback c, const data_sink<T> &parent) : parent_sink(parent), buffer(max_chunk_size), callback{ std::forward<Callback>(c) } {}

        explicit continuous_listener(std::shared_ptr<poller> poller, bool do_block, const data_sink<T> &parent) : parent_sink(parent), block(do_block), polling_handler{ std::move(poller) } {}

        inline void
        call_callback(std::span<const T> data, std::span<const tag_t> tags) {
            if constexpr (std::is_invocable_v<Callback, std::span<const T>, std::span<const tag_t>, const data_sink<T> &>) {
                callback(std::move(data), std::move(tags), parent_sink);
            } else if constexpr (std::is_invocable_v<Callback, std::span<const T>, std::span<const tag_t>>) {
                callback(std::move(data), std::move(tags));
            } else {
                callback(std::move(data));
            }
        }

        void
        process(std::span<const T>, std::span<const T> data, std::optional<property_map> tag_data0) override {
            using namespace fair::graph::detail;

            if constexpr (has_callback) {
                // if there's pending data, fill buffer and send out
                if (buffer_fill > 0) {
                    const auto n = std::min(data.size(), buffer.size() - buffer_fill);
                    detail::copy_span(data.first(n), std::span(buffer).subspan(buffer_fill, n));
                    if constexpr (callback_takes_tags) {
                        if (tag_data0) {
                            tag_buffer.push_back({ static_cast<tag_t::signed_index_type>(buffer_fill), *tag_data0 });
                            tag_data0.reset();
                        }
                    }
                    buffer_fill += n;
                    if (buffer_fill == buffer.size()) {
                        call_callback(std::span(buffer), std::span(tag_buffer));
                        samples_written += buffer.size();
                        buffer_fill = 0;
                        tag_buffer.clear();
                    }

                    data = data.last(data.size() - n);
                }

                // send out complete chunks directly
                while (data.size() >= buffer.size()) {
                    if constexpr (callback_takes_tags) {
                        std::vector<tag_t> tags;
                        if (tag_data0) {
                            tags.push_back({ 0, std::move(*tag_data0) });
                            tag_data0.reset();
                        }
                        call_callback(data.first(buffer.size()), std::span(tags));
                    } else {
                        callback(data.first(buffer.size()));
                    }
                    samples_written += buffer.size();
                    data = data.last(data.size() - buffer.size());
                }

                // write remaining data to the buffer
                if (!data.empty()) {
                    detail::copy_span(data, std::span(buffer).first(data.size()));
                    buffer_fill = data.size();
                    if constexpr (callback_takes_tags) {
                        if (tag_data0) {
                            tag_buffer.push_back({ 0, std::move(*tag_data0) });
                        }
                    }
                }
            } else {
                auto poller = polling_handler.lock();
                if (!poller) {
                    this->set_expired();
                    return;
                }

                const auto to_write = block ? data.size() : std::min(data.size(), poller->writer.available());

                if (to_write > 0) {
                    if (tag_data0) {
                        auto tw = poller->tag_writer.reserve_output_range(1);
                        tw[0]   = { static_cast<tag_t::signed_index_type>(samples_written), std::move(*tag_data0) };
                        tw.publish(1);
                    }
                    auto write_data = poller->writer.reserve_output_range(to_write);
                    detail::copy_span(data.first(to_write), std::span(write_data));
                    write_data.publish(write_data.size());
                }
                poller->drop_count += data.size() - to_write;
                samples_written += to_write;
            }
        }

        void
        stop() override {
            if constexpr (has_callback) {
                if (buffer_fill > 0) {
                    call_callback(std::span(buffer).first(buffer_fill), std::span(tag_buffer));
                    tag_buffer.clear();
                    buffer_fill = 0;
                }
            } else {
                if (auto p = polling_handler.lock()) {
                    p->finished = true;
                }
            }
        }
    };

    struct pending_window {
        DataSet<T>  dataset;
        std::size_t pending_post_samples = 0;
    };

    template<typename Callback, TriggerMatcher M>
    struct trigger_listener : public abstract_listener {
        bool                          block        = false;
        std::size_t                   pre_samples  = 0;
        std::size_t                   post_samples = 0;

        DataSet<T>                    dataset_template;
        M                             trigger_matcher = {};
        std::deque<pending_window>    pending_trigger_windows; // triggers that still didn't receive all their data
        std::weak_ptr<dataset_poller> polling_handler = {};

        Callback                      callback;

        explicit trigger_listener(M matcher, std::shared_ptr<dataset_poller> handler, std::size_t pre, std::size_t post, bool do_block)
            : block(do_block), pre_samples(pre), post_samples(post), trigger_matcher(std::move(matcher)), polling_handler{ std::move(handler) } {}

        explicit trigger_listener(M matcher, std::size_t pre, std::size_t post, Callback cb) : pre_samples(pre), post_samples(post), trigger_matcher(std::move(matcher)), callback{ std::move(cb) } {}

        void
        set_dataset_template(DataSet<T> dst) override {
            dataset_template = std::move(dst);
        }

        inline void
        publish_dataset(DataSet<T> &&data) {
            if constexpr (!std::is_same_v<Callback, fair::meta::null_type>) {
                callback(std::move(data));
            } else {
                auto poller = polling_handler.lock();
                if (!poller) {
                    this->set_expired();
                    return;
                }

                auto write_data = poller->writer.reserve_output_range(1);
                if (block) {
                    write_data[0] = std::move(data);
                    write_data.publish(1);
                } else {
                    if (poller->writer.available() > 0) {
                        write_data[0] = std::move(data);
                        write_data.publish(1);
                    } else {
                        poller->drop_count++;
                    }
                }
            }
        }

        void
        process(std::span<const T> history, std::span<const T> in_data, std::optional<property_map> tag_data0) override {
            if (tag_data0 && trigger_matcher(tag_t{ 0, *tag_data0 }) == trigger_match_result::Matching) {
                DataSet<T> dataset = dataset_template;
                dataset.signal_values.reserve(pre_samples + post_samples); // TODO maybe make the circ. buffer smaller but preallocate these

                const auto pre_sample_view = history.last(std::min(pre_samples, history.size()));
                dataset.signal_values.insert(dataset.signal_values.end(), pre_sample_view.begin(), pre_sample_view.end());

                dataset.timing_events = { { { static_cast<tag_t::signed_index_type>(pre_sample_view.size()), *tag_data0 } } };
                pending_trigger_windows.push_back({ .dataset = std::move(dataset), .pending_post_samples = post_samples });
            }

            auto window = pending_trigger_windows.begin();
            while (window != pending_trigger_windows.end()) {
                const auto post_sample_view = in_data.first(std::min(window->pending_post_samples, in_data.size()));
                window->dataset.signal_values.insert(window->dataset.signal_values.end(), post_sample_view.begin(), post_sample_view.end());
                window->pending_post_samples -= post_sample_view.size();

                if (window->pending_post_samples == 0) {
                    this->publish_dataset(std::move(window->dataset));
                    window = pending_trigger_windows.erase(window);
                } else {
                    ++window;
                }
            }
        }

        void
        stop() override {
            for (auto &window : pending_trigger_windows) {
                if (!window.dataset.signal_values.empty()) {
                    this->publish_dataset(std::move(window.dataset));
                }
            }
            pending_trigger_windows.clear();
            if (auto p = polling_handler.lock()) {
                p->finished = true;
            }
        }
    };

    template<typename Callback, TriggerMatcher M>
    struct multiplexed_listener : public abstract_listener {
        bool                          block = false;
        M                             matcher;
        DataSet<T>                    dataset_template;
        std::optional<DataSet<T>>     pending_dataset;
        std::size_t                   maximum_window_size;
        std::weak_ptr<dataset_poller> polling_handler = {};
        Callback                      callback;

        explicit multiplexed_listener(M matcher_, std::size_t max_window_size, Callback cb) : matcher(std::move(matcher_)), maximum_window_size(max_window_size), callback(cb) {}

        explicit multiplexed_listener(M matcher_, std::size_t max_window_size, std::shared_ptr<dataset_poller> handler, bool do_block)
            : block(do_block), matcher(std::move(matcher_)), maximum_window_size(max_window_size), polling_handler{ std::move(handler) } {}

        void
        set_dataset_template(DataSet<T> dst) override {
            dataset_template = std::move(dst);
        }

        inline void
        publish_dataset(DataSet<T> &&data) {
            if constexpr (!std::is_same_v<Callback, fair::meta::null_type>) {
                callback(std::move(data));
            } else {
                auto poller = polling_handler.lock();
                if (!poller) {
                    this->set_expired();
                    return;
                }

                auto write_data = poller->writer.reserve_output_range(1);
                if (block) {
                    write_data[0] = std::move(data);
                    write_data.publish(1);
                } else {
                    if (poller->writer.available() > 0) {
                        write_data[0] = std::move(data);
                        write_data.publish(1);
                    } else {
                        poller->drop_count++;
                    }
                }
            }
        }

        void
        process(std::span<const T>, std::span<const T> in_data, std::optional<property_map> tag_data0) override {
            if (tag_data0) {
                const auto obsr = matcher(tag_t{ 0, *tag_data0 });
                if (obsr == trigger_match_result::NotMatching || obsr == trigger_match_result::Matching) {
                    if (pending_dataset) {
                        if (obsr == trigger_match_result::NotMatching) {
                            pending_dataset->timing_events[0].push_back({ static_cast<tag_t::signed_index_type>(pending_dataset->signal_values.size()), *tag_data0 });
                        }
                        this->publish_dataset(std::move(*pending_dataset));
                        pending_dataset.reset();
                    }
                }
                if (obsr == trigger_match_result::Matching) {
                    pending_dataset = dataset_template;
                    pending_dataset->signal_values.reserve(maximum_window_size); // TODO might be too much?
                    pending_dataset->timing_events = { { { 0, *tag_data0 } } };
                }
            }
            if (pending_dataset) {
                const auto to_write = std::min(in_data.size(), maximum_window_size - pending_dataset->signal_values.size());
                const auto view     = in_data.first(to_write);
                pending_dataset->signal_values.insert(pending_dataset->signal_values.end(), view.begin(), view.end());

                if (pending_dataset->signal_values.size() == maximum_window_size) {
                    this->publish_dataset(std::move(*pending_dataset));
                    pending_dataset.reset();
                }
            }
        }

        void
        stop() override {
            if (pending_dataset) {
                this->publish_dataset(std::move(*pending_dataset));
                pending_dataset.reset();
            }
            if (auto p = polling_handler.lock()) {
                p->finished = true;
            }
        }
    };

    struct pending_snapshot {
        property_map tag_data;
        std::size_t  delay           = 0;
        std::size_t  pending_samples = 0;
    };

    template<typename Callback, TriggerMatcher M>
    struct snapshot_listener : public abstract_listener {
        bool                          block = false;
        std::chrono::nanoseconds      time_delay;
        std::size_t                   sample_delay = 0;
        DataSet<T>                    dataset_template;
        M                             trigger_matcher = {};
        std::deque<pending_snapshot>  pending;
        std::weak_ptr<dataset_poller> polling_handler = {};
        Callback                      callback;

        explicit snapshot_listener(M matcher, std::chrono::nanoseconds delay, std::shared_ptr<dataset_poller> poller, bool do_block)
            : block(do_block), time_delay(delay), trigger_matcher(std::move(matcher)), polling_handler{ std::move(poller) } {}

        explicit snapshot_listener(M matcher, std::chrono::nanoseconds delay, Callback cb) : time_delay(delay), trigger_matcher(std::move(matcher)), callback(std::move(cb)) {}

        void
        set_dataset_template(DataSet<T> dst) override {
            dataset_template = std::move(dst);
        }

        void
        apply_sample_rate(float rateHz) override {
            sample_delay = std::round(std::chrono::duration_cast<std::chrono::duration<float>>(time_delay).count() * rateHz);
            // TODO do we need to update the requested_samples of pending here? (considering both old and new time_delay)
        }

        inline void
        publish_dataset(DataSet<T> &&data) {
            if constexpr (!std::is_same_v<Callback, fair::meta::null_type>) {
                callback(std::move(data));
            } else {
                auto poller = polling_handler.lock();
                if (!poller) {
                    this->set_expired();
                    return;
                }

                auto write_data = poller->writer.reserve_output_range(1);
                if (block) {
                    write_data[0] = std::move(data);
                    write_data.publish(1);
                } else {
                    if (poller->writer.available() > 0) {
                        write_data[0] = std::move(data);
                        write_data.publish(1);
                    } else {
                        poller->drop_count++;
                    }
                }
            }
        }

        void
        process(std::span<const T>, std::span<const T> in_data, std::optional<property_map> tag_data0) override {
            if (tag_data0 && trigger_matcher({ 0, *tag_data0 }) == trigger_match_result::Matching) {
                auto new_pending = pending_snapshot{ *tag_data0, sample_delay, sample_delay };
                // make sure pending is sorted by number of pending_samples (insertion might be not at end if sample rate decreased; TODO unless we adapt them in apply_sample_rate, see there)
                auto rit = std::find_if(pending.rbegin(), pending.rend(), [delay = sample_delay](const auto &other) { return other.pending_samples < delay; });
                pending.insert(rit.base(), std::move(new_pending));
            }

            auto it = pending.begin();
            while (it != pending.end()) {
                if (it->pending_samples >= in_data.size()) {
                    it->pending_samples -= in_data.size();
                    break;
                }

                DataSet<T> dataset    = dataset_template;
                dataset.timing_events = { { { -static_cast<tag_t::signed_index_type>(it->delay), std::move(it->tag_data) } } };
                dataset.signal_values = { in_data[it->pending_samples] };
                this->publish_dataset(std::move(dataset));

                it = pending.erase(it);
            }
        }

        void
        stop() override {
            pending.clear();
            if (auto p = polling_handler.lock()) {
                p->finished = true;
            }
        }
    };
};

} // namespace fair::graph

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (fair::graph::data_sink<T>), in, sample_rate, signal_name, signal_unit, signal_min, signal_max);

#endif
