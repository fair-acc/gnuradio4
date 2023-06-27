#ifndef GNURADIO_DATA_SINK_HPP
#define GNURADIO_DATA_SINK_HPP

#include "circular_buffer.hpp"
#include "dataset.hpp"
#include "history_buffer.hpp"
#include "node.hpp"
#include "tag.hpp"

#include <any>
#include <chrono>

namespace fair::graph {

enum class blocking_mode { NonBlocking, Blocking };

enum class trigger_observer_state {
    Start,        ///< Start a new dataset
    Stop,         ///< Finish dataset
    StopAndStart, ///< Finish pending dataset, start a new one
    Ignore        ///< Ignore tag
};

// Until clang-format can handle concepts
// clang-format off
template<typename T>
concept TriggerPredicate = requires(const T p, tag_t tag) {
    { p(tag) } -> std::convertible_to<bool>;
};

/**
 * For the 'Multiplexed' acquisition mode: Stateful object checking all incoming tags to control which data should be sent
 * to the listener.
 *
 * A new dataset is started when the observer returns @c Start or @c StopAndStart.
 * A dataset is closed and sent when @c Stop or @StopAndStart is returned.
 *
 * The observer can rely on being called with each incoming tag exactly once, in the order they arrive.
 *
 * Example:
 *
 * @code
 * // Observer observing three possible tag values, "green", "yellow", "red".
 * // starting a dataset when seeing "green", stopping on "red", starting a new dataset on "yellow"
 * struct color_observer {
 *     trigger_observer_state operator()(const tag_t &tag) {
 *         if (tag == green || tag == yellow) {
 *             return trigger_observer_state::StopAndStart;
 *         }
 *         if (tag == red) {
 *             return trigger_observer_state::Stop;
 *         }
 *
 *         return trigger_observer_state::Ignore;
 *     }
 * };
 * @endcode
 *
 * @see trigger_observer_state
 */
template<typename T>
concept TriggerObserver = requires(T o, tag_t tag) {
    { o(tag) } -> std::convertible_to<trigger_observer_state>;
};

template<typename T>
concept TriggerObserverFactory = requires(T f) {
    { f() } -> TriggerObserver;
};
// clang-format on

template<typename T>
class data_sink;

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

    template<typename T, TriggerPredicate P>
    std::shared_ptr<typename data_sink<T>::dataset_poller>
    get_trigger_poller(const data_sink_query &query, P p, std::size_t pre_samples, std::size_t post_samples, blocking_mode block = blocking_mode::Blocking) {
        std::lock_guard lg{ _mutex };
        auto            sink = find_sink<T>(query);
        return sink ? sink->get_trigger_poller(std::forward<P>(p), pre_samples, post_samples, block) : nullptr;
    }

    template<typename T, TriggerObserverFactory F>
    std::shared_ptr<typename data_sink<T>::dataset_poller>
    get_multiplexed_poller(const data_sink_query &query, F triggerObserverFactory, std::size_t maximum_window_size, blocking_mode block = blocking_mode::Blocking) {
        std::lock_guard lg{ _mutex };
        auto            sink = find_sink<T>(query);
        return sink ? sink->get_multiplexed_poller(std::forward<F>(triggerObserverFactory), maximum_window_size, block) : nullptr;
    }

    template<typename T, TriggerPredicate P>
    std::shared_ptr<typename data_sink<T>::dataset_poller>
    get_snapshot_poller(const data_sink_query &query, P p, std::chrono::nanoseconds delay, blocking_mode block = blocking_mode::Blocking) {
        std::lock_guard lg{ _mutex };
        auto            sink = find_sink<T>(query);
        return sink ? sink->get_snapshot_poller(std::forward<P>(p), delay, block) : nullptr;
    }

    template<typename T, typename Callback>
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

    template<typename T, TriggerPredicate P, typename Callback>
    bool
    register_trigger_callback(const data_sink_query &query, P p, std::size_t pre_samples, std::size_t post_samples, Callback callback) {
        std::lock_guard lg{ _mutex };
        auto            sink = find_sink<T>(query);
        if (!sink) {
            return false;
        }

        sink->register_trigger_callback(std::forward<P>(p), pre_samples, post_samples, std::forward<Callback>(callback));
        return true;
    }

    template<typename T, TriggerObserver O, typename Callback>
    bool
    register_multiplexed_callback(const data_sink_query &query, std::size_t maximum_window_size, Callback callback) {
        std::lock_guard lg{ _mutex };
        auto            sink = find_sink<T>(query);
        if (!sink) {
            return false;
        }

        sink->template register_multiplexed_callback<O, Callback>(maximum_window_size, std::move(callback));
        return true;
    }

    template<typename T, TriggerPredicate P, typename Callback>
    bool
    register_snapshot_callback(const data_sink_query &query, P p, std::chrono::nanoseconds delay, Callback callback) {
        std::lock_guard lg{ _mutex };
        auto            sink = find_sink<T>(query);
        if (!sink) {
            return false;
        }

        sink->template register_snapshot_callback<P, Callback>(std::forward<P>(p), delay, std::forward<Callback>(callback));
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
template<typename T, typename P>
std::span<T>
find_matching_prefix(std::span<T> s, P predicate) {
    const auto nm = std::find_if_not(s.begin(), s.end(), predicate);
    if (nm == s.end()) {
        return s;
    }
    return s.first(std::distance(s.begin(), nm));
}

template<typename T>
bool
copy_span(std::span<const T> src, std::span<T> dst) {
    assert(src.size() <= dst.size());
    if (src.size() > dst.size()) {
        return false;
    }
    std::copy(src.begin(), src.end(), dst.begin());
    return true;
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
 *        ━╢ :signal_name  ║←--------------------→│  user code...) │
 *        ━╢ :signal_unit  ║  register            │                │
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
    gr::history_buffer<T>                          _history = gr::history_buffer<T>(1);

public:
    Annotated<float, "sample rate", Doc<"signal sample rate">, Unit<"Hz">>           sample_rate = 10000.f;
    Annotated<std::string, "signal name", Visible>                                   signal_name;
    Annotated<std::string, "signal unit", Visible, Doc<"signal's physical SI unit">> signal_unit;
    Annotated<T, "signal min", Doc<"signal physical min. (e.g. DAQ) limit">>         signal_min;
    Annotated<T, "signal max", Doc<"signal physical max. (e.g. DAQ) limit">>         signal_max;

    IN<T, std::dynamic_extent, _listener_buffer_size>                                in;

    struct poller {
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
        process_bulk(std::invocable<std::span<DataSet<T>>> auto fnc) {
            const auto available = reader.available();
            if (available == 0) {
                return false;
            }

            const auto read_data = reader.get(available);
            fnc(read_data);
            std::ignore = reader.consume(available);
            return true;
        }

        [[nodiscard]] bool
        process_one(std::invocable<DataSet<T>> auto fnc) {
            const auto available = reader.available();
            if (available == 0) {
                return false;
            }

            const auto read_data = reader.get(1);
            fnc(read_data[0]);
            std::ignore = reader.consume(1);
            return true;
        }
    };

    data_sink() { data_sink_registry::instance().register_sink(this); }

    ~data_sink() { data_sink_registry::instance().unregister_sink(this); }

    std::shared_ptr<poller>
    get_streaming_poller(blocking_mode block_mode = blocking_mode::Blocking) {
        std::lock_guard lg(_listener_mutex);
        const auto      block   = block_mode == blocking_mode::Blocking;
        auto            handler = std::make_shared<poller>();
        add_listener(std::make_unique<continuous_listener<fair::meta::null_type>>(handler, block), block);
        return handler;
    }

    template<typename TriggerPredicate>
    std::shared_ptr<dataset_poller>
    get_trigger_poller(TriggerPredicate p, std::size_t pre_samples, std::size_t post_samples, blocking_mode block_mode = blocking_mode::Blocking) {
        const auto      block   = block_mode == blocking_mode::Blocking;
        auto            handler = std::make_shared<dataset_poller>();
        std::lock_guard lg(_listener_mutex);
        add_listener(std::make_unique<trigger_listener<fair::meta::null_type, TriggerPredicate>>(std::forward<TriggerPredicate>(p), handler, pre_samples, post_samples, block), block);
        ensure_history_size(pre_samples);
        return handler;
    }

    template<TriggerObserverFactory F>
    std::shared_ptr<dataset_poller>
    get_multiplexed_poller(F triggerObserverFactory, std::size_t maximum_window_size, blocking_mode block_mode = blocking_mode::Blocking) {
        std::lock_guard lg(_listener_mutex);
        const auto      block   = block_mode == blocking_mode::Blocking;
        auto            handler = std::make_shared<dataset_poller>();
        add_listener(std::make_unique<multiplexed_listener<fair::meta::null_type, F>>(std::move(triggerObserverFactory), maximum_window_size, handler, block), block);
        return handler;
    }

    template<TriggerPredicate P>
    std::shared_ptr<dataset_poller>
    get_snapshot_poller(P p, std::chrono::nanoseconds delay, blocking_mode block_mode = blocking_mode::Blocking) {
        const auto      block   = block_mode == blocking_mode::Blocking;
        auto            handler = std::make_shared<dataset_poller>();
        std::lock_guard lg(_listener_mutex);
        add_listener(std::make_unique<snapshot_listener<fair::meta::null_type, P>>(std::forward<P>(p), delay, handler, block), block);
        return handler;
    }

    template<typename Callback>
    void
    register_streaming_callback(std::size_t max_chunk_size, Callback callback) {
        add_listener(std::make_unique<continuous_listener<Callback>>(max_chunk_size, std::forward<Callback>(callback)), false);
    }

    template<TriggerPredicate P, typename Callback>
    void
    register_trigger_callback(P p, std::size_t pre_samples, std::size_t post_samples, Callback callback) {
        add_listener(std::make_unique<trigger_listener<Callback, P>>(std::forward<P>(p), pre_samples, post_samples, std::forward<Callback>(callback)), false);
        ensure_history_size(pre_samples);
    }

    template<TriggerObserverFactory F, typename Callback>
    void
    register_multiplexed_callback(F triggerObserverFactory, std::size_t maximum_window_size, Callback callback) {
        std::lock_guard lg(_listener_mutex);
        add_listener(std::make_unique<multiplexed_listener>(std::move(triggerObserverFactory), maximum_window_size, std::forward<Callback>(callback)), false);
    }

    template<TriggerPredicate P, typename Callback>
    void
    register_snapshot_callback(P p, std::chrono::nanoseconds delay, Callback callback) {
        std::lock_guard lg(_listener_mutex);
        add_listener(std::make_unique<snapshot_listener>(std::forward<P>(p), delay, std::forward<Callback>(callback)), false);
    }

    // TODO this code should be called at the end of graph processing
    void
    stop() noexcept {
        std::lock_guard lg(_listener_mutex);
        for (auto &listener : _listeners) {
            listener->flush();
        }
    }

    [[nodiscard]] work_return_t
    process_bulk(std::span<const T> in_data) noexcept {
        std::optional<property_map> tagData;
        if (this->input_tags_present()) {
            assert(this->input_tags()[0].index == 0);
            tagData = this->input_tags()[0].map;
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
    void
    ensure_history_size(std::size_t new_size) {
        if (new_size <= _history.capacity()) {
            return;
        }
        // TODO transitional, do not reallocate/copy, but create a shared buffer with size N,
        // and a per-listener history buffer where more than N samples is needed.
        auto new_history = gr::history_buffer<T>(std::max(new_size, _history.capacity()));
        new_history.push_back_bulk(_history.begin(), _history.end());
        std::swap(_history, new_history);
    }

    void
    add_listener(std::unique_ptr<abstract_listener> &&l, bool block) {
        l->apply_sample_rate(sample_rate); // TODO also call when sample_rate changes
        if (block) {
            _listeners.push_back(std::move(l));
        } else {
            _listeners.push_front(std::move(l));
        }
    }

    struct abstract_listener {
        bool expired = false;

        virtual ~abstract_listener() = default;

        void set_expired() { expired = true; }

        virtual void
        apply_sample_rate(float) {}

        virtual void
        process(std::span<const T> history, std::span<const T> data, std::optional<property_map> tag_data0)
                = 0;
        virtual void
        flush() = 0;
    };

    template<typename Callback>
    struct continuous_listener : public abstract_listener {
        static constexpr auto has_callback        = !std::is_same_v<Callback, fair::meta::null_type>;
        static constexpr auto callback_takes_tags = std::is_invocable_v<Callback, std::span<const T>, std::span<const tag_t>>;

        bool                  block               = false;
        std::size_t           samples_written     = 0;

        // callback-only
        std::size_t        buffer_fill = 0;
        std::vector<T>     buffer;
        std::vector<tag_t> tag_buffer;

        // polling-only
        std::weak_ptr<poller> polling_handler = {};

        Callback              callback;

        explicit continuous_listener(std::size_t max_chunk_size, Callback c) : buffer(max_chunk_size), callback{ std::forward<Callback>(c) } {}

        explicit continuous_listener(std::shared_ptr<poller> poller, bool do_block) : block(do_block), polling_handler{ std::move(poller) } {}

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
                        if constexpr (callback_takes_tags) {
                            callback(std::span(buffer), std::span(tag_buffer));
                        } else {
                            callback(std::span(buffer));
                        }
                        samples_written += buffer.size();
                        buffer_fill = 0;
                        tag_buffer.clear();
                    }

                    data = data.last(data.size() - n);
                }

                // send out complete chunks directly
                while (data.size() > buffer.size()) {
                    if constexpr (callback_takes_tags) {
                        std::vector<tag_t> tags;
                        if (tag_data0) {
                            tags.push_back({ 0, std::move(*tag_data0) });
                            tag_data0.reset();
                        }
                        callback(data.first(buffer.size()), std::span(tags));
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
        flush() override {
            if constexpr (has_callback) {
                if (buffer_fill > 0) {
                    if constexpr (callback_takes_tags) {
                        callback(std::span(buffer).first(buffer_fill), std::span(tag_buffer));
                        tag_buffer.clear();
                    } else {
                        callback(std::span(buffer).first(buffer_fill));
                    }
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

    template<typename Callback, TriggerPredicate P>
    struct trigger_listener : public abstract_listener {
        bool                          block             = false;
        std::size_t                   pre_samples       = 0;
        std::size_t                   post_samples      = 0;

        P                             trigger_predicate = {};
        std::deque<pending_window>    pending_trigger_windows; // triggers that still didn't receive all their data
        std::weak_ptr<dataset_poller> polling_handler = {};

        Callback                      callback;

        explicit trigger_listener(P predicate, std::shared_ptr<dataset_poller> handler, std::size_t pre, std::size_t post, bool do_block)
            : block(do_block), pre_samples(pre), post_samples(post), trigger_predicate(std::forward<P>(predicate)), polling_handler{ std::move(handler) } {}

        explicit trigger_listener(P predicate, std::size_t pre, std::size_t post, Callback cb)
            : pre_samples(pre), post_samples(post), trigger_predicate(std::forward<P>(predicate)), callback{ std::forward<Callback>(cb) } {}

        // TODO all the dataset-based listeners could share publish_dataset and parts of flush (closing pollers),
        // but if we want to use different datastructures/pass additional info, this might become moot again, so
        // I leave it as is for now.
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
            if (tag_data0 && trigger_predicate(tag_t{ 0, *tag_data0 })) {
                // TODO fill dataset with metadata etc.
                DataSet<T> dataset;
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
                    publish_dataset(std::move(window->dataset));
                    window = pending_trigger_windows.erase(window);
                } else {
                    ++window;
                }
            }
        }

        void
        flush() override {
            for (auto &window : pending_trigger_windows) {
                if (!window.dataset.signal_values.empty()) {
                    publish_dataset(std::move(window.dataset));
                }
            }
            pending_trigger_windows.clear();
            if (auto p = polling_handler.lock()) {
                p->finished = true;
            }
        }
    };

    template<typename Callback, TriggerObserverFactory F>
    struct multiplexed_listener : public abstract_listener {
        bool                          block = false;
        F                             observerFactory;
        decltype(observerFactory())   observer;
        std::optional<DataSet<T>>     pending_dataset;
        std::size_t                   maximum_window_size;
        std::weak_ptr<dataset_poller> polling_handler = {};
        Callback                      callback;

        explicit multiplexed_listener(F factory, std::size_t max_window_size, Callback cb)
            : observerFactory(factory), observer(observerFactory()), maximum_window_size(max_window_size), callback(cb) {}

        explicit multiplexed_listener(F factory, std::size_t max_window_size, std::shared_ptr<dataset_poller> handler, bool do_block)
            : block(do_block), observerFactory(factory), observer(observerFactory()), maximum_window_size(max_window_size), polling_handler{ std::move(handler) } {}

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
                const auto obsr = observer(tag_t{ 0, *tag_data0 });
                // TODO set proper error state instead of throwing
                if (obsr == trigger_observer_state::Stop || obsr == trigger_observer_state::StopAndStart) {
                    if (obsr == trigger_observer_state::Stop && !pending_dataset) {
                        throw std::runtime_error("multiplexed: Stop without start");
                    }

                    if (pending_dataset) {
                        if (obsr == trigger_observer_state::Stop) {
                            pending_dataset->timing_events[0].push_back({ static_cast<tag_t::signed_index_type>(pending_dataset->signal_values.size()), *tag_data0 });
                        }
                        publish_dataset(std::move(*pending_dataset));
                        pending_dataset.reset();
                    }
                }
                if (obsr == trigger_observer_state::Start || obsr == trigger_observer_state::StopAndStart) {
                    if (obsr == trigger_observer_state::Start && pending_dataset) {
                        throw std::runtime_error("multiplexed: Two starts without stop");
                    }
                    pending_dataset = DataSet<T>();
                    pending_dataset->signal_values.reserve(maximum_window_size); // TODO might be too much?
                    pending_dataset->timing_events = { { { 0, *tag_data0 } } };
                }
            }
            if (pending_dataset) {
                const auto to_write = std::min(in_data.size(), maximum_window_size - pending_dataset->signal_values.size());
                const auto view     = in_data.first(to_write);
                pending_dataset->signal_values.insert(pending_dataset->signal_values.end(), view.begin(), view.end());

                if (pending_dataset->signal_values.size() == maximum_window_size) {
                    publish_dataset(std::move(*pending_dataset));
                    pending_dataset.reset();
                }
            }
        }

        void
        flush() override {
            if (pending_dataset) {
                publish_dataset(std::move(*pending_dataset));
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

    template<typename Callback, TriggerPredicate P>
    struct snapshot_listener : public abstract_listener {
        bool                          block = false;
        std::chrono::nanoseconds      time_delay;
        std::size_t                   sample_delay      = 0;
        P                             trigger_predicate = {};
        std::deque<pending_snapshot>  pending;
        std::weak_ptr<dataset_poller> polling_handler = {};
        Callback                      callback;

        explicit snapshot_listener(P p, std::chrono::nanoseconds delay, std::shared_ptr<dataset_poller> poller, bool do_block)
            : block(do_block), time_delay(delay), trigger_predicate(std::forward<P>(p)), polling_handler{ std::move(poller) } {}

        explicit snapshot_listener(P p, std::chrono::nanoseconds delay, Callback cb) : trigger_predicate(std::forward<P>(p)), time_delay(std::forward<Callback>(cb)) {}

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
        apply_sample_rate(float rateHz) override {
            sample_delay = std::round(std::chrono::duration_cast<std::chrono::duration<float>>(time_delay).count() * rateHz);
            // TODO do we need to update the requested_samples of pending here? (considering both old and new time_delay)
        }

        void
        process(std::span<const T>, std::span<const T> in_data, std::optional<property_map> tag_data0) override {
            if (tag_data0 && trigger_predicate({ 0, *tag_data0 })) {
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

                DataSet<T> dataset;
                dataset.timing_events = { { { -static_cast<tag_t::signed_index_type>(it->delay), std::move(it->tag_data) } } };
                dataset.signal_values = { in_data[it->pending_samples] };
                publish_dataset(std::move(dataset));

                it = pending.erase(it);
            }
        }

        void
        flush() override {
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
