#ifndef GNURADIO_DATA_SINK_HPP
#define GNURADIO_DATA_SINK_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/Tag.hpp>

#include <any>
#include <chrono>
#include <deque>
#include <limits>

namespace gr::basic {

enum class BlockingMode { NonBlocking, Blocking };

enum class TriggerMatchResult {
    Matching,    ///< Start a new dataset
    NotMatching, ///< Finish dataset
    Ignore       ///< Ignore tag
};

template<typename T>
class DataSink;

// Until clang-format can handle concepts
// clang-format off

template<typename T, typename V>
concept DataSetCallback = std::invocable<T, DataSet<V>>;

/**
 * Stream callback functions receive the span of data, with optional tags and reference to the sink.
 */
template<typename T, typename V>
concept StreamCallback = std::invocable<T, std::span<const V>> || std::invocable<T, std::span<const V>, std::span<const Tag>> || std::invocable<T, std::span<const V>, std::span<const Tag>, const DataSink<V>&>;

/**
 * Used for testing whether a tag should trigger data acquisition.
 *
 * For the 'Triggered' (data window) and 'Snapshot' (single sample) acquisition modes:
 * Stateless predicate to check whether a tag matches the trigger criteria.
 *
 * @code
 * auto matcher = [](const auto &tag) {
 *     const auto isTrigger = ...check if tag is trigger...;
 *     return isTrigger ? TriggerMatchResult::Matching : TriggerMatchResult::Ignore;
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
 * struct ColorMatcher {
 *     TriggerMatcherResult operator()(const Tag &tag) {
 *         if (tag == green || tag == yellow) {
 *             return TriggerMatchResult::Matching;
 *         }
 *         if (tag == red) {
 *             return TriggerMatchResult::NotMatching;
 *         }
 *
 *         return TriggerMatchResult::Ignore;
 *     }
 * };
 * @endcode
 *
 * @see TriggerMatchResult
 */
template<typename T>
concept TriggerMatcher = requires(T matcher, Tag tag) {
    { matcher(tag) } -> std::convertible_to<TriggerMatchResult>;
};

// clang-format on

struct DataSinkQuery {
    std::optional<std::string> _sink_name;
    std::optional<std::string> _signal_name;

    static DataSinkQuery
    signalName(std::string_view name) {
        return { {}, std::string{ name } };
    }

    static DataSinkQuery
    sinkName(std::string_view name) {
        return { std::string{ name }, {} };
    }
};

class DataSinkRegistry {
    std::mutex            _mutex;
    std::vector<std::any> _sinks;

public:
    // TODO this shouldn't be a singleton but associated with the flow graph (?)
    // TODO reconsider mutex usage when moving to the graph
    static DataSinkRegistry &
    instance() {
        static DataSinkRegistry s_instance;
        return s_instance;
    }

    template<typename T>
    void
    registerSink(DataSink<T> *sink) {
        std::lock_guard lg{ _mutex };
        _sinks.push_back(sink);
    }

    template<typename T>
    void
    unregisterSink(DataSink<T> *sink) {
        std::lock_guard lg{ _mutex };
        std::erase_if(_sinks, [sink](const std::any &v) {
            try {
                return std::any_cast<DataSink<T> *>(v) == sink;
            } catch (...) {
                return false;
            }
        });
    }

    template<typename T>
    std::shared_ptr<typename DataSink<T>::Poller>
    getStreamingPoller(const DataSinkQuery &query, BlockingMode block = BlockingMode::Blocking) {
        std::lock_guard lg{ _mutex };
        auto            sink = findSink<T>(query);
        return sink ? sink->getStreamingPoller(block) : nullptr;
    }

    template<typename T, TriggerMatcher M>
    std::shared_ptr<typename DataSink<T>::DataSetPoller>
    getTriggerPoller(const DataSinkQuery &query, M &&matcher, std::size_t preSamples, std::size_t postSamples, BlockingMode block = BlockingMode::Blocking) {
        std::lock_guard lg{ _mutex };
        auto            sink = findSink<T>(query);
        return sink ? sink->getTriggerPoller(std::forward<M>(matcher), preSamples, postSamples, block) : nullptr;
    }

    template<typename T, TriggerMatcher M>
    std::shared_ptr<typename DataSink<T>::DataSetPoller>
    getMultiplexedPoller(const DataSinkQuery &query, M &&matcher, std::size_t maximumWindowSize, BlockingMode block = BlockingMode::Blocking) {
        std::lock_guard lg{ _mutex };
        auto            sink = findSink<T>(query);
        return sink ? sink->getMultiplexedPoller(std::forward<M>(matcher), maximumWindowSize, block) : nullptr;
    }

    template<typename T, TriggerMatcher M>
    std::shared_ptr<typename DataSink<T>::DataSetPoller>
    getSnapshotPoller(const DataSinkQuery &query, M &&matcher, std::chrono::nanoseconds delay, BlockingMode block = BlockingMode::Blocking) {
        std::lock_guard lg{ _mutex };
        auto            sink = findSink<T>(query);
        return sink ? sink->getSnapshotPoller(std::forward<M>(matcher), delay, block) : nullptr;
    }

    template<typename T, StreamCallback<T> Callback>
    bool
    registerStreamingCallback(const DataSinkQuery &query, std::size_t maxChunkSize, Callback &&callback) {
        std::lock_guard lg{ _mutex };
        auto            sink = findSink<T>(query);
        if (!sink) {
            return false;
        }

        sink->registerStreamingCallback(maxChunkSize, std::forward<Callback>(callback));
        return true;
    }

    template<typename T, DataSetCallback<T> Callback, TriggerMatcher M>
    bool
    registerTriggerCallback(const DataSinkQuery &query, M &&matcher, std::size_t preSamples, std::size_t postSamples, Callback &&callback) {
        std::lock_guard lg{ _mutex };
        auto            sink = findSink<T>(query);
        if (!sink) {
            return false;
        }

        sink->registerTriggerCallback(std::forward<M>(matcher), preSamples, postSamples, std::forward<Callback>(callback));
        return true;
    }

    template<typename T, DataSetCallback<T> Callback, TriggerMatcher M>
    bool
    registerMultiplexedCallback(const DataSinkQuery &query, M &&matcher, std::size_t maximumWindowSize, Callback &&callback) {
        std::lock_guard lg{ _mutex };
        auto            sink = findSink<T>(query);
        if (!sink) {
            return false;
        }

        sink->registerMultiplexedCallback(std::forward<M>(matcher), maximumWindowSize, std::forward<Callback>(callback));
        return true;
    }

    template<typename T, DataSetCallback<T> Callback, TriggerMatcher M>
    bool
    registerSnapshotCallback(const DataSinkQuery &query, M &&matcher, std::chrono::nanoseconds delay, Callback &&callback) {
        std::lock_guard lg{ _mutex };
        auto            sink = findSink<T>(query);
        if (!sink) {
            return false;
        }

        sink->registerSnapshotCallback(std::forward<M>(matcher), delay, std::forward<Callback>(callback));
        return true;
    }

private:
    template<typename T>
    DataSink<T> *
    findSink(const DataSinkQuery &query) {
        auto matches = [&query](const std::any &v) {
            try {
                auto       sink              = std::any_cast<DataSink<T> *>(v);
                const auto sinkNameMatches   = !query._sink_name || *query._sink_name == sink->name;
                const auto signalNameMatches = !query._signal_name || *query._signal_name == sink->signal_name;
                return sinkNameMatches && signalNameMatches;
            } catch (...) {
                return false;
            }
        };

        const auto it = std::find_if(_sinks.begin(), _sinks.end(), matches);
        if (it == _sinks.end()) {
            return nullptr;
        }

        return std::any_cast<DataSink<T> *>(*it);
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
    const auto it = m.find(std::string(key));
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
 * synchronously (/asynchronously) if handled by the same (/different) sink block.
 *
 * @tparam T input sample type
 */
template<typename T>
class DataSink : public Block<DataSink<T>> {
    struct AbstractListener;

    static constexpr std::size_t                  _listener_buffer_size = 65536;
    std::deque<std::unique_ptr<AbstractListener>> _listeners;
    std::mutex                                    _listener_mutex;
    std::optional<gr::HistoryBuffer<T>>           _history;
    bool                                          _has_signal_info_from_settings = false;

public:
    Annotated<float, "sample rate", Doc<"signal sample rate">, Unit<"Hz">>           sample_rate = 1.f;
    Annotated<std::string, "signal name", Visible>                                   signal_name = "unknown signal";
    Annotated<std::string, "signal unit", Visible, Doc<"signal's physical SI unit">> signal_unit = "a.u.";
    Annotated<T, "signal min", Doc<"signal physical min. (e.g. DAQ) limit">>         signal_min  = std::numeric_limits<T>::lowest();
    Annotated<T, "signal max", Doc<"signal physical max. (e.g. DAQ) limit">>         signal_max  = std::numeric_limits<T>::max();

    PortIn<T, RequiredSamples<std::dynamic_extent, _listener_buffer_size>>           in;

    struct Poller {
        // TODO consider whether reusing port<T> here makes sense
        gr::CircularBuffer<T>             buffer       = gr::CircularBuffer<T>(_listener_buffer_size);
        decltype(buffer.new_reader())     reader       = buffer.new_reader();
        decltype(buffer.new_writer())     writer       = buffer.new_writer();
        gr::CircularBuffer<Tag>           tag_buffer   = gr::CircularBuffer<Tag>(1024);
        decltype(tag_buffer.new_reader()) tag_reader   = tag_buffer.new_reader();
        decltype(tag_buffer.new_writer()) tag_writer   = tag_buffer.new_writer();
        std::size_t                       samples_read = 0; // reader thread
        std::atomic<bool>                 finished     = false;
        std::atomic<std::size_t>          drop_count   = 0;

        template<typename Handler>
        [[nodiscard]] bool
        process(Handler fnc, std::size_t requested = std::numeric_limits<std::size_t>::max()) {
            const auto nProcess = std::min(reader.available(), requested);
            if (nProcess == 0) {
                return false;
            }

            const auto readData = reader.get(nProcess);
            if constexpr (requires { fnc(std::span<const T>(), std::span<const Tag>()); }) {
                const auto tags         = tag_reader.get();
                const auto it           = std::find_if_not(tags.begin(), tags.end(), [until = static_cast<int64_t>(samples_read + nProcess)](const auto &tag) { return tag.index < until; });
                auto       relevantTags = std::vector<Tag>(tags.begin(), it);
                for (auto &t : relevantTags) {
                    t.index -= static_cast<int64_t>(samples_read);
                }
                fnc(readData, std::span<const Tag>(relevantTags));
                std::ignore = tag_reader.consume(relevantTags.size());
            } else {
                std::ignore = tag_reader.consume(tag_reader.available());
                fnc(readData);
            }

            std::ignore = reader.consume(nProcess);
            samples_read += nProcess;
            return true;
        }
    };

    struct DataSetPoller {
        gr::CircularBuffer<DataSet<T>> buffer     = gr::CircularBuffer<DataSet<T>>(_listener_buffer_size);
        decltype(buffer.new_reader())  reader     = buffer.new_reader();
        decltype(buffer.new_writer())  writer     = buffer.new_writer();

        std::atomic<bool>              finished   = false;
        std::atomic<std::size_t>       drop_count = 0;

        [[nodiscard]] bool
        process(std::invocable<std::span<DataSet<T>>> auto fnc, std::size_t requested = std::numeric_limits<std::size_t>::max()) {
            const auto nProcess = std::min(reader.available(), requested);
            if (nProcess == 0) {
                return false;
            }

            const auto readData = reader.get(nProcess);
            fnc(readData);
            std::ignore = reader.consume(nProcess);
            return true;
        }
    };

    DataSink() { DataSinkRegistry::instance().registerSink(this); }

    ~DataSink() {
        stop();
        DataSinkRegistry::instance().unregisterSink(this);
    }

    void
    settingsChanged(const property_map & /*oldSettings*/, const property_map &newSettings) {
        if (applySignalInfo(newSettings)) {
            _has_signal_info_from_settings = true;
        }
    }

    std::shared_ptr<Poller>
    getStreamingPoller(BlockingMode blockMode = BlockingMode::Blocking) {
        std::lock_guard lg(_listener_mutex);
        const auto      block   = blockMode == BlockingMode::Blocking;
        auto            handler = std::make_shared<Poller>();
        addListener(std::make_unique<ContinuousListener<gr::meta::null_type>>(handler, block, *this), block);
        return handler;
    }

    template<TriggerMatcher M>
    std::shared_ptr<DataSetPoller>
    getTriggerPoller(M &&matcher, std::size_t preSamples, std::size_t postSamples, BlockingMode blockMode = BlockingMode::Blocking) {
        const auto      block   = blockMode == BlockingMode::Blocking;
        auto            handler = std::make_shared<DataSetPoller>();
        std::lock_guard lg(_listener_mutex);
        addListener(std::make_unique<TriggerListener<gr::meta::null_type, M>>(std::forward<M>(matcher), handler, preSamples, postSamples, block), block);
        ensureHistorySize(preSamples);
        return handler;
    }

    template<TriggerMatcher M>
    std::shared_ptr<DataSetPoller>
    getMultiplexedPoller(M &&matcher, std::size_t maximumWindowSize, BlockingMode blockMode = BlockingMode::Blocking) {
        std::lock_guard lg(_listener_mutex);
        const auto      block   = blockMode == BlockingMode::Blocking;
        auto            handler = std::make_shared<DataSetPoller>();
        addListener(std::make_unique<MultiplexedListener<gr::meta::null_type, M>>(std::forward<M>(matcher), maximumWindowSize, handler, block), block);
        return handler;
    }

    template<TriggerMatcher M>
    std::shared_ptr<DataSetPoller>
    getSnapshotPoller(M &&matcher, std::chrono::nanoseconds delay, BlockingMode blockMode = BlockingMode::Blocking) {
        const auto      block   = blockMode == BlockingMode::Blocking;
        auto            handler = std::make_shared<DataSetPoller>();
        std::lock_guard lg(_listener_mutex);
        addListener(std::make_unique<SnapshotListener<gr::meta::null_type, M>>(std::forward<M>(matcher), delay, handler, block), block);
        return handler;
    }

    template<StreamCallback<T> Callback>
    void
    registerStreamingCallback(std::size_t maxChunkSize, Callback &&callback) {
        addListener(std::make_unique<ContinuousListener<Callback>>(maxChunkSize, std::forward<Callback>(callback), *this), false);
    }

    template<TriggerMatcher M, DataSetCallback<T> Callback>
    void
    registerTriggerCallback(M &&matcher, std::size_t preSamples, std::size_t postSamples, Callback &&callback) {
        addListener(std::make_unique<TriggerListener<Callback, M>>(std::forward<M>(matcher), preSamples, postSamples, std::forward<Callback>(callback)), false);
        ensureHistorySize(preSamples);
    }

    template<TriggerMatcher M, DataSetCallback<T> Callback>
    void
    registerMultiplexedCallback(M &&matcher, std::size_t maximumWindowSize, Callback &&callback) {
        std::lock_guard lg(_listener_mutex);
        addListener(std::make_unique<MultiplexedListener<Callback, M>>(std::forward<M>(matcher), maximumWindowSize, std::forward<Callback>(callback)), false);
    }

    template<TriggerMatcher M, DataSetCallback<T> Callback>
    void
    registerSnapshotCallback(M &&matcher, std::chrono::nanoseconds delay, Callback &&callback) {
        std::lock_guard lg(_listener_mutex);
        addListener(std::make_unique<SnapshotListener<Callback, M>>(std::forward<M>(matcher), delay, std::forward<Callback>(callback)), false);
    }

    // TODO this code should be called at the end of graph processing
    void
    stop() noexcept {
        std::lock_guard lg(_listener_mutex);
        for (auto &listener : _listeners) {
            listener->stop();
        }
    }

    [[nodiscard]] work::Status
    processBulk(std::span<const T> inData) noexcept {
        std::optional<property_map> tagData;
        if (this->input_tags_present()) {
            assert(this->input_tags()[0].index == 0);
            tagData = this->input_tags()[0].map;
            // signal info from settings overrides info from tags
            if (!_has_signal_info_from_settings) {
                applySignalInfo(this->input_tags()[0].map);
            }
        }

        {
            std::lock_guard lg(_listener_mutex); // TODO review/profile if a lock-free data structure should be used here
            const auto      historyView = _history ? _history->get_span(0) : std::span<const T>();
            std::erase_if(_listeners, [](const auto &l) { return l->expired; });
            for (auto &listener : _listeners) {
                listener->process(historyView, inData, tagData);
            }
            if (_history) {
                // store potential pre-samples for triggers at the beginning of the next chunk
                const auto toWrite = std::min(inData.size(), _history->capacity());
                _history->push_back_bulk(inData.last(toWrite));
            }
        }

        return work::Status::OK;
    }

private:
    bool
    applySignalInfo(const property_map &properties) {
        try {
            const auto rate_ = detail::get<float>(properties, tag::SAMPLE_RATE.key());
            const auto name_ = detail::get<std::string>(properties, tag::SIGNAL_NAME.key());
            const auto unit_ = detail::get<std::string>(properties, tag::SIGNAL_UNIT.key());
            const auto min_  = detail::get<T>(properties, tag::SIGNAL_MIN.key());
            const auto max_  = detail::get<T>(properties, tag::SIGNAL_MAX.key());

            // commit
            if (rate_) {
                sample_rate = *rate_;
            }
            if (name_) {
                signal_name = *name_;
            }
            if (unit_) {
                signal_unit = *unit_;
            }
            if (min_) {
                signal_min = *min_;
            }
            if (max_) {
                signal_max = *max_;
            }

            // forward to listeners
            if (rate_ || name_ || unit_ || min_ || max_) {
                const auto      dstempl = makeDataSetTemplate();

                std::lock_guard lg{ _listener_mutex };
                for (auto &l : _listeners) {
                    if (rate_) {
                        l->applySampleRate(sample_rate);
                    }
                    if (name_ || unit_ || min_ || max_) {
                        l->setDataSetTemplate(dstempl);
                    }
                }
            }
            return name_ || unit_ || min_ || max_;
        } catch (const std::bad_variant_access &) {
            // TODO log?
            return false;
        }
    }

    DataSet<T>
    makeDataSetTemplate() const {
        DataSet<T> dstempl;
        dstempl.signal_names  = { signal_name };
        dstempl.signal_units  = { signal_unit };
        dstempl.signal_ranges = { { signal_min, signal_max } };
        return dstempl;
    }

    void
    ensureHistorySize(std::size_t new_size) {
        const auto old_size = _history ? _history->capacity() : std::size_t{ 0 };
        if (new_size <= old_size) {
            return;
        }
        // TODO Important!
        //  - History size must be limited to avoid users causing OOM
        //  - History should shrink again

        // transitional, do not reallocate/copy, but create a shared buffer with size N,
        // and a per-listener history buffer where more than N samples is needed.
        auto new_history = gr::HistoryBuffer<T>(new_size);
        if (_history) {
            new_history.push_back_bulk(_history->begin(), _history->end());
        }
        _history = new_history;
    }

    void
    addListener(std::unique_ptr<AbstractListener> &&l, bool block) {
        l->setDataSetTemplate(makeDataSetTemplate());
        l->applySampleRate(sample_rate);
        if (block) {
            _listeners.push_back(std::move(l));
        } else {
            _listeners.push_front(std::move(l));
        }
    }

    struct AbstractListener {
        bool expired                = false;

        virtual ~AbstractListener() = default;

        void
        setExpired() {
            expired = true;
        }

        virtual void
        applySampleRate(float /*sample_rate*/) {}

        virtual void
        setDataSetTemplate(DataSet<T>) {}

        virtual void
        process(std::span<const T> history, std::span<const T> data, std::optional<property_map> tagData0)
                = 0;
        virtual void
        stop() = 0;
    };

    template<typename Callback>
    struct ContinuousListener : public AbstractListener {
        static constexpr auto hasCallback       = !std::is_same_v<Callback, gr::meta::null_type>;
        static constexpr auto callbackTakesTags = std::is_invocable_v<Callback, std::span<const T>, std::span<const Tag>>
                                               || std::is_invocable_v<Callback, std::span<const T>, std::span<const Tag>, const DataSink<T> &>;

        const DataSink<T> &parent_sink;
        bool               block           = false;
        std::size_t        samples_written = 0;

        // callback-only
        std::size_t      buffer_fill = 0;
        std::vector<T>   buffer;
        std::vector<Tag> tag_buffer;

        // polling-only
        std::weak_ptr<Poller> polling_handler = {};

        Callback              callback;

        template<typename CallbackFW>
        explicit ContinuousListener(std::size_t maxChunkSize, CallbackFW &&c, const DataSink<T> &parent) : parent_sink(parent), buffer(maxChunkSize), callback{ std::forward<CallbackFW>(c) } {}

        explicit ContinuousListener(std::shared_ptr<Poller> poller, bool doBlock, const DataSink<T> &parent) : parent_sink(parent), block(doBlock), polling_handler{ std::move(poller) } {}

        inline void
        callCallback(std::span<const T> data, std::span<const Tag> tags) {
            if constexpr (std::is_invocable_v<Callback, std::span<const T>, std::span<const Tag>, const DataSink<T> &>) {
                callback(std::move(data), std::move(tags), parent_sink);
            } else if constexpr (std::is_invocable_v<Callback, std::span<const T>, std::span<const Tag>>) {
                callback(std::move(data), std::move(tags));
            } else {
                callback(std::move(data));
            }
        }

        void
        process(std::span<const T>, std::span<const T> data, std::optional<property_map> tagData0) override {
            using namespace gr::detail;

            if constexpr (hasCallback) {
                // if there's pending data, fill buffer and send out
                if (buffer_fill > 0) {
                    const auto n = std::min(data.size(), buffer.size() - buffer_fill);
                    detail::copy_span(data.first(n), std::span(buffer).subspan(buffer_fill, n));
                    if constexpr (callbackTakesTags) {
                        if (tagData0) {
                            tag_buffer.push_back({ static_cast<Tag::signed_index_type>(buffer_fill), *tagData0 });
                            tagData0.reset();
                        }
                    }
                    buffer_fill += n;
                    if (buffer_fill == buffer.size()) {
                        callCallback(std::span(buffer), std::span(tag_buffer));
                        samples_written += buffer.size();
                        buffer_fill = 0;
                        tag_buffer.clear();
                    }

                    data = data.last(data.size() - n);
                }

                // send out complete chunks directly
                while (data.size() >= buffer.size()) {
                    if constexpr (callbackTakesTags) {
                        std::vector<Tag> tags;
                        if (tagData0) {
                            tags.push_back({ 0, std::move(*tagData0) });
                            tagData0.reset();
                        }
                        callCallback(data.first(buffer.size()), std::span(tags));
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
                    if constexpr (callbackTakesTags) {
                        if (tagData0) {
                            tag_buffer.push_back({ 0, std::move(*tagData0) });
                        }
                    }
                }
            } else {
                auto poller = polling_handler.lock();
                if (!poller) {
                    this->setExpired();
                    return;
                }

                const auto toWrite = block ? data.size() : std::min(data.size(), poller->writer.available());

                if (toWrite > 0) {
                    if (tagData0) {
                        auto tw = poller->tag_writer.reserve_output_range(1);
                        tw[0]   = { static_cast<Tag::signed_index_type>(samples_written), std::move(*tagData0) };
                        tw.publish(1);
                    }
                    auto writeData = poller->writer.reserve_output_range(toWrite);
                    detail::copy_span(data.first(toWrite), std::span(writeData));
                    writeData.publish(writeData.size());
                }
                poller->drop_count += data.size() - toWrite;
                samples_written += toWrite;
            }
        }

        void
        stop() override {
            if constexpr (hasCallback) {
                if (buffer_fill > 0) {
                    callCallback(std::span(buffer).first(buffer_fill), std::span(tag_buffer));
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

    struct PendingWindow {
        DataSet<T>  dataset;
        std::size_t pending_post_samples = 0;
    };

    template<typename Callback, TriggerMatcher M>
    struct TriggerListener : public AbstractListener {
        bool                         block       = false;
        std::size_t                  preSamples  = 0;
        std::size_t                  postSamples = 0;

        DataSet<T>                   dataset_template;
        M                            trigger_matcher = {};
        std::deque<PendingWindow>    pending_trigger_windows; // triggers that still didn't receive all their data
        std::weak_ptr<DataSetPoller> polling_handler = {};

        Callback                     callback;

        template<TriggerMatcher Matcher>
        explicit TriggerListener(Matcher &&matcher, std::shared_ptr<DataSetPoller> handler, std::size_t pre, std::size_t post, bool doBlock)
            : block(doBlock), preSamples(pre), postSamples(post), trigger_matcher(std::forward<Matcher>(matcher)), polling_handler{ std::move(handler) } {}

        template<typename CallbackFW, TriggerMatcher Matcher>
        explicit TriggerListener(Matcher &&matcher, std::size_t pre, std::size_t post, CallbackFW &&cb)
            : preSamples(pre), postSamples(post), trigger_matcher(std::forward<Matcher>(matcher)), callback{ std::forward<CallbackFW>(cb) } {}

        void
        setDataSetTemplate(DataSet<T> dst) override {
            dataset_template = std::move(dst);
        }

        inline void
        publishDataSet(DataSet<T> &&data) {
            if constexpr (!std::is_same_v<Callback, gr::meta::null_type>) {
                callback(std::move(data));
            } else {
                auto poller = polling_handler.lock();
                if (!poller) {
                    this->setExpired();
                    return;
                }

                auto writeData = poller->writer.reserve_output_range(1);
                if (block) {
                    writeData[0] = std::move(data);
                    writeData.publish(1);
                } else {
                    if (poller->writer.available() > 0) {
                        writeData[0] = std::move(data);
                        writeData.publish(1);
                    } else {
                        poller->drop_count++;
                    }
                }
            }
        }

        void
        process(std::span<const T> history, std::span<const T> inData, std::optional<property_map> tagData0) override {
            if (tagData0 && trigger_matcher(Tag{ 0, *tagData0 }) == TriggerMatchResult::Matching) {
                DataSet<T> dataset = dataset_template;
                dataset.signal_values.reserve(preSamples + postSamples); // TODO maybe make the circ. buffer smaller but preallocate these

                const auto preSampleView = history.last(std::min(preSamples, history.size()));
                dataset.signal_values.insert(dataset.signal_values.end(), preSampleView.begin(), preSampleView.end());

                dataset.timing_events = { { { static_cast<Tag::signed_index_type>(preSampleView.size()), *tagData0 } } };
                pending_trigger_windows.push_back({ .dataset = std::move(dataset), .pending_post_samples = postSamples });
            }

            auto window = pending_trigger_windows.begin();
            while (window != pending_trigger_windows.end()) {
                const auto postSampleView = inData.first(std::min(window->pending_post_samples, inData.size()));
                window->dataset.signal_values.insert(window->dataset.signal_values.end(), postSampleView.begin(), postSampleView.end());
                window->pending_post_samples -= postSampleView.size();

                if (window->pending_post_samples == 0) {
                    this->publishDataSet(std::move(window->dataset));
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
                    this->publishDataSet(std::move(window.dataset));
                }
            }
            pending_trigger_windows.clear();
            if (auto p = polling_handler.lock()) {
                p->finished = true;
            }
        }
    };

    template<typename Callback, TriggerMatcher M>
    struct MultiplexedListener : public AbstractListener {
        bool                         block = false;
        M                            matcher;
        DataSet<T>                   dataset_template;
        std::optional<DataSet<T>>    pending_dataset;
        std::size_t                  maximumWindowSize;
        std::weak_ptr<DataSetPoller> polling_handler = {};
        Callback                     callback;

        template<typename CallbackFW, TriggerMatcher Matcher>
        explicit MultiplexedListener(Matcher &&matcher_, std::size_t maxWindowSize, CallbackFW &&cb)
            : matcher(std::forward<Matcher>(matcher_)), maximumWindowSize(maxWindowSize), callback(std::forward<CallbackFW>(cb)) {}

        template<TriggerMatcher Matcher>
        explicit MultiplexedListener(Matcher &&matcher_, std::size_t maxWindowSize, std::shared_ptr<DataSetPoller> handler, bool doBlock)
            : block(doBlock), matcher(std::forward<Matcher>(matcher_)), maximumWindowSize(maxWindowSize), polling_handler{ std::move(handler) } {}

        void
        setDataSetTemplate(DataSet<T> dst) override {
            dataset_template = std::move(dst);
        }

        inline void
        publishDataSet(DataSet<T> &&data) {
            if constexpr (!std::is_same_v<Callback, gr::meta::null_type>) {
                callback(std::move(data));
            } else {
                auto poller = polling_handler.lock();
                if (!poller) {
                    this->setExpired();
                    return;
                }

                auto writeData = poller->writer.reserve_output_range(1);
                if (block) {
                    writeData[0] = std::move(data);
                    writeData.publish(1);
                } else {
                    if (poller->writer.available() > 0) {
                        writeData[0] = std::move(data);
                        writeData.publish(1);
                    } else {
                        poller->drop_count++;
                    }
                }
            }
        }

        void
        process(std::span<const T>, std::span<const T> inData, std::optional<property_map> tagData0) override {
            if (tagData0) {
                const auto obsr = matcher(Tag{ 0, *tagData0 });
                if (obsr == TriggerMatchResult::NotMatching || obsr == TriggerMatchResult::Matching) {
                    if (pending_dataset) {
                        if (obsr == TriggerMatchResult::NotMatching) {
                            pending_dataset->timing_events[0].push_back({ static_cast<Tag::signed_index_type>(pending_dataset->signal_values.size()), *tagData0 });
                        }
                        this->publishDataSet(std::move(*pending_dataset));
                        pending_dataset.reset();
                    }
                }
                if (obsr == TriggerMatchResult::Matching) {
                    pending_dataset = dataset_template;
                    pending_dataset->signal_values.reserve(maximumWindowSize); // TODO might be too much?
                    pending_dataset->timing_events = { { { 0, *tagData0 } } };
                }
            }
            if (pending_dataset) {
                const auto toWrite = std::min(inData.size(), maximumWindowSize - pending_dataset->signal_values.size());
                const auto view    = inData.first(toWrite);
                pending_dataset->signal_values.insert(pending_dataset->signal_values.end(), view.begin(), view.end());

                if (pending_dataset->signal_values.size() == maximumWindowSize) {
                    this->publishDataSet(std::move(*pending_dataset));
                    pending_dataset.reset();
                }
            }
        }

        void
        stop() override {
            if (pending_dataset) {
                this->publishDataSet(std::move(*pending_dataset));
                pending_dataset.reset();
            }
            if (auto p = polling_handler.lock()) {
                p->finished = true;
            }
        }
    };

    struct PendingSnapshot {
        property_map tag_data;
        std::size_t  delay           = 0;
        std::size_t  pending_samples = 0;
    };

    template<typename Callback, TriggerMatcher M>
    struct SnapshotListener : public AbstractListener {
        bool                         block = false;
        std::chrono::nanoseconds     time_delay;
        std::size_t                  sample_delay = 0;
        DataSet<T>                   dataset_template;
        M                            trigger_matcher = {};
        std::deque<PendingSnapshot>  pending;
        std::weak_ptr<DataSetPoller> polling_handler = {};
        Callback                     callback;

        template<TriggerMatcher Matcher>
        explicit SnapshotListener(Matcher &&matcher, std::chrono::nanoseconds delay, std::shared_ptr<DataSetPoller> poller, bool doBlock)
            : block(doBlock), time_delay(delay), trigger_matcher(std::forward<Matcher>(matcher)), polling_handler{ std::move(poller) } {}

        template<typename CallbackFW, TriggerMatcher Matcher>
        explicit SnapshotListener(Matcher &&matcher, std::chrono::nanoseconds delay, CallbackFW &&cb)
            : time_delay(delay), trigger_matcher(std::forward<Matcher>(matcher)), callback(std::forward<CallbackFW>(cb)) {}

        void
        setDataSetTemplate(DataSet<T> dst) override {
            dataset_template = std::move(dst);
        }

        void
        applySampleRate(float rateHz) override {
            sample_delay = static_cast<std::size_t>(std::round(std::chrono::duration_cast<std::chrono::duration<float>>(time_delay).count() * rateHz));
            // TODO do we need to update the requested_samples of pending here? (considering both old and new time_delay)
        }

        inline void
        publishDataSet(DataSet<T> &&data) {
            if constexpr (!std::is_same_v<Callback, gr::meta::null_type>) {
                callback(std::move(data));
            } else {
                auto poller = polling_handler.lock();
                if (!poller) {
                    this->setExpired();
                    return;
                }

                auto writeData = poller->writer.reserve_output_range(1);
                if (block) {
                    writeData[0] = std::move(data);
                    writeData.publish(1);
                } else {
                    if (poller->writer.available() > 0) {
                        writeData[0] = std::move(data);
                        writeData.publish(1);
                    } else {
                        poller->drop_count++;
                    }
                }
            }
        }

        void
        process(std::span<const T>, std::span<const T> inData, std::optional<property_map> tagData0) override {
            if (tagData0 && trigger_matcher({ 0, *tagData0 }) == TriggerMatchResult::Matching) {
                auto new_pending = PendingSnapshot{ *tagData0, sample_delay, sample_delay };
                // make sure pending is sorted by number of pending_samples (insertion might be not at end if sample rate decreased; TODO unless we adapt them in applySampleRate, see there)
                auto rit = std::find_if(pending.rbegin(), pending.rend(), [delay = sample_delay](const auto &other) { return other.pending_samples < delay; });
                pending.insert(rit.base(), std::move(new_pending));
            }

            auto it = pending.begin();
            while (it != pending.end()) {
                if (it->pending_samples >= inData.size()) {
                    it->pending_samples -= inData.size();
                    break;
                }

                DataSet<T> dataset    = dataset_template;
                dataset.timing_events = { { { -static_cast<Tag::signed_index_type>(it->delay), std::move(it->tag_data) } } };
                dataset.signal_values = { inData[it->pending_samples] };
                this->publishDataSet(std::move(dataset));

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

} // namespace gr::basic

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (gr::basic::DataSink<T>), in, sample_rate, signal_name, signal_unit, signal_min, signal_max);

#endif
