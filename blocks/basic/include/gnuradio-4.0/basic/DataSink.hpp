#ifndef GNURADIO_DATA_SINK_HPP
#define GNURADIO_DATA_SINK_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>

#include <any>
#include <chrono>
#include <deque>
#include <limits>

namespace gr::basic {

enum class BlockingMode { NonBlocking, Blocking };

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

// clang-format on

struct DataSinkQuery {
    std::optional<std::string> _sink_name;
    std::optional<std::string> _signal_name;

    static DataSinkQuery signalName(std::string_view name) { return {{}, std::string{name}}; }

    static DataSinkQuery sinkName(std::string_view name) { return {std::string{name}, {}}; }
};

class DataSinkRegistry {
    std::mutex                      _mutex;
    std::vector<std::any>           _sinks;
    std::map<std::string, std::any> _sink_by_signal_name;

public:
    // TODO this shouldn't be a singleton but associated with the flow graph (?)
    // TODO reconsider mutex usage when moving to the graph
    static DataSinkRegistry& instance() {
        static DataSinkRegistry s_instance;
        return s_instance;
    }

    template<typename T>
    void registerSink(DataSink<T>* sink) {
        std::lock_guard lg{_mutex};
        _sinks.push_back(sink);
        _sink_by_signal_name[sink->signal_name] = sink;
    }

    template<typename T>
    void unregisterSink(DataSink<T>* sink) {
        std::lock_guard lg{_mutex};
        std::erase_if(_sinks, [sink](const std::any& v) -> bool {
            auto ptr = std::any_cast<DataSink<T>*>(v);
            return ptr && ptr == sink;
        });
        _sink_by_signal_name.erase(sink->signal_name);
    }

    template<typename T>
    void updateSignalName(DataSink<T>* sink, std::string_view oldName, std::string_view newName) {
        std::lock_guard lg{_mutex};
        _sink_by_signal_name.erase(std::string(oldName));
        _sink_by_signal_name[std::string(newName)] = sink;
    }

    template<typename T>
    std::shared_ptr<typename DataSink<T>::Poller> getStreamingPoller(const DataSinkQuery& query, BlockingMode block = BlockingMode::Blocking) {
        std::lock_guard lg{_mutex};
        auto            sink = findSink<T>(query);
        return sink ? sink->getStreamingPoller(block) : nullptr;
    }

    template<typename T, trigger::Matcher M>
    std::shared_ptr<typename DataSink<T>::DataSetPoller> getTriggerPoller(const DataSinkQuery& query, M&& matcher, std::size_t preSamples, std::size_t postSamples, BlockingMode block = BlockingMode::Blocking) {
        std::lock_guard lg{_mutex};
        auto            sink = findSink<T>(query);
        return sink ? sink->getTriggerPoller(std::forward<M>(matcher), preSamples, postSamples, block) : nullptr;
    }

    template<typename T, trigger::Matcher M>
    std::shared_ptr<typename DataSink<T>::DataSetPoller> getMultiplexedPoller(const DataSinkQuery& query, M&& matcher, std::size_t maximumWindowSize, BlockingMode block = BlockingMode::Blocking) {
        std::lock_guard lg{_mutex};
        auto            sink = findSink<T>(query);
        return sink ? sink->getMultiplexedPoller(std::forward<M>(matcher), maximumWindowSize, block) : nullptr;
    }

    template<typename T, trigger::Matcher M>
    std::shared_ptr<typename DataSink<T>::DataSetPoller> getSnapshotPoller(const DataSinkQuery& query, M&& matcher, std::chrono::nanoseconds delay, BlockingMode block = BlockingMode::Blocking) {
        std::lock_guard lg{_mutex};
        auto            sink = findSink<T>(query);
        return sink ? sink->getSnapshotPoller(std::forward<M>(matcher), delay, block) : nullptr;
    }

    template<typename T, StreamCallback<T> Callback>
    bool registerStreamingCallback(const DataSinkQuery& query, std::size_t maxChunkSize, Callback&& callback) {
        std::lock_guard lg{_mutex};
        auto            sink = findSink<T>(query);
        if (!sink) {
            return false;
        }

        sink->registerStreamingCallback(maxChunkSize, std::forward<Callback>(callback));
        return true;
    }

    template<typename T, DataSetCallback<T> Callback, trigger::Matcher M>
    bool registerTriggerCallback(const DataSinkQuery& query, M&& matcher, std::size_t preSamples, std::size_t postSamples, Callback&& callback) {
        std::lock_guard lg{_mutex};
        auto            sink = findSink<T>(query);
        if (!sink) {
            return false;
        }

        sink->registerTriggerCallback(std::forward<M>(matcher), preSamples, postSamples, std::forward<Callback>(callback));
        return true;
    }

    template<typename T, DataSetCallback<T> Callback, trigger::Matcher M>
    bool registerMultiplexedCallback(const DataSinkQuery& query, M&& matcher, std::size_t maximumWindowSize, Callback&& callback) {
        std::lock_guard lg{_mutex};
        auto            sink = findSink<T>(query);
        if (!sink) {
            return false;
        }

        sink->registerMultiplexedCallback(std::forward<M>(matcher), maximumWindowSize, std::forward<Callback>(callback));
        return true;
    }

    template<typename T, DataSetCallback<T> Callback, trigger::Matcher M>
    bool registerSnapshotCallback(const DataSinkQuery& query, M&& matcher, std::chrono::nanoseconds delay, Callback&& callback) {
        std::lock_guard lg{_mutex};
        auto            sink = findSink<T>(query);
        if (!sink) {
            return false;
        }

        sink->registerSnapshotCallback(std::forward<M>(matcher), delay, std::forward<Callback>(callback));
        return true;
    }

private:
    template<typename T>
    DataSink<T>* findSink(const DataSinkQuery& query) {
        if (query._signal_name) {
            const auto it = _sink_by_signal_name.find(*query._signal_name);
            if (it != _sink_by_signal_name.end()) {
                try {
                    return std::any_cast<DataSink<T>*>(it->second);
                } catch (...) {
                    return nullptr;
                }
            }
            return nullptr;
        }
        auto sinkNameMatches = [&query](const std::any& v) {
            try {
                const auto sink = std::any_cast<DataSink<T>*>(v);
                return query._sink_name == sink->name;
            } catch (...) {
                return false;
            }
        };

        const auto it = std::find_if(_sinks.begin(), _sinks.end(), sinkNameMatches);
        if (it == _sinks.end()) {
            return nullptr;
        }

        return std::any_cast<DataSink<T>*>(*it);
    }
};

namespace detail {
template<typename U>
[[nodiscard]] constexpr inline std::optional<U> getProperty(const property_map& m, std::string_view key) {
    const auto it = m.find(std::string(key));
    if (it == m.end()) {
        return std::nullopt;
    }
    try {
        return std::get<U>(it->second);
    } catch (const std::bad_variant_access&) {
        return std::nullopt;
    }
};

struct Metadata {
    float       sampleRate;
    std::string signalName;
    std::string signalUnit;
    float       signalMin;
    float       signalMax;

    property_map toTagMap() const { return {{std::string(tag::SIGNAL_RATE.shortKey()), sampleRate}, {std::string(tag::SIGNAL_NAME.shortKey()), signalName}, {std::string(tag::SIGNAL_UNIT.shortKey()), signalUnit}, {std::string(tag::SIGNAL_MIN.shortKey()), signalMin}, {std::string(tag::SIGNAL_MAX.shortKey()), signalMax}}; }
};

inline std::optional<property_map> tagAndMetadata(const std::optional<property_map>& tagData, const std::optional<Metadata>& metadata) {
    if (!tagData && !metadata) {
        return std::nullopt;
    }

    property_map merged;
    if (metadata) {
        merged = metadata->toTagMap();
    }
    if (tagData) {
        merged.insert(tagData->begin(), tagData->end());
    }

    return merged;
}

template<typename T>
[[nodiscard]] inline DataSet<T> makeDataSetTemplate(Metadata metadata) {
    DataSet<T> tmpl;
    tmpl.signal_names  = {std::move(metadata.signalName)};
    tmpl.signal_units  = {std::move(metadata.signalUnit)};
    tmpl.signal_ranges = {{static_cast<T>(metadata.signalMin), static_cast<T>(metadata.signalMax)}};
    return tmpl;
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
    using Description = Doc<R""(@brief generic data sink for exporting arbitrary-typed streams to non-GR C++ APIs.

Each sink registers with a (user-defined/exchangeable) global registry that can be
queried by the non-GR caller to find the sink responsible for a given signal name, etc.
and either retrieve a poller handler that allows asynchronous data from a different thread,
or register a callback that is invoked by the sink if the user-conditions are met.

<pre>
@code
        ╔═══════════════╗
   in0 ━╢   data sink   ║                      ┌──── caller ────┐
(err0) ━╢ (opt. error)  ║                      │                │
        ║               ║  retrieve poller or  │ (custom non-GR │
        ║ :signal_name  ║←--------------------→│  user code...) │
        ║ :signal_unit  ║  register            │                │
        ║ :...          ║  callback function   └───┬────────────┘
        ╚═ GR block ═╤══╝                          │
                     │                             │
                     │                             │
                     │      ╭─registry─╮           │
           register/ │      ╞══════════╡           │ queries for specific
         deregister  ╰─────→│ [sinks]  │←──────────╯ signal_info_t list/criteria
                            ╞══════════╡
                            ╰──────────╯

</pre>
Pollers can be configured to be blocking, i.e. blocks the flow-graph
if data is not being retrieved in time, or non-blocking, i.e. data being dropped when
the user-defined buffer size is full.
N.B. due to the nature of the GR scheduler, signals from the same sink are notified
synchronously (/asynchronously) if handled by the same (/different) sink block.

@tparam T input sample type
)"">;
    struct AbstractListener;

    static constexpr std::size_t                  _listener_buffer_size = 65536;
    std::deque<std::unique_ptr<AbstractListener>> _listeners;
    bool                                          _listeners_finished = false;
    std::mutex                                    _listener_mutex;
    std::optional<gr::HistoryBuffer<T>>           _history;
    bool                                          _registered = false;

public:
    PortIn<T, RequiredSamples<std::dynamic_extent, _listener_buffer_size>> in;

    Annotated<float, "sample rate", Doc<"signal sample rate">, Unit<"Hz">>           sample_rate = 1.f;
    Annotated<std::string, "signal name", Visible>                                   signal_name = "unknown signal";
    Annotated<std::string, "signal unit", Visible, Doc<"signal's physical SI unit">> signal_unit = "a.u.";
    Annotated<float, "signal min", Doc<"signal physical min. (e.g. DAQ) limit">>     signal_min  = -1.0f;
    Annotated<float, "signal max", Doc<"signal physical max. (e.g. DAQ) limit">>     signal_max  = +1.0f;

    using Block<DataSink<T>>::Block; // needed to inherit mandatory base-class Block(property_map) constructor

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
        [[nodiscard]] bool process(Handler fnc, std::size_t requested = std::numeric_limits<std::size_t>::max()) {
            const auto nProcess = std::min(reader.available(), requested);
            if (nProcess == 0) {
                return false;
            }

            auto readData = reader.get(nProcess);
            if constexpr (requires { fnc(std::span<const T>(), std::span<const Tag>()); }) {
                auto       tags         = tag_reader.get();
                const auto it           = std::find_if_not(tags.begin(), tags.end(), [until = static_cast<int64_t>(samples_read + nProcess)](const auto& tag) { return tag.index < until; });
                auto       relevantTags = std::vector<Tag>(tags.begin(), it);
                for (auto& t : relevantTags) {
                    t.index -= static_cast<int64_t>(samples_read);
                }
                fnc(readData, std::span<const Tag>(relevantTags));
                std::ignore = tags.consume(relevantTags.size());
            } else {
                auto tags   = tag_reader.get();
                std::ignore = tags.consume(tags.size());
                fnc(readData);
            }

            std::ignore = readData.consume(nProcess);
            samples_read += nProcess;
            return true;
        }
    };

    struct DataSetPoller {
        gr::CircularBuffer<DataSet<T>> buffer = gr::CircularBuffer<DataSet<T>>(_listener_buffer_size);
        decltype(buffer.new_reader())  reader = buffer.new_reader();
        decltype(buffer.new_writer())  writer = buffer.new_writer();

        std::atomic<bool>        finished   = false;
        std::atomic<std::size_t> drop_count = 0;

        [[nodiscard]] bool process(std::invocable<std::span<DataSet<T>>> auto fnc, std::size_t requested = std::numeric_limits<std::size_t>::max()) {
            const auto nProcess = std::min(reader.available(), requested);
            if (nProcess == 0) {
                return false;
            }

            auto readData = reader.get(nProcess);
            fnc(readData);
            std::ignore = readData.consume(nProcess);
            return true;
        }
    };

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/) {
        const auto oldSignalName = detail::getProperty<std::string>(oldSettings, "signal_name");
        if (oldSignalName != signal_name && _registered) {
            DataSinkRegistry::instance().updateSignalName(this, oldSignalName.value_or(""), signal_name);
        }
        std::lock_guard lg{_listener_mutex};
        for (auto& listener : _listeners) {
            listener->setMetadata(detail::Metadata{sample_rate, signal_name, signal_unit, signal_min, signal_max});
        }
    }

    std::shared_ptr<Poller> getStreamingPoller(BlockingMode blockMode = BlockingMode::Blocking) {
        const auto      block   = blockMode == BlockingMode::Blocking;
        auto            handler = std::make_shared<Poller>();
        std::lock_guard lg(_listener_mutex);
        handler->finished = _listeners_finished;
        addListener(std::make_unique<ContinuousListener<gr::meta::null_type>>(handler, block, *this), block);
        return handler;
    }

    template<trigger::Matcher M>
    std::shared_ptr<DataSetPoller> getTriggerPoller(M&& matcher, std::size_t preSamples, std::size_t postSamples, BlockingMode blockMode = BlockingMode::Blocking) {
        const auto      block   = blockMode == BlockingMode::Blocking;
        auto            handler = std::make_shared<DataSetPoller>();
        std::lock_guard lg(_listener_mutex);
        handler->finished = _listeners_finished;
        addListener(std::make_unique<TriggerListener<gr::meta::null_type, M>>(std::forward<M>(matcher), handler, preSamples, postSamples, block), block);
        ensureHistorySize(preSamples);
        return handler;
    }

    template<trigger::Matcher M>
    std::shared_ptr<DataSetPoller> getMultiplexedPoller(M&& matcher, std::size_t maximumWindowSize, BlockingMode blockMode = BlockingMode::Blocking) {
        std::lock_guard lg(_listener_mutex);
        const auto      block   = blockMode == BlockingMode::Blocking;
        auto            handler = std::make_shared<DataSetPoller>();
        addListener(std::make_unique<MultiplexedListener<gr::meta::null_type, M>>(std::forward<M>(matcher), maximumWindowSize, handler, block), block);
        return handler;
    }

    template<trigger::Matcher M>
    std::shared_ptr<DataSetPoller> getSnapshotPoller(M&& matcher, std::chrono::nanoseconds delay, BlockingMode blockMode = BlockingMode::Blocking) {
        const auto      block   = blockMode == BlockingMode::Blocking;
        auto            handler = std::make_shared<DataSetPoller>();
        std::lock_guard lg(_listener_mutex);
        addListener(std::make_unique<SnapshotListener<gr::meta::null_type, M>>(std::forward<M>(matcher), delay, handler, block), block);
        return handler;
    }

    template<StreamCallback<T> Callback>
    void registerStreamingCallback(std::size_t maxChunkSize, Callback&& callback) {
        addListener(std::make_unique<ContinuousListener<Callback>>(maxChunkSize, std::forward<Callback>(callback), *this), false);
    }

    template<trigger::Matcher M, DataSetCallback<T> Callback>
    void registerTriggerCallback(M&& matcher, std::size_t preSamples, std::size_t postSamples, Callback&& callback) {
        addListener(std::make_unique<TriggerListener<Callback, M>>(std::forward<M>(matcher), preSamples, postSamples, std::forward<Callback>(callback)), false);
        ensureHistorySize(preSamples);
    }

    template<trigger::Matcher M, DataSetCallback<T> Callback>
    void registerMultiplexedCallback(M&& matcher, std::size_t maximumWindowSize, Callback&& callback) {
        std::lock_guard lg(_listener_mutex);
        addListener(std::make_unique<MultiplexedListener<Callback, M>>(std::forward<M>(matcher), maximumWindowSize, std::forward<Callback>(callback)), false);
    }

    template<trigger::Matcher M, DataSetCallback<T> Callback>
    void registerSnapshotCallback(M&& matcher, std::chrono::nanoseconds delay, Callback&& callback) {
        std::lock_guard lg(_listener_mutex);
        addListener(std::make_unique<SnapshotListener<Callback, M>>(std::forward<M>(matcher), delay, std::forward<Callback>(callback)), false);
    }

    void start() noexcept {
        DataSinkRegistry::instance().registerSink(this);
        _registered = true;
    }

    void stop() noexcept {
        DataSinkRegistry::instance().unregisterSink(this);
        _registered = false;
        std::lock_guard lg(_listener_mutex);
        for (auto& listener : _listeners) {
            listener->stop();
        }
        _listeners_finished = true;
    }

    [[nodiscard]] work::Status processBulk(std::span<const T>& inData) noexcept {
        std::optional<property_map> tagData;
        if (this->inputTagsPresent()) {
            assert(this->mergedInputTag().index == 0);
            tagData = this->mergedInputTag().map;
        }

        {
            std::lock_guard lg(_listener_mutex); // TODO review/profile if a lock-free data structure should be used here
            const auto      historyView = _history ? _history->get_span(0) : std::span<const T>();
            std::erase_if(_listeners, [](const auto& l) { return l->expired; });
            for (auto& listener : _listeners) {
                listener->process(historyView, inData, tagData);
            }
            if (_history) {
                // store potential pre-samples for triggers at the beginning of the next chunk
                const auto toWrite = std::min(inData.size(), _history->capacity());
                _history->push_back_bulk(inData.last(toWrite).begin(), inData.last(toWrite).end());
            }
        }
        return work::Status::OK;
    }

private:
    void ensureHistorySize(std::size_t new_size) {
        const auto old_size = _history ? _history->capacity() : std::size_t{0};
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

    void addListener(std::unique_ptr<AbstractListener>&& l, bool block) {
        l->setMetadata(detail::Metadata{sample_rate, signal_name, signal_unit, signal_min, signal_max});
        if (block) {
            _listeners.push_back(std::move(l));
        } else {
            _listeners.push_front(std::move(l));
        }
    }

    struct AbstractListener {
        bool expired = false;

        virtual ~AbstractListener() = default;

        void setExpired() { expired = true; }

        virtual void setMetadata(detail::Metadata) = 0;

        virtual void process(std::span<const T> history, std::span<const T> data, std::optional<property_map> tagData0) = 0;
        virtual void stop()                                                                                             = 0;
    };

    template<typename Callback>
    struct ContinuousListener : public AbstractListener {
        static constexpr auto hasCallback       = !std::is_same_v<Callback, gr::meta::null_type>;
        static constexpr auto callbackTakesTags = std::is_invocable_v<Callback, std::span<const T>, std::span<const Tag>> || std::is_invocable_v<Callback, std::span<const T>, std::span<const Tag>, const DataSink<T>&>;

        const DataSink<T>&              parent_sink;
        bool                            block           = false;
        std::size_t                     samples_written = 0;
        std::optional<detail::Metadata> _pendingMetadata;

        // callback-only
        std::size_t      buffer_fill = 0;
        std::vector<T>   buffer;
        std::vector<Tag> tag_buffer;

        // polling-only
        std::weak_ptr<Poller> polling_handler = {};
        Callback              callback;

        template<typename CallbackFW>
        explicit ContinuousListener(std::size_t maxChunkSize, CallbackFW&& c, const DataSink<T>& parent) : parent_sink(parent), buffer(maxChunkSize), callback{std::forward<CallbackFW>(c)} {}

        explicit ContinuousListener(std::shared_ptr<Poller> poller, bool doBlock, const DataSink<T>& parent) : parent_sink(parent), block(doBlock), polling_handler{std::move(poller)} {}

        inline void callCallback(std::span<const T> data, std::span<const Tag> tags) {
            if constexpr (std::is_invocable_v<Callback, std::span<const T>, std::span<const Tag>, const DataSink<T>&>) {
                callback(std::move(data), tags, parent_sink);
            } else if constexpr (std::is_invocable_v<Callback, std::span<const T>, std::span<const Tag>>) {
                callback(std::move(data), tags);
            } else {
                callback(std::move(data));
            }
        }

        void setMetadata(detail::Metadata metadata) override { _pendingMetadata = std::move(metadata); }

        void process(std::span<const T>, std::span<const T> data, std::optional<property_map> tagData0) override {
            if constexpr (hasCallback) {
                // if there's pending data, fill buffer and send out
                if (buffer_fill > 0) {
                    const auto n = std::min(data.size(), buffer.size() - buffer_fill);
                    std::ranges::copy(data.first(n), buffer.begin() + static_cast<std::ptrdiff_t>(buffer_fill));
                    if constexpr (callbackTakesTags) {
                        if (auto tag = detail::tagAndMetadata(tagData0, _pendingMetadata)) {
                            tag_buffer.emplace_back(static_cast<Tag::signed_index_type>(buffer_fill), std::move(*tag));
                        }
                        _pendingMetadata.reset();
                        tagData0.reset();
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
                        if (auto tag = detail::tagAndMetadata(tagData0, _pendingMetadata)) {
                            tags.emplace_back(0, std::move(*tag));
                        }
                        tagData0.reset();
                        _pendingMetadata.reset();
                        callCallback(data.first(buffer.size()), std::span(tags));
                    } else {
                        callback(data.first(buffer.size()));
                    }
                    samples_written += buffer.size();
                    data = data.last(data.size() - buffer.size());
                }

                // write remaining data to the buffer
                if (!data.empty()) {
                    std::ranges::copy(data, buffer.begin());
                    buffer_fill = data.size();
                    if constexpr (callbackTakesTags) {
                        if (auto tag = detail::tagAndMetadata(tagData0, _pendingMetadata)) {
                            tag_buffer.emplace_back(0, std::move(*tag));
                        }
                        _pendingMetadata.reset();
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
                    if (auto tag = detail::tagAndMetadata(tagData0, _pendingMetadata)) {
                        auto tw = poller->tag_writer.reserve(1);
                        tw[0]   = {static_cast<Tag::signed_index_type>(samples_written), std::move(*tag)};
                        tw.publish(1);
                    }
                    _pendingMetadata.reset();
                    auto writeData = poller->writer.reserve(toWrite);
                    std::ranges::copy(data | std::views::take(toWrite), writeData.begin());
                    writeData.publish(writeData.size());
                }
                poller->drop_count += data.size() - toWrite;
                samples_written += toWrite;
            }
        }

        void stop() override {
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

    template<typename Callback, trigger::Matcher TMatcher>
    struct TriggerListener : public AbstractListener {
        bool        block       = false;
        std::size_t preSamples  = 0;
        std::size_t postSamples = 0;

        DataSet<T>                   dataset_template;
        TMatcher                     trigger_matcher = {};
        std::deque<PendingWindow>    pending_trigger_windows; // triggers that still didn't receive all their data
        std::weak_ptr<DataSetPoller> polling_handler = {};

        Callback callback;

        template<trigger::Matcher Matcher>
        explicit TriggerListener(Matcher&& matcher, std::shared_ptr<DataSetPoller> handler, std::size_t pre, std::size_t post, bool doBlock) : block(doBlock), preSamples(pre), postSamples(post), trigger_matcher(std::forward<Matcher>(matcher)), polling_handler{std::move(handler)} {}

        template<typename CallbackFW, trigger::Matcher Matcher>
        explicit TriggerListener(Matcher&& matcher, std::size_t pre, std::size_t post, CallbackFW&& cb) : preSamples(pre), postSamples(post), trigger_matcher(std::forward<Matcher>(matcher)), callback{std::forward<CallbackFW>(cb)} {}

        void setMetadata(detail::Metadata metadata) override { dataset_template = detail::makeDataSetTemplate<T>(std::move(metadata)); }

        inline bool publishDataSet(DataSet<T>&& data) {
            if constexpr (!std::is_same_v<Callback, gr::meta::null_type>) {
                callback(std::move(data));
                return true;
            } else {
                auto poller = polling_handler.lock();
                if (!poller) {
                    this->setExpired();
                    return false;
                }

                auto writeData = poller->writer.tryReserve(1);
                if (writeData.empty()) {
                    return false;
                } else {
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
                return true;
            }
        }

        void process(std::span<const T> history, std::span<const T> inData, std::optional<property_map> tagData0) override {
            if (tagData0 && trigger_matcher("", Tag{0, *tagData0}, {}) == trigger::MatchResult::Matching) {
                DataSet<T> dataset = dataset_template;
                dataset.signal_values.reserve(preSamples + postSamples); // TODO maybe make the circ. buffer smaller but preallocate these

                const auto preSampleView = history.subspan(0UZ, std::min(preSamples, history.size()));
                dataset.signal_values.insert(dataset.signal_values.end(), preSampleView.rbegin(), preSampleView.rend());

                dataset.timing_events = {{{static_cast<Tag::signed_index_type>(preSampleView.size()), *tagData0}}};
                pending_trigger_windows.push_back({.dataset = std::move(dataset), .pending_post_samples = postSamples});
            }

            auto window = pending_trigger_windows.begin();
            while (window != pending_trigger_windows.end()) {
                const auto postSampleView = inData.first(std::min(window->pending_post_samples, inData.size()));
                window->dataset.signal_values.insert(window->dataset.signal_values.end(), postSampleView.begin(), postSampleView.end());
                window->pending_post_samples -= postSampleView.size();

                if (window->pending_post_samples == 0) {
                    bool published = this->publishDataSet(std::move(window->dataset));
                    if (published) {
                        window = pending_trigger_windows.erase(window);
                    }
                } else {
                    ++window;
                }
            }
        }

        void stop() override {
            for (auto& window : pending_trigger_windows) {
                if (!window.dataset.signal_values.empty()) {
                    bool published = this->publishDataSet(std::move(window.dataset));
                    if (!published) {
                        throw gr::exception("DataSink::TriggerListener::stop() can not publish pending datasets\n");
                    }
                }
            }
            pending_trigger_windows.clear();
            if (auto p = polling_handler.lock()) {
                p->finished = true;
            }
        }
    };

    template<typename Callback, trigger::Matcher M>
    struct MultiplexedListener : public AbstractListener {
        bool                         block = false;
        M                            matcher;
        DataSet<T>                   dataset_template;
        std::optional<DataSet<T>>    pending_dataset;
        std::size_t                  maximumWindowSize;
        std::weak_ptr<DataSetPoller> polling_handler = {};
        Callback                     callback;

        template<typename CallbackFW, trigger::Matcher Matcher>
        explicit MultiplexedListener(Matcher&& matcher_, std::size_t maxWindowSize, CallbackFW&& cb) : matcher(std::forward<Matcher>(matcher_)), maximumWindowSize(maxWindowSize), callback(std::forward<CallbackFW>(cb)) {}

        template<trigger::Matcher Matcher>
        explicit MultiplexedListener(Matcher&& matcher_, std::size_t maxWindowSize, std::shared_ptr<DataSetPoller> handler, bool doBlock) : block(doBlock), matcher(std::forward<Matcher>(matcher_)), maximumWindowSize(maxWindowSize), polling_handler{std::move(handler)} {}

        void setMetadata(detail::Metadata metadata) override { dataset_template = detail::makeDataSetTemplate<T>(std::move(metadata)); }

        inline bool publishDataSet(DataSet<T>&& data) {
            if constexpr (!std::is_same_v<Callback, gr::meta::null_type>) {
                callback(std::move(data));
                return true;
            } else {
                auto poller = polling_handler.lock();
                if (!poller) {
                    this->setExpired();
                    return false;
                }

                auto writeData = poller->writer.tryReserve(1);
                if (writeData.empty()) {
                    return false;
                } else {
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
                return true;
            }
        }

        void process(std::span<const T>, std::span<const T> inData, std::optional<property_map> tagData0) override {
            if (tagData0) {
                const auto obsr = matcher("", Tag{0, *tagData0}, {});
                if (obsr == trigger::MatchResult::NotMatching || obsr == trigger::MatchResult::Matching) {
                    if (pending_dataset) {
                        if (obsr == trigger::MatchResult::NotMatching) {
                            pending_dataset->timing_events[0].push_back({static_cast<Tag::signed_index_type>(pending_dataset->signal_values.size()), *tagData0});
                        }
                        bool published = this->publishDataSet(std::move(*pending_dataset));
                        if (published) {
                            pending_dataset.reset();
                        }
                    }
                }
                if (obsr == trigger::MatchResult::Matching) {
                    pending_dataset = dataset_template;
                    pending_dataset->signal_values.reserve(maximumWindowSize); // TODO might be too much?
                    pending_dataset->timing_events = {{{0, *tagData0}}};
                }
            }
            if (pending_dataset) {
                const auto toWrite = std::min(inData.size(), maximumWindowSize - pending_dataset->signal_values.size());
                const auto view    = inData.first(toWrite);
                pending_dataset->signal_values.insert(pending_dataset->signal_values.end(), view.begin(), view.end());

                if (pending_dataset->signal_values.size() == maximumWindowSize) {
                    bool published = this->publishDataSet(std::move(*pending_dataset));
                    if (published) {
                        pending_dataset.reset();
                    }
                }
            }
        }

        void stop() override {
            if (pending_dataset) {
                bool published = this->publishDataSet(std::move(*pending_dataset));
                if (!published) {
                    throw gr::exception("DataSink::MultiplexedListener::stop() can not publish pending datasets\n");
                }
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

    template<typename Callback, trigger::Matcher M>
    struct SnapshotListener : public AbstractListener {
        bool                         block = false;
        std::chrono::nanoseconds     time_delay;
        std::size_t                  sample_delay = 0;
        DataSet<T>                   dataset_template;
        M                            trigger_matcher = {};
        std::deque<PendingSnapshot>  pending;
        std::weak_ptr<DataSetPoller> polling_handler = {};
        Callback                     callback;

        template<trigger::Matcher Matcher>
        explicit SnapshotListener(Matcher&& matcher, std::chrono::nanoseconds delay, std::shared_ptr<DataSetPoller> poller, bool doBlock) : block(doBlock), time_delay(delay), trigger_matcher(std::forward<Matcher>(matcher)), polling_handler{std::move(poller)} {}

        template<typename CallbackFW, trigger::Matcher Matcher>
        explicit SnapshotListener(Matcher&& matcher, std::chrono::nanoseconds delay, CallbackFW&& cb) : time_delay(delay), trigger_matcher(std::forward<Matcher>(matcher)), callback(std::forward<CallbackFW>(cb)) {}

        void setMetadata(detail::Metadata metadata) override {
            sample_delay     = static_cast<std::size_t>(std::round(std::chrono::duration_cast<std::chrono::duration<float>>(time_delay).count() * metadata.sampleRate));
            dataset_template = detail::makeDataSetTemplate<T>(std::move(metadata));
        }

        inline bool publishDataSet(DataSet<T>&& data) {
            if constexpr (!std::is_same_v<Callback, gr::meta::null_type>) {
                callback(std::move(data));
                return true;
            } else {
                auto poller = polling_handler.lock();
                if (!poller) {
                    this->setExpired();
                    return false;
                }

                auto writeData = poller->writer.tryReserve(1);
                if (writeData.empty()) {
                    return false;
                } else {
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
                return true;
            }
        }

        void process(std::span<const T>, std::span<const T> inData, std::optional<property_map> tagData0) override {
            if (tagData0 && trigger_matcher("", {0, *tagData0}, {}) == trigger::MatchResult::Matching) {
                auto new_pending = PendingSnapshot{*tagData0, sample_delay, sample_delay};
                // make sure pending is sorted by number of pending_samples (insertion might be not at end if sample rate decreased)
                auto rit = std::find_if(pending.rbegin(), pending.rend(), [delay = sample_delay](const auto& other) { return other.pending_samples < delay; });
                pending.insert(rit.base(), std::move(new_pending));
            }

            auto it = pending.begin();
            while (it != pending.end()) {
                if (it->pending_samples >= inData.size()) {
                    it->pending_samples -= inData.size();
                    break;
                }

                DataSet<T> dataset    = dataset_template;
                dataset.timing_events = {{{-static_cast<Tag::signed_index_type>(it->delay), std::move(it->tag_data)}}};
                dataset.signal_values = {inData[it->pending_samples]};
                bool published        = this->publishDataSet(std::move(dataset));
                if (published) {
                    it = pending.erase(it);
                }
            }
        }

        void stop() override {
            pending.clear();
            if (auto p = polling_handler.lock()) {
                p->finished = true;
            }
        }
    };
};

} // namespace gr::basic

ENABLE_REFLECTION_FOR_TEMPLATE(gr::basic::DataSink, in, sample_rate, signal_name, signal_unit, signal_min, signal_max);
auto registerDataSink = gr::registerBlock<gr::basic::DataSink, float, double>(gr::globalBlockRegistry());

#endif
