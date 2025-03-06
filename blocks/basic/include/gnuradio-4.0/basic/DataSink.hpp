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

enum class OverflowPolicy : std::uint8_t {
    Backpressure = 0U, /// creates backpressure on the upstream flow-graph, guaranteeing reliable delivery at the expense of potential stalling.
    Drop               /// drop arriving samples to avoid blocking upstream flow-graph, potentially losing some data (e.g. UI use-case).
};

struct PollerConfig {
    OverflowPolicy overflowPolicy     = OverflowPolicy::Backpressure;
    std::size_t    minRequiredSamples = 1UZ;                                     // Minimum number of samples required before `process` call. Higher values optimize throughput by reducing frequent small `process` calls.
    std::size_t    maxRequiredSamples = std::numeric_limits<std::size_t>::max(); // Maximum number of samples that can be processed in a single `process` call. Lower values optimize latency by allowing faster processing of small batches.

    std::size_t preSamples  = 100; // Only used in trigger mode. Number of samples to keep before a trigger.
    std::size_t postSamples = 100; // Only used in trigger mode. Number of samples to keep after a trigger.

    std::size_t maximumWindowSize = 1000; // Only used in multiplexed mode.

    std::chrono::nanoseconds delay = std::chrono::milliseconds{10}; // Only used in snapshot mode. Defines the time interval for snapshot.
};

template<typename T>
class DataSink;

template<typename T>
class DataSetSink;

template<typename T, typename V>
concept DataSetCallback = std::invocable<T, DataSet<V>>;

/**
 * Stream callback functions receive the span of data, with optional tags and reference to the sink.
 */
template<typename T, typename V>
concept StreamCallback = std::invocable<T, std::span<const V>> || std::invocable<T, std::span<const V>, std::span<const Tag>> || std::invocable<T, std::span<const V>, std::span<const Tag>, const DataSink<V>&>;

template<typename T, typename V>
concept DataSetMatcher = std::invocable<T, gr::DataSet<V>> && std::is_same_v<std::invoke_result_t<T, gr::DataSet<V>>, bool>;

template<typename T>
concept DataSinkOrDataSetSinkLike = gr::meta::is_instantiation_of<T, DataSink> || gr::meta::is_instantiation_of<T, DataSetSink>;

static_assert(DataSinkOrDataSetSinkLike<DataSink<float>>);
static_assert(DataSinkOrDataSetSinkLike<DataSetSink<float>>);

namespace detail {
constexpr std::size_t data_sink_buffer_size          = 65536;
constexpr std::size_t data_sink_tag_buffer_size      = 1024;
constexpr std::size_t data_sink_data_set_buffer_size = 1024;

inline std::size_t calculateNSamplesToProcess(std::size_t available, std::size_t requested, std::size_t minRequired, std::size_t maxRequired) {
    const std::size_t clampRequested = std::clamp(requested, minRequired, maxRequired);
    return std::min(available, clampRequested);
}

} // namespace detail

template<typename T>
struct StreamingPoller {
    // TODO consider whether reusing port<T> here makes sense
    gr::CircularBuffer<T>            buffer      = gr::CircularBuffer<T>(detail::data_sink_buffer_size);
    decltype(buffer.new_reader())    reader      = buffer.new_reader();
    decltype(buffer.new_writer())    writer      = buffer.new_writer();
    gr::CircularBuffer<Tag>          tagBuffer   = gr::CircularBuffer<Tag>(detail::data_sink_tag_buffer_size);
    decltype(tagBuffer.new_reader()) tagReader   = tagBuffer.new_reader();
    decltype(tagBuffer.new_writer()) tagWriter   = tagBuffer.new_writer();
    std::size_t                      samplesRead = 0; // reader thread
    std::atomic<bool>                finished    = false;
    std::atomic<std::size_t>         dropCount   = 0;
    std::size_t                      minRequiredSamples; // the number of samples to process must be in a range [minRequiredSamples, maxRequiredSamples]
    std::size_t                      maxRequiredSamples;

    StreamingPoller(std::size_t minRequiredSamples_, std::size_t maxRequiredSamples_) : minRequiredSamples(minRequiredSamples_), maxRequiredSamples(maxRequiredSamples_) {
        if (minRequiredSamples > maxRequiredSamples) {
            throw gr::exception(fmt::format("Failed to create StreamingPoller: minRequiredSamples ({}) > maxRequiredSamples ({})", minRequiredSamples, maxRequiredSamples));
        }
    }

    template<typename Handler>
    [[nodiscard]] bool process(Handler fnc, std::size_t requested = std::numeric_limits<std::size_t>::max()) {
        const std::size_t nProcess = detail::calculateNSamplesToProcess(reader.available(), requested, minRequiredSamples, maxRequiredSamples);
        if (nProcess < minRequiredSamples) {
            return false;
        }

        ReaderSpanLike auto readData = reader.get(nProcess);
        if constexpr (requires { fnc(std::span<const T>(), std::span<const Tag>()); }) {
            ReaderSpanLike auto tags             = tagReader.get();
            const auto          it               = std::ranges::find_if_not(tags, [until = samplesRead + nProcess](const auto& tag) { return tag.index < until; });
            auto                relevantTagsView = std::span(tags.begin(), it) | std::views::transform([this](const auto& v) { return Tag{v.index - samplesRead, v.map}; });
            auto                relevantTags     = std::vector(relevantTagsView.begin(), relevantTagsView.end());

            fnc(readData, relevantTags);
            std::ignore = tags.consume(relevantTags.size());
        } else {
            ReaderSpanLike auto tags = tagReader.get();
            std::ignore              = tags.consume(tags.size());
            fnc(readData);
        }

        std::ignore = readData.consume(nProcess);
        samplesRead += nProcess;
        return true;
    }
};

template<typename T>
struct DataSetPoller {
    gr::CircularBuffer<DataSet<T>> buffer    = gr::CircularBuffer<DataSet<T>>(detail::data_sink_data_set_buffer_size);
    decltype(buffer.new_reader())  reader    = buffer.new_reader();
    decltype(buffer.new_writer())  writer    = buffer.new_writer();
    std::atomic<bool>              finished  = false;
    std::atomic<std::size_t>       dropCount = 0;
    std::size_t                    minRequiredSamples; // the number of samples (DataSets) to process must be in a range [minRequiredSamples, maxRequiredSamples]
    std::size_t                    maxRequiredSamples;

    DataSetPoller(std::size_t minRequiredSamples_, std::size_t maxRequiredSamples_) : minRequiredSamples(minRequiredSamples_), maxRequiredSamples(maxRequiredSamples_) {
        if (minRequiredSamples > maxRequiredSamples) {
            throw gr::exception(fmt::format("Failed to create DataSetPoller: minRequiredSamples ({}) > maxRequiredSamples ({})", minRequiredSamples, maxRequiredSamples));
        }
    }

    [[nodiscard]] bool process(std::invocable<std::span<DataSet<T>>> auto fnc, std::size_t requested = std::numeric_limits<std::size_t>::max()) {
        const std::size_t nProcess = detail::calculateNSamplesToProcess(reader.available(), requested, minRequiredSamples, maxRequiredSamples);
        if (nProcess < minRequiredSamples) {
            return false;
        }
        if (nProcess < minRequiredSamples) {
            return false;
        }

        ReaderSpanLike auto readData = reader.get(nProcess);
        fnc(readData);
        std::ignore = readData.consume(nProcess);
        return true;
    }
};

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

    template<DataSinkOrDataSetSinkLike TSink>
    void registerSink(TSink* sink, std::source_location location = std::source_location::current()) {
        std::lock_guard lg{_mutex};

        if (sink->_registered || sink->signal_name.value.empty()) {
            return;
        }

        if (_sink_by_signal_name.contains(sink->signal_name)) {
            throw gr::exception(fmt::format("Failed to register sink `{}`. Sink with the signal_name `{}` is already registered.", sink->name, sink->signal_name), location);
        }

        _sinks.push_back(sink);
        _sink_by_signal_name[sink->signal_name] = sink;
        sink->_registered                       = true;
    }

    template<DataSinkOrDataSetSinkLike TSink>
    void unregisterSink(TSink* sink) {
        std::lock_guard lg{_mutex};
        std::erase_if(_sinks, [sink](const std::any& v) -> bool {
            auto ptr = std::any_cast<TSink*>(v);
            return ptr && ptr == sink;
        });
        _sink_by_signal_name.erase(sink->signal_name);
        sink->_registered = false;
    }

    template<DataSinkOrDataSetSinkLike TSink>
    void updateSignalName(TSink* sink, std::string_view oldName, std::string_view newName, std::source_location location = std::source_location::current()) {
        std::lock_guard lg{_mutex};

        if (oldName.empty()) {
            throw gr::exception(fmt::format("Failed to update signal_name of sink `{}`. The old signal_name is an empty string.", sink->name), location);
        }

        if (newName.empty()) {
            throw gr::exception(fmt::format("Failed to update signal_name of sink `{}`. The new signal_name is an empty string.", sink->name), location);
        }

        if (_sink_by_signal_name.contains(std::string{newName})) {
            throw gr::exception(fmt::format("Failed to update signal_name of sink `{}`. Sink with signal_name `{}` is already registered.", sink->name, newName), location);
        }

        _sink_by_signal_name.erase(std::string(oldName));
        _sink_by_signal_name[std::string(newName)] = sink;
    }

    template<typename T>
    std::shared_ptr<StreamingPoller<T>> getStreamingPoller(const DataSinkQuery& query, PollerConfig config = {}) {
        std::lock_guard lg{_mutex};
        auto            sink = find<DataSink<T>>(query);
        return sink ? sink->getStreamingPoller(config) : nullptr;
    }

    template<typename T, trigger::Matcher M>
    std::shared_ptr<DataSetPoller<T>> getTriggerPoller(const DataSinkQuery& query, M&& matcher, PollerConfig config = {}) {
        std::lock_guard lg{_mutex};
        auto            sink = find<DataSink<T>>(query);
        return sink ? sink->getTriggerPoller(std::forward<M>(matcher), config) : nullptr;
    }

    template<typename T, trigger::Matcher M>
    std::shared_ptr<DataSetPoller<T>> getMultiplexedPoller(const DataSinkQuery& query, M&& matcher, PollerConfig config = {}) {
        std::lock_guard lg{_mutex};
        auto            sink = find<DataSink<T>>(query);
        return sink ? sink->getMultiplexedPoller(std::forward<M>(matcher), config) : nullptr;
    }

    template<typename T, trigger::Matcher M>
    std::shared_ptr<DataSetPoller<T>> getSnapshotPoller(const DataSinkQuery& query, M&& matcher, PollerConfig config = {}) {
        std::lock_guard lg{_mutex};
        auto            sink = find<DataSink<T>>(query);
        return sink ? sink->getSnapshotPoller(std::forward<M>(matcher), config) : nullptr;
    }

    template<typename T, DataSetMatcher<T> M>
    std::shared_ptr<DataSetPoller<T>> getDataSetPoller(const DataSinkQuery& query, M&& matcher, PollerConfig config = {}) {
        std::lock_guard lg{_mutex};
        auto            sink = find<DataSetSink<T>>(query);
        return sink ? sink->getPoller(std::forward<M>(matcher), config) : nullptr;
    }

    template<typename T>
    std::shared_ptr<DataSetPoller<T>> getDataSetPoller(const DataSinkQuery& query, PollerConfig config = {}) {
        std::lock_guard lg{_mutex};
        auto            sink = find<DataSetSink<T>>(query);
        return sink ? sink->getPoller(config) : nullptr;
    }

    template<typename T, StreamCallback<T> Callback>
    bool registerStreamingCallback(const DataSinkQuery& query, std::size_t maxChunkSize, Callback&& callback) {
        std::lock_guard lg{_mutex};
        auto            sink = find<DataSink<T>>(query);
        if (!sink) {
            return false;
        }

        sink->registerStreamingCallback(maxChunkSize, std::forward<Callback>(callback));
        return true;
    }

    template<typename T, DataSetCallback<T> Callback, trigger::Matcher M>
    bool registerTriggerCallback(const DataSinkQuery& query, M&& matcher, std::size_t preSamples, std::size_t postSamples, Callback&& callback) {
        std::lock_guard lg{_mutex};
        auto            sink = find<DataSink<T>>(query);
        if (!sink) {
            return false;
        }

        sink->registerTriggerCallback(std::forward<M>(matcher), preSamples, postSamples, std::forward<Callback>(callback));
        return true;
    }

    template<typename T, DataSetCallback<T> Callback, trigger::Matcher M>
    bool registerMultiplexedCallback(const DataSinkQuery& query, M&& matcher, std::size_t maximumWindowSize, Callback&& callback) {
        std::lock_guard lg{_mutex};
        auto            sink = find<DataSink<T>>(query);
        if (!sink) {
            return false;
        }

        sink->registerMultiplexedCallback(std::forward<M>(matcher), maximumWindowSize, std::forward<Callback>(callback));
        return true;
    }

    template<typename T, DataSetCallback<T> Callback, trigger::Matcher M>
    bool registerSnapshotCallback(const DataSinkQuery& query, M&& matcher, std::chrono::nanoseconds delay, Callback&& callback) {
        std::lock_guard lg{_mutex};
        auto            sink = find<DataSink<T>>(query);
        if (!sink) {
            return false;
        }

        sink->registerSnapshotCallback(std::forward<M>(matcher), delay, std::forward<Callback>(callback));
        return true;
    }

    template<typename T, DataSetCallback<T> Callback, DataSetMatcher<T> M>
    bool registerDataSetCallback(const DataSinkQuery& query, M&& matcher, Callback&& callback) {
        std::lock_guard lg{_mutex};
        auto            sink = find<DataSetSink<T>>(query);
        if (!sink) {
            return false;
        }

        sink->registerCallback(std::forward<M>(matcher), std::forward<Callback>(callback));
        return true;
    }

    template<typename T, DataSetCallback<T> Callback>
    bool registerDataSetCallback(const DataSinkQuery& query, Callback&& callback) {
        std::lock_guard lg{_mutex};
        auto            sink = find<DataSetSink<T>>(query);
        if (!sink) {
            return false;
        }

        sink->registerCallback(std::forward<Callback>(callback));
        return true;
    }

private:
    template<typename T>
    T* find(const DataSinkQuery& query) {
        if (query._signal_name) {
            const auto it = _sink_by_signal_name.find(*query._signal_name);
            if (it != _sink_by_signal_name.end()) {
                try {
                    return std::any_cast<T*>(it->second);
                } catch (...) {
                    return nullptr;
                }
            }
            return nullptr;
        }
        auto sinkNameMatches = [&query](const std::any& v) {
            try {
                const auto sink = std::any_cast<T*>(v);
                return query._sink_name == sink->name;
            } catch (...) {
                return false;
            }
        };

        const auto it = std::ranges::find_if(_sinks, sinkNameMatches);
        if (it == _sinks.end()) {
            return nullptr;
        }

        return std::any_cast<T*>(*it);
    }
};

namespace detail {

struct Metadata {
    float       sampleRate;
    std::string signalName;
    std::string signalQuantity;
    std::string signalUnit;
    float       signalMin;
    float       signalMax;

    property_map toTagMap() const { return {{std::string(tag::SIGNAL_RATE.shortKey()), sampleRate}, {std::string(tag::SIGNAL_NAME.shortKey()), signalName}, {std::string(tag::SIGNAL_QUANTITY.shortKey()), signalQuantity}, {std::string(tag::SIGNAL_UNIT.shortKey()), signalUnit}, {std::string(tag::SIGNAL_MIN.shortKey()), signalMin}, {std::string(tag::SIGNAL_MAX.shortKey()), signalMax}}; }
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
 * @brief generic data sink for exporting streams to non-GR C++ APIs.
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
 * This block type is mean for non-data set input streams. For input streams of type DataSet<T>, use
 * @see DataSetSink<T>.
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

This block type is mean for non-data set input streams. For input streams of type DataSet<T>, use
@see DataSetSink<T>.

@tparam T input sample type
)"">;
    struct AbstractListener;

    std::deque<std::unique_ptr<AbstractListener>> _listeners;
    bool                                          _listeners_finished = false;
    std::mutex                                    _listener_mutex;
    std::optional<gr::HistoryBuffer<T>>           _history;

public:
    PortIn<T, RequiredSamples<std::dynamic_extent, detail::data_sink_buffer_size>> in;

    Annotated<float, "sample rate", Doc<"signal sample rate">, Unit<"Hz">>                                           sample_rate = 1.f;
    Annotated<std::string, "signal name", Visible>                                                                   signal_name = ""; // DataSink cannot be registered with an empty string; it must be set either by the user or via a tag.
    Annotated<std::string, "signal quantity", Doc<"physical quantity (e.g., 'voltage'). Follows ISO 80000-1:2022.">> signal_quantity{};
    Annotated<std::string, "signal unit", Doc<"unit of measurement (e.g., '[V]', '[m]'). Follows ISO 80000-1:2022">> signal_unit{"a.u."};
    Annotated<float, "signal min", Doc<"signal physical min. (e.g. DAQ) limit">>                                     signal_min = -1.0f;
    Annotated<float, "signal max", Doc<"signal physical max. (e.g. DAQ) limit">>                                     signal_max = +1.0f;

    GR_MAKE_REFLECTABLE(DataSink, in, sample_rate, signal_name, signal_unit, signal_min, signal_max);

    bool _registered = false; // status should be updated by DataSinkRegistry

    using Block<DataSink<T>>::Block; // needed to inherit mandatory base-class Block(property_map) constructor

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/) {
        if (oldSettings.contains("signal_name")) {
            const std::string oldSignalName = std::get<std::string>(oldSettings.at("signal_name"));
            if (oldSignalName.empty()) {
                DataSinkRegistry::instance().registerSink(this);
            } else if (oldSignalName != signal_name && _registered) {
                DataSinkRegistry::instance().updateSignalName(this, oldSignalName, signal_name);
            }
        }
        std::lock_guard lg{_listener_mutex};
        for (auto& listener : _listeners) {
            listener->setMetadata(detail::Metadata{sample_rate, signal_name, signal_quantity, signal_unit, signal_min, signal_max});
        }
    }

    std::shared_ptr<StreamingPoller<T>> getStreamingPoller(PollerConfig config) {
        const auto      withBackpressure = config.overflowPolicy == OverflowPolicy::Backpressure;
        auto            handler          = std::make_shared<StreamingPoller<T>>(config.minRequiredSamples, config.maxRequiredSamples);
        std::lock_guard lg(_listener_mutex);
        handler->finished = _listeners_finished;
        addListener(std::make_unique<ContinuousListener<gr::meta::null_type>>(handler, withBackpressure, *this), withBackpressure);
        return handler;
    }

    template<trigger::Matcher M>
    std::shared_ptr<DataSetPoller<T>> getTriggerPoller(M&& matcher, PollerConfig config = {}) {
        const auto      withBackpressure = config.overflowPolicy == OverflowPolicy::Backpressure;
        auto            handler          = std::make_shared<DataSetPoller<T>>(config.minRequiredSamples, config.maxRequiredSamples);
        std::lock_guard lg(_listener_mutex);
        handler->finished = _listeners_finished;
        addListener(std::make_unique<TriggerListener<gr::meta::null_type, M>>(std::forward<M>(matcher), handler, config.preSamples, config.postSamples, withBackpressure), withBackpressure);
        ensureHistorySize(config.preSamples);
        return handler;
    }

    template<trigger::Matcher M>
    std::shared_ptr<DataSetPoller<T>> getMultiplexedPoller(M&& matcher, PollerConfig config = {}) {
        std::lock_guard lg(_listener_mutex);
        const auto      withBackpressure = config.overflowPolicy == OverflowPolicy::Backpressure;
        auto            handler          = std::make_shared<DataSetPoller<T>>(config.minRequiredSamples, config.maxRequiredSamples);
        addListener(std::make_unique<MultiplexedListener<gr::meta::null_type, M>>(std::forward<M>(matcher), config.maximumWindowSize, handler, withBackpressure), withBackpressure);
        return handler;
    }

    template<trigger::Matcher M>
    std::shared_ptr<DataSetPoller<T>> getSnapshotPoller(M&& matcher, PollerConfig config = {}) {
        const auto      withBackpressure = config.overflowPolicy == OverflowPolicy::Backpressure;
        auto            handler          = std::make_shared<DataSetPoller<T>>(config.minRequiredSamples, config.maxRequiredSamples);
        std::lock_guard lg(_listener_mutex);
        addListener(std::make_unique<SnapshotListener<gr::meta::null_type, M>>(std::forward<M>(matcher), config.delay, handler, withBackpressure), withBackpressure);
        return handler;
    }

    template<StreamCallback<T> Callback>
    void registerStreamingCallback(std::size_t maxChunkSize, Callback&& callback) {
        std::lock_guard lg(_listener_mutex);
        addListener(std::make_unique<ContinuousListener<Callback>>(maxChunkSize, std::forward<Callback>(callback), *this), false);
    }

    template<trigger::Matcher M, DataSetCallback<T> Callback>
    void registerTriggerCallback(M&& matcher, std::size_t preSamples, std::size_t postSamples, Callback&& callback) {
        std::lock_guard lg(_listener_mutex);
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

    void stop() noexcept {
        DataSinkRegistry::instance().unregisterSink(this);
        std::lock_guard lg(_listener_mutex);
        for (auto& listener : _listeners) {
            listener->stop();
        }
        _listeners_finished = true;
    }

    [[nodiscard]] work::Status processBulk(InputSpanLike auto& inData) noexcept {
        // Note: AbstractListener::process currently accepts a property_map for a single Tag only.
        // Consider updating the method signature to handle multiple Tags simultaneously.
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
                _history->push_front(inData);
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

        auto new_history = gr::HistoryBuffer<T>(new_size);
        if (_history) {
            new_history.push_front(_history->begin(), _history->end());
        }
        _history = std::move(new_history);
    }

    void addListener(std::unique_ptr<AbstractListener>&& l, bool withBackpressure) {
        l->setMetadata(detail::Metadata{sample_rate, signal_name, signal_quantity, signal_unit, signal_min, signal_max});
        if (withBackpressure) {
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
    struct DataSetBaseListener : public AbstractListener {
        bool                            isBlocking = false;
        std::weak_ptr<DataSetPoller<T>> poller;
        Callback                        callback;

        template<typename CallbackFW>
        explicit DataSetBaseListener(CallbackFW&& callback_) : callback(std::forward<CallbackFW>(callback_)) {}

        explicit DataSetBaseListener(std::shared_ptr<DataSetPoller<T>> poller_, bool isBlocking_) : isBlocking(isBlocking_), poller(std::move(poller_)) {}

        inline void publishDataSet(DataSet<T>&& data) {
            if constexpr (!std::is_same_v<Callback, gr::meta::null_type>) {
                callback(std::move(data));
            } else {
                auto pollerPtr = poller.lock();
                if (!pollerPtr) {
                    this->setExpired();
                    return;
                }

                if (isBlocking || pollerPtr->writer.available() > 0) {
                    auto writeData = pollerPtr->writer.reserve(1);
                    writeData[0]   = std::move(data);
                    writeData.publish(1);
                } else {
                    pollerPtr->dropCount++;
                }
            }
        }
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
        std::weak_ptr<StreamingPoller<T>> polling_handler = {};
        Callback                          callback;

        template<typename CallbackFW>
        explicit ContinuousListener(std::size_t maxChunkSize, CallbackFW&& c, const DataSink<T>& parent) : parent_sink(parent), buffer(maxChunkSize), callback{std::forward<CallbackFW>(c)} {}

        explicit ContinuousListener(std::shared_ptr<StreamingPoller<T>> poller_, bool doBlock, const DataSink<T>& parent) : parent_sink(parent), block(doBlock), polling_handler{std::move(poller_)} {}

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
                            tag_buffer.emplace_back(static_cast<std::ptrdiff_t>(buffer_fill), std::move(*tag));
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
                        auto tw = poller->tagWriter.reserve(1);
                        tw[0]   = {samples_written, std::move(*tag)};
                        tw.publish(1);
                    }
                    _pendingMetadata.reset();
                    auto writeData = poller->writer.reserve(toWrite);
                    std::ranges::copy(data | std::views::take(toWrite), writeData.begin());
                    writeData.publish(writeData.size());
                }
                poller->dropCount += data.size() - toWrite;
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
    struct TriggerListener : public DataSetBaseListener<Callback> {
        std::size_t preSamples  = 0;
        std::size_t postSamples = 0;

        DataSet<T>                dataset_template;
        gr::property_map          trigger_matcher_state;
        TMatcher                  trigger_matcher = {};
        std::deque<PendingWindow> pending_trigger_windows; // triggers that still didn't receive all their data

        template<trigger::Matcher Matcher>
        explicit TriggerListener(Matcher&& matcher, std::shared_ptr<DataSetPoller<T>> poller_, std::size_t pre, std::size_t post, bool doBlock) : DataSetBaseListener<Callback>(std::move(poller_), doBlock), preSamples(pre), postSamples(post), trigger_matcher(std::forward<Matcher>(matcher)) {}

        template<typename CallbackFW, trigger::Matcher Matcher>
        explicit TriggerListener(Matcher&& matcher, std::size_t pre, std::size_t post, CallbackFW&& cb) : DataSetBaseListener<Callback>(std::forward<CallbackFW>(cb)), preSamples(pre), postSamples(post), trigger_matcher(std::forward<Matcher>(matcher)) {}

        void setMetadata(detail::Metadata metadata) override { dataset_template = detail::makeDataSetTemplate<T>(std::move(metadata)); }

        void process(std::span<const T> history, std::span<const T> inData, std::optional<property_map> tagData0) override {
            if (tagData0 && trigger_matcher("", Tag{0, *tagData0}, trigger_matcher_state) == trigger::MatchResult::Matching) {
                DataSet<T> dataset = dataset_template;
                dataset.signal_values.reserve(preSamples + postSamples); // TODO maybe make the circ. buffer smaller but preallocate these

                const auto preSampleView = history.subspan(0UZ, std::min(preSamples, history.size()));
                dataset.signal_values.insert(dataset.signal_values.end(), preSampleView.rbegin(), preSampleView.rend());

                dataset.timing_events = {{{static_cast<std::ptrdiff_t>(preSampleView.size()), *tagData0}}};
                pending_trigger_windows.push_back({.dataset = std::move(dataset), .pending_post_samples = postSamples});
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

        void stop() override {
            for (auto& window : pending_trigger_windows) {
                if (!window.dataset.signal_values.empty()) {
                    this->publishDataSet(std::move(window.dataset));
                }
            }
            pending_trigger_windows.clear();
            if (auto p = this->poller.lock()) {
                p->finished = true;
            }
        }
    };

    template<typename Callback, trigger::Matcher M>
    struct MultiplexedListener : public DataSetBaseListener<Callback> {
        M                         matcher;
        gr::property_map          matcher_state;
        DataSet<T>                dataset_template;
        std::optional<DataSet<T>> pending_dataset;
        std::size_t               maximumWindowSize;

        template<typename CallbackFW, trigger::Matcher Matcher>
        explicit MultiplexedListener(Matcher&& matcher_, std::size_t maxWindowSize, CallbackFW&& cb) : DataSetBaseListener<Callback>(std::forward<CallbackFW>(cb)), matcher(std::forward<Matcher>(matcher_)), maximumWindowSize(maxWindowSize) {}

        template<trigger::Matcher Matcher>
        explicit MultiplexedListener(Matcher&& matcher_, std::size_t maxWindowSize, std::shared_ptr<DataSetPoller<T>> poller_, bool doBlock) : DataSetBaseListener<Callback>(std::move(poller_), doBlock), matcher(std::forward<Matcher>(matcher_)), maximumWindowSize(maxWindowSize) {}

        void setMetadata(detail::Metadata metadata) override { dataset_template = detail::makeDataSetTemplate<T>(std::move(metadata)); }

        void process(std::span<const T>, std::span<const T> inData, std::optional<property_map> tagData0) override {
            if (tagData0) {
                const auto obsr = matcher("", Tag{0, *tagData0}, matcher_state);
                if (obsr == trigger::MatchResult::NotMatching || obsr == trigger::MatchResult::Matching) {
                    if (pending_dataset) {
                        if (obsr == trigger::MatchResult::NotMatching) {
                            pending_dataset->timing_events[0].emplace_back(pending_dataset->signal_values.size(), *tagData0);
                        }
                        this->publishDataSet(std::move(*pending_dataset));
                        pending_dataset.reset();
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
                    this->publishDataSet(std::move(*pending_dataset));
                    pending_dataset.reset();
                }
            }
        }

        void stop() override {
            if (pending_dataset) {
                this->publishDataSet(std::move(*pending_dataset));
                pending_dataset.reset();
            }
            if (auto p = this->poller.lock()) {
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
    struct SnapshotListener : public DataSetBaseListener<Callback> {
        std::chrono::nanoseconds    time_delay;
        std::size_t                 sample_delay = 0;
        DataSet<T>                  dataset_template;
        M                           trigger_matcher = {};
        gr::property_map            trigger_matcher_state;
        std::deque<PendingSnapshot> pending;

        template<trigger::Matcher Matcher>
        explicit SnapshotListener(Matcher&& matcher, std::chrono::nanoseconds delay, std::shared_ptr<DataSetPoller<T>> poller_, bool doBlock) : DataSetBaseListener<Callback>(std::move(poller_), doBlock), time_delay(delay), trigger_matcher(std::forward<Matcher>(matcher)) {}

        template<typename CallbackFW, trigger::Matcher Matcher>
        explicit SnapshotListener(Matcher&& matcher, std::chrono::nanoseconds delay, CallbackFW&& cb) : DataSetBaseListener<Callback>(std::forward<CallbackFW>(cb)), time_delay(delay), trigger_matcher(std::forward<Matcher>(matcher)) {}

        void setMetadata(detail::Metadata metadata) override {
            sample_delay     = static_cast<std::size_t>(std::round(std::chrono::duration_cast<std::chrono::duration<float>>(time_delay).count() * metadata.sampleRate));
            dataset_template = detail::makeDataSetTemplate<T>(std::move(metadata));
        }

        void process(std::span<const T>, std::span<const T> inData, std::optional<property_map> tagData0) override {
            if (tagData0 && trigger_matcher("", {0, *tagData0}, trigger_matcher_state) == trigger::MatchResult::Matching) {
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
                dataset.timing_events = {{{-static_cast<std::ptrdiff_t>(it->delay), std::move(it->tag_data)}}};
                dataset.signal_values = {inData[it->pending_samples]};
                this->publishDataSet(std::move(dataset));
                it = pending.erase(it);
            }
        }

        void stop() override {
            pending.clear();
            if (auto p = this->poller.lock()) {
                p->finished = true;
            }
        }
    };
};

/**
 * @brief data sink for exporting data set streams to non-GR C++ APIs.
 *
 * Like DataSink, but for exporting DataSet objects via poller or registered callback.
 * It provides a similar but simpler API to DataSink, basically handing out the data sets
 * as is, allowing basic filtering via a predicate on data set objects.
 *
 * @tparam T sample type in the data set
 */
template<typename T>
class DataSetSink : public Block<DataSetSink<T>> {
    using Description = Doc<R""(@brief data sink for exporting data set streams to non-GR C++ APIs.

 Like DataSink, but for exporting DataSet objects via poller or registered callback.
 It provides a similar but simpler API to DataSink, basically handing out the data sets
 as is, allowing basic filtering via a predicate on data set objects.

 @tparam T sample type in the data set
)"">;
    struct AbstractListener;
    std::deque<std::unique_ptr<AbstractListener>> _listeners;
    bool                                          _listeners_finished = false;
    std::mutex                                    _listener_mutex;

public:
    PortIn<DataSet<T>>                              in;
    Annotated<std::string, "DataSet name", Visible> signal_name = ""; // DataSetSink cannot be registered with an empty string; it must be set either by the user or via a tag.

    GR_MAKE_REFLECTABLE(DataSetSink, in, signal_name);

    bool _registered = false; // status should be updated by DataSinkRegistry

    using Block<DataSetSink<T>>::Block; // needed to inherit mandatory base-class Block(property_map) constructor

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/) {
        if (oldSettings.contains("signal_name")) {
            const std::string oldSignalName = std::get<std::string>(oldSettings.at("signal_name"));
            if (oldSignalName.empty()) {
                DataSinkRegistry::instance().registerSink(this);
            } else if (oldSignalName != signal_name && _registered) {
                DataSinkRegistry::instance().updateSignalName(this, oldSignalName, signal_name);
            }
        }
    }

    template<DataSetMatcher<T> M>
    std::shared_ptr<DataSetPoller<T>> getPoller(M&& matcher, PollerConfig config = {}) {
        const auto      withBackpressure = config.overflowPolicy == OverflowPolicy::Backpressure;
        auto            handler          = std::make_shared<DataSetPoller<T>>(config.minRequiredSamples, config.maxRequiredSamples);
        std::lock_guard lg(_listener_mutex);
        handler->finished = _listeners_finished;
        addListener(std::make_unique<Listener<gr::meta::null_type, M>>(std::forward<M>(matcher), handler, withBackpressure), withBackpressure);
        return handler;
    }

    std::shared_ptr<DataSetPoller<T>> getPoller(PollerConfig config = {}) {
        return getPoller([](const auto&) { return true; }, config);
    }

    template<DataSetMatcher<T> M, DataSetCallback<T> Callback>
    void registerCallback(M&& matcher, Callback&& callback) {
        std::lock_guard lg(_listener_mutex);
        addListener(std::make_unique<Listener<Callback, M>>(std::forward<M>(matcher), std::forward<Callback>(callback)), false);
    }

    template<DataSetCallback<T> Callback>
    void registerCallback(Callback&& callback) {
        registerCallback([](const auto&) { return true; }, std::forward<Callback>(callback));
    }

    void stop() noexcept {
        DataSinkRegistry::instance().unregisterSink(this);
        std::lock_guard lg(_listener_mutex);
        for (auto& listener : _listeners) {
            listener->stop();
        }
        _listeners_finished = true;
    }

    [[nodiscard]] work::Status processBulk(std::span<const DataSet<T>>& inData) noexcept {
        std::lock_guard lg(_listener_mutex);
        for (auto& listener : _listeners) {
            listener->process(inData);
        }
        return work::Status::OK;
    }

private:
    void addListener(std::unique_ptr<AbstractListener>&& l, bool block) {
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

        virtual void process(std::span<const DataSet<T>> data) = 0;
        virtual void stop()                                    = 0;
    };

    template<typename Callback, DataSetMatcher<T> TMatcher>
    struct Listener : public AbstractListener {
        bool                            block   = false;
        TMatcher                        matcher = {};
        gr::property_map                trigger_state;
        std::weak_ptr<DataSetPoller<T>> polling_handler = {};

        Callback callback;

        template<DataSetMatcher<T> Matcher>
        explicit Listener(Matcher&& matcher_, std::shared_ptr<DataSetPoller<T>> handler, bool doBlock) : block(doBlock), matcher(std::forward<Matcher>(matcher_)), polling_handler{std::move(handler)} {}

        template<typename CallbackFW, DataSetMatcher<T> Matcher>
        explicit Listener(Matcher&& matcher_, CallbackFW&& cb) : matcher(std::forward<Matcher>(matcher_)), callback{std::forward<CallbackFW>(cb)} {}

        inline void publishDataSet(const DataSet<T>& data) {
            if constexpr (!std::is_same_v<Callback, gr::meta::null_type>) {
                callback(std::move(data));
            } else {
                auto poller = polling_handler.lock();
                if (!poller) {
                    this->setExpired();
                    return;
                }

                if (block || poller->writer.available() > 0) {
                    auto writeData = poller->writer.reserve(1);
                    writeData[0]   = std::move(data);
                    writeData.publish(1);
                } else {
                    poller->dropCount++;
                }
            }
        }

        void process(std::span<const DataSet<T>> inData) override {
            for (const auto& ds : inData) {
                if (matcher(ds)) {
                    publishDataSet(ds);
                }
            }
        }

        void stop() override {
            if (auto p = polling_handler.lock()) {
                p->finished = true;
            }
        }
    }; // Listener
};

} // namespace gr::basic

auto registerDataSink    = gr::registerBlock<gr::basic::DataSink, float, double>(gr::globalBlockRegistry());
auto registerDataSetSink = gr::registerBlock<gr::basic::DataSetSink, float, double>(gr::globalBlockRegistry());

#endif
