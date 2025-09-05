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
    gr::CircularBuffer<T>            buffer             = gr::CircularBuffer<T>(detail::data_sink_buffer_size);
    decltype(buffer.new_reader())    reader             = buffer.new_reader();
    decltype(buffer.new_writer())    writer             = buffer.new_writer();
    gr::CircularBuffer<Tag>          tagBuffer          = gr::CircularBuffer<Tag>(detail::data_sink_tag_buffer_size);
    decltype(tagBuffer.new_reader()) tagReader          = tagBuffer.new_reader();
    decltype(tagBuffer.new_writer()) tagWriter          = tagBuffer.new_writer();
    std::size_t                      samplesRead        = 0; // reader thread
    std::atomic<bool>                finished           = false;
    std::atomic<std::size_t>         droppedSampleCount = 0;
    std::atomic<std::size_t>         droppedTagCount    = 0;
    std::size_t                      minRequiredSamples; // the number of samples to process must be in a range [minRequiredSamples, maxRequiredSamples]
    std::size_t                      maxRequiredSamples;

    StreamingPoller(std::size_t minRequiredSamples_, std::size_t maxRequiredSamples_) : minRequiredSamples(minRequiredSamples_), maxRequiredSamples(maxRequiredSamples_) {
        if (minRequiredSamples > maxRequiredSamples) {
            throw gr::exception(std::format("Failed to create StreamingPoller: minRequiredSamples ({}) > maxRequiredSamples ({})", minRequiredSamples, maxRequiredSamples));
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
            throw gr::exception(std::format("Failed to create DataSetPoller: minRequiredSamples ({}) > maxRequiredSamples ({})", minRequiredSamples, maxRequiredSamples));
        }
    }

    [[nodiscard]] bool process(std::invocable<std::span<DataSet<T>>> auto fnc, std::size_t requested = std::numeric_limits<std::size_t>::max()) {
        const std::size_t nProcess = detail::calculateNSamplesToProcess(reader.available(), requested, minRequiredSamples, maxRequiredSamples);
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
    template<DataSinkOrDataSetSinkLike TSink>
    void registerSink(TSink* sink, std::source_location location = std::source_location::current()) {
        std::lock_guard lg{_mutex};

        if (sink->_registered || sink->signal_name.value.empty()) {
            return;
        }

        if (_sink_by_signal_name.contains(sink->signal_name)) {
            throw gr::exception(std::format("Failed to register sink `{}`. Sink with the signal_name `{}` is already registered.", sink->name, sink->signal_name), location);
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
            throw gr::exception(std::format("Failed to update signal_name of sink `{}`. The old signal_name is an empty string.", sink->name), location);
        }

        if (newName.empty()) {
            throw gr::exception(std::format("Failed to update signal_name of sink `{}`. The new signal_name is an empty string.", sink->name), location);
        }

        if (_sink_by_signal_name.contains(std::string{newName})) {
            throw gr::exception(std::format("Failed to update signal_name of sink `{}`. Sink with signal_name `{}` is already registered.", sink->name, newName), location);
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

__attribute__((visibility("default"))) inline DataSinkRegistry& globalDataSinkRegistry() {
    static DataSinkRegistry instance;
    return instance;
}

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

template<typename T>
[[nodiscard]] inline DataSet<T> createDataset(const Metadata& metadata, std::size_t reserveSize = 0UZ) {
    DataSet<T> ds;
    ds.signal_names      = {metadata.signalName};
    ds.signal_quantities = {metadata.signalQuantity};
    ds.signal_units      = {metadata.signalUnit};
    ds.signal_ranges     = {{static_cast<T>(metadata.signalMin), static_cast<T>(metadata.signalMax)}};

    ds.timestamp = 0ULL;

    ds.axis_names = {"Time"};
    ds.axis_units = {"a.u."};

    ds.extents.resize(1UZ);
    ds.layout = gr::LayoutRight{};

    ds.timing_events.resize(1UZ);
    ds.axis_values.resize(1UZ);

    ds.meta_information.resize(1UZ);

    if (reserveSize != 0UZ) {
        ds.signal_values.reserve(reserveSize);
        ds.axis_values[0].reserve(reserveSize);
    }

    return ds;
}

inline constexpr void checkTag([[maybe_unused]] const Tag& tag) {
#ifndef NDEBUG
    if (tag.index != 0) { // DataSink enforces input_chunk_size == 1
        std::println(stderr, "DataSink always has input_chunk_size == 1 and only Tags with index == 0 are available; tag.index:{}, tag.map:{}", tag.index, tag.map);
        std::abort();
    }
#endif
}
} // namespace detail

GR_REGISTER_BLOCK(gr::basic::DataSink, [T], [ float, double ])

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

    GR_MAKE_REFLECTABLE(DataSink, in, sample_rate, signal_name, signal_quantity, signal_unit, signal_min, signal_max);

    bool _registered = false; // status should be updated by DataSinkRegistry

    using Block<DataSink<T>>::Block; // needed to inherit mandatory base-class Block(property_map) constructor

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/) {
        if (oldSettings.contains("signal_name")) {
            const std::string oldSignalName = std::get<std::string>(oldSettings.at("signal_name"));
            if (oldSignalName.empty()) {
                globalDataSinkRegistry().registerSink(this);
            } else if (oldSignalName != signal_name && _registered) {
                globalDataSinkRegistry().updateSignalName(this, oldSignalName, signal_name);
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
        addListener(std::make_unique<ContinuousListener<>>(handler, withBackpressure, *this), withBackpressure);
        return handler;
    }

    template<trigger::Matcher TMatcher>
    std::shared_ptr<DataSetPoller<T>> getTriggerPoller(TMatcher&& matcher, PollerConfig config = {}) {
        const auto      withBackpressure = config.overflowPolicy == OverflowPolicy::Backpressure;
        auto            handler          = std::make_shared<DataSetPoller<T>>(config.minRequiredSamples, config.maxRequiredSamples);
        std::lock_guard lg(_listener_mutex);
        handler->finished = _listeners_finished;
        addListener(std::make_unique<TriggerListener<TMatcher>>(std::forward<TMatcher>(matcher), handler, config.preSamples, config.postSamples, withBackpressure), withBackpressure);
        ensureHistorySize(config.preSamples);
        return handler;
    }

    template<trigger::Matcher TMatcher>
    std::shared_ptr<DataSetPoller<T>> getMultiplexedPoller(TMatcher&& matcher, PollerConfig config = {}) {
        std::lock_guard lg(_listener_mutex);
        const auto      withBackpressure = config.overflowPolicy == OverflowPolicy::Backpressure;
        auto            handler          = std::make_shared<DataSetPoller<T>>(config.minRequiredSamples, config.maxRequiredSamples);
        addListener(std::make_unique<MultiplexedListener<TMatcher>>(std::forward<TMatcher>(matcher), config.maximumWindowSize, handler, withBackpressure), withBackpressure);
        return handler;
    }

    template<trigger::Matcher TMatcher>
    std::shared_ptr<DataSetPoller<T>> getSnapshotPoller(TMatcher&& matcher, PollerConfig config = {}) {
        const auto      withBackpressure = config.overflowPolicy == OverflowPolicy::Backpressure;
        auto            handler          = std::make_shared<DataSetPoller<T>>(config.minRequiredSamples, config.maxRequiredSamples);
        std::lock_guard lg(_listener_mutex);
        addListener(std::make_unique<SnapshotListener<TMatcher>>(std::forward<TMatcher>(matcher), config.delay, handler, withBackpressure), withBackpressure);
        return handler;
    }

    template<StreamCallback<T> TCallback>
    void registerStreamingCallback(std::size_t maxChunkSize, TCallback&& callback) {
        std::lock_guard lg(_listener_mutex);
        addListener(std::make_unique<ContinuousListener<TCallback>>(maxChunkSize, std::forward<TCallback>(callback), *this), false);
    }

    template<trigger::Matcher TMatcher, DataSetCallback<T> TCallback>
    void registerTriggerCallback(TMatcher&& matcher, std::size_t preSamples, std::size_t postSamples, TCallback&& callback) {
        std::lock_guard lg(_listener_mutex);
        addListener(std::make_unique<TriggerListener<TMatcher, TCallback>>(std::forward<TMatcher>(matcher), preSamples, postSamples, std::forward<TCallback>(callback)), false);
        ensureHistorySize(preSamples);
    }

    template<trigger::Matcher TMatcher, DataSetCallback<T> TCallback>
    void registerMultiplexedCallback(TMatcher&& matcher, std::size_t maximumWindowSize, TCallback&& callback) {
        std::lock_guard lg(_listener_mutex);
        addListener(std::make_unique<MultiplexedListener<TMatcher, TCallback>>(std::forward<TMatcher>(matcher), maximumWindowSize, std::forward<TCallback>(callback)), false);
    }

    template<trigger::Matcher TMatcher, DataSetCallback<T> TCallback>
    void registerSnapshotCallback(TMatcher&& matcher, std::chrono::nanoseconds delay, TCallback&& callback) {
        std::lock_guard lg(_listener_mutex);
        addListener(std::make_unique<SnapshotListener<TMatcher, TCallback>>(std::forward<TMatcher>(matcher), delay, std::forward<TCallback>(callback)), false);
    }

    void stop() noexcept {
        globalDataSinkRegistry().unregisterSink(this);
        std::lock_guard lg(_listener_mutex);
        for (auto& listener : _listeners) {
            listener->stop();
        }
        _listeners_finished = true;
    }

    [[nodiscard]] work::Status processBulk(InputSpanLike auto& inData) noexcept {
        std::lock_guard lg(_listener_mutex); // TODO review/profile if a lock-free data structure should be used here
        const auto      historyView = _history ? _history->get_span(0UZ) : std::span<const T>();
        std::erase_if(_listeners, [](const auto& l) { return l->expired; });
        for (auto& listener : _listeners) {
            listener->process(historyView, inData, this->inputTags());
        }
        if (_history) {
            _history->push_front(inData);
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

        virtual void process(std::span<const T> history, std::span<const T> data, std::span<const Tag> tags) = 0;
        virtual void stop()                                                                                  = 0;
    };

    template<trigger::Matcher TMatcher, typename TCallback>
    struct DataSetBaseListener : public AbstractListener {
        static constexpr bool hasCallback = !std::is_same_v<TCallback, gr::meta::null_type>;

        TMatcher                        _matcher;
        gr::property_map                _matcherState;
        detail::Metadata                _metadata;
        std::weak_ptr<DataSetPoller<T>> _poller;
        TCallback                       _callback;
        bool                            _isBlocking = false;

        template<trigger::Matcher TMatcherFW>
        explicit DataSetBaseListener(TMatcherFW&& matcher, std::shared_ptr<DataSetPoller<T>> poller, bool isBlocking)
        requires(!hasCallback)
            : _matcher(std::forward<TMatcherFW>(matcher)), _poller(std::move(poller)), _isBlocking(isBlocking) {}

        template<trigger::Matcher TMatcherFW, typename TCallbackFW>
        explicit DataSetBaseListener(TMatcherFW&& matcher, TCallbackFW&& callback)
        requires(hasCallback)
            : _matcher(std::forward<TMatcherFW>(matcher)), _callback(std::forward<TCallbackFW>(callback)) {}

        inline void publishDataSet(DataSet<T>&& data) {
            if constexpr (hasCallback) {
                _callback(std::move(data));
            } else {
                auto pollerPtr = _poller.lock();
                if (!pollerPtr) {
                    this->setExpired();
                    return;
                }

                if (_isBlocking || pollerPtr->writer.available() > 0) {
                    auto writeData = pollerPtr->writer.reserve(1UZ);
                    writeData[0UZ] = std::move(data);
                    writeData.publish(1UZ);
                } else {
                    pollerPtr->dropCount++;
                }
            }
        }

        void setMetadata(detail::Metadata metadata) override { _metadata = std::move(metadata); }
    };

    template<typename TCallback = gr::meta::null_type>
    struct ContinuousListener : public AbstractListener {
        static constexpr bool hasCallback       = !std::is_same_v<TCallback, gr::meta::null_type>;
        static constexpr bool callbackTakesTags = std::is_invocable_v<TCallback, std::span<const T>, std::span<const Tag>> || std::is_invocable_v<TCallback, std::span<const T>, std::span<const Tag>, const DataSink<T>&>;

        const DataSink<T>&              _parentSink;
        bool                            _isBlocking        = false;
        std::size_t                     _nPublishedSamples = 0;
        std::optional<detail::Metadata> _pendingMetadata;

        // callback mode only
        std::size_t _maxChunkSize; // // 0 = no limit; process all available samples at once
        TCallback   _callback;

        // poller mode only
        std::weak_ptr<StreamingPoller<T>> _poller = {};

        template<typename TCallbackFW>
        explicit ContinuousListener(std::size_t maxChunkSize, TCallbackFW&& c, const DataSink<T>& parent)
        requires(hasCallback)
            : _parentSink(parent), _maxChunkSize(maxChunkSize), _callback{std::forward<TCallbackFW>(c)} {}

        explicit ContinuousListener(std::shared_ptr<StreamingPoller<T>> poller, bool isBlocking, const DataSink<T>& parent)
        requires(!hasCallback)
            : _parentSink(parent), _isBlocking(isBlocking), _poller{std::move(poller)} {}

        inline void callCallback(std::span<const T> data, std::span<const Tag> tags) {
            if constexpr (std::is_invocable_v<TCallback, std::span<const T>, std::span<const Tag>, const DataSink<T>&>) {
                _callback(data, tags, _parentSink);
            } else if constexpr (std::is_invocable_v<TCallback, std::span<const T>, std::span<const Tag>>) {
                _callback(data, tags);
            } else {
                _callback(data);
            }
        }

        void setMetadata(detail::Metadata metadata) override { _pendingMetadata = std::move(metadata); }

        void process(std::span<const T>, std::span<const T> data, std::span<const Tag> tags) override {
            if constexpr (hasCallback) {
                const std::size_t nSamples = data.size();
                const std::size_t chunk    = _maxChunkSize != 0 ? _maxChunkSize : nSamples;
                std::size_t       offset   = 0;

                while (offset < nSamples) {
                    const std::size_t len   = std::min(chunk, nSamples - offset);
                    auto              slice = data.subspan(offset, len);
                    if constexpr (callbackTakesTags) {
                        const std::size_t end            = offset + len;
                        const auto        nTagsInRange   = static_cast<std::size_t>(std::ranges::count_if(tags, [offset, end](const Tag& t) { return t.index >= offset && t.index < end; }));
                        const std::size_t nTagsToPublish = _pendingMetadata.has_value() ? nTagsInRange + 1 : nTagsInRange;
                        std::vector<Tag>  tmpTags;
                        if (nTagsToPublish > 0) {
                            tmpTags.reserve(nTagsToPublish);
                            if (_pendingMetadata.has_value()) {
                                tmpTags.emplace_back(0, _pendingMetadata->toTagMap());
                                _pendingMetadata.reset();
                            }

                            for (const Tag& tag : tags) {
                                if (tag.index >= offset && tag.index < end) {
                                    tmpTags.emplace_back(tag.index - offset, tag.map);
                                }
                            }
                        }
                        callCallback(slice, std::span(tmpTags));
                    } else {
                        _callback(slice);
                    }
                    _nPublishedSamples += len;
                    offset += len;
                }
            } else {
                auto poller = _poller.lock();
                if (!poller) {
                    this->setExpired();
                    return;
                }

                const std::size_t nSamplesToPublish  = _isBlocking ? data.size() : std::min(data.size(), poller->writer.available());
                const auto        nTagsInRange       = static_cast<std::size_t>(std::ranges::count_if(tags, [nSamplesToPublish](const Tag& t) { return t.index < nSamplesToPublish; }));
                const std::size_t availableInputTags = _pendingMetadata.has_value() ? nTagsInRange + 1 : nTagsInRange;
                const std::size_t nTagsToPublish     = _isBlocking ? availableInputTags : std::min(availableInputTags, poller->tagWriter.available());

                if (nSamplesToPublish > 0) {
                    if (nTagsToPublish > 0) {
                        WriterSpanLike auto outTags = poller->tagWriter.reserve(nTagsToPublish);
                        std::size_t         counter = 0UZ;
                        if (_pendingMetadata.has_value()) {
                            outTags[counter++] = {_nPublishedSamples, _pendingMetadata->toTagMap()};
                            _pendingMetadata.reset();
                        }
                        for (const Tag& tag : tags) {
                            if (counter >= nTagsToPublish) {
                                break;
                            }
                            if (tag.index < nSamplesToPublish) {
                                outTags[counter++] = {_nPublishedSamples + tag.index, tag.map};
                            }
                        }
                        outTags.publish(nTagsToPublish);
                    }
                    WriterSpanLike auto outSamples = poller->writer.reserve(nSamplesToPublish);
                    std::ranges::copy(data | std::views::take(nSamplesToPublish), outSamples.begin());
                    outSamples.publish(nSamplesToPublish);
                }
                poller->droppedSampleCount += data.size() - nSamplesToPublish;
                poller->droppedTagCount += availableInputTags - nTagsToPublish;
                _nPublishedSamples += nSamplesToPublish;
            }
        }

        void stop() override {
            if constexpr (!hasCallback) {
                if (auto p = _poller.lock()) {
                    p->finished = true;
                }
            }
        }
    }; // struct ContinuousListener

    struct TriggerCapture {
        DataSet<T>  dataset;
        std::size_t nRemainingPostSamples = 0;
    };

    template<trigger::Matcher TMatcher, typename TCallback = gr::meta::null_type>
    struct TriggerListener : public DataSetBaseListener<TMatcher, TCallback> {
        using Base = DataSetBaseListener<TMatcher, TCallback>;

        std::size_t                _preSamples  = 0;
        std::size_t                _postSamples = 0;
        std::deque<TriggerCapture> _triggerCaptures; // trigger captures that still didn't receive all their data

        template<trigger::Matcher TMatcherFW>
        explicit TriggerListener(TMatcherFW&& matcher, std::shared_ptr<DataSetPoller<T>> poller_, std::size_t pre, std::size_t post, bool isBlocking)
        requires(!Base::hasCallback)
            : DataSetBaseListener<TMatcher, TCallback>(std::forward<TMatcherFW>(matcher), std::move(poller_), isBlocking), _preSamples(pre), _postSamples(post) {}

        template<trigger::Matcher TMatcherFW, typename TCallbackFW>
        explicit TriggerListener(TMatcherFW&& matcher, std::size_t pre, std::size_t post, TCallbackFW&& cb)
        requires(Base::hasCallback)
            : DataSetBaseListener<TMatcher, TCallback>(std::forward<TMatcherFW>(matcher), std::forward<TCallbackFW>(cb)), _preSamples(pre), _postSamples(post) {}

        void process(std::span<const T> history, std::span<const T> inData, std::span<const Tag> tags) override {
            for (const Tag& tag : tags) {
                detail::checkTag(tag);

                if (!tag.map.empty() && this->_matcher("", tag, this->_matcherState) == trigger::MatchResult::Matching) {
                    const std::size_t minPreSamples = std::min(_preSamples, history.size());
                    DataSet<T>        dataset       = detail::createDataset<T>(this->_metadata, minPreSamples + _postSamples);
                    std::ranges::copy(history.subspan(0UZ, minPreSamples) | std::views::reverse, std::back_inserter(dataset.signal_values));
                    std::ranges::copy(std::views::iota(0UZ, minPreSamples), std::back_inserter(dataset.axis_values[0]));
                    dataset.extents[0UZ] = static_cast<std::int32_t>(dataset.signalValues(0UZ).size());
                    dataset.timing_events[0UZ].emplace_back(static_cast<std::ptrdiff_t>(minPreSamples), tag.map);
                    _triggerCaptures.push_back({.dataset = std::move(dataset), .nRemainingPostSamples = _postSamples});
                }
            }

            auto capture = _triggerCaptures.begin();
            while (capture != _triggerCaptures.end()) {
                const std::size_t nPostSamples = std::min(capture->nRemainingPostSamples, inData.size());
                const std::size_t oldSize      = capture->dataset.signalValues(0UZ).size();
                std::ranges::copy(inData.first(nPostSamples), std::back_inserter(capture->dataset.signal_values));
                const std::size_t newSize = capture->dataset.signalValues(0UZ).size();
                std::ranges::copy(std::views::iota(oldSize, newSize), std::back_inserter(capture->dataset.axis_values[0UZ]));
                capture->dataset.extents[0UZ] = static_cast<std::int32_t>(newSize);

                capture->nRemainingPostSamples -= nPostSamples;

                if (capture->nRemainingPostSamples == 0) {
                    this->publishDataSet(std::move(capture->dataset));
                    capture = _triggerCaptures.erase(capture);
                } else {
                    ++capture;
                }
            }
        }

        void stop() override {
            for (auto& window : _triggerCaptures) {
                if (!window.dataset.signal_values.empty()) {
                    this->publishDataSet(std::move(window.dataset));
                }
            }
            _triggerCaptures.clear();
            if (auto p = this->_poller.lock()) {
                p->finished = true;
            }
        }
    }; // struct TriggerListener

    template<trigger::Matcher TMatcher, typename TCallback = gr::meta::null_type>
    struct MultiplexedListener : public DataSetBaseListener<TMatcher, TCallback> {
        using Base = DataSetBaseListener<TMatcher, TCallback>;

        std::optional<DataSet<T>> _curDataset;
        std::size_t               _maxDataSetSize; // 0 == unlimited

        template<typename TCallbackFW, trigger::Matcher TMatcherFW>
        explicit MultiplexedListener(TMatcherFW&& matcher, std::size_t maxDataSetSize, TCallbackFW&& callback)
        requires(Base::hasCallback)
            : DataSetBaseListener<TMatcher, TCallback>(std::forward<TMatcherFW>(matcher), std::forward<TCallbackFW>(callback)), _maxDataSetSize(maxDataSetSize) {}

        template<trigger::Matcher TMatcherFW>
        explicit MultiplexedListener(TMatcherFW&& matcher, std::size_t maxDataSetSize, std::shared_ptr<DataSetPoller<T>> poller, bool isBlocking)
        requires(!Base::hasCallback)
            : DataSetBaseListener<TMatcher, TCallback>(std::forward<TMatcherFW>(matcher), std::move(poller), isBlocking), _maxDataSetSize(maxDataSetSize) {}

        void process(std::span<const T>, std::span<const T> inData, std::span<const Tag> tags) override {
            // Overlapping trigger windows are not supported.
            // Examples:
            //   Start1–Stop1–Start2–Stop2 → OK: produces two DataSets: [Start1, Stop1) and [Start2, Stop2).
            //   Start1–Start2–Stop1–Stop2 → When a capture is active, a new Start ends and publishes the current DataSet, then immediately starts a new one.
            //   Result: two DataSets — [Start1, Start2) and [Start2, Stop1). Stop2 is ignored.
            for (const Tag& tag : tags) {
                detail::checkTag(tag);
                // MatchResult::Matching == START trigger, MatchResult::NotMatching == STOP trigger
                const auto matchResult = this->_matcher("", tag, this->_matcherState);
                // Close current capture on either explicit Stop or implicit Stop (new Start)
                if (matchResult == trigger::MatchResult::NotMatching || matchResult == trigger::MatchResult::Matching) {
                    if (_curDataset) {
                        _curDataset->timing_events[0UZ].emplace_back(static_cast<std::ptrdiff_t>(_curDataset->signalValues(0UZ).size()), tag.map);
                        this->publishDataSet(std::move(*_curDataset));
                        _curDataset.reset();
                    }
                }
                if (matchResult == trigger::MatchResult::Matching) {
                    _curDataset = detail::createDataset<T>(this->_metadata, _maxDataSetSize);
                    _curDataset->timing_events[0UZ].emplace_back(0, tag.map);

                    // Multiple Start/Stop tags may share the same sample index; we start on the first Start and
                    // ignore any further tags at this index to prevent empty DataSets at the same point.
                    // This should not happen in practice.
                    continue;
                }
            }
            if (_curDataset) {
                const std::size_t maxSize = (_maxDataSetSize == 0) ? std::numeric_limits<std::size_t>::max() : _maxDataSetSize;
                const std::size_t oldSize = _curDataset->signalValues(0UZ).size();
                const std::size_t toWrite = std::min(inData.size(), maxSize - oldSize);
                const std::size_t newSize = oldSize + toWrite;
                if (toWrite > 0) {
                    std::ranges::copy(inData.first(toWrite), std::back_inserter(_curDataset->signal_values));
                    std::ranges::copy(std::views::iota(oldSize, newSize), std::back_inserter(_curDataset->axis_values[0UZ]));
                    _curDataset->extents[0UZ] = static_cast<std::int32_t>(newSize);
                }
                if (newSize == maxSize) {
                    this->publishDataSet(std::move(*_curDataset));
                    _curDataset.reset();
                }
            }
        }

        void stop() override {
            if (_curDataset) {
                this->publishDataSet(std::move(*_curDataset));
                _curDataset.reset();
            }
            if (auto p = this->_poller.lock()) {
                p->finished = true;
            }
        }
    }; // struct MultiplexedListener

    struct Snapshot {
        property_map tagMap;
        std::size_t  triggerIndex      = 0;
        std::size_t  nRemainingSamples = 0;
    };

    template<trigger::Matcher TMatcher, typename TCallback = gr::meta::null_type>
    struct SnapshotListener : public DataSetBaseListener<TMatcher, TCallback> {
        using Base = DataSetBaseListener<TMatcher, TCallback>;

        std::chrono::nanoseconds _timeDelay;       // set by the user
        std::size_t              _sampleDelay = 0; // calculated from _timeDelay and sample rate
        std::deque<Snapshot>     _snapshots;

        template<trigger::Matcher TMatcherFW>
        explicit SnapshotListener(TMatcherFW&& matcher, std::chrono::nanoseconds delay, std::shared_ptr<DataSetPoller<T>> poller, bool isBlocking)
        requires(!Base::hasCallback)
            : DataSetBaseListener<TMatcher, TCallback>(std::forward<TMatcherFW>(matcher), std::move(poller), isBlocking), _timeDelay(delay) {}

        template<typename TCallbackFW, trigger::Matcher TMatcherFW>
        explicit SnapshotListener(TMatcherFW&& matcher, std::chrono::nanoseconds delay, TCallbackFW&& callback)
        requires(Base::hasCallback)
            : DataSetBaseListener<TMatcher, TCallback>(std::forward<TMatcherFW>(matcher), std::forward<TCallbackFW>(callback)), _timeDelay(delay) {}

        void setMetadata(detail::Metadata metadata) override {
            _sampleDelay    = static_cast<std::size_t>(std::round(std::chrono::duration_cast<std::chrono::duration<float>>(_timeDelay).count() * metadata.sampleRate));
            this->_metadata = std::move(metadata);
        }

        void process(std::span<const T>, std::span<const T> inData, std::span<const Tag> tags) override {
            for (const Tag& tag : tags) {
                detail::checkTag(tag);
                if (!tag.map.empty() && this->_matcher("", tag, this->_matcherState) == trigger::MatchResult::Matching) {
                    _snapshots.push_back(Snapshot{tag.map, _sampleDelay, _sampleDelay});
                }
            }

            auto it = _snapshots.begin();
            while (it != _snapshots.end()) {
                if (it->nRemainingSamples >= inData.size()) {
                    it->nRemainingSamples -= inData.size();
                    ++it;
                } else {
                    DataSet<T> dataset = detail::createDataset<T>(this->_metadata, 1UZ);
                    dataset.timing_events[0UZ].emplace_back(-static_cast<std::ptrdiff_t>(it->triggerIndex), std::move(it->tagMap));
                    dataset.signal_values.push_back(inData[it->nRemainingSamples]);
                    dataset.axis_values[0].push_back(0);
                    dataset.extents[0UZ] = std::int32_t(1);
                    this->publishDataSet(std::move(dataset));
                    it = _snapshots.erase(it);
                }
            }
        }

        void stop() override {
            _snapshots.clear();
            if (auto p = this->_poller.lock()) {
                p->finished = true;
            }
        }
    }; // struct SnapshotListener
};

GR_REGISTER_BLOCK(gr::basic::DataSetSink, [T], [ float, double ])
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
                globalDataSinkRegistry().registerSink(this);
            } else if (oldSignalName != signal_name && _registered) {
                globalDataSinkRegistry().updateSignalName(this, oldSignalName, signal_name);
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
        globalDataSinkRegistry().unregisterSink(this);
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
                    auto writeData = poller->writer.reserve(1UZ);
                    writeData[0UZ] = std::move(data);
                    writeData.publish(1UZ);
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

#endif
