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
#include <functional>
#include <limits>

namespace gr::basic {

enum class CaptureMode : std::uint8_t { Trigger, Multiplexed, Snapshot };

enum class OverflowPolicy : std::uint8_t { Backpressure, Drop };

struct PollerConfig {
    OverflowPolicy           overflowPolicy    = OverflowPolicy::Backpressure;
    std::size_t              preSamples        = 100;                           // CaptureMode::Trigger
    std::size_t              postSamples       = 100;                           // CaptureMode::Trigger
    std::size_t              maximumWindowSize = 1000;                          // CaptureMode::Multiplexed, 0: unbounded
    std::chrono::nanoseconds delay             = std::chrono::milliseconds{10}; // CaptureMode::Snapshot: from the trigger to the captured sample
};

template<typename T>
struct DataSink;

template<typename T>
using DataSetSink = DataSink<DataSet<T>>;

template<typename T, typename V>
concept DataSetCallback = std::invocable<T, DataSet<V>>;

// Stream callback functions receive the span of data, with optional tags and reference to the sink.
template<typename T, typename V>
concept StreamCallback = std::invocable<T, std::span<const V>> || std::invocable<T, std::span<const V>, std::span<const Tag>> || std::invocable<T, std::span<const V>, std::span<const Tag>, const DataSink<V>&>;

namespace detail {
constexpr std::size_t data_sink_buffer_size          = 65536;
constexpr std::size_t data_sink_tag_buffer_size      = 1024;
constexpr std::size_t data_sink_data_set_buffer_size = 1024;
} // namespace detail

template<typename T>
struct Poller {
    gr::CircularBuffer<T>                                    buffer             = gr::CircularBuffer<T>(DataSetLike<T> ? detail::data_sink_data_set_buffer_size : detail::data_sink_buffer_size);
    decltype(buffer.new_reader())                            reader             = buffer.new_reader();
    decltype(buffer.new_writer())                            writer             = buffer.new_writer();
    gr::CircularBuffer<std::pair<std::size_t, property_map>> tagBuffer          = gr::CircularBuffer<std::pair<std::size_t, property_map>>(detail::data_sink_tag_buffer_size); // owning: Tag only views its map
    decltype(tagBuffer.new_reader())                         tagReader          = tagBuffer.new_reader();
    decltype(tagBuffer.new_writer())                         tagWriter          = tagBuffer.new_writer();
    std::size_t                                              samplesRead        = 0;
    std::atomic<bool>                                        finished           = false;
    std::atomic<std::size_t>                                 droppedSampleCount = 0;
    std::atomic<std::size_t>                                 droppedTagCount    = 0;
    std::shared_ptr<gr::Sequence>                            _progress;

    explicit Poller(std::shared_ptr<gr::Sequence> progress) : _progress(std::move(progress)) {}

    template<typename THandler>
    requires std::invocable<THandler&, std::span<const T>> || std::invocable<THandler&, std::span<const T>, std::span<const Tag>>
    [[nodiscard]] bool process(THandler fnc, std::size_t requested = std::numeric_limits<std::size_t>::max()) {
        auto callHandler = [&fnc](std::span<const T> data, std::span<const Tag> tags) {
            if constexpr (std::invocable<THandler&, std::span<const T>, std::span<const Tag>>) {
                fnc(data, tags);
            } else {
                fnc(data);
            }
        };

        const std::size_t nProcess = std::min(reader.available(), requested);
        if (nProcess == 0UZ) {
            return false;
        }

        {
            ReaderSpanLike auto readData     = reader.get(nProcess);
            ReaderSpanLike auto tags         = tagReader.get();
            const auto          it           = std::ranges::find_if_not(tags, [until = samplesRead + nProcess](const auto& tag) { return tag.first < until; });
            const auto          relevantTags = std::span(tags.begin(), it) | std::views::transform([this](const auto& v) { return Tag{v.first - samplesRead, v.second}; }) | std::ranges::to<std::vector>();
            callHandler(readData, relevantTags);
            std::ignore = tags.consume(relevantTags.size());
            std::ignore = readData.consume(nProcess);
        }
        samplesRead += nProcess;
        _progress->incrementAndGet();
        _progress->notify_all();
        return true;
    }
};

struct DataSinkQuery {
    std::optional<std::string> _sink_name;
    std::optional<std::string> _signal_name;

    static DataSinkQuery signalName(std::string_view name) { return {{}, std::string{name}}; }

    static DataSinkQuery sinkName(std::string_view name) { return {std::string{name}, {}}; }
};

struct DataSinkRegistry {
    std::mutex                      _mutex;
    std::map<std::string, std::any> _sink_by_signal_name;

    template<typename T>
    [[nodiscard]] std::expected<void, gr::Error> registerSink(DataSink<T>* sink, std::source_location location = std::source_location::current()) {
        std::lock_guard lg{_mutex};

        if (sink->_registered || sink->signal_name.value.empty()) {
            return {};
        }

        const std::string signalName{std::string_view(sink->signal_name)};
        if (_sink_by_signal_name.contains(signalName)) {
            return std::unexpected(gr::Error(std::format("Failed to register sink `{}`. Sink with the signal_name `{}` is already registered.", sink->name, signalName), location));
        }

        _sink_by_signal_name[signalName] = sink;
        sink->_registered                = true;
        return {};
    }

    template<typename T>
    void unregisterSink(DataSink<T>* sink) {
        std::lock_guard lg{_mutex};
        std::erase_if(_sink_by_signal_name, [sink](const auto& entry) { return asSink<DataSink<T>>(entry.second) == sink; });
        sink->_registered = false;
    }

    template<typename T>
    [[nodiscard]] std::expected<void, gr::Error> updateSignalName(DataSink<T>* sink, std::string_view newName, std::source_location location = std::source_location::current()) {
        std::lock_guard lg{_mutex};

        if (newName.empty()) {
            return std::unexpected(gr::Error(std::format("Failed to update signal_name of sink `{}`. The new signal_name is an empty string.", sink->name), location));
        }

        const auto it = _sink_by_signal_name.find(std::string{newName});
        if (it != _sink_by_signal_name.end() && asSink<DataSink<T>>(it->second) != sink) {
            return std::unexpected(gr::Error(std::format("Failed to update signal_name of sink `{}`. Sink with signal_name `{}` is already registered.", sink->name, newName), location));
        }

        std::erase_if(_sink_by_signal_name, [sink](const auto& entry) { return asSink<DataSink<T>>(entry.second) == sink; });
        _sink_by_signal_name[std::string(newName)] = sink;
        return {};
    }

    template<typename T>
    std::shared_ptr<Poller<T>> getPoller(const DataSinkQuery& query, PollerConfig config = {}) {
        std::lock_guard lg{_mutex};
        auto            sink = find<DataSink<T>>(query);
        return sink ? sink->getPoller(config) : nullptr;
    }

    template<typename T, trigger::Matcher TMatcher>
    std::shared_ptr<Poller<DataSet<T>>> getPoller(const DataSinkQuery& query, CaptureMode mode, TMatcher&& matcher, PollerConfig config = {}) {
        std::lock_guard lg{_mutex};
        auto            sink = find<DataSink<T>>(query);
        return sink ? sink->getPoller(mode, std::forward<TMatcher>(matcher), config) : nullptr;
    }

    template<typename T, StreamCallback<T> TCallback>
    bool registerCallback(const DataSinkQuery& query, TCallback&& callback) {
        std::lock_guard lg{_mutex};
        auto            sink = find<DataSink<T>>(query);
        if (sink) {
            sink->registerCallback(std::forward<TCallback>(callback));
        }
        return sink != nullptr;
    }

    template<typename T, trigger::Matcher TMatcher, DataSetCallback<T> TCallback>
    bool registerCallback(const DataSinkQuery& query, CaptureMode mode, TMatcher&& matcher, PollerConfig config, TCallback&& callback) {
        std::lock_guard lg{_mutex};
        auto            sink = find<DataSink<T>>(query);
        if (sink) {
            sink->registerCallback(mode, std::forward<TMatcher>(matcher), config, std::forward<TCallback>(callback));
        }
        return sink != nullptr;
    }

    template<typename TSink>
    static TSink* asSink(const std::any& value) {
        const auto* sink = std::any_cast<TSink*>(&value);
        return sink != nullptr ? *sink : nullptr;
    }

    template<typename T>
    T* find(const DataSinkQuery& query) {
        if (query._signal_name) {
            const auto it = _sink_by_signal_name.find(*query._signal_name);
            return it != _sink_by_signal_name.end() ? asSink<T>(it->second) : nullptr;
        }
        for (const std::any& value : std::views::values(_sink_by_signal_name)) {
            T* sink = asSink<T>(value);
            if (sink != nullptr && query._sink_name && std::string_view{*query._sink_name} == std::string_view{sink->name.value}) {
                return sink;
            }
        }
        return nullptr;
    }
};

__attribute__((visibility("default"))) inline DataSinkRegistry& globalDataSinkRegistry() {
    static DataSinkRegistry instance;
    return instance;
}

namespace detail {

struct Metadata {
    float            sampleRate;
    std::pmr::string signalName;
    std::pmr::string signalQuantity;
    std::pmr::string signalUnit;
    float            signalMin;
    float            signalMax;

    property_map toTagMap() const {
        return property_map{                                          //
            {std::pmr::string(tag::kSampleRate), sampleRate},         //
            {std::pmr::string(tag::kSignalName), signalName},         //
            {std::pmr::string(tag::kSignalQuantity), signalQuantity}, //
            {std::pmr::string(tag::kSignalUnit), signalUnit},         //
            {std::pmr::string(tag::kSignalMin), signalMin},           //
            {std::pmr::string(tag::kSignalMax), signalMax}};
    }
};

template<typename T>
[[nodiscard]] inline DataSet<T> createDataset(const Metadata& metadata, std::size_t reserveSize = 0UZ) {
    DataSet<T> ds;
    ds.signal_names.emplace_back(metadata.signalName);
    ds.signal_quantities.emplace_back(metadata.signalQuantity);
    ds.signal_units.emplace_back(metadata.signalUnit);
    ds.signal_ranges = {{static_cast<T>(metadata.signalMin), static_cast<T>(metadata.signalMax)}};

    ds.axis_names = {"Time"};
    ds.axis_units = {"a.u."};

    ds.extents.resize(1UZ);
    ds.layout = gr::LayoutRight{};

    ds.timing_events.resize(1UZ);
    ds.axis_values.resize(1UZ);

    ds.meta_information.resize(1UZ);

    if (reserveSize != 0UZ) {
        ds.signal_values.reserve(reserveSize);
        ds.axis_values[0UZ].reserve(reserveSize);
    }

    return ds;
}

template<typename T>
void appendSamples(DataSet<T>& dataset, std::ranges::input_range auto&& samples) {
    const std::size_t oldSize = dataset.signal_values.size();
    std::ranges::copy(samples, std::back_inserter(dataset.signal_values));
    const std::size_t newSize = dataset.signal_values.size();
    std::ranges::copy(std::views::iota(oldSize, newSize), std::back_inserter(dataset.axis_values[0UZ]));
    dataset.extents[0UZ] = static_cast<std::int32_t>(newSize);
}
} // namespace detail

GR_REGISTER_BLOCK(gr::basic::DataSink, [T], [ float, double ])
GR_REGISTER_BLOCK("gr::basic::DataSetSink", gr::basic::DataSetSink, [T], [ float, double ])

template<typename T>
struct DataSink : Block<DataSink<T>> {
    using Description = Doc<R""(@brief generic data sink for exporting arbitrary-typed streams to non-GR C++ APIs.

Each sink registers with a (user-defined/exchangeable) global registry that can be
queried by the non-GR caller to find the sink responsible for a given signal name, etc.
and either retrieve a poller handler that allows asynchronous data from a different thread,
or register a callback that is invoked by the sink if the user-conditions are met.

<pre>
@code
        ╔═══════════════╗
    in ━╢   data sink   ║                      ┌──── caller ────┐
        ║               ║                      │                │
        ║               ║  retrieve poller or  │ (custom non-GR │
        ║ :signal_name  ║←--------------------→│  user code...) │
        ║ :signal_unit  ║  register            │                │
        ║ :...          ║  callback function   └───┬────────────┘
        ╚═ GR block ═╤══╝                          │
                     │                             │
                     │                             │
                     │      ╭─registry─╮           │
           register/ │      ╞══════════╡           │ queries by
         deregister  ╰─────→│ [sinks]  │←──────────╯ DataSinkQuery
                            ╞══════════╡
                            ╰──────────╯

</pre>
A full poller either holds back the flow graph (OverflowPolicy::Backpressure)
or drops what does not fit (OverflowPolicy::Drop); callbacks are called synchronously and never drop.
N.B. due to the nature of the GR scheduler, signals from the same sink are notified
synchronously (/asynchronously) if handled by the same (/different) sink block.

For DataSet<U> input streams only streaming pollers and callbacks are available; they hand out
the data sets as they arrive, without the stream's tags.

@tparam T input sample type, or DataSet<U>
)"">;

    struct AbstractListener;

    PortIn<T, RequiredSamples<std::dynamic_extent, detail::data_sink_buffer_size>> in;

    gr::SampleRate     sample_rate = 1.f;
    gr::SignalName     signal_name = ""; // empty: not registered until set by the user or a tag
    gr::SignalQuantity signal_quantity{};
    gr::SignalUnit     signal_unit{"a.u."};
    gr::SignalMin      signal_min = -1.0f;
    gr::SignalMax      signal_max = +1.0f;

    GR_MAKE_REFLECTABLE(DataSink, in, sample_rate, signal_name, signal_quantity, signal_unit, signal_min, signal_max);

    bool                                           _registered = false;
    std::vector<std::unique_ptr<AbstractListener>> _listeners;
    bool                                           _stopped = false;
    std::mutex                                     _listener_mutex;
    std::optional<gr::HistoryBuffer<T>>            _history;

    ~DataSink() { globalDataSinkRegistry().unregisterSink(this); }

    void stop() noexcept {
        globalDataSinkRegistry().unregisterSink(this);
        std::lock_guard lg(_listener_mutex);
        for (auto& listener : _listeners) {
            this->invokeUserProvidedFunction("stop()", [&listener] { listener->stop(); });
        }
        _stopped = true;
    }

    void reset() {
        {
            std::lock_guard lg(_listener_mutex);
            for (auto& listener : _listeners) {
                this->invokeUserProvidedFunction("reset()", [&listener] { listener->stop(); });
            }
            _listeners.clear();
            _history.reset();
            _stopped = false;
        }
        this->emitErrorMessageIfAny("reset()", globalDataSinkRegistry().registerSink(this));
    }

    [[nodiscard]] work::Status processBulk(InputSpanLike auto& inData) {
        std::vector<Tag> tags;
        if constexpr (!DataSetLike<T>) {
            for (const auto& [relIndex, tagMapRef] : inData.tags()) {
                if (relIndex == 0) {
                    tags.emplace_back(0UZ, tagMapRef.get());
                }
            }
        }

        std::lock_guard lg(_listener_mutex);
        if (lifecycle::isShuttingDown(this->state())) {
            return work::Status::DONE;
        }
        if (std::erase_if(_listeners, [](const auto& listener) { return listener->isExpired(); }) > 0UZ) {
            updateHistorySize();
        }
        const auto        minAcceptable = [&tags](std::size_t nAvailable, const auto& listener) { return std::min(nAvailable, listener->nAcceptable(nAvailable, tags)); };
        const std::size_t nSamples      = std::ranges::fold_left(_listeners, inData.size(), minAcceptable);
        if (nSamples == 0UZ && inData.size() > 0UZ) {
            return work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }
        if constexpr (!DataSetLike<T>) {
            for (const auto& [relIndex, tagMapRef] : inData.tags()) {
                if (relIndex != 0 && relIndex < static_cast<std::ptrdiff_t>(nSamples)) [[unlikely]] {
                    this->emitErrorMessage("processBulk()", std::format("dropped tag at chunk offset {}, expected 0: {}", relIndex, tagMapRef.get()));
                }
            }
        }
        std::erase_if(tags, [nSamples](const Tag& tag) { return tag.index >= nSamples; });

        const std::span<const T> samples     = inData.first(nSamples);
        const std::span<const T> historyView = _history ? _history->get_span(0UZ) : std::span<const T>();
        for (auto& listener : _listeners) {
            listener->process(historyView, samples, tags);
        }
        if (_history) {
            _history->push_front(samples);
        }
        std::ignore = inData.consume(nSamples);
        return work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& /*newSettings*/) {
        const std::string oldSignalName = oldSettings.value_or<std::string>("signal_name", std::string{});
        if (oldSignalName != signal_name) {
            if (_registered) {
                this->emitErrorMessageIfAny("settingsChanged()", globalDataSinkRegistry().updateSignalName(this, signal_name));
            } else {
                this->emitErrorMessageIfAny("settingsChanged()", globalDataSinkRegistry().registerSink(this));
            }
        }
        std::lock_guard lg{_listener_mutex};
        for (auto& listener : _listeners) {
            updateMetadata(*listener);
        }
    }

    std::shared_ptr<Poller<T>> getPoller(PollerConfig config = {}) {
        std::lock_guard lg(_listener_mutex);
        auto            poller = makePoller<T>();
        addListener(std::make_unique<StreamingListener>(poller, config.overflowPolicy));
        return poller;
    }

    template<trigger::Matcher TMatcher>
    requires(!DataSetLike<T>)
    std::shared_ptr<Poller<DataSet<T>>> getPoller(CaptureMode mode, TMatcher&& matcher, PollerConfig config = {}) {
        std::lock_guard lg(_listener_mutex);
        auto            poller = makePoller<DataSet<T>>();
        addListener(std::make_unique<CaptureListener<std::decay_t<TMatcher>>>(mode, config, std::forward<TMatcher>(matcher), poller, nullptr));
        return poller;
    }

    // callbacks run on the sink's thread with the sink locked: they must not call into this sink, the registry or the scheduler
    template<StreamCallback<T> TCallback>
    void registerCallback(TCallback&& callback) {
        std::function<void(std::span<const T>, std::span<const Tag>)> streamCallback;
        if constexpr (std::invocable<std::decay_t<TCallback>&, std::span<const T>, std::span<const Tag>, const DataSink<T>&>) {
            streamCallback = [this, callback = std::forward<TCallback>(callback)](std::span<const T> data, std::span<const Tag> tags) mutable { callback(data, tags, *this); };
        } else if constexpr (std::invocable<std::decay_t<TCallback>&, std::span<const T>, std::span<const Tag>>) {
            streamCallback = std::forward<TCallback>(callback);
        } else {
            streamCallback = [callback = std::forward<TCallback>(callback)](std::span<const T> data, std::span<const Tag>) mutable { callback(data); };
        }
        std::lock_guard lg(_listener_mutex);
        addListener(std::make_unique<StreamingListener>(std::move(streamCallback)));
    }

    template<trigger::Matcher TMatcher, DataSetCallback<T> TCallback>
    requires(!DataSetLike<T>)
    void registerCallback(CaptureMode mode, TMatcher&& matcher, PollerConfig config, TCallback&& callback) {
        std::lock_guard lg(_listener_mutex);
        addListener(std::make_unique<CaptureListener<std::decay_t<TMatcher>>>(mode, config, std::forward<TMatcher>(matcher), nullptr, std::forward<TCallback>(callback)));
    }

    void updateHistorySize() {
        const std::size_t nRequired = std::ranges::fold_left(_listeners | std::views::transform([](const auto& listener) { return listener->nHistorySamples(); }), 0UZ, std::ranges::max);
        if (nRequired == 0UZ) {
            _history.reset();
        } else if (!_history) {
            _history.emplace(nRequired);
        } else if (_history->capacity() != nRequired) {
            _history->resize(nRequired);
        }
    }

    template<typename U>
    std::shared_ptr<Poller<U>> makePoller() {
        auto poller      = std::make_shared<Poller<U>>(this->progress);
        poller->finished = _stopped;
        return poller;
    }

    void addListener(std::unique_ptr<AbstractListener>&& l) {
        updateMetadata(*l);
        _listeners.push_back(std::move(l));
        updateHistorySize();
    }

    void updateMetadata(AbstractListener& listener) {
        if constexpr (!DataSetLike<T>) {
            listener.setMetadata(detail::Metadata{sample_rate, signal_name, signal_quantity, signal_unit, signal_min, signal_max});
        }
    }

    struct AbstractListener {
        virtual ~AbstractListener() = default;

        [[nodiscard]] virtual bool        isExpired() const                                              = 0;
        [[nodiscard]] virtual std::size_t nAcceptable(std::size_t nAvailable, std::span<const Tag> tags) = 0;
        [[nodiscard]] virtual std::size_t nHistorySamples() const { return 0UZ; }

        virtual void setMetadata(detail::Metadata) = 0;

        virtual void process(std::span<const T> history, std::span<const T> data, std::span<const Tag> tags) = 0;
        virtual void stop()                                                                                  = 0;
    };

    struct StreamingListener : AbstractListener {
        std::weak_ptr<Poller<T>>                                      _poller;
        std::function<void(std::span<const T>, std::span<const Tag>)> _callback;
        OverflowPolicy                                                _overflowPolicy    = OverflowPolicy::Backpressure;
        std::size_t                                                   _nPublishedSamples = 0;
        std::optional<detail::Metadata>                               _pendingMetadata;

        StreamingListener(std::shared_ptr<Poller<T>> poller, OverflowPolicy overflowPolicy) : _poller(std::move(poller)), _overflowPolicy(overflowPolicy) {}

        explicit StreamingListener(std::function<void(std::span<const T>, std::span<const Tag>)> callback) : _callback(std::move(callback)) {}

        [[nodiscard]] bool isExpired() const override { return !_callback && _poller.expired(); }

        [[nodiscard]] std::size_t nAcceptable(std::size_t nAvailable, std::span<const Tag> tags) override {
            const auto poller = _poller.lock();
            if (!poller || _overflowPolicy == OverflowPolicy::Drop) {
                return nAvailable;
            }
            std::size_t nTagSlots = poller->tagWriter.available();
            if (_pendingMetadata.has_value()) {
                if (nTagSlots == 0UZ) {
                    return 0UZ;
                }
                --nTagSlots;
            }
            const std::size_t nSamples = std::min(nAvailable, poller->writer.available());
            return tags.size() > nTagSlots ? std::min(nSamples, tags[nTagSlots].index) : nSamples;
        }

        void setMetadata(detail::Metadata metadata) override { _pendingMetadata = std::move(metadata); }

        void process(std::span<const T>, std::span<const T> data, std::span<const Tag> tags) override {
            if (_callback) {
                const property_map metadataMap = _pendingMetadata ? _pendingMetadata->toTagMap() : property_map{};
                std::vector<Tag>   allTags;
                if (_pendingMetadata.has_value()) {
                    allTags.emplace_back(0UZ, metadataMap);
                    _pendingMetadata.reset();
                }
                allTags.insert(allTags.end(), tags.begin(), tags.end());
                _callback(data, allTags);
                return;
            }

            const auto poller = _poller.lock();
            if (!poller) {
                return;
            }

            const std::size_t nSamplesToPublish = std::min(data.size(), poller->writer.available());
            const std::size_t nTagSlots         = nSamplesToPublish > 0UZ ? poller->tagWriter.available() : 0UZ;
            const bool        publishMetadata   = _pendingMetadata.has_value() && nTagSlots > 0UZ;
            const std::size_t nTagsToPublish    = std::min(tags.size(), publishMetadata ? nTagSlots - 1UZ : nTagSlots);

            if (nSamplesToPublish > 0UZ) {
                const std::size_t nTagEntries = publishMetadata ? nTagsToPublish + 1UZ : nTagsToPublish;
                if (nTagEntries > 0UZ) {
                    WriterSpanLike auto outTags = poller->tagWriter.reserve(nTagEntries);
                    std::size_t         counter = 0UZ;
                    if (publishMetadata) {
                        outTags[counter++] = {_nPublishedSamples, _pendingMetadata->toTagMap()};
                        _pendingMetadata.reset();
                    }
                    for (const Tag& tag : tags | std::views::take(nTagsToPublish)) {
                        outTags[counter++] = {_nPublishedSamples + tag.index, property_map(tag.map)};
                    }
                    outTags.publish(nTagEntries);
                }
                WriterSpanLike auto outSamples = poller->writer.reserve(nSamplesToPublish);
                std::ranges::copy(data | std::views::take(nSamplesToPublish), outSamples.begin());
                outSamples.publish(nSamplesToPublish);
            }
            poller->droppedSampleCount += data.size() - nSamplesToPublish;
            poller->droppedTagCount += tags.size() - nTagsToPublish;
            _nPublishedSamples += nSamplesToPublish;
        }

        void stop() override {
            if (const auto poller = _poller.lock()) {
                poller->finished = true;
            }
        }
    };

    struct TriggerCapture {
        DataSet<T>  dataset;
        std::size_t nRemainingPostSamples = 0;
    };

    struct Snapshot {
        property_map tagMap;
        std::size_t  triggerIndex      = 0;
        std::size_t  nRemainingSamples = 0;
    };

    template<trigger::Matcher TMatcher>
    struct CaptureListener : AbstractListener {
        CaptureMode                       _mode;
        PollerConfig                      _config;
        TMatcher                          _matcher;
        std::weak_ptr<Poller<DataSet<T>>> _poller;
        std::function<void(DataSet<T>)>   _callback;
        gr::property_map                  _matcherState;
        detail::Metadata                  _metadata{};
        std::deque<TriggerCapture>        _triggerCaptures;
        std::optional<DataSet<T>>         _window;
        std::deque<Snapshot>              _snapshots; // ascending nRemainingSamples: after the sample rate drops, a later trigger is due first

        template<trigger::Matcher TMatcherFW>
        CaptureListener(CaptureMode mode, const PollerConfig& config, TMatcherFW&& matcher, std::shared_ptr<Poller<DataSet<T>>> poller, std::function<void(DataSet<T>)> callback) : _mode(mode), _config(config), _matcher(std::forward<TMatcherFW>(matcher)), _poller(std::move(poller)), _callback(std::move(callback)) {}

        [[nodiscard]] bool isExpired() const override { return !_callback && _poller.expired(); }

        [[nodiscard]] std::size_t nHistorySamples() const override { return _mode == CaptureMode::Trigger ? _config.preSamples : 0UZ; }

        void setMetadata(detail::Metadata metadata) override { _metadata = std::move(metadata); }

        [[nodiscard]] std::size_t nDelaySamples() const { return static_cast<std::size_t>(std::round(std::chrono::duration_cast<std::chrono::duration<float>>(_config.delay).count() * _metadata.sampleRate)); }

        [[nodiscard]] std::size_t nAcceptable(std::size_t nAvailable, std::span<const Tag> tags) override {
            const auto poller = _poller.lock();
            if (!poller || _config.overflowPolicy == OverflowPolicy::Drop) {
                return nAvailable;
            }
            const std::size_t nMaxDataSets = _triggerCaptures.size() + (_window ? 1UZ : 0UZ) + _snapshots.size() + tags.size();
            const std::size_t nFree        = poller->writer.available();
            if (nFree >= nMaxDataSets) {
                return nAvailable;
            }
            return nFree == poller->buffer.size() ? 1UZ : 0UZ; // more open captures than the ring holds: advance sample by sample so none is dropped
        }

        void process(std::span<const T> history, std::span<const T> inData, std::span<const Tag> tags) override {
            if (_mode == CaptureMode::Trigger) {
                for (const Tag& tag : tags) {
                    if (!tag.map.empty() && _matcher("", tag, _matcherState) == trigger::MatchResult::Matching) {
                        const std::size_t minPreSamples = std::min(_config.preSamples, history.size());
                        DataSet<T>        dataset       = detail::createDataset<T>(_metadata, minPreSamples + _config.postSamples);
                        detail::appendSamples(dataset, history.first(minPreSamples) | std::views::reverse);
                        dataset.timing_events[0UZ].emplace_back(static_cast<std::ptrdiff_t>(minPreSamples), tag.map);
                        _triggerCaptures.push_back({.dataset = std::move(dataset), .nRemainingPostSamples = _config.postSamples});
                    }
                }

                auto capture = _triggerCaptures.begin();
                while (capture != _triggerCaptures.end()) {
                    const std::size_t nPostSamples = std::min(capture->nRemainingPostSamples, inData.size());
                    detail::appendSamples(capture->dataset, inData.first(nPostSamples));
                    capture->nRemainingPostSamples -= nPostSamples;

                    if (capture->nRemainingPostSamples == 0UZ) {
                        publish(std::move(capture->dataset));
                        capture = _triggerCaptures.erase(capture);
                    } else {
                        ++capture;
                    }
                }
            } else if (_mode == CaptureMode::Multiplexed) {
                for (const Tag& tag : tags) {
                    const auto matchResult = _matcher("", tag, _matcherState);
                    if (matchResult == trigger::MatchResult::NotMatching || matchResult == trigger::MatchResult::Matching) {
                        if (_window) {
                            _window->timing_events[0UZ].emplace_back(static_cast<std::ptrdiff_t>(_window->signalValues(0UZ).size()), tag.map);
                            publish(std::move(*_window));
                            _window.reset();
                        }
                    }
                    if (matchResult == trigger::MatchResult::Matching) {
                        _window = detail::createDataset<T>(_metadata, _config.maximumWindowSize);
                        _window->timing_events[0UZ].emplace_back(0, tag.map);
                    }
                }
                if (_window) {
                    const std::size_t maxSize = _config.maximumWindowSize == 0UZ ? std::numeric_limits<std::size_t>::max() : _config.maximumWindowSize;
                    detail::appendSamples(*_window, inData.first(std::min(inData.size(), maxSize - _window->signal_values.size())));
                    if (_window->signal_values.size() == maxSize) {
                        publish(std::move(*_window));
                        _window.reset();
                    }
                }
            } else {
                const std::size_t nDelay = nDelaySamples();
                for (const Tag& tag : tags) {
                    if (!tag.map.empty() && _matcher("", tag, _matcherState) == trigger::MatchResult::Matching) {
                        _snapshots.insert(std::ranges::upper_bound(_snapshots, nDelay, {}, &Snapshot::nRemainingSamples), Snapshot{tag.map, nDelay, nDelay});
                    }
                }

                auto it = _snapshots.begin();
                while (it != _snapshots.end()) {
                    if (it->nRemainingSamples >= inData.size()) {
                        it->nRemainingSamples -= inData.size();
                        ++it;
                    } else {
                        DataSet<T> dataset = detail::createDataset<T>(_metadata, 1UZ);
                        dataset.timing_events[0UZ].emplace_back(-static_cast<std::ptrdiff_t>(it->triggerIndex), std::move(it->tagMap));
                        detail::appendSamples(dataset, inData.subspan(it->nRemainingSamples, 1UZ));
                        publish(std::move(dataset));
                        it = _snapshots.erase(it);
                    }
                }
            }
        }

        void publish(DataSet<T>&& dataSet) {
            if (_callback) {
                _callback(std::move(dataSet));
                return;
            }
            const auto poller = _poller.lock();
            if (!poller) {
                return;
            }
            if (poller->writer.available() == 0UZ) {
                ++poller->droppedSampleCount;
                return;
            }
            WriterSpanLike auto out = poller->writer.reserve(1UZ);
            out[0UZ]                = std::move(dataSet);
            out.publish(1UZ);
        }

        void stop() override {
            for (TriggerCapture& capture : _triggerCaptures) {
                publish(std::move(capture.dataset));
            }
            if (_window) {
                publish(std::move(*_window));
            }
            _triggerCaptures.clear();
            _window.reset();
            _snapshots.clear();
            if (const auto poller = _poller.lock()) {
                poller->finished = true;
            }
        }
    };
};

} // namespace gr::basic

#endif
