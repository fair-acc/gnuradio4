#ifndef GNURADIO_SCHEDULER_HPP
#define GNURADIO_SCHEDULER_HPP

#include <bit>
#include <chrono>
#include <mutex>
#include <queue>
#include <set>

#include <thread>
#include <utility>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Graph_yaml_importer.hpp>
#include <gnuradio-4.0/LifeCycle.hpp>
#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Profiler.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#include <emscripten/threading.h>
#endif

// Under Windows windows.h defines ERROR as 0.  This messes the ERROR function work::status::ERROR.
#ifdef _WIN32
#ifdef ERROR
#undef ERROR
#endif // #ifdef ERROR
#endif // #ifdef _WIN32

template<typename T>
inline void waitUntilChanged(gr::Sequence& sequence, T oldValue, [[maybe_unused]] unsigned int delay_ms = 1U) {
    if (sequence.value() != oldValue) {
        return;
    }
    do {
#ifdef __EMSCRIPTEN__
#ifdef __EMSCRIPTEN_PTHREADS__
        sequence.wait(oldValue); // only works in worker threads with PThreads
#else
        std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms)); // fallback spin sleep
#endif
#else
        sequence.wait(oldValue); // C++ native
#endif
    } while (sequence.value() == oldValue);
}

namespace gr::scheduler {
using namespace gr::message;

namespace property {

inline static const char* kEmplaceBlock = "EmplaceBlock";
inline static const char* kRemoveBlock  = "RemoveBlock";
inline static const char* kReplaceBlock = "ReplaceBlock";
inline static const char* kEmplaceEdge  = "EmplaceEdge";
inline static const char* kRemoveEdge   = "RemoveEdge";

inline static const char* kBlockEmplaced = "BlockEmplaced";
inline static const char* kBlockRemoved  = "BlockRemoved";
inline static const char* kBlockReplaced = "BlockReplaced";
inline static const char* kEdgeEmplaced  = "EdgeEmplaced";
inline static const char* kEdgeRemoved   = "EdgeRemoved";

inline static const char* kGraphGRC           = "GraphGRC";
inline static const char* kSchedulerInspect   = "SchedulerInspect";
inline static const char* kSchedulerInspected = "SchedulerInspected";
} // namespace property

enum class ExecutionPolicy {
    singleThreaded,        ///
    multiThreaded,         ///
    singleThreadedBlocking /// blocks with a time-out if none of the blocks in the graph made progress (N.B. a CPU/battery power-saving measures)
};

using JobLists = std::vector<std::vector<std::shared_ptr<BlockModel>>>;

template<typename Derived, ExecutionPolicy execution = ExecutionPolicy::singleThreaded, profiling::ProfilerLike TProfiler = profiling::null::Profiler>
struct SchedulerBase : Block<Derived> {
    friend class lifecycle::StateMachine<Derived>;
    using TaskExecutor = gr::thread_pool::TaskExecutor;
    using enum block::Category;

private:
    static consteval void _forbid_reserved_overrides() {
        using Base = SchedulerBase<Derived, execution, TProfiler>;
        // Lifecycle callback functions init/start/stop/pause/resume/reset need to remain reserved to SchedulerBase<Derived, ...>.
        // Do NOT re-implement them in the Derived custom user-defined scheduler.
        static_assert(std::same_as<decltype(&Derived::init), decltype(&Base::init)>, "Derived defines 'init()' (reserved). Use 'customInit()' instead.");
        static_assert(std::same_as<decltype(&Derived::start), decltype(&Base::start)>, "Derived defines 'start()' (reserved). Use 'customStart()' instead.");
        static_assert(std::same_as<decltype(&Derived::stop), decltype(&Base::stop)>, "Derived defines 'stop()' (reserved). Use 'customStop()' instead.");
        static_assert(std::same_as<decltype(&Derived::pause), decltype(&Base::pause)>, "Derived defines 'pause()' (reserved). Use 'customPause()' instead.");
        static_assert(std::same_as<decltype(&Derived::resume), decltype(&Base::resume)>, "Derived defines 'resume()' (reserved). Use 'customResume()' instead.");
        static_assert(std::same_as<decltype(&Derived::reset), decltype(&Base::reset)>, "Derived defines 'reset()' (reserved). Use 'customReset()' instead.");
    }

    gr::Graph* findTargetSubGraph(const gr::property_map& data) {
        auto it = data.find("_targetGraph");
        if (it == data.end()) {
            return std::addressof(*_graph);
        } else if (it->second.value_or(std::string()) == _graph->unique_name || it->second.value_or(std::string()) == this->unique_name) {
            return std::addressof(*_graph);
        } else {
            const auto targetGraphName = it->second.value_or(std::string_view{});
            if (targetGraphName.empty()) {
                return nullptr;
            }
            auto result = graph::findBlock(*_graph, std::string_view(targetGraphName));
            if (!result) {
                return nullptr;
            }

            if (result.value()->typeName() != "gr::Graph") {
                return nullptr;
            }

            return static_cast<gr::Graph*>(result.value()->raw());
        }
    }

protected:
    using ProfileHandle = decltype(std::declval<TProfiler&>().forThisThread());

    std::atomic_bool              _valid{true};
    std::atomic<std::size_t>      _nWatchdogsRunning{0};
    meta::indirect<gr::Graph>     _graph{};
    TProfiler                     _profiler{};
    ProfileHandle                 _profilerHandler{_profiler.forThisThread()};
    std::shared_ptr<TaskExecutor> _pool{gr::thread_pool::Manager::instance().defaultCpuPool()};
    std::shared_ptr<gr::Sequence> _nRunningJobs = std::make_shared<gr::Sequence>();
    std::recursive_mutex          _executionOrderMutex; // only used when modifying and copying the graph->local job list
    std::shared_ptr<JobLists>     _executionOrder = std::make_shared<JobLists>();

    std::mutex                               _zombieBlocksMutex;
    std::vector<std::shared_ptr<BlockModel>> _zombieBlocks;

    // for blocks that were added while scheduler was running. They need to be adopted by a thread
    std::mutex _adoptionBlocksMutex;
    // fixed-sized vector indexed by runnerId. Cheaper than a map.
    std::vector<std::vector<std::shared_ptr<BlockModel>>> _adoptionBlocks;

    MsgPortOutForChildren    _toChildMessagePort;
    MsgPortInFromChildren    _fromChildMessagePort;
    std::vector<gr::Message> _pendingMessagesToChildren;
    bool                     _messagePortsConnected = false;

    std::atomic_flag _processingScheduledMessages;

    void rebuildProfiler(const profiling::Options& opt) {
        std::destroy_at(std::addressof(_profiler));
        std::construct_at(std::addressof(_profiler), opt);
        _profilerHandler = _profiler.forThisThread();
    }

    void registerPropertyCallbacks() noexcept {
        _forbid_reserved_overrides();
        using PropertyCallback                            = BlockBase::PropertyCallback;
        auto& callbacks                                   = this->propertyCallbacks;
        callbacks[scheduler::property::kEmplaceBlock]     = static_cast<PropertyCallback>(&SchedulerBase::propertyCallbackEmplaceBlock);
        callbacks[scheduler::property::kRemoveBlock]      = static_cast<PropertyCallback>(&SchedulerBase::propertyCallbackRemoveBlock);
        callbacks[scheduler::property::kRemoveEdge]       = static_cast<PropertyCallback>(&SchedulerBase::propertyCallbackRemoveEdge);
        callbacks[scheduler::property::kEmplaceEdge]      = static_cast<PropertyCallback>(&SchedulerBase::propertyCallbackEmplaceEdge);
        callbacks[scheduler::property::kReplaceBlock]     = static_cast<PropertyCallback>(&SchedulerBase::propertyCallbackReplaceBlock);
        callbacks[scheduler::property::kGraphGRC]         = static_cast<PropertyCallback>(&SchedulerBase::propertyCallbackGraphGRC);
        callbacks[scheduler::property::kSchedulerInspect] = static_cast<PropertyCallback>(&SchedulerBase::propertyCallbackSchedulerInspect);
        callbacks[graph::property::kInspectBlock]         = static_cast<PropertyCallback>(&SchedulerBase::propertyCallbackInspectBlock);
        this->settings().updateActiveParameters();
    }

public:
    using base_t = Block<Derived>;

    Annotated<gr::Size_t, "timeout", Unit<"ms">, Doc<"sleep timeout to wait if graph has made no progress ">>                  timeout_ms                      = 100U;
    Annotated<gr::Size_t, "watchdog_timeout", Unit<"ms">, Doc<"sleep timeout for watchdog">>                                   watchdog_timeout                = 1000U;
    Annotated<gr::Size_t, "timeout_inactivity_count", Doc<"number of inactive cycles w/o progress before sleep is triggered">> timeout_inactivity_count        = 5U;
    Annotated<gr::Size_t, "process_stream_to_message_ratio", Doc<"number of stream to msg processing">>                        process_stream_to_message_ratio = 16U;
    Annotated<std::string, "pool name", Doc<"default pool name">>                                                              poolName                        = std::string(gr::thread_pool::kDefaultCpuPoolId);
    Annotated<std::size_t, "max_work_items", Doc<"number of work items per work scheduling interval (controls latency)">>      max_work_items                  = std::numeric_limits<std::size_t>::max(); // TODO: check whether we can keep this std::size_t or more consistently to gr::Size_t
    Annotated<property_map, "sched_settings", Doc<"scheduler implementation specific settings">>                               sched_settings{};

    GR_MAKE_REFLECTABLE(SchedulerBase, timeout_ms, timeout_inactivity_count, process_stream_to_message_ratio, max_work_items, sched_settings);

    constexpr static block::Category blockCategory = block::Category::ScheduledBlockGroup;

    [[nodiscard]] static constexpr auto executionPolicy() { return execution; }

    SchedulerBase() : base_t(gr::property_map()) { registerPropertyCallbacks(); }

    SchedulerBase(std::initializer_list<std::pair<const std::pmr::string, pmt::Value>> initParameter) noexcept(false) : base_t(initParameter) {
        registerPropertyCallbacks();
        std::ignore = this->settings().set(initParameter);
        std::ignore = this->settings().activateContext();
        std::ignore = this->settings().applyStagedParameters();
    }

    explicit SchedulerBase(property_map initParameters) noexcept(false) : base_t(initParameters) {
        registerPropertyCallbacks();
        std::ignore = this->settings().set(initParameters);
        std::ignore = this->settings().activateContext();
        std::ignore = this->settings().applyStagedParameters();
    }

    ~SchedulerBase() {
        if (this->state() == lifecycle::RUNNING) {
            if (auto e = this->changeStateTo(lifecycle::REQUESTED_STOP); !e) {
                std::println(std::cerr, "Failed to stop execution at destruction of scheduler: {} ({})", e.error().message, e.error().srcLoc());
                std::abort();
            }
        }
        waitDone();

        _valid.store(false, std::memory_order_release); // Mark as invalid

        // the watchdog dereferences SchedulerBase, wait until it finishes
        while (_nWatchdogsRunning.load() != 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        _executionOrder.reset(); // force earlier crashes if this is accessed after destruction (e.g. from thread that was kept running)
    }

    [[nodiscard]] std::expected<meta::indirect<Graph>, Error> exchange(meta::indirect<Graph>&& newGraph, std::string_view defaultPoolName = gr::thread_pool::kDefaultCpuPoolId, const profiling::Options& option = {}) {
        using enum lifecycle::State;
        const auto oldState = this->state();
        if (lifecycle::isActive(oldState)) { // need to stop running scheduler
            if (auto result = this->changeStateTo(REQUESTED_STOP); !result) {
                return std::unexpected(result.error());
            }
            waitDone(); // wait for all jobs to complete

            if (auto result = this->changeStateTo(STOPPED); !result) {
                return std::unexpected(result.error());
            }
        }

        if (this->state() == ERROR || this->state() == STOPPED) {
            reset(); // reset internal states
        }

        auto oldGraph = std::exchange(_graph, std::move(newGraph));

        if ((option != profiling::Options{})) { // need to update profiler
            rebuildProfiler(option);
        }

        if (_pool->name() != defaultPoolName) { // need to update thread pool
            _pool = gr::thread_pool::Manager::instance().get(defaultPoolName);
        }

        // restore the original lifecycle state
        if (lifecycle::isActive(oldState)) {
            if (auto result = this->changeStateTo(INITIALISED); !result) { // Need to go to INITIALISED first
                return std::unexpected(result.error());
            }
            if (auto result = this->changeStateTo(RUNNING); !result) {
                return std::unexpected(result.error());
            }

            if (oldState == REQUESTED_PAUSE) {
                if (auto result = this->changeStateTo(REQUESTED_PAUSE); !result) {
                    return std::unexpected(result.error());
                }
            } else if (oldState == PAUSED) {
                if (auto result = this->changeStateTo(REQUESTED_PAUSE); !result) {
                    return std::unexpected(result.error());
                }
                if (auto result = this->changeStateTo(PAUSED); !result) {
                    return std::unexpected(result.error());
                }
            }
        }
        return oldGraph;
    }

    [[nodiscard]] const gr::Graph& graph() const noexcept { return *_graph; }
    [[nodiscard]] gr::Graph&       graph() noexcept { return *_graph; }

    [[nodiscard]] const TProfiler& profiler() const noexcept { return _profiler; }

    [[nodiscard]] bool isProcessing() const
    requires(executionPolicy() == ExecutionPolicy::multiThreaded)
    {
        return _nRunningJobs->value() > 0UZ;
    }

    void stateChanged(lifecycle::State newState) { this->notifyListeners(block::property::kLifeCycleState, {{"state", std::string(magic_enum::enum_name(newState))}}); }

    [[nodiscard]] std::span<std::shared_ptr<BlockModel>>       blocks() noexcept { return _graph->blocks(); }
    [[nodiscard]] std::span<const std::shared_ptr<BlockModel>> blocks() const noexcept { return _graph->blocks(); }
    [[nodiscard]] std::span<Edge>                              edges() noexcept { return _graph->edges(); }
    [[nodiscard]] std::span<const Edge>                        edges() const noexcept { return _graph->edges(); }

    void connectBlockMessagePorts() {
        const auto available = _graph->msgIn.streamReader().available();
        if (available != 0UZ) {
            ReaderSpanLike auto msgInSpan = _graph->msgIn.streamReader().get<SpanReleasePolicy::ProcessAll>(available);
            _pendingMessagesToChildren.insert(_pendingMessagesToChildren.end(), msgInSpan.begin(), msgInSpan.end());
        }

        auto toSchedulerBuffer = _fromChildMessagePort.buffer();
        std::ignore            = _toChildMessagePort.connect(_graph->msgIn);
        _graph->msgOut.setBuffer(toSchedulerBuffer.streamBuffer, toSchedulerBuffer.tagBuffer);

        graph::forEachBlock<TransparentBlockGroup>(*_graph, [this, &toSchedulerBuffer](auto& block) {
            if (ConnectionResult::SUCCESS != _toChildMessagePort.connect(*block->msgIn)) {
                this->emitErrorMessage("connectBlockMessagePorts()", std::format("Failed to connect scheduler input message port to child '{}'", block->uniqueName()));
            }

            block->msgOut->setBuffer(toSchedulerBuffer.streamBuffer, toSchedulerBuffer.tagBuffer);
        });

        // Forward any messages to children that were received before the scheduler was initialised
        _messagePortsConnected = true;

        WriterSpanLike auto msgSpan = _toChildMessagePort.streamWriter().reserve<SpanReleasePolicy::ProcessAll>(_pendingMessagesToChildren.size());
        std::ranges::move(_pendingMessagesToChildren, msgSpan.begin());
        _pendingMessagesToChildren.clear();
    }

    void processMessages(gr::MsgPortInBuiltin& port, std::span<const gr::Message> messages) {
        base_t::processMessages(port, messages); // filters messages and calls own property handler

        for (const gr::Message& msg : messages) {
            if (msg.serviceName != this->unique_name && msg.serviceName != this->name && msg.endpoint != block::property::kLifeCycleState) {
                // only forward wildcard, non-scheduler messages, and non-lifecycle messages (N.B. the latter is exclusively handled by the scheduler)
                if (_messagePortsConnected) {
                    WriterSpanLike auto msgSpan = _toChildMessagePort.streamWriter().reserve<SpanReleasePolicy::ProcessAll>(1UZ);
                    msgSpan[0]                  = msg;
                } else {
                    // if not yet connected, keep messages to children in cache and forward when connecting
                    _pendingMessagesToChildren.push_back(msg);
                }
            }
        }
    }

    void processScheduledMessages() {
        if (std::atomic_flag_test_and_set_explicit(&_processingScheduledMessages, std::memory_order_acquire)) {
            return;
        }

        on_scope_exit _ = [&] { std::atomic_flag_clear_explicit(&_processingScheduledMessages, std::memory_order_release); };

        base_t::processScheduledMessages(); // filters messages and calls own property handler

        // Process messages in the graph
        _graph->processScheduledMessages();
        if (_nRunningJobs->value() == 0UZ) {
            graph::forEachBlock<TransparentBlockGroup>(*_graph, [](auto& block) { block->processScheduledMessages(); });
        }

        ReaderSpanLike auto messagesFromChildren = _fromChildMessagePort.streamReader().get();
        if (messagesFromChildren.size() == 0) {
            return;
        }

        if (this->msgOut.buffer().streamBuffer.n_readers() == 0) {
            // nobody is listening on messages -> convert errors to exceptions
            for (const auto& msg : messagesFromChildren) {
                if (!msg.data.has_value()) {
                    throw gr::exception(std::format("scheduler {}: throwing ignored exception {:t}", this->name, msg.data.error()));
                }
            }
            return;
        }

        {
            WriterSpanLike auto msgSpan = this->msgOut.streamWriter().template reserve<SpanReleasePolicy::ProcessAll>(messagesFromChildren.size());
            std::ranges::copy(messagesFromChildren, msgSpan.begin());
            msgSpan.publish(messagesFromChildren.size());
        } // to force publish
        if (!messagesFromChildren.consume(messagesFromChildren.size())) {
            this->emitErrorMessage("process child return messages", "Failed to consume messages from child message port");
        }
    }

    std::expected<void, Error> runAndWait() {
        using enum lifecycle::State;
        [[maybe_unused]] const auto pe = this->_profilerHandler->startCompleteEvent("scheduler_base.runAndWait");
        processScheduledMessages(); // make sure initial subscriptions are processed
        if (this->state() == STOPPED || this->state() == ERROR) {
            if (auto e = this->changeStateTo(INITIALISED); !e) {
                this->emitErrorMessage("runAndWait() -> LifecycleState", e.error());
                return std::unexpected(e.error());
            }
        }
        if (this->state() == IDLE) {
            if (auto e = this->changeStateTo(INITIALISED); !e) {
                this->emitErrorMessage("runAndWait() -> LifecycleState", e.error());
                return std::unexpected(e.error());
            }
        }
        if (auto e = this->changeStateTo(RUNNING); !e) {
            this->emitErrorMessage("runAndWait() -> LifecycleState", e.error());
            return std::unexpected(e.error());
        }

        // N.B. the transition to lifecycle::State::RUNNING will for the ExecutionPolicy:
        // * singleThreaded[Blocking] naturally block in the calling thread
        // * multiThreaded[Blocking] spawn two worker and block on 'waitDone()'
        waitDone();
        processScheduledMessages();

        if (this->state() == RUNNING) {
            if (auto e = this->changeStateTo(REQUESTED_STOP); !e) {
                this->emitErrorMessage("runAndWait() -> LifecycleState", e.error());
                return std::unexpected(e.error());
            }
        }
        if (this->state() == REQUESTED_STOP) {
            if (auto e = this->changeStateTo(STOPPED); !e) {
                this->emitErrorMessage("runAndWait() -> LifecycleState", e.error());
            }
        }
        processScheduledMessages();
        return {};
    }

    void waitDone() {
        [[maybe_unused]] const auto pe = _profilerHandler->startCompleteEvent("scheduler_base.waitDone");
        while (_nRunningJobs->value() > 0UZ) {
            std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
        }
    }

    [[nodiscard]] std::shared_ptr<JobLists> jobs() const noexcept { return _executionOrder; }

protected:
    void disconnectAllEdges() {
        _graph->disconnectAllEdges();
        graph::forEachBlock<TransparentBlockGroup>(*_graph, [&](auto& block) {
            if (block->blockCategory() == TransparentBlockGroup) {
                auto* graph = static_cast<GraphWrapper<gr::Graph>*>(block.get());
                graph->blockRef().disconnectAllEdges();
            }
        });
    }

    bool connectPendingEdges() {
        auto primeFeedbackPorts = [&](const gr::Graph& graph) {
            std::vector<graph::FeedbackLoop> feedbackLoops = gr::graph::detectFeedbackLoops(graph);
            for (auto& loop : feedbackLoops) {
                if (std::expected<std::size_t, Error> nPrimeSamples = gr::graph::calculateLoopPrimingSize(loop); nPrimeSamples) {
                    if (auto ret = gr::graph::primeLoop(loop, nPrimeSamples.value()); !ret) {
                        this->emitErrorMessage("connectPendingEdges()", std::format("failed to prime feedback loop: {}\nloop: {}", ret.error(), loop.edges));
                    }
                } else {
                    this->emitErrorMessage("connectPendingEdges()", std::format("failed to prime feedback loop: {}\nloop: {}", nPrimeSamples.error(), loop.edges));
                }
            }
        };

        bool result = _graph->connectPendingEdges();
        primeFeedbackPorts(gr::graph::flatten(*_graph)); // need to flatten graph due to potential loops from within the subgraph to blocks in the parents.
        graph::forEachBlock<TransparentBlockGroup>(*_graph, [&](auto& block) {
            if (block->blockCategory() == TransparentBlockGroup) {
                auto* graph = static_cast<GraphWrapper<gr::Graph>*>(block.get());
                result      = result && graph->blockRef().connectPendingEdges();
                primeFeedbackPorts(gr::graph::flatten(graph->blockRef()));
            }
        });
        return result;
    }

    work::Result traverseBlockListOnce(const std::vector<std::shared_ptr<BlockModel>>& blocks) const {
        const std::size_t requestedWorkAllBlocks = max_work_items;
        std::size_t       performedWorkAllBlocks = 0UZ;
        bool              unfinishedBlocksExist  = false; // i.e. at least one block returned OK, INSUFFICIENT_INPUT_ITEMS, or INSUFFICIENT_OUTPU_ITEMS
        for (auto& currentBlock : blocks) {
            const auto [requested_work, performed_work, status] = currentBlock->work(requestedWorkAllBlocks);
            performedWorkAllBlocks += performed_work;

            if (status == work::Status::ERROR) {
                return {requested_work, performedWorkAllBlocks, work::Status::ERROR};
            } else if (status != work::Status::DONE) {
                unfinishedBlocksExist = true;
            }
        }
#ifdef __EMSCRIPTEN__
        std::this_thread::sleep_for(std::chrono::microseconds(10u)); // workaround for incomplete std::atomic implementation (at least it seems for nodejs)
#endif
        return {max_work_items, performedWorkAllBlocks, unfinishedBlocksExist ? work::Status::OK : work::Status::DONE};
    }

    void init() {
        [[maybe_unused]] const auto pe = _profilerHandler->startCompleteEvent("scheduler_base.init");
        base_t::processScheduledMessages(); // make sure initial subscriptions are processed
        connectBlockMessagePorts();

        if constexpr (requires(Derived& d) { d.customInit(); }) {
            static_cast<Derived*>(this)->customInit();
        }
    }

    void reset() {
        graph::forEachBlock<TransparentBlockGroup>(*_graph, [this](auto& block) { this->emitErrorMessageIfAny("reset() -> LifecycleState", block->changeStateTo(lifecycle::INITIALISED)); });
        disconnectAllEdges();

        if constexpr (requires(Derived& d) { d.customReset(); }) {
            static_cast<Derived*>(this)->customReset();
        }
    }

    void start() {
        using enum gr::lifecycle::State;

        disconnectAllEdges();
        if (auto result = connectPendingEdges(); !result) {
            this->emitErrorMessage("start()", "Failed to connect blocks in graph");
        }
        if (this->state() == IDLE) {
            if (auto result = this->changeStateTo(INITIALISED); !result) { // Need to go to INITIALISED first
                this->emitErrorMessage("start()", result.error());
            }
        }

        std::lock_guard lock(_executionOrderMutex);

        graph::forEachBlock<TransparentBlockGroup>(*_graph, [this](auto& block) { //
            if (block->blockCategory() == ScheduledBlockGroup) {
                // We don't simply move to RUNNING, as schedulers block. This code path
                // uses a separate thread.
                auto* schedulerModel = dynamic_cast<SchedulerModel*>(block.get());
                if (schedulerModel) {
                    schedulerModel->start();
                } else {
                    throw gr::exception(std::format("ScheduledBlockGroup is not a SchedulerModel {}", block->uniqueName()));
                }
            } else {
                this->emitErrorMessageIfAny("LifecycleState -> RUNNING", block->changeStateTo(lifecycle::RUNNING));
            }
        });

        // start watchdog
        auto ioThreadPool = gr::thread_pool::Manager::defaultIoPool();

        // keep outside of the lambda, as ~SchedulerBase() might finish before watchdog even starts
        _nWatchdogsRunning.fetch_add(1, std::memory_order_acq_rel);

        ioThreadPool->execute([this] { this->runWatchDog(watchdog_timeout.value, timeout_inactivity_count.value); });

        assert(_nRunningJobs->value() == 0UZ);
        assert(!_executionOrder->empty());
        if constexpr (executionPolicy() == ExecutionPolicy::singleThreaded || executionPolicy() == ExecutionPolicy::singleThreadedBlocking) {
            static_cast<Derived*>(this)->poolWorker(0UZ, _executionOrder);
        } else { // run on processing thread pool
            [[maybe_unused]] const auto pe           = _profilerHandler->startCompleteEvent("scheduler_base.runOnPool");
            auto                        jobListsCopy = _executionOrder;
            for (std::size_t runnerID = 0UZ; runnerID < _executionOrder->size(); runnerID++) {
                _pool->execute([this, runnerID, jobListsCopy]() { static_cast<Derived*>(this)->poolWorker(runnerID, jobListsCopy); });
            }
            if (!_executionOrder->empty()) {
                _nRunningJobs->wait(0UZ); // waits until at least one pool worker started
            }
        }
        if constexpr (requires(Derived& d) { d.customStart(); }) {
            static_cast<Derived*>(this)->customStart();
        }
    }

    void poolWorker(const std::size_t runnerID, std::shared_ptr<std::vector<std::vector<std::shared_ptr<BlockModel>>>> jobList) noexcept {
        using enum lifecycle::State;
        std::shared_ptr<gr::Sequence> progress     = _graph->_progress; // life-time guaranteed
        std::shared_ptr<gr::Sequence> nRunningJobs = _nRunningJobs;

        nRunningJobs->incrementAndGet();
        nRunningJobs->notify_all();
        gr::thread_pool::thread::setThreadName(std::format("pW{}-{}", runnerID, gr::meta::shorten_type_name(this->unique_name)));

        [[maybe_unused]] auto profiler_handler = _profiler.forThisThread();

        std::vector<std::shared_ptr<BlockModel>> localBlockList;
        {
            assert(jobList->size() > runnerID);
            std::lock_guard                          lock(_executionOrderMutex);
            std::vector<std::shared_ptr<BlockModel>> blocks = jobList->at(runnerID);
            localBlockList.reserve(blocks.size());
            std::ranges::copy(blocks, std::back_inserter(localBlockList));
        }

        [[maybe_unused]] auto currentProgress    = this->_graph->progress().value();
        std::size_t           inactiveCycleCount = 0UZ;
        std::size_t           msgToCount         = 0UZ;
        auto                  activeState        = this->state();
        do {
            [[maybe_unused]] auto pe = profiler_handler->startCompleteEvent("scheduler_base.work");
            if constexpr (executionPolicy() == ExecutionPolicy::singleThreadedBlocking) {
                // optionally tracking progress and block if there is none
                currentProgress = progress->value();
            }

            bool hasMessagesToProcess = msgToCount == 0UZ;
            if (hasMessagesToProcess) {
                if (runnerID == 0UZ || nRunningJobs->value() == 0UZ) {
                    this->processScheduledMessages(); // execute the scheduler- and Graph-specific message handler only once globally
                }

                // Zombies are cleaned per-thread, as we remove from the localBlockList as well.
                // Cleaning zombies has low priority, so uses process_stream_to_message_ratio (a different ratio could be introduced)
                cleanupZombieBlocks(localBlockList);

                adoptBlocks(runnerID, localBlockList);

                std::ranges::for_each(localBlockList, &BlockModel::processScheduledMessages);
                activeState = this->state();
                msgToCount++;
            } else {
                if (std::has_single_bit(process_stream_to_message_ratio.value)) {
                    msgToCount = (msgToCount + 1U) & (process_stream_to_message_ratio.value - 1);
                } else {
                    msgToCount = (msgToCount + 1U) % process_stream_to_message_ratio.value;
                }
            }

            if (activeState == RUNNING) {
                gr::work::Result result = traverseBlockListOnce(localBlockList);
                if (result.status == work::Status::DONE) {
                    break; // nothing happened -> shutdown this worker
                } else if (result.status == work::Status::ERROR) {
                    this->emitErrorMessageIfAny("LifecycleState (ERROR)", this->changeStateTo(ERROR));
                    break;
                }
            } else if (activeState == PAUSED) {
                std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
                msgToCount = 0UZ;
            } else { // other states
                std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
                msgToCount = 0UZ;
            }

            // optionally tracking progress and block if there is none
            if constexpr (executionPolicy() == ExecutionPolicy::singleThreadedBlocking) {
                auto progressAfter = progress->value();
                if (currentProgress == progressAfter) {
                    inactiveCycleCount++;
                } else {
                    inactiveCycleCount = 0UZ;
                }

                currentProgress = progressAfter;
                if (inactiveCycleCount > timeout_inactivity_count) {
                    // allow a scheduler process to wait on progress before retrying (N.B. intended to save CPU/battery power)
                    // N.B. a watchdog will periodically update the progress to check for non-responsive blocks.
                    waitUntilChanged(*progress, currentProgress, timeout_ms);
                    msgToCount = 0UZ;
                }
            }
        } while (lifecycle::isActive(activeState));
        std::ignore = nRunningJobs->subAndGet(1UZ);
        nRunningJobs->notify_all();
    }

    void runWatchDog(std::size_t timeOut_ms, std::size_t timeOut_count) {
        on_scope_exit _ = [this] { _nWatchdogsRunning.fetch_sub(1, std::memory_order_acq_rel); };

        auto thisName = gr::meta::shorten_type_name(this->unique_name);
        gr::thread_pool::thread::setThreadName(std::format("WatchDog-{}", thisName));

        const auto deadline      = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
        const auto checkInterval = std::chrono::milliseconds(std::max(timeout_ms / 10UZ, 1UZ));
        while (_valid.load(std::memory_order_acquire) && _nRunningJobs->value() == 0UZ && std::chrono::steady_clock::now() < deadline && lifecycle::isActive(this->state())) {
            std::this_thread::sleep_for(checkInterval);
        }

        if (!_valid.load(std::memory_order_acquire) || _nRunningJobs->value() == 0UZ || !lifecycle::isActive(this->state())) {
            return; // abort watchdog: scheduler inactive or jobs already finished.
        }

        std::size_t lastProgress = _graph->_progress->value();
        std::size_t nWarnings    = 0;
        do {
            std::this_thread::sleep_for(std::chrono::milliseconds(timeOut_ms));
            // check and increase progress if there hasn't been none.

            std::size_t currentProgress = _graph->_progress->value();
            if ((_nRunningJobs->value() > 0UZ) && (currentProgress == lastProgress)) {
                nWarnings++;
                lastProgress = _graph->_progress->incrementAndGet(); // watchdog triggered manual update
                _graph->_progress->notify_all();
                if (nWarnings >= timeOut_count) {
                    std::println(stderr, "trigger watchdog update {} of {} in {}", nWarnings, timeOut_count, thisName);
                    // log or escalate (e.g., throw, abort, notify external watchdog)
                }
            } else {
                lastProgress = currentProgress;
                nWarnings    = 0UZ;
            }
        } while (_nRunningJobs->value() > 0UZ);
    }

    void stop() {
        using enum lifecycle::State;
        graph::forEachBlock<TransparentBlockGroup>(*_graph, [this](auto& block) {
            if (block->blockCategory() == ScheduledBlockGroup) {
                auto* schedulerModel = dynamic_cast<SchedulerModel*>(block.get());
                if (schedulerModel) {
                    schedulerModel->stop();
                } else {
                    throw gr::exception(std::format("ScheduledBlockGroup is not a SchedulerModel {}", block->uniqueName()));
                }
            } else {
                this->emitErrorMessageIfAny("forEachBlock -> stop() -> LifecycleState", block->changeStateTo(REQUESTED_STOP));
                if (!block->isBlocking()) { // N.B. no other thread/constraint to consider before shutting down
                    this->emitErrorMessageIfAny("forEachBlock -> stop() -> LifecycleState", block->changeStateTo(STOPPED));
                }
            }
        });

        this->emitErrorMessageIfAny("stop() -> LifecycleState ->STOPPED", this->changeStateTo(STOPPED));
        if constexpr (requires(Derived& d) { d.customStop(); }) {
            static_cast<Derived*>(this)->customStop();
        }
    }

    void pause() {
        using enum lifecycle::State;
        graph::forEachBlock<TransparentBlockGroup>(*_graph, [this](auto& block) {
            this->emitErrorMessageIfAny("pause() -> LifecycleState", block->changeStateTo(REQUESTED_PAUSE));
            if (!block->isBlocking()) { // N.B. no other thread/constraint to consider before shutting down
                this->emitErrorMessageIfAny("pause() -> LifecycleState", block->changeStateTo(PAUSED));
            }
        });
        this->emitErrorMessageIfAny("pause() -> LifecycleState", this->changeStateTo(PAUSED));
        if constexpr (requires(Derived& d) { d.customPause(); }) {
            static_cast<Derived*>(this)->customPause();
        }
    }

    void resume() {
        using enum lifecycle::State;
        auto result = connectPendingEdges();
        if (!result) {
            this->emitErrorMessage("init()", "Failed to connect blocks in graph");
        }
        graph::forEachBlock<TransparentBlockGroup>(*_graph, [this](auto& block) { this->emitErrorMessageIfAny("resume() -> LifecycleState", block->changeStateTo(RUNNING)); });
        if constexpr (requires(Derived& d) { d.customResume(); }) {
            static_cast<Derived*>(this)->customResume();
        }
    }

    std::optional<Message> propertyCallbackEmplaceBlock([[maybe_unused]] std::string_view propertyName, Message message) {
        using enum lifecycle::State;
        assert(propertyName == scheduler::property::kEmplaceBlock);
        using namespace std::string_literals;
        const auto& messageData = message.data.value();
        const auto  type        = messageData.at("type").value_or(std::string_view{});

        if (type.empty()) {
            message.data = std::unexpected(Error{std::format("No type specified for the message {}", message)});
            return message;
        }

        const property_map& properties = [&] {
            if (auto it = messageData.find("properties"); it != messageData.end()) {
                auto* result = it->second.get_if<property_map>();
                if (result == nullptr) {
                    return property_map{};
                } else {
                    return *result;
                }
            } else {
                return property_map{};
            }
        }();

        message.endpoint = scheduler::property::kBlockEmplaced;

        auto* targetGraph = findTargetSubGraph(messageData);

        if (targetGraph == nullptr) {
            message.data = std::unexpected(Error{std::format("No target graph for the message {}", message)});
            return message;
        }

        auto& newBlock = targetGraph->emplaceBlock(type, properties);

        if (ConnectionResult::SUCCESS != _toChildMessagePort.connect(*newBlock->msgIn)) {
            this->emitErrorMessage("connectBlockMessagePorts()", std::format("Failed to connect scheduler input message port to child '{}'", newBlock->uniqueName()));
        }

        auto toSchedulerBuffer = _fromChildMessagePort.buffer();
        newBlock->msgOut->setBuffer(toSchedulerBuffer.streamBuffer, toSchedulerBuffer.tagBuffer);

        if (lifecycle::isActive(this->state())) {
            // Block is being added while scheduler is running. Will be adopted by a thread.
            const auto nBatches = _adoptionBlocks.size();
            if (nBatches > 0) {
                std::lock_guard guard(_adoptionBlocksMutex);
                // pseudo-randomize which thread gets it
                auto blockAddress = reinterpret_cast<std::uintptr_t>(&newBlock);
                auto runnerIndex  = (blockAddress / sizeof(void*)) % nBatches;
                _adoptionBlocks[runnerIndex].push_back(newBlock);

                switch (newBlock->state()) {
                case STOPPED:
                case IDLE: //
                    this->emitErrorMessageIfAny("adoptBlocks -> INITIALIZED", newBlock->changeStateTo(INITIALISED));
                    this->emitErrorMessageIfAny("adoptBlocks -> INITIALIZED", newBlock->changeStateTo(RUNNING));
                    break;
                case INITIALISED: //
                    this->emitErrorMessageIfAny("adoptBlocks -> INITIALIZED", newBlock->changeStateTo(RUNNING));
                    break;
                case RUNNING:
                case REQUESTED_PAUSE:
                case PAUSED:
                case REQUESTED_STOP:
                case ERROR: //
                    this->emitErrorMessage("propertyCallbackEmplaceBlock", std::format("Unexpected block state during emplacement: {}", magic_enum::enum_name(newBlock->state())));
                    break;
                }
            }
        }

        auto replyData = serializeBlock(gr::globalPluginLoader(), newBlock, BlockSerializationFlags::All);

        replyData["_targetGraph"] = targetGraph->unique_name.value();

        this->emitMessage(scheduler::property::kBlockEmplaced, std::move(replyData));

        // Message is sent as a reaction to emplaceBlock, no need for a separate one
        return {};
    }

    std::optional<Message> propertyCallbackRemoveBlock([[maybe_unused]] std::string_view propertyName, Message message) {
        assert(propertyName == scheduler::property::kRemoveBlock);
        using namespace std::string_literals;
        auto&      messageData = message.data.value();
        const auto uniqueName  = messageData.at("uniqueName").value_or(std::string_view{});
        if (uniqueName.empty()) {
            message.data = std::unexpected(Error{std::format("No uniqueName in the message {}", message)});
            return message;
        }

        message.endpoint = scheduler::property::kBlockRemoved;

        auto* targetGraph = findTargetSubGraph(messageData);

        if (targetGraph == nullptr) {
            message.data = std::unexpected(Error{std::format("No target graph for the message {}", message)});
            return message;
        }

        messageData["_targetGraph"] = targetGraph->unique_name.value();
        auto removedBlock           = targetGraph->removeBlockByName(uniqueName);
        makeZombie(std::move(removedBlock));

        return {message};
    }

    std::optional<Message> propertyCallbackRemoveEdge([[maybe_unused]] std::string_view propertyName, Message message) {
        assert(propertyName == scheduler::property::kRemoveEdge);
        using namespace std::string_literals;
        auto&      messageData = message.data.value();
        const auto sourceBlock = messageData.at(std::pmr::string(gr::serialization_fields::EDGE_SOURCE_BLOCK)).value_or(std::string_view{});
        const auto sourcePort  = messageData.at(std::pmr::string(gr::serialization_fields::EDGE_SOURCE_PORT)).value_or(std::string_view{});
        if (sourceBlock.empty() || sourcePort.empty()) {
            message.data = std::unexpected(Error{std::format("No source definition for the message {}", message)});
            return message;
        }

        message.endpoint = scheduler::property::kEdgeRemoved;

        auto* targetGraph = findTargetSubGraph(messageData);

        if (targetGraph == nullptr) {
            message.data = std::unexpected(Error{std::format("No target graph for the message {}", message)});
            return message;
        }

        messageData["_targetGraph"] = targetGraph->unique_name.value();
        targetGraph->removeEdgeBySourcePort(sourceBlock, sourcePort);

        return message;
    }

    std::optional<Message> propertyCallbackEmplaceEdge([[maybe_unused]] std::string_view propertyName, Message message) {
        assert(propertyName == scheduler::property::kEmplaceEdge);
        using namespace std::string_literals;
        auto&                       messageData      = message.data.value();
        const auto                  sourceBlock      = messageData.at(std::pmr::string(gr::serialization_fields::EDGE_SOURCE_BLOCK)).value_or(std::string_view{});
        const auto                  sourcePort       = messageData.at(std::pmr::string(gr::serialization_fields::EDGE_SOURCE_PORT)).value_or(std::string_view{});
        const auto                  destinationBlock = messageData.at(std::pmr::string(gr::serialization_fields::EDGE_DESTINATION_BLOCK)).value_or(std::string_view{});
        const auto                  destinationPort  = messageData.at(std::pmr::string(gr::serialization_fields::EDGE_DESTINATION_PORT)).value_or(std::string_view{});
        [[maybe_unused]] const auto minBufferSize    = checked_access_ptr{messageData.at(std::pmr::string(gr::serialization_fields::EDGE_MIN_BUFFER_SIZE)).get_if<gr::Size_t>()};
        [[maybe_unused]] const auto weight           = checked_access_ptr{messageData.at(std::pmr::string(gr::serialization_fields::EDGE_WEIGHT)).get_if<std::int32_t>()};
        const auto                  edgeName         = messageData.at(std::pmr::string(gr::serialization_fields::EDGE_NAME)).value_or(std::string_view{});

        if (sourceBlock.empty() || sourcePort.empty() || destinationBlock.empty() || destinationPort.empty() || minBufferSize == nullptr || weight == nullptr || edgeName.empty()) {
            message.data = std::unexpected(Error{std::format("Message is incomplete {}", message)});
            return message;
        }

        message.endpoint = scheduler::property::kEdgeEmplaced;

        auto* targetGraph = findTargetSubGraph(messageData);

        if (targetGraph == nullptr) {
            message.data = std::unexpected(Error{std::format("No target graph for the message {}", message)});
            return message;
        }

        messageData["_targetGraph"] = targetGraph->unique_name.value();
        targetGraph->emplaceEdge(sourceBlock, std::string(sourcePort), destinationBlock, std::string(destinationPort), *minBufferSize, *weight, edgeName);

        return message;
    }

    /*
      Zombie Tutorial:

      Blocks cannot be deleted unless stopped, but stopping may take time (asynchronous).
      We therefore move such blocks to the "zombie list" and disconnect them immediately from the graph,
      allowing them to stop and be deleted safely.

      Each worker thread periodically calls cleanupZombieBlocks(), which:
      - removes fully stopped zombies from the zombie list
      - erases corresponding entries from its own localBlockList
      - updates the shared _executionOrder to ensure zombies do not reappear on restart

      This mechanism supports safe dynamic block removal while the scheduler is running, without blocking execution.
    */
    void cleanupZombieBlocks(std::vector<std::shared_ptr<BlockModel>>& localBlockList) {
        using enum lifecycle::State;
        if (localBlockList.empty()) {
            return;
        }

        std::lock_guard guard(_zombieBlocksMutex);

        auto it = _zombieBlocks.begin();

        while (it != _zombieBlocks.end()) {
            const auto localBlockIt = std::ranges::find(localBlockList, *it);
            if (localBlockIt == localBlockList.end()) {
                // we only care about the blocks local to our thread.
                ++it;
                continue;
            }

            bool shouldDelete = false;

            switch ((*it)->state()) {
            case IDLE:
            case STOPPED:
            case INITIALISED: // block can be deleted immediately
                shouldDelete = true;
                break;
            case ERROR: // delete as well
                shouldDelete = true;
                break;
            case REQUESTED_STOP: // block will be deleted later
                break;
            case REQUESTED_PAUSE: // block will be deleted later
                // There's no transition from REQUESTED_PAUSE to REQUESTED_STOP
                // Will be moved to REQUESTED_STOP as soon as it's possible
                break;
            case PAUSED: // zombie was in REQUESTED_PAUSE and now finally in PAUSED. Can be stopped now.
                // Will be deleted in a next zombie maintenance period
                this->emitErrorMessageIfAny("cleanupZombieBlocks", (*it)->changeStateTo(REQUESTED_STOP));
                break;
            case RUNNING: assert(false && "Doesn't happen: zombie blocks are never running"); break;
            }

            if (shouldDelete) {
                localBlockList.erase(localBlockIt);

                std::shared_ptr<BlockModel> zombieRaw = *it;
                it                                    = _zombieBlocks.erase(it); // ~Block() runs here

                // We need to remove zombieRaw from jobLists as well, in case Scheduler ever goes to INITIALIZED again.
                std::lock_guard lock(_executionOrderMutex);
                for (auto& jobList : *this->_executionOrder) {
                    auto job_it = std::remove(jobList.begin(), jobList.end(), zombieRaw);
                    if (job_it != jobList.end()) {
                        jobList.erase(job_it, jobList.end());
                        break;
                    }
                }

            } else {
                ++it;
            }
        }
    }

    void adoptBlocks(std::size_t runnerID, std::vector<std::shared_ptr<BlockModel>>& localBlockList) {
        std::lock_guard guard(_adoptionBlocksMutex);

        assert(_adoptionBlocks.size() > runnerID);
        auto& newBlocks = _adoptionBlocks[runnerID];

        localBlockList.reserve(localBlockList.size() + newBlocks.size());
        localBlockList.insert(localBlockList.end(), newBlocks.begin(), newBlocks.end());
        newBlocks.clear();
    }

    /*
      Moves a block to the zombie list:

      - Requests stop if the block is still running or paused.
      - Removes the block from adoption lists (to handle edge cases such as Add Block → Remove Block).
      - Adds it to the zombie list.

      The block will be physically deleted by cleanupZombieBlocks() when it reaches a safe state.
    */
    void makeZombie(std::shared_ptr<BlockModel> block) {
        using enum lifecycle::State;
        if (block->state() == PAUSED || block->state() == RUNNING) {
            this->emitErrorMessageIfAny("makeZombie", block->changeStateTo(REQUESTED_STOP));
        }

        {
            // Handle edge case: If we receive two consecutive "Add Block X" "Remove Block X" messages
            // it would be zombie before being adopted, so we need to remove it from adoption list
            std::lock_guard guard(_adoptionBlocksMutex);
            for (std::vector<std::shared_ptr<BlockModel>>& adoptionList : _adoptionBlocks) {
                if (auto it = std::ranges::find(adoptionList, block); it != adoptionList.end()) {
                    adoptionList.erase(it);
                    break;
                }
            }
        }

        std::lock_guard guard(_zombieBlocksMutex);
        _zombieBlocks.push_back(std::move(block));
    }

    // Moves all blocks into the zombie list
    // Useful for bulk operations such as "set grc yaml" message
    void makeAllZombies() {
        using enum lifecycle::State;
        std::lock_guard guard(_zombieBlocksMutex);

        for (auto& block : this->_graph->blocks()) {
            switch (block->state()) {
            case RUNNING:
            case REQUESTED_PAUSE:
            case PAUSED: //
                this->emitErrorMessageIfAny("makeAllZombies", block->changeStateTo(REQUESTED_STOP));
                break;

            case INITIALISED: //
                this->emitErrorMessageIfAny("makeAllZombies", block->changeStateTo(STOPPED));
                break;
            case IDLE:
            case STOPPED:
            case ERROR:
            case REQUESTED_STOP:
                // Can go into the zombie list and deleted
                break;
            default:;
            }

            _zombieBlocks.push_back(std::move(block));
        }

        this->_graph->clear();
    }

    std::optional<Message> propertyCallbackGraphGRC([[maybe_unused]] std::string_view propertyName, Message message) {
        using enum lifecycle::State;
        assert(propertyName == scheduler::property::kGraphGRC);

        auto& pluginLoader = gr::globalPluginLoader();
        if (message.cmd == message::Command::Get) {
            message.data = property_map{{"value", gr::saveGrc(pluginLoader, *_graph)}};
        } else if (message.cmd == message::Command::Set) {
            const auto& messageData = message.data.value();
            auto        yamlContent = messageData.at("value").value_or(std::string_view{});
            if (yamlContent.empty()) {
                message.data = std::unexpected(Error{std::format("Yaml content not found")});
            } else {
                try {
                    auto newGraph = gr::loadGrc(pluginLoader, yamlContent);

                    makeAllZombies();

                    const auto originalState = this->state();

                    if (auto result = this->exchange(std::move(newGraph)); !result) {
                        this->emitErrorMessage("propertyCallbackGraphGRC", "Failed to exchange graph");
                        return {};
                    }

                    message.data = property_map{{"originalSchedulerState", static_cast<int>(originalState)}};
                } catch (const std::exception& e) {
                    message.data = std::unexpected(Error{std::format("Error parsing YAML: {}", e.what())});
                }
            }

        } else {
            throw gr::exception(std::format("Unexpected command type {}", message.cmd));
        }

        return message;
    }

    std::optional<Message> propertyCallbackSchedulerInspect([[maybe_unused]] std::string_view propertyName, Message message) {
        assert(propertyName == scheduler::property::kSchedulerInspect);
        message.data = [&] {
            property_map result;
            result[std::pmr::string(serialization_fields::BLOCK_NAME)]        = std::string(this->name);
            result[std::pmr::string(serialization_fields::BLOCK_UNIQUE_NAME)] = std::string(this->unique_name);
            result[std::pmr::string(serialization_fields::BLOCK_CATEGORY)]    = std::string(magic_enum::enum_name(blockCategory));

            // Requesting graph serialization
            property_map serializedChildren;
            auto         graphData = _graph->propertyCallbackGraphInspect(graph::property::kGraphInspect, {});
            if (!graphData.has_value()) {
                return result;
            }
            serializedChildren[std::pmr::string(_graph->unique_name)] = graphData->data.value();

            result[std::pmr::string(serialization_fields::BLOCK_CHILDREN)] = std::move(serializedChildren);
            return result;
        }();

        message.endpoint = scheduler::property::kSchedulerInspected;
        return message;
    }

    std::optional<Message> propertyCallbackInspectBlock([[maybe_unused]] std::string_view propertyName, Message message) {
        auto result = _graph->propertyCallbackInspectBlock(propertyName, message);
        if (result) {
            result->serviceName = this->unique_name;
        }
        return result;
    }

    std::optional<Message> propertyCallbackReplaceBlock([[maybe_unused]] std::string_view propertyName, Message message) {
        assert(propertyName == scheduler::property::kReplaceBlock);
        using namespace std::string_literals;
        const auto& messageData = message.data.value();
        const auto  uniqueName  = messageData.at("uniqueName").value_or(std::string_view{});
        const auto  type        = messageData.at("type").value_or(std::string_view{});
        if (uniqueName.empty() || type.empty()) {
            message.data = std::unexpected(Error{std::format("No uniqueName or type in the message {}", message)});
            return message;
        }
        const property_map& properties = [&] {
            if (auto it = messageData.find("properties"); it != messageData.end()) {
                auto* result = it->second.get_if<property_map>();
                if (result == nullptr) {
                    return property_map{};
                } else {
                    return *result;
                }
            } else {
                return property_map{};
            }
        }();

        auto* targetGraph = findTargetSubGraph(messageData);

        if (targetGraph == nullptr) {
            message.data = std::unexpected(Error{std::format("No target graph for the message {}", message)});
            return message;
        }

        auto [oldBlock, newBlockRaw] = targetGraph->replaceBlock(uniqueName, type, properties);
        makeZombie(std::move(oldBlock));

        std::optional<Message> result = gr::Message{};
        result->endpoint              = scheduler::property::kBlockReplaced;
        result->data                  = serializeBlock(gr::globalPluginLoader(), newBlockRaw, BlockSerializationFlags::All);

        (*result->data)["_targetGraph"]            = targetGraph->unique_name.value();
        (*result->data)["replacedBlockUniqueName"] = uniqueName;

        return result;
    }
};

template<ExecutionPolicy execution = ExecutionPolicy::singleThreaded, profiling::ProfilerLike TProfiler = profiling::null::Profiler>
struct Simple : SchedulerBase<Simple<execution, TProfiler>, execution, TProfiler> {
    using Description = Doc<R""(Simple loop based Scheduler, which iterates over all blocks in the order they have beein defined and emplaced definition in the graph.)"">;

    using SchedulerBase<Simple<execution, TProfiler>, execution, TProfiler>::SchedulerBase;

    void customInit() {
        [[maybe_unused]] const auto pe = this->_profilerHandler->startCompleteEvent("scheduler_simple.init");

        // generate job list
        const gr::Graph   flatGraph = graph::flatten(*this->_graph);
        const std::size_t nBlocks   = flatGraph.blocks().size();

        std::size_t n_batches = 1UZ;
        switch (this->executionPolicy()) {
        case ExecutionPolicy::singleThreaded:
        case ExecutionPolicy::singleThreadedBlocking: break;
        case ExecutionPolicy::multiThreaded: n_batches = std::min(static_cast<std::size_t>(this->_pool->maxThreads()), nBlocks); break;
        default:;
        }

        std::lock_guard lock(this->_executionOrderMutex);
        this->_adoptionBlocks.clear();
        this->_adoptionBlocks.resize(n_batches);
        this->_executionOrder->clear();
        this->_executionOrder->reserve(n_batches);
        for (std::size_t i = 0; i < n_batches; i++) {
            // create job-set for thread
            auto& job = this->_executionOrder->emplace_back(std::vector<std::shared_ptr<BlockModel>>());
            job.reserve(nBlocks / n_batches + 1);
            for (std::size_t j = i; j < nBlocks; j += n_batches) {
                job.push_back(flatGraph.blocks()[j]);
            }
        }
    }
};

namespace detail {
inline JobLists batchBlocks(const std::vector<std::shared_ptr<BlockModel>>& blocks, std::size_t n_batches) {
    JobLists result(n_batches);
    for (std::size_t batch = 0UZ; batch < n_batches; ++batch) {
        result[batch].reserve(blocks.size() / n_batches + 1UZ);
        for (std::size_t i = batch; i < blocks.size(); i += n_batches) {
            result[batch].push_back(blocks[i]);
        }
    }
    return result;
}

inline void printExecutionOrder(const std::vector<std::vector<std::shared_ptr<BlockModel>>>& executionOrder) {
    std::size_t batchIndex = 0;
    for (const auto& batch : executionOrder) {
        std::print("Batch #{}:\n", batchIndex++);
        for (const auto& block : batch) {
            std::print("  - {} ({})\n", block->name(), block->uniqueName());
        }
    }
}

} // namespace detail

template<ExecutionPolicy execution = ExecutionPolicy::singleThreaded, profiling::ProfilerLike TProfiler = profiling::null::Profiler>
struct BreadthFirst : SchedulerBase<BreadthFirst<execution, TProfiler>, execution, TProfiler> {
    using Description = Doc<R""(Breadth First Scheduler which traverses the graph starting from the source blocks in a breath first fashion
detecting cycles and blocks which can be reached from several source blocks.)"">;

    static_assert(execution == ExecutionPolicy::singleThreaded || execution == ExecutionPolicy::multiThreaded, "Unsupported execution policy");

    void customInit() {
        /* implements Breadth-first search scheduling algorithm (https://en.wikipedia.org/wiki/Breadth-first_search)
         * 1. compute 'adjacencyList'
         * 2. determine all 'sourceBlocks' S (no incoming edges)
         * 3. initialise queue Q with S
         * 4. while Q not empty:
         *   - dequeue Block B
         *   - if B not visited:
         *     - mark visited
         *     - add B to result
         *   - for each outgoing edge from B:
         *     - if target not yet reached, enqueue target
         *
         * For more details see also:
         * [1] T. H. Cormen, C. E. Leiserson, R. L. Rivest, and C. Stein, "Introduction to Algorithms", 3rd ed., MIT Press, 2009, ch. 22.2.
         * [2] P. Morin, "Open Data Structures". [Online]. available at: https://opendatastructures.org/
         */
        using block_t                  = std::shared_ptr<BlockModel>;
        [[maybe_unused]] const auto pe = this->_profilerHandler->startCompleteEvent("breadth_first.init");

        gr::Graph                      flatGraph     = gr::graph::flatten(*this->_graph);
        const gr::graph::AdjacencyList adjacencyList = graph::computeAdjacencyList(flatGraph);
        const std::vector<block_t>     sourceBlocks  = graph::findSourceBlocks(adjacencyList);

        std::vector<block_t>        blockList;
        std::unordered_set<block_t> visited;
        std::queue<block_t>         queue;
        std::set<block_t>           reached;

        for (const auto& src : sourceBlocks) {
            if (reached.insert(src).second) {
                queue.push(src);
            }
        }

        while (!queue.empty()) {
            block_t current = queue.front();
            queue.pop();

            if (visited.insert(current).second) {
                blockList.push_back(current);
            }

            // enqueue outgoing neighbours, but only once
            if (adjacencyList.contains(current)) {
                for (const auto& edges : adjacencyList.at(current) | std::views::values) {
                    for (const auto* edge : edges) {
                        const auto& dst = edge->destinationBlock();
                        if (reached.insert(dst).second) {
                            queue.push(dst);
                        }
                    }
                }
            }
        }

        const std::size_t n_batches = (execution == ExecutionPolicy::multiThreaded) ? std::min(static_cast<std::size_t>(this->_pool->maxThreads()), blockList.size()) : 1UZ;

        std::lock_guard guard(this->_adoptionBlocksMutex);
        std::lock_guard lock(this->_executionOrderMutex);
        this->_adoptionBlocks.clear();
        this->_adoptionBlocks.resize(n_batches);
        *this->_executionOrder = detail::batchBlocks(blockList, n_batches);
    }
};

template<ExecutionPolicy execution = ExecutionPolicy::singleThreaded, profiling::ProfilerLike TProfiler = profiling::null::Profiler>
struct DepthFirst : SchedulerBase<DepthFirst<execution, TProfiler>, execution, TProfiler> {
    using Description = Doc<R""(Depth First Scheduler which traverses the graph starting from the source blocks in a depth-first manner.)"">;
    static_assert(execution == ExecutionPolicy::singleThreaded || execution == ExecutionPolicy::multiThreaded, "Unsupported execution policy");

    void customInit() {
        /**
         * implements Depth-first search scheduling algorithm (https://en.wikipedia.org/wiki/Depth-first_search)
         * 1. compute 'adjacencyList'
         * 2. determine all `sourceBlocks' S (no incoming edges)
         * 3. initialise visited set
         * 4. for each source s in S:
         *   - recursively visit(s)
         * 5. visit(Block B):
         *   - if B visited: return
         *   - mark B visited
         *   - add B to result
         *   - for each outgoing edge from B:
         *     - recursively visit(destination)
         *
         * For more details see also:
         * [1] T. H. Cormen, C. E. Leiserson, R. L. Rivest, and C. Stein, "Introduction to Algorithms", 3rd ed., MIT Press, 2009, ch. 22.2.
         * [2] P. Morin, "Open Data Structures". [Online]. available at: https://opendatastructures.org/
         */
        using block_t                  = std::shared_ptr<BlockModel>;
        [[maybe_unused]] const auto pe = this->_profilerHandler->startCompleteEvent("depth_first.init");

        gr::Graph                  flatGraph     = gr::graph::flatten(*this->_graph);
        const graph::AdjacencyList adjacencyList = graph::computeAdjacencyList(flatGraph);
        const std::vector<block_t> sourceBlocks  = graph::findSourceBlocks(adjacencyList);

        std::vector<block_t> blockList;
        std::set<block_t>    visited;

        auto dfs = [&](this auto&& self, const block_t& node) -> void {
            if (!visited.insert(node).second) {
                return; // already visited
            }
            blockList.push_back(node);

            if (adjacencyList.contains(node)) {
                for (const auto& edges : adjacencyList.at(node) | std::views::values) {
                    for (const auto* edge : edges) {
                        self(edge->destinationBlock());
                    }
                }
            }
        };

        for (const auto& src : sourceBlocks) {
            dfs(src);
        }

        const std::size_t n_batches = (execution == ExecutionPolicy::multiThreaded) ? std::min(static_cast<std::size_t>(this->_pool->maxThreads()), blockList.size()) : 1UZ;

        std::lock_guard guard(this->_adoptionBlocksMutex);
        std::lock_guard lock(this->_executionOrderMutex);
        this->_adoptionBlocks.clear();
        this->_adoptionBlocks.resize(n_batches);
        *this->_executionOrder = detail::batchBlocks(blockList, n_batches);
    }
};

} // namespace gr::scheduler

#endif // GNURADIO_SCHEDULER_HPP
