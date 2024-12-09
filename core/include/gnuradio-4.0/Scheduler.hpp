#ifndef GNURADIO_SCHEDULER_HPP
#define GNURADIO_SCHEDULER_HPP

#include <bit>
#include <chrono>
#include <mutex>
#include <queue>
#include <set>
#include <source_location>
#include <thread>
#include <utility>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/LifeCycle.hpp>
#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Profiler.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

namespace gr::scheduler {
using gr::thread_pool::BasicThreadPool;
using namespace gr::message;

enum class ExecutionPolicy {
    singleThreaded,        ///
    multiThreaded,         ///
    singleThreadedBlocking /// blocks with a time-out if none of the blocks in the graph made progress (N.B. a CPU/battery power-saving measures)
};

template<typename Derived, ExecutionPolicy execution = ExecutionPolicy::singleThreaded, profiling::ProfilerLike TProfiler = profiling::null::Profiler>
class SchedulerBase : public Block<Derived> {
    friend class lifecycle::StateMachine<Derived>;
    using JobLists = std::shared_ptr<std::vector<std::vector<BlockModel*>>>;

protected:
    gr::Graph                           _graph;
    TProfiler                           _profiler;
    decltype(_profiler.forThisThread()) _profilerHandler;
    std::shared_ptr<BasicThreadPool>    _pool;
    std::atomic_size_t                  _nRunningJobs{0UZ};
    std::recursive_mutex                _jobListsMutex; // only used when modifying and copying the graph->local job list
    JobLists                            _jobLists = std::make_shared<std::vector<std::vector<BlockModel*>>>();

    MsgPortOutForChildren    _toChildMessagePort;
    MsgPortInFromChildren    _fromChildMessagePort;
    std::vector<gr::Message> _pendingMessagesToChildren;
    bool                     _messagePortsConnected = false;

public:
    using base_t = Block<Derived>;

    Annotated<gr::Size_t, "timeout", Doc<"sleep timeout to wait if graph has made no progress ">>                              timeout_ms                      = 10U;
    Annotated<gr::Size_t, "timeout_inactivity_count", Doc<"number of inactive cycles w/o progress before sleep is triggered">> timeout_inactivity_count        = 20U;
    Annotated<gr::Size_t, "process_stream_to_message_ratio", Doc<"number of stream to msg processing">>                        process_stream_to_message_ratio = 16U;

    GR_MAKE_REFLECTABLE(SchedulerBase, timeout_ms, timeout_inactivity_count, process_stream_to_message_ratio);

    constexpr static block::Category blockCategory = block::Category::ScheduledBlockGroup;

    [[nodiscard]] static constexpr auto executionPolicy() { return execution; }

    explicit SchedulerBase(gr::Graph&&   graph,                                                                                                  //
        std::shared_ptr<BasicThreadPool> thread_pool       = std::make_shared<BasicThreadPool>("simple-scheduler-pool", thread_pool::CPU_BOUND), //
        const profiling::Options&        profiling_options = {})                                                                                        //
        : _graph(std::move(graph)), _profiler{profiling_options}, _profilerHandler{_profiler.forThisThread()}, _pool(std::move(thread_pool)) {}

    ~SchedulerBase() {
        if (this->state() == lifecycle::RUNNING) {
            if (auto e = this->changeStateTo(lifecycle::State::REQUESTED_STOP); !e) {
                fmt::println(std::cerr, "Failed to stop execution at destruction of scheduler: {} ({})", e.error().message, e.error().srcLoc());
                std::abort();
            }
        }
        waitDone();
    }

    [[nodiscard]] bool isProcessing() const
    requires(executionPolicy() == ExecutionPolicy::multiThreaded)
    {
        return _nRunningJobs.load(std::memory_order_acquire) > 0UZ;
    }

    void stateChanged(lifecycle::State newState) { this->notifyListeners(block::property::kLifeCycleState, {{"state", std::string(magic_enum::enum_name(newState))}}); }

    void connectBlockMessagePorts() {
        auto toSchedulerBuffer = _fromChildMessagePort.buffer();
        std::ignore            = _toChildMessagePort.connect(_graph.msgIn);
        _graph.msgOut.setBuffer(toSchedulerBuffer.streamBuffer, toSchedulerBuffer.tagBuffer);

        auto connectNestedBlocks = [this, &toSchedulerBuffer](auto& connectNestedBlocks_, auto& parent) -> void {
            const auto& blocks = [&parent] -> std::span<std::unique_ptr<BlockModel>> {
                if constexpr (requires { parent.blocks(); }) {
                    return parent.blocks();
                } else {
                    return parent->blocks();
                }
            }();
            for (auto& block : blocks) {
                if (ConnectionResult::SUCCESS != _toChildMessagePort.connect(*block->msgIn)) {
                    this->emitErrorMessage("connectBlockMessagePorts()", fmt::format("Failed to connect scheduler input message port to child '{}'", block->uniqueName()));
                }

                block->msgOut->setBuffer(toSchedulerBuffer.streamBuffer, toSchedulerBuffer.tagBuffer);

                if (block->blockCategory() != block::Category::NormalBlock) {
                    connectNestedBlocks_(connectNestedBlocks_, block);
                }
            }
        };
        connectNestedBlocks(connectNestedBlocks, _graph);

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
                    msgSpan[0]                  = std::move(msg);
                } else {
                    // if not yet connected, keep messages to children in cache and forward when connecting
                    _pendingMessagesToChildren.push_back(msg);
                }
            }
        }
    }

    void processScheduledMessages() {
        base_t::processScheduledMessages(); // filters messages and calls own property handler

        // Process messages in the graph
        _graph.processScheduledMessages();
        if (_nRunningJobs.load(std::memory_order_acquire) == 0UZ) {
            auto processNestedBlocks = [](auto& processNestedBlocks_, auto& parent) -> void {
                const auto& blocks = [&parent] -> std::span<std::unique_ptr<BlockModel>> {
                    if constexpr (requires { parent.blocks(); }) {
                        return parent.blocks();
                    } else {
                        return parent->blocks();
                    }
                }();

                for (auto& block : blocks) {
                    block->processScheduledMessages();

                    if (block->blockCategory() != block::Category::NormalBlock) {
                        processNestedBlocks_(processNestedBlocks_, block);
                    }
                }
            };
            processNestedBlocks(processNestedBlocks, _graph);
        }

        ReaderSpanLike auto messagesFromChildren = _fromChildMessagePort.streamReader().get();
        if (messagesFromChildren.size() == 0) {
            return;
        }

        if (this->msgOut.buffer().streamBuffer.n_readers() == 0) {
            // nobody is listening on messages -> convert errors to exceptions
            for (const auto& msg : messagesFromChildren) {
                if (!msg.data.has_value()) {
                    throw gr::exception(fmt::format("scheduler {}: throwing ignored exception {:t}", this->name, msg.data.error()));
                }
            }
            return;
        }

        {
            WriterSpanLike auto msgSpan = this->msgOut.streamWriter().template reserve<SpanReleasePolicy::ProcessAll>(messagesFromChildren.size());
            std::ranges::copy(messagesFromChildren, msgSpan.begin());
        } // to force publish
        if (!messagesFromChildren.consume(messagesFromChildren.size())) {
            this->emitErrorMessage("process child return messages", "Failed to consume messages from child message port");
        }
    }

    [[nodiscard]] constexpr auto& graph() { return _graph; }

    std::expected<void, Error> runAndWait() {
        [[maybe_unused]] const auto pe = this->_profilerHandler.startCompleteEvent("scheduler_base.runAndWait");
        base_t::processScheduledMessages(); // make sure initial subscriptions are processed
        if (this->state() == lifecycle::State::IDLE) {
            if (auto e = this->changeStateTo(lifecycle::State::INITIALISED); !e) {
                this->emitErrorMessage("runAndWait() -> LifecycleState", e.error());
                return std::unexpected(e.error());
            }
        }
        if (auto e = this->changeStateTo(lifecycle::State::RUNNING); !e) {
            this->emitErrorMessage("runAndWait() -> LifecycleState", e.error());
            return std::unexpected(e.error());
        }

        // N.B. the transition to lifecycle::State::RUNNING will for the ExecutionPolicy:
        // * singleThreaded[Blocking] naturally block in the calling thread
        // * multiThreaded[Blocking] spawn two worker and block on 'waitDone()'
        waitDone();
        this->processScheduledMessages();

        if (this->state() == lifecycle::State::RUNNING) {
            if (auto e = this->changeStateTo(lifecycle::State::REQUESTED_STOP); !e) {
                this->emitErrorMessage("runAndWait() -> LifecycleState", e.error());
                return std::unexpected(e.error());
            }
        }
        if (this->state() == lifecycle::State::REQUESTED_STOP) {
            if (auto e = this->changeStateTo(lifecycle::State::STOPPED); !e) {
                this->emitErrorMessage("runAndWait() -> LifecycleState", e.error());
            }
        }
        this->processScheduledMessages();
        return {};
    }

    void waitDone() {
        [[maybe_unused]] const auto pe = _profilerHandler.startCompleteEvent("scheduler_base.waitDone");
        while (_nRunningJobs.load(std::memory_order_acquire) > 0UZ) {
            std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
        }
    }

    [[nodiscard]] const JobLists& jobs() const noexcept { return _jobLists; }

protected:
    forceinline work::Result traverseBlockListOnce(const std::vector<BlockModel*>& blocks) noexcept {
        constexpr std::size_t requestedWorkAllBlocks = std::numeric_limits<std::size_t>::max();
        std::size_t           performedWorkAllBlocks = 0UZ;
        bool                  unfinishedBlocksExist  = false; // i.e. at least one block returned OK, INSUFFICIENT_INPUT_ITEMS, or INSUFFICIENT_OUTPU_ITEMS
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
        return {requestedWorkAllBlocks, performedWorkAllBlocks, unfinishedBlocksExist ? work::Status::OK : work::Status::DONE};
    }

    void init() {
        [[maybe_unused]] const auto pe = _profilerHandler.startCompleteEvent("scheduler_base.init");
        base_t::processScheduledMessages(); // make sure initial subscriptions are processed
        connectBlockMessagePorts();
    }

    void reset() {
        _graph.forEachBlockMutable([this](auto& block) { this->emitErrorMessageIfAny("reset() -> LifecycleState", block.changeState(lifecycle::INITIALISED)); });
        _graph.disconnectAllEdges();
    }

    void start() {
        const bool result = _graph.reconnectAllEdges();
        if (!result) {
            this->emitErrorMessage("init()", "Failed to connect blocks in graph");
        }

        std::lock_guard lock(_jobListsMutex);
        _graph.forEachBlockMutable([this](auto& block) { this->emitErrorMessageIfAny("LifecycleState -> RUNNING", block.changeState(lifecycle::RUNNING)); });
        if constexpr (executionPolicy() == ExecutionPolicy::singleThreaded || executionPolicy() == ExecutionPolicy::singleThreadedBlocking) {
            assert(_nRunningJobs.load(std::memory_order_acquire) == 0UZ);
            static_cast<Derived*>(this)->poolWorker(0UZ, _jobLists);
        } else { // run on processing thread pool
            [[maybe_unused]] const auto pe = _profilerHandler.startCompleteEvent("scheduler_base.runOnPool");
            assert(_nRunningJobs.load(std::memory_order_acquire) == 0UZ);
            for (std::size_t runnerID = 0UZ; runnerID < _jobLists->size(); runnerID++) {
                _pool->execute([this, runnerID]() { static_cast<Derived*>(this)->poolWorker(runnerID, _jobLists); });
            }
            if (!_jobLists->empty()) {
                _nRunningJobs.wait(0UZ, std::memory_order_acquire); // waits until at least one pool worker started
            }
        }
    }

    void poolWorker(const std::size_t runnerID, std::shared_ptr<std::vector<std::vector<BlockModel*>>> jobList) noexcept {
        _nRunningJobs.fetch_add(1UZ, std::memory_order_acq_rel);
        _nRunningJobs.notify_all();

        [[maybe_unused]] auto& profiler_handler = _profiler.forThisThread();

        std::vector<BlockModel*> localBlockList;
        {
            std::lock_guard          lock(_jobListsMutex);
            std::vector<BlockModel*> blocks = jobList->at(runnerID);
            localBlockList.reserve(blocks.size());
            for (const auto& block : blocks) {
                localBlockList.push_back(block);
            }
        }

        [[maybe_unused]] auto currentProgress    = this->_graph.progress().value();
        std::size_t           inactiveCycleCount = 0UZ;
        std::size_t           msgToCount         = 0UZ;
        auto                  activeState        = this->state();
        do {
            [[maybe_unused]] auto pe = profiler_handler.startCompleteEvent("scheduler_base.work");
            if constexpr (executionPolicy() == ExecutionPolicy::singleThreadedBlocking) {
                // optionally tracking progress and block if there is none
                currentProgress = this->_graph.progress().value();
            }

            bool processMessages = msgToCount == 0UZ;
            if (processMessages) {
                if (runnerID == 0UZ || _nRunningJobs.load(std::memory_order_acquire) == 0UZ) {
                    this->processScheduledMessages(); // execute the scheduler- and Graph-specific message handler only once globally
                }
                std::ranges::for_each(localBlockList, [](auto& block) { block->processScheduledMessages(); });
                activeState = this->state();
                msgToCount++;
            } else {
                if (std::has_single_bit(process_stream_to_message_ratio.value)) {
                    msgToCount = (msgToCount + 1U) & (process_stream_to_message_ratio.value - 1);
                } else {
                    msgToCount = (msgToCount + 1U) % process_stream_to_message_ratio.value;
                }
            }

            if (activeState == lifecycle::State::RUNNING) {
                gr::work::Result result = traverseBlockListOnce(localBlockList);
                if (result.status == work::Status::DONE) {
                    break; // nothing happened -> shutdown this worker
                } else if (result.status == work::Status::ERROR) {
                    this->emitErrorMessageIfAny("LifecycleState (ERROR)", this->changeStateTo(lifecycle::State::ERROR));
                    break;
                }
            } else if (activeState == lifecycle::State::PAUSED) {
                if (_graph.hasTopologyChanged()) {
                    // TODO: update localBlockList topology if needed
                    _graph.ackTopologyChange();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
                msgToCount = 0UZ;
            } else { // other states
                std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
                msgToCount = 0UZ;
            }

            // optionally tracking progress and block if there is none
            if constexpr (executionPolicy() == ExecutionPolicy::singleThreadedBlocking) {
                auto progressAfter = this->_graph.progress().value();
                if (currentProgress == progressAfter) {
                    inactiveCycleCount++;
                } else {
                    inactiveCycleCount = 0UZ;
                }

                currentProgress = progressAfter;
                if (inactiveCycleCount > timeout_inactivity_count) {
                    // allow scheduler process to sleep before retrying (N.B. intended to save CPU/battery power)
                    std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
                    msgToCount = 0UZ;
                }
            }
        } while (lifecycle::isActive(activeState));
        _nRunningJobs.fetch_sub(1UZ, std::memory_order_acq_rel);
        _nRunningJobs.notify_all();
        waitDone(); // wait for the other workers to finish.
    }

    void stop() {
        _graph.forEachBlockMutable([this](auto& block) {
            this->emitErrorMessageIfAny("forEachBlock -> stop() -> LifecycleState", block.changeState(lifecycle::State::REQUESTED_STOP));
            if (!block.isBlocking()) { // N.B. no other thread/constraint to consider before shutting down
                this->emitErrorMessageIfAny("forEachBlock -> stop() -> LifecycleState", block.changeState(lifecycle::State::STOPPED));
            }
        });
        this->emitErrorMessageIfAny("stop() -> LifecycleState ->STOPPED", this->changeStateTo(lifecycle::State::STOPPED));
        this->emitErrorMessageIfAny("stop() -> LifecycleState ->IDLE", this->changeStateTo(lifecycle::State::IDLE));
    }

    void pause() {
        _graph.forEachBlockMutable([this](auto& block) {
            this->emitErrorMessageIfAny("pause() -> LifecycleState", block.changeState(lifecycle::State::REQUESTED_PAUSE));
            if (!block.isBlocking()) { // N.B. no other thread/constraint to consider before shutting down
                this->emitErrorMessageIfAny("pause() -> LifecycleState", block.changeState(lifecycle::State::PAUSED));
            }
        });
        this->emitErrorMessageIfAny("pause() -> LifecycleState", this->changeStateTo(lifecycle::State::PAUSED));
    }

    void resume() {
        const bool result = _graph.connectPendingEdges();
        if (!result) {
            this->emitErrorMessage("init()", "Failed to connect blocks in graph");
        }
        _graph.forEachBlockMutable([this](auto& block) { this->emitErrorMessageIfAny("resume() -> LifecycleState", block.changeState(lifecycle::RUNNING)); });
    }
};

template<ExecutionPolicy execution = ExecutionPolicy::singleThreaded, profiling::ProfilerLike TProfiler = profiling::null::Profiler>
class Simple : public SchedulerBase<Simple<execution, TProfiler>, execution, TProfiler> {
    using Description = Doc<R""(Simple loop based Scheduler, which iterates over all blocks in the order they have beein defined and emplaced definition in the graph.)"">;

    friend class lifecycle::StateMachine<Simple<execution, TProfiler>>;
    friend class SchedulerBase<Simple<execution, TProfiler>, execution, TProfiler>;

public:
    using base_t = SchedulerBase<Simple<execution, TProfiler>, execution, TProfiler>;

    explicit Simple(gr::Graph&& graph, std::shared_ptr<BasicThreadPool> thread_pool = std::make_shared<BasicThreadPool>("simple-scheduler-pool", thread_pool::CPU_BOUND), const profiling::Options& profiling_options = {}) : base_t(std::move(graph), thread_pool, profiling_options) {}

private:
    void init() {
        base_t::init();
        [[maybe_unused]] const auto pe = this->_profilerHandler.startCompleteEvent("scheduler_simple.init");

        // generate job list
        std::size_t n_batches = 1UZ;
        switch (base_t::executionPolicy()) {
        case ExecutionPolicy::singleThreaded:
        case ExecutionPolicy::singleThreadedBlocking: break;
        case ExecutionPolicy::multiThreaded: n_batches = std::min(static_cast<std::size_t>(this->_pool->maxThreads()), this->_graph.blocks().size()); break;
        }

        std::lock_guard lock(base_t::_jobListsMutex);
        this->_jobLists->reserve(n_batches);
        for (std::size_t i = 0; i < n_batches; i++) {
            // create job-set for thread
            auto& job = this->_jobLists->emplace_back(std::vector<BlockModel*>());
            job.reserve(this->_graph.blocks().size() / n_batches + 1);
            for (std::size_t j = i; j < this->_graph.blocks().size(); j += n_batches) {
                job.push_back(this->_graph.blocks()[j].get());
            }
        }
    }
};

template<ExecutionPolicy execution = ExecutionPolicy::singleThreaded, profiling::ProfilerLike TProfiler = profiling::null::Profiler>
class BreadthFirst : public SchedulerBase<BreadthFirst<execution, TProfiler>, execution, TProfiler> {
    using Description = Doc<R""(Breadth First Scheduler which traverses the graph starting from the source blocks in a breath first fashion
detecting cycles and blocks which can be reached from several source blocks.)"">;

    friend class lifecycle::StateMachine<BreadthFirst<execution, TProfiler>>;
    friend class SchedulerBase<BreadthFirst<execution, TProfiler>, execution, TProfiler>;
    static_assert(execution == ExecutionPolicy::singleThreaded || execution == ExecutionPolicy::multiThreaded, "Unsupported execution policy");
    std::vector<BlockModel*> _blocklist;

public:
    using base_t = SchedulerBase<BreadthFirst<execution, TProfiler>, execution, TProfiler>;

    explicit BreadthFirst(gr::Graph&& graph, std::shared_ptr<BasicThreadPool> thread_pool = std::make_shared<BasicThreadPool>("breadth-first-pool", thread_pool::CPU_BOUND), const profiling::Options& profiling_options = {}) : base_t(std::move(graph), thread_pool, profiling_options) {}

private:
    void init() {
        [[maybe_unused]] const auto pe = this->_profilerHandler.startCompleteEvent("breadth_first.init");
        using block_t                  = BlockModel*;
        base_t::init();
        // calculate adjacency list
        std::map<block_t, std::vector<block_t>> _adjacency_list{};
        std::vector<block_t>                    _source_blocks{};
        // compute the adjacency list
        std::set<block_t> block_reached;
        for (auto& e : this->_graph.edges()) {
            _adjacency_list[e._sourceBlock].push_back(e._destinationBlock);
            _source_blocks.push_back(e._sourceBlock);
            block_reached.insert(e._destinationBlock);
        }
        _source_blocks.erase(std::remove_if(_source_blocks.begin(), _source_blocks.end(), [&block_reached](auto currentBlock) { return block_reached.contains(currentBlock); }), _source_blocks.end());
        // traverse graph
        std::queue<block_t> queue{};
        std::set<block_t>   reached;
        // add all source blocks to queue
        for (block_t sourceBlock : _source_blocks) {
            if (!reached.contains(sourceBlock)) {
                queue.push(sourceBlock);
            }
            reached.insert(sourceBlock);
        }

        // process all blocks, adding all unvisited child blocks to the queue
        while (!queue.empty()) {
            block_t currentBlock = queue.front();
            queue.pop();
            _blocklist.push_back(currentBlock);
            if (_adjacency_list.contains(currentBlock)) { // node has outgoing edges
                for (auto& dst : _adjacency_list.at(currentBlock)) {
                    if (!reached.contains(dst)) { // detect cycles. this could be removed if we guarantee cycle free graphs earlier
                        queue.push(dst);
                        reached.insert(dst);
                    }
                }
            }
        }

        // generate job list
        std::size_t n_batches = 1UZ;
        switch (base_t::executionPolicy()) {
        case ExecutionPolicy::singleThreaded:
        case ExecutionPolicy::singleThreadedBlocking: break;
        case ExecutionPolicy::multiThreaded: n_batches = std::min(static_cast<std::size_t>(this->_pool->maxThreads()), _blocklist.size()); break;
        }

        std::lock_guard lock(base_t::_jobListsMutex);
        this->_jobLists->reserve(n_batches);
        for (std::size_t i = 0; i < n_batches; i++) {
            // create job-set for thread
            auto& job = this->_jobLists->emplace_back(std::vector<BlockModel*>());
            job.reserve(_blocklist.size() / n_batches + 1);
            for (std::size_t j = i; j < _blocklist.size(); j += n_batches) {
                job.push_back(_blocklist[j]);
            }
        }
    }
};
} // namespace gr::scheduler

#endif // GNURADIO_SCHEDULER_HPP
