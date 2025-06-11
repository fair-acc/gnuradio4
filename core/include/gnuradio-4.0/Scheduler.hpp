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

inline static const char* kGraphGRC = "GraphGRC";
} // namespace property

enum class ExecutionPolicy {
    singleThreaded,        ///
    multiThreaded,         ///
    singleThreadedBlocking /// blocks with a time-out if none of the blocks in the graph made progress (N.B. a CPU/battery power-saving measures)
};

template<typename Derived, ExecutionPolicy execution = ExecutionPolicy::singleThreaded, profiling::ProfilerLike TProfiler = profiling::null::Profiler>
class SchedulerBase : public Block<Derived> {
    friend class lifecycle::StateMachine<Derived>;
    using JobLists     = std::shared_ptr<std::vector<std::vector<BlockModel*>>>;
    using TaskExecutor = gr::thread_pool::TaskExecutor;

protected:
    gr::Graph                           _graph;
    TProfiler                           _profiler;
    decltype(_profiler.forThisThread()) _profilerHandler;
    std::shared_ptr<TaskExecutor>       _pool;
    std::atomic_size_t                  _nRunningJobs{0UZ};
    std::recursive_mutex                _jobListsMutex; // only used when modifying and copying the graph->local job list
    JobLists                            _jobLists = std::make_shared<std::vector<std::vector<BlockModel*>>>();

    std::mutex                               _zombieBlocksMutex;
    std::vector<std::unique_ptr<BlockModel>> _zombieBlocks;

    // for blocks that were added while scheduler was running. They need to be adopted by a thread
    std::mutex _adoptionBlocksMutex;
    // fixed-sized vector indexed by runnerId. Cheaper than a map.
    std::vector<std::vector<BlockModel*>> _adoptionBlocks;

    MsgPortOutForChildren    _toChildMessagePort;
    MsgPortInFromChildren    _fromChildMessagePort;
    std::vector<gr::Message> _pendingMessagesToChildren;
    bool                     _messagePortsConnected = false;

    template<typename Fn>
    void forAllUnmanagedBlocks(Fn&& function) {
        auto doForNestedBlocks = [&function](auto& doForNestedBlocks_, auto& parent) -> void {
            const auto& blocks = [&parent] -> std::span<std::unique_ptr<BlockModel>> {
                if constexpr (requires { parent.blocks(); }) {
                    return parent.blocks();
                } else {
                    return parent->blocks();
                }
                // Silence warnings
                return {};
            }();
            for (auto& block : blocks) {
                function(block);

                if (block->blockCategory() == block::Category::TransparentBlockGroup) {
                    doForNestedBlocks_(doForNestedBlocks_, block);
                }
            }
        };
        doForNestedBlocks(doForNestedBlocks, _graph);
    }

public:
    using base_t = Block<Derived>;

    Annotated<gr::Size_t, "timeout", Doc<"sleep timeout to wait if graph has made no progress ">>                              timeout_ms                      = 10U;
    Annotated<gr::Size_t, "timeout_inactivity_count", Doc<"number of inactive cycles w/o progress before sleep is triggered">> timeout_inactivity_count        = 20U;
    Annotated<gr::Size_t, "process_stream_to_message_ratio", Doc<"number of stream to msg processing">>                        process_stream_to_message_ratio = 16U;
    Annotated<std::string, "pool name", Doc<"default pool name">>                                                              poolName                        = std::string(gr::thread_pool::kDefaultCpuPoolId);

    GR_MAKE_REFLECTABLE(SchedulerBase, timeout_ms, timeout_inactivity_count, process_stream_to_message_ratio);

    constexpr static block::Category blockCategory = block::Category::ScheduledBlockGroup;

    [[nodiscard]] static constexpr auto executionPolicy() { return execution; }

    explicit SchedulerBase(gr::Graph&& graph, std::string_view defaultPoolName, const profiling::Options& profiling_options = {}) //
        : _graph(std::move(graph)), _profiler{profiling_options}, _profilerHandler{_profiler.forThisThread()}, _pool(gr::thread_pool::Manager::instance().get(defaultPoolName)) {
        this->propertyCallbacks[scheduler::property::kEmplaceBlock] = std::mem_fn(&SchedulerBase::propertyCallbackEmplaceBlock);
        this->propertyCallbacks[scheduler::property::kRemoveBlock]  = std::mem_fn(&SchedulerBase::propertyCallbackRemoveBlock);
        this->propertyCallbacks[scheduler::property::kRemoveEdge]   = std::mem_fn(&SchedulerBase::propertyCallbackRemoveEdge);
        this->propertyCallbacks[scheduler::property::kEmplaceEdge]  = std::mem_fn(&SchedulerBase::propertyCallbackEmplaceEdge);
        this->propertyCallbacks[scheduler::property::kReplaceBlock] = std::mem_fn(&SchedulerBase::propertyCallbackReplaceBlock);
        this->propertyCallbacks[scheduler::property::kGraphGRC]     = std::mem_fn(&SchedulerBase::propertyCallbackGraphGRC);
    }

    ~SchedulerBase() {
        if (this->state() == lifecycle::RUNNING) {
            if (auto e = this->changeStateTo(lifecycle::State::REQUESTED_STOP); !e) {
                std::println(std::cerr, "Failed to stop execution at destruction of scheduler: {} ({})", e.error().message, e.error().srcLoc());
                std::abort();
            }
        }
        waitDone();
        _jobLists.reset(); // force earlier crashes is accessed after destruction
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

        forAllUnmanagedBlocks([this, &toSchedulerBuffer](auto& block) {
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
            forAllUnmanagedBlocks([](auto& block) { block->processScheduledMessages(); });
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
        } // to force publish
        if (!messagesFromChildren.consume(messagesFromChildren.size())) {
            this->emitErrorMessage("process child return messages", "Failed to consume messages from child message port");
        }
    }

    [[nodiscard]] constexpr auto& graph() { return _graph; }

    std::expected<void, Error> runAndWait() {
        [[maybe_unused]] const auto pe = this->_profilerHandler.startCompleteEvent("scheduler_base.runAndWait");
        processScheduledMessages(); // make sure initial subscriptions are processed
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
        processScheduledMessages();

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
        processScheduledMessages();
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
    void disconnectAllEdges() {
        _graph.disconnectAllEdges();
        forAllUnmanagedBlocks([&](auto& block) {
            if (block->blockCategory() == block::Category::TransparentBlockGroup) {
                auto* graph = static_cast<GraphWrapper<gr::Graph>*>(block.get());
                graph->blockRef().disconnectAllEdges();
            }
        });
    }

    bool connectPendingEdges() {
        bool result = _graph.connectPendingEdges();
        this->forAllUnmanagedBlocks([&](auto& block) {
            if (block->blockCategory() == block::Category::TransparentBlockGroup) {
                auto* graph = static_cast<GraphWrapper<gr::Graph>*>(block.get());
                result      = result && graph->blockRef().connectPendingEdges();
            }
        });
        return result;
    }

    work::Result traverseBlockListOnce(const std::vector<BlockModel*>& blocks) noexcept {
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
        forAllUnmanagedBlocks([this](auto& block) { this->emitErrorMessageIfAny("reset() -> LifecycleState", block->changeStateTo(lifecycle::INITIALISED)); });
        disconnectAllEdges();
    }

    void start() {
        disconnectAllEdges();
        auto result = connectPendingEdges();

        if (!result) {
            this->emitErrorMessage("init()", "Failed to connect blocks in graph");
        }

        std::lock_guard lock(_jobListsMutex);
        forAllUnmanagedBlocks([this](auto& block) { //
            this->emitErrorMessageIfAny("LifecycleState -> RUNNING", block->changeStateTo(lifecycle::RUNNING));
        });
        if constexpr (executionPolicy() == ExecutionPolicy::singleThreaded || executionPolicy() == ExecutionPolicy::singleThreadedBlocking) {
            assert(_nRunningJobs.load(std::memory_order_acquire) == 0UZ);
            static_cast<Derived*>(this)->poolWorker(0UZ, _jobLists);
        } else { // run on processing thread pool
            [[maybe_unused]] const auto pe = _profilerHandler.startCompleteEvent("scheduler_base.runOnPool");
            assert(_nRunningJobs.load(std::memory_order_acquire) == 0UZ);
            auto jobListsCopy = _jobLists;
            for (std::size_t runnerID = 0UZ; runnerID < _jobLists->size(); runnerID++) {
                _pool->execute([this, runnerID, jobListsCopy]() { static_cast<Derived*>(this)->poolWorker(runnerID, jobListsCopy); });
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

                // Zombies are cleaned per-thread, as we remove from the localBlockList as well.
                // Cleaning zombies has low priority, so uses process_stream_to_message_ratio (a different ratio could be introduced)
                cleanupZombieBlocks(localBlockList);

                adoptBlocks(runnerID, localBlockList);

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
    }

    void stop() {
        forAllUnmanagedBlocks([this](auto& block) {
            this->emitErrorMessageIfAny("forEachBlock -> stop() -> LifecycleState", block->changeStateTo(lifecycle::State::REQUESTED_STOP));
            if (!block->isBlocking()) { // N.B. no other thread/constraint to consider before shutting down
                this->emitErrorMessageIfAny("forEachBlock -> stop() -> LifecycleState", block->changeStateTo(lifecycle::State::STOPPED));
            }
        });

        this->emitErrorMessageIfAny("stop() -> LifecycleState ->STOPPED", this->changeStateTo(lifecycle::State::STOPPED));
    }

    void pause() {
        forAllUnmanagedBlocks([this](auto& block) {
            this->emitErrorMessageIfAny("pause() -> LifecycleState", block->changeStateTo(lifecycle::State::REQUESTED_PAUSE));
            if (!block->isBlocking()) { // N.B. no other thread/constraint to consider before shutting down
                this->emitErrorMessageIfAny("pause() -> LifecycleState", block->changeStateTo(lifecycle::State::PAUSED));
            }
        });
        this->emitErrorMessageIfAny("pause() -> LifecycleState", this->changeStateTo(lifecycle::State::PAUSED));
    }

    void resume() {
        auto result = connectPendingEdges();
        if (!result) {
            this->emitErrorMessage("init()", "Failed to connect blocks in graph");
        }
        forAllUnmanagedBlocks([this](auto& block) { this->emitErrorMessageIfAny("resume() -> LifecycleState", block->changeStateTo(lifecycle::RUNNING)); });
    }

    std::optional<Message> propertyCallbackEmplaceBlock([[maybe_unused]] std::string_view propertyName, Message message) {
        assert(propertyName == scheduler::property::kEmplaceBlock);
        using namespace std::string_literals;
        const auto&         data       = message.data.value();
        const std::string&  type       = std::get<std::string>(data.at("type"s));
        const property_map& properties = [&] {
            if (auto it = data.find("properties"s); it != data.end()) {
                return std::get<property_map>(it->second);
            } else {
                return property_map{};
            }
        }();

        auto& newBlock = _graph.emplaceBlock(type, properties);

        if (lifecycle::isActive(this->state())) {
            // Block is being added while scheduler is running. Will be adopted by a thread.
            const auto nBatches = _adoptionBlocks.size();
            if (nBatches > 0) {
                std::lock_guard guard(_adoptionBlocksMutex);
                // pseudo-randomize which thread gets it
                auto blockAddress = reinterpret_cast<std::uintptr_t>(&newBlock);
                auto runnerIndex  = (blockAddress / sizeof(void*)) % nBatches;
                _adoptionBlocks[runnerIndex].push_back(&newBlock);

                switch (newBlock.state()) {
                case lifecycle::State::STOPPED:
                case lifecycle::State::IDLE: //
                    this->emitErrorMessageIfAny("adoptBlocks -> INITIALIZED", newBlock.changeStateTo(lifecycle::State::INITIALISED));
                    this->emitErrorMessageIfAny("adoptBlocks -> INITIALIZED", newBlock.changeStateTo(lifecycle::State::RUNNING));
                    break;
                case lifecycle::State::INITIALISED: //
                    this->emitErrorMessageIfAny("adoptBlocks -> INITIALIZED", newBlock.changeStateTo(lifecycle::State::RUNNING));
                    break;
                case lifecycle::State::RUNNING:
                case lifecycle::State::REQUESTED_PAUSE:
                case lifecycle::State::PAUSED:
                case lifecycle::State::REQUESTED_STOP:
                case lifecycle::State::ERROR: //
                    this->emitErrorMessage("propertyCallbackEmplaceBlock", std::format("Unexpected block state during emplacement: {}", magic_enum::enum_name(newBlock.state())));
                    break;
                }
            }
        }

        this->emitMessage(scheduler::property::kBlockEmplaced, Graph::serializeBlock(std::addressof(newBlock)));

        // Message is sent as a reaction to emplaceBlock, no need for a separate one
        return {};
    }

    std::optional<Message> propertyCallbackRemoveBlock([[maybe_unused]] std::string_view propertyName, Message message) {
        assert(propertyName == scheduler::property::kRemoveBlock);
        using namespace std::string_literals;
        const auto&        data       = message.data.value();
        const std::string& uniqueName = std::get<std::string>(data.at("uniqueName"s));

        auto removedBlock = _graph.removeBlockByName(uniqueName);
        makeZombie(std::move(removedBlock));

        message.endpoint = scheduler::property::kBlockRemoved;
        return {message};
    }

    std::optional<Message> propertyCallbackRemoveEdge([[maybe_unused]] std::string_view propertyName, Message message) {
        assert(propertyName == scheduler::property::kRemoveEdge);
        using namespace std::string_literals;
        const auto&        data        = message.data.value();
        const std::string& sourceBlock = std::get<std::string>(data.at("sourceBlock"s));
        const std::string& sourcePort  = std::get<std::string>(data.at("sourcePort"s));

        _graph.removeEdgeBySourcePort(sourceBlock, sourcePort);

        message.endpoint = scheduler::property::kEdgeRemoved;
        return message;
    }

    std::optional<Message> propertyCallbackEmplaceEdge([[maybe_unused]] std::string_view propertyName, Message message) {
        assert(propertyName == scheduler::property::kEmplaceEdge);
        using namespace std::string_literals;
        const auto&                         data             = message.data.value();
        const std::string&                  sourceBlock      = std::get<std::string>(data.at("sourceBlock"s));
        const std::string&                  sourcePort       = std::get<std::string>(data.at("sourcePort"s));
        const std::string&                  destinationBlock = std::get<std::string>(data.at("destinationBlock"s));
        const std::string&                  destinationPort  = std::get<std::string>(data.at("destinationPort"s));
        [[maybe_unused]] const std::size_t  minBufferSize    = std::get<gr::Size_t>(data.at("minBufferSize"s));
        [[maybe_unused]] const std::int32_t weight           = std::get<std::int32_t>(data.at("weight"s));
        const std::string                   edgeName         = std::get<std::string>(data.at("edgeName"s));

        _graph.emplaceEdge(sourceBlock, sourcePort, destinationBlock, destinationPort, minBufferSize, weight, edgeName);

        message.endpoint = scheduler::property::kEdgeEmplaced;
        return message;
    }

    /*
      Zombie Tutorial:

      Blocks can't be deleted unless stopped, but since stopping can take time (async) we move such blocks
      to the "zombie list" and disconnect them immediately from the graph. This allows them to stop and be deleted
      safely.

      Periodically, we call cleanupZombieBlocks(), which iterates the zombie list and deletes the blocks that are now stopped.

      cleanupZombieBlocks() is called *per-thread*, since we also need to update the localBlockList, i.e.: removing dangling block pointers
      from the localBlockList.

      We also update the _jobLists member variable, but probably that member can be removed, seems unneeded and only used so unit-tests can
      query it.
     */
    void cleanupZombieBlocks(std::vector<BlockModel*>& localBlockList) {
        if (localBlockList.empty()) {
            return;
        }

        std::lock_guard guard(_zombieBlocksMutex);

        auto it = _zombieBlocks.begin();

        while (it != _zombieBlocks.end()) {
            auto localBlockIt = std::find(localBlockList.begin(), localBlockList.end(), it->get());
            if (localBlockIt == localBlockList.end()) {
                // we only care about the blocks local to our thread.
                ++it;
                continue;
            }

            bool shouldDelete = false;

            switch ((*it)->state()) {
            case lifecycle::State::IDLE:
            case lifecycle::State::STOPPED:
            case lifecycle::State::INITIALISED:
                // This block can be deleted immediately
                shouldDelete = true;
                break;
            case lifecycle::State::ERROR:
                // Delete as well. (Separate case block, as better ideas welcome)
                shouldDelete = true;
                break;
            case lifecycle::State::REQUESTED_STOP:
                // This block will be deleted later
                break;
            case lifecycle::State::REQUESTED_PAUSE:
                // This block will be deleted later
                // There's no transition from REQUESTED_PAUSE to REQUESTED_STOP
                // Will be moved to REQUESTED_STOP as soon as it's possible
                break;
            case lifecycle::State::PAUSED:
                // This zombie was in REQUESTED_PAUSE and now finally in PAUSED. Can be stopped now.
                // Will be deleted in a next zombie maintenance period
                this->emitErrorMessageIfAny("cleanupZombieBlocks", (*it)->changeStateTo(lifecycle::State::REQUESTED_STOP));
                break;
            case lifecycle::State::RUNNING: assert(false && "Doesn't happen: zombie blocks are never running"); break;
            }

            if (shouldDelete) {
                localBlockList.erase(localBlockIt);

                BlockModel* zombieRaw = it->get();
                it                    = _zombieBlocks.erase(it); // ~Block() runs here

                // We need to remove zombieRaw from jobLists as well, in case Scheduler ever goes to INITIALIZED
                // again.
                // TODO: I'd argue we should remove _jobLists to minimize having to maintain state. Instead, a job list can be
                // calculated in start().
                std::lock_guard lock(_jobListsMutex);
                for (auto& jobList : *this->_jobLists) {
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

    void adoptBlocks(std::size_t runnerID, std::vector<BlockModel*>& localBlockList) {
        std::lock_guard guard(_adoptionBlocksMutex);

        assert(_adoptionBlocks.size() > runnerID);
        auto& newBlocks = _adoptionBlocks[runnerID];

        localBlockList.reserve(localBlockList.size() + newBlocks.size());
        localBlockList.insert(localBlockList.end(), newBlocks.begin(), newBlocks.end());
        newBlocks.clear();
    }

    void makeZombie(std::unique_ptr<BlockModel> block) {
        if (block->state() == lifecycle::State::PAUSED || block->state() == lifecycle::State::RUNNING) {
            this->emitErrorMessageIfAny("makeZombie", block->changeStateTo(lifecycle::State::REQUESTED_STOP));
        }

        {
            // Handle edge case: If we receive two consecutive "Add Block X" "Remove Block X" messages
            // it would be zombie before being adopted, so we need to remove it from adoption list
            std::lock_guard guard(_adoptionBlocksMutex);
            for (auto& adoptionList : _adoptionBlocks) {
                auto it = std::find(adoptionList.begin(), adoptionList.end(), block.get());
                if (it != adoptionList.end()) {
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
        std::lock_guard guard(_zombieBlocksMutex);

        for (auto& block : this->_graph.blocks()) {
            switch (block->state()) {
            case lifecycle::State::RUNNING:
            case lifecycle::State::REQUESTED_PAUSE:
            case lifecycle::State::PAUSED: //
                this->emitErrorMessageIfAny("makeAllZombies", block->changeStateTo(lifecycle::State::REQUESTED_STOP));
                break;

            case lifecycle::State::INITIALISED: //
                this->emitErrorMessageIfAny("makeAllZombies", block->changeStateTo(lifecycle::State::STOPPED));
                break;
            case lifecycle::State::IDLE:
            case lifecycle::State::STOPPED:
            case lifecycle::State::ERROR:
            case lifecycle::State::REQUESTED_STOP:
                // Can go into the zombie list and deleted
                break;
            }

            _zombieBlocks.push_back(std::move(block));
        }

        this->_graph.clear();
    }

    std::optional<Message> propertyCallbackGraphGRC([[maybe_unused]] std::string_view propertyName, Message message) {
        assert(propertyName == scheduler::property::kGraphGRC);

        auto& pluginLoader = gr::globalPluginLoader();
        if (message.cmd == message::Command::Get) {
            message.data = property_map{{"value", gr::saveGrc(pluginLoader, _graph)}};
        } else if (message.cmd == message::Command::Set) {
            const auto& data        = message.data.value();
            auto        yamlContent = std::get<std::string>(data.at("value"s));

            try {
                Graph newGraph = gr::loadGrc(pluginLoader, yamlContent);

                makeAllZombies();

                const auto originalState = this->state();

                switch (originalState) {
                case lifecycle::State::RUNNING:
                case lifecycle::State::REQUESTED_PAUSE:
                case lifecycle::State::PAUSED: //
                    this->emitErrorMessageIfAny("propertyCallbackGraphGRC -> REQUESTED_STOP", this->changeStateTo(lifecycle::State::REQUESTED_STOP));
                    this->emitErrorMessageIfAny("propertyCallbackGraphGRC -> STOPPED", this->changeStateTo(lifecycle::State::STOPPED));
                    break;
                case lifecycle::State::REQUESTED_STOP:
                case lifecycle::State::INITIALISED: //
                    this->emitErrorMessageIfAny("propertyCallbackGraphGRC -> REQUESTED_STOP", this->changeStateTo(lifecycle::State::STOPPED));
                    break;
                case lifecycle::State::IDLE:
                    assert(false); // doesn't happen
                    break;
                case lifecycle::State::STOPPED:
                case lifecycle::State::ERROR: break;
                }

                _graph = std::move(newGraph);

                // Now ideally we'd just restart the Scheduler, but we can't since we're processing a message inside a working thread.
                // When the scheduler starts running it asserts that _nRunningJobs is 0, so we can't start it now, we're in the job.
                // We need to let poolWorker() unwind, decrement _nRunningJobs and then move scheduler to its original value.
                // Alternatively, we could forbid kGraphGRC unless Scheduler was in STOPPED state. That would simplify logic, but
                // put more burden on the client.

                message.data = property_map{{"originalSchedulerState", int(originalState)}};
            } catch (const std::exception& e) {
                message.data = std::unexpected(Error{std::format("Error parsing YAML: {}", e.what())});
            }

        } else {
            throw gr::exception(std::format("Unexpected command type {}", message.cmd));
        }

        return message;
    }

    std::optional<Message> propertyCallbackReplaceBlock([[maybe_unused]] std::string_view propertyName, Message message) {
        assert(propertyName == scheduler::property::kReplaceBlock);
        using namespace std::string_literals;
        const auto&         data       = message.data.value();
        const std::string&  uniqueName = std::get<std::string>(data.at("uniqueName"s));
        const std::string&  type       = std::get<std::string>(data.at("type"s));
        const property_map& properties = [&] {
            if (auto it = data.find("properties"s); it != data.end()) {
                return std::get<property_map>(it->second);
            } else {
                return property_map{};
            }
        }();

        auto [oldBlock, newBlockRaw] = _graph.replaceBlock(uniqueName, type, properties);
        makeZombie(std::move(oldBlock));

        std::optional<Message> result = gr::Message{};
        result->endpoint              = scheduler::property::kBlockReplaced;
        result->data                  = Graph::serializeBlock(newBlockRaw);

        (*result->data)["replacedBlockUniqueName"s] = uniqueName;

        return result;
    }
};

template<ExecutionPolicy execution = ExecutionPolicy::singleThreaded, profiling::ProfilerLike TProfiler = profiling::null::Profiler>
class Simple : public SchedulerBase<Simple<execution, TProfiler>, execution, TProfiler> {
    using Description = Doc<R""(Simple loop based Scheduler, which iterates over all blocks in the order they have beein defined and emplaced definition in the graph.)"">;

    friend class lifecycle::StateMachine<Simple<execution, TProfiler>>;
    friend class SchedulerBase<Simple<execution, TProfiler>, execution, TProfiler>;

public:
    using base_t = SchedulerBase<Simple<execution, TProfiler>, execution, TProfiler>;

    explicit Simple(gr::Graph&& graph, std::string_view defaultThreadPool = gr::thread_pool::kDefaultCpuPoolId, const profiling::Options& profiling_options = {}) : base_t(std::move(graph), defaultThreadPool, profiling_options) {}

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

        std::size_t blockCount = 0UZ;
        this->forAllUnmanagedBlocks([&blockCount](auto&& /*block*/) { blockCount++; });
        std::vector<BlockModel*> allBlocks;
        allBlocks.reserve(blockCount);
        this->forAllUnmanagedBlocks([&allBlocks](auto&& block) { allBlocks.push_back(block.get()); });

        this->_adoptionBlocks.clear();
        this->_adoptionBlocks.resize(n_batches);
        this->_jobLists->clear();
        this->_jobLists->reserve(n_batches);
        for (std::size_t i = 0; i < n_batches; i++) {
            // create job-set for thread
            auto& job = this->_jobLists->emplace_back(std::vector<BlockModel*>());
            job.reserve(allBlocks.size() / n_batches + 1);
            for (std::size_t j = i; j < allBlocks.size(); j += n_batches) {
                job.push_back(allBlocks[j]);
            }
        }
    }

    void reset() {
        base_t::reset();
        init();
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

    explicit BreadthFirst(gr::Graph&& graph, std::string_view defaultThreadPool = gr::thread_pool::kDefaultCpuPoolId, const profiling::Options& profiling_options = {}) : base_t(std::move(graph), defaultThreadPool, profiling_options) {}

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

        this->_adoptionBlocks.clear();
        this->_adoptionBlocks.resize(n_batches);
        std::lock_guard lock(base_t::_jobListsMutex);
        this->_jobLists->clear();
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

    void reset() {
        base_t::reset();
        init();
    }
};
} // namespace gr::scheduler

#endif // GNURADIO_SCHEDULER_HPP
