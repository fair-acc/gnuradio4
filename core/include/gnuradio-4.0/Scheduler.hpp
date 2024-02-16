#ifndef GNURADIO_SCHEDULER_HPP
#define GNURADIO_SCHEDULER_HPP

#include <chrono>
#include <set>
#include <source_location>
#include <thread>
#include <utility>
#include <queue>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/LifeCycle.hpp>
#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Profiler.hpp>
#include <gnuradio-4.0/thread/thread_pool.hpp>

namespace gr::scheduler {
using gr::thread_pool::BasicThreadPool;
using namespace gr::message;
using namespace std::string_literals;

enum ExecutionPolicy { singleThreaded, multiThreaded };

constexpr std::chrono::milliseconds kMessagePollInterval{ 10 };

template<typename Derived, ExecutionPolicy execution = ExecutionPolicy::singleThreaded, profiling::ProfilerLike TProfiler = profiling::null::Profiler>
class SchedulerBase : public lifecycle::StateMachine<Derived> {
    friend class lifecycle::StateMachine<Derived>;

protected:
    gr::Graph                              _graph;
    TProfiler                              _profiler;
    decltype(_profiler.forThisThread())    _profiler_handler;
    std::shared_ptr<BasicThreadPool>       _pool;
    std::atomic_size_t                     _running_jobs;
    std::atomic_bool                       _stop_requested;
    std::vector<std::vector<BlockModel *>> _job_lists;

    MsgPortOutNamed<"__ForChildren"> _toChildMessagePort;
    MsgPortInNamed<"__FromChildren"> _fromChildMessagePort;

    std::string unique_name = gr::meta::type_name<Derived>(); // TODO: to be replaced if scheduler derives from Block<T>

public:
    MsgPortInNamed<"__Builtin">  msgIn;
    MsgPortOutNamed<"__Builtin"> msgOut;

    [[nodiscard]] static constexpr auto
    executionPolicy() {
        return execution;
    }

    explicit SchedulerBase(gr::Graph &&graph, std::shared_ptr<BasicThreadPool> thread_pool = std::make_shared<BasicThreadPool>("simple-scheduler-pool", thread_pool::CPU_BOUND),
                           const profiling::Options &profiling_options = {})
        : _graph(std::move(graph)), _profiler{ profiling_options }, _profiler_handler{ _profiler.forThisThread() }, _pool(std::move(thread_pool)) {}

    ~SchedulerBase() { std::ignore = this->changeStateTo(lifecycle::State::REQUESTED_STOP); }

    [[nodiscard]] bool
    isProcessing() const
        requires(executionPolicy() == multiThreaded)
    {
        return _running_jobs.load() > 0;
    }

    void
    stateChanged(lifecycle::State newState) {
        const auto stateStr = [](lifecycle::State s) {
            using enum lifecycle::State;
            using namespace message::scheduler::update;
            switch (s) {
            case RUNNING: return Started;
            case REQUESTED_PAUSE: return Pausing;
            case PAUSED: return Paused;
            case REQUESTED_STOP: return Stopping;
            case STOPPED: return Stopped;
            case ERROR: return Error;
            default: return std::string{};
            }
        }(newState);

        if (!stateStr.empty()) {
            emitMessage(msgOut, { { kind::SchedulerUpdate, stateStr } });
        }
    }

    void
    connectBlockMessagePorts() {
        _graph.forEachBlock([this](auto &block) {
            if (ConnectionResult::SUCCESS != _toChildMessagePort.connect(*block.msgIn)) {
                emitError(msgOut, lifecycle::ErrorType(fmt::format("Failed to connect scheduler input message port to child '{}'", block.uniqueName()), std::source_location::current()));
            }

            auto buffer = _fromChildMessagePort.buffer();
            block.msgOut->setBuffer(buffer.streamBuffer, buffer.tagBuffer);
        });
    }

    void
    processScheduledMessages() {
        auto &msgInReader = msgIn.streamReader();
        if (const auto available = msgInReader.available(); available > 0) {
            const auto &input = msgInReader.get(available);
            for (const auto &msg : input) {
                const auto                                kind = std::get<std::string>(msg.at(key::Kind));
                std::expected<void, lifecycle::ErrorType> e;
                if (kind == gr::message::scheduler::command::Start || kind == gr::message::scheduler::command::Resume) {
                    e = this->changeStateTo(lifecycle::State::RUNNING);
                } else if (kind == gr::message::scheduler::command::Stop) {
                    e = this->changeStateTo(lifecycle::State::REQUESTED_STOP);
                } else if (kind == gr::message::scheduler::command::Pause) {
                    e = this->changeStateTo(lifecycle::State::REQUESTED_PAUSE);
                } else {
                    _toChildMessagePort.streamWriter().publish([&](auto &output) { output[0] = msg; });
                }
                if (!e) {
                    emitError(msgOut, e.error());
                }
            }

            if (!msgInReader.consume(available)) {
                emitError(msgOut, lifecycle::ErrorType("Failed to consume messages from msgIn port", std::source_location::current()));
            }
        }

        auto &fromChildReader = _fromChildMessagePort.streamReader();
        if (const auto available = fromChildReader.available(); available > 0) {
            const auto &input = fromChildReader.get(available);
            msgOut.streamWriter().publish([&](auto &output) { std::ranges::copy(input, output.begin()); }, available);
            if (!fromChildReader.consume(available)) {
                emitError(msgOut, lifecycle::ErrorType("Failed to consume messages from child message port", std::source_location::current()));
            }
        }

        // Process messages in the graph
        _graph.processScheduledMessages();
        if (_running_jobs.load() == 0) {
            _graph.forEachBlock(&BlockModel::processScheduledMessages);
        }
    }

    [[nodiscard]] constexpr auto &
    graph() {
        return _graph;
    }

    std::expected<void, std::string>
    runAndWait() {
        [[maybe_unused]] const auto pe = this->_profiler_handler.startCompleteEvent("scheduler_base.runAndWait");
        if (this->state() == lifecycle::State::IDLE) {
            if (auto e = this->changeStateTo(lifecycle::State::INITIALISED); !e) {
                emitError(msgOut, e.error());
                return std::unexpected(e.error().message);
            }
        }
        if (auto e = this->changeStateTo(lifecycle::State::RUNNING); !e) {
            emitError(msgOut, e.error());
            return std::unexpected(e.error().message);
        }

        if constexpr (executionPolicy() == ExecutionPolicy::multiThreaded) {
            waitDone();

            if (this->state() == lifecycle::State::RUNNING) {
                if (auto e = this->changeStateTo(lifecycle::State::REQUESTED_STOP); !e) {
                    emitError(msgOut, e.error());
                    return std::unexpected(e.error().message);
                }
            }
        }
        return {};
    }

    void
    waitDone() {
        [[maybe_unused]] const auto pe = _profiler_handler.startCompleteEvent("scheduler_base.waitDone");
        for (auto running = _running_jobs.load(); this->state() == lifecycle::REQUESTED_PAUSE || this->state() == lifecycle::PAUSED || running > 0ul; running = _running_jobs.load()) {
            std::this_thread::sleep_for(kMessagePollInterval);
            processScheduledMessages();
        }
        processScheduledMessages();
    }

    [[nodiscard]] const std::vector<std::vector<BlockModel *>> &
    jobs() const {
        return _job_lists;
    }

protected:
    void
    emitMessage(auto &port, Message message) { // TODO: to be replaced if scheduler derives from Block<T>
        message[key::Sender] = unique_name;
        port.streamWriter().publish([&](auto &out) { out[0] = std::move(message); }, 1);
    }

    void
    emitError(auto &port, const lifecycle::ErrorType &error) {
        emitMessage(port, { { key::Kind, kind::Error }, { key::ErrorInfo, error.message }, { key::Location, error.srcLoc() } });
    }

    void
    setBlocksState(lifecycle::State state) {
        _graph.forEachBlock([state, this](auto &block) {
            if (auto e = block.changeState(state); !e) {
                emitError(msgOut, e.error());
            }
        });
    }

    template<typename block_type>
    work::Result
    workOnce(const std::span<block_type> &blocks) {
        constexpr std::size_t requestedWorkAllBlocks = std::numeric_limits<std::size_t>::max();
        std::size_t           performedWorkAllBlocks = 0UZ;
        bool                  something_happened     = false;
        for (auto &currentBlock : blocks) {
            currentBlock->processScheduledMessages();
            const auto [requested_work, performed_work, status] = currentBlock->work(requestedWorkAllBlocks);
            performedWorkAllBlocks += performed_work;
            if (status == work::Status::ERROR) {
                return { requested_work, performed_work, work::Status::ERROR };
            } else if (status != work::Status::DONE) {
                something_happened = true;
            }
        }

        return { requestedWorkAllBlocks, performedWorkAllBlocks, something_happened ? work::Status::OK : work::Status::DONE };
    }

    void
    init() {
        [[maybe_unused]] const auto pe     = _profiler_handler.startCompleteEvent("scheduler_base.init");
        const auto                  result = _graph.performConnections();
        if (!result) {
            emitError(msgOut, lifecycle::ErrorType("Failed to connect blocks in graph", std::source_location::current()));
        }
        connectBlockMessagePorts();
    }

private:
    void
    stop() {
        _stop_requested = true;
        waitJobsDone();
        _graph.forEachBlock([this](auto &block) {
            if (auto e = block.changeState(lifecycle::State::REQUESTED_STOP); !e) {
                emitError(msgOut, e.error());
            }
            if (!block.isBlocking()) { // N.B. no other thread/constraint to consider before shutting down
                if (auto e = block.changeState(lifecycle::State::STOPPED); !e) {
                    emitError(msgOut, e.error());
                }
            }
        });
        if (auto e = this->changeStateTo(lifecycle::State::STOPPED); !e) {
            emitError(msgOut, e.error());
        }
    }

    void
    pause() {
        _stop_requested = true;
        waitJobsDone();
        _graph.forEachBlock([this](auto &block) {
            if (auto e = block.changeState(lifecycle::State::REQUESTED_PAUSE); !e) {
                emitError(msgOut, e.error());
            }
            if (!block.isBlocking()) { // N.B. no other thread/constraint to consider before shutting down
                if (auto e = block.changeState(lifecycle::State::PAUSED); !e) {
                    emitError(msgOut, e.error());
                }
            }
        });
        if (auto e = this->changeStateTo(lifecycle::State::PAUSED); !e) {
            emitError(msgOut, e.error());
        }
    }

    void
    reset() {
        setBlocksState(lifecycle::INITIALISED);

        // since it is not possible to set up the graph connections a second time, this method leaves the graph in the initialized state with clear buffers.
        // clear buffers
        // std::for_each(_graph.edges().begin(), _graph.edges().end(), [](auto &edge) {
        //
        // });
    }

    void
    resume() {
        _stop_requested = false;
        setBlocksState(lifecycle::RUNNING);
        if (executionPolicy() == ExecutionPolicy::multiThreaded) {
            // TODO review _job_lists usage
            this->runOnPool(_job_lists, [this](auto &job) { return this->workOnce(job); });
        }
    }

    void
    start() {
        // TODO merge with resume?
        _stop_requested = false;
        setBlocksState(lifecycle::RUNNING);
        if constexpr (executionPolicy() == singleThreaded) {
            self().runSingleThreaded();
        } else {
            runOnPool(_job_lists, [this](auto &job) { return this->workOnce(job); });
        }
    }

    void
    waitJobsDone() {
        for (auto running = _running_jobs.load(); running > 0ul; running = _running_jobs.load()) {
            _running_jobs.wait(running);
        }
    }

    void
    runOnPool(const std::vector<std::vector<BlockModel *>> &jobs, const std::function<work::Result(const std::span<BlockModel *const> &)> work_function) {
        [[maybe_unused]] const auto pe = _profiler_handler.startCompleteEvent("scheduler_base.runOnPool");
        _running_jobs                  = jobs.size();
        for (auto &jobset : jobs) {
            _pool->execute([this, &jobset, work_function]() { poolWorker([&work_function, &jobset]() { return work_function(jobset); }); });
        }
    }

    void
    poolWorker(const std::function<work::Result()> &work) {
        auto &profiler_handler   = _profiler.forThisThread();
        bool  something_happened = true;
        while (something_happened && !_stop_requested) {
            auto                   pe         = profiler_handler.startCompleteEvent("scheduler_base.work");
            const gr::work::Result workResult = work();
            pe.finish();

            something_happened = workResult.status == work::Status::OK;
        }
        _running_jobs.fetch_sub(1);
        _running_jobs.notify_all();
    }

    [[nodiscard]] constexpr auto &
    self() noexcept {
        return *static_cast<Derived *>(this);
    }

    [[nodiscard]] constexpr const auto &
    self() const noexcept {
        return *static_cast<const Derived *>(this);
    }
};

/**
 * Trivial loop based scheduler, which iterates over all blocks in definition order in the graph until no node did any processing
 */
template<ExecutionPolicy execution = ExecutionPolicy::singleThreaded, profiling::ProfilerLike TProfiler = profiling::null::Profiler>
class Simple : public SchedulerBase<Simple<execution, TProfiler>, execution, TProfiler> {
    friend class lifecycle::StateMachine<Simple<execution, TProfiler>>;
    friend class SchedulerBase<Simple<execution, TProfiler>, execution, TProfiler>;
    static_assert(execution == ExecutionPolicy::singleThreaded || execution == ExecutionPolicy::multiThreaded, "Unsupported execution policy");
    using base_t = SchedulerBase<Simple<execution, TProfiler>, execution, TProfiler>;

public:
    explicit Simple(gr::Graph &&graph, std::shared_ptr<BasicThreadPool> thread_pool = std::make_shared<BasicThreadPool>("simple-scheduler-pool", thread_pool::CPU_BOUND),
                    const profiling::Options &profiling_options = {})
        : base_t(std::move(graph), thread_pool, profiling_options) {}

private:
    void
    init() {
        base_t::init();
        [[maybe_unused]] const auto pe = this->_profiler_handler.startCompleteEvent("scheduler_simple.init");
        // generate job list
        if constexpr (base_t::executionPolicy() == ExecutionPolicy::multiThreaded) {
            const auto n_batches = std::min(static_cast<std::size_t>(this->_pool->maxThreads()), this->_graph.blocks().size());
            this->_job_lists.reserve(n_batches);
            for (std::size_t i = 0; i < n_batches; i++) {
                // create job-set for thread
                auto &job = this->_job_lists.emplace_back();
                job.reserve(this->_graph.blocks().size() / n_batches + 1);
                for (std::size_t j = i; j < this->_graph.blocks().size(); j += n_batches) {
                    job.push_back(this->_graph.blocks()[j].get());
                }
            }
        }
    }

    void
    runSingleThreaded()
        requires(base_t::executionPolicy() == ExecutionPolicy::singleThreaded)
    {
        work::Result result;
        auto         blocklist = std::span{ this->_graph.blocks() };
        do {
            this->processScheduledMessages();
            if (this->state() == lifecycle::State::RUNNING) {
                result = this->workOnce(blocklist);
                if (result.status == work::Status::ERROR) {
                    if (auto e = this->changeStateTo(lifecycle::State::ERROR); !e) {
                        this->emitError(this->msgOut, e.error());
                    }
                }
            } else {
                std::this_thread::sleep_for(kMessagePollInterval);
                result = { 0, 0, work::Status::OK };
            }
        } while (result.status == work::Status::OK && this->state() != lifecycle::State::ERROR && this->state() != lifecycle::State::STOPPED);
        if (this->state() == lifecycle::State::RUNNING) {
            if (auto e = this->changeStateTo(lifecycle::State::REQUESTED_STOP); !e) {
                this->emitError(this->msgOut, e.error());
            }
        }
    }
};

/**
 * Breadth first traversal scheduler which traverses the graph starting from the source blocks in a breath first fashion
 * detecting cycles and blocks which can be reached from several source blocks.
 */
template<ExecutionPolicy execution = ExecutionPolicy::singleThreaded, profiling::ProfilerLike TProfiler = profiling::null::Profiler>
class BreadthFirst : public SchedulerBase<BreadthFirst<execution, TProfiler>, execution, TProfiler> {
    friend class lifecycle::StateMachine<BreadthFirst<execution, TProfiler>>;
    friend class SchedulerBase<BreadthFirst<execution, TProfiler>, execution, TProfiler>;
    static_assert(execution == ExecutionPolicy::singleThreaded || execution == ExecutionPolicy::multiThreaded, "Unsupported execution policy");
    using base_t = SchedulerBase<BreadthFirst<execution, TProfiler>, execution, TProfiler>;
    std::vector<BlockModel *> _blocklist;

public:
    explicit BreadthFirst(gr::Graph &&graph, std::shared_ptr<BasicThreadPool> thread_pool = std::make_shared<BasicThreadPool>("breadth-first-pool", thread_pool::CPU_BOUND),
                          const profiling::Options &profiling_options = {})
        : base_t(std::move(graph), thread_pool, profiling_options) {}

private:
    void
    init() {
        [[maybe_unused]] const auto pe = this->_profiler_handler.startCompleteEvent("breadth_first.init");
        using block_t                  = BlockModel *;
        base_t::init();
        // calculate adjacency list
        std::map<block_t, std::vector<block_t>> _adjacency_list{};
        std::vector<block_t>                    _source_blocks{};
        // compute the adjacency list
        std::set<block_t> block_reached;
        for (auto &e : this->_graph.edges()) {
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
                for (auto &dst : _adjacency_list.at(currentBlock)) {
                    if (!reached.contains(dst)) { // detect cycles. this could be removed if we guarantee cycle free graphs earlier
                        queue.push(dst);
                        reached.insert(dst);
                    }
                }
            }
        }
        // generate job list
        if constexpr (base_t::executionPolicy() == ExecutionPolicy::multiThreaded) {
            const auto n_batches = std::min(static_cast<std::size_t>(this->_pool->maxThreads()), _blocklist.size());
            this->_job_lists.reserve(n_batches);
            for (std::size_t i = 0; i < n_batches; i++) {
                // create job-set for thread
                auto &job = this->_job_lists.emplace_back();
                job.reserve(_blocklist.size() / n_batches + 1);
                for (std::size_t j = i; j < _blocklist.size(); j += n_batches) {
                    job.push_back(_blocklist[j]);
                }
            }
        }
    }

    void
    runSingleThreaded()
        requires(base_t::executionPolicy() == ExecutionPolicy::singleThreaded)
    {
        work::Result result;
        auto         blocklist = std::span{ this->_blocklist };
        do {
            this->processScheduledMessages();
            if (this->state() == lifecycle::State::RUNNING) {
                result = this->workOnce(blocklist);
                if (result.status == work::Status::ERROR) {
                    if (auto e = this->changeStateTo(lifecycle::State::ERROR); !e) {
                        this->emitError(this->msgOut, e.error());
                    }
                }
            } else {
                std::this_thread::sleep_for(kMessagePollInterval);
                result = { 0, 0, work::Status::OK };
            }
        } while (result.status == work::Status::OK && this->state() != lifecycle::State::ERROR && this->state() != lifecycle::State::STOPPED);
        if (this->state() == lifecycle::State::RUNNING) {
            if (auto e = this->changeStateTo(lifecycle::State::REQUESTED_STOP); !e) {
                this->emitError(this->msgOut, e.error());
            }
        }
    }
};
} // namespace gr::scheduler

#endif // GNURADIO_SCHEDULER_HPP
