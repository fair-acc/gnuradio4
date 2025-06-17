#ifndef GNURADIO_SCHEDULER_MODEL_HPP
#define GNURADIO_SCHEDULER_MODEL_HPP

#include <gnuradio-4.0/BlockModel.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <charconv>

namespace gr {

class SchedulerModel {
public:
    SchedulerModel() = default;

public:
    SchedulerModel(const SchedulerModel&)             = delete;
    SchedulerModel& operator=(const SchedulerModel&)  = delete;
    SchedulerModel(SchedulerModel&& other)            = delete;
    SchedulerModel& operator=(SchedulerModel&& other) = delete;

    virtual ~SchedulerModel() = default;

    virtual std::expected<void, gr::Error> start()  = 0;
    virtual std::expected<void, gr::Error> stop()   = 0;
    virtual std::expected<void, gr::Error> pause()  = 0;
    virtual std::expected<void, gr::Error> resume() = 0;

    virtual std::expected<gr::Graph, gr::Error> extractGraph() = 0;
};

template<BlockLike TScheduler> // TODO: SchedulerLike and GraphLike?
class SchedulerWrapper : public GraphWrapper<TScheduler, gr::Graph>, public SchedulerModel {
    static_assert(std::is_same_v<TScheduler, std::remove_reference_t<TScheduler>>);

public:
    explicit SchedulerWrapper(gr::Graph&&           graph,                                                                                                                                                         //
        std::shared_ptr<scheduler::BasicThreadPool> thread_pool = std::make_shared<scheduler::BasicThreadPool>("simple-scheduler-pool", thread_pool::CPU_BOUND), const profiling::Options& profiling_options = {}) //
        : GraphWrapper<TScheduler, gr::Graph>(std::move(graph), thread_pool, profiling_options) {
        this->_graph = std::addressof(this->_block.graph());
    }

    SchedulerWrapper(const SchedulerWrapper& other)            = delete;
    SchedulerWrapper(SchedulerWrapper&& other)                 = delete;
    SchedulerWrapper& operator=(const SchedulerWrapper& other) = delete;
    SchedulerWrapper& operator=(SchedulerWrapper&& other)      = delete;

    ~SchedulerWrapper() override = default;

    std::expected<void, gr::Error> start() final { return this->blockRef().changeStateTo(gr::lifecycle::State::RUNNING); }
    std::expected<void, gr::Error> stop() final { return this->blockRef().changeStateTo(gr::lifecycle::State::REQUESTED_STOP); }
    std::expected<void, gr::Error> pause() final { return this->blockRef().changeStateTo(gr::lifecycle::State::REQUESTED_PAUSE); }
    std::expected<void, gr::Error> resume() final { return this->blockRef().changeStateTo(gr::lifecycle::State::RUNNING); }

    virtual std::expected<gr::Graph, gr::Error> extractGraph() override {
        auto stopResult = stop();
        if (!stopResult) {
            return std::unexpected(stopResult.error());
        }

        return std::move(this->blockRef().graph());
    }
};

} // namespace gr

#endif // GNURADIO_BLOCK_SCHEDULER_HPP
