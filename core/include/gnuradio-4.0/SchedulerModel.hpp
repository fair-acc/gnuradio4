#ifndef GNURADIO_SCHEDULER_MODEL_HPP
#define GNURADIO_SCHEDULER_MODEL_HPP

#include <gnuradio-4.0/BlockModel.hpp>
#include <gnuradio-4.0/Graph.hpp>

#include <thread>

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

    virtual void        setGraph(gr::Graph&&) = 0;
    virtual BlockModel* asBlockModel()        = 0;

    static std::shared_ptr<BlockModel> asBlockModelPtr(std::shared_ptr<SchedulerModel> ptr) {
        if (!ptr) {
            return {};
        }

        return std::shared_ptr<BlockModel>(ptr, ptr->asBlockModel());
    }

    virtual void start() = 0;
    virtual void stop()  = 0;

    virtual void requestWorkQuiescence() = 0;
    virtual void releaseWorkQuiescence() = 0;
};

template<BlockLike TScheduler>
class SchedulerWrapper : public GraphWrapper<TScheduler, gr::Graph>, public SchedulerModel {
    static_assert(std::is_same_v<TScheduler, std::remove_reference_t<TScheduler>>);

public:
    explicit SchedulerWrapper(const gr::property_map& props = {}) //
        : GraphWrapper<TScheduler, gr::Graph>(props) {}

    SchedulerWrapper(const SchedulerWrapper& other)            = delete;
    SchedulerWrapper(SchedulerWrapper&& other)                 = delete;
    SchedulerWrapper& operator=(const SchedulerWrapper& other) = delete;
    SchedulerWrapper& operator=(SchedulerWrapper&& other)      = delete;

    ~SchedulerWrapper() override = default;

    void setGraph(gr::Graph&& graph) final { std::ignore = this->blockRef().exchange(std::move(graph)); }

    BlockModel* asBlockModel() final { return static_cast<BlockModel*>(this); }

    void start() override {
        auto& sched = this->blockRef();

        if (sched.state() == gr::lifecycle::State::IDLE) {
            std::ignore = sched.changeStateTo(gr::lifecycle::State::INITIALISED);
        }

        if (sched.state() != gr::lifecycle::State::INITIALISED) {
            return;
        }

        _schedulerThread = std::thread([&sched] {
            // this will invoke scheduler's start(), which blocks
            if (!sched.changeStateTo(gr::lifecycle::State::RUNNING)) {
                std::ignore = sched.changeStateTo(gr::lifecycle::State::ERROR);
            }
        });
    }

    void requestWorkQuiescence() override { this->blockRef().requestWorkQuiescence(); }
    void releaseWorkQuiescence() override { this->blockRef().releaseWorkQuiescence(); }

    void stop() override {
        if (this->blockRef().changeStateTo(gr::lifecycle::State::REQUESTED_STOP)) {
            // transitions to stopped/error do not fail
            std::ignore = this->blockRef().changeStateTo(gr::lifecycle::State::STOPPED);
        } else {
            std::ignore = this->blockRef().changeStateTo(gr::lifecycle::State::ERROR);
        }

        if (_schedulerThread.joinable()) {
            _schedulerThread.join();
        }
    }

    std::thread _schedulerThread;
};

} // namespace gr

#endif // GNURADIO_BLOCK_SCHEDULER_HPP
