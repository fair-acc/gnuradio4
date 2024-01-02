#ifndef GNURADIO_CLOCK_SOURCE_HPP
#define GNURADIO_CLOCK_SOURCE_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <optional>
#include <random>
#include <thread>
#include <queue>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/reflection.hpp>
#include <gnuradio-4.0/Tag.hpp>

#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace gr::basic {

// optional shortening
template<typename T, gr::meta::fixed_string description = "", typename... Arguments>
using A = gr::Annotated<T, description, Arguments...>;
using namespace gr;

template<typename T, bool useIoThread = true, typename ClockSourceType = std::chrono::system_clock, bool basicPeriodAlgorithm = true>
struct ClockSource : public gr::Block<ClockSource<T, useIoThread, ClockSourceType>, BlockingIO<useIoThread>, Doc<R""(
ClockSource Documentation -- add here
)"">> {
    std::chrono::time_point<ClockSourceType> nextTimePoint = ClockSourceType::now();
    //
    PortOut<T>       out;
    std::vector<Tag> tags{};
    std::size_t      next_tag{ 0 };
    //
    A<std::uint32_t, "n_samples_max", Visible, Doc<"0: unlimited">>              n_samples_max = 1024;
    std::uint32_t                                                                n_samples_produced{ 0 };
    A<float, "avg. sample rate", Visible>                                        sample_rate = 1000.f;
    A<std::uint32_t, "chunk_size", Visible, Doc<"number of samples per update">> chunk_size  = 100;
    std::shared_ptr<std::thread>                                                 userProvidedThread;
    bool                                                                         verbose_console = true;

    void
    start() {
        if (verbose_console) {
            fmt::println("starting {}", this->name);
        }
        n_samples_produced = 0U;
        tryStartThread();
        if (verbose_console) {
            fmt::println("started {}", this->name);
        }
    }

    void
    stop() {
        if (verbose_console) {
            fmt::println("stop {}", this->name);
        }
        this->requestStop();
        if constexpr (!useIoThread) {
            if (verbose_console) {
                fmt::println("joining user-provided {}joinable thread in block {}", userProvidedThread->joinable() ? "" : "non-", this->name);
            }
            userProvidedThread->join();
        }
    }

    void
    settingsChanged(const property_map & /*old_settings*/, const property_map & /*new_settings*/) {
        nextTimePoint = ClockSourceType::now();
    }

    work::Status
    processBulk(PublishableSpan auto &output) noexcept {
        if (n_samples_max > 0 && n_samples_produced >= n_samples_max) {
            output.publish(0UZ);
            return work::Status::DONE;
        }

        if constexpr (useIoThread) { // using scheduler-graph provided user thread
            // sleep until next update period -- the following call blocks
            std::this_thread::sleep_until(nextTimePoint);
        }
        const auto writableSamples = static_cast<std::uint32_t>(output.size());
        if (writableSamples < chunk_size) {
            output.publish(0UZ);
            return work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }

        const std::uint32_t remaining_samples = n_samples_max - n_samples_produced;
        const std::uint32_t limit             = std::min(writableSamples, remaining_samples);
        const std::uint32_t n_available       = std::min(limit, chunk_size.value);

        std::uint32_t samples_to_produce = n_available;
        while (next_tag < tags.size() && tags[next_tag].index <= static_cast<std::make_signed_t<std::size_t>>(n_samples_produced + n_available)) {
            const auto tagDeltaIndex = tags[next_tag].index - static_cast<Tag::signed_index_type>(n_samples_produced); // position w.r.t. start of this chunk
            if (verbose_console) {
                gr::testing::print_tag(tags[next_tag], fmt::format("{}::processBulk(...)\t publish tag at  {:6}", this->name, n_samples_produced + tagDeltaIndex));
            }
            out.publishTag(tags[next_tag].map, tagDeltaIndex);
            samples_to_produce = static_cast<std::uint32_t>(tags[next_tag].index) - n_samples_produced;
            next_tag++;
        }
        samples_to_produce = std::min(samples_to_produce, n_samples_max.value);

        output.publish(static_cast<std::size_t>(samples_to_produce));
        n_samples_produced += samples_to_produce;

        if constexpr (basicPeriodAlgorithm) {
            const auto updatePeriod = std::chrono::microseconds(static_cast<long>(1e6f * static_cast<float>(chunk_size) / sample_rate));
            nextTimePoint += updatePeriod;
        } else {
            const auto updatePeriod = std::chrono::microseconds(static_cast<long>(1e6f * static_cast<float>(chunk_size) / sample_rate));
            // verify the actual rate
            const auto actual_elapsed_time   = std::chrono::duration_cast<std::chrono::microseconds>(ClockSourceType::now() - nextTimePoint + updatePeriod).count();
            const auto expected_elapsed_time = 1e6f * static_cast<float>(chunk_size) / sample_rate;

            // adjust the next update period
            const float ratio = static_cast<float>(actual_elapsed_time) / expected_elapsed_time;
            nextTimePoint += std::chrono::microseconds(static_cast<long>(static_cast<float>(updatePeriod.count()) * ratio));
        }

        return work::Status::OK;
    }

private:
    [[maybe_unused]] bool
    tryStartThread() {
        if constexpr (useIoThread) {
            return false; // use Block<T>::work generated thread
        }
        if (verbose_console) {
            fmt::println("initial ClockSource state: {}", magic_enum::enum_name(this->state.load()));
        }
        if (lifecycle::State expectedThreadState = lifecycle::State::INITIALISED; this->state.compare_exchange_strong(expectedThreadState, lifecycle::State::RUNNING, std::memory_order_acq_rel)) {
            // mocks re-using a user-provided thread
            if (verbose_console) {
                fmt::println("mocking a user-provided io-Thread for {}", this->name);
            }
            this->state.notify_all();
            auto createManagedThread = [](auto &&threadFunction, auto &&threadDeleter) {
                return std::shared_ptr<std::thread>(new std::thread(std::forward<decltype(threadFunction)>(threadFunction)), std::forward<decltype(threadDeleter)>(threadDeleter));
            };
            userProvidedThread = createManagedThread(
                    [this]() {
                        if (verbose_console) {
                            fmt::println("started user-provided thread");
                        }
                        lifecycle::State actualThreadState = this->state.load();
                        while (lifecycle::isActive(actualThreadState)) {
                            std::this_thread::sleep_until(nextTimePoint);
                            // invoke and execute work function from user-provided thread
                            const work::Status status = this->invokeWork();
                            if (status == work::Status::DONE) {
                                std::atomic_store_explicit(&this->state, lifecycle::State::REQUESTED_STOP, std::memory_order_release);
                                this->state.notify_all();
                                break;
                            }
                            actualThreadState = this->state.load();
                            this->ioLastWorkStatus.exchange(status, std::memory_order_relaxed);
                        }

                        if (verbose_console) {
                            fmt::println("stopped user-provided thread - state: {}", magic_enum::enum_name(this->state.load()));
                        }
                        std::atomic_store_explicit(&this->state, lifecycle::State::STOPPED, std::memory_order_release);
                        this->state.notify_all();
                    },
                    [this](std::thread *t) {
                        std::atomic_store_explicit(&this->state, lifecycle::State::STOPPED, std::memory_order_release);
                        this->state.notify_all();
                        if (t->joinable()) {
                            t->join();
                        }
                        delete t;
                        fmt::println("user-provided thread deleted");
                    });
            if (verbose_console) {
                fmt::println("launched user-provided thread");
            }
            return true;
        }
        return false;
    }
};

} // namespace gr::basic

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, bool useIoThread, typename ClockSourceType), (gr::basic::ClockSource<T, useIoThread, ClockSourceType>), out, n_samples_max, chunk_size, sample_rate,
                                    verbose_console);

namespace gr::basic {
static_assert(gr::HasProcessBulkFunction<ClockSource<float>>);
} // namespace gr::basic

#endif // GNURADIO_CLOCK_SOURCE_HPP
