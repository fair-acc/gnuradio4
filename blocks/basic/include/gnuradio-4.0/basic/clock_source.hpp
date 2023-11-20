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

#include <gnuradio-4.0/testing/tag_monitors.hpp>

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
    std::thread                                                                  userProvidedThread;

    ~ClockSource() { stopThread(); }

    [[maybe_unused]] bool
    tryStartThread() {
        if constexpr (useIoThread) {
            return false;
        }
        if (bool expectedThreadState = false; this->ioThreadShallRun.compare_exchange_strong(expectedThreadState, true, std::memory_order_acq_rel)) {
            // mocks re-using a user-provided thread
            fmt::print("mocking a user-provided io-Thread for {}\n", this->name);
            std::atomic_store_explicit(&this->ioThreadShallRun, true, std::memory_order_release);
            this->ioThreadShallRun.notify_all();
            userProvidedThread = std::thread([this]() {
                fmt::print("started user-provided thread\n");
                for (int retry = 2; this->ioThreadShallRun.load() && retry > 0; --retry) {
                    while (this->ioThreadShallRun.load()) {
                        std::this_thread::sleep_until(nextTimePoint);
                        // invoke and execute work function from user-provided thread
                        if (this->invokeWork() == work::Status::DONE) {
                            break;
                        } else {
                            retry = 2;
                        }
                    }
                    // delayed shut-down in case there are more tasks to be processed
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                this->stopThread();
                fmt::print("stopped user-provided thread\n");
            });
            userProvidedThread.detach();
            return true;
        }
        return false;
    }

    void
    stopThread() {
        std::atomic_store_explicit(&this->ioThreadShallRun, false, std::memory_order_release);
        this->ioThreadShallRun.notify_all();
    }

    void
    settingsChanged(const property_map & /*old_settings*/, const property_map & /*new_settings*/) {
        nextTimePoint = ClockSourceType::now();
    }

    work::Status
    processBulk(PublishableSpan auto &output) noexcept {
        if (n_samples_max > 0 && n_samples_produced >= n_samples_max) {
            output.publish(0_UZ);
            return work::Status::DONE;
        }

        if constexpr (useIoThread) { // using scheduler-graph provided user thread
            // sleep until next update period -- the following call blocks
            std::this_thread::sleep_until(nextTimePoint);
        }
        const auto writableSamples = static_cast<std::uint32_t>(output.size());
        if (writableSamples < chunk_size) {
            output.publish(0_UZ);
            return work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }

        const std::uint32_t remaining_samples = n_samples_max - n_samples_produced;
        const std::uint32_t limit             = std::min(writableSamples, remaining_samples);
        const std::uint32_t n_available       = std::min(limit, chunk_size.value);

        std::uint32_t samples_to_produce = n_available;
        while (next_tag < tags.size() && tags[next_tag].index <= static_cast<std::make_signed_t<std::size_t>>(n_samples_produced + n_available)) {
            gr::testing::print_tag(tags[next_tag], fmt::format("{}::processBulk(...)\t publish tag at  {:6}", this->name, n_samples_produced));
            Tag &out_tag       = this->output_tags()[0];
            out_tag            = tags[next_tag];
            out_tag.index      = tags[next_tag].index - static_cast<std::make_signed_t<std::size_t>>(n_samples_produced);
            samples_to_produce = static_cast<std::uint32_t>(tags[next_tag].index) - n_samples_produced;
            this->forwardTags();
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
};

} // namespace gr::basic

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, bool useIoThread, typename ClockSourceType), (gr::basic::ClockSource<T, useIoThread, ClockSourceType>), out, n_samples_max, chunk_size, sample_rate);

namespace gr::basic {
static_assert(gr::HasProcessBulkFunction<ClockSource<float>>);
} // namespace gr::basic

#endif // GNURADIO_CLOCK_SOURCE_HPP
