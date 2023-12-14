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

#include <fmt/chrono.h>
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

using ClockSourceDoc = Doc<R""(
ClockSource Documentation -- add here
)"">;

template<typename T, bool useIoThread = true, typename ClockSourceType = std::chrono::system_clock, bool basicPeriodAlgorithm = true>
struct ClockSource : public gr::Block<ClockSource<T, useIoThread, ClockSourceType>, BlockingIO<useIoThread>, ClockSourceDoc> {
    std::chrono::time_point<ClockSourceType> nextTimePoint = ClockSourceType::now();

    PortOut<T> out;

    // Ready-to-use tags set by user
    std::vector<Tag> tags{};
    std::size_t      next_tag{ 0 };

    // Time-string tags
    std::vector<std::uint64_t> tag_times; // time in nanoseconds
    std::vector<std::string>   tag_values;
    std::size_t                next_time_tag{ 0 };
    std::uint64_t              repeat_period{ 0 };          // if repeat_period > last tag_time -> restart tags, in nanoseconds
    bool                       do_zero_order_hold{ false }; // if more tag_times than values: true=publish last tag, false=publish empty

    A<std::uint32_t, "n_samples_max", Visible, Doc<"0: unlimited">>              n_samples_max = 1024;
    std::uint32_t                                                                n_samples_produced{ 0 };
    A<float, "avg. sample rate", Visible>                                        sample_rate = 1000.f;
    A<std::uint32_t, "chunk_size", Visible, Doc<"number of samples per update">> chunk_size  = 100;
    std::shared_ptr<std::thread>                                                 userProvidedThread;
    bool                                                                         verbose_console = true;

private:
    std::chrono::time_point<ClockSourceType> _beginSequenceTimePoint = ClockSourceType::now();
    bool                                     _beginSequenceTimePointInitialized{ false };

public:
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
        // TODO: does one need to check every processBulk call
        bool isAscending = std::ranges::adjacent_find(tag_times, std::greater_equal()) == tag_times.end();
        if (!isAscending) {
            throw std::invalid_argument("The input tag_times vector should be ascending.");
        }

        if (n_samples_max > 0 && n_samples_produced >= n_samples_max) {
            output.publish(0UZ);
            return work::Status::DONE;
        }

        if constexpr (useIoThread) { // using scheduler-graph provided user thread
            // sleep until next update period -- the following call blocks
            std::this_thread::sleep_until(nextTimePoint);
        }

        const std::uint32_t remainingSamples = n_samples_max - n_samples_produced;
        std::uint32_t       samplesToProduce = std::min(remainingSamples, chunk_size.value);

        std::uint32_t samplesToNextTimeTag = std::numeric_limits<uint32_t>::max();
        if (!tag_times.empty()) {
            if (next_time_tag == 0 && !_beginSequenceTimePointInitialized) {
                _beginSequenceTimePoint            = nextTimePoint;
                _beginSequenceTimePointInitialized = true;
            }
            if (next_time_tag >= tag_times.size() && repeat_period >= tag_times.back()) {
                _beginSequenceTimePoint += std::chrono::microseconds(repeat_period / 1000); // ns -> μs
                _beginSequenceTimePointInitialized = true;
                next_time_tag                      = 0;
            }
            if (next_time_tag < tag_times.size()) {
                const auto currentTagTime = std::chrono::microseconds(tag_times[next_time_tag] / 1000); // ns -> μs
                const auto timeToNextTag  = std::chrono::duration_cast<std::chrono::microseconds>((_beginSequenceTimePoint + currentTagTime - nextTimePoint));
                samplesToNextTimeTag      = static_cast<std::uint32_t>((static_cast<double>(timeToNextTag.count()) / 1.e6) * static_cast<double>(sample_rate));
            }
        }

        auto samplesToNextTag = tags.empty() || next_tag >= tags.size() ? std::numeric_limits<uint32_t>::max() : static_cast<std::uint32_t>(tags[next_tag].index) - n_samples_produced;

        if (samplesToNextTag < samplesToNextTimeTag) {
            if (next_tag < tags.size() && samplesToNextTag <= samplesToProduce) {
                const auto tagDeltaIndex = tags[next_tag].index - static_cast<Tag::signed_index_type>(n_samples_produced); // position w.r.t. start of this chunk
                if (verbose_console) {
                    gr::testing::print_tag(tags[next_tag], fmt::format("{}::processBulk(...)\t publish tag at  {:6}", this->name, n_samples_produced + tagDeltaIndex));
                }
                out.publishTag(tags[next_tag].map, tagDeltaIndex);
                samplesToProduce = samplesToNextTag;
                next_tag++;
            }
        } else {
            if (!tag_times.empty() && next_time_tag < tag_times.size() && samplesToNextTimeTag <= samplesToProduce) {
                std::string  value    = next_time_tag < tag_values.size() ? tag_values[next_time_tag] : do_zero_order_hold ? tag_values.back() : "";
                property_map context  = { { "context", value } };
                property_map metaInfo = { { "trigger_meta_info", context } };
                if (verbose_console) {
                    fmt::println("{}::processBulk(...)\t publish tag-time at  {:6}, time:{}ns", this->name, samplesToNextTimeTag, tag_times[next_time_tag]);
                }
                out.publishTag(metaInfo, samplesToNextTimeTag);
                samplesToProduce = samplesToNextTimeTag;
                next_time_tag++;
            }
        }
        samplesToProduce = std::min(samplesToProduce, n_samples_max.value);

        if (static_cast<std::uint32_t>(output.size()) < samplesToProduce) {
            output.publish(0UZ);
            return work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }

        output.publish(static_cast<std::size_t>(samplesToProduce));
        n_samples_produced += samplesToProduce;

        if constexpr (basicPeriodAlgorithm) {
            const auto updatePeriod = std::chrono::microseconds(static_cast<long>(1e6f * static_cast<float>(samplesToProduce) / sample_rate));
            nextTimePoint += updatePeriod;
        } else {
            const auto updatePeriod = std::chrono::microseconds(static_cast<long>(1e6f * static_cast<float>(samplesToProduce) / sample_rate));
            // verify the actual rate
            const auto actual_elapsed_time   = std::chrono::duration_cast<std::chrono::microseconds>(ClockSourceType::now() - nextTimePoint + updatePeriod).count();
            const auto expected_elapsed_time = 1e6f * static_cast<float>(samplesToProduce) / sample_rate;

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
