#ifndef GNURADIO_CLOCK_SOURCE_HPP
#define GNURADIO_CLOCK_SOURCE_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <iostream>
#include <optional>
#include <queue>
#include <random>
#include <thread>

#include <format>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <gnuradio-4.0/TriggerMatcher.hpp>

namespace gr::basic {

// optional shortening
template<typename T, gr::meta::fixed_string description = "", typename... Arguments>
using A = gr::Annotated<T, description, Arguments...>;
using namespace gr;

GR_REGISTER_BLOCK("gr::basic::ClockSource", gr::basic::DefaultClockSource)
GR_REGISTER_BLOCK("gr::basic::ClockSourceNonBlocking", gr::basic::DefaultClockSourceNonBlocking)

template<typename T, bool useIoThread = true, typename ClockSourceType = std::chrono::system_clock, bool basicPeriodAlgorithm = true>
struct ClockSource : Block<ClockSource<T, useIoThread, ClockSourceType>, BlockingIO<useIoThread>> {
    using Description = Doc<R""(A source block that generates clock signals with specified timing intervals.
This block can generate periodic signals based on a system clock and allows for customization of the sample rate and chunk size.
The 'tag_times[ns]:tag_value(string)' vectors control the emission of tags with a single 'context' keys at specified times after the block started.)"">;
    using TimePoint   = std::chrono::time_point<ClockSourceType>;

    PortOut<T> out;

    A<gr::Size_t, "n_samples_max", Visible, Doc<"0: unlimited">>                                                   n_samples_max = 1024;
    gr::Size_t                                                                                                     n_samples_produced{0};
    A<float, "avg. sample rate", Visible>                                                                          sample_rate = 1000.f;
    A<gr::Size_t, "chunk_size", Visible, Doc<"number of samples per update">>                                      chunk_size  = 100;
    A<std::vector<std::uint64_t>, "tag times", Doc<"times when tags should be emitted [ns]">>                      tag_times;
    A<std::vector<std::string>, "tag values", Doc<"list of '<trigger name>/<ctx>' formatted tags">>                tag_values;
    A<std::uint64_t, "repeat period", Visible, Doc<"if repeat_period > last tag_time -> restart tags, in [ns]">>   repeat_period{0U}; //
    A<bool, "perform zero-order-hold", Doc<"if tag_times>tag_values: true=publish last tag, false=publish empty">> do_zero_order_hold{false};
    A<bool, "verbose console">                                                                                     verbose_console = false;

    GR_MAKE_REFLECTABLE(ClockSource, out, tag_times, tag_values, repeat_period, do_zero_order_hold, n_samples_max, chunk_size, sample_rate, verbose_console);

    // Ready-to-use tags set by user
    std::vector<Tag>             tags{};
    std::shared_ptr<std::thread> userProvidedThread;

    TimePoint   _beginSequenceTimePoint = ClockSourceType::now();
    bool        _beginSequenceTimePointInitialized{false};
    TimePoint   _nextTimePoint = ClockSourceType::now();
    std::size_t _nextTimeTag{0};
    std::size_t _nextTagIndex{0};

    void start() {
        if (verbose_console) {
            std::println("starting {}", this->name);
        }
        n_samples_produced = 0U;
        tryStartThread();
        if (verbose_console) {
            std::println("started {}", this->name);
        }
        _nextTimePoint = ClockSourceType::now();
    }

    void stop() {
        if (verbose_console) {
            std::println("stop {}", this->name);
        }
        this->requestStop();
        if constexpr (!useIoThread) {
            if (verbose_console) {
                std::println("joining user-provided {}joinable thread in block {}", userProvidedThread->joinable() ? "" : "non-", this->name);
            }
            userProvidedThread->join();
        }
    }

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        _nextTimePoint = ClockSourceType::now();
        if (newSettings.contains("tag_times")) {
            if (std::ranges::adjacent_find(tag_times.value, std::greater_equal()) != tag_times.value.end()) { // check time being monotonic
                using namespace gr::message;
                throw gr::exception("The input tag_times vector should be ascending.");
            }
        }
        if (newSettings.contains("chunk_size")) {
            if (chunk_size.value < 1) {
                using namespace gr::message;
                throw gr::exception("The number of samples per update (`chunk_size`) must be >= 1.");
            }
        }
    }

    work::Status processBulk(OutputSpanLike auto& outSpan) noexcept {
        if (n_samples_max > 0UZ && n_samples_produced >= n_samples_max) {
            outSpan.publish(0UZ);
            return work::Status::DONE;
        }

        if constexpr (useIoThread) { // using scheduler-graph provided user thread
            // sleep until next update period -- the following call blocks
            std::this_thread::sleep_until(_nextTimePoint);
        }
        const auto now = ClockSourceType::now();

        const gr::Size_t remainingSamples = (n_samples_max > 0UZ) ? n_samples_max - n_samples_produced : std::numeric_limits<gr::Size_t>::max();
        gr::Size_t       samplesToProduce = std::min({remainingSamples, chunk_size.value, static_cast<gr::Size_t>(outSpan.size())});

        gr::Size_t samplesToNextTimeTag = std::numeric_limits<uint32_t>::max();
        if (!tag_times.value.empty()) {
            if (_nextTimeTag == 0 && !_beginSequenceTimePointInitialized) {
                _beginSequenceTimePoint            = _nextTimePoint;
                _beginSequenceTimePointInitialized = true;
            }
            if (_nextTimeTag >= tag_times.value.size() && repeat_period >= tag_times.value.back()) {
                _beginSequenceTimePoint += std::chrono::microseconds(repeat_period / 1000); // ns -> μs
                _beginSequenceTimePointInitialized = true;
                _nextTimeTag                       = 0;
            }
            if (_nextTimeTag < tag_times.value.size()) {
                const auto currentTagTime = std::chrono::microseconds(tag_times.value[_nextTimeTag] / 1000); // ns -> μs
                const auto timeToNextTag  = std::chrono::duration_cast<std::chrono::microseconds>((_beginSequenceTimePoint + currentTagTime - _nextTimePoint));
                samplesToNextTimeTag      = static_cast<gr::Size_t>((static_cast<double>(timeToNextTag.count()) / 1.e6) * static_cast<double>(sample_rate));
            }
        }

        auto samplesToNextTag = tags.empty() || _nextTagIndex >= tags.size() ? std::numeric_limits<uint32_t>::max() : static_cast<gr::Size_t>(tags[_nextTagIndex].index) - n_samples_produced;

        if (samplesToNextTag < samplesToNextTimeTag) {
            if (_nextTagIndex < tags.size() && samplesToNextTag <= samplesToProduce) {
                const auto tagDeltaIndex = tags[_nextTagIndex].index - static_cast<std::size_t>(n_samples_produced); // position w.r.t. start of this chunk
                if (verbose_console) {
                    gr::testing::print_tag(tags[_nextTagIndex], std::format("{}::processBulk(...)\t publish tag at  {:6}", this->name, n_samples_produced + tagDeltaIndex));
                }
                outSpan.publishTag(tags[_nextTagIndex].map, tagDeltaIndex);
                samplesToProduce = samplesToNextTag;
                _nextTagIndex++;
            }
        } else {
            if (!tag_times.value.empty() && _nextTimeTag < tag_times.value.size() && samplesToNextTimeTag <= samplesToProduce) {
                const std::string     value = _nextTimeTag < tag_values.value.size() ? tag_values.value[_nextTimeTag] : do_zero_order_hold ? tag_values.value.back() : "";
                std::string           triggerName;
                [[maybe_unused]] bool triggerNameNegated;
                std::string           triggerContext;
                [[maybe_unused]] bool triggerContextNegated;
                gr::basic::trigger::detail::parse(value, triggerName, triggerNameNegated, triggerContext, triggerContextNegated);
                property_map triggerTag;
                uint64_t     triggerTime = static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
                triggerTime += static_cast<std::uint64_t>(static_cast<float>(samplesToNextTimeTag) * 1e9f / sample_rate);
                triggerTag[tag::TRIGGER_NAME.shortKey()]      = triggerName;
                triggerTag[tag::TRIGGER_TIME.shortKey()]      = triggerTime;
                triggerTag[tag::TRIGGER_OFFSET.shortKey()]    = 0.f;
                triggerTag[tag::CONTEXT.shortKey()]           = triggerContext;
                triggerTag[tag::TRIGGER_META_INFO.shortKey()] = property_map{};
                if (verbose_console) {
                    std::println("{}::processBulk(...)\t publish tag-time at  {:6}, time:{}ns", this->name, samplesToNextTimeTag, tag_times.value[_nextTimeTag]);
                }
                outSpan.publishTag(triggerTag, samplesToNextTimeTag);
                samplesToProduce = samplesToNextTimeTag;
                _nextTimeTag++;
            }
        }

        outSpan.publish(static_cast<std::size_t>(samplesToProduce));
        n_samples_produced += samplesToProduce;

        if constexpr (basicPeriodAlgorithm) {
            const auto updatePeriod = std::chrono::microseconds(static_cast<long>(1e6f * static_cast<float>(samplesToProduce) / sample_rate));
            _nextTimePoint += updatePeriod;
        } else {
            const auto updatePeriod = std::chrono::microseconds(static_cast<long>(1e6f * static_cast<float>(samplesToProduce) / sample_rate));
            // verify the actual rate
            const auto actual_elapsed_time   = std::chrono::duration_cast<std::chrono::microseconds>(ClockSourceType::now() - _nextTimePoint + updatePeriod).count();
            const auto expected_elapsed_time = 1e6f * static_cast<float>(samplesToProduce) / sample_rate;

            // adjust the next update period
            const float ratio = static_cast<float>(actual_elapsed_time) / expected_elapsed_time;
            _nextTimePoint += std::chrono::microseconds(static_cast<long>(static_cast<float>(updatePeriod.count()) * ratio));
        }

        return work::Status::OK;
    }

private:
    [[maybe_unused]] bool tryStartThread() {
        if constexpr (useIoThread) {
            return false; // use Block<T>::work generated thread
        }
        if (verbose_console) {
            std::println("initial ClockSource state: {}", magic_enum::enum_name(this->state()));
        }
        if (lifecycle::State expectedThreadState = lifecycle::State::INITIALISED; this->_state.compare_exchange_strong(expectedThreadState, lifecycle::State::RUNNING, std::memory_order_acq_rel)) {
            // mocks re-using a user-provided thread
            if (verbose_console) {
                std::println("mocking a user-provided io-Thread for {}", this->name);
            }
            this->_state.notify_all();
            auto createManagedThread = [](auto&& threadFunction, auto&& threadDeleter) { return std::shared_ptr<std::thread>(new std::thread(std::forward<decltype(threadFunction)>(threadFunction)), std::forward<decltype(threadDeleter)>(threadDeleter)); };
            userProvidedThread       = createManagedThread(
                [this]() {
                    if (verbose_console) {
                        std::println("started user-provided thread");
                    }
                    lifecycle::State actualThreadState = this->state();
                    while (lifecycle::isActive(actualThreadState)) {
                        std::this_thread::sleep_until(_nextTimePoint);
                        // invoke and execute work function from user-provided thread
                        const work::Status status = this->invokeWork();
                        if (status == work::Status::DONE) {
                            this->requestStop();
                            break;
                        }
                        actualThreadState = this->state();
                        this->ioLastWorkStatus.exchange(status, std::memory_order_relaxed);
                    }

                    if (verbose_console) {
                        std::println("stopped user-provided thread - state: {}", magic_enum::enum_name(this->state()));
                    }
                    if (auto ret = this->changeStateTo(lifecycle::State::STOPPED); !ret) {
                        using namespace gr::message;
                        this->emitErrorMessage("requested STOPPED", ret.error());
                    }
                },
                [this](std::thread* t) {
                    if (auto ret = this->changeStateTo(lifecycle::State::STOPPED); !ret) {
                        using namespace gr::message;
                        this->emitErrorMessage("requested STOPPED", ret.error());
                    }
                    if (t->joinable()) {
                        t->join();
                    }
                    delete t;
                    std::println("user-provided thread deleted");
                });
            if (verbose_console) {
                std::println("launched user-provided thread");
            }
            return true;
        }
        return false;
    }
};

using DefaultClockSource            = ClockSource<std::uint8_t, true, std::chrono::system_clock, true>;
using DefaultClockSourceNonBlocking = ClockSource<std::uint8_t, false, std::chrono::system_clock, true>;
} // namespace gr::basic

static_assert(gr::HasProcessBulkFunction<gr::basic::ClockSource<std::uint8_t>>);

#endif // GNURADIO_CLOCK_SOURCE_HPP
