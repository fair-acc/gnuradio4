#ifndef GNURADIO_CLOCK_SOURCE_HPP
#define GNURADIO_CLOCK_SOURCE_HPP

#include <atomic>
#include <chrono>
#include <format>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/BlockingSync.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>

namespace gr::basic {

template<typename T, gr::meta::fixed_string description = "", typename... Arguments>
using A = gr::Annotated<T, description, Arguments...>;
using namespace gr;

GR_REGISTER_BLOCK("gr::basic::ClockSource", gr::basic::DefaultClockSource)

template<typename T, typename ClockSourceType = std::chrono::system_clock>
struct ClockSource : Block<ClockSource<T, ClockSourceType>>, BlockingSync<ClockSource<T, ClockSourceType>, ClockSourceType> {
    using Description = Doc<R""(@brief generates clock signals with specified timing intervals.)"">;

    using TimePoint = std::chrono::time_point<ClockSourceType>;

    PortOut<T> out;

    constexpr static gr::Size_t kDefaultChunkSize = 100U;

    A<gr::Size_t, "n_samples_max", Visible, Doc<"stops after this many samples; 0: unlimited">>                                                                                       n_samples_max = 1024;
    gr::SampleRate                                                                                                                                                                    sample_rate   = 1000.f;
    A<gr::Size_t, "chunk_size", Doc<"number of samples per update">, Limits<1U, std::numeric_limits<gr::Size_t>::max()>>                                                              chunk_size    = kDefaultChunkSize;
    A<Tensor<std::uint64_t>, "tag_times", Doc<"times after block start when tags should be emitted [ns]">>                                                                            tag_times;
    A<std::vector<std::string>, "tag_values", Doc<"list of '<trigger name>/<ctx>' formatted tags">>                                                                                   tag_values;
    A<std::uint64_t, "repeat_period", Doc<"if repeat_period > last tag_time -> restart tags, in [ns]">>                                                                               repeat_period{0U};
    A<std::uint64_t, "start_time", Doc<"UTC time of the first sample, in [ns], published as its trigger_time unless a tag time of 0 already dates it; 0: use the wall clock">>        start_time{0U};
    A<bool, "do_zero_order_hold", Doc<"if tag_times>tag_values: true=publish last tag, false=publish empty">>                                                                         do_zero_order_hold  = false;
    A<bool, "use_internal_thread", Doc<"true: GR4 timer thread wakes the scheduler every chunk_size / sample_rate; false: on-demand/external timing, samples from the elapsed time">> use_internal_thread = true;
    A<bool, "verbose_console">                                                                                                                                                        verbose_console     = false;

    GR_MAKE_REFLECTABLE(ClockSource, out, n_samples_max, sample_rate, chunk_size, tag_times, tag_values, repeat_period, start_time, do_zero_order_hold, use_internal_thread, verbose_console);

    struct ClockTag {
        std::size_t  index{0UZ};
        property_map map{};
    };

    std::vector<ClockTag> tags{};
    gr::Size_t            n_samples_produced{0};

    TimePoint   _beginSequenceTimePoint{};
    bool        _beginSequenceTimePointInitialized{false};
    std::size_t _nextTimeTag{0};
    std::size_t _nextTagIndex{0};

    std::uint64_t _counterOriginNs{0U}; /// [ns] clock of sample 0 on a device

    void start() {
        if (verbose_console) {
            std::println("ClockSource::start() - use_internal_thread: {}", static_cast<bool>(use_internal_thread));
        }
        n_samples_produced                 = 0U;
        _nextTimeTag                       = 0;
        _nextTagIndex                      = 0;
        _beginSequenceTimePointInitialized = false;
        _counterOriginNs                   = start_time != 0U ? start_time.value : hostNowNs();

        this->blockingSyncStart();

        if (verbose_console) {
            std::println("started {}", this->name);
        }
    }

    void stop() {
        if (verbose_console) {
            std::println("stopping {}", this->name);
        }
        this->blockingSyncStop();
    }

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        if (newSettings.contains("tag_times") && std::ranges::adjacent_find(tag_times.value, std::greater_equal()) != tag_times.value.end()) {
            tag_times.value = oldSettings.value_or<Tensor<std::uint64_t>>("tag_times", Tensor<std::uint64_t>{});
            this->emitErrorMessage("settingsChanged()", "tag_times must be strictly ascending; keeping the previous tag_times");
        }
        if (newSettings.contains("sample_rate") && !(sample_rate > 0.f)) {
            const float previous = oldSettings.value_or<float>("sample_rate", 1000.f);
            this->emitErrorMessage("settingsChanged()", std::format("sample_rate {} cannot pace a clock; keeping {}", sample_rate.value, previous));
            sample_rate = previous;
        }
        if (newSettings.contains("start_time")) {
            _counterOriginNs = start_time != 0U ? start_time.value : hostNowNs() - elapsedNs(n_samples_produced);
        }
    }

    [[nodiscard]] std::string_view hostOnlyReason() const noexcept {
        constexpr std::string_view kTimerThreadReason = "the internal timer thread runs on the host; set use_internal_thread = false for a device";
        return use_internal_thread ? kTimerThreadReason : "";
    }

    work::Status processBulk(OutputSpanLike auto& outSpan) noexcept {
        return produce(outSpan, this->syncSamples(outSpan), [] { return hostNowNs(); });
    }

    work::Status processBulk(gr::device::DeviceContext& /*ctx*/, OutputSpanLike auto& outSpan) noexcept {
        const std::size_t available = std::min<std::size_t>(outSpan.size(), chunk_size.value);
        return produce(outSpan, available, [this] { return _counterOriginNs + elapsedNs(n_samples_produced); });
    }

    [[nodiscard]] std::uint64_t elapsedNs(gr::Size_t nSamples) const noexcept { return static_cast<std::uint64_t>(static_cast<double>(nSamples) * 1e9 / static_cast<double>(sample_rate)); }

    [[nodiscard]] static std::uint64_t hostNowNs() noexcept {
        const auto sinceEpoch = ClockSourceType::now().time_since_epoch();
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(sinceEpoch).count());
    }

    /// `nowNs()`: the clock of the next sample, read only when a time tag needs it
    work::Status produce(OutputSpanLike auto& outSpan, std::size_t availableSamples, auto nowNs) noexcept {
        if (n_samples_max > 0UZ && n_samples_produced >= n_samples_max) {
            outSpan.publish(0UZ);
            return work::Status::DONE;
        }
        const gr::Size_t remainingSamples = (n_samples_max > 0UZ) ? n_samples_max - n_samples_produced : std::numeric_limits<gr::Size_t>::max();
        gr::Size_t       samplesToProduce = std::min({remainingSamples, static_cast<gr::Size_t>(availableSamples)});

        if (samplesToProduce == 0) {
            outSpan.publish(0UZ);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        const bool firstTimeTagDatesSampleZero = !tag_times.value.empty() && std::round(static_cast<double>(tag_times.value[0]) * 1e-9 * static_cast<double>(sample_rate)) == 0.;
        if (n_samples_produced == 0U && start_time != 0U && !firstTimeTagDatesSampleZero) {
            outSpan.publishTag(property_map{{std::string(tag::kTriggerTime.key()), start_time.value}, {std::string(tag::kTriggerOffset.key()), 0.f}}, 0UZ);
        }

        gr::Size_t samplesToNextTimeTag = std::numeric_limits<gr::Size_t>::max();
        if (!tag_times.value.empty()) {
            if (_nextTimeTag == 0 && !_beginSequenceTimePointInitialized) {
                _beginSequenceTimePoint            = this->blockingSyncStartTime();
                _beginSequenceTimePointInitialized = true;
            }
            if (_nextTimeTag >= tag_times.value.size() && repeat_period >= tag_times.value.back()) {
                _beginSequenceTimePoint += std::chrono::microseconds(repeat_period / 1000);
                _beginSequenceTimePointInitialized = true;
                _nextTimeTag                       = 0;
            }
            if (_nextTimeTag < tag_times.value.size()) {
                const auto tagOffsetNs       = std::chrono::duration_cast<std::chrono::nanoseconds>(_beginSequenceTimePoint - this->blockingSyncStartTime()).count() + static_cast<std::int64_t>(tag_times.value[_nextTimeTag]);
                const auto absoluteTagSample = static_cast<gr::Size_t>(std::max(0L, std::lround(static_cast<double>(tagOffsetNs) / 1.e9 * static_cast<double>(sample_rate))));
                samplesToNextTimeTag         = (absoluteTagSample >= n_samples_produced) ? (absoluteTagSample - n_samples_produced) : 0U;
            }
        }

        auto samplesToNextTag = tags.empty() || _nextTagIndex >= tags.size() ? std::numeric_limits<gr::Size_t>::max() : static_cast<gr::Size_t>(tags[_nextTagIndex].index) - n_samples_produced;

        if (samplesToNextTag < samplesToNextTimeTag) {
            if (_nextTagIndex < tags.size() && samplesToNextTag <= samplesToProduce) {
                const auto tagDeltaIndex = tags[_nextTagIndex].index - static_cast<std::size_t>(n_samples_produced);
                if (verbose_console) {
                    std::println("{}::processBulk(...)\t publish tag at {:6} @index= {}", this->name, n_samples_produced + tagDeltaIndex, tags[_nextTagIndex].index);
                }
                outSpan.publishTag(tags[_nextTagIndex].map, tagDeltaIndex);
                samplesToProduce = std::max(samplesToNextTag, gr::Size_t{1});
                _nextTagIndex++;
            }
        } else {
            if (!tag_times.value.empty() && _nextTimeTag < tag_times.value.size() && samplesToNextTimeTag <= samplesToProduce) {
                const std::string value = _nextTimeTag < tag_values.value.size() ? tag_values.value[_nextTimeTag] : (do_zero_order_hold ? tag_values.value.back() : "");

                std::string           triggerName;
                [[maybe_unused]] bool triggerNameNegated;
                std::string           triggerContext;
                [[maybe_unused]] bool triggerContextNegated;
                gr::basic::trigger::detail::parse(value, triggerName, triggerNameNegated, triggerContext, triggerContextNegated);

                property_map triggerTag;
                uint64_t     triggerTime = 0;
                if (start_time == 0U) {
                    triggerTime = nowNs();
                    triggerTime += static_cast<std::uint64_t>(static_cast<float>(samplesToNextTimeTag) * 1e9f / sample_rate);
                } else {
                    triggerTime = start_time + static_cast<std::uint64_t>(static_cast<double>(n_samples_produced + samplesToNextTimeTag) * 1e9 / static_cast<double>(sample_rate));
                }

                triggerTag[tag::kTriggerName]     = triggerName;
                triggerTag[tag::kTriggerTime]     = triggerTime;
                triggerTag[tag::kTriggerOffset]   = 0.f;
                triggerTag[tag::kContext]         = triggerContext;
                triggerTag[tag::kTriggerMetaInfo] = property_map{};

                if (verbose_console) {
                    std::println("{}::processBulk(...)\t publish tag-time at {:6}, time:{}ns", this->name, samplesToNextTimeTag, tag_times.value[_nextTimeTag]);
                }
                outSpan.publishTag(triggerTag, samplesToNextTimeTag);
                samplesToProduce = std::max(samplesToNextTimeTag, gr::Size_t{1});
                _nextTimeTag++;
            }
        }

        outSpan.publish(static_cast<std::size_t>(samplesToProduce));
        n_samples_produced += samplesToProduce;

        return work::Status::OK;
    }
};

using DefaultClockSource = ClockSource<std::uint8_t, std::chrono::system_clock>;

} // namespace gr::basic

static_assert(gr::HasProcessBulkFunction<gr::basic::ClockSource<std::uint8_t>>);

#endif // GNURADIO_CLOCK_SOURCE_HPP
