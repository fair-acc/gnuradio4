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
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

namespace gr::basic {

template<typename T, gr::meta::fixed_string description = "", typename... Arguments>
using A = gr::Annotated<T, description, Arguments...>;
using namespace gr;

GR_REGISTER_BLOCK("gr::basic::ClockSource", gr::basic::DefaultClockSource)

template<typename T, typename ClockSourceType = std::chrono::system_clock>
struct ClockSource : Block<ClockSource<T, ClockSourceType>>, BlockingSync<ClockSource<T, ClockSourceType>, ClockSourceType> {
    using Description = Doc<R""(@brief generates clock signals with specified timing intervals.

Operating modes (controlled by `use_internal_thread`):
  true (default): internal timer thread wakes scheduler every (chunk_size / sample_rate) seconds
  false: block produces samples based on elapsed time when scheduler calls processBulk

The 'tag_times[ns]:tag_value(string)' vectors control the emission of tags at specified times after block start.
Terminates when n_samples_max is reached (0 = unlimited).)"">;

    using TimePoint = std::chrono::time_point<ClockSourceType>;

    PortOut<T> out;

    A<gr::Size_t, "n_samples_max", Visible, Doc<"0: unlimited">>                                                 n_samples_max = 1024;
    A<float, "sample_rate", Visible, Doc<"average sample rate in Hz">>                                           sample_rate   = 1000.f;
    A<gr::Size_t, "chunk_size", Visible, Doc<"number of samples per update">>                                    chunk_size    = 100;
    A<Tensor<std::uint64_t>, "tag_times", Doc<"times when tags should be emitted [ns]">>                         tag_times;
    A<Tensor<pmt::Value>, "tag_values", Doc<"list of '<trigger name>/<ctx>' formatted tags">>                    tag_values;
    A<std::uint64_t, "repeat_period", Visible, Doc<"if repeat_period > last tag_time -> restart tags, in [ns]">> repeat_period{0U};
    A<bool, "do_zero_order_hold", Doc<"if tag_times>tag_values: true=publish last tag, false=publish empty">>    do_zero_order_hold  = false;
    A<bool, "use_internal_thread", Doc<"true: GR4 timer thread; false: on-demand/external timing">>              use_internal_thread = true;
    A<bool, "verbose_console">                                                                                   verbose_console     = false;

    GR_MAKE_REFLECTABLE(ClockSource, out, n_samples_max, sample_rate, chunk_size, tag_times, tag_values, repeat_period, do_zero_order_hold, use_internal_thread, verbose_console);

    std::vector<Tag> tags{};
    gr::Size_t       n_samples_produced{0};

    TimePoint   _beginSequenceTimePoint{};
    bool        _beginSequenceTimePointInitialized{false};
    std::size_t _nextTimeTag{0};
    std::size_t _nextTagIndex{0};

    void start() {
        if (verbose_console) {
            std::println("ClockSource::start() - use_internal_thread: {}", static_cast<bool>(use_internal_thread));
        }
        n_samples_produced                 = 0U;
        _nextTimeTag                       = 0;
        _nextTagIndex                      = 0;
        _beginSequenceTimePointInitialized = false;

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

    void settingsChanged(const property_map& /*oldSettings*/, const property_map& newSettings) {
        if (newSettings.contains("tag_times")) {
            if (std::ranges::adjacent_find(tag_times.value, std::greater_equal()) != tag_times.value.end()) {
                throw gr::exception("tag_times must be strictly ascending");
            }
        }
        if (newSettings.contains("chunk_size") && chunk_size.value < 1) {
            throw gr::exception("chunk_size must be >= 1");
        }
    }

    work::Status processBulk(OutputSpanLike auto& outSpan) noexcept {
        if (n_samples_max > 0UZ && n_samples_produced >= n_samples_max) {
            outSpan.publish(0UZ);
            return work::Status::DONE;
        }

        const gr::Size_t remainingSamples = (n_samples_max > 0UZ) ? n_samples_max - n_samples_produced : std::numeric_limits<gr::Size_t>::max();
        const auto       syncDrivenCount  = this->syncSamples(outSpan);
        gr::Size_t       samplesToProduce = std::min({remainingSamples, static_cast<gr::Size_t>(syncDrivenCount)});

        if (samplesToProduce == 0) {
            outSpan.publish(0UZ);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        const auto now = ClockSourceType::now();

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
                    gr::testing::print_tag(tags[_nextTagIndex], std::format("{}::processBulk(...)\t publish tag at {:6}", this->name, n_samples_produced + tagDeltaIndex));
                }
                outSpan.publishTag(tags[_nextTagIndex].map, tagDeltaIndex);
                samplesToProduce = std::max(samplesToNextTag, gr::Size_t{1});
                _nextTagIndex++;
            }
        } else {
            if (!tag_times.value.empty() && _nextTimeTag < tag_times.value.size() && samplesToNextTimeTag <= samplesToProduce) {
                const std::string value = _nextTimeTag < tag_values.value.size() ? tag_values.value[_nextTimeTag].value_or(std::string()) : (do_zero_order_hold ? tag_values.value.back().value_or(std::string()) : "");

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
