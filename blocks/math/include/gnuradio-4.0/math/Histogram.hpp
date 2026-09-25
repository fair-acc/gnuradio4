#ifndef GNURADIO_MATH_HISTOGRAM_HPP
#define GNURADIO_MATH_HISTOGRAM_HPP

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/TriggerMatcher.hpp>
#include <gnuradio-4.0/algorithm/Histogram.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

namespace gr::blocks::math {

GR_REGISTER_BLOCK("gr::blocks::math::Histogram", gr::blocks::math::Histogram, [T], [ float, double, gr::UncertainValue<float>, gr::UncertainValue<double> ])

template<typename T>
struct Histogram : gr::Block<Histogram<T>, gr::NoTagPropagation> {
    using TValue      = gr::meta::fundamental_base_value_type_t<T>;
    using Description = Doc<R"(@brief bin a stream into a distribution, published on an event or periodically

    in    ─0.4─0.5─0.5─0.6─▶       (no RxMarbles equivalent: a reduction to a distribution)
    evtIn ──────────────S──▶       S = snapshot
    out   ──────────────D──▶       D = DataSet: counts against bin centres, figures in its meta

A measurement arriving as single values -- an interval between two triggers, a peak amplitude, a time of flight -- is
read as a distribution, and the figures that go with it are the mean and the deviation. The range is explicit: a
sample outside `[bin_min, bin_max)` is counted as under- or overflow rather than widening the axis, so two runs of one
configuration are comparable. A snapshot is published on an event, every `n_samples`, or after `timeout` -- whichever
comes first. Figures come from the values, not the bin centres, by Welford. An `UncertainValue` input is binned by
its value: a count has no uncertainty of its own.

@code
auto& jitter = graph.emplaceBlock<Histogram<double>>({{"bin_min", 0.}, {"bin_max", 1e-6}, {"n_bins", 100U},
                                                     {"n_samples", 1000U}, {"axis_name", "interval"}, {"axis_unit", "s"}});
@endcode
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::PortIn<T>                               in;
    gr::EventPortIn                             evtIn;
    gr::PortOut<gr::DataSet<TValue>, gr::Async> out;

    A<double, "bin min", Doc<"lower edge of the first bin">>                             bin_min = 0.;
    A<double, "bin max", Doc<"upper edge of the last bin">>                              bin_max = 1.;
    A<gr::Size_t, "n bins">                                                              n_bins  = 100U;
    A<std::string, "trigger name", Doc<"filter naming the snapshot event; empty = any">> trigger_name;
    A<gr::Size_t, "n samples", Doc<"publish after this many samples, 0 = never">>        n_samples        = 0U;
    A<double, "timeout", Doc<"s, time since last snapshot; 0 = never">>                  timeout          = 0.;
    A<bool, "reset on publish", Doc<"snapshot covers only samples since the last one">>  reset_on_publish = true;
    A<std::string, "axis name">                                                          axis_name        = std::string("value");
    A<std::string, "axis unit">                                                          axis_unit        = std::string("a.u.");
    A<std::string, "signal name">                                                        signal_name      = std::string("counts");

    A<double, "mean", Doc<"of the values counted, not of the bins they fell in">>       mean        = 0.;
    A<double, "stddev", Doc<"of the values counted, not of the bins they fell in">>     stddev      = 0.;
    A<double, "min value", Doc<"smallest value counted; meaningless if n_entries = 0">> min_value   = 0.;
    A<double, "max value", Doc<"largest value counted; meaningless if n_entries = 0">>  max_value   = 0.;
    A<gr::Size_t, "n entries", Doc<"samples inside the range">>                         n_entries   = 0U;
    A<gr::Size_t, "n underflow">                                                        n_underflow = 0U;
    A<gr::Size_t, "n overflow">                                                         n_overflow  = 0U;
    A<gr::Size_t, "n published">                                                        n_published = 0U;

    GR_MAKE_REFLECTABLE(Histogram, in, evtIn, out, bin_min, bin_max, n_bins, trigger_name, n_samples, timeout, reset_on_publish, axis_name, axis_unit, signal_name, mean, stddev, min_value, max_value, n_entries, n_underflow, n_overflow, n_published);

    std::vector<std::uint64_t>                          _counts; // the storage the accumulator bins into
    gr::algorithm::HistogramAccumulator<double>         _histogram;
    gr::trigger::BasicTriggerNameCtxMatcher::MatchState _snapshotFilter{};
    bool                                                _filtered         = false;
    std::size_t                                         _sinceLastPublish = 0UZ;
    std::chrono::steady_clock::time_point               _lastPublish      = std::chrono::steady_clock::now();
    bool                                                _asked            = false;

    void start() { resize(); }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSamples, gr::InputSpanLike auto& evtSpan, gr::OutputSpanLike auto& outSets) {
        for (const gr::property_map_view& event : evtSpan) {
            _asked = _asked || asksForSnapshot(event);
        }
        std::ignore = evtSpan.consume(evtSpan.size());

        for (const T& sample : inSamples) {
            _histogram.add(static_cast<double>(gr::value(sample)));
        }
        _sinceLastPublish += inSamples.size();
        report();

        if (!due()) {
            outSets.publish(0UZ);
            return gr::work::Status::OK;
        }
        if (outSets.empty()) {
            return gr::work::Status::OK; // the snapshot stays due until there is somewhere to put it
        }
        outSets[0UZ] = snapshot();
        outSets.publish(1UZ);
        n_published       = n_published + 1U;
        _asked            = false;
        _sinceLastPublish = 0UZ;
        _lastPublish      = std::chrono::steady_clock::now();
        if (reset_on_publish) {
            _histogram.clear();
            report();
        }
        return gr::work::Status::OK;
    }

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& /*newSettings*/) {
        if (n_bins == 0U) {
            keepPrevious(n_bins, "n_bins", oldSettings, gr::Size_t(100), "a histogram of no bins counts nothing");
        }
        if (bin_max <= bin_min) {
            keepPrevious(bin_max, "bin_max", oldSettings, bin_min.value + 1., "an empty or inverted range has no bin to fill");
        }
        _filtered = !trigger_name.value.empty();
        if (_filtered) {
            if (const auto compiled = gr::trigger::BasicTriggerNameCtxMatcher::compile(trigger_name.value)) {
                _snapshotFilter = *compiled;
            } else {
                keepPrevious(trigger_name, "trigger_name", oldSettings, std::string(), compiled.error().message);
                _filtered = false;
            }
        }
        if (_counts.size() != static_cast<std::size_t>(n_bins.value)) { // only a change of shape may throw counts away
            resize();
        } else {
            _histogram.binMin = bin_min;
            _histogram.binMax = bin_max;
        }
    }

private:
    void resize() {
        _counts.assign(static_cast<std::size_t>(n_bins.value), 0U);
        _histogram = gr::algorithm::HistogramAccumulator<double>{.binMin = bin_min, .binMax = bin_max, .bins = std::span<std::uint64_t>{_counts}};
        report();
    }

    /// the figures a UI reads, taken from the accumulator once per work call rather than per sample
    void report() {
        mean        = _histogram.mean;
        stddev      = _histogram.stddev();
        min_value   = _histogram.smallest;
        max_value   = _histogram.largest;
        n_entries   = static_cast<gr::Size_t>(_histogram.entries);
        n_underflow = static_cast<gr::Size_t>(_histogram.underflow);
        n_overflow  = static_cast<gr::Size_t>(_histogram.overflow);
    }

    void keepPrevious(auto& setting, std::string_view key, const gr::property_map& oldSettings, auto fallback, std::string_view reason) {
        const auto previous = oldSettings.value_or<std::remove_cvref_t<decltype(fallback)>>(std::string(key), fallback);
        gr::log::warning("Histogram: '{}' refused ({}); keeping {}", key, reason, previous);
        setting = previous;
    }

    [[nodiscard]] bool asksForSnapshot(const gr::property_map_view& event) { // the matcher carries state, so it cannot be const
        if (event.empty()) {
            return false;
        }
        return !_filtered || gr::trigger::BasicTriggerNameCtxMatcher::match(_snapshotFilter, event) == gr::trigger::MatchResult::Matching;
    }

    [[nodiscard]] bool due() const {
        if (_asked) {
            return true;
        }
        if (n_samples > 0U && _sinceLastPublish >= static_cast<std::size_t>(n_samples.value)) {
            return true;
        }
        return timeout > 0. && std::chrono::duration<double>(std::chrono::steady_clock::now() - _lastPublish).count() >= timeout.value;
    }

    [[nodiscard]] gr::DataSet<TValue> snapshot() const {
        gr::DataSet<TValue> dataSet;
        dataSet.timestamp = 0;

        dataSet.axis_names.emplace_back(axis_name);
        dataSet.axis_units.emplace_back(axis_unit);
        dataSet.axis_values.resize(1UZ);
        dataSet.axis_values[0].reserve(_counts.size());
        std::ranges::transform(std::views::iota(0UZ, _counts.size()), std::back_inserter(dataSet.axis_values[0]), [this](std::size_t bin) { return static_cast<TValue>(_histogram.binCentre(bin)); });
        dataSet.extents.emplace_back(static_cast<std::int32_t>(_counts.size()));

        dataSet.signal_names.emplace_back(signal_name);
        dataSet.signal_quantities.emplace_back("count");
        dataSet.signal_units.emplace_back("#");
        dataSet.signal_values.reserve(_counts.size());
        std::ranges::transform(_counts, std::back_inserter(dataSet.signal_values), [](std::uint64_t counted) { return static_cast<TValue>(counted); });
        const std::uint64_t peak = _counts.empty() ? 0U : std::ranges::max(_counts);
        dataSet.signal_ranges.push_back(gr::Range<TValue>{TValue(0), static_cast<TValue>(peak)});

        dataSet.meta_information.resize(1UZ);
        dataSet.meta_information[0].insert_or_assign(std::string_view{"mean"}, mean.value);
        dataSet.meta_information[0].insert_or_assign(std::string_view{"stddev"}, stddev.value);
        dataSet.meta_information[0].insert_or_assign(std::string_view{"min_value"}, min_value.value);
        dataSet.meta_information[0].insert_or_assign(std::string_view{"max_value"}, max_value.value);
        dataSet.meta_information[0].insert_or_assign(std::string_view{"n_entries"}, n_entries.value);
        dataSet.meta_information[0].insert_or_assign(std::string_view{"n_underflow"}, n_underflow.value);
        dataSet.meta_information[0].insert_or_assign(std::string_view{"n_overflow"}, n_overflow.value);
        dataSet.timing_events.resize(1UZ);
        return dataSet;
    }
};

} // namespace gr::blocks::math

#endif // GNURADIO_MATH_HISTOGRAM_HPP
