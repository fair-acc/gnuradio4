#ifndef GNURADIO_MATH_HISTOGRAM_HPP
#define GNURADIO_MATH_HISTOGRAM_HPP

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <format>
#include <string>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/algorithm/Histogram.hpp>
#include <gnuradio-4.0/math/SnapshotSchedule.hpp>
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
                                                     {"n_samples", 1000U}, {"axis_name", gr::Tensor<gr::pmt::Value>(gr::data_from, {gr::pmt::Value("interval")})}, {"axis_unit", gr::Tensor<gr::pmt::Value>(gr::data_from, {gr::pmt::Value("s")})}});
@endcode
)">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::PortIn<T>                               in;
    gr::EventPortIn                             evtIn;
    gr::PortOut<gr::DataSet<TValue>, gr::Async> out;

    A<double, "bin min", Doc<"lower edge of the first bin">>                            bin_min = 0.;
    A<double, "bin max", Doc<"upper edge of the last bin">>                             bin_max = 1.;
    A<gr::Size_t, "n bins">                                                             n_bins  = kDefaultBins;
    gr::TriggerName                                                                     trigger_name;
    A<gr::Size_t, "n samples", Doc<"publish after this many samples, 0 = never">>       n_samples        = 0U;
    A<double, "timeout", Doc<"s, time since last snapshot; 0 = never">>                 timeout          = 0.;
    A<bool, "reset on publish", Doc<"snapshot covers only samples since the last one">> reset_on_publish = true;
    gr::AxisName                                                                        axis_name        = gr::Tensor<gr::pmt::Value>(gr::data_from, {gr::pmt::Value("value")});
    gr::AxisUnit                                                                        axis_unit        = gr::Tensor<gr::pmt::Value>(gr::data_from, {gr::pmt::Value("a.u.")});
    gr::SignalName                                                                      signal_name      = std::string("counts");

    /** of the values counted, not of the bins they fell in; as is `_stdDev` */
    double _mean   = 0.;
    double _stdDev = 0.;
    /** meaningless while `_entriesInRange` is 0; as is `_max` */
    double     _min                = 0.;
    double     _max                = 0.;
    gr::Size_t _entriesInRange     = 0U;
    gr::Size_t _belowRange         = 0U;
    gr::Size_t _aboveRange         = 0U;
    gr::Size_t _snapshotsPublished = 0U;

    GR_MAKE_REFLECTABLE(Histogram, in, evtIn, out, bin_min, bin_max, n_bins, trigger_name, n_samples, timeout, reset_on_publish, axis_name, axis_unit, signal_name, _mean, _stdDev, _min, _max, _entriesInRange, _belowRange, _aboveRange, _snapshotsPublished);

    std::vector<std::uint64_t>          _counts;
    gr::algorithm::HistogramAccumulator _histogram;
    SnapshotSchedule                    _snapshot;

    void start() {
        resize();
        _snapshot.published(std::chrono::steady_clock::now());
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& inSamples, gr::InputSpanLike auto& evtSpan, gr::OutputSpanLike auto& outSets) {
        _snapshot.noteEvents(evtSpan);

        for (const T& sample : inSamples) {
            _histogram.add(static_cast<double>(gr::value(sample)));
        }
        _snapshot.sinceLastPublish += inSamples.size();
        report();

        const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        if (!snapshotDue(now)) {
            outSets.publish(0UZ);
            return gr::work::Status::OK;
        }
        if (outSets.empty()) {
            return gr::work::Status::OK;
        }
        outSets[0UZ] = snapshot();
        outSets.publish(1UZ);
        ++_snapshotsPublished;
        _snapshot.published(now);
        if (reset_on_publish) {
            _histogram.clear();
            report();
        }
        return gr::work::Status::OK;
    }

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& /*newSettings*/) {
        if (n_bins == 0U) {
            keepPreviousSetting(*this, n_bins, "n_bins", oldSettings, kDefaultBins, "a histogram of no bins counts nothing");
        }
        if (bin_max <= bin_min) {
            keepPreviousSetting(*this, bin_max, "bin_max", oldSettings, bin_min.value + 1., "an empty or inverted range has no bin to fill");
        }
        if (const auto set = _snapshot.setFilter(trigger_name.value); !set) {
            keepPreviousSetting(*this, trigger_name, "trigger_name", oldSettings, std::string(), set.error());
        }
        if (_counts.size() != static_cast<std::size_t>(n_bins.value)) {
            resize();
        } else {
            _histogram.binMin = bin_min;
            _histogram.binMax = bin_max;
        }
    }

    [[nodiscard]] bool snapshotDue(std::chrono::steady_clock::time_point now) const { return _snapshot.due(now, static_cast<std::size_t>(n_samples.value), timeout.value); }

private:
    void resize() {
        _counts.assign(static_cast<std::size_t>(n_bins.value), 0U);
        _histogram = gr::algorithm::HistogramAccumulator{.binMin = bin_min, .binMax = bin_max, .bins = std::span<std::uint64_t>{_counts}};
        report();
    }

    /** the figures a UI reads, taken from the accumulator once per work call rather than per sample */
    void report() {
        _mean           = _histogram.mean;
        _stdDev         = _histogram.stddev();
        _min            = _histogram.smallest;
        _max            = _histogram.largest;
        _entriesInRange = static_cast<gr::Size_t>(_histogram.entriesInRange);
        _belowRange     = static_cast<gr::Size_t>(_histogram.underflow);
        _aboveRange     = static_cast<gr::Size_t>(_histogram.overflow);
    }

    [[nodiscard]] gr::DataSet<TValue> snapshot() const {
        gr::DataSet<TValue> dataSet;
        dataSet.timestamp = 0;

        const auto firstLabel = [](const gr::Tensor<gr::pmt::Value>& labels) { return labels.empty() ? std::string{} : std::string(labels[0UZ].value_or(std::string_view{})); };
        dataSet.axis_names.emplace_back(firstLabel(axis_name.value));
        dataSet.axis_units.emplace_back(firstLabel(axis_unit.value));
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
        gr::property_map& figures = dataSet.meta_information[0];
        figures.insert_or_assign(std::string_view{"mean"}, _mean);
        figures.insert_or_assign(std::string_view{"stddev"}, _stdDev);
        figures.insert_or_assign(std::string_view{"min_value"}, _min);
        figures.insert_or_assign(std::string_view{"max_value"}, _max);
        figures.insert_or_assign(std::string_view{"n_entries"}, _entriesInRange);
        figures.insert_or_assign(std::string_view{"n_underflow"}, _belowRange);
        figures.insert_or_assign(std::string_view{"n_overflow"}, _aboveRange);
        dataSet.timing_events.resize(1UZ);
        return dataSet;
    }
};

GR_REGISTER_BLOCK(gr::blocks::math::Histogram2D, [T], [ float, double ])

template<std::floating_point T>
struct Histogram2D : gr::Block<Histogram2D<T>, gr::NoTagPropagation> {
    using Description = Doc<R""(@brief bin (x, y) pairs into a map [n_bins_y, n_bins_x], published on an event or periodically; `TensorPeakFinder` finds its clusters.

The pairs come sample by sample from `x` and `y`, or as the rows of a tensor [K, >=2] on `points` (columns 0 and 1; column `weight_column` weighs each, else each counts 1): whichever is connected is binned.
All three are synchronous, so connected together they are read in step, one tensor per pair.
A pair outside `[x_min, x_max) x [y_min, y_max)`, or with a NaN coordinate, is counted apart rather than clamped into an edge bin.
With `time_constant` > 0 the map forgets exponentially, by exp(-dt / time_constant) over the wall-clock time dt; 0 accumulates forever.
A snapshot is published on an event, every `n_samples` pairs, or after `timeout` -- whichever comes first -- and, with `reset_on_publish`, the next one starts empty.
Its axes go out as `gr:axis_*` tags, y first, at the bin centres, with the first map and after each settings change.)"">;

    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = gr::Annotated<U, description, Arguments...>;

    gr::PortIn<T, gr::Optional>       x;
    gr::PortIn<T, gr::Optional>       y;
    gr::TensorPortIn<T, gr::Optional> points;
    gr::EventPortIn                   evtIn;
    gr::TensorPortOut<T, gr::Async>   out;

    A<gr::Size_t, "n bins x">                                                                                n_bins_x      = kDefaultBins;
    A<gr::Size_t, "n bins y">                                                                                n_bins_y      = kDefaultBins;
    A<double, "x min", Doc<"lower edge of the first column">>                                                x_min         = 0.;
    A<double, "x max", Doc<"upper edge of the last column">>                                                 x_max         = 1.;
    A<double, "y min", Doc<"lower edge of the first row">>                                                   y_min         = 0.;
    A<double, "y max", Doc<"upper edge of the last row">>                                                    y_max         = 1.;
    A<std::int32_t, "weight column", Doc<"column of `points` weighing each pair, 2 or more; -1 = counts 1">> weight_column = -1;
    A<double, "time constant", Unit<"s">, Doc<"of the exponential forgetting; 0 = accumulate forever">>      time_constant = 0.;
    gr::TriggerName                                                                                          trigger_name;
    A<gr::Size_t, "n samples", Doc<"publish after this many pairs, 0 = never">>                              n_samples        = 0U;
    A<double, "timeout", Unit<"s">, Doc<"time since last snapshot; 0 = never">>                              timeout          = 0.;
    A<bool, "reset on publish", Doc<"snapshot covers only pairs since the last one">>                        reset_on_publish = true;
    gr::AxisName                                                                                             axis_name        = gr::Tensor<gr::pmt::Value>(gr::data_from, {gr::pmt::Value("y"), gr::pmt::Value("x")});
    gr::AxisUnit                                                                                             axis_unit        = gr::Tensor<gr::pmt::Value>(gr::data_from, {gr::pmt::Value("a.u."), gr::pmt::Value("a.u.")});

    /** sums of the weights binned, aged as the map */
    double     _entriesInRange     = 0.;
    double     _outsideRange       = 0.;
    gr::Size_t _snapshotsPublished = 0U;

    GR_MAKE_REFLECTABLE(Histogram2D, x, y, points, evtIn, out, n_bins_x, n_bins_y, x_min, x_max, y_min, y_max, weight_column, time_constant, trigger_name, n_samples, timeout, reset_on_publish, axis_name, axis_unit, _entriesInRange, _outsideRange, _snapshotsPublished);

    /** [n_bins_y, n_bins_x], row-major */
    std::vector<double>                   _bins;
    gr::algorithm::HistogramAccumulator2D _histogram;
    SnapshotSchedule                      _snapshot;
    std::chrono::steady_clock::time_point _lastAged;
    bool                                  _axesPublished = false;

    void start() {
        resize();
        _snapshot.published(std::chrono::steady_clock::now());
        _lastAged      = _snapshot.lastPublish;
        _axesPublished = false;
    }

    gr::work::Status processBulk(gr::InputSpanLike auto& xSpan, gr::InputSpanLike auto& ySpan, gr::InputSpanLike auto& pointsSpan, gr::InputSpanLike auto& evtSpan, gr::OutputSpanLike auto& outMaps) {
        _snapshot.noteEvents(evtSpan);

        const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        age(now);

        std::string refusal;
        if (xSpan.isConnected != ySpan.isConnected) {
            refusal = "x and y are binned as pairs: connect both or neither";
        }
        const std::size_t nPairs = refusal.empty() ? std::min(xSpan.size(), ySpan.size()) : 0UZ;
        for (std::size_t i = 0UZ; i < nPairs; ++i) {
            _histogram.add(static_cast<double>(xSpan[i]), static_cast<double>(ySpan[i]));
        }
        _snapshot.sinceLastPublish += nPairs;

        std::size_t k = 0UZ;
        for (; k < pointsSpan.size() && refusal.empty(); ++k) {
            const gr::TensorView<T>& rows    = pointsSpan[k];
            const std::size_t        columns = rows.rank() == 2UZ ? rows.extent(1UZ) : 0UZ;
            if (columns < 2UZ || (weight_column >= 0 && columns <= static_cast<std::size_t>(weight_column.value))) {
                refusal = std::format("points frame {} has shape {}, expects [K, >=2] with column {} for the weight", pointsSpan.streamIndex + k, rows.extents(), weight_column.value);
                break;
            }
            const T* const elements = rows.data();
            for (std::size_t row = 0UZ; row < rows.extent(0UZ); ++row) {
                const T* const pair   = elements + row * columns;
                const double   weight = weight_column >= 0 ? static_cast<double>(pair[weight_column.value]) : 1.;
                _histogram.add(static_cast<double>(pair[0]), static_cast<double>(pair[1]), weight);
            }
            _snapshot.sinceLastPublish += rows.extent(0UZ);
        }
        std::ignore = pointsSpan.consume(k);
        report();
        if (!refusal.empty()) [[unlikely]] {
            this->emitErrorMessage("processBulk()", refusal);
            this->requestStop();
            outMaps.publish(0UZ);
            return gr::work::Status::ERROR;
        }

        if (!snapshotDue(now)) {
            outMaps.publish(0UZ);
            return gr::work::Status::OK;
        }
        if (outMaps.size() == 0UZ) {
            return gr::work::Status::OK;
        }
        const std::array<std::size_t, 2> extents{static_cast<std::size_t>(n_bins_y.value), static_cast<std::size_t>(n_bins_x.value)};
        auto                             map = gr::emit(*this, outMaps, 0UZ, extents);
        if (!map) [[unlikely]] {
            outMaps.publish(0UZ);
            return gr::work::Status::ERROR;
        }
        std::ranges::transform(_bins, map->begin(), [](double content) { return static_cast<T>(content); });
        if (!_axesPublished) {
            outMaps.publishTag(axes(), 0UZ);
            _axesPublished = true;
        }
        outMaps.publish(1UZ);
        ++_snapshotsPublished;
        _snapshot.published(now);
        if (reset_on_publish) {
            _histogram.clear();
            report();
        }
        return gr::work::Status::OK;
    }

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& /*newSettings*/) {
        if (n_bins_x == 0U) {
            keepPreviousSetting(*this, n_bins_x, "n_bins_x", oldSettings, kDefaultBins, "a map of no columns counts nothing");
        }
        if (n_bins_y == 0U) {
            keepPreviousSetting(*this, n_bins_y, "n_bins_y", oldSettings, kDefaultBins, "a map of no rows counts nothing");
        }
        if (x_max <= x_min) {
            keepPreviousSetting(*this, x_max, "x_max", oldSettings, x_min.value + 1., "an empty or inverted range has no bin to fill");
        }
        if (y_max <= y_min) {
            keepPreviousSetting(*this, y_max, "y_max", oldSettings, y_min.value + 1., "an empty or inverted range has no bin to fill");
        }
        if (weight_column < -1 || weight_column == 0 || weight_column == 1) {
            keepPreviousSetting(*this, weight_column, "weight_column", oldSettings, std::int32_t(-1), "columns 0 and 1 are x and y");
        }
        if (!(time_constant >= 0.)) {
            keepPreviousSetting(*this, time_constant, "time_constant", oldSettings, 0., "a negative time constant grows the map without bound");
        }
        if (const auto set = _snapshot.setFilter(trigger_name.value); !set) {
            keepPreviousSetting(*this, trigger_name, "trigger_name", oldSettings, std::string(), set.error());
        }
        const auto nBinsX = static_cast<std::size_t>(n_bins_x.value);
        const auto nBinsY = static_cast<std::size_t>(n_bins_y.value);
        if (_bins.size() != nBinsX * nBinsY || _histogram.nBinsX != nBinsX) {
            resize();
        } else {
            _histogram.xMin = x_min;
            _histogram.xMax = x_max;
            _histogram.yMin = y_min;
            _histogram.yMax = y_max;
        }
        _axesPublished = false;
    }

    [[nodiscard]] bool snapshotDue(std::chrono::steady_clock::time_point now) const { return _snapshot.due(now, static_cast<std::size_t>(n_samples.value), timeout.value); }

    void age(std::chrono::steady_clock::time_point now) {
        if (time_constant > 0.) {
            _histogram.decay(std::exp(-std::chrono::duration<double>(now - _lastAged).count() / time_constant.value));
        }
        _lastAged = now;
    }

private:
    void resize() {
        _bins.assign(static_cast<std::size_t>(n_bins_x.value) * static_cast<std::size_t>(n_bins_y.value), 0.);
        _histogram = gr::algorithm::HistogramAccumulator2D{.xMin = x_min, .xMax = x_max, .yMin = y_min, .yMax = y_max, .nBinsX = static_cast<std::size_t>(n_bins_x.value), .bins = std::span<double>{_bins}};
        report();
    }

    void report() {
        _entriesInRange = _histogram.entriesInRange;
        _outsideRange   = _histogram.outsideRange;
    }

    [[nodiscard]] gr::property_map axes() const {
        const auto                 labelOf = [](const gr::Tensor<gr::pmt::Value>& labels, std::size_t d) { return labels.size() == 2UZ ? labels[d] : gr::pmt::Value(std::string_view{}); };
        gr::Tensor<double>         origins(gr::data_from, {_histogram.binCentreY(0UZ), _histogram.binCentreX(0UZ)});
        gr::Tensor<double>         steps(gr::data_from, {_histogram.binWidthY(), _histogram.binWidthX()});
        gr::Tensor<gr::pmt::Value> units(gr::data_from, {labelOf(axis_unit.value, 0UZ), labelOf(axis_unit.value, 1UZ)});
        gr::Tensor<gr::pmt::Value> names(gr::data_from, {labelOf(axis_name.value, 0UZ), labelOf(axis_name.value, 1UZ)});
        return {{std::string(gr::tag::kAxisOrigin.key()), origins}, {std::string(gr::tag::kAxisStep.key()), steps}, //
            {std::string(gr::tag::kAxisUnit.key()), units}, {std::string(gr::tag::kAxisName.key()), names}};
    }
};

} // namespace gr::blocks::math

#endif // GNURADIO_MATH_HISTOGRAM_HPP
