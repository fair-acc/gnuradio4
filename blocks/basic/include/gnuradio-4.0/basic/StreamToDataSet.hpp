#ifndef GNURADIO_STREAMTODATASET_HPP
#define GNURADIO_STREAMTODATASET_HPP

#include "gnuradio-4.0/TriggerMatcher.hpp"
#include <gnuradio-4.0/algorithm/dataset/DataSetUtils.hpp>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::basic {

template<typename T, bool streamOut = true, trigger::Matcher TMatcher = trigger::BasicTriggerNameCtxMatcher::Filter>
    requires(std::is_arithmetic_v<T> || gr::meta::complex_like<T>)
struct StreamFilterImpl : Block<StreamFilterImpl<T, streamOut, TMatcher>, Doc<R""(
@brief Converts stream of input data into chunked discrete DataSet<T> based on tag-based pre- / post-conditions

)"">> {
    constexpr static std::size_t MIN_BUFFER_SIZE = 1024U;
    template<typename U, gr::meta::fixed_string description = "", typename... Arguments> // optional annotation shortening
    using A = Annotated<U, description, Arguments...>;

    // port definitions
    using OutType = std::conditional_t<streamOut, T, DataSet<T>>;
    PortIn<T>               in;
    PortOut<OutType, Async> out;

    // settings
    A<std::string, "filter", Visible, Doc<"syntax: '[<start trigger name>/<ctx1>, <stop trigger name>/<ctx2>]'">> filter;
    A<property_map, "filter state", Doc<"">>                                                                      filterState;
    A<gr::Size_t, "n samples pre", Visible, Doc<"number of pre-trigger samples">>                                 n_pre  = 0U;
    A<gr::Size_t, "n samples post", Visible, Doc<"number of post-trigger samples">>                               n_post = 0U;
    A<gr::Size_t, "n samples max", Doc<"maximum number of samples (0: infinite)">>                                n_max  = 0U;

    // meta information (will be usually set by incoming tags/upstream sources
    A<float, "sample_rate", Doc<"signal sample rate">>                                                       sample_rate = 1.f;
    A<std::string, "signal_name", Doc<"signal name">>                                                        signal_name;
    A<std::string, "signal quantity", Doc<"physical quantity (e.g., 'voltage'). Follows ISO 80000-1:2022.">> signal_quantity;
    A<std::string, "signal unit", Doc<"unit of measurement (e.g., '[V]', '[m]'). Follows ISO 80000-1:2022">> signal_unit;
    A<float, "signal_min", Doc<"signal physical max. (e.g. DAQ) limit">>                                     signal_min = 0.f;
    A<float, "signal_max", Doc<"signal physical max. (e.g. DAQ) limit">>                                     signal_max = 1.f;

    // internal trigger state
    HistoryBuffer<T> _history{ MIN_BUFFER_SIZE + n_pre };
    TMatcher         _matcher{};
    bool             _accumulationActive  = false;
    std::size_t      _nSamplesWritten     = 0UZ;
    std::size_t      _nPostSamplesWritten = 0UZ;
    DataSet<T>       _tempDataSet;
    bool             _dataSetAccumulation = false;

    void
    reset() {
        filterState.value.clear();
        _accumulationActive  = false;
        _dataSetAccumulation = false;
    }

    void
    settingsChanged(const gr::property_map & /*oldSettings*/, const gr::property_map &newSettings) {
        if (newSettings.contains("n_pre")) {
            auto newBuffer = HistoryBuffer<T>(MIN_BUFFER_SIZE + std::get<gr::Size_t>(newSettings.at("n_pre")));
            newBuffer.push_back_bulk(_history);
            _history = std::move(newBuffer);
        }
    }

    gr::work::Status
    processBulk(ConsumableSpan auto inSamples /* equivalent to std::span<const T> */, PublishableSpan auto &outSamples /* equivalent to std::span<T> */) {
        bool firstTrigger = false;
        bool lastTrigger  = false;
        if (const trigger::MatchResult matchResult = _matcher(filter.value, this->mergedInputTag(), filterState.value); matchResult != trigger::MatchResult::Ignore) {
            assert(filterState.value.contains("isSingleTrigger"));
            _accumulationActive = (matchResult == trigger::MatchResult::Matching);
            firstTrigger        = _accumulationActive;
            lastTrigger         = !_accumulationActive || std::get<bool>(filterState.value.at("isSingleTrigger"));
            if (std::get<bool>(filterState.value.at("isSingleTrigger"))) { // handled by the n_pre and n_post settings
                _accumulationActive = false;
            }
            if constexpr (!streamOut) {
                if (firstTrigger) {
                    _tempDataSet = DataSet<T>();
                    initNewDataSet(_tempDataSet);
                }
            }
        }
        if (lastTrigger) {
            _nPostSamplesWritten = n_post;
        }

        const std::size_t nInAvailable      = inSamples.size();
        std::size_t       nOutAvailable     = outSamples.size();
        std::size_t       nSamplesToPublish = 0UZ;

        if (firstTrigger) { // handle pre-trigger samples kept in _history
            _nSamplesWritten = 0UZ;
            if constexpr (streamOut) {
                if (n_pre >= nOutAvailable) {
                    std::ignore = inSamples.consume(0UZ);
                    outSamples.publish(0UZ);
                    return work::Status::INSUFFICIENT_OUTPUT_ITEMS;
                }
                const auto nPreSamplesToPublish = static_cast<std::size_t>(n_pre.value);
                std::ranges::copy_n(_history.cbegin(), static_cast<std::ptrdiff_t>(std::min(nPreSamplesToPublish, nOutAvailable)), outSamples.begin());
                nSamplesToPublish += n_pre;
                nOutAvailable -= n_pre;
            } else {
                _tempDataSet.signal_values.insert(_tempDataSet.signal_values.end(), _history.cbegin(), std::next(_history.cbegin(), static_cast<std::ptrdiff_t>(n_pre.value)));
                for (int i = -static_cast<int>(n_pre); i < 0; i++) {
                    _tempDataSet.axis_values[0].emplace_back(static_cast<float>(i) / sample_rate);
                }
            }
            _nSamplesWritten += n_pre;
        }

        if constexpr (!streamOut) { // move tags into DataSet
            const Tag &mergedTag = this->mergedInputTag();
            if (!_tempDataSet.timing_events.empty() && !mergedTag.map.empty() && (_accumulationActive || _nPostSamplesWritten > 0 || std::get<bool>(filterState.value.at("isSingleTrigger")))) {
                _tempDataSet.timing_events[0].emplace_back(Tag{ static_cast<Tag::signed_index_type>(_nSamplesWritten), mergedTag.map });
            }
            this->_mergedInputTag.map.clear(); // ensure that the input tag is only propagated once
        }

        std::size_t nSamplesToCopy = 0UZ;
        if (_accumulationActive) { // handle normal data accumulation
            if constexpr (streamOut) {
                nSamplesToCopy += std::min(nInAvailable, nOutAvailable);
                std::copy_n(inSamples.begin(), static_cast<std::ptrdiff_t>(nSamplesToCopy), std::next(outSamples.begin(), static_cast<std::ptrdiff_t>(nSamplesToPublish)));
                nOutAvailable -= nSamplesToCopy;
            } else {
                nSamplesToCopy += nInAvailable;
                _tempDataSet.signal_values.insert(_tempDataSet.signal_values.end(), inSamples.begin(), inSamples.end());
                _tempDataSet.axis_values[0].reserve(_nSamplesWritten + nInAvailable);
                for (auto i = 0U; i < nInAvailable; i++) {
                    _tempDataSet.axis_values[0].emplace_back(static_cast<float>(_nSamplesWritten - n_pre + i) / sample_rate);
                }
            }
            _nSamplesWritten += nSamplesToCopy;
            nSamplesToPublish += nSamplesToCopy;
        }

        if (_nPostSamplesWritten > 0) { // handle post-trigger samples
            if constexpr (streamOut) {
                const std::size_t nPostSamplesToPublish = std::min(_nPostSamplesWritten, std::min(nInAvailable, nOutAvailable));
                std::copy_n(inSamples.begin(), nPostSamplesToPublish, std::next(outSamples.begin(), static_cast<std::ptrdiff_t>(nSamplesToPublish)));
                nSamplesToCopy += nPostSamplesToPublish;
                nSamplesToPublish += nPostSamplesToPublish;
                _nSamplesWritten += nPostSamplesToPublish;
                _nPostSamplesWritten -= nPostSamplesToPublish;
            } else {
                const std::size_t nPostSamplesToPublish = std::min(_nPostSamplesWritten, nInAvailable);
                _tempDataSet.signal_values.insert(_tempDataSet.signal_values.end(), inSamples.begin(), std::next(inSamples.begin(), static_cast<std::ptrdiff_t>(nPostSamplesToPublish)));
                for (std::size_t i = 0; i < nPostSamplesToPublish; i++) {
                    _tempDataSet.axis_values[0].emplace_back(static_cast<float>(_nSamplesWritten - n_pre + i) / sample_rate);
                }
                _nSamplesWritten += nPostSamplesToPublish;
                _nPostSamplesWritten -= nPostSamplesToPublish;
            }
        }

        if (n_pre > 0) { // copy the last min(n_pre, nInAvailable) samples to the history buffer
            _history.push_back_bulk(std::prev(inSamples.end(), static_cast<std::ptrdiff_t>(std::min(static_cast<std::size_t>(n_pre), nInAvailable))), inSamples.end());
        }

        std::ignore = inSamples.consume(nInAvailable);
        if constexpr (streamOut) {
            outSamples.publish(nSamplesToPublish);
        } else {
            // publish a single sample if the DataSet<T> accumulation is complete, otherwise publish 0UZ
            if (((lastTrigger || (std::get<bool>(filterState.value.at("isSingleTrigger")) && _nSamplesWritten > 0)) && _nPostSamplesWritten == 0)
                || (_dataSetAccumulation && _tempDataSet.signal_values.size() >= n_max)) {
                assert(!_tempDataSet.extents.empty());
                _tempDataSet.extents[1] = static_cast<std::int32_t>(_nSamplesWritten);
                gr::dataset::updateMinMax(_tempDataSet);
                outSamples[0] = std::move(_tempDataSet);
                outSamples.publish(1UZ);
                _nSamplesWritten = 0UZ;
            } else {
                outSamples.publish(0UZ);
            }
        }
        return work::Status::OK;
    }

private:
    void
    initNewDataSet(DataSet<T> &dataSet) const {
        dataSet.axis_names.emplace_back("time");
        dataSet.axis_units.emplace_back("s");
        dataSet.axis_values.resize(1UZ);
        dataSet.extents.emplace_back(1); // 1-dim data
        dataSet.extents.emplace_back(0); // size of 1-dim data

        dataSet.signal_names.emplace_back(signal_name);
        dataSet.signal_quantities.emplace_back(signal_quantity);
        dataSet.signal_units.emplace_back(signal_unit);
        dataSet.signal_ranges.resize(1UZ);    // one data set
        dataSet.signal_ranges[0].resize(2UZ); // [min, max]s
        dataSet.meta_information.resize(1);   // one data set
        dataSet.meta_information[0]["ctx"]    = filter;
        dataSet.meta_information[0]["n_pre"]  = n_pre;
        dataSet.meta_information[0]["n_post"] = n_post;
        dataSet.meta_information[0]["n_max"]  = n_max;

        dataSet.timing_events.resize(1UZ); // one data set
    }
};

template<typename T>
using StreamToDataSet = StreamFilterImpl<T, false>;

template<typename T>
using StreamFilter = StreamFilterImpl<T, true>;

} // namespace gr::basic

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T, bool streamOut, typename Matcher), (gr::basic::StreamFilterImpl<T, streamOut, Matcher>), filter, in, out, filter, n_pre, n_post, n_max, sample_rate,
                                    signal_name, signal_quantity, signal_unit, signal_min, signal_max);
static_assert(gr::HasProcessBulkFunction<gr::basic::StreamFilterImpl<float>>);

inline static auto registerStreamFilters
        = gr::registerBlock<gr::basic::StreamToDataSet, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>>(
                  gr::globalBlockRegistry())
        | gr::registerBlock<gr::basic::StreamFilter, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>>(
                  gr::globalBlockRegistry());

#endif // GNURADIO_STREAMTODATASET_HPP
