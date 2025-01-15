#ifndef GNURADIO_STREAMTODATASET_HPP
#define GNURADIO_STREAMTODATASET_HPP

#include "gnuradio-4.0/TriggerMatcher.hpp"
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetUtils.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::basic {

template<typename T, bool streamOut = true, trigger::Matcher TMatcher = trigger::BasicTriggerNameCtxMatcher::Filter>
requires(std::is_arithmetic_v<T> || gr::meta::complex_like<T>)
struct StreamFilterImpl : Block<StreamFilterImpl<T, streamOut, TMatcher>> {
    using Description = Doc<R"(
@brief Converts stream of input data into chunked discrete DataSet<T> based on tag-based pre- / post-conditions
Notes:
* n_pre must be less than the size of the output port's CircularBuffer. The block waits until all n_pre samples can be written to the output in a single iteration.
* In stream-to-stream mode, the 'stop' Tag is not included in the output stream.
* In stream-to-dataset mode, the 'stop' Tag is included in the DataSet output.
* Stream-to-stream mode has limited support for overlapping triggers. Configurations such as Start-Stop-Start-Stop work as expected,
but configurations like Start-Start-Stop-Stop result in undefined behavior.
If data accumulation is active, the second 'start' is ignored, and the output may vary depending on subsequent tags.
* Tags for pre-samples are not stored and are not included in the output.
* It is expected that 'start' and 'stop' Tags will come from different iterations.
If multiple 'start' or 'stop' Tags arrive in a single merged tag, only one DataSet is created. This may lead to incorrect or unexpected output.)">;

    constexpr static std::size_t MIN_BUFFER_SIZE = 1024U;
    template<typename U, gr::meta::fixed_string description = "", typename... Arguments> // optional annotation shortening
    using A = Annotated<U, description, Arguments...>;

    // port definitions
    using OutType = std::conditional_t<streamOut, T, DataSet<T>>;
    PortIn<T>               in;
    PortOut<OutType, Async> out;

    // settings
    A<std::string, "filter", Visible, Doc<"syntax: '[<start trigger name>/<ctx1>, <stop trigger name>/<ctx2>]'">> filter;
    A<gr::Size_t, "n samples pre", Visible, Doc<"number of pre-trigger samples">>                                 n_pre  = 0U; // Note: It is assumed that n_pre <= output port CircularBuffer size, and we wait until all n_pre samples can be written to the output in a single iteration.
    A<gr::Size_t, "n samples post", Visible, Doc<"number of post-trigger samples">>                               n_post = 0U;
    A<gr::Size_t, "n samples max", Doc<"maximum number of samples (0: infinite)">>                                n_max  = 0U;

    // meta information (will be usually set by incoming tags/upstream sources
    A<float, "sample_rate", Doc<"signal sample rate">>                                                       sample_rate = 1.f;
    A<std::string, "signal_name", Doc<"signal name">>                                                        signal_name;
    A<std::string, "signal quantity", Doc<"physical quantity (e.g., 'voltage'). Follows ISO 80000-1:2022.">> signal_quantity;
    A<std::string, "signal unit", Doc<"unit of measurement (e.g., '[V]', '[m]'). Follows ISO 80000-1:2022">> signal_unit;
    A<float, "signal_min", Doc<"signal physical max. (e.g. DAQ) limit">>                                     signal_min = 0.f;
    A<float, "signal_max", Doc<"signal physical max. (e.g. DAQ) limit">>                                     signal_max = 1.f;

    GR_MAKE_REFLECTABLE(StreamFilterImpl, filter, in, out, filter, n_pre, n_post, n_max, sample_rate, signal_name, signal_quantity, signal_unit, signal_min, signal_max);

    // internal trigger state
    HistoryBuffer<T> _history{MIN_BUFFER_SIZE + n_pre};
    TMatcher         _matcher{};

    struct AccumulationState {
        bool        isActive           = false;
        bool        isPreActive        = false;
        bool        isPostActive       = false;
        bool        isSingleTrigger    = false;
        std::size_t nPostSamplesRemain = 0UZ;
        std::size_t nPreSamples        = 0UZ;
        std::size_t nSamples           = 0UZ;

        void update(bool startTrigger, bool endTrigger, bool isSingle, gr::Size_t nPre, gr::Size_t nPost) {
            isSingleTrigger = isSingle;
            if (!isActive) {
                if (startTrigger) {
                    isPreActive = nPre > 0; // No pre samples -> Done
                    isActive    = true;
                    nSamples    = 0UZ;
                    if (isSingleTrigger) {
                        isPostActive       = true;
                        nPostSamplesRemain = nPost;
                    }
                }
            }

            if (isActive && !isPostActive && endTrigger) {
                isPostActive       = true;
                nPostSamplesRemain = nPost;
            }
        }

        void updatePostSamples(std::size_t nPostSamplesToCopy) {
            nPostSamplesRemain -= nPostSamplesToCopy;
            nSamples += nPostSamplesToCopy;

            if (nPostSamplesRemain == 0UZ) {
                isActive     = false;
                isPostActive = false;
            }
        }

        void reset() {
            isActive           = false;
            isPreActive        = false;
            isPostActive       = false;
            nPostSamplesRemain = 0UZ;
        }
    };

    std::conditional_t<streamOut, AccumulationState, std::deque<AccumulationState>> _accState{};
    std::deque<DataSet<T>>                                                          _tempDataSets;
    std::conditional_t<streamOut, property_map, std::deque<property_map>>           _filterState;

    void reset() {
        _filterState.clear();
        if constexpr (streamOut) {
            _accState.reset();
        } else {
            _tempDataSets.clear();
            _accState.clear();
        }
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        if (newSettings.contains("n_pre")) {
            if constexpr (streamOut) {
                if (n_pre.value > out.buffer().streamBuffer.size()) {
                    using namespace gr::message;
                    throw gr::exception("n_pre must be <= output port CircularBuffer size");
                }
            }
            _history.resize(MIN_BUFFER_SIZE + std::get<gr::Size_t>(newSettings.at("n_pre")));
        }

        if constexpr (!streamOut) {
            if (newSettings.contains("n_pre") || newSettings.contains("n_post") || newSettings.contains("n_max")) {
                if (n_max != 0UZ && n_pre + n_post > n_max) {
                    using namespace gr::message;
                    throw gr::exception(fmt::format("ill-formed settings: n_pre({}) + n_post({}) > n_max({})", n_pre, n_post, n_max));
                }
            }
        }
    }

    gr::work::Status processBulk(InputSpanLike auto& inSamples, OutputSpanLike auto& outSamples) {
        if constexpr (streamOut) {
            return processBulkStream(inSamples, outSamples);
        } else {
            return processBulkDataSet(inSamples, outSamples);
        }
    }

    gr::work::Status processBulkStream(InputSpanLike auto& inSamples, OutputSpanLike auto& outSamples) {
        const auto [startTrigger, endTrigger, isSingleTrigger] = detectTrigger(_filterState);
        _accState.update(startTrigger, endTrigger, isSingleTrigger, n_pre, n_post);

        if (!_accState.isActive) { // If accumulation is not active, consume all input samples and publish 0 samples.
            copyInputSamplesToHistory(inSamples, inSamples.size());
            std::ignore = inSamples.consume(inSamples.size());
            outSamples.publish(0UZ);
        } else { // accumulation is active
            std::size_t nOutAvailable     = outSamples.size();
            std::size_t nSamplesToPublish = 0UZ;

            // pre samples data accumulation
            auto nPreSamplesToCopy = 0UZ;
            if (_accState.isPreActive) {
                // Note: It is assumed that n_pre <= output port CircularBuffer size, and we wait until all n_pre samples can be written to the output in a single iteration.
                nPreSamplesToCopy = std::min(static_cast<std::size_t>(n_pre.value), _history.size()); // partially write pre samples if not enough samples stored in HistoryBuffer
                if (nPreSamplesToCopy > nOutAvailable) {
                    std::ignore = inSamples.consume(0UZ);
                    outSamples.publish(0UZ);
                    return work::Status::INSUFFICIENT_OUTPUT_ITEMS;
                }
                auto startIt = std::next(_history.begin(), static_cast<std::ptrdiff_t>(nPreSamplesToCopy));
                std::ranges::copy_n(std::make_reverse_iterator(startIt), static_cast<std::ptrdiff_t>(nPreSamplesToCopy), outSamples.begin());
                nSamplesToPublish += nPreSamplesToCopy;
                nOutAvailable -= nPreSamplesToCopy;
                _accState.isPreActive = false;
                _accState.nSamples += nPreSamplesToCopy;
            }

            if (!_accState.isPostActive) { // normal data accumulation
                const std::size_t nSamplesToCopy = std::min(inSamples.size(), nOutAvailable);
                std::ranges::copy_n(inSamples.begin(), static_cast<std::ptrdiff_t>(nSamplesToCopy), std::next(outSamples.begin(), static_cast<std::ptrdiff_t>(nSamplesToPublish)));
                nSamplesToPublish += nSamplesToCopy;
                _accState.nSamples += nSamplesToCopy;
            } else { // post samples data accumulation
                const std::size_t nPostSamplesToCopy = std::min(_accState.nPostSamplesRemain, std::min(inSamples.size(), nOutAvailable));
                std::ranges::copy_n(inSamples.begin(), static_cast<std::ptrdiff_t>(nPostSamplesToCopy), std::next(outSamples.begin(), static_cast<std::ptrdiff_t>(nSamplesToPublish)));
                nSamplesToPublish += nPostSamplesToCopy;
                _accState.updatePostSamples(nPostSamplesToCopy);
            }

            copyInputSamplesToHistory(inSamples, nSamplesToPublish - nPreSamplesToCopy);
            std::ignore = inSamples.consume(nSamplesToPublish - nPreSamplesToCopy);
            outSamples.publish(nSamplesToPublish);
        }
        return work::Status::OK;
    }

    gr::work::Status processBulkDataSet(InputSpanLike auto& inSamples, OutputSpanLike auto& outSamples) {
        //    This is a workaround to support cases of overlapping datasets, for example, Start1-Start2-Stop1-Stop2 case.
        //    always add new DataSet when Start trigger is present
        property_map tmpFilterState;
        const auto [startTrigger, endTrigger, isSingleTrigger] = detectTrigger(tmpFilterState);
        if (startTrigger) {
            _tempDataSets.emplace_back();
            initNewDataSet(_tempDataSets.back());

            _accState.emplace_back();
            _accState.back().update(startTrigger, endTrigger, isSingleTrigger, n_pre, n_post);

            _filterState.push_back(tmpFilterState);
        }

        // Update state only for the front dataset which is not in the isPostActiveState
        for (std::size_t i = 0; i < _tempDataSets.size(); i++) {
            if (!_accState[i].isPostActive) {
                const auto [startTrigger2, endTrigger2, isSingleTrigger2] = detectTrigger(_filterState[i]);
                if (endTrigger2) {
                    _accState[i].update(startTrigger2, endTrigger2, isSingleTrigger2, n_pre, n_post);
                }
                break; // only the first one should be updated
            }
        }

        if (_tempDataSets.empty()) { // If accumulation is not active (no active DataSets), consume all input samples and publish 0 samples.
            copyInputSamplesToHistory(inSamples, inSamples.size());
            std::ignore = inSamples.consume(inSamples.size());
            outSamples.publish(0UZ);
            return work::Status::OK;
        } else { // accumulation is active (at least one DataSets is active)
            for (std::size_t i = 0; i < _tempDataSets.size(); i++) {
                auto& ds       = _tempDataSets[i];
                auto& accState = _accState[i];

                // Note: Tags for pre-samples are not added to DataSet
                if (n_max.value > ds.signal_values.size()) { // do not add Tags if DataSet is full
                    const Tag& mergedTag = this->mergedInputTag();
                    if (!ds.timing_events.empty() && !mergedTag.map.empty() && accState.isActive) {
                        ds.timing_events[0].emplace_back(static_cast<std::ptrdiff_t>(ds.signal_values.size()), mergedTag.map);
                    }
                }

                // pre samples data accumulation
                if (accState.isPreActive) {
                    // no need to check for n_max here: n_pre + n_post <= n_max
                    const std::size_t nPreSamplesToCopy = std::min(static_cast<std::size_t>(n_pre.value), _history.size()); // partially write pre samples if not enough samples stored in HistoryBuffer
                    const auto        historyEnd        = std::next(_history.cbegin(), static_cast<std::ptrdiff_t>(nPreSamplesToCopy));
                    ds.signal_values.insert(ds.signal_values.end(), std::make_reverse_iterator(historyEnd), std::make_reverse_iterator(_history.cbegin()));
                    fillAxisValues(ds, -static_cast<int>(nPreSamplesToCopy), nPreSamplesToCopy);
                    accState.isPreActive = false;
                    accState.nPreSamples = nPreSamplesToCopy;
                    accState.nSamples += nPreSamplesToCopy;
                }

                if (!accState.isPostActive) { // normal data accumulation
                    const std::size_t nSamplesToCopy = std::min(n_max.value - ds.signal_values.size(), inSamples.size());
                    if (nSamplesToCopy > 0) {
                        ds.signal_values.insert(ds.signal_values.end(), inSamples.begin(), inSamples.begin() + static_cast<std::ptrdiff_t>(nSamplesToCopy));
                        fillAxisValues(ds, static_cast<int>(accState.nSamples - accState.nPreSamples), nSamplesToCopy);
                        accState.nSamples += nSamplesToCopy;
                    }
                } else { // post samples data accumulation
                    const std::size_t nPostSamplesToCopy = std::min({n_max.value - ds.signal_values.size(), accState.nPostSamplesRemain, inSamples.size()});
                    if (nPostSamplesToCopy > 0) {
                        ds.signal_values.insert(ds.signal_values.end(), inSamples.begin(), std::next(inSamples.begin(), static_cast<std::ptrdiff_t>(nPostSamplesToCopy)));
                        fillAxisValues(ds, static_cast<int>(accState.nSamples - accState.nPreSamples), nPostSamplesToCopy);
                        accState.updatePostSamples(nPostSamplesToCopy);
                    } else {
                        accState.isActive = false;
                    }
                }
            }
            this->_mergedInputTag.map.clear(); // ensure that the input tag is only propagated once
        }

        copyInputSamplesToHistory(inSamples, inSamples.size());
        std::ignore = inSamples.consume(inSamples.size());

        // publish all completed DataSet<T>
        std::size_t publishedCounter = 0UZ;
        while (true) {
            if (!_tempDataSets.empty() && !_accState.front().isActive) {
                auto& ds = _tempDataSets.front();
                assert(!ds.extents.empty());
                ds.extents[1] = static_cast<std::int32_t>(ds.signal_values.size());
                if (!ds.signal_values.empty()) { // TODO: do we need to publish empty  DataSet at all, empty DataSet can occur when n_max is set.
                    gr::dataset::updateMinMax(ds);
                }
                outSamples[publishedCounter] = std::move(ds);
                _tempDataSets.pop_front();
                _accState.pop_front();
                _filterState.pop_front();
                publishedCounter++;
            } else {
                break;
            }
        }
        outSamples.publish(publishedCounter);

        return work::Status::OK;
    }

private:
    auto detectTrigger(property_map& filterState) {
        struct {
            bool startTrigger    = false;
            bool endTrigger      = false;
            bool isSingleTrigger = false;
        } result;

        const trigger::MatchResult matchResult = _matcher(filter.value, this->mergedInputTag(), filterState);
        if (matchResult != trigger::MatchResult::Ignore) {
            assert(filterState.contains("isSingleTrigger"));
            result.startTrigger    = matchResult == trigger::MatchResult::Matching;
            result.endTrigger      = matchResult == trigger::MatchResult::NotMatching;
            result.isSingleTrigger = std::get<bool>(filterState.at("isSingleTrigger"));
        }
        return result;
    }

    void copyInputSamplesToHistory(InputSpanLike auto& inSamples, std::size_t maxSamplesToCopy) {
        if (n_pre > 0) {
            const auto samplesToCopy = std::min(maxSamplesToCopy, inSamples.size());
            _history.push_front(inSamples.begin(), std::next(inSamples.begin(), static_cast<std::ptrdiff_t>(samplesToCopy)));
        }
    }

    void fillAxisValues(DataSet<T>& ds, int start, std::size_t nSamples) {
        ds.axis_values[0].reserve(ds.axis_values[0].size() + nSamples);
        for (int j = 0; j < static_cast<int>(nSamples); j++) {
            ds.axis_values[0].emplace_back(static_cast<float>(start + j) / sample_rate);
        }
    }

    void initNewDataSet(DataSet<T>& dataSet) const {
        dataSet.axis_names.emplace_back("time");
        dataSet.axis_units.emplace_back("s");
        dataSet.axis_values.resize(1UZ);
        dataSet.extents.emplace_back(1); // 1-dim data
        dataSet.extents.emplace_back(0); // size of 1-dim data

        dataSet.signal_names.emplace_back(signal_name);
        dataSet.signal_quantities.emplace_back(signal_quantity);
        dataSet.signal_units.emplace_back(signal_unit);
        dataSet.signal_ranges.resize(1UZ);  // one data set
        dataSet.meta_information.resize(1); // one data set
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

static_assert(gr::HasProcessBulkFunction<gr::basic::StreamFilterImpl<float>>);

inline static auto registerStreamFilters = gr::registerBlock<gr::basic::StreamToDataSet, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>>(gr::globalBlockRegistry()) | gr::registerBlock<gr::basic::StreamFilter, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>>(gr::globalBlockRegistry());

#endif // GNURADIO_STREAMTODATASET_HPP
