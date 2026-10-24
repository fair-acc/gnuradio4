#ifndef GNURADIO_STREAMTODATASET_HPP
#define GNURADIO_STREAMTODATASET_HPP

#include "gnuradio-4.0/TriggerMatcher.hpp"
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetUtils.hpp>
#include <gnuradio-4.0/meta/utils.hpp>
#include <gnuradio-4.0/trigger/EventStore.hpp>
#include <gnuradio-4.0/trigger/TimeBase.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <deque>
#include <iterator>
#include <optional>
#include <ranges>
#include <vector>

namespace gr::basic {

GR_REGISTER_BLOCK("gr::basic::StreamToDataSet", gr::basic::StreamFilterImpl, ([T], false), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ]);
GR_REGISTER_BLOCK("gr::basic::StreamFilter", gr::basic::StreamFilterImpl, ([T], true), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double> ]);

template<typename T, bool streamOut = false, trigger::Matcher TMatcher = trigger::BasicTriggerNameCtxMatcher::Filter>
requires(std::is_arithmetic_v<T> || gr::meta::complex_like<T>)
struct StreamFilterImpl : Block<StreamFilterImpl<T, streamOut, TMatcher>, NoTagPropagation> {
    using Description = Doc<R"(@brief Selects ranges from an input stream using trigger tags and publishes them as either a stream or DataSet<T>.
One merged tag (every key seen, the newest value winning, a setting's key carrying its current value) is published once at the next output position.
A tag in the selected range keeps its sample position, in `DataSet::timing_events` for StreamToDataSet.)">;

    constexpr static std::size_t MIN_BUFFER_SIZE = 1024U;
    template<typename U, gr::meta::fixed_string description = "", typename... Arguments>
    using A = Annotated<U, description, Arguments...>;

    using OutType = std::conditional_t<streamOut, T, DataSet<T>>;
    PortIn<T>               in;
    EventPortIn             evtIn;
    PortOut<OutType, Async> out;
    EventPortOut            evtOut{{.streamSlotsPerPublish = 8UZ}};

    A<std::string, "filter", Visible, Doc<"syntax: '[<start trigger name>/<ctx1>, <stop trigger name>/<ctx2>]'; the first matching tag at a sample controls the range; nested starts are undefined, StreamToDataSet overlaps ranges whose starts and stops come in separate work calls">> filter;
    A<gr::Size_t, "n samples pre", Visible, Doc<"number of samples before the start trigger">>                                                                                                                                                                                            n_pre  = 0U; // Note: It is assumed that n_pre <= output port CircularBuffer size, and we wait until all n_pre samples can be written to the output in a single iteration.
    A<gr::Size_t, "n samples post", Visible, Doc<"number of samples after the stop trigger; with 0 and a stop matcher that excludes the stop sample, the stop tag closes the range but is not published">>                                                                                n_post = 0U;
    A<gr::Size_t, "n samples max", Doc<"maximum number of samples (0: infinite)">>                                                                                                                                                                                                        n_max  = 0U;
    A<std::string, "arm filter", Doc<"trigger filter that must be seen before a range may open, empty = always armed">>                                                                                                                                                                   arm_filter;
    A<std::string, "rearm", Doc<"auto = every trigger may open a range, manual = one per arming, and an arming re-opens a stopped single mode">>                                                                                                                                          rearm           = std::string("auto");
    A<gr::Size_t, "n segments", Doc<"ranges to open before the block stops opening more, 0 = unbounded">>                                                                                                                                                                                 n_segments      = 0U;
    A<gr::Size_t, "holdoff samples", Doc<"samples after a range opens during which no further one may, 0 = none">>                                                                                                                                                                        holdoff_samples = 0U;
    A<float, "holdoff seconds", Doc<"as above in seconds, where sample_rate is known, <= 0 = none">>                                                                                                                                                                                      holdoff_seconds = 0.f;
    A<gr::Size_t, "holdoff count", Doc<"start triggers refused after each admitted one">>                                                                                                                                                                                                 holdoff_count   = 0U;
    A<std::string, "mode", Doc<"normal|single|auto: single opens one range; auto forces one after auto_timeout without">>                                                                                                                                                                 mode            = std::string("normal");
    A<float, "auto timeout", Doc<"'auto': seconds without an admitted start before a range is forced; needs sample_rate">>                                                                                                                                                                auto_timeout    = 0.f;
    A<std::string, "force filter", Doc<"trigger filter opening a single range whatever arming, holdoff and mode say">>                                                                                                                                                                    force_filter    = std::string("forced");
    A<std::string, "stage", Doc<"priming|unarmed|armed|awaiting_samples|holdoff|stopped, as the block last saw itself; read-only">>                                                                                                                                                       stage           = std::string("priming");

    // meta information (will be usually set by incoming tags/upstream sources
    gr::SampleRate     sample_rate = 1.f;
    gr::SignalName     signal_name;
    gr::SignalQuantity signal_quantity;
    gr::SignalUnit     signal_unit;
    gr::SignalMin      signal_min = 0.f;
    gr::SignalMax      signal_max = 1.f;

    gr::Size_t _segmentsCaptured = 0U;
    /** armings that re-opened a stopped single mode, each allowing one more range */
    gr::Size_t _singleRearms = 0U;
    /** events on evtIn carrying no time, which act at the start of the call they are found in */
    gr::Size_t _undatedEvents = 0U;
    /** start triggers the arming, the holdoff or the segment limit turned down */
    gr::Size_t _startsRefused = 0U;
    /** start triggers that arrived while a range was already open (either output mode) */
    gr::Size_t _startsWhileBusy = 0U;

    GR_MAKE_REFLECTABLE(StreamFilterImpl, filter, in, evtIn, out, evtOut, n_pre, n_post, n_max, arm_filter, rearm, n_segments, holdoff_samples, holdoff_seconds, holdoff_count, mode, auto_timeout, force_filter, stage, sample_rate, signal_name, signal_quantity, signal_unit, signal_min, signal_max, _segmentsCaptured, _singleRearms, _undatedEvents, _startsRefused, _startsWhileBusy);

    static constexpr std::string_view kBusyTrigger = "busy"; /**< the trigger name of a report about a start seen while a range was open */

    HistoryBuffer<T>                                 _history{MIN_BUFFER_SIZE + n_pre};
    std::deque<std::pair<std::size_t, property_map>> _historyTags; // owning: Tag is view-only, history outlives the input span whose blob it aliases
    property_map                                     _mergedForwardedTag;
    TMatcher                                         _matcher{};
    property_map                                     _armState;
    property_map                                     _forceState;
    gr::Size_t                                       _holdoffLeft    = 0U;
    gr::Size_t                                       _forcedCaptured = 0U;
    std::size_t                                      _lastAdmittedAt = 0UZ;
    std::string_view                                 _origin; /// why the newest range opened when no start trigger did: forced, auto_timeout
    bool                                             _armed          = false;
    std::size_t                                      _samplesTakenIn = 0UZ;
    std::size_t                                      _deafUntil      = 0UZ;
    std::vector<property_map>                        _injected;
    gr::blocks::trigger::EventStore                  _events;
    gr::blocks::trigger::TimeBase                    _time;
    std::size_t                                      _spanIndex = 0UZ; /// stream index of the first sample of this work call
    std::deque<property_map>                         _busyNotices;

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
        _armState.clear();
        _forceState.clear();
        _armed          = false;
        _holdoffLeft    = 0U;
        _forcedCaptured = 0U;
        _lastAdmittedAt = 0UZ;
        _origin         = {};
        _samplesTakenIn = 0UZ;
        _deafUntil      = 0UZ;
        _injected.clear();
        _events.clear();
        _time.reset();
        _undatedEvents = 0U;
        _busyNotices.clear();
        _segmentsCaptured = 0U;
        _singleRearms     = 0U;
        stage             = std::string("priming");
        _startsRefused    = 0U;
        _startsWhileBusy  = 0U;
        if constexpr (streamOut) {
            _accState.reset();
        } else {
            _tempDataSets.clear();
            _accState.clear();
        }
        _historyTags.clear();
        _mergedForwardedTag.clear();
    }

    void settingsChanged(const gr::property_map& oldSettings, const gr::property_map& newSettings) {
        if (newSettings.contains("sample_rate")) {
            _time.setRate(static_cast<double>(sample_rate));
        }
        if (newSettings.contains("rearm") && rearm.value != "auto" && rearm.value != "manual") {
            const std::string previous = oldSettings.value_or<std::string>("rearm", std::string("auto"));
            this->emitErrorMessage("settingsChanged()", std::format("rearm '{}' is neither 'auto' nor 'manual': keeps '{}'", rearm.value, previous));
            rearm = previous;
        }
        if (const auto it = newSettings.find("n_pre"); it != newSettings.end()) {
            if constexpr (streamOut) {
                if (n_pre.value > out.buffer().streamBuffer.size()) { // reported, not refused: before the graph is connected the buffer is not yet sized
                    this->emitErrorMessage("settingsChanged()", std::format("n_pre {} exceeds the output buffer's {} samples", n_pre.value, out.buffer().streamBuffer.size()));
                }
            }
            const ValueView entry  = (*it).second; // ValueMap iter yields ValueView; get_if<Size_t> works on ValueView
            auto            prePtr = entry.get_if<gr::Size_t>();
            if (prePtr != nullptr) {
                _history.resize(MIN_BUFFER_SIZE + *prePtr);
            }
        }

        if constexpr (!streamOut) {
            if (newSettings.contains("n_pre") || newSettings.contains("n_post") || newSettings.contains("n_max")) {
                if (n_max != 0UZ && n_pre + n_post > n_max) {
                    this->emitErrorMessage("settingsChanged()", std::format("ill-formed settings: n_pre({}) + n_post({}) > n_max({})", n_pre, n_post, n_max));
                }
            }
        }
    }

    gr::work::Status processBulk(InputSpanLike auto& inSamples, InputSpanLike auto& evtSpan, OutputSpanLike auto& outSamples, OutputSpanLike auto& evtOutSpan) {
        _spanIndex                        = inSamples.streamIndex;
        const std::size_t      nAvailable = takeInjected(evtSpan, inSamples);
        const gr::work::Status status     = [&] {
            if constexpr (streamOut) {
                return processBulkStream(inSamples, outSamples, nAvailable);
            } else {
                return processBulkDataSet(inSamples, outSamples, nAvailable);
            }
        }();
        publishReports(evtOutSpan);
        noteStage();
        return status;
    }

    /**
     * an event stands in for the tag a stream does not carry: a dated one acts at the sample its `trigger_time` falls on,
     * so the call ends there; one that cannot be placed acts at the start of this call. Returns the samples this call may take.
     */
    std::size_t takeInjected(InputSpanLike auto& evtSpan, InputSpanLike auto& inSamples) {
        _injected.clear();
        for (const auto& [relIndex, tagMap] : inSamples.tags()) {
            std::ignore = _time.adopt(tagMap.get(), _spanIndex + (relIndex < 0 ? 0UZ : static_cast<std::size_t>(relIndex)));
        }
        _events.drain(evtSpan);
        std::size_t available = inSamples.size();
        std::size_t ready     = _events.ready();
        for (std::size_t k = 0UZ; k < ready;) {
            const gr::blocks::trigger::StoredEvent& stored = _events.entries[k];
            const std::optional<std::size_t>        placed = stored.dated() ? _time.indexAt(*stored.at) : std::nullopt;
            if (placed && *placed > _spanIndex) {
                available = std::min(available, *placed - _spanIndex);
                ++k;
                continue;
            }
            if (!_injected.empty()) { // one event acts per call: the rest wait for the next, which starts at the same sample
                available = 0UZ;
                break;
            }
            _undatedEvents = _undatedEvents + (stored.dated() ? 0U : 1U);
            _injected.emplace_back(stored.event);
            _events.entries.erase(_events.entries.begin() + static_cast<std::ptrdiff_t>(k));
            --ready;
        }
        return available;
    }

    void publishReports(OutputSpanLike auto& evtOutSpan) {
        std::size_t published = 0UZ;
        while (published < evtOutSpan.size() && !_busyNotices.empty()) {
            if (!gr::emit(evtOutSpan, published, property_map_view{_busyNotices.front()})) {
                break;
            }
            _busyNotices.pop_front();
            ++published;
        }
        evtOutSpan.publish(published);
    }

    gr::work::Status processBulkStream(InputSpanLike auto& inSamples, OutputSpanLike auto& outSamples, std::size_t nAvailable) {
        const auto                       inTags          = inputTags(inSamples);
        const std::optional<std::size_t> matchedTagIndex = findFirstTriggerTag(inTags);
        const Tag                        matchedTag      = triggerTagOf(inTags, matchedTagIndex);

        const auto detected = detectTrigger(matchedTag, _filterState);
        arm(matchedTag);
        const bool busy = detected.startTrigger && _accState.isActive;
        if (busy) {
            noteBusy();
        }
        const bool imposed      = !_accState.isActive && (forcedBy(matchedTag) || (!detected.startTrigger && autoDue())); // a real start on auto's deadline is a real start
        const bool startTrigger = admits(detected.startTrigger && !busy, imposed);
        _accState.update(startTrigger, detected.endTrigger, detected.isSingleTrigger || (startTrigger && imposed), n_pre, n_post);

        if (!_accState.isActive) {
            const std::size_t taken = untilAutoDeadline(nAvailable);
            updateHistory(inSamples, taken, true);
            std::ignore = inSamples.consume(taken);
            outSamples.publish(0UZ);
        } else {
            std::size_t nOutAvailable     = outSamples.size();
            std::size_t nSamplesToPublish = 0UZ;

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

            if (!_accState.isPostActive) {
                const std::size_t nSamplesToCopy = std::min(nAvailable, nOutAvailable);
                std::ranges::copy_n(inSamples.begin(), static_cast<std::ptrdiff_t>(nSamplesToCopy), std::next(outSamples.begin(), static_cast<std::ptrdiff_t>(nSamplesToPublish)));
                nSamplesToPublish += nSamplesToCopy;
                _accState.nSamples += nSamplesToCopy;
            } else {
                const std::size_t nPostSamplesToCopy = std::min(_accState.nPostSamplesRemain, std::min(nAvailable, nOutAvailable));
                std::ranges::copy_n(inSamples.begin(), static_cast<std::ptrdiff_t>(nPostSamplesToCopy), std::next(outSamples.begin(), static_cast<std::ptrdiff_t>(nSamplesToPublish)));
                nSamplesToPublish += nPostSamplesToCopy;
                _accState.updatePostSamples(nPostSamplesToCopy);
            }

            if (nSamplesToPublish == 0UZ && _accState.isActive) {
                std::ignore = inSamples.consume(0UZ);
                outSamples.publish(0UZ);
                return nOutAvailable == 0UZ ? work::Status::INSUFFICIENT_OUTPUT_ITEMS : work::Status::INSUFFICIENT_INPUT_ITEMS;
            }

            bool inTagsPublished = false;
            if (nSamplesToPublish > 0UZ) {
                publishMergedForwardedTag(outSamples);

                for (const auto& [tagIndex, tagMap] : _historyTags) {
                    const std::size_t offset = (n_pre > 0 && tagIndex < nPreSamplesToCopy) ? (nPreSamplesToCopy - tagIndex) : 0UZ;
                    outSamples.publishTag(tagMap, offset);
                }
                _historyTags.clear();

                const std::size_t nCurrentSamplesCopied = nSamplesToPublish - nPreSamplesToCopy;
                for (const Tag& tag : inTags) {
                    if (tag.index >= nCurrentSamplesCopied) {
                        continue;
                    }
                    const std::size_t offset = nPreSamplesToCopy + tag.index;
                    outSamples.publishTag(tag.map, offset);
                }
                inTagsPublished = true;
            }

            if (_accState.isActive) {
                updateHistory(inSamples, nSamplesToPublish - nPreSamplesToCopy, !inTagsPublished);
                std::ignore = inSamples.consume(nSamplesToPublish - nPreSamplesToCopy);
            } else { // the range ended here: the call stops at auto's next deadline, or where the range ended if later
                const std::size_t taken = std::max(nSamplesToPublish - nPreSamplesToCopy, untilAutoDeadline(nAvailable));
                updateHistory(inSamples, taken, !inTagsPublished);
                std::ignore = inSamples.consume(taken);
            }
            outSamples.publish(nSamplesToPublish);
        }
        return work::Status::OK;
    }

    gr::work::Status processBulkDataSet(InputSpanLike auto& inSamples, OutputSpanLike auto& outSamples, std::size_t nAvailable) {
        if (!_tempDataSets.empty() && !_accState.front().isActive && outSamples.size() == 0UZ) {
            std::ignore = inSamples.consume(0UZ);
            outSamples.publish(0UZ);
            return work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }

        const auto                       inTags          = inputTags(inSamples);
        const std::optional<std::size_t> matchedTagIndex = findFirstTriggerTag(inTags);
        const Tag                        matchedTag      = triggerTagOf(inTags, matchedTagIndex);

        //    This is a workaround to support cases of overlapping datasets, for example, Start1-Start2-Stop1-Stop2 case.
        //    always add new DataSet when Start trigger is present
        property_map tmpFilterState;
        const auto   detected = detectTrigger(matchedTag, tmpFilterState);
        arm(matchedTag);
        if (detected.startTrigger && !_accState.empty() && _accState.front().isActive) {
            noteBusy();
        }
        const auto [detectedStart, endTrigger, isSingleTrigger] = detected;
        const bool imposed                                      = _tempDataSets.empty() && (forcedBy(matchedTag) || (!detectedStart && autoDue()));
        const bool startTrigger                                 = admits(detectedStart, imposed);
        if (startTrigger) {
            _tempDataSets.emplace_back();
            initNewDataSet(_tempDataSets.back());
            if (imposed) {
                setAcquisitionFlag(_tempDataSets.back().meta_information[0], _origin);
            }

            _accState.emplace_back();
            _accState.back().update(startTrigger, endTrigger, isSingleTrigger || imposed, n_pre, n_post);

            _filterState.push_back(tmpFilterState);
        }

        // Update state only for the front dataset which is not in the isPostActiveState
        for (std::size_t i = 0; i < _tempDataSets.size(); i++) {
            if (!_accState[i].isActive) {
                continue;
            }
            if (!_accState[i].isPostActive) {
                const auto [startTrigger2, endTrigger2, isSingleTrigger2] = detectTrigger(matchedTag, _filterState[i]);
                if (endTrigger2) {
                    _accState[i].update(startTrigger2, endTrigger2, isSingleTrigger2, n_pre, n_post);
                }
                break; // only the first one should be updated
            }
        }

        std::size_t used = 0UZ; /// the latest input sample a range took in this call
        if (_tempDataSets.empty()) {
            const std::size_t taken = untilAutoDeadline(nAvailable);
            updateHistory(inSamples, taken, true);
            std::ignore = inSamples.consume(taken);
            outSamples.publish(0UZ);
            return work::Status::OK;
        } else {
            for (std::size_t i = 0; i < _tempDataSets.size(); i++) {
                auto& ds       = _tempDataSets[i];
                auto& accState = _accState[i];
                if (!accState.isActive) {
                    continue;
                }

                if (accState.isPreActive) {
                    // no need to check for n_max here: n_pre + n_post <= n_max
                    const std::size_t nPreSamplesToCopy = std::min(static_cast<std::size_t>(n_pre.value), _history.size()); // partially write pre samples if not enough samples stored in HistoryBuffer
                    const auto        historyEnd        = std::next(_history.cbegin(), static_cast<std::ptrdiff_t>(nPreSamplesToCopy));
                    ds.signal_values.insert(ds.signal_values.end(), std::make_reverse_iterator(historyEnd), std::make_reverse_iterator(_history.cbegin()));
                    fillAxisValues(ds, -static_cast<int>(nPreSamplesToCopy), nPreSamplesToCopy);
                    if (nPreSamplesToCopy < n_pre.value) {
                        setAcquisitionFlag(ds.meta_information[0], "incomplete"); // a range reaching before the first held sample is cut, never padded
                    }
                    accState.isPreActive = false;
                    accState.nPreSamples = nPreSamplesToCopy;
                    accState.nSamples += nPreSamplesToCopy;

                    if (nPreSamplesToCopy > 0UZ) {
                        for (const auto& [tagIndex, tagMap] : _historyTags) {
                            if (tagIndex <= nPreSamplesToCopy && !tagMap.empty()) {
                                ds.timing_events[0].emplace_back(static_cast<std::ptrdiff_t>(nPreSamplesToCopy - tagIndex), tagMap);
                            }
                        }
                    }
                }

                std::size_t nNonPreSamplesCopied = 0UZ;
                if (!accState.isPostActive) {
                    const std::size_t nSamplesToCopy = n_max.value == 0UZ ? nAvailable : std::min(n_max.value - ds.signal_values.size(), nAvailable);
                    if (nSamplesToCopy > 0) {
                        ds.signal_values.insert(ds.signal_values.end(), inSamples.begin(), inSamples.begin() + static_cast<std::ptrdiff_t>(nSamplesToCopy));
                        fillAxisValues(ds, static_cast<int>(accState.nSamples - accState.nPreSamples), nSamplesToCopy);
                        accState.nSamples += nSamplesToCopy;
                        nNonPreSamplesCopied += nSamplesToCopy;
                    }
                } else {
                    const std::size_t nPostSamplesToCopy = n_max.value == 0UZ ? std::min(accState.nPostSamplesRemain, nAvailable) : //
                                                               std::min({n_max.value - ds.signal_values.size(), accState.nPostSamplesRemain, nAvailable});
                    if (nPostSamplesToCopy > 0) {
                        ds.signal_values.insert(ds.signal_values.end(), inSamples.begin(), std::next(inSamples.begin(), static_cast<std::ptrdiff_t>(nPostSamplesToCopy)));
                        fillAxisValues(ds, static_cast<int>(accState.nSamples - accState.nPreSamples), nPostSamplesToCopy);
                        accState.updatePostSamples(nPostSamplesToCopy);
                        nNonPreSamplesCopied += nPostSamplesToCopy;
                    } else if (nAvailable > 0UZ) {
                        accState.isActive = false;
                    }
                }

                // Add tags only if at least one sample from current iteration is copied to DataSet
                used = std::max(used, nNonPreSamplesCopied);
                if (nNonPreSamplesCopied > 0UZ && !ds.timing_events.empty() && !inTags.empty()) {
                    for (const Tag& tag : inTags) {
                        if (tag.index < nNonPreSamplesCopied && !tag.map.empty()) {
                            ds.timing_events[0].emplace_back(static_cast<std::ptrdiff_t>(accState.nSamples - nNonPreSamplesCopied + tag.index), tag.map);
                        }
                    }
                }
            }
        }

        // once every range has ended, a call stops at auto's next deadline (or where the last range ended, if later)
        const bool        idle  = std::ranges::none_of(_accState, [](const AccumulationState& range) { return range.isActive; });
        const std::size_t taken = idle ? std::max(used, untilAutoDeadline(nAvailable)) : nAvailable;
        updateHistory(inSamples, taken, true);
        std::ignore = inSamples.consume(taken);

        std::size_t publishedCounter = 0UZ;
        while (!_tempDataSets.empty() && !_accState.front().isActive) {
            if (publishedCounter >= outSamples.size()) {
                break;
            }
            auto& ds = _tempDataSets.front();
            assert(!ds.extents.empty());
            ds.extents[0UZ] = static_cast<std::int32_t>(ds.signal_values.size());
            if (!ds.signal_values.empty()) { // TODO: do we need to publish empty  DataSet at all, empty DataSet can occur when n_max is set.
                gr::dataset::updateMinMax(ds);
            }
            outSamples[publishedCounter] = std::move(ds);
            _tempDataSets.pop_front();
            _accState.pop_front();
            _filterState.pop_front();
            publishedCounter++;
        }
        if (publishedCounter > 0UZ) {
            publishMergedForwardedTag(outSamples);
        }
        outSamples.publish(publishedCounter);

        return work::Status::OK;
    }

private:
    template<typename TInputSpan>
    [[nodiscard]] std::vector<Tag> inputTags(TInputSpan& inSpan) const {
        std::vector<Tag> tags;
        for (const auto& [relIndex, tagMapRef] : inSpan.tags()) {
            tags.emplace_back(relIndex < 0 ? 0UZ : static_cast<std::size_t>(relIndex), tagMapRef.get());
        }
        return tags;
    }

    [[nodiscard]] std::optional<std::size_t> findFirstTriggerTag(const std::vector<Tag>& inTags) {
        // If multiple Start/Stop/SingleTrigger trigger tags arrive at the same sample index ,
        // we process only the first one (regardless of Start, Stop or singleTrigger) and ignore the rest.
        // This should not occur in practice because it guarantees at most one trigger per sample index.
        // Note: StreamToDataSet have only Tags at index 0, since input_chunk_size == 1
        if (inTags.empty()) {
            return std::nullopt;
        }

        for (std::size_t i = 0UZ; i < inTags.size(); ++i) {
            const Tag& tag = inTags[i];
            if (tag.index != 0UZ) {
                continue;
            }

            if constexpr (streamOut) {
                property_map tmpFilterState                            = _filterState;
                const auto [startTrigger, endTrigger, isSingleTrigger] = detectTrigger(tag, tmpFilterState);
                if (startTrigger || endTrigger || isSingleTrigger) {
                    return i;
                }
            } else {
                property_map tmpEmptyFilterState;
                const auto [startTriggerEmpty, endTriggerEmpty, isSingleTriggerEmpty] = detectTrigger(tag, tmpEmptyFilterState);
                if (startTriggerEmpty || endTriggerEmpty || isSingleTriggerEmpty) {
                    return i;
                }

                for (const property_map& filterState : _filterState) {
                    property_map tmpFilterState                            = filterState;
                    const auto [startTrigger, endTrigger, isSingleTrigger] = detectTrigger(tag, tmpFilterState);
                    if (startTrigger || endTrigger || isSingleTrigger) {
                        return i;
                    }
                }
            }
        }

        const auto firstCurrentTag = std::ranges::find_if(inTags, [](const Tag& tag) { return tag.index == 0UZ; });
        if (firstCurrentTag != inTags.end()) {
            return static_cast<std::size_t>(std::distance(inTags.begin(), firstCurrentTag));
        }
        return std::nullopt;
    }

    /** an imposed start (forced, or auto's deadline) opens a range whatever arming and holdoff say; a stopped block opens none */
    [[nodiscard]] bool admits(bool startTrigger, bool imposed) {
        const bool forced = imposed && _origin == "forced";
        if ((stopped() && !forced) || (!startTrigger && !imposed)) {
            _startsRefused = _startsRefused + (startTrigger ? 1U : 0U);
            return false;
        }
        if (!imposed) {
            const bool counted = _holdoffLeft > 0U;
            if (!_armed || _samplesTakenIn < _deafUntil || counted) {
                _holdoffLeft = counted ? _holdoffLeft - 1U : _holdoffLeft;
                ++_startsRefused;
                return false;
            }
            if (rearm.value == "manual") {
                _armed = false;
            }
        }
        ++_segmentsCaptured;
        _forcedCaptured = _forcedCaptured + (forced ? 1U : 0U);
        _deafUntil      = _samplesTakenIn + holdoffSamples();
        _holdoffLeft    = holdoff_count;
        _lastAdmittedAt = _samplesTakenIn;
        if (!imposed) {
            _origin = {};
        } else if constexpr (streamOut) {
            setAcquisitionFlag(_mergedForwardedTag, _origin);
        }
        return true;
    }

    /** a forced range is an explicit request: it does not use up the segments, nor single mode's one range */
    [[nodiscard]] bool stopped() const noexcept {
        const gr::Size_t counted = _segmentsCaptured - _forcedCaptured;
        return (n_segments > 0U && counted >= n_segments) || singleSpent();
    }

    [[nodiscard]] bool singleSpent() const noexcept { return mode == "single" && _segmentsCaptured - _forcedCaptured >= 1U + _singleRearms; }

    [[nodiscard]] bool forcedBy(const Tag& tag) {
        if (force_filter.value.empty() || tag.map.empty() || _matcher(force_filter.value, tag, _forceState) != trigger::MatchResult::Matching) {
            return false;
        }
        _origin = "forced";
        return true;
    }

    [[nodiscard]] std::optional<std::size_t> autoTimeoutSamples() const noexcept {
        if (mode != "auto" || auto_timeout <= 0.f || sample_rate <= 0.f) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(std::round(static_cast<double>(auto_timeout) * static_cast<double>(sample_rate)));
    }

    [[nodiscard]] bool autoDue() {
        const auto wait = autoTimeoutSamples();
        if (!wait || *wait == 0UZ || stopped() || _samplesTakenIn < _lastAdmittedAt + *wait) {
            return false;
        }
        _origin = "auto_timeout";
        return true;
    }

    /** while waiting for auto's deadline, a call ends there, so the forced range starts at that sample however the stream is cut */
    [[nodiscard]] std::size_t untilAutoDeadline(std::size_t available) const noexcept {
        const auto wait = autoTimeoutSamples();
        if (!wait || *wait == 0UZ || stopped() || _lastAdmittedAt + *wait <= _samplesTakenIn) {
            return available;
        }
        return std::min(available, _lastAdmittedAt + *wait - _samplesTakenIn);
    }

    /** sets `trigger_meta_info.acquisition_flags.<flag>`, keeping what the map already holds */
    static void setAcquisitionFlag(property_map& meta, std::string_view flag) {
        const std::string_view key{gr::tag::TRIGGER_META_INFO.key()};
        property_map           details;
        if (const auto existing = property_map_view{meta}.get_if<property_map>(key)) {
            details = *existing;
        }
        property_map flags;
        if (const auto existing = property_map_view{details}.get_if<property_map>(gr::tag::ACQUISITION_FLAGS.key())) {
            flags = *existing;
        }
        flags.insert_or_assign(flag, true);
        details.insert_or_assign(std::string_view{gr::tag::ACQUISITION_FLAGS.key()}, std::move(flags));
        meta.insert_or_assign(key, std::move(details));
    }

    void noteStage() {
        const bool awaitsSamples = [this] {
            if constexpr (streamOut) {
                return _accState.isActive;
            } else {
                return !_tempDataSets.empty();
            }
        }();
        std::string_view now = "armed";
        if (awaitsSamples) {
            now = "awaiting_samples";
        } else if (stopped()) {
            now = "stopped";
        } else if (!_armed) {
            now = "unarmed";
        } else if (_holdoffLeft > 0U || _samplesTakenIn < _deafUntil) {
            now = "holdoff";
        } else if (_samplesTakenIn < n_pre) {
            now = "priming";
        }
        if (stage.value != now) {
            stage = std::string(now);
        }
    }

    [[nodiscard]] std::size_t holdoffSamples() const {
        if (holdoff_samples > 0U) {
            return static_cast<std::size_t>(holdoff_samples.value);
        }
        if (holdoff_seconds > 0.f && sample_rate > 0.f) {
            return static_cast<std::size_t>(std::llround(static_cast<double>(holdoff_seconds) * static_cast<double>(sample_rate)));
        }
        return 0UZ;
    }

    void arm(const Tag& tag) {
        if (arm_filter.value.empty()) {
            _armed = true;
            return;
        }
        if (_matcher(arm_filter.value, tag, _armState) == trigger::MatchResult::Matching) {
            _armed = true;
            if (rearm.value == "manual" && singleSpent()) {
                ++_singleRearms;
            }
        }
    }

    [[nodiscard]] Tag triggerTagOf(const std::vector<Tag>& inTags, std::optional<std::size_t> matchedTagIndex) const {
        if (matchedTagIndex.has_value()) {
            return inTags[*matchedTagIndex];
        }
        return _injected.empty() ? Tag{} : Tag{0UZ, ValueMapView(_injected.front())};
    }

    void noteBusy() {
        ++_startsWhileBusy;
        property_map report;
        report[gr::tag::TRIGGER_NAME.key()] = std::string(kBusyTrigger);
        _busyNotices.push_back(std::move(report));
    }

    [[nodiscard]] auto detectTrigger(const Tag& tag, property_map& filterState) {
        struct {
            bool startTrigger    = false;
            bool endTrigger      = false;
            bool isSingleTrigger = false;
        } result;

        const trigger::MatchResult matchResult = _matcher(filter.value, tag, filterState);
        if (matchResult != trigger::MatchResult::Ignore) {
            assert(filterState.contains("isSingleTrigger"));
            result.startTrigger    = matchResult == trigger::MatchResult::Matching;
            result.endTrigger      = matchResult == trigger::MatchResult::NotMatching;
            result.isSingleTrigger = filterState.value_or<bool>("isSingleTrigger", false);
        }
        return result;
    }

    void updateHistory(InputSpanLike auto& inSamples, std::size_t maxSamplesToCopy, bool copyInputTags) {
        const auto samplesToCopy = std::min(maxSamplesToCopy, inSamples.size());
        _samplesTakenIn += samplesToCopy;
        if (samplesToCopy > 0UZ) {
            const auto inTags = inputTags(inSamples);
            if constexpr (streamOut) {
                if (copyInputTags) {
                    if (n_pre > 0) {
                        for (const Tag& t : inTags) { // materialise the non-owning views into owning history entries
                            _historyTags.emplace_back(t.index, property_map(t.map));
                        }
                    } else {
                        mergeForwardedTags(inTags);
                    }
                }
            } else {
                if (copyInputTags && n_pre > 0) {
                    for (const Tag& t : inTags) {
                        _historyTags.emplace_back(t.index, property_map(t.map));
                    }
                }
                mergeForwardedTags(inTags);
            }

            if (n_pre > 0) {
                const auto samplesEnd = std::next(inSamples.begin(), static_cast<std::ptrdiff_t>(samplesToCopy));
                _history.push_front(inSamples.begin(), samplesEnd);
                for (auto& entry : _historyTags) {
                    entry.first += samplesToCopy;
                }
            }

            if constexpr (streamOut) {
                if (n_pre > 0) {
                    std::vector<std::pair<std::size_t, property_map>> expiredTags;
                    std::erase_if(_historyTags, [N = static_cast<std::size_t>(n_pre.value), &expiredTags](const auto& entry) {
                        if (entry.first <= N) {
                            return false;
                        }
                        expiredTags.push_back(entry);
                        return true;
                    });
                    mergeForwardedTags(expiredTags);
                }
            } else {
                std::erase_if(_historyTags, [N = static_cast<std::size_t>(n_pre.value)](const auto& entry) { return entry.first > N; });
            }
        }
    }

    void mergeForwardedTags(const std::vector<Tag>& tags) {
        for (const Tag& tag : tags) {
            this->forwardedTag(tag.map, [this](const auto& forwarded) {
                for (const auto& [key, value] : forwarded) {
                    _mergedForwardedTag.insert_or_assign(key, value);
                }
            });
        }
    }

    void mergeForwardedTags(const std::vector<std::pair<std::size_t, property_map>>& ownedTags) {
        std::vector<Tag> tagViews;
        tagViews.reserve(ownedTags.size());
        for (const auto& [index, map] : ownedTags) {
            tagViews.emplace_back(index, map);
        }
        mergeForwardedTags(tagViews);
    }

    void publishMergedForwardedTag(OutputSpanLike auto& outSpan) {
        if (!_mergedForwardedTag.empty()) {
            outSpan.publishTag(_mergedForwardedTag, 0UZ);
            _mergedForwardedTag.clear();
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
        dataSet.extents.emplace_back(0); // size of 1-dim data

        dataSet.signal_names.emplace_back(signal_name);
        dataSet.signal_quantities.emplace_back(signal_quantity);
        dataSet.signal_units.emplace_back(signal_unit);
        dataSet.signal_ranges.resize(1UZ);
        dataSet.meta_information.resize(1);
        dataSet.meta_information[0].insert_or_assign(std::string_view{"ctx"}, filter.value);
        dataSet.meta_information[0].insert_or_assign(std::string_view{"n_pre"}, n_pre.value);
        dataSet.meta_information[0].insert_or_assign(std::string_view{"n_post"}, n_post.value);
        dataSet.meta_information[0].insert_or_assign(std::string_view{"n_max"}, n_max.value);

        dataSet.timing_events.resize(1UZ);
    }
};

template<typename T>
using StreamToDataSet = StreamFilterImpl<T, false>;

template<typename T>
using StreamFilter = StreamFilterImpl<T, true>;

} // namespace gr::basic

#endif // GNURADIO_STREAMTODATASET_HPP
