#ifndef GNURADIO_STREAMTODATASET_HPP
#define GNURADIO_STREAMTODATASET_HPP

#include "gnuradio-4.0/TriggerMatcher.hpp"
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/algorithm/dataset/DataSetUtils.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include <algorithm>
#include <cassert>
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
Basic behavior:
* filter selects the trigger tags that open and close a range.
* n_pre adds samples before the start trigger. n_post adds samples after the stop trigger. n_max limits the DataSet size.
* Sample tags are published only when their attached sample is copied to the output. They keep the correct sample position.
* The block also publishes a merged auto-forward tag. It contains only keys from the block's auto-forward list, which is initialized from kDefaultTags. It is published once at the next output position. If the same key is seen more than once, the newest value is used.
* Tags at the same input sample are checked one by one. They are not merged before trigger matching. If several tags match, the first matching tag controls the range.

StreamFilter output:
* Publishes selected samples as a normal stream.
* Publishes tags in the selected range at the correct output samples.
* Drops old sample tags once they are outside the n_pre history window. Their auto-forward keys can still be included in the next merged auto-forward tag.
* Supports simple non-overlapping ranges. Start-Stop-Start-Stop works. Nested Start-Start-Stop-Stop is undefined and later start tags can be ignored.

StreamToDataSet output:
* Publishes one DataSet<T> for each completed range.
* Publishes all tags in the selected range to DataSet::timing_events, attached to the matching DataSet samples.
* Publishes the merged auto-forward tag as a normal output tag on the published DataSet.
* Supports overlapping DataSets when starts and stops are processed in separate work calls.
* If the stop matcher excludes the stop sample and n_post == 0, the stop tag closes the range but is not published as a sample tag.)">;

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

    GR_MAKE_REFLECTABLE(StreamFilterImpl, filter, in, out, n_pre, n_post, n_max, sample_rate, signal_name, signal_quantity, signal_unit, signal_min, signal_max);

    // internal trigger state
    HistoryBuffer<T>    _history{MIN_BUFFER_SIZE + n_pre};
    std::deque<gr::Tag> _historyTags;
    property_map        _mergedAutoForwardTag;
    TMatcher            _matcher{};

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
        _historyTags.clear();
        _mergedAutoForwardTag.clear();
    }

    void settingsChanged(const gr::property_map& /*oldSettings*/, const gr::property_map& newSettings) {
        if (const auto it = newSettings.find("n_pre"); it != newSettings.end()) {
            if constexpr (streamOut) {
                if (n_pre.value > out.buffer().streamBuffer.size()) {
                    using namespace gr::message;
                    throw gr::exception("n_pre must be <= output port CircularBuffer size");
                }
            }
            auto prePtr = it->second.get_if<gr::Size_t>();
            if (prePtr != nullptr) {
                _history.resize(MIN_BUFFER_SIZE + *prePtr);
            }
        }

        if constexpr (!streamOut) {
            if (newSettings.contains("n_pre") || newSettings.contains("n_post") || newSettings.contains("n_max")) {
                if (n_max != 0UZ && n_pre + n_post > n_max) {
                    using namespace gr::message;
                    throw gr::exception(std::format("ill-formed settings: n_pre({}) + n_post({}) > n_max({})", n_pre, n_post, n_max));
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
        const auto                       inTags          = inputTags(inSamples);
        const std::optional<std::size_t> matchedTagIndex = findFirstTriggerTag(inTags);
        const Tag                        emptyTag{};
        const Tag&                       matchedTag = matchedTagIndex.has_value() ? inTags[*matchedTagIndex] : emptyTag;

        const auto [startTrigger, endTrigger, isSingleTrigger] = detectTrigger(matchedTag, _filterState);
        _accState.update(startTrigger, endTrigger, isSingleTrigger, n_pre, n_post);

        if (!_accState.isActive) { // If accumulation is not active, consume all input samples and publish 0 samples.
            updateHistory(inSamples, inSamples.size(), true);
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

            if (nSamplesToPublish == 0UZ && _accState.isActive) {
                std::ignore = inSamples.consume(0UZ);
                outSamples.publish(0UZ);
                return nOutAvailable == 0UZ ? work::Status::INSUFFICIENT_OUTPUT_ITEMS : work::Status::INSUFFICIENT_INPUT_ITEMS;
            }

            bool inTagsPublished = false;
            if (nSamplesToPublish > 0UZ) {
                publishMergedAutoForwardTag(outSamples);

                for (const Tag& tag : _historyTags) {
                    const std::size_t offset = (n_pre > 0 && tag.index < nPreSamplesToCopy) ? (nPreSamplesToCopy - tag.index) : 0UZ;
                    outSamples.publishTag(tag.map, offset);
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
            } else {
                updateHistory(inSamples, inSamples.size(), !inTagsPublished);
                std::ignore = inSamples.consume(inSamples.size());
            }
            outSamples.publish(nSamplesToPublish);
        }
        return work::Status::OK;
    }

    gr::work::Status processBulkDataSet(InputSpanLike auto& inSamples, OutputSpanLike auto& outSamples) {
        if (!_tempDataSets.empty() && !_accState.front().isActive && outSamples.size() == 0UZ) {
            std::ignore = inSamples.consume(0UZ);
            outSamples.publish(0UZ);
            return work::Status::INSUFFICIENT_OUTPUT_ITEMS;
        }

        const auto                       inTags          = inputTags(inSamples);
        const std::optional<std::size_t> matchedTagIndex = findFirstTriggerTag(inTags);
        const Tag                        emptyTag{};
        const Tag&                       matchedTag = matchedTagIndex.has_value() ? inTags[*matchedTagIndex] : emptyTag;

        //    This is a workaround to support cases of overlapping datasets, for example, Start1-Start2-Stop1-Stop2 case.
        //    always add new DataSet when Start trigger is present
        property_map tmpFilterState;
        const auto [startTrigger, endTrigger, isSingleTrigger] = detectTrigger(matchedTag, tmpFilterState);
        if (startTrigger) {
            _tempDataSets.emplace_back();
            initNewDataSet(_tempDataSets.back());

            _accState.emplace_back();
            _accState.back().update(startTrigger, endTrigger, isSingleTrigger, n_pre, n_post);

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

        if (_tempDataSets.empty()) { // If accumulation is not active (no active DataSets) -> update history, consume all input samples and publish 0 samples.
            updateHistory(inSamples, inSamples.size(), true);
            std::ignore = inSamples.consume(inSamples.size());
            outSamples.publish(0UZ);
            return work::Status::OK;
        } else { // accumulation is active (at least one DataSets is active)
            for (std::size_t i = 0; i < _tempDataSets.size(); i++) {
                auto& ds       = _tempDataSets[i];
                auto& accState = _accState[i];
                if (!accState.isActive) {
                    continue;
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

                    if (nPreSamplesToCopy > 0UZ) {
                        for (const Tag& tag : _historyTags) {
                            if (tag.index <= nPreSamplesToCopy && !tag.map.empty()) {
                                ds.timing_events[0].emplace_back(static_cast<std::ptrdiff_t>(nPreSamplesToCopy - tag.index), tag.map);
                            }
                        }
                    }
                }

                std::size_t nNonPreSamplesCopied = 0UZ;
                if (!accState.isPostActive) { // normal data accumulation
                    const std::size_t nSamplesToCopy = n_max.value == 0UZ ? inSamples.size() : std::min(n_max.value - ds.signal_values.size(), inSamples.size());
                    if (nSamplesToCopy > 0) {
                        ds.signal_values.insert(ds.signal_values.end(), inSamples.begin(), inSamples.begin() + static_cast<std::ptrdiff_t>(nSamplesToCopy));
                        fillAxisValues(ds, static_cast<int>(accState.nSamples - accState.nPreSamples), nSamplesToCopy);
                        accState.nSamples += nSamplesToCopy;
                        nNonPreSamplesCopied += nSamplesToCopy;
                    }
                } else {                                                                                                                  // post samples data accumulation
                    const std::size_t nPostSamplesToCopy = n_max.value == 0UZ ? std::min(accState.nPostSamplesRemain, inSamples.size()) : //
                                                               std::min({n_max.value - ds.signal_values.size(), accState.nPostSamplesRemain, inSamples.size()});
                    if (nPostSamplesToCopy > 0) {
                        ds.signal_values.insert(ds.signal_values.end(), inSamples.begin(), std::next(inSamples.begin(), static_cast<std::ptrdiff_t>(nPostSamplesToCopy)));
                        fillAxisValues(ds, static_cast<int>(accState.nSamples - accState.nPreSamples), nPostSamplesToCopy);
                        accState.updatePostSamples(nPostSamplesToCopy);
                        nNonPreSamplesCopied += nPostSamplesToCopy;
                    } else {
                        accState.isActive = false;
                    }
                }

                // Add tags only if at least one sample from current iteration is copied to DataSet
                if (nNonPreSamplesCopied > 0UZ && !ds.timing_events.empty() && !inTags.empty()) {
                    for (const Tag& tag : inTags) {
                        if (tag.index < nNonPreSamplesCopied && !tag.map.empty()) {
                            ds.timing_events[0].emplace_back(static_cast<std::ptrdiff_t>(accState.nSamples - nNonPreSamplesCopied + tag.index), tag.map);
                        }
                    }
                }
            }
        }

        updateHistory(inSamples, inSamples.size(), true);
        std::ignore = inSamples.consume(inSamples.size());

        // publish all completed DataSet<T>
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
            publishMergedAutoForwardTag(outSamples);
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
            result.isSingleTrigger = filterState.at("isSingleTrigger").value_or(false);
        }
        return result;
    }

    void updateHistory(InputSpanLike auto& inSamples, std::size_t maxSamplesToCopy, bool copyInputTags) {
        const auto samplesToCopy = std::min(maxSamplesToCopy, inSamples.size());
        if (samplesToCopy > 0UZ) {
            const auto inTags = inputTags(inSamples);
            if constexpr (streamOut) {
                if (copyInputTags) {
                    if (n_pre > 0) {
                        _historyTags.insert(_historyTags.end(), inTags.begin(), inTags.end());
                    } else {
                        mergeAutoForwardTags(inTags);
                    }
                }
            } else {
                if (copyInputTags && n_pre > 0) {
                    _historyTags.insert(_historyTags.end(), inTags.begin(), inTags.end());
                }
                mergeAutoForwardTags(inTags);
            }

            if (n_pre > 0) {
                const auto samplesEnd = std::next(inSamples.begin(), static_cast<std::ptrdiff_t>(samplesToCopy));
                _history.push_front(inSamples.begin(), samplesEnd);
                for (Tag& tag : _historyTags) {
                    tag.index += samplesToCopy;
                }
            }

            if constexpr (streamOut) {
                if (n_pre > 0) {
                    std::vector<Tag> expiredTags;
                    std::erase_if(_historyTags, [N = static_cast<std::size_t>(n_pre.value), &expiredTags](const Tag& tag) {
                        if (tag.index <= N) {
                            return false;
                        }
                        expiredTags.push_back(tag);
                        return true;
                    });
                    mergeAutoForwardTags(expiredTags);
                }
            } else {
                std::erase_if(_historyTags, [N = static_cast<std::size_t>(n_pre.value)](const Tag& tag) { return tag.index > N; });
            }
        }
    }

    void mergeAutoForwardTags(const std::vector<Tag>& tags) {
        const auto&                 autoForwardKeys = this->settings().autoForwardParameters();
        const auto&                 blockSettings   = CtxSettings<StreamFilterImpl<T, streamOut, TMatcher>>::allWritableMembers();
        std::optional<property_map> cachedSettings;

        for (const Tag& tag : tags) {
            for (const auto& [key, value] : tag.map) {
                const auto shortKey = convert_string_domain(key);
                if (!autoForwardKeys.contains(shortKey)) {
                    continue;
                }
                if (!cachedSettings) {
                    cachedSettings.emplace(this->settings().get());
                }
                if (const auto it = cachedSettings->find(key); blockSettings.contains(shortKey) && it != cachedSettings->end()) {
                    _mergedAutoForwardTag.insert_or_assign(key, it->second);
                } else {
                    _mergedAutoForwardTag.insert_or_assign(key, value);
                }
            }
        }
    }

    void publishMergedAutoForwardTag(OutputSpanLike auto& outSpan) {
        if (!_mergedAutoForwardTag.empty()) {
            outSpan.publishTag(_mergedAutoForwardTag, 0UZ);
            _mergedAutoForwardTag.clear();
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
        dataSet.signal_ranges.resize(1UZ);  // one data set
        dataSet.meta_information.resize(1); // one data set
        dataSet.meta_information[0]["ctx"]    = filter.value;
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

#endif // GNURADIO_STREAMTODATASET_HPP
