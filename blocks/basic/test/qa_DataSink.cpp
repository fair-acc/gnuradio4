#include <future>
#include <ranges>

#include <boost/ut.hpp>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

#include <gnuradio-4.0/basic/DataSink.hpp>
#include <gnuradio-4.0/basic/StreamToDataSet.hpp>
#include <gnuradio-4.0/testing/Delay.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

using namespace std::chrono_literals;

namespace gr::basic::data_sink_test {

constexpr auto kProcessingDelayMs = 600u;

template<typename T>
std::vector<T> getIota(std::size_t n, const T& first = {}) {
    std::vector<T> v(n);
    std::iota(v.begin(), v.end(), first);
    return v;
}

/**
 * Example tag matcher (trigger::Matcher implementation) for the multiplexed listener case (interleaved data). As a toy example, we use
 * data tagged as Year/Month/Day.
 *
 * For each of year, month, day, the user can specify whether:
 *
 *  - option not set: The field is to be ignored
 *  - -1: Whenever a change between the previous and the current tag is observed, start a new data set (StopAndStart)
 *  - other values >= 0: A new dataset is started when the tag matches, and stopped, when a tag doesn't match
 *
 * (Note that this Matcher is stateful and remembers the last tag seen)
 */
struct Matcher {
    std::optional<int>                       year;
    std::optional<int>                       month;
    std::optional<int>                       day;
    std::optional<std::tuple<int, int, int>> last_seen;
    bool                                     last_matched = false;

    explicit Matcher(std::optional<int> year_, std::optional<int> month_, std::optional<int> day_) : year(year_), month(month_), day(day_) {}

    static inline bool same(int x, std::optional<int> other) { return other && x == *other; }

    static inline bool changed(int x, std::optional<int> other) { return !same(x, other); }

    [[nodiscard]] trigger::MatchResult operator()(std::string_view /* filterSpec */, const Tag& tag, const property_map& /* filter state */) {
        const auto ty = tag.get("YEAR");
        const auto tm = tag.get("MONTH");
        const auto td = tag.get("DAY");
        if (!ty || !tm || !td) {
            return trigger::MatchResult::Ignore;
        }

        const auto tup        = std::make_tuple(std::get<int>(ty->get()), std::get<int>(tm->get()), std::get<int>(td->get()));
        const auto& [y, m, d] = tup;
        const auto ly         = last_seen ? std::optional<int>(std::get<0>(*last_seen)) : std::nullopt;
        const auto lm         = last_seen ? std::optional<int>(std::get<1>(*last_seen)) : std::nullopt;
        const auto ld         = last_seen ? std::optional<int>(std::get<2>(*last_seen)) : std::nullopt;

        const auto yearRestart  = year && *year == -1 && changed(y, ly);
        const auto yearMatches  = !year || *year == -1 || same(y, year);
        const auto monthRestart = month && *month == -1 && changed(m, lm);
        const auto monthMatches = !month || *month == -1 || same(m, month);
        const auto dayRestart   = day && *day == -1 && changed(d, ld);
        const auto dayMatches   = !day || *day == -1 || same(d, day);
        const auto matches      = yearMatches && monthMatches && dayMatches;
        const auto restart      = yearRestart || monthRestart || dayRestart;

        auto r = trigger::MatchResult::Ignore;

        if (!matches) {
            r = trigger::MatchResult::NotMatching;
        } else if (!last_matched || restart) {
            r = trigger::MatchResult::Matching;
        }

        last_seen    = tup;
        last_matched = matches;
        return r;
    }
};

static Tag makeTag(std::size_t index, int year, int month, int day) { return Tag{index, {{"YEAR", year}, {"MONTH", month}, {"DAY", day}}}; }

static std::vector<Tag> makeTestTags(std::size_t firstIndex, std::size_t interval, std::size_t nTagsPerIndex = 1UZ) {
    std::vector<Tag> tags;
    for (int y = 1; y <= 3; ++y) {
        for (int m = 1; m <= 2; ++m) {
            for (int d = 1; d <= 3; ++d) {
                for (int iT = 0; iT < static_cast<int>(nTagsPerIndex); iT++) {
                    tags.push_back(makeTag(firstIndex, y + iT, m + iT, d + iT));
                }
                firstIndex += interval;
            }
        }
    }
    return tags;
}

static std::string toAsciiArt(std::span<trigger::MatchResult> states) {
    bool        started = false;
    std::string r;
    for (auto s : states) {
        switch (s) {
        case trigger::MatchResult::Matching:
            r += started ? "||#" : "|#";
            started = true;
            break;
        case trigger::MatchResult::NotMatching:
            r += started ? "|_" : "_";
            started = false;
            break;
        case trigger::MatchResult::Ignore: r += started ? "#" : "_"; break;
        }
    };
    return r;
}

template<trigger::Matcher TMatcher>
std::string runMatcherTest(std::span<const Tag> tags, TMatcher matcher) {
    std::vector<trigger::MatchResult> result;
    result.reserve(tags.size());
    for (const auto& tag : tags) {
        result.push_back(matcher("", tag, {}));
    }
    return toAsciiArt(result);
}

bool spinUntil(std::chrono::milliseconds timeout, auto fnc) {
    const auto start = std::chrono::steady_clock::now();

    while (std::chrono::steady_clock::now() - start < timeout) {
        if (fnc()) {
            return true;
        }
    }

    return false;
}

trigger::MatchResult isTrigger(std::string_view /* filterSpec */, const Tag& tag, const property_map& /* filter state */) {
    const auto v = tag.get(gr::tag::TRIGGER_NAME.shortKey());
    return (v && std::get<std::string>(v->get()) == "TRIGGER") ? trigger::MatchResult::Matching : trigger::MatchResult::Ignore;
};

struct DataSetTestParams {
    std::size_t nSignalValues  = 0;
    std::string signalName     = "TestName";
    std::string signalUnit     = "TestUnit";
    std::string signalQuantity = "TestQuantity";
    float       rangeMin       = -2.f;
    float       rangeMax       = 2.f;

    std::ptrdiff_t timingEventIndex = 0;
    std::string    triggerName      = "TRIGGER";
    std::uint64_t  triggerTimeMax   = 0;
};

void checkDataSet(const DataSet<float>& dataset, std::vector<float>& receivedData, std::size_t& nReceivedTags, DataSetTestParams expectedParams = {}, std::source_location location = std::source_location::current()) {
    using namespace boost::ut;
    using namespace gr::tag;

    const auto                     locationStr = std::format("{}:{} ", location.file_name(), location.line());
    std::expected<void, gr::Error> dsCheck     = dataset::checkConsistency(dataset, "dataset - blocking polling trigger mode overlapping");
    expect(dsCheck.has_value()) << [&] { return std::format("unexpected: {}", dsCheck.error()); } << fatal;
    expect(eq(dataset.size(), 1UZ)) << locationStr;

    expect(eq(dataset.signal_values.size(), expectedParams.nSignalValues)) << locationStr;

    expect(eq(dataset.signalName(0UZ), expectedParams.signalName)) << locationStr;
    expect(eq(dataset.signalUnit(0UZ), expectedParams.signalUnit)) << locationStr;
    expect(eq(dataset.signalQuantity(0UZ), expectedParams.signalQuantity)) << locationStr;
    expect(eq(dataset.signalRange(0UZ), gr::Range<float>{expectedParams.rangeMin, expectedParams.rangeMax})) << locationStr;

    expect(eq(dataset.timingEvents(0UZ).size(), 1UZ)) << locationStr; // only trigger is stored
    expect(eq(dataset.timingEvents(0UZ)[0UZ].first, static_cast<std::ptrdiff_t>(expectedParams.timingEventIndex))) << locationStr;
    const auto eventMap = dataset.timingEvents(0UZ)[0UZ].second;
    expect(eq(eventMap.size(), 2UZ)) << locationStr;
    const bool containsTriggerName = eventMap.contains(TRIGGER_NAME.shortKey());
    expect(containsTriggerName) << locationStr;
    if (containsTriggerName) {
        expect(eq(std::get<std::string>(eventMap.at(TRIGGER_NAME.shortKey())), expectedParams.triggerName)) << locationStr;
    }
    const bool containsTriggerTime = eventMap.contains(TRIGGER_TIME.shortKey());
    expect(containsTriggerTime) << locationStr;
    if (containsTriggerTime) {
        expect(lt(std::get<std::uint64_t>(eventMap.at(TRIGGER_TIME.shortKey())), static_cast<std::uint64_t>(expectedParams.triggerTimeMax))) << locationStr;
    }

    receivedData.insert(receivedData.end(), dataset.signal_values.begin(), dataset.signal_values.end());
    nReceivedTags += dataset.timingEvents(0UZ).size();
};

} // namespace gr::basic::data_sink_test

using Scheduler = gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded>;

const boost::ut::suite DataSinkTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::basic;
    using namespace gr::basic::data_sink_test;
    using namespace std::string_literals;
    using namespace gr::test;

    "continuous mode - callback"_test = [] {
        using namespace gr::tag;
        constexpr gr::Size_t  kSamples      = 200005;
        constexpr std::size_t kMaxChunkSize = 1000;

        const std::vector<Tag> srcTags = makeTestTags(0, 1234, 1);

        gr::Graph testGraph;
        auto&     src   = testGraph.emplaceBlock<gr::testing::TagSource<float>>({{"n_samples_max", kSamples}, {"mark_tag", false}, //
                  {SIGNAL_NAME.shortKey(), "TestName"}, {SIGNAL_UNIT.shortKey(), "TestUnit"}, {SIGNAL_QUANTITY.shortKey(), "TestQuantity"}, {SIGNAL_MIN.shortKey(), -42.f}, {SIGNAL_MAX.shortKey(), 42.f}});
        auto&     delay = testGraph.emplaceBlock<testing::Delay<float>>({{"delay_ms", kProcessingDelayMs}});
        delay.settings().autoForwardParameters().insert({"DAY", "MONTH", "YEAR"}); // custom auto forward keys
        auto& sink = testGraph.emplaceBlock<DataSink<float>>({{"name", "test_sink"}, {SIGNAL_NAME.shortKey(), "TestName"}});
        src._tags  = srcTags;

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(delay)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(delay).to<"in">(sink)));

        std::atomic<std::size_t> samplesSeen1 = 0;
        std::atomic<std::size_t> chunksSeen1  = 0;
        auto                     callback     = [&samplesSeen1, &chunksSeen1, &kMaxChunkSize](std::span<const float> buffer) {
            for (std::size_t i = 0; i < buffer.size(); ++i) {
                expect(eq(buffer[i], static_cast<float>(samplesSeen1 + i)));
            }
            samplesSeen1 += buffer.size();
            chunksSeen1++;
            expect(le(buffer.size(), kMaxChunkSize));
        };

        std::mutex       m2;
        std::size_t      samplesSeen2 = 0;
        std::size_t      chunksSeen2  = 0;
        std::vector<Tag> receivedTags;
        auto             callbackWithTags = [&samplesSeen2, &chunksSeen2, &m2, &receivedTags, &kMaxChunkSize](std::span<const float> buffer, std::span<const Tag> tags) {
            for (std::size_t i = 0; i < buffer.size(); ++i) {
                expect(eq(buffer[i], static_cast<float>(samplesSeen2 + i)));
            }

            for (const auto& tag : tags) {
                expect(ge(tag.index, 0UZ));
                expect(lt(tag.index, buffer.size()));
            }

            auto lg       = std::lock_guard{m2};
            auto absolute = tags | std::views::transform([&samplesSeen2](const auto& t) { return gr::Tag{t.index + samplesSeen2, t.map}; });
            receivedTags.insert(receivedTags.end(), absolute.begin(), absolute.end());
            samplesSeen2 += buffer.size();
            chunksSeen2++;
            expect(le(buffer.size(), kMaxChunkSize));
        };

        auto callbackWithTagsAndSink = [&sink](std::span<const float>, std::span<const Tag>, const DataSink<float>& passedSink) {
            expect(eq(passedSink.name.value, "test_sink"s));
            expect(eq(sink.unique_name, passedSink.unique_name));
        };

        auto registerThread = std::thread([&] {
            gr::thread_pool::thread::setThreadName("qa_DS:registerThread");
            bool callbackRegistered                = false;
            bool callbackWithTagsRegistered        = false;
            bool callbackWithTagsAndSinkRegistered = false;

            expect(spinUntil(4s, [&] {
                if (!callbackRegistered) {
                    callbackRegistered = globalDataSinkRegistry().registerStreamingCallback<float>(DataSinkQuery::sinkName("test_sink"), kMaxChunkSize, callback);
                }
                if (!callbackWithTagsRegistered) {
                    callbackWithTagsRegistered = globalDataSinkRegistry().registerStreamingCallback<float>(DataSinkQuery::sinkName("test_sink"), kMaxChunkSize, callbackWithTags);
                }
                if (!callbackWithTagsAndSinkRegistered) {
                    callbackWithTagsAndSinkRegistered = globalDataSinkRegistry().registerStreamingCallback<float>(DataSinkQuery::sinkName("test_sink"), kMaxChunkSize, callbackWithTagsAndSink);
                }
                return callbackRegistered && callbackWithTagsRegistered && callbackWithTagsAndSinkRegistered;
            })) << boost::ut::fatal;
        });

        Scheduler sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());
        registerThread.join();

        expect(eq(src._nSamplesProduced, kSamples));

        auto lg = std::lock_guard{m2};
        expect(ge(chunksSeen1.load(), 201UZ));
        expect(ge(chunksSeen2, 201UZ));
        expect(eq(samplesSeen1.load(), static_cast<std::size_t>(kSamples)));
        expect(eq(samplesSeen2, static_cast<std::size_t>(kSamples)));

        std::vector<Tag> srcAndMetaTags;
        // Add metadata tag published by DataSink
        srcAndMetaTags.emplace_back(0UZ, property_map{{SAMPLE_RATE.shortKey(), 1.f}, {SIGNAL_MAX.shortKey(), 42.f}, {SIGNAL_MIN.shortKey(), -42.f}, //
                                             {SIGNAL_NAME.shortKey(), "TestName"}, {SIGNAL_QUANTITY.shortKey(), "TestQuantity"}, {SIGNAL_UNIT.shortKey(), "TestUnit"}});
        // Add tag published by TagSource init with parameters
        srcAndMetaTags.emplace_back(0UZ, property_map{{SIGNAL_MAX.shortKey(), 42.f}, {SIGNAL_MIN.shortKey(), -42.f}, {SIGNAL_NAME.shortKey(), "TestName"}, {SIGNAL_UNIT.shortKey(), "TestUnit"}, //
                                             {SIGNAL_QUANTITY.shortKey(), "TestQuantity"}});

        // Add sources tags
        srcAndMetaTags.insert(srcAndMetaTags.end(), srcTags.begin(), srcTags.end());
        expect(gr::testing::equal_tag_lists(receivedTags, srcAndMetaTags));
    };

    "continuous mode - blocking/non-blocking polling"_test = [](bool isBlocking) {
        using namespace gr::tag;
        OverflowPolicy       overflowPolicy = isBlocking ? OverflowPolicy::Backpressure : OverflowPolicy::Drop;
        constexpr gr::Size_t kSamples       = 200005;

        gr::Graph  testGraph;
        const auto srcTags = makeTestTags(0, 1234, 2);
        auto&      src     = testGraph.emplaceBlock<gr::testing::TagSource<float>>({{"n_samples_max", kSamples}, {"mark_tag", false}, //
                     {SIGNAL_NAME.shortKey(), "TestName"}, {SIGNAL_UNIT.shortKey(), "TestUnit"}, {SIGNAL_QUANTITY.shortKey(), "TestQuantity"}, {SIGNAL_MIN.shortKey(), -42.f}, {SIGNAL_MAX.shortKey(), 42.f}});
        src._tags          = srcTags;
        auto& delay        = testGraph.emplaceBlock<testing::Delay<float>>({{"delay_ms", kProcessingDelayMs}});
        delay.settings().autoForwardParameters().insert({"DAY", "MONTH", "YEAR"}); // custom aut forward keys
        auto& sink = testGraph.emplaceBlock<DataSink<float>>({{"name", "test_sink"}, {SIGNAL_NAME.shortKey(), "TestName"}});

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(delay)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(delay).to<"in">(sink)));

        auto invalidTypePoller = globalDataSinkRegistry().getStreamingPoller<double>(DataSinkQuery::sinkName("test_sink"));
        expect(eq(invalidTypePoller, nullptr));

        auto runner1 = std::async([overflowPolicy] {
            std::shared_ptr<StreamingPoller<float>> poller;
            expect(spinUntil(4s, [&poller, overflowPolicy] {
                poller = globalDataSinkRegistry().getStreamingPoller<float>(DataSinkQuery::sinkName("test_sink"), {.overflowPolicy = overflowPolicy});
                return poller != nullptr;
            })) << boost::ut::fatal;

            std::vector<float> received;
            bool               seenFinished = false;
            while (!seenFinished) {
                std::this_thread::sleep_for(50ms); // always wait to force full buffer
                seenFinished = poller->finished;
                while (poller->process([&received](const auto& data) { received.insert(received.end(), data.begin(), data.end()); })) {
                }
            }

            return std::make_tuple(poller, received);
        });

        auto runner2 = std::async([overflowPolicy] {
            std::shared_ptr<StreamingPoller<float>> poller;
            expect(spinUntil(4s, [&poller, overflowPolicy] {
                poller = globalDataSinkRegistry().getStreamingPoller<float>(DataSinkQuery::signalName("TestName"), {.overflowPolicy = overflowPolicy});
                return poller != nullptr;
            })) << boost::ut::fatal;
            std::vector<float> received;
            std::vector<Tag>   receivedTags;
            bool               seenFinished = false;
            while (!seenFinished) {
                std::this_thread::sleep_for(50ms); // always wait to force full buffer
                seenFinished = poller->finished;
                while (poller->process([&received, &receivedTags](const auto& data, const auto& tags_) {
                    auto absolute = tags_ | std::views::transform([&received](const auto& t) { return gr::Tag{t.index + received.size(), t.map}; });
                    receivedTags.insert(receivedTags.end(), absolute.begin(), absolute.end());
                    received.insert(received.end(), data.begin(), data.end());
                })) {
                }
            }

            return std::make_tuple(poller, received, receivedTags);
        });

        {
            Scheduler sched;
            if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
                throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
            }
            expect(sched.runAndWait().has_value());

            const auto pollerAfterStop = globalDataSinkRegistry().getStreamingPoller<float>(DataSinkQuery::sinkName("test_sink"));
            expect(eq(pollerAfterStop, nullptr));
        }

        const auto pollerAfterDestruction = globalDataSinkRegistry().getStreamingPoller<float>(DataSinkQuery::sinkName("test_sink"));
        expect(!pollerAfterDestruction);

        std::vector<float> expected(kSamples);
        std::iota(expected.begin(), expected.end(), 0.0);

        const auto& [pollerDataOnly, received1]               = runner1.get();
        const auto& [pollerWithTags, received2, receivedTags] = runner2.get();

        if (isBlocking) {
            std::vector<Tag> srcAndMetaTags;
            // Add metadata tag published by DataSink
            srcAndMetaTags.emplace_back(0UZ, property_map{{SAMPLE_RATE.shortKey(), 1.f}, {SIGNAL_MAX.shortKey(), 42.f}, {SIGNAL_MIN.shortKey(), -42.f}, //
                                                 {SIGNAL_NAME.shortKey(), "TestName"}, {SIGNAL_QUANTITY.shortKey(), "TestQuantity"}, {SIGNAL_UNIT.shortKey(), "TestUnit"}});
            // Add tag published by TagSource init with parameters
            srcAndMetaTags.emplace_back(0UZ, property_map{{SIGNAL_MAX.shortKey(), 42.f}, {SIGNAL_MIN.shortKey(), -42.f}, {SIGNAL_NAME.shortKey(), "TestName"}, {SIGNAL_UNIT.shortKey(), "TestUnit"}, //
                                                 {SIGNAL_QUANTITY.shortKey(), "TestQuantity"}});
            // Add sources tags
            srcAndMetaTags.insert(srcAndMetaTags.end(), srcTags.begin(), srcTags.end());
            expect(gr::testing::equal_tag_lists(receivedTags, srcAndMetaTags));

            expect(eq(received1.size(), expected.size()));
            expect(eq(received1, expected));
            expect(eq(received2.size(), expected.size()));
            expect(eq(received2, expected));

            expect(eq(pollerWithTags->droppedSampleCount.load(), 0UZ));
            expect(eq(pollerWithTags->droppedTagCount.load(), 0UZ));
            expect(eq(pollerDataOnly->droppedSampleCount.load(), 0UZ));
            expect(eq(pollerDataOnly->droppedTagCount.load(), 0UZ));
        } else {
            expect(eq(received1.size() + pollerDataOnly->droppedSampleCount.load(), expected.size()));
            expect(eq(received2.size() + pollerWithTags->droppedSampleCount.load(), expected.size()));
            expect(gt(pollerWithTags->droppedSampleCount.load(), 0UZ));
            expect(gt(pollerDataOnly->droppedSampleCount.load(), 0UZ));
        }
    } | std::vector<bool>{true, false};

    "trigger mode - polling/callback overlapping/non-overlapping"_test = [] {
        using namespace gr::tag;
        const std::uint32_t nSamples    = 150000;
        const std::size_t   preSamples  = 5;
        const std::size_t   postSamples = 7;

        const std::vector<std::size_t> triggerIndices = {1001, 1001, 1002, 1003, 1003, 1005, 1007, 10000, 10000, 20000};
        std::vector<Tag>               srcTags;
        std::uint64_t                  timeCounter = 0;
        for (std::size_t i : triggerIndices) {
            // tags should not be identical, duplicates are ignored by Block::inputTags()
            srcTags.push_back(Tag{i, {{TRIGGER_NAME.shortKey(), "TRIGGER"}, {TRIGGER_TIME.shortKey(), timeCounter++}}});
        }
        srcTags.push_back(Tag{21000, {{TRIGGER_NAME.shortKey(), "NO_TRIGGER1"}, {TRIGGER_TIME.shortKey(), timeCounter + 1}}});
        srcTags.push_back(Tag{21000, {{TRIGGER_NAME.shortKey(), "NO_TRIGGER2"}, {TRIGGER_TIME.shortKey(), timeCounter + 2}}});
        srcTags.push_back(Tag{22000, {{TRIGGER_NAME.shortKey(), "NO_TRIGGER3"}, {TRIGGER_TIME.shortKey(), timeCounter + 3}}});

        gr::Graph testGraph;
        auto&     src = testGraph.emplaceBlock<gr::testing::TagSource<float>>({{"n_samples_max", nSamples}, {"mark_tag", false}, {"verbose_console", true}, //
                {SIGNAL_NAME.shortKey(), "TestName"}, {SIGNAL_UNIT.shortKey(), "TestUnit"}, {SIGNAL_QUANTITY.shortKey(), "TestQuantity"}, {SIGNAL_MIN.shortKey(), -2.f}, {SIGNAL_MAX.shortKey(), 2.f}});

        src._tags   = srcTags;
        auto& delay = testGraph.emplaceBlock<testing::Delay<float>>({{"delay_ms", kProcessingDelayMs}});
        auto& sink  = testGraph.emplaceBlock<DataSink<float>>({{"name", "test_sink"}, {"signal_name", "TestName"}});

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(delay)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(delay).to<"in">(sink)));

        std::mutex         mutexCallback;
        std::vector<float> receivedDataCallback;
        std::size_t        nReceivedTagsCallback = 0;
        auto               callback              = [&receivedDataCallback, &nReceivedTagsCallback, &mutexCallback, timeCounter](auto&& dataset) {
            std::lock_guard lg{mutexCallback};
            checkDataSet(dataset, receivedDataCallback, nReceivedTagsCallback, {.nSignalValues = preSamples + postSamples, .timingEventIndex = preSamples, .triggerTimeMax = timeCounter});
        };
        auto callbackThread = std::thread([&] { //
            gr::thread_pool::thread::setThreadName("qa_DS:registerThread");
            expect(spinUntil(4s, [&] { return globalDataSinkRegistry().registerTriggerCallback<float>(DataSinkQuery::sinkName("test_sink"), isTrigger, preSamples, postSamples, callback); })) << boost::ut::fatal;
        });

        auto polling = std::async([timeCounter, preSamples, postSamples] {
            std::shared_ptr<DataSetPoller<float>> poller;
            expect(spinUntil(4s, [&] {
                poller = globalDataSinkRegistry().getTriggerPoller<float>(DataSinkQuery::sinkName("test_sink"), isTrigger, {.preSamples = preSamples, .postSamples = postSamples});
                return poller != nullptr;
            })) << boost::ut::fatal;
            std::vector<float> receivedData;
            std::size_t        nReceivedTags = 0;
            bool               seenFinished  = false;
            while (!seenFinished) {
                seenFinished = poller->finished.load();
                while (poller->process([&receivedData, &nReceivedTags, timeCounter, preSamples, postSamples](const auto& datasets) {
                    for (const auto& dataset : datasets) {
                        checkDataSet(dataset, receivedData, nReceivedTags, {.nSignalValues = preSamples + postSamples, .timingEventIndex = preSamples, .triggerTimeMax = timeCounter});
                    }
                })) {
                }
            }
            return std::make_tuple(poller, receivedData, nReceivedTags);
        });

        Scheduler sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());
        callbackThread.join();

        expect(eq(src._nSamplesProduced, nSamples));

        const auto& [poller, receivedData, nReceivedTags] = polling.get();
        expect(eq(poller->dropCount.load(), 0u));

        std::vector<float> expectedData;
        expectedData.reserve(triggerIndices.size() * (preSamples + postSamples));
        for (std::size_t i : triggerIndices) {
            for (std::size_t k = i - preSamples; k < i + postSamples; k++) {
                expectedData.push_back(static_cast<float>(k));
            }
        }

        // polling
        expect(eq(receivedData.size(), triggerIndices.size() * (preSamples + postSamples)));
        expect(eq(receivedData, expectedData));
        expect(eq(nReceivedTags, triggerIndices.size()));

        // callback
        expect(eq(receivedDataCallback.size(), triggerIndices.size() * (preSamples + postSamples)));
        expect(eq(receivedDataCallback, expectedData));
        expect(eq(nReceivedTagsCallback, triggerIndices.size()));
    };

    "snapshot mode - polling/callback"_test = [] {
        using namespace gr::tag;
        constexpr gr::Size_t kSamples      = 200000;
        constexpr float      kSampleRate   = 10000.f;
        constexpr auto       kDelay        = std::chrono::milliseconds{500}; // sample rate 10000 -> 5000 samples
        const double         seconds       = std::chrono::duration<double>(kDelay).count();
        const std::size_t    nSamplesDelay = static_cast<std::size_t>(seconds * static_cast<double>(kSampleRate));

        const std::vector<std::size_t> triggerIndices = {1001, 1001, 1002, 1003, 1003, 1005, 1007, 10000, 10000, 20000};
        std::vector<Tag>               srcTags;
        std::uint64_t                  timeCounter = 0;
        for (std::size_t i : triggerIndices) {
            srcTags.push_back(Tag{i, {{TRIGGER_NAME.shortKey(), "TRIGGER"}, {TRIGGER_TIME.shortKey(), timeCounter++}}});
        }
        srcTags.push_back(Tag{21000, {{TRIGGER_NAME.shortKey(), "NO_TRIGGER1"}, {TRIGGER_TIME.shortKey(), timeCounter + 1}}});
        srcTags.push_back(Tag{21000, {{TRIGGER_NAME.shortKey(), "NO_TRIGGER2"}, {TRIGGER_TIME.shortKey(), timeCounter + 2}}});
        srcTags.push_back(Tag{22000, {{TRIGGER_NAME.shortKey(), "NO_TRIGGER3"}, {TRIGGER_TIME.shortKey(), timeCounter + 3}}});

        gr::Graph testGraph;
        auto&     src = testGraph.emplaceBlock<gr::testing::TagSource<float>>({{"n_samples_max", kSamples}, {"mark_tag", false},                                                 //
                {SAMPLE_RATE.shortKey(), kSampleRate}, {SIGNAL_NAME.shortKey(), "TestName"}, {SIGNAL_UNIT.shortKey(), "TestUnit"}, {SIGNAL_QUANTITY.shortKey(), "TestQuantity"}, //
                {SIGNAL_MIN.shortKey(), 0.f}, {SIGNAL_MAX.shortKey(), static_cast<float>(kSamples - 1)}});
        src._tags     = srcTags;
        auto& delay   = testGraph.emplaceBlock<testing::Delay<float>>({{"delay_ms", kProcessingDelayMs}});
        auto& sink    = testGraph.emplaceBlock<DataSink<float>>({{"name", "test_sink"}, {SIGNAL_NAME.shortKey(), "TestName"}});

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(delay)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(delay).to<"in">(sink)));

        DataSetTestParams dataSetTestParams = {.nSignalValues = 1, .rangeMin = 0.f, .rangeMax = static_cast<float>(kSamples - 1.), .timingEventIndex = -static_cast<std::ptrdiff_t>(nSamplesDelay), .triggerTimeMax = timeCounter};

        std::vector<float> receivedDataCallback;
        std::size_t        nReceivedTagsCallback = 0;
        auto               callback              = [&receivedDataCallback, &nReceivedTagsCallback, &dataSetTestParams](const auto& dataset) { //
            checkDataSet(dataset, receivedDataCallback, nReceivedTagsCallback, dataSetTestParams);
        };

        auto callbackThread = std::thread([&] {
            gr::thread_pool::thread::setThreadName("qa_DS:registerThread");
            expect(spinUntil(4s, [&] { return globalDataSinkRegistry().registerSnapshotCallback<float>(DataSinkQuery::sinkName("test_sink"), isTrigger, kDelay, callback); })) << boost::ut::fatal;
        });

        auto polling = std::async([kDelay, &dataSetTestParams] {
            std::shared_ptr<DataSetPoller<float>> poller;
            expect(spinUntil(4s, [&] {
                poller = globalDataSinkRegistry().getSnapshotPoller<float>(DataSinkQuery::sinkName("test_sink"), isTrigger, {.delay = kDelay});
                return poller != nullptr;
            })) << boost::ut::fatal;

            std::vector<float> receivedData;
            std::size_t        nReceivedTags = 0;

            bool seenFinished = false;
            while (!seenFinished) {
                seenFinished            = poller->finished;
                [[maybe_unused]] auto r = poller->process([&receivedData, &nReceivedTags, &dataSetTestParams](const auto& datasets) {
                    for (const auto& dataset : datasets) {
                        checkDataSet(dataset, receivedData, nReceivedTags, dataSetTestParams);
                    }
                });
            }
            return std::make_tuple(poller, receivedData, nReceivedTags);
        });

        Scheduler sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());
        callbackThread.join();

        expect(eq(src._nSamplesProduced, kSamples));

        const auto& [poller, receivedData, nReceivedTags] = polling.get();
        expect(eq(poller->dropCount.load(), 0u));

        std::vector<float> expectedData;
        std::ranges::transform(triggerIndices, std::back_inserter(expectedData), [nSamplesDelay](std::size_t i) { return static_cast<float>(i + nSamplesDelay); });

        // polling
        expect(eq(receivedData.size(), triggerIndices.size()));
        expect(eq(receivedData, expectedData));
        expect(eq(nReceivedTags, triggerIndices.size()));

        // callback
        expect(eq(receivedDataCallback.size(), triggerIndices.size()));
        expect(eq(receivedDataCallback, expectedData));
        expect(eq(nReceivedTagsCallback, triggerIndices.size()));
    };

    "multiplexed mode - blocking polling"_test = [] {
        // Use large delay and timeout to ensure this also works in debug/coverage builds
        const auto tags = makeTestTags(0, 10000);

        const gr::Size_t n_samples = static_cast<gr::Size_t>(tags.size() * 10000 + 100000);
        gr::Graph        testGraph;
        auto&            src = testGraph.emplaceBlock<gr::testing::TagSource<int32_t>>({{"n_samples_max", n_samples}, {"mark_tag", false}});
        src._tags            = tags;
        auto& delay          = testGraph.emplaceBlock<testing::Delay<int32_t>>({{"delay_ms", 2500u}});
        delay.settings().autoForwardParameters().insert({"DAY", "MONTH", "YEAR"}); // custom auto forward keys
        auto& sink = testGraph.emplaceBlock<DataSink<int32_t>>({{"name", "test_sink"}, {"signal_name", "test signal"}});

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(delay)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(delay).to<"in">(sink)));

        {
            const auto t = std::span(tags);

            // Test the test matcher
            expect(eq(runMatcherTest(t, Matcher({}, -1, {})), "|###||###||###||###||###||###"s));
            expect(eq(runMatcherTest(t, Matcher(-1, {}, {})), "|######||######||######"s));
            expect(eq(runMatcherTest(t, Matcher(1, {}, {})), "|######|____________"s));
            expect(eq(runMatcherTest(t, Matcher(1, {}, 2)), "_|#|__|#|_____________"s));
            expect(eq(runMatcherTest(t, Matcher({}, {}, 1)), "|#|__|#|__|#|__|#|__|#|__|#|__"s));
        }

        const auto matchers = std::array{Matcher({}, -1, {}), Matcher(-1, {}, {}), Matcher(1, {}, {}), Matcher(1, {}, 2), Matcher({}, {}, 1)};

        // Following the patterns above, where each #/_ is 10000 samples
        const auto                                        expected = std::array<std::vector<int32_t>, matchers.size()>{{{0, 29999, 30000, 59999, 60000, 89999, 90000, 119999, 120000, 149999, 150000, 249999}, {0, 59999, 60000, 119999, 120000, 219999}, {0, 59999}, {10000, 19999, 40000, 49999}, {0, 9999, 30000, 39999, 60000, 69999, 90000, 99999, 120000, 129999, 150000, 159999}}};
        std::vector<std::future<std::vector<int32_t>>>    results;
        std::array<std::vector<int32_t>, matchers.size()> resultsCb;

        auto registerThread = std::thread([&] {
            gr::thread_pool::thread::setThreadName("qa_DS:registerThread");
            std::array<bool, resultsCb.size()> registered;
            std::ranges::fill(registered, false);
            expect(spinUntil(3s, [&] {
                for (auto i = 0UZ; i < registered.size(); ++i) {
                    if (!registered[i]) {
                        auto callback = [&r = resultsCb[i]](const auto& dataset) {
                            r.push_back(dataset.signal_values.front());
                            r.push_back(dataset.signal_values.back());
                        };
                        registered[i] = globalDataSinkRegistry().registerMultiplexedCallback<int32_t>(DataSinkQuery::sinkName("test_sink"), Matcher(matchers[i]), 100000, std::move(callback));
                    }
                }
                return std::ranges::all_of(registered, [](bool b) { return b; });
            })) << boost::ut::fatal;
        });

        for (std::size_t i = 0; i < matchers.size(); ++i) {
            auto f = std::async([i, &matchers]() {
                std::shared_ptr<DataSetPoller<int32_t>> poller;
                expect(spinUntil(4s, [&] {
                    poller = globalDataSinkRegistry().getMultiplexedPoller<int32_t>(DataSinkQuery::sinkName("test_sink"), Matcher(matchers[i]), {.maximumWindowSize = 100000});
                    return poller != nullptr;
                })) << boost::ut::fatal;
                std::vector<int32_t> ranges;
                bool                 seenFinished = false;
                while (!seenFinished) {
                    seenFinished = poller->finished.load();
                    while (poller->process([&ranges](const auto& datasets) {
                        for (const auto& dataset : datasets) {
                            std::expected<void, gr::Error> dsCheck = dataset::checkConsistency(dataset, "dataset - blocking multiplexed mode");
                            expect(dsCheck.has_value()) << [&] { return std::format("unexpected: {}", dsCheck.error()); } << fatal;
                            expect(eq(dataset.size(), 1UZ)) << "DataSink supports only 1 signal per DataSet<T> (for the time being)";

                            // default signal info, we didn't set anything
                            expect(eq(dataset.signalName(0UZ), "test signal"s));
                            expect(eq(dataset.signalUnit(0UZ), "a.u."s));
                            ranges.push_back(dataset.signal_values.front());
                            ranges.push_back(dataset.signal_values.back());
                        }
                    })) {
                    }
                }
                return ranges;
            });
            results.push_back(std::move(f));
        }

        Scheduler sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());
        registerThread.join();

        for (std::size_t i = 0UZ; i < results.size(); ++i) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnull-dereference" // false positive for GCC13
            expect(eq(results[i].get(), expected[i]));
            expect(eq(resultsCb[i], expected[i]));
#pragma GCC diagnostic pop
        }
    };

    "DataSet - polling"_test = [] {
        gr::Graph testGraph;
        auto&     source          = testGraph.emplaceBlock<testing::TagSource<float, testing::ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", static_cast<gr::Size_t>(1024)}, {"signal_name", "test signal"}, {"signal_unit", "test unit"}, {"mark_tag", false}});
        auto&     delay           = testGraph.emplaceBlock<testing::Delay<float>>({{"delay_ms", kProcessingDelayMs}});
        auto&     streamToDataSet = testGraph.emplaceBlock<StreamToDataSet<float>>({{"filter", "CMD_DIAG_TRIGGER1"}, {"n_pre", static_cast<gr::Size_t>(100)}, {"n_post", static_cast<gr::Size_t>(200)}});
        auto&     sink            = testGraph.emplaceBlock<DataSetSink<float>>({{"name", "test_sink"}, {"signal_name", "test signal"}});
        expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect<"out">(source).to<"in">(delay)));
        expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect<"out">(delay).to<"in">(streamToDataSet)));
        expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect<"out">(streamToDataSet).to<"in">(sink)));

        auto genTrigger = [](std::size_t index, std::string triggerName, std::string triggerCtx = {}) {
            return Tag{index, {{gr::tag::TRIGGER_NAME.shortKey(), triggerName}, {gr::tag::TRIGGER_TIME.shortKey(), std::uint64_t(0)}, {gr::tag::TRIGGER_OFFSET.shortKey(), 0.f}, //
                                  {gr::tag::CONTEXT.shortKey(), triggerCtx},                                                                                                     //
                                  {gr::tag::TRIGGER_META_INFO.shortKey(), gr::property_map{}}}};
        };

        source._tags.push_back(genTrigger(400, "CMD_DIAG_TRIGGER1", ""));
        source._tags.push_back(genTrigger(800, "CMD_DIAG_TRIGGER1", ""));

        auto polling = std::async([] {
            std::shared_ptr<DataSetPoller<float>> poller;
            expect(spinUntil(4s, [&poller] {
                poller = globalDataSinkRegistry().getDataSetPoller<float>(DataSinkQuery::sinkName("test_sink"));
                return poller != nullptr;
            })) << boost::ut::fatal;
            std::vector<DataSet<float>> receivedDataSets;
            bool                        seenFinished = false;
            while (!seenFinished) {
                seenFinished = poller->finished.load();
                while (poller->process([&receivedDataSets](const auto& dataSets) { receivedDataSets.insert(receivedDataSets.end(), dataSets.begin(), dataSets.end()); })) {
                }
            }
            return std::pair(poller, receivedDataSets);
        });

        Scheduler sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());

        const auto& [poller, receivedDataSets] = polling.get();
        expect(eq(receivedDataSets.size(), 2UZ));
        std::expected<void, gr::Error> dsCheck = dataset::checkConsistency(receivedDataSets[1UZ], "receivedDataSets[1UZ]");
        expect(dsCheck.has_value()) << [&] { return std::format("unexpected: {}", dsCheck.error()); } << fatal;
        expect(eq(receivedDataSets[1UZ].size(), 1UZ)) << "DataSink supports only 1 signal per DataSet<T> (for the time being)";

        expect(eq_collections(receivedDataSets[0UZ].signalValues(0UZ), getIota(300, 300.f)));
        expect(eq(receivedDataSets[0UZ].signalName(0UZ), "test signal"s));
        expect(eq(receivedDataSets[0UZ].signalUnit(0UZ), "test unit"s));
        expect(eq(receivedDataSets[0UZ].timingEvents(0UZ).size(), 1UZ));
        expect(eq(receivedDataSets[0UZ].timingEvents(0UZ)[0UZ].first, 100));
        expect(eq_collections(receivedDataSets[1UZ].signalValues(0UZ), getIota(300, 700.f)));
        expect(eq(receivedDataSets[1UZ].signalName(0UZ), "test signal"s));
        expect(eq(receivedDataSets[1UZ].signalUnit(0UZ), "test unit"s));
        expect(eq(receivedDataSets[1UZ].timingEvents(0UZ).size(), 1UZ));
        expect(eq(receivedDataSets[1UZ].timingEvents(0UZ)[0UZ].first, 100));
    };

    "DataSet - callback"_test = [] {
        gr::Graph testGraph;
        auto&     source          = testGraph.emplaceBlock<testing::TagSource<float, testing::ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", static_cast<gr::Size_t>(1024)}, {"signal_name", "test signal"}, {"signal_unit", "test unit"}, {"mark_tag", false}});
        auto&     delay           = testGraph.emplaceBlock<testing::Delay<float>>({{"delay_ms", kProcessingDelayMs}});
        auto&     streamToDataSet = testGraph.emplaceBlock<StreamToDataSet<float>>({{"filter", "CMD_DIAG_TRIGGER1"}, {"n_pre", static_cast<gr::Size_t>(100)}, {"n_post", static_cast<gr::Size_t>(200)}});
        auto&     sink            = testGraph.emplaceBlock<DataSetSink<float>>({{"name", "test_sink"}, {"signal_name", "test signal"}});
        expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect<"out">(source).to<"in">(delay)));
        expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect<"out">(delay).to<"in">(streamToDataSet)));
        expect(eq(gr::ConnectionResult::SUCCESS, testGraph.connect<"out">(streamToDataSet).to<"in">(sink)));

        auto genTrigger = [](std::size_t index, std::string triggerName, std::string triggerCtx = {}) {
            return Tag{index, {{gr::tag::TRIGGER_NAME.shortKey(), triggerName}, {gr::tag::TRIGGER_TIME.shortKey(), std::uint64_t(0)}, {gr::tag::TRIGGER_OFFSET.shortKey(), 0.f}, //
                                  {gr::tag::CONTEXT.shortKey(), triggerCtx},                                                                                                     //
                                  {gr::tag::TRIGGER_META_INFO.shortKey(), gr::property_map{}}}};
        };

        source._tags.push_back(genTrigger(400, "CMD_DIAG_TRIGGER1", ""));
        source._tags.push_back(genTrigger(800, "CMD_DIAG_TRIGGER1", ""));

        std::vector<DataSet<float>> receivedDataSets;
        auto                        callback = [&receivedDataSets](const auto& ds) { receivedDataSets.push_back(ds); };

        auto registerThread = std::thread([&] { //
            gr::thread_pool::thread::setThreadName("qa_DS:registerThread");
            expect(spinUntil(4s, [&] { return globalDataSinkRegistry().registerDataSetCallback<float>(DataSinkQuery::sinkName("test_sink"), callback); })) << boost::ut::fatal;
        });

        Scheduler sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());
        registerThread.join();

        expect(eq(receivedDataSets.size(), 2UZ));
        std::expected<void, gr::Error> dsCheck = dataset::checkConsistency(receivedDataSets[1UZ], "receivedDataSets[1UZ]");
        expect(dsCheck.has_value()) << [&] { return std::format("unexpected: {}", dsCheck.error()); } << fatal;
        expect(eq(receivedDataSets[1UZ].size(), 1UZ)) << "DataSink supports only 1 signal per DataSet<T> (for the time being)";

        expect(eq_collections(receivedDataSets[0UZ].signalValues(0UZ), getIota(300, 300.f)));
        expect(eq(receivedDataSets[0UZ].signalName(0UZ), "test signal"s));
        expect(eq(receivedDataSets[0UZ].signalUnit(0UZ), "test unit"s));
        expect(eq(receivedDataSets[0UZ].timingEvents(0UZ).size(), 1UZ));
        expect(eq(receivedDataSets[0UZ].timingEvents(0UZ)[0UZ].first, 100));
        expect(eq_collections(receivedDataSets[1UZ].signalValues(0UZ), getIota(300, 700.f)));
        expect(eq(receivedDataSets[1UZ].signalName(0UZ), "test signal"s));
        expect(eq(receivedDataSets[1UZ].signalUnit(0UZ), "test unit"s));
        expect(eq(receivedDataSets[1UZ].timingEvents(0UZ).size(), 1UZ));
        expect(eq(receivedDataSets[1UZ].timingEvents(0UZ)[0UZ].first, 100));
    };
};

int main() { /* tests are statically executed */ }
