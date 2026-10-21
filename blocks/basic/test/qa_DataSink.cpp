#include <atomic>
#include <future>
#include <ranges>
#include <thread>

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
#include <gnuradio-4.0/testing/Sleep.hpp>
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

    [[nodiscard]] trigger::MatchResult operator()(std::string_view /* filterSpec */, const auto& tag, const property_map& /* filter state */) {
        const auto ty = tag.map.find_value("YEAR");
        const auto tm = tag.map.find_value("MONTH");
        const auto td = tag.map.find_value("DAY");
        if (!ty || !tm || !td) {
            return trigger::MatchResult::Ignore;
        }

        const auto tup        = std::make_tuple(ty->value_or(0), tm->value_or(0), td->value_or(0));
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

static gr::testing::OwningTag makeTag(std::size_t index, int year, int month, int day) { return gr::testing::OwningTag{index, {{"YEAR", year}, {"MONTH", month}, {"DAY", day}}}; }

static std::vector<gr::testing::OwningTag> makeTestTags(std::size_t firstIndex, std::size_t interval, std::size_t nTagsPerIndex = 1UZ) {
    std::vector<gr::testing::OwningTag> tags;
    for (int y = 1; y <= 3; ++y) {
        for (int m = 1; m <= 2; ++m) {
            for (int d = 1; d <= 3; ++d) {
                for (int iT = 0; iT < static_cast<int>(nTagsPerIndex); ++iT) {
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
std::string runMatcherTest(std::span<const gr::testing::OwningTag> tags, TMatcher matcher) {
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
    const auto v = tag.map.find_value(gr::tag::kTriggerName);
    return (v && gr::test::get_value_or_fail<std::string>(*v) == "TRIGGER") ? trigger::MatchResult::Matching : trigger::MatchResult::Ignore;
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
    expect(eq(dataset.size(), 1UZ)) << locationStr << fatal;

    expect(eq(dataset.signal_values.size(), expectedParams.nSignalValues)) << locationStr;

    expect(eq(dataset.signalName(0UZ), expectedParams.signalName)) << locationStr;
    expect(eq(dataset.signalUnit(0UZ), expectedParams.signalUnit)) << locationStr;
    expect(eq(dataset.signalQuantity(0UZ), expectedParams.signalQuantity)) << locationStr;
    expect(eq(dataset.signalRange(0UZ), gr::Range<float>{expectedParams.rangeMin, expectedParams.rangeMax})) << locationStr;

    expect(eq(dataset.timingEvents(0UZ).size(), 1UZ)) << locationStr << fatal;
    expect(eq(dataset.timingEvents(0UZ)[0UZ].first, static_cast<std::ptrdiff_t>(expectedParams.timingEventIndex))) << locationStr;
    const auto eventMap = dataset.timingEvents(0UZ)[0UZ].second;
    expect(eq(eventMap.size(), 2UZ)) << locationStr;
    const bool containsTriggerName = eventMap.contains(kTriggerName);
    expect(containsTriggerName) << locationStr;
    if (containsTriggerName) {
        expect(eq(gr::test::get_value_or_fail<std::string>(eventMap.find_value(kTriggerName).value()), expectedParams.triggerName)) << locationStr;
    }
    const bool containsTriggerTime = eventMap.contains(kTriggerTime);
    expect(containsTriggerTime) << locationStr;
    if (containsTriggerTime) {
        expect(lt(gr::test::get_value_or_fail<std::uint64_t>(eventMap.find_value(kTriggerTime).value()), static_cast<std::uint64_t>(expectedParams.triggerTimeMax))) << locationStr;
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

        const std::vector<gr::testing::OwningTag> srcTags = makeTestTags(0, 1234, 1);

        gr::Graph testGraph;
        auto&     src       = testGraph.emplaceBlock<gr::testing::TagSource<float>>({{"n_samples_max", kSamples}, {"mark_tag", false}, //
                      {kSignalName, "TestName"}, {kSignalUnit, "TestUnit"}, {kSignalQuantity, "TestQuantity"}, {kSignalMin, -42.f}, {kSignalMax, 42.f}});
        auto&     delay     = testGraph.emplaceBlock<testing::Sleep<float>>({{"delay_ms", kProcessingDelayMs}});
        auto&     sink      = testGraph.emplaceBlock<DataSink<float>>({{"name", "test_sink"}, {kSignalName, "TestName"}});
        src._tags           = srcTags;
        sink.in.max_samples = kMaxChunkSize;

        expect(testGraph.connect<"out", "in">(src, delay).has_value());
        expect(testGraph.connect<"out", "in">(delay, sink).has_value());

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

        std::mutex                          m2;
        std::size_t                         samplesSeen2 = 0;
        std::size_t                         chunksSeen2  = 0;
        std::vector<gr::testing::OwningTag> receivedTags;
        auto                                callbackWithTags = [&samplesSeen2, &chunksSeen2, &m2, &receivedTags, &kMaxChunkSize](std::span<const float> buffer, std::span<const Tag> tags) {
            for (std::size_t i = 0; i < buffer.size(); ++i) {
                expect(eq(buffer[i], static_cast<float>(samplesSeen2 + i)));
            }

            for (const auto& tag : tags) {
                expect(ge(tag.index, 0UZ));
                expect(lt(tag.index, buffer.size()));
            }

            auto lg       = std::lock_guard{m2};
            auto absolute = tags | std::views::transform([&samplesSeen2](const auto& t) { return gr::testing::OwningTag{t.index + samplesSeen2, t.map}; });
            receivedTags.insert(receivedTags.end(), absolute.begin(), absolute.end());
            samplesSeen2 += buffer.size();
            chunksSeen2++;
            expect(le(buffer.size(), kMaxChunkSize));
        };

        std::atomic<std::size_t> chunksSeen3             = 0;
        auto                     callbackWithTagsAndSink = [&sink, &chunksSeen3](std::span<const float>, std::span<const Tag>, const DataSink<float>& passedSink) {
            chunksSeen3++;
            expect(eq(std::string_view{passedSink.name.value}, "test_sink"sv));
            expect(eq(sink.unique_name, passedSink.unique_name));
        };

        auto registerThread = std::thread([&] {
            gr::thread_pool::thread::setThreadName("qa_DS:registerThread");
            bool callbackRegistered                = false;
            bool callbackWithTagsRegistered        = false;
            bool callbackWithTagsAndSinkRegistered = false;

            expect(spinUntil(4s, [&] {
                if (!callbackRegistered) {
                    callbackRegistered = globalDataSinkRegistry().registerCallback<float>(DataSinkQuery::sinkName("test_sink"), callback);
                }
                if (!callbackWithTagsRegistered) {
                    callbackWithTagsRegistered = globalDataSinkRegistry().registerCallback<float>(DataSinkQuery::sinkName("test_sink"), callbackWithTags);
                }
                if (!callbackWithTagsAndSinkRegistered) {
                    callbackWithTagsAndSinkRegistered = globalDataSinkRegistry().registerCallback<float>(DataSinkQuery::sinkName("test_sink"), callbackWithTagsAndSink);
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
        expect(ge(chunksSeen3.load(), 201UZ));
        expect(eq(samplesSeen1.load(), static_cast<std::size_t>(kSamples)));
        expect(eq(samplesSeen2, static_cast<std::size_t>(kSamples)));

        std::vector<gr::testing::OwningTag> srcAndMetaTags;
        srcAndMetaTags.emplace_back(0UZ, property_map{{kSampleRate, 1.f}, {kSignalMax, 42.f}, {kSignalMin, -42.f}, //
                                             {kSignalName, "TestName"}, {kSignalQuantity, "TestQuantity"}, {kSignalUnit, "TestUnit"}});
        srcAndMetaTags.emplace_back(0UZ, property_map{{kSignalMax, 42.f}, {kSignalMin, -42.f}, {kSignalName, "TestName"}, {kSignalUnit, "TestUnit"}, //
                                             {kSignalQuantity, "TestQuantity"}});

        srcAndMetaTags.insert(srcAndMetaTags.end(), srcTags.begin(), srcTags.end());
        expect(gr::testing::equal_tag_lists(receivedTags, srcAndMetaTags));
    };

    "continuous mode - blocking/non-blocking polling"_test = [](bool isBlocking) {
        using namespace gr::tag;
        OverflowPolicy       overflowPolicy = isBlocking ? OverflowPolicy::Backpressure : OverflowPolicy::Drop;
        constexpr gr::Size_t kSamples       = 200005;

        gr::Graph                           testGraph;
        std::vector<gr::testing::OwningTag> srcTags = makeTestTags(0, 1, 128); // a burst beyond the poller's tag buffer
        std::ranges::copy(makeTestTags(1000, 11000, 2), std::back_inserter(srcTags));
        auto& src   = testGraph.emplaceBlock<gr::testing::TagSource<float>>({{"n_samples_max", kSamples}, {"mark_tag", false}, //
              {kSignalName, "TestName"}, {kSignalUnit, "TestUnit"}, {kSignalQuantity, "TestQuantity"}, {kSignalMin, -42.f}, {kSignalMax, 42.f}});
        src._tags   = srcTags;
        auto& delay = testGraph.emplaceBlock<testing::Sleep<float>>({{"delay_ms", kProcessingDelayMs}});
        auto& sink  = testGraph.emplaceBlock<DataSink<float>>({{"name", "test_sink"}, {kSignalName, "TestName"}});

        expect(testGraph.connect<"out", "in">(src, delay).has_value());
        expect(testGraph.connect<"out", "in">(delay, sink).has_value());

        auto invalidTypePoller = globalDataSinkRegistry().getPoller<double>(DataSinkQuery::sinkName("test_sink"));
        expect(eq(invalidTypePoller, nullptr));

        auto runner1 = std::async(std::launch::async, [overflowPolicy] {
            std::shared_ptr<Poller<float>> poller;
            expect(spinUntil(4s, [&poller, overflowPolicy] {
                poller = globalDataSinkRegistry().getPoller<float>(DataSinkQuery::sinkName("test_sink"), {.overflowPolicy = overflowPolicy});
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

        auto runner2 = std::async(std::launch::async, [overflowPolicy] {
            std::shared_ptr<Poller<float>> poller;
            expect(spinUntil(4s, [&poller, overflowPolicy] {
                poller = globalDataSinkRegistry().getPoller<float>(DataSinkQuery::signalName("TestName"), {.overflowPolicy = overflowPolicy});
                return poller != nullptr;
            })) << boost::ut::fatal;
            std::vector<float>                  received;
            std::vector<gr::testing::OwningTag> receivedTags;
            bool                                seenFinished = false;
            while (!seenFinished) {
                std::this_thread::sleep_for(50ms); // always wait to force full buffer
                seenFinished = poller->finished;
                while (poller->process([&received, &receivedTags](const auto& data, const auto& tags_) {
                    auto absolute = tags_ | std::views::transform([&received](const auto& t) { return gr::testing::OwningTag{t.index + received.size(), t.map}; });
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

            const auto pollerAfterStop = globalDataSinkRegistry().getPoller<float>(DataSinkQuery::sinkName("test_sink"));
            expect(eq(pollerAfterStop, nullptr));
        }

        const auto pollerAfterDestruction = globalDataSinkRegistry().getPoller<float>(DataSinkQuery::sinkName("test_sink"));
        expect(!pollerAfterDestruction);

        std::vector<float> expected(kSamples);
        std::iota(expected.begin(), expected.end(), 0.0);

        const auto& [pollerDataOnly, received1]               = runner1.get();
        const auto& [pollerWithTags, received2, receivedTags] = runner2.get();

        std::vector<gr::testing::OwningTag> srcAndMetaTags;
        srcAndMetaTags.emplace_back(0UZ, property_map{{kSampleRate, 1.f}, {kSignalMax, 42.f}, {kSignalMin, -42.f}, //
                                             {kSignalName, "TestName"}, {kSignalQuantity, "TestQuantity"}, {kSignalUnit, "TestUnit"}});
        srcAndMetaTags.emplace_back(0UZ, property_map{{kSignalMax, 42.f}, {kSignalMin, -42.f}, {kSignalName, "TestName"}, {kSignalUnit, "TestUnit"}, //
                                             {kSignalQuantity, "TestQuantity"}});
        srcAndMetaTags.insert(srcAndMetaTags.end(), srcTags.begin(), srcTags.end());

        if (isBlocking) {
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
            expect(eq(receivedTags.size() + pollerWithTags->droppedTagCount.load(), srcAndMetaTags.size()));
            expect(gt(pollerWithTags->droppedTagCount.load(), 0UZ));
        }
    } | std::vector<bool>{true, false};

    "trigger mode - polling/callback overlapping/non-overlapping"_test = [] {
        using namespace gr::tag;
        const std::uint32_t nSamples    = 150000;
        const std::size_t   preSamples  = 5;
        const std::size_t   postSamples = 7;

        const std::vector<std::size_t>      triggerIndices = {1001, 1001, 1002, 1003, 1003, 1005, 1007, 10000, 10000, 20000};
        std::vector<gr::testing::OwningTag> srcTags;
        std::uint64_t                       timeCounter = 0;
        for (std::size_t i : triggerIndices) {
            // tags should not be identical, duplicates are ignored by Block::inputTags()
            srcTags.push_back(gr::testing::OwningTag{i, {{kTriggerName, "TRIGGER"}, {kTriggerTime, timeCounter++}}});
        }
        srcTags.push_back(gr::testing::OwningTag{21000, {{kTriggerName, "NO_TRIGGER1"}, {kTriggerTime, timeCounter + 1}}});
        srcTags.push_back(gr::testing::OwningTag{21000, {{kTriggerName, "NO_TRIGGER2"}, {kTriggerTime, timeCounter + 2}}});
        srcTags.push_back(gr::testing::OwningTag{22000, {{kTriggerName, "NO_TRIGGER3"}, {kTriggerTime, timeCounter + 3}}});

        gr::Graph testGraph;
        auto&     src = testGraph.emplaceBlock<gr::testing::TagSource<float>>({{"n_samples_max", nSamples}, {"mark_tag", false}, {"verbose_console", true}, //
                {kSignalName, "TestName"}, {kSignalUnit, "TestUnit"}, {kSignalQuantity, "TestQuantity"}, {kSignalMin, -2.f}, {kSignalMax, 2.f}});

        src._tags   = srcTags;
        auto& delay = testGraph.emplaceBlock<testing::Sleep<float>>({{"delay_ms", kProcessingDelayMs}});
        auto& sink  = testGraph.emplaceBlock<DataSink<float>>({{"name", "test_sink"}, {"signal_name", "TestName"}});

        expect(testGraph.connect<"out", "in">(src, delay).has_value());
        expect(testGraph.connect<"out", "in">(delay, sink).has_value());

        std::mutex         mutexCallback;
        std::vector<float> receivedDataCallback;
        std::size_t        nReceivedTagsCallback = 0;
        auto               callback              = [&receivedDataCallback, &nReceivedTagsCallback, &mutexCallback, timeCounter](auto&& dataset) {
            std::lock_guard lg{mutexCallback};
            checkDataSet(dataset, receivedDataCallback, nReceivedTagsCallback, {.nSignalValues = preSamples + postSamples, .timingEventIndex = preSamples, .triggerTimeMax = timeCounter});
        };
        auto callbackThread = std::thread([&] { //
            gr::thread_pool::thread::setThreadName("qa_DS:registerThread");
            expect(spinUntil(4s, [&] { return globalDataSinkRegistry().registerCallback<float>(DataSinkQuery::sinkName("test_sink"), CaptureMode::Trigger, isTrigger, {.preSamples = preSamples, .postSamples = postSamples}, callback); })) << boost::ut::fatal;
        });

        auto polling = std::async(std::launch::async, [timeCounter, preSamples, postSamples] {
            std::shared_ptr<Poller<DataSet<float>>> poller;
            expect(spinUntil(4s, [&] {
                poller = globalDataSinkRegistry().getPoller<float>(DataSinkQuery::sinkName("test_sink"), CaptureMode::Trigger, isTrigger, {.preSamples = preSamples, .postSamples = postSamples});
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
        expect(eq(poller->droppedSampleCount.load(), 0u));

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
        constexpr gr::Size_t kSamples         = 200000;
        constexpr float      kInputSampleRate = 10000.f;
        constexpr auto       kDelay           = std::chrono::milliseconds{500}; // sample rate 10000 -> 5000 samples
        const double         seconds          = std::chrono::duration<double>(kDelay).count();
        const std::size_t    nSamplesDelay    = static_cast<std::size_t>(seconds * static_cast<double>(kInputSampleRate));

        const std::vector<std::size_t>      triggerIndices = {1001, 1001, 1002, 1003, 1003, 1005, 1007, 10000, 10000, 20000};
        std::vector<gr::testing::OwningTag> srcTags;
        std::uint64_t                       timeCounter = 0;
        for (std::size_t i : triggerIndices) {
            srcTags.push_back(gr::testing::OwningTag{i, {{kTriggerName, "TRIGGER"}, {kTriggerTime, timeCounter++}}});
        }
        srcTags.push_back(gr::testing::OwningTag{21000, {{kTriggerName, "NO_TRIGGER1"}, {kTriggerTime, timeCounter + 1}}});
        srcTags.push_back(gr::testing::OwningTag{21000, {{kTriggerName, "NO_TRIGGER2"}, {kTriggerTime, timeCounter + 2}}});
        srcTags.push_back(gr::testing::OwningTag{22000, {{kTriggerName, "NO_TRIGGER3"}, {kTriggerTime, timeCounter + 3}}});

        gr::Graph testGraph;
        auto&     src = testGraph.emplaceBlock<gr::testing::TagSource<float>>({{"n_samples_max", kSamples}, {"mark_tag", false},          //
                {kSampleRate, kInputSampleRate}, {kSignalName, "TestName"}, {kSignalUnit, "TestUnit"}, {kSignalQuantity, "TestQuantity"}, //
                {kSignalMin, 0.f}, {kSignalMax, static_cast<float>(kSamples - 1)}});
        src._tags     = srcTags;
        auto& delay   = testGraph.emplaceBlock<testing::Sleep<float>>({{"delay_ms", kProcessingDelayMs}});
        auto& sink    = testGraph.emplaceBlock<DataSink<float>>({{"name", "test_sink"}, {kSignalName, "TestName"}});

        expect(testGraph.connect<"out", "in">(src, delay).has_value());
        expect(testGraph.connect<"out", "in">(delay, sink).has_value());

        DataSetTestParams dataSetTestParams = {.nSignalValues = 1, .rangeMin = 0.f, .rangeMax = static_cast<float>(kSamples - 1.), .timingEventIndex = -static_cast<std::ptrdiff_t>(nSamplesDelay), .triggerTimeMax = timeCounter};

        std::vector<float> receivedDataCallback;
        std::size_t        nReceivedTagsCallback = 0;
        auto               callback              = [&receivedDataCallback, &nReceivedTagsCallback, &dataSetTestParams](const auto& dataset) { //
            checkDataSet(dataset, receivedDataCallback, nReceivedTagsCallback, dataSetTestParams);
        };

        auto callbackThread = std::thread([&] {
            gr::thread_pool::thread::setThreadName("qa_DS:registerThread");
            expect(spinUntil(4s, [&] { return globalDataSinkRegistry().registerCallback<float>(DataSinkQuery::sinkName("test_sink"), CaptureMode::Snapshot, isTrigger, {.delay = kDelay}, callback); })) << boost::ut::fatal;
        });

        auto polling = std::async(std::launch::async, [kDelay, &dataSetTestParams] {
            std::shared_ptr<Poller<DataSet<float>>> poller;
            expect(spinUntil(4s, [&] {
                poller = globalDataSinkRegistry().getPoller<float>(DataSinkQuery::sinkName("test_sink"), CaptureMode::Snapshot, isTrigger, {.delay = kDelay});
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
        expect(eq(poller->droppedSampleCount.load(), 0u));

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
        auto& delay          = testGraph.emplaceBlock<testing::Sleep<int32_t>>({{"delay_ms", 2500u}});
        auto& sink           = testGraph.emplaceBlock<DataSink<int32_t>>({{"name", "test_sink"}, {"signal_name", "test signal"}});

        expect(testGraph.connect<"out", "in">(src, delay).has_value());
        expect(testGraph.connect<"out", "in">(delay, sink).has_value());

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
                        registered[i] = globalDataSinkRegistry().registerCallback<int32_t>(DataSinkQuery::sinkName("test_sink"), CaptureMode::Multiplexed, Matcher(matchers[i]), {.maximumWindowSize = 100000}, std::move(callback));
                    }
                }
                return std::ranges::all_of(registered, [](bool b) { return b; });
            })) << boost::ut::fatal;
        });

        for (std::size_t i = 0; i < matchers.size(); ++i) {
            auto f = std::async(std::launch::async, [i, &matchers]() {
                std::shared_ptr<Poller<DataSet<int32_t>>> poller;
                expect(spinUntil(4s, [&] {
                    poller = globalDataSinkRegistry().getPoller<int32_t>(DataSinkQuery::sinkName("test_sink"), CaptureMode::Multiplexed, Matcher(matchers[i]), {.maximumWindowSize = 100000});
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
                            expect(eq(dataset.signalName(0UZ), "test signal"sv));
                            expect(eq(dataset.signalUnit(0UZ), "a.u."sv));
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

    "continuous mode - more tags than the tag buffer holds"_test = [] {
        constexpr std::size_t kTags       = 3000UZ;
        constexpr std::size_t kTagSpacing = 2UZ;
        constexpr gr::Size_t  kSamples    = static_cast<gr::Size_t>(kTagSpacing * (kTags + 1UZ));
        gr::Graph             testGraph;
        auto&                 source = testGraph.emplaceBlock<testing::TagSource<float>>({{"n_samples_max", kSamples}, {"mark_tag", false}});
        auto&                 sink   = testGraph.emplaceBlock<DataSink<float>>({{"name", "test_sink"}, {"signal_name", "test signal"}});
        expect(testGraph.connect<"out", "in">(source, sink).has_value()) << fatal;
        for (std::size_t i = 1UZ; i <= kTags; ++i) {
            source._tags.push_back(gr::testing::OwningTag{kTagSpacing * i, {{"index", static_cast<std::uint64_t>(i)}}});
        }

        const auto  poller    = sink.getPoller();
        std::size_t nReceived = 0UZ;
        std::size_t nTags     = 0UZ;
        const auto  receive   = [&nReceived, &nTags](std::span<const float> data, std::span<const Tag> tags) {
            expect(eq_collections(data, getIota(data.size(), static_cast<float>(nReceived))));
            for (const Tag& tag : tags) {
                if (const auto index = tag.map.find_value("index")) {
                    ++nTags;
                    expect(eq(nReceived + tag.index, kTagSpacing * nTags));
                    expect(eq(gr::test::get_value_or_fail<std::uint64_t>(*index), static_cast<std::uint64_t>(nTags)));
                }
            }
            nReceived += data.size();
        };

        Scheduler sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        auto reader = std::async(std::launch::async, [&poller, &receive] {
            expect(spinUntil(30s, [&poller] { return poller->tagReader.available() == poller->tagBuffer.size(); })) << "the tag buffer fills";
            expect(spinUntil(30s, [&poller, &receive] {
                const bool seenFinished = poller->finished.load();
                while (poller->process(receive)) {
                }
                return seenFinished;
            }));
        });
        expect(sched.runAndWait().has_value()) << fatal;
        reader.get();

        expect(eq(nReceived, static_cast<std::size_t>(kSamples)));
        expect(eq(nTags, kTags));
        expect(eq(poller->droppedSampleCount.load(), 0UZ));
        expect(eq(poller->droppedTagCount.load(), 0UZ));
    };

    "snapshot mode - more pending snapshots than the poller holds"_test = [] {
        using namespace gr::tag;
        constexpr std::size_t kSnapshots = 3000UZ; // below the 4096 entries of the port's tag ring, which drops when full
        constexpr std::size_t kDelay     = 1500UZ; // a 1500 s delay at the sink's default sample_rate of 1 Hz
        constexpr gr::Size_t  kSamples   = static_cast<gr::Size_t>(kSnapshots + kDelay + 1UZ);
        gr::Graph             testGraph;
        auto&                 source = testGraph.emplaceBlock<testing::TagSource<float>>({{"n_samples_max", kSamples}, {"mark_tag", false}});
        auto&                 sink   = testGraph.emplaceBlock<DataSink<float>>({{"name", "test_sink"}, {"signal_name", "test signal"}});
        expect(testGraph.connect<"out", "in">(source, sink).has_value()) << fatal;
        for (std::size_t i = 1UZ; i <= kSnapshots; ++i) {
            source._tags.push_back(gr::testing::OwningTag{i, {{kTriggerName, "TRIGGER"}, {kTriggerTime, static_cast<std::uint64_t>(i)}}});
        }

        const auto  poller    = sink.getPoller(CaptureMode::Snapshot, isTrigger, {.delay = std::chrono::seconds(kDelay)});
        std::size_t nReceived = 0UZ;
        const auto  receive   = [&nReceived](std::span<const DataSet<float>> dataSets) {
            for (const auto& dataSet : dataSets) {
                ++nReceived;
                expect(eq_collections(dataSet.signalValues(0UZ), getIota(1UZ, static_cast<float>(nReceived + kDelay))));
                expect(eq(dataSet.timingEvents(0UZ)[0UZ].first, -static_cast<std::ptrdiff_t>(kDelay)));
            }
        };

        Scheduler sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        auto reader = std::async(std::launch::async, [&poller, &receive] {
            expect(spinUntil(30s, [&poller, &receive] {
                const bool seenFinished = poller->finished.load();
                while (poller->process(receive)) {
                }
                return seenFinished;
            }));
        });
        expect(sched.runAndWait().has_value()) << fatal;
        reader.get();

        expect(eq(nReceived, kSnapshots));
        expect(eq(poller->droppedSampleCount.load(), 0UZ));
    };

    "pollers - overflow"_test = [](OverflowPolicy overflowPolicy) {
        using namespace gr::tag;
        constexpr std::size_t kOverflowDataSets = 3UZ;
        constexpr std::size_t kPostSamples      = 2UZ;
        constexpr std::size_t kTriggerSpacing   = 4UZ;
        const bool            isBackpressure    = overflowPolicy == OverflowPolicy::Backpressure;

        constexpr std::array<std::optional<CaptureMode>, 4UZ> kCaptureModes{CaptureMode::Trigger, CaptureMode::Multiplexed, CaptureMode::Snapshot, std::nullopt}; // nullopt: a DataSet input stream
        for (const std::optional<CaptureMode> mode : kCaptureModes) {
            gr::Graph                               testGraph;
            auto&                                   source = testGraph.emplaceBlock<testing::TagSource<float>>({{"signal_name", "test signal"}, {"signal_unit", "test unit"}, {"mark_tag", false}});
            std::shared_ptr<Poller<DataSet<float>>> poller;
            if (!mode) {
                auto& streamToDataSet = testGraph.emplaceBlock<StreamToDataSet<float>>({{"filter", "TRIGGER"}, {"n_pre", gr::Size_t{0}}, {"n_post", static_cast<gr::Size_t>(kPostSamples)}});
                auto& sink            = testGraph.emplaceBlock<DataSink<DataSet<float>>>({{"name", "test_sink"}, {"signal_name", "test signal"}});
                expect(testGraph.connect<"out", "in">(source, streamToDataSet).has_value()) << fatal;
                expect(testGraph.connect<"out", "in">(streamToDataSet, sink).has_value()) << fatal;
                poller = sink.getPoller({.overflowPolicy = overflowPolicy});
            } else {
                auto& sink = testGraph.emplaceBlock<DataSink<float>>({{"name", "test_sink"}, {"signal_name", "test signal"}});
                expect(testGraph.connect<"out", "in">(source, sink).has_value()) << fatal;
                poller = sink.getPoller(*mode, isTrigger, {.overflowPolicy = overflowPolicy, .preSamples = 0UZ, .postSamples = kPostSamples, .maximumWindowSize = kPostSamples, .delay = 1s});
            }

            const std::size_t capacity  = poller->buffer.size();
            const std::size_t nDataSets = capacity + kOverflowDataSets;
            const auto        nSamples  = static_cast<gr::Size_t>(kTriggerSpacing * nDataSets + kPostSamples);
            expect(source.settings().setStaged({{"n_samples_max", nSamples}}).empty()) << fatal;
            for (std::size_t i = 0UZ; i < nDataSets; ++i) {
                source._tags.push_back(gr::testing::OwningTag{kTriggerSpacing * (i + 1UZ), {{kTriggerName, "TRIGGER"}, {kTriggerTime, static_cast<std::uint64_t>(i)}}});
            }

            std::vector<DataSet<float>> received;
            auto                        drain = [&poller, &received] {
                while (poller->process([&received](const auto& dataSets) { received.insert(received.end(), dataSets.begin(), dataSets.end()); })) {
                }
            };

            Scheduler sched;
            if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
                throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
            }
            if (isBackpressure) {
                auto reader = std::async(std::launch::async, [&poller, &drain, capacity] {
                    expect(spinUntil(30s, [&poller, capacity] { return poller->reader.available() == capacity; }));
                    expect(spinUntil(30s, [&poller, &drain] {
                        const bool seenFinished = poller->finished.load();
                        drain();
                        return seenFinished;
                    }));
                });
                expect(sched.runAndWait().has_value()) << fatal;
                reader.get();
            } else {
                expect(sched.runAndWait().has_value()) << fatal;
                drain();
            }
            expect(eq(source._nSamplesProduced, nSamples));
            expect(poller->finished.load());
            expect(eq(poller->droppedSampleCount.load(), isBackpressure ? 0UZ : kOverflowDataSets));

            expect(eq(received.size(), isBackpressure ? nDataSets : capacity)) << (mode ? gr::meta::enumName(*mode).value_or("") : "DataSet input") << fatal;
            const bool        isSnapshot = mode == CaptureMode::Snapshot;
            const std::size_t nValues    = isSnapshot ? 1UZ : kPostSamples;
            const std::size_t nDelay     = isSnapshot ? 1UZ : 0UZ; // a 1 s delay at the sink's default sample_rate of 1 Hz
            for (std::size_t i = 0UZ; i < received.size(); ++i) {
                const auto& dataSet = received[i];
                const auto  dsCheck = dataset::checkConsistency(dataSet, "retained dataset");
                expect(dsCheck.has_value()) << [&] { return std::format("unexpected: {}", dsCheck.error()); } << fatal;
                expect(eq(dataSet.size(), 1UZ)) << fatal;
                expect(eq_collections(dataSet.signalValues(0UZ), getIota(nValues, static_cast<float>(kTriggerSpacing * (i + 1UZ) + nDelay))));
                expect(eq(dataSet.signalName(0UZ), "test signal"sv));
                expect(eq(dataSet.signalUnit(0UZ), "test unit"sv));
                expect(eq(dataSet.timingEvents(0UZ).size(), 1UZ)) << fatal;
                expect(eq(dataSet.timingEvents(0UZ)[0UZ].first, -static_cast<std::ptrdiff_t>(nDelay)));
                expect(dataSet.timingEvents(0UZ)[0UZ].second == source._tags[i].map);
            }
        }
    } | std::vector{OverflowPolicy::Drop, OverflowPolicy::Backpressure};

    "DataSet - polling and callback"_test = [] {
        using namespace gr::tag;
        gr::Graph testGraph;
        auto&     source          = testGraph.emplaceBlock<testing::TagSource<float, testing::ProcessFunction::USE_PROCESS_BULK>>({{"n_samples_max", static_cast<gr::Size_t>(1024)}, {"signal_name", "test signal"}, {"signal_unit", "test unit"}, {"mark_tag", false}});
        auto&     delay           = testGraph.emplaceBlock<testing::Sleep<float>>({{"delay_ms", kProcessingDelayMs}});
        auto&     streamToDataSet = testGraph.emplaceBlock<StreamToDataSet<float>>({{"filter", "CMD_DIAG_TRIGGER1"}, {"n_pre", static_cast<gr::Size_t>(100)}, {"n_post", static_cast<gr::Size_t>(200)}});
        auto&     sink            = testGraph.emplaceBlock<DataSink<DataSet<float>>>({{"name", "test_sink"}, {"signal_name", "test signal"}});
        expect(testGraph.connect<"out", "in">(source, delay).has_value());
        expect(testGraph.connect<"out", "in">(delay, streamToDataSet).has_value());
        expect(testGraph.connect<"out", "in">(streamToDataSet, sink).has_value());

        for (const auto index : {400UZ, 800UZ}) {
            source._tags.push_back(gr::testing::OwningTag{index, {kTriggerName("CMD_DIAG_TRIGGER1"s), kTriggerTime(std::uint64_t(0)), kTriggerOffset(0.f), kContext(""s), kTriggerMetaInfo(gr::property_map{})}});
        }

        auto&                       registry = globalDataSinkRegistry();
        const auto                  query    = DataSinkQuery::sinkName("test_sink");
        std::vector<DataSet<float>> receivedCallback;

        Scheduler sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }

        auto polling = std::async(std::launch::async, [&registry, query, &receivedCallback] {
            auto callback = [&receivedCallback](std::span<const DataSet<float>> dataSets) { receivedCallback.insert(receivedCallback.end(), dataSets.begin(), dataSets.end()); };
            expect(spinUntil(4s, [&] { return registry.registerCallback<DataSet<float>>(query, std::move(callback)); })) << fatal;

            std::shared_ptr<Poller<DataSet<float>>> poller;
            expect(spinUntil(4s, [&] {
                poller = registry.getPoller<DataSet<float>>(query);
                return poller != nullptr;
            })) << fatal;

            std::vector<DataSet<float>>         received;
            std::vector<gr::testing::OwningTag> receivedTags;
            bool                                seenFinished = false;
            while (!seenFinished) {
                seenFinished = poller->finished.load();
                while (poller->process([&received, &receivedTags](const auto& dataSets, const auto& tags) {
                    auto absolute = tags | std::views::transform([&received](const auto& t) { return gr::testing::OwningTag{t.index + received.size(), t.map}; });
                    receivedTags.insert(receivedTags.end(), absolute.begin(), absolute.end());
                    received.insert(received.end(), dataSets.begin(), dataSets.end());
                })) {
                }
            }
            return std::make_tuple(std::move(poller), std::move(received), std::move(receivedTags));
        });

        expect(sched.runAndWait().has_value());
        const auto [poller, receivedPolling, receivedTags] = polling.get();

        expect(poller->finished.load());
        expect(eq(poller->droppedSampleCount.load(), 0UZ));
        expect(!registry.getPoller<DataSet<float>>(query));
        expect(receivedTags.empty());
        expect(eq(receivedPolling.size(), 2UZ)) << fatal;
        expect(eq(receivedCallback.size(), 2UZ)) << fatal;

        for (std::size_t i = 0; i < receivedPolling.size(); ++i) {
            const auto& polled   = receivedPolling[i];
            const auto& callback = receivedCallback[i];
            const auto  dsCheck  = dataset::checkConsistency(polled, "polled dataset");
            expect(dsCheck.has_value()) << [&] { return std::format("unexpected: {}", dsCheck.error()); } << fatal;
            expect(eq(polled.size(), 1UZ)) << fatal;
            expect(eq_collections(polled.signalValues(0UZ), getIota(300, 300.f + 400.f * static_cast<float>(i))));
            expect(eq(polled.signalName(0UZ), "test signal"sv));
            expect(eq(polled.signalUnit(0UZ), "test unit"sv));
            expect(eq(polled.timingEvents(0UZ).size(), 1UZ)) << fatal;
            expect(eq(polled.timingEvents(0UZ)[0UZ].first, 100));

            expect(eq(polled.default_value, callback.default_value));
            expect(eq(polled.timestamp, callback.timestamp));
            expect(eq(polled.axis_names, callback.axis_names));
            expect(eq(polled.axis_units, callback.axis_units));
            expect(polled.axis_values == callback.axis_values);
            expect(eq(polled.extents, callback.extents));
            expect(std::holds_alternative<LayoutRight>(polled.layout));
            expect(std::holds_alternative<LayoutRight>(callback.layout));
            expect(eq(polled.signal_names, callback.signal_names));
            expect(eq(polled.signal_quantities, callback.signal_quantities));
            expect(eq(polled.signal_units, callback.signal_units));
            expect(eq(polled.signal_values, callback.signal_values));
            expect(eq(polled.signal_ranges, callback.signal_ranges));
            expect(polled.meta_information == callback.meta_information);
            expect(polled.timing_events == callback.timing_events);
        }
    };

    "registry - mixed sink types"_test = [] {
        gr::Graph testGraph;
        auto&     source          = testGraph.emplaceBlock<testing::TagSource<float>>({{"n_samples_max", gr::Size_t{10}}, {"mark_tag", false}});
        auto&     streamSink      = testGraph.emplaceBlock<DataSink<float>>({{"name", "stream_sink"}, {"signal_name", "stream signal"}});
        auto&     streamToDataSet = testGraph.emplaceBlock<StreamToDataSet<float>>({{"filter", "TRIGGER"}});
        auto&     dataSetSink     = testGraph.emplaceBlock<DataSink<DataSet<float>>>({{"name", "dataset_sink"}, {"signal_name", "dataset signal"}});
        expect(testGraph.connect<"out", "in">(source, streamSink).has_value()) << fatal;
        expect(testGraph.connect<"out", "in">(source, streamToDataSet).has_value()) << fatal;
        expect(testGraph.connect<"out", "in">(streamToDataSet, dataSetSink).has_value()) << fatal;
        expect(streamSink._registered) << fatal;
        expect(dataSetSink._registered) << fatal;

        Scheduler sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value()) << fatal;

        expect(!streamSink._registered);
        expect(!dataSetSink._registered);
        expect(eq(globalDataSinkRegistry().getPoller<float>(DataSinkQuery::signalName("stream signal")), nullptr));
        expect(eq(globalDataSinkRegistry().getPoller<DataSet<float>>(DataSinkQuery::signalName("dataset signal")), nullptr));
    };

    "registry - duplicate signal_name"_test = [] {
        gr::Graph liveGraph;
        auto&     registeredSink = liveGraph.emplaceBlock<DataSink<float>>({{"name", "registered_sink"}, {"signal_name", "duplicate signal"}});
        expect(registeredSink._registered) << fatal;

        gr::Graph testGraph;
        auto&     source        = testGraph.emplaceBlock<testing::TagSource<float>>({{"n_samples_max", gr::Size_t{10}}, {"mark_tag", false}});
        auto&     duplicateSink = testGraph.emplaceBlock<DataSink<float>>({{"name", "duplicate_sink"}, {"signal_name", "duplicate signal"}});
        expect(!duplicateSink._registered) << fatal;
        expect(duplicateSink.settings().setStaged({{"signal_name", "unique signal"}}).empty()) << fatal;
        std::ignore = duplicateSink.settings().applyStagedParameters();
        expect(duplicateSink._registered) << "a rejected sink registers once renamed";
        expect(globalDataSinkRegistry().find<DataSink<float>>(DataSinkQuery::signalName("unique signal")) == &duplicateSink);
        expect(testGraph.connect<"out", "in">(source, duplicateSink).has_value()) << fatal;

        Scheduler sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value()) << fatal;

        expect(globalDataSinkRegistry().getPoller<float>(DataSinkQuery::signalName("duplicate signal")) != nullptr);
    };

    "registry - sink destroyed without stop"_test = [] {
        auto emplaceSinks = [](gr::Graph& graph) {
            auto& sink        = graph.emplaceBlock<DataSink<float>>({{"name", "unstarted_sink"}, {"signal_name", "unstarted signal"}});
            auto& dataSetSink = graph.emplaceBlock<DataSink<DataSet<float>>>({{"name", "unstarted_dataset_sink"}, {"signal_name", "unstarted dataset signal"}});
            return sink._registered && dataSetSink._registered;
        };
        {
            gr::Graph discardedGraph;
            expect(emplaceSinks(discardedGraph)) << fatal;
        }
        gr::Graph graph;
        expect(emplaceSinks(graph));
    };

    "registry - restarted graph"_test = [] {
        gr::Graph testGraph;
        auto&     source = testGraph.emplaceBlock<testing::TagSource<float>>({{"n_samples_max", gr::Size_t{10}}, {"mark_tag", false}});
        auto&     sink   = testGraph.emplaceBlock<DataSink<float>>({{"name", "restarted_sink"}, {"signal_name", "restarted signal"}});
        expect(testGraph.connect<"out", "in">(source, sink).has_value()) << fatal;

        Scheduler sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value()) << fatal;
        expect(!sink._registered) << fatal;

        expect(sched.changeStateTo(lifecycle::State::INITIALISED).has_value()) << fatal;
        expect(sink._registered);
        expect(globalDataSinkRegistry().find<DataSink<float>>(DataSinkQuery::signalName("restarted signal")) == &sink);
        expect(!sink.getPoller({})->finished.load()) << "a poller of the new run is not finished";
    };

    "pollers - created after stop"_test = [] {
        gr::Graph testGraph;
        auto&     source = testGraph.emplaceBlock<testing::TagSource<float>>({{"n_samples_max", gr::Size_t{10}}, {"mark_tag", false}});
        auto&     sink   = testGraph.emplaceBlock<DataSink<float>>({{"name", "test_sink"}, {"signal_name", "test signal"}});
        expect(testGraph.connect<"out", "in">(source, sink).has_value()) << fatal;

        Scheduler sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value()) << fatal;

        expect(sink.getPoller({})->finished.load());
        expect(sink.getPoller(CaptureMode::Trigger, isTrigger)->finished.load());
        expect(sink.getPoller(CaptureMode::Multiplexed, isTrigger)->finished.load());
        expect(sink.getPoller(CaptureMode::Snapshot, isTrigger)->finished.load());
    };

    "pollers - restart after error"_test = [] {
        DataSink<float> sink(property_map{{"name", "test_sink"}});
        const auto      streamPoller  = sink.getPoller();
        const auto      capturePoller = sink.getPoller(CaptureMode::Trigger, isTrigger);

        expect(sink.changeStateTo(lifecycle::State::ERROR).has_value()) << fatal;
        expect(sink.changeStateTo(lifecycle::State::INITIALISED).has_value()) << fatal;
        expect(streamPoller->finished.load()) << "a restart from ERROR skips stop()";
        expect(capturePoller->finished.load());
    };

    "callbacks - errors"_test = [](lifecycle::State failureState) {
        DataSink<float> sink;
        PortOut<float>  source;
        MsgPortIn       errors;
        expect(source.connect(sink.in).has_value()) << fatal;
        expect(sink.msgOut.connect(errors).has_value()) << fatal;
        sink.init(sink.progress);
        expect(sink.changeStateTo(lifecycle::State::RUNNING).has_value()) << fatal;

        std::size_t nCalls   = 0UZ;
        auto        callback = [&nCalls](auto) {
            ++nCalls;
            throw std::runtime_error("callback failed");
        };
        if (failureState == lifecycle::State::RUNNING) {
            sink.registerCallback(callback);
        } else {
            sink.registerCallback(CaptureMode::Trigger, isTrigger, {.preSamples = 0UZ, .postSamples = 2UZ}, callback);
        }
        const auto poller = sink.getPoller();
        source.publishTag(property_map{{gr::tag::kTriggerName, "TRIGGER"}});
        {
            auto output = source.streamWriter().reserve(1UZ);
            output[0UZ] = 42.f;
            output.publish(1UZ);
        }
        expect(sink.work(1UZ).status == (failureState == lifecycle::State::RUNNING ? work::Status::ERROR : work::Status::OK));
        if (failureState == lifecycle::State::INITIALISED) {
            expect(sink.changeStateTo(lifecycle::State::ERROR).has_value()) << fatal;
            expect(sink.changeStateTo(lifecycle::State::INITIALISED).has_value()) << fatal;
        } else {
            expect(sink.changeStateTo(lifecycle::State::REQUESTED_STOP).has_value()) << fatal;
        }
        expect(poller->finished.load());
        expect(eq(nCalls, 1UZ));
        const auto messages = errors.streamReader().get();
        expect(eq(messages.size(), 1UZ)) << fatal;
        expect(!messages[0UZ].data.has_value());
    } | std::vector{lifecycle::State::RUNNING, lifecycle::State::REQUESTED_STOP, lifecycle::State::INITIALISED};

    "DataSet - released poller"_test = [] {
        using namespace gr::tag;
        gr::Graph testGraph;
        auto&     source          = testGraph.emplaceBlock<testing::TagSource<float>>({{"n_samples_max", gr::Size_t{20}}, {"mark_tag", false}});
        auto&     streamToDataSet = testGraph.emplaceBlock<StreamToDataSet<float>>({{"filter", "TRIGGER"}, {"n_pre", gr::Size_t{0}}, {"n_post", gr::Size_t{2}}});
        auto&     sink            = testGraph.emplaceBlock<DataSink<DataSet<float>>>({{"name", "test_sink"}, {"signal_name", "test signal"}});
        expect(testGraph.connect<"out", "in">(source, streamToDataSet).has_value()) << fatal;
        expect(testGraph.connect<"out", "in">(streamToDataSet, sink).has_value()) << fatal;
        for (const std::size_t index : {4UZ, 8UZ}) {
            source._tags.push_back(gr::testing::OwningTag{index, {{kTriggerName, "TRIGGER"}, {kTriggerTime, static_cast<std::uint64_t>(index)}}});
        }

        std::ignore           = sink.getPoller({.overflowPolicy = OverflowPolicy::Backpressure});
        std::size_t nDataSets = 0UZ;
        sink.registerCallback([&nDataSets](std::span<const DataSet<float>> dataSets) { nDataSets += dataSets.size(); });

        Scheduler sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value()) << fatal;

        expect(eq(nDataSets, 2UZ));
    };

    "capture mode - open capture at stop"_test = [] {
        using namespace gr::tag;
        constexpr std::size_t kPostSamples = 6UZ;
        gr::Graph             testGraph;
        auto&                 source = testGraph.emplaceBlock<testing::TagSource<float>>({{"n_samples_max", gr::Size_t{20}}, {"mark_tag", false}});
        auto&                 sink   = testGraph.emplaceBlock<DataSink<float>>({{"name", "test_sink"}, {"signal_name", "test signal"}});
        expect(testGraph.connect<"out", "in">(source, sink).has_value()) << fatal;
        source._tags = {gr::testing::OwningTag{4UZ, {{kTriggerName, "TRIGGER"}, {kTriggerTime, std::uint64_t{0}}}}, gr::testing::OwningTag{18UZ, {{kTriggerName, "TRIGGER"}, {kTriggerTime, std::uint64_t{1}}}}};

        const auto triggerPoller     = sink.getPoller(CaptureMode::Trigger, isTrigger, {.preSamples = 0UZ, .postSamples = kPostSamples});
        const auto multiplexedPoller = sink.getPoller(CaptureMode::Multiplexed, isTrigger, {.maximumWindowSize = 0UZ});

        Scheduler sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());

        const auto receive = [](const std::shared_ptr<Poller<DataSet<float>>>& poller) {
            std::vector<DataSet<float>> received;
            while (poller->process([&received](const auto& dataSets) { received.insert(received.end(), dataSets.begin(), dataSets.end()); })) {
            }
            return received;
        };
        const std::vector<DataSet<float>> captures = receive(triggerPoller);
        expect(eq(captures.size(), 2UZ)) << fatal;
        expect(eq_collections(captures[0UZ].signalValues(0UZ), getIota(kPostSamples, 4.f)));
        expect(eq_collections(captures[1UZ].signalValues(0UZ), getIota(2UZ, 18.f))) << "the stream ends 2 samples after the trigger";
        const std::vector<DataSet<float>> windows = receive(multiplexedPoller);
        expect(eq(windows.size(), 2UZ)) << fatal;
        expect(eq_collections(windows[0UZ].signalValues(0UZ), getIota(14UZ, 4.f)));
        expect(eq_collections(windows[1UZ].signalValues(0UZ), getIota(2UZ, 18.f))) << "the end of the stream closes the window";
    };

    "trigger mode - stop with full poller"_test = [] {
        using namespace gr::tag;
        constexpr std::size_t kTriggerSpacing = 4UZ;
        constexpr std::size_t kPostSamples    = 6UZ;
        gr::Graph             testGraph;
        auto&                 source = testGraph.emplaceBlock<testing::TagSource<float>>({{"n_samples_max", gr::Size_t{0}}, {"mark_tag", false}});
        auto&                 sink   = testGraph.emplaceBlock<DataSink<float>>({{"name", "test_sink"}, {"signal_name", "test signal"}});
        expect(testGraph.connect<"out", "in">(source, sink).has_value()) << fatal;

        const auto        poller   = sink.getPoller(CaptureMode::Trigger, isTrigger, {.overflowPolicy = OverflowPolicy::Backpressure, .preSamples = 0UZ, .postSamples = kPostSamples});
        const std::size_t capacity = poller->buffer.size();
        for (std::size_t i = 0UZ; i < 2UZ * capacity; ++i) {
            source._tags.push_back(gr::testing::OwningTag{kTriggerSpacing * (i + 1UZ), {{kTriggerName, "TRIGGER"}, {kTriggerTime, static_cast<std::uint64_t>(i)}}});
        }

        Scheduler sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        auto running = std::async(std::launch::async, [&sched] { return sched.runAndWait(); });

        expect(spinUntil(30s, [&poller, capacity] { return poller->reader.available() == capacity - 1UZ; })) << "the open capture keeps one slot";
        expect(globalDataSinkRegistry().getPoller<float>(DataSinkQuery::signalName("test signal")) != nullptr);
        sched.requestStop();
        expect(running.get().has_value());

        expect(poller->finished.load());
        expect(eq(poller->droppedSampleCount.load(), 0UZ));

        std::vector<DataSet<float>> received;
        while (poller->process([&received](const auto& dataSets) { received.insert(received.end(), dataSets.begin(), dataSets.end()); })) {
        }
        expect(eq(received.size(), capacity)) << fatal;
        for (std::size_t i = 0UZ; i + 1UZ < received.size(); ++i) {
            expect(eq_collections(received[i].signalValues(0UZ), getIota(kPostSamples, static_cast<float>(kTriggerSpacing * (i + 1UZ)))));
        }
        expect(eq_collections(received.back().signalValues(0UZ), getIota(kTriggerSpacing, static_cast<float>(kTriggerSpacing * capacity)))) << "the capture open at the stop";
    };

    "trigger mode - released poller"_test = [] {
        using namespace gr::tag;
        gr::Graph testGraph;
        auto&     source = testGraph.emplaceBlock<testing::TagSource<float>>({{"n_samples_max", gr::Size_t{40}}, {"mark_tag", false}});
        auto&     sink   = testGraph.emplaceBlock<DataSink<float>>({{"name", "test_sink"}, {"signal_name", "test signal"}});
        expect(testGraph.connect<"out", "in">(source, sink).has_value()) << fatal;
        source._tags = {gr::testing::OwningTag{20UZ, {{kTriggerName, "TRIGGER"}, {kTriggerTime, std::uint64_t{0}}}}};

        std::ignore       = sink.getPoller(CaptureMode::Trigger, isTrigger, {.preSamples = 8UZ, .postSamples = 1UZ});
        const auto poller = sink.getPoller(CaptureMode::Trigger, isTrigger, {.preSamples = 3UZ, .postSamples = 2UZ});

        Scheduler sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());

        std::vector<DataSet<float>> received;
        while (poller->process([&received](const auto& dataSets) { received.insert(received.end(), dataSets.begin(), dataSets.end()); })) {
        }
        expect(eq(received.size(), 1UZ)) << fatal;
        expect(eq_collections(received[0UZ].signalValues(0UZ), getIota(5UZ, 17.f)));
    };

    "snapshot mode - sample rate change"_test = [] {
        using namespace gr::tag;
        constexpr float kInputSampleRate = 10.f; // a delay of 1 s is 10 samples, and 5 samples after the rate is halved
        gr::Graph       testGraph;
        auto&           source = testGraph.emplaceBlock<testing::TagSource<float>>({{"n_samples_max", gr::Size_t{40}}, {"mark_tag", false}, {kSampleRate, kInputSampleRate}});
        auto&           sink   = testGraph.emplaceBlock<DataSink<float>>({{"name", "test_sink"}, {"signal_name", "test signal"}}); // an explicitly set sample_rate would no longer follow tags
        expect(testGraph.connect<"out", "in">(source, sink).has_value()) << fatal;
        source._tags = {gr::testing::OwningTag{4UZ, {{kTriggerName, "TRIGGER"}, {kTriggerTime, std::uint64_t{0}}}}, gr::testing::OwningTag{8UZ, {{kSampleRate, kInputSampleRate / 2.f}, {kTriggerName, "TRIGGER"}, {kTriggerTime, std::uint64_t{1}}}}};

        const auto poller = sink.getPoller(CaptureMode::Snapshot, isTrigger, {.delay = 1s});

        Scheduler sched;
        if (auto ret = sched.exchange(std::move(testGraph)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(sched.runAndWait().has_value());
        expect(eq(poller->droppedSampleCount.load(), 0UZ));

        std::vector<DataSet<float>> received;
        while (poller->process([&received](const auto& dataSets) { received.insert(received.end(), dataSets.begin(), dataSets.end()); })) {
        }
        constexpr std::array                      kSnapshots{13.f, 14.f}; // the pending trigger at 4 keeps its 10 samples, the one at 8 is due first
        constexpr std::array<std::ptrdiff_t, 2UZ> kTriggerOffsets{-5, -10};
        expect(eq(received.size(), kSnapshots.size())) << fatal;
        for (std::size_t i = 0UZ; i < received.size(); ++i) {
            expect(eq_collections(received[i].signalValues(0UZ), std::vector{kSnapshots[i]}));
            expect(eq(received[i].timingEvents(0UZ).size(), 1UZ)) << fatal;
            expect(eq(received[i].timingEvents(0UZ)[0UZ].first, kTriggerOffsets[i]));
        }
    };
};

const boost::ut::suite DataSinkRegistryTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::basic;

    // a local registry keeps these tests independent of the process-wide one
    "a second sink with an already registered signal name is rejected"_test = [] {
        DataSinkRegistry registry;
        DataSink<float>  first(property_map{{"name", "first"}});
        DataSink<float>  second(property_map{{"name", "second"}});
        DataSink<float>  unnamed(property_map{{"name", "unnamed"}});
        first.signal_name  = "qa_registry_signal";
        second.signal_name = "qa_registry_signal";

        expect(registry.registerSink(&first).has_value());
        expect(first._registered);
        expect(!registry.registerSink(&second).has_value()) << "duplicate signal name";
        expect(not second._registered);

        expect(registry.registerSink(&unnamed).has_value());
        expect(not unnamed._registered) << "a sink without a signal name is never registered";

        expect(registry.getPoller<DataSet<float>>(DataSinkQuery::signalName("qa_registry_signal")) == nullptr) << "a sink is only found as the type it was registered as";

        registry.unregisterSink(&first);
        expect(not first._registered);
        expect(registry.getPoller<float>(DataSinkQuery::signalName("qa_registry_signal")) == nullptr) << "unregistered sinks cannot be found";
    };

    "renaming a registered sink re-keys it in the registry"_test = [] {
        DataSinkRegistry registry;
        DataSink<float>  sink(property_map{{"name", "renamed"}});
        sink.signal_name = "qa_old_name";
        expect(registry.registerSink(&sink).has_value());
        expect(registry.getPoller<float>(DataSinkQuery::signalName("qa_old_name")) != nullptr);

        expect(registry.updateSignalName(&sink, "qa_new_name").has_value());
        expect(registry.getPoller<float>(DataSinkQuery::signalName("qa_old_name")) == nullptr) << "the old name is gone";
        expect(registry.getPoller<float>(DataSinkQuery::signalName("qa_new_name")) != nullptr) << "the new name finds the sink";
    };

    "renaming a sink to an empty or taken name is rejected"_test = [] {
        DataSinkRegistry registry;
        DataSink<float>  sinkA(property_map{{"name", "a"}});
        DataSink<float>  sinkB(property_map{{"name", "b"}});
        sinkA.signal_name = "qa_name_a";
        sinkB.signal_name = "qa_name_b";
        expect(registry.registerSink(&sinkA).has_value());
        expect(registry.registerSink(&sinkB).has_value());

        expect(!registry.updateSignalName(&sinkA, "").has_value()) << "empty new name";
        expect(!registry.updateSignalName(&sinkA, "qa_name_b").has_value()) << "taken new name";
        expect(registry.getPoller<float>(DataSinkQuery::signalName("qa_name_a")) != nullptr) << "a rejected rename leaves the sink where it was";

        expect(registry.updateSignalName(&sinkA, "qa_name_c").has_value());
        expect(registry.find<DataSink<float>>(DataSinkQuery::signalName("qa_name_b")) == &sinkB) << "a later rename does not touch the other sink";
        expect(registry.find<DataSink<float>>(DataSinkQuery::signalName("qa_name_c")) == &sinkA);
        expect(registry.find<DataSink<float>>(DataSinkQuery::signalName("qa_name_a")) == nullptr) << "the previous name is gone";
        expect(registry.updateSignalName(&sinkA, "qa_name_c").has_value()) << "the sink's own name is not taken";
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
