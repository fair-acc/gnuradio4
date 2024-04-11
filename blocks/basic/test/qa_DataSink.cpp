#include <future>
#include <ranges>

#include <boost/ut.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/reflection.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <gnuradio-4.0/basic/DataSink.hpp>
#include <gnuradio-4.0/testing/Delay.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

using namespace std::chrono_literals;

template<>
struct fmt::formatter<gr::Tag> {
    template<typename ParseContext>
    constexpr auto
    parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    constexpr auto
    format(const gr::Tag &tag, FormatContext &ctx) const {
        return fmt::format_to(ctx.out(), "{}", tag.index);
    }
};

namespace gr::basic::data_sink_test {

constexpr auto kProcessingDelayMs = 600u;

/**
 * Example tag matcher (TriggerMatcher implementation) for the multiplexed listener case (interleaved data). As a toy example, we use
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

    static inline bool
    same(int x, std::optional<int> other) {
        return other && x == *other;
    }

    static inline bool
    changed(int x, std::optional<int> other) {
        return !same(x, other);
    }

    TriggerMatchResult
    operator()(const Tag &tag) {
        const auto ty = tag.get("YEAR");
        const auto tm = tag.get("MONTH");
        const auto td = tag.get("DAY");
        if (!ty || !tm || !td) {
            return TriggerMatchResult::Ignore;
        }

        const auto tup        = std::make_tuple(std::get<int>(ty->get()), std::get<int>(tm->get()), std::get<int>(td->get()));
        const auto &[y, m, d] = tup;
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

        auto r = TriggerMatchResult::Ignore;

        if (!matches) {
            r = TriggerMatchResult::NotMatching;
        } else if (!last_matched || restart) {
            r = TriggerMatchResult::Matching;
        }

        last_seen    = tup;
        last_matched = matches;
        return r;
    }
};

static Tag
makeTag(Tag::signed_index_type index, int year, int month, int day) {
    return Tag{ index, { { "YEAR", year }, { "MONTH", month }, { "DAY", day } } };
}

static std::vector<Tag>
makeTestTags(Tag::signed_index_type firstIndex, Tag::signed_index_type interval) {
    std::vector<Tag> tags;
    for (int y = 1; y <= 3; ++y) {
        for (int m = 1; m <= 2; ++m) {
            for (int d = 1; d <= 3; ++d) {
                tags.push_back(makeTag(firstIndex, y, m, d));
                firstIndex += interval;
            }
        }
    }
    return tags;
}

static std::string
toAsciiArt(std::span<TriggerMatchResult> states) {
    bool        started = false;
    std::string r;
    for (auto s : states) {
        switch (s) {
        case TriggerMatchResult::Matching:
            r += started ? "||#" : "|#";
            started = true;
            break;
        case TriggerMatchResult::NotMatching:
            r += started ? "|_" : "_";
            started = false;
            break;
        case TriggerMatchResult::Ignore: r += started ? "#" : "_"; break;
        }
    };
    return r;
}

template<TriggerMatcher M>
std::string
runMatcherTest(std::span<const Tag> tags, M o) {
    std::vector<TriggerMatchResult> r;
    r.reserve(tags.size());
    for (const auto &tag : tags) {
        r.push_back(o(tag));
    }
    return toAsciiArt(r);
}

std::pair<std::vector<Tag>, std::vector<Tag>>
extractMetadataTags(const std::vector<Tag> &tags) {
    constexpr auto   tagsToExtract = std::array{ gr::tag::SAMPLE_RATE.shortKey(), gr::tag::SIGNAL_NAME.shortKey(), gr::tag::SIGNAL_UNIT.shortKey(), gr::tag::SIGNAL_MIN.shortKey(),
                                               gr::tag::SIGNAL_MAX.shortKey() };
    std::vector<Tag> metadataTags;
    std::vector<Tag> nonMetadataTags;
    for (const auto &tag : tags) {
        Tag metadata;
        Tag nonMetadata;
        metadata.index    = tag.index;
        nonMetadata.index = tag.index;
        for (const auto &[key, value] : tag.map) {
            if (std::find(tagsToExtract.begin(), tagsToExtract.end(), key) != tagsToExtract.end()) {
                metadata.map[key] = value;
            } else {
                nonMetadata.map[key] = value;
            }
        }
        if (!metadata.map.empty()) {
            metadataTags.push_back(metadata);
        }
        if (!nonMetadata.map.empty()) {
            nonMetadataTags.push_back(nonMetadata);
        }
    }
    return { metadataTags, nonMetadataTags };
}

struct Metadata {
    std::optional<std::string> signal_name;
    std::optional<std::string> signal_unit;
    std::optional<float>       signal_min;
    std::optional<float>       signal_max;
    std::optional<float>       sample_rate;
};

Metadata
metadataFromTag(const Tag &tag) {
    Metadata m;
    for (const auto &[key, value] : tag.map) {
        if (key == gr::tag::SIGNAL_NAME.shortKey()) {
            m.signal_name = std::get<std::string>(value);
        } else if (key == gr::tag::SIGNAL_UNIT.shortKey()) {
            m.signal_unit = std::get<std::string>(value);
        } else if (key == gr::tag::SIGNAL_MIN.shortKey()) {
            m.signal_min = std::get<float>(value);
        } else if (key == gr::tag::SIGNAL_MAX.shortKey()) {
            m.signal_max = std::get<float>(value);
        } else if (key == gr::tag::SAMPLE_RATE.shortKey()) {
            m.sample_rate = std::get<float>(value);
        }
    }
    return m;
}

Metadata
latestMetadata(const std::vector<Tag> &tags) {
    Metadata metadata;
    for (const auto &tag : tags | std::views::reverse) {
        const auto m = metadataFromTag(tag);
        if (!metadata.signal_name) {
            metadata.signal_name = m.signal_name;
        }
        if (!metadata.signal_unit) {
            metadata.signal_unit = m.signal_unit;
        }
        if (!metadata.signal_min) {
            metadata.signal_min = m.signal_min;
        }
        if (!metadata.signal_max) {
            metadata.signal_max = m.signal_max;
        }
        if (!metadata.sample_rate) {
            metadata.sample_rate = m.sample_rate;
        }
    }
    return metadata;
}

bool
spinUntil(std::chrono::milliseconds timeout, auto fnc) {
    const auto start = std::chrono::steady_clock::now();

    while (std::chrono::steady_clock::now() - start < timeout) {
        if (fnc()) {
            return true;
        }
    }

    return false;
}

} // namespace gr::basic::data_sink_test

template<typename T>
std::string
formatList(const T &l) {
    return fmt::format("[{}]", fmt::join(l, ", "));
}

template<typename T, typename U>
bool
indexesMatch(const T &lhs, const U &rhs) {
    auto index_match = [](const auto &l, const auto &r) { return l.index == r.index; };

    return std::equal(std::begin(lhs), std::end(lhs), std::begin(rhs), std::end(rhs), index_match);
}

using Scheduler = gr::scheduler::Simple<gr::scheduler::multiThreaded>;

const boost::ut::suite DataSinkTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::basic;
    using namespace gr::basic::data_sink_test;
    using namespace std::string_literals;

    "callback continuous mode"_test = [] {
        constexpr gr::Size_t  kSamples   = 200005;
        constexpr std::size_t kChunkSize = 1000;

        const auto srcTags = makeTestTags(0, 1000);

        gr::Graph testGraph;
        auto     &src = testGraph.emplaceBlock<gr::testing::TagSource<float>>(
                { { "n_samples_max", kSamples }, { "mark_tag", false }, { "signal_name", "test source" }, { "signal_unit", "test unit" }, { "signal_min", -42.f }, { "signal_max", 42.f } });
        auto &delay = testGraph.emplaceBlock<testing::Delay<float>>({ { "delay_ms", kProcessingDelayMs } });
        auto &sink  = testGraph.emplaceBlock<DataSink<float>>({ { "name", "test_sink" } });
        src.tags    = srcTags;

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(delay)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(delay).to<"in">(sink)));

        std::atomic<std::size_t> samplesSeen1 = 0;
        std::atomic<std::size_t> chunksSeen1  = 0;
        auto                     callback     = [&samplesSeen1, &chunksSeen1, &kChunkSize](std::span<const float> buffer) {
            for (std::size_t i = 0; i < buffer.size(); ++i) {
                expect(eq(buffer[i], static_cast<float>(samplesSeen1 + i)));
            }

            samplesSeen1 += buffer.size();
            chunksSeen1++;
            if (chunksSeen1 < 201) {
                expect(eq(buffer.size(), kChunkSize));
            } else {
                expect(eq(buffer.size(), 5UZ));
            }
        };

        std::mutex       m2;
        std::size_t      samplesSeen2 = 0;
        std::size_t      chunksSeen2  = 0;
        std::vector<Tag> receivedTags;
        auto             callbackWithTags = [&samplesSeen2, &chunksSeen2, &m2, &receivedTags, &kChunkSize](std::span<const float> buffer, std::span<const Tag> tags) {
            for (std::size_t i = 0; i < buffer.size(); ++i) {
                expect(eq(buffer[i], static_cast<float>(samplesSeen2 + i)));
            }

            for (const auto &tag : tags) {
                expect(ge(tag.index, 0));
                expect(lt(tag.index, static_cast<decltype(tag.index)>(buffer.size())));
            }

            auto             lg = std::lock_guard{ m2 };
            std::vector<Tag> adjusted;
            std::transform(tags.begin(), tags.end(), std::back_inserter(adjusted), [samplesSeen2](const auto &tag) {
                return Tag{ static_cast<Tag::signed_index_type>(samplesSeen2) + tag.index, tag.map };
            });
            receivedTags.insert(receivedTags.end(), adjusted.begin(), adjusted.end());
            samplesSeen2 += buffer.size();
            chunksSeen2++;
            if (chunksSeen2 < 201) {
                expect(eq(buffer.size(), kChunkSize));
            } else {
                expect(eq(buffer.size(), 5UZ));
            }
        };

        auto callbackWithTagsAndSink = [&sink](std::span<const float>, std::span<const Tag>, const DataSink<float> &passedSink) {
            expect(eq(passedSink.name.value, "test_sink"s));
            expect(eq(sink.unique_name, passedSink.unique_name));
        };

        auto registerThread = std::thread([&] {
            bool callbackRegistered                = false;
            bool callbackWithTagsRegistered        = false;
            bool callbackWithTagsAndSinkRegistered = false;

            expect(spinUntil(4s, [&] {
                if (!callbackRegistered) {
                    callbackRegistered = DataSinkRegistry::instance().registerStreamingCallback<float>(DataSinkQuery::sinkName("test_sink"), kChunkSize, callback);
                }
                if (!callbackWithTagsRegistered) {
                    callbackWithTagsRegistered = DataSinkRegistry::instance().registerStreamingCallback<float>(DataSinkQuery::sinkName("test_sink"), kChunkSize, callbackWithTags);
                }
                if (!callbackWithTagsAndSinkRegistered) {
                    callbackWithTagsAndSinkRegistered = DataSinkRegistry::instance().registerStreamingCallback<float>(DataSinkQuery::sinkName("test_sink"), kChunkSize, callbackWithTagsAndSink);
                }
                return callbackRegistered && callbackWithTagsRegistered && callbackWithTagsAndSinkRegistered;
            })) << boost::ut::fatal;
        });

        Scheduler sched{ std::move(testGraph) };
        expect(sched.runAndWait().has_value());
        registerThread.join();

        auto lg = std::lock_guard{ m2 };
        expect(eq(chunksSeen1.load(), 201UZ));
        expect(eq(chunksSeen2, 201UZ));
        expect(eq(samplesSeen1.load(), static_cast<std::size_t>(kSamples)));
        expect(eq(samplesSeen2, static_cast<std::size_t>(kSamples)));
        const auto &[metadataTags, nonMetadataTags] = extractMetadataTags(receivedTags);
        expect(eq(nonMetadataTags.size(), srcTags.size()));
        expect(eq(indexesMatch(nonMetadataTags, srcTags), true)) << fmt::format("{} != {}", formatList(receivedTags), formatList(srcTags));
        const auto metadata = latestMetadata(metadataTags);
        expect(eq(metadata.signal_name.value_or("<unset>"), "test source"s));
        expect(eq(metadata.signal_unit.value_or("<unset>"), "test unit"s));
        expect(eq(metadata.signal_min.value_or(-1234567.f), -42.f));
        expect(eq(metadata.signal_max.value_or(-1234567.f), 42.f));
    };

    "blocking polling continuous mode"_test = [] {
        constexpr gr::Size_t kSamples = 200000;

        gr::Graph  testGraph;
        const auto tags = makeTestTags(0, 1000);
        auto      &src  = testGraph.emplaceBlock<gr::testing::TagSource<float>>(
                { { "n_samples_max", kSamples }, { "mark_tag", false }, { "signal_name", "test source" }, { "signal_unit", "test unit" }, { "signal_min", -42.f }, { "signal_max", 42.f } });
        src.tags    = tags;
        auto &delay = testGraph.emplaceBlock<testing::Delay<float>>({ { "delay_ms", kProcessingDelayMs } });
        auto &sink  = testGraph.emplaceBlock<DataSink<float>>({ { "name", "test_sink" }, { "signal_name", "test signal" } });

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(delay)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(delay).to<"in">(sink)));

        auto runner1 = std::async([] {
            std::shared_ptr<DataSink<float>::Poller> poller;
            expect(spinUntil(4s, [&poller] {
                poller = DataSinkRegistry::instance().getStreamingPoller<float>(DataSinkQuery::sinkName("test_sink"), BlockingMode::Blocking);
                return poller != nullptr;
            })) << boost::ut::fatal;

            std::vector<float> received;
            bool               seenFinished = false;
            while (!seenFinished) {
                seenFinished = poller->finished;
                while (poller->process([&received](const auto &data) { received.insert(received.end(), data.begin(), data.end()); })) {
                }
            }

            return std::make_tuple(poller, received);
        });

        auto runner2 = std::async([] {
            std::shared_ptr<DataSink<float>::Poller> poller;
            expect(spinUntil(4s, [&poller] {
                poller = DataSinkRegistry::instance().getStreamingPoller<float>(DataSinkQuery::signalName("test signal"), BlockingMode::Blocking);
                return poller != nullptr;
            })) << boost::ut::fatal;
            std::vector<float> received;
            std::vector<Tag>   receivedTags;
            bool               seenFinished = false;
            while (!seenFinished) {
                seenFinished = poller->finished;
                while (poller->process([&received, &receivedTags](const auto &data, const auto &tags_) {
                    auto rtags = std::vector<Tag>(tags_.begin(), tags_.end());
                    for (auto &t : rtags) {
                        t.index += static_cast<int64_t>(received.size());
                    }
                    receivedTags.insert(receivedTags.end(), rtags.begin(), rtags.end());
                    received.insert(received.end(), data.begin(), data.end());
                })) {
                }
            }

            return std::make_tuple(poller, received, receivedTags);
        });

        {
            Scheduler sched{ std::move(testGraph) };
            expect(sched.runAndWait().has_value());

            const auto pollerAfterStop = DataSinkRegistry::instance().getStreamingPoller<float>(DataSinkQuery::sinkName("test_sink"));
            expect(eq(pollerAfterStop, nullptr));
        }

        const auto pollerAfterDestruction = DataSinkRegistry::instance().getStreamingPoller<float>(DataSinkQuery::sinkName("test_sink"));
        expect(!pollerAfterDestruction);

        std::vector<float> expected(kSamples);
        std::iota(expected.begin(), expected.end(), 0.0);

        const auto &[pollerDataOnly, received1]               = runner1.get();
        const auto &[pollerWithTags, received2, receivedTags] = runner2.get();
        const auto &[metadataTags, nonMetadataTags]           = extractMetadataTags(receivedTags);
        expect(eq(received1.size(), expected.size()));
        expect(eq(received1, expected));
        expect(eq(pollerDataOnly->drop_count.load(), 0UZ));
        expect(eq(received2.size(), expected.size()));
        expect(eq(received2, expected));
        expect(eq(nonMetadataTags.size(), tags.size()));
        expect(eq(indexesMatch(nonMetadataTags, tags), true)) << fmt::format("{} != {}", formatList(nonMetadataTags), formatList(tags));
        expect(eq(metadataTags.size(), 1UZ));
        expect(eq(metadataTags[0].index, 0));
        const auto metadata = latestMetadata(metadataTags);
        expect(eq(metadata.signal_name.value_or("<unset>"), "test source"s));
        expect(eq(metadata.signal_unit.value_or("<unset>"), "test unit"s));
        expect(eq(metadata.signal_min.value_or(-1234567.f), -42.f));
        expect(eq(metadata.signal_max.value_or(-1234567.f), 42.f));
        expect(eq(pollerWithTags->drop_count.load(), 0UZ));
    };

    "blocking polling trigger mode non-overlapping"_test = [] {
        constexpr gr::Size_t kSamples = 200000;

        gr::Graph testGraph;
        auto     &src = testGraph.emplaceBlock<gr::testing::TagSource<int32_t>>({ { "n_samples_max", kSamples }, { "mark_tag", false } });

        const auto tags = std::vector<Tag>{ { 3000, { { "TYPE", "TRIGGER" } } }, { 8000, { { "TYPE", "NO_TRIGGER" } } }, { 180000, { { "TYPE", "TRIGGER" } } } };
        src.tags        = tags;
        auto &delay     = testGraph.emplaceBlock<testing::Delay<int32_t>>({ { "delay_ms", kProcessingDelayMs } });
        auto &sink      = testGraph.emplaceBlock<DataSink<int32_t>>(
                { { "name", "test_sink" }, { "signal_name", "test signal" }, { "signal_unit", "none" }, { "signal_min", -2.0f }, { "signal_max", 2.0f } });

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(delay)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(delay).to<"in">(sink)));

        auto polling = std::async([] {
            auto isTrigger = [](const Tag &tag) {
                const auto v = tag.get("TYPE");
                return v && std::get<std::string>(v->get()) == "TRIGGER" ? TriggerMatchResult::Matching : TriggerMatchResult::Ignore;
            };

            std::shared_ptr<DataSink<int32_t>::DataSetPoller> poller;
            expect(spinUntil(4s, [&] {
                // lookup by signal name
                poller = DataSinkRegistry::instance().getTriggerPoller<int32_t>(DataSinkQuery::signalName("test signal"), isTrigger, 3, 2, BlockingMode::Blocking);
                return poller != nullptr;
            })) << boost::ut::fatal;
            std::vector<int32_t> receivedData;
            std::vector<Tag>     receivedTags;
            bool                 seenFinished = false;
            while (!seenFinished) {
                seenFinished            = poller->finished;
                [[maybe_unused]] auto r = poller->process([&receivedData, &receivedTags](const auto &datasets) {
                    for (const auto &dataset : datasets) {
                        receivedData.insert(receivedData.end(), dataset.signal_values.begin(), dataset.signal_values.end());
                        // signal info from sink settings
                        expect(eq(dataset.signal_names.size(), 1u));
                        expect(eq(dataset.signal_units.size(), 1u));
                        expect(eq(dataset.signal_ranges.size(), 1u));
                        expect(eq(dataset.timing_events.size(), 1u));
                        expect(eq(dataset.signal_names[0], "test signal"s));
                        expect(eq(dataset.signal_units[0], "none"s));
                        expect(eq(dataset.signal_ranges[0], std::vector<int32_t>{ -2, +2 }));
                        expect(eq(dataset.timing_events[0].size(), 1u));
                        expect(eq(dataset.timing_events[0][0].index, 3));
                        receivedTags.insert(receivedTags.end(), dataset.timing_events[0].begin(), dataset.timing_events[0].end());
                    }
                });
            }
            return std::make_tuple(poller, receivedData, receivedTags);
        });

        Scheduler sched{ std::move(testGraph) };
        expect(sched.runAndWait().has_value());

        const auto &[poller, receivedData, receivedTags] = polling.get();
        const auto expected_tags                         = { tags[0], tags[2] }; // triggers-only

        expect(eq(receivedData.size(), 10UZ));
        expect(eq(receivedData, std::vector<int32_t>{ 2997, 2998, 2999, 3000, 3001, 179997, 179998, 179999, 180000, 180001 }));
        expect(eq(receivedTags.size(), expected_tags.size()));

        expect(eq(poller->drop_count.load(), 0UZ));
    };

    "propagation of signal metadata per data set"_test = [] {
        constexpr gr::Size_t kSamples = 40000000;

        gr::Graph testGraph;
        auto     &src = testGraph.emplaceBlock<gr::testing::TagSource<int32_t>>(
                { { "n_samples_max", kSamples }, { "mark_tag", false }, { "signal_name", "test signal" }, { "signal_unit", "no unit" }, { "signal_min", -2.f }, { "signal_max", 2.f } });
        const auto tags = std::vector<Tag>{ { 39000000, { { "TYPE", "TRIGGER" } } } };
        src.tags        = tags;
        auto &delay     = testGraph.emplaceBlock<testing::Delay<int32_t>>({ { "delay_ms", kProcessingDelayMs } });
        auto &sink      = testGraph.emplaceBlock<DataSink<int32_t>>();

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(delay)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(delay).to<"in">(sink)));

        auto polling = std::async([] {
            std::vector<int32_t>                              receivedData;
            std::vector<Tag>                                  receivedTags;
            bool                                              seenFinished = false;
            auto                                              isTrigger    = [](const Tag &) { return TriggerMatchResult::Matching; };
            std::shared_ptr<DataSink<int32_t>::DataSetPoller> poller;
            expect(spinUntil(4s, [&] {
                poller = DataSinkRegistry::instance().getTriggerPoller<int32_t>(DataSinkQuery::signalName("test signal"), isTrigger, 0UZ, 2UZ, BlockingMode::Blocking);
                return poller != nullptr;
            })) << boost::ut::fatal;

            if (!poller) {
                return std::make_tuple(poller, receivedData, receivedTags);
            }

            while (!seenFinished) {
                seenFinished            = poller->finished;
                [[maybe_unused]] auto r = poller->process([&receivedData, &receivedTags](const auto &datasets) {
                    for (const auto &dataset : datasets) {
                        receivedData.insert(receivedData.end(), dataset.signal_values.begin(), dataset.signal_values.end());
                        // signal info from sink settings
                        expect(eq(dataset.signal_names.size(), 1u));
                        expect(eq(dataset.signal_units.size(), 1u));
                        expect(eq(dataset.signal_ranges.size(), 1u));
                        expect(eq(dataset.timing_events.size(), 1u));
                        expect(eq(dataset.signal_names[0], "test signal"s));
                        expect(eq(dataset.signal_units[0], "no unit"s));
                        expect(eq(dataset.signal_ranges[0], std::vector{ -2, +2 }));
                        expect(eq(dataset.timing_events[0].size(), 1u));
                        expect(eq(dataset.timing_events[0][0].index, 0));
                        receivedTags.insert(receivedTags.end(), dataset.timing_events[0].begin(), dataset.timing_events[0].end());
                    }
                });
            }
            return std::make_tuple(poller, receivedData, receivedTags);
        });

        Scheduler sched{ std::move(testGraph) };
        expect(sched.runAndWait().has_value());

        const auto &[_, receivedData, receivedTags] = polling.get();
        expect(eq(receivedData, std::vector<int32_t>{ 39000000, 39000001 }));
    };

    "blocking snapshot mode"_test = [] {
        constexpr gr::Size_t kSamples = 200000;

        gr::Graph testGraph;
        auto     &src = testGraph.emplaceBlock<gr::testing::TagSource<int32_t>>({ { "n_samples_max", kSamples },
                                                                                  { "mark_tag", false },
                                                                                  { "sample_rate", 10000.f },
                                                                                  { "signal_name", "test signal" },
                                                                                  { "signal_unit", "none" },
                                                                                  { "signal_min", 0.f },
                                                                                  { "signal_max", static_cast<float>(kSamples - 1) } });
        src.tags      = { { 3000, { { "TYPE", "TRIGGER" } } }, { 8000, { { "TYPE", "NO_TRIGGER" } } }, { 180000, { { "TYPE", "TRIGGER" } } } };
        auto &delay   = testGraph.emplaceBlock<testing::Delay<int32_t>>({ { "delay_ms", kProcessingDelayMs } });
        auto &sink    = testGraph.emplaceBlock<DataSink<int32_t>>({ { "name", "test_sink" } });

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(delay)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(delay).to<"in">(sink)));

        constexpr auto kDelay = std::chrono::milliseconds{ 500 }; // sample rate 10000 -> 5000 samples

        std::vector<int32_t> receivedDataCb;

        auto callback = [&receivedDataCb](const auto &dataset) { receivedDataCb.insert(receivedDataCb.end(), dataset.signal_values.begin(), dataset.signal_values.end()); };

        auto isTrigger = [](const Tag &tag) {
            const auto v = tag.get("TYPE");
            return (v && std::get<std::string>(v->get()) == "TRIGGER") ? TriggerMatchResult::Matching : TriggerMatchResult::Ignore;
        };

        auto registerThread = std::thread([&] {
            expect(spinUntil(4s, [&] { return DataSinkRegistry::instance().registerSnapshotCallback<int32_t>(DataSinkQuery::sinkName("test_sink"), isTrigger, kDelay, callback); }))
                    << boost::ut::fatal;
        });

        auto poller_result = std::async([isTrigger, kDelay] {
            std::shared_ptr<DataSink<int32_t>::DataSetPoller> poller;
            expect(spinUntil(4s, [&] {
                poller = DataSinkRegistry::instance().getSnapshotPoller<int32_t>(DataSinkQuery::sinkName("test_sink"), isTrigger, kDelay, BlockingMode::Blocking);
                return poller != nullptr;
            })) << boost::ut::fatal;

            std::vector<int32_t> receivedData;

            bool seenFinished = false;
            while (!seenFinished) {
                seenFinished            = poller->finished;
                [[maybe_unused]] auto r = poller->process([&receivedData](const auto &datasets) {
                    for (const auto &dataset : datasets) {
                        // signal info propagated from source to sink
                        expect(eq(dataset.signal_names.size(), 1u));
                        expect(eq(dataset.signal_units.size(), 1u));
                        expect(eq(dataset.signal_ranges.size(), 1u));
                        expect(eq(dataset.timing_events.size(), 1u));
                        expect(eq(dataset.signal_names[0], "test signal"s));
                        expect(eq(dataset.signal_units[0], "none"s));
                        expect(eq(dataset.signal_ranges[0], std::vector<int32_t>{ 0, kSamples - 1 }));
                        expect(eq(dataset.timing_events[0].size(), 1u));
                        expect(eq(dataset.timing_events[0][0].index, -5000));
                        receivedData.insert(receivedData.end(), dataset.signal_values.begin(), dataset.signal_values.end());
                    }
                });
            }

            return std::make_tuple(poller, receivedData);
        });

        Scheduler sched{ std::move(testGraph) };
        expect(sched.runAndWait().has_value());

        registerThread.join();

        const auto &[poller, receivedData] = poller_result.get();
        expect(eq(receivedDataCb, receivedData));
        expect(eq(receivedData, std::vector<int32_t>{ 8000, 185000 }));
        expect(eq(poller->drop_count.load(), 0UZ));
    };

    "blocking multiplexed mode"_test = [] {
        // Use large delay and timeout to ensure this also works in debug/coverage builds
        const auto tags = makeTestTags(0, 10000);

        const gr::Size_t n_samples = static_cast<gr::Size_t>(tags.size() * 10000 + 100000);
        gr::Graph        testGraph;
        auto            &src = testGraph.emplaceBlock<gr::testing::TagSource<int32_t>>({ { "n_samples_max", n_samples }, { "mark_tag", false } });
        src.tags             = tags;
        auto &delay          = testGraph.emplaceBlock<testing::Delay<int32_t>>({ { "delay_ms", 2500u } });
        auto &sink           = testGraph.emplaceBlock<DataSink<int32_t>>({ { "name", "test_sink" } });

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

        const auto matchers = std::array{ Matcher({}, -1, {}), Matcher(-1, {}, {}), Matcher(1, {}, {}), Matcher(1, {}, 2), Matcher({}, {}, 1) };

        // Following the patterns above, where each #/_ is 10000 samples
        const auto expected = std::array<std::vector<int32_t>, matchers.size()>{ { { 0, 29999, 30000, 59999, 60000, 89999, 90000, 119999, 120000, 149999, 150000, 249999 },
                                                                                   { 0, 59999, 60000, 119999, 120000, 219999 },
                                                                                   { 0, 59999 },
                                                                                   { 10000, 19999, 40000, 49999 },
                                                                                   { 0, 9999, 30000, 39999, 60000, 69999, 90000, 99999, 120000, 129999, 150000, 159999 } } };
        std::vector<std::future<std::vector<int32_t>>>    results;
        std::array<std::vector<int32_t>, matchers.size()> resultsCb;

        auto registerThread = std::thread([&] {
            std::array<bool, resultsCb.size()> registered;
            std::ranges::fill(registered, false);
            expect(spinUntil(3s, [&] {
                for (auto i = 0UZ; i < registered.size(); ++i) {
                    if (!registered[i]) {
                        auto callback = [&r = resultsCb[i]](const auto &dataset) {
                            r.push_back(dataset.signal_values.front());
                            r.push_back(dataset.signal_values.back());
                        };
                        registered[i] = DataSinkRegistry::instance().registerMultiplexedCallback<int32_t>(DataSinkQuery::sinkName("test_sink"), Matcher(matchers[i]), 100000, std::move(callback));
                    }
                }
                return std::ranges::all_of(registered, [](bool b) { return b; });
            })) << boost::ut::fatal;
        });

        for (std::size_t i = 0; i < matchers.size(); ++i) {
            auto f = std::async([i, &matchers]() {
                std::shared_ptr<DataSink<int32_t>::DataSetPoller> poller;
                expect(spinUntil(4s, [&] {
                    poller = DataSinkRegistry::instance().getMultiplexedPoller<int32_t>(DataSinkQuery::sinkName("test_sink"), Matcher(matchers[i]), 100000, BlockingMode::Blocking);
                    return poller != nullptr;
                })) << boost::ut::fatal;
                std::vector<int32_t> ranges;
                bool                 seenFinished = false;
                while (!seenFinished) {
                    seenFinished = poller->finished.load();
                    while (poller->process([&ranges](const auto &datasets) {
                        for (const auto &dataset : datasets) {
                            // default signal info, we didn't set anything
                            expect(eq(dataset.signal_names.size(), 1u));
                            expect(eq(dataset.signal_units.size(), 1u));
                            expect(eq(dataset.timing_events.size(), 1u));
                            expect(eq(dataset.signal_names[0], "unknown signal"s));
                            expect(eq(dataset.signal_units[0], "a.u."s));
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

        Scheduler sched{ std::move(testGraph) };
        expect(sched.runAndWait().has_value());
        registerThread.join();

        for (std::size_t i = 0; i < results.size(); ++i) {
            expect(eq(results[i].get(), expected[i]));
            expect(eq(resultsCb[i], expected[i]));
        }
    };

    "blocking polling trigger mode overlapping"_test = [] {
        constexpr std::uint32_t kSamples  = 150000;
        constexpr std::size_t   kTriggers = 300;

        gr::Graph testGraph;
        auto     &src = testGraph.emplaceBlock<gr::testing::TagSource<float>>({ { "n_samples_max", kSamples }, { "mark_tag", false } });

        for (std::size_t i = 0; i < kTriggers; ++i) {
            src.tags.push_back(Tag{ static_cast<Tag::signed_index_type>(60000 + i), { { "TYPE", "TRIGGER" } } });
        }

        auto &delay = testGraph.emplaceBlock<testing::Delay<float>>({ { "delay_ms", kProcessingDelayMs } });
        auto &sink  = testGraph.emplaceBlock<DataSink<float>>({ { "name", "test_sink" } });

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(delay)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(delay).to<"in">(sink)));

        auto polling = std::async([] {
            auto isTrigger = [](const Tag &) { return TriggerMatchResult::Matching; };

            std::shared_ptr<DataSink<float>::DataSetPoller> poller;
            expect(spinUntil(4s, [&] {
                poller = DataSinkRegistry::instance().getTriggerPoller<float>(DataSinkQuery::sinkName("test_sink"), isTrigger, 3000, 2000, BlockingMode::Blocking);
                return poller != nullptr;
            })) << boost::ut::fatal;
            std::vector<float> receivedData;
            std::vector<Tag>   receivedTags;
            bool               seenFinished = false;
            while (!seenFinished) {
                seenFinished = poller->finished.load();
                while (poller->process([&receivedData, &receivedTags](const auto &datasets) {
                    for (const auto &dataset : datasets) {
                        expect(eq(dataset.signal_values.size(), 5000u) >> fatal);
                        receivedData.push_back(dataset.signal_values.front());
                        receivedData.push_back(dataset.signal_values.back());
                        expect(eq(dataset.timing_events.size(), 1u));
                        expect(eq(dataset.timing_events[0].size(), 1u));
                        expect(eq(dataset.timing_events[0][0].index, 3000));
                        receivedTags.insert(receivedTags.end(), dataset.timing_events[0].begin(), dataset.timing_events[0].end());
                    }
                })) {
                }
            }
            return std::make_tuple(poller, receivedData, receivedTags);
        });

        Scheduler sched{ std::move(testGraph) };
        expect(sched.runAndWait().has_value());

        const auto &[poller, receivedData, receivedTags] = polling.get();
        auto expectedStart                               = std::vector<float>{ 57000, 61999, 57001, 62000, 57002 };
        expect(eq(poller->drop_count.load(), 0u));
        expect(eq(receivedData.size(), 2 * kTriggers) >> fatal);
        expect(eq(std::vector(receivedData.begin(), receivedData.begin() + 5), expectedStart));
        expect(eq(receivedTags.size(), kTriggers));
    };

    "callback trigger mode overlapping"_test = [] {
        constexpr std::uint32_t kSamples  = 150000;
        constexpr std::size_t   kTriggers = 300;

        gr::Graph testGraph;
        auto     &src = testGraph.emplaceBlock<gr::testing::TagSource<float>>({ { "n_samples_max", kSamples }, { "mark_tag", false } });

        for (std::size_t i = 0; i < kTriggers; ++i) {
            src.tags.push_back(Tag{ static_cast<Tag::signed_index_type>(60000 + i), { { "TYPE", "TRIGGER" } } });
        }

        auto &delay = testGraph.emplaceBlock<testing::Delay<float>>({ { "delay_ms", kProcessingDelayMs } });
        auto &sink  = testGraph.emplaceBlock<DataSink<float>>({ { "name", "test_sink" } });

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(delay)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(delay).to<"in">(sink)));

        auto isTrigger = [](const Tag &) { return TriggerMatchResult::Matching; };

        std::mutex         m;
        std::vector<float> receivedData;

        auto callback = [&receivedData, &m](auto &&dataset) {
            std::lock_guard lg{ m };
            expect(eq(dataset.signal_values.size(), 5000u));
            receivedData.push_back(dataset.signal_values.front());
            receivedData.push_back(dataset.signal_values.back());
        };

        auto registerThread = std::thread([&] {
            expect(spinUntil(4s, [&] { return DataSinkRegistry::instance().registerTriggerCallback<float>(DataSinkQuery::sinkName("test_sink"), isTrigger, 3000, 2000, callback); }))
                    << boost::ut::fatal;
        });

        Scheduler sched{ std::move(testGraph) };
        expect(sched.runAndWait().has_value());
        registerThread.join();

        std::lock_guard lg{ m };
        auto            expectedStart = std::vector<float>{ 57000, 61999, 57001, 62000, 57002 };
        expect(eq(receivedData.size(), 2 * kTriggers));
        expect(eq(std::vector(receivedData.begin(), receivedData.begin() + 5), expectedStart));
    };

    "non-blocking polling continuous mode"_test = [] {
        constexpr std::uint32_t kSamples = 200000;

        gr::Graph testGraph;
        auto     &src   = testGraph.emplaceBlock<gr::testing::TagSource<float>>({ { "n_samples_max", kSamples }, { "mark_tag", false } });
        auto     &delay = testGraph.emplaceBlock<testing::Delay<float>>({ { "delay_ms", kProcessingDelayMs } });
        auto     &sink  = testGraph.emplaceBlock<DataSink<float>>({ { "name", "test_sink" } });

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(delay)));
        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(delay).to<"in">(sink)));

        auto invalid_type_poller = DataSinkRegistry::instance().getStreamingPoller<double>(DataSinkQuery::sinkName("test_sink"));
        expect(eq(invalid_type_poller, nullptr));

        auto polling = std::async([] {
            std::shared_ptr<DataSink<float>::Poller> poller;
            expect(spinUntil(4s, [&poller] {
                poller = DataSinkRegistry::instance().getStreamingPoller<float>(DataSinkQuery::sinkName("test_sink"));
                return poller != nullptr;
            })) << boost::ut::fatal;

            std::size_t samplesSeen  = 0;
            bool        seenFinished = false;
            while (!seenFinished) {
                using namespace std::chrono_literals;
                std::this_thread::sleep_for(20ms);

                seenFinished = poller->finished.load();
                while (poller->process([&samplesSeen](const auto &data) { samplesSeen += data.size(); })) {
                }
            }

            return std::make_tuple(poller, samplesSeen);
        });

        Scheduler sched{ std::move(testGraph) };
        expect(sched.runAndWait().has_value());

        const auto &[poller, samplesSeen] = polling.get();
        expect(eq(samplesSeen + poller->drop_count, static_cast<std::size_t>(kSamples)));
    };
};

int
main() { /* tests are statically executed */
}
