#include <future>

#include <boost/ut.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/reflection.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

#include <gnuradio-4.0/basic/DataSink.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

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

template<typename T>
struct Source : public Block<Source<T>> {
    PortOut<T>       out;
    std::int32_t     n_samples_produced = 0;
    std::int32_t     n_samples_max      = 1024;
    std::size_t      n_tag_offset       = 0;
    float            sample_rate        = 1000.0f;
    T                next_value         = {};
    std::size_t      next_tag           = 0;
    std::vector<Tag> tags; // must be sorted by index, only one tag per sample

    void
    settingsChanged(const property_map &, const property_map &) {
        // optional init function that is called after construction and whenever settings change
        gr::publish_tag(out, { { "n_samples_max", n_samples_max } }, n_tag_offset);
    }

    work::Status
    processBulk(PublishableSpan auto &output) noexcept {
        auto nSamples = nSamplesToPublish();
        nSamples      = std::min(nSamples, static_cast<std::make_signed_t<std::size_t>>(output.size()));

        if (nSamples == -1) {
            this->requestStop();
            output.publish(0UZ);
            return work::Status::DONE;
        }
        if (next_tag < tags.size() && tags[next_tag].index <= static_cast<std::make_signed_t<std::size_t>>(n_samples_produced)) {
            Tag &out_tag  = this->output_tags()[0];
            out_tag       = Tag{ 0, tags[next_tag].map };
            out_tag.index = 0;
            this->forwardTags();
            next_tag++;
        }

        std::ranges::for_each(output | std::views::take(nSamples), [this](auto &elem) { elem = this->next_value++; });

        n_samples_produced += static_cast<std::int64_t>(nSamples);
        output.publish(static_cast<std::size_t>(nSamples));
        return n_samples_produced < n_samples_max ? work::Status::OK : work::Status::DONE;
    }

private:
    constexpr std::make_signed_t<std::size_t>
    nSamplesToPublish() noexcept {
        // TODO unify with other test sources
        // split into chunks so that we have a single tag at index 0 (or none)
        auto ret = static_cast<std::make_signed_t<std::size_t>>(n_samples_max - n_samples_produced);
        if (next_tag < tags.size()) {
            if (n_samples_produced < tags[next_tag].index) {
                ret = tags[next_tag].index - n_samples_produced;
            } else if (next_tag + 1 < tags.size()) {
                // tag at first sample? then read up until before next tag
                ret = tags[next_tag + 1].index - n_samples_produced;
            }
        }

        return ret > 0 ? ret : -1; // '-1' -> DONE, produced enough samples
    }
};

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

} // namespace gr::basic::data_sink_test

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (gr::basic::data_sink_test::Source<T>), out, n_samples_produced, n_samples_max, n_tag_offset, sample_rate);

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

const boost::ut::suite DataSinkTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::basic;
    using namespace gr::basic::data_sink_test;
    using namespace std::string_literals;

    "callback continuous mode"_test = [] {
        constexpr std::int32_t kSamples   = 200005;
        constexpr std::size_t  kChunkSize = 1000;

        const auto srcTags = makeTestTags(0, 1000);

        gr::Graph testGraph;
        auto     &src  = testGraph.emplaceBlock<Source<float>>({ { "n_samples_max", kSamples } });
        auto     &sink = testGraph.emplaceBlock<DataSink<float>>({ { "name", "test_sink" } });
        src.tags       = srcTags;

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(sink)));

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

        expect(DataSinkRegistry::instance().registerStreamingCallback<float>(DataSinkQuery::sinkName("test_sink"), kChunkSize, callback));
        expect(DataSinkRegistry::instance().registerStreamingCallback<float>(DataSinkQuery::sinkName("test_sink"), kChunkSize, callbackWithTags));
        expect(DataSinkRegistry::instance().registerStreamingCallback<float>(DataSinkQuery::sinkName("test_sink"), kChunkSize, callbackWithTagsAndSink));

        gr::scheduler::Simple sched{ std::move(testGraph) };
        sched.runAndWait();

        sink.stop(); // TODO the scheduler should call this

        auto lg = std::lock_guard{ m2 };
        expect(eq(chunksSeen1.load(), 201UZ));
        expect(eq(chunksSeen2, 201UZ));
        expect(eq(samplesSeen1.load(), static_cast<std::size_t>(kSamples)));
        expect(eq(samplesSeen2, static_cast<std::size_t>(kSamples)));
        expect(eq(indexesMatch(receivedTags, srcTags), true)) << fmt::format("{} != {}", formatList(receivedTags), formatList(srcTags));
    };

    "blocking polling continuous mode"_test = [] {
        constexpr std::int32_t kSamples = 200000;

        gr::Graph  testGraph;
        const auto tags = makeTestTags(0, 1000);
        auto      &src  = testGraph.emplaceBlock<Source<float>>({ { "n_samples_max", kSamples } });
        src.tags        = tags;
        auto &sink      = testGraph.emplaceBlock<DataSink<float>>({ { "name", "test_sink" } });

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(sink)));

        auto pollerDataOnly = DataSinkRegistry::instance().getStreamingPoller<float>(DataSinkQuery::sinkName("test_sink"), BlockingMode::Blocking);
        expect(neq(pollerDataOnly, nullptr));

        auto pollerWithTags = DataSinkRegistry::instance().getStreamingPoller<float>(DataSinkQuery::sinkName("test_sink"), BlockingMode::Blocking);
        expect(neq(pollerWithTags, nullptr));

        auto runner1 = std::async([poller = pollerDataOnly] {
            std::vector<float> received;
            bool               seenFinished = false;
            while (!seenFinished) {
                seenFinished = poller->finished;
                while (poller->process([&received](const auto &data) { received.insert(received.end(), data.begin(), data.end()); })) {
                }
            }

            return received;
        });

        auto runner2 = std::async([poller = pollerWithTags] {
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

            return std::make_tuple(received, receivedTags);
        });

        {
            gr::scheduler::Simple sched{ std::move(testGraph) };
            sched.runAndWait();

            sink.stop(); // TODO the scheduler should call this

            const auto pollerAfterStop = DataSinkRegistry::instance().getStreamingPoller<float>(DataSinkQuery::sinkName("test_sink"));
            expect(pollerAfterStop->finished.load());
        }

        const auto pollerAfterDestruction = DataSinkRegistry::instance().getStreamingPoller<float>(DataSinkQuery::sinkName("test_sink"));
        expect(!pollerAfterDestruction);

        std::vector<float> expected(kSamples);
        std::iota(expected.begin(), expected.end(), 0.0);

        const auto received1                  = runner1.get();
        const auto &[received2, receivedTags] = runner2.get();
        expect(eq(received1.size(), expected.size()));
        expect(eq(received1, expected));
        expect(eq(pollerDataOnly->drop_count.load(), 0UZ));
        expect(eq(received2.size(), expected.size()));
        expect(eq(received2, expected));
        expect(eq(receivedTags.size(), tags.size()));
        expect(eq(indexesMatch(receivedTags, tags), true)) << fmt::format("{} != {}", formatList(receivedTags), formatList(tags));
        expect(eq(pollerWithTags->drop_count.load(), 0UZ));
    };

    "blocking polling trigger mode non-overlapping"_test = [] {
        constexpr std::int32_t kSamples = 200000;

        gr::Graph  testGraph;
        auto      &src  = testGraph.emplaceBlock<Source<int32_t>>({ { "n_samples_max", kSamples } });
        const auto tags = std::vector<Tag>{ { 3000, { { "TYPE", "TRIGGER" } } }, { 8000, { { "TYPE", "NO_TRIGGER" } } }, { 180000, { { "TYPE", "TRIGGER" } } } };
        src.tags        = tags;
        auto &sink      = testGraph.emplaceBlock<DataSink<int32_t>>(
                { { "name", "test_sink" }, { "signal_name", "test signal" }, { "signal_unit", "none" }, { "signal_min", int32_t{ 0 } }, { "signal_max", int32_t{ kSamples - 1 } } });

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(sink)));

        auto isTrigger = [](const Tag &tag) {
            const auto v = tag.get("TYPE");
            return v && std::get<std::string>(v->get()) == "TRIGGER" ? TriggerMatchResult::Matching : TriggerMatchResult::Ignore;
        };

        // lookup by signal name
        auto poller = DataSinkRegistry::instance().getTriggerPoller<int32_t>(DataSinkQuery::signalName("test signal"), isTrigger, 3, 2, BlockingMode::Blocking);
        expect(neq(poller, nullptr));

        auto polling = std::async([poller] {
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
                        expect(eq(dataset.signal_ranges[0], std::vector<int32_t>{ 0, kSamples - 1 }));
                        expect(eq(dataset.timing_events[0].size(), 1u));
                        expect(eq(dataset.timing_events[0][0].index, 3));
                        receivedTags.insert(receivedTags.end(), dataset.timing_events[0].begin(), dataset.timing_events[0].end());
                    }
                });
            }
            return std::make_tuple(receivedData, receivedTags);
        });

        gr::scheduler::Simple sched{ std::move(testGraph) };
        sched.runAndWait();

        sink.stop(); // TODO the scheduler should call this

        const auto &[receivedData, receivedTags] = polling.get();
        const auto expected_tags                 = { tags[0], tags[2] }; // triggers-only

        expect(eq(receivedData.size(), 10UZ));
        expect(eq(receivedData, std::vector<int32_t>{ 2997, 2998, 2999, 3000, 3001, 179997, 179998, 179999, 180000, 180001 }));
        expect(eq(receivedTags.size(), expected_tags.size()));

        expect(eq(poller->drop_count.load(), 0UZ));
    };

    "blocking snapshot mode"_test = [] {
        constexpr std::int32_t kSamples = 200000;

        gr::Graph testGraph;
        auto     &src = testGraph.emplaceBlock<Source<int32_t>>({ { "n_samples_max", kSamples } });
        src.tags      = { { 0,
                            { { std::string(tag::SIGNAL_NAME.key()), "test signal" },
                              { std::string(tag::SIGNAL_UNIT.key()), "none" },
                              { std::string(tag::SIGNAL_MIN.key()), int32_t{ 0 } },
                              { std::string(tag::SIGNAL_MAX.key()), kSamples - 1 } } },
                          { 3000, { { "TYPE", "TRIGGER" } } },
                          { 8000, { { "TYPE", "NO_TRIGGER" } } },
                          { 180000, { { "TYPE", "TRIGGER" } } } };
        auto &sink    = testGraph.emplaceBlock<DataSink<int32_t>>({ { "name", "test_sink" }, { "sample_rate", 10000.f } });

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(sink)));

        auto isTrigger = [](const Tag &tag) {
            const auto v = tag.get("TYPE");
            return (v && std::get<std::string>(v->get()) == "TRIGGER") ? TriggerMatchResult::Matching : TriggerMatchResult::Ignore;
        };

        const auto delay  = std::chrono::milliseconds{ 500 }; // sample rate 10000 -> 5000 samples
        auto       poller = DataSinkRegistry::instance().getSnapshotPoller<int32_t>(DataSinkQuery::sinkName("test_sink"), isTrigger, delay, BlockingMode::Blocking);
        expect(neq(poller, nullptr));

        std::vector<int32_t> receivedDataCb;

        auto callback = [&receivedDataCb](const auto &dataset) { receivedDataCb.insert(receivedDataCb.end(), dataset.signal_values.begin(), dataset.signal_values.end()); };

        expect(DataSinkRegistry::instance().registerSnapshotCallback<int32_t>(DataSinkQuery::sinkName("test_sink"), isTrigger, delay, callback));

        auto poller_result = std::async([poller] {
            std::vector<int32_t> receivedData;

            bool seenFinished = false;
            while (!seenFinished) {
                seenFinished            = poller->finished;
                [[maybe_unused]] auto r = poller->process([&receivedData](const auto &datasets) {
                    for (const auto &dataset : datasets) {
                        // signal info from tags
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

            return receivedData;
        });

        gr::scheduler::Simple sched{ std::move(testGraph) };
        sched.runAndWait();

        sink.stop(); // TODO the scheduler should call this

        const auto receivedData = poller_result.get();
        expect(eq(receivedDataCb, receivedData));
        expect(eq(receivedData, std::vector<int32_t>{ 8000, 185000 }));
        expect(eq(poller->drop_count.load(), 0UZ));
    };

    "blocking multiplexed mode"_test = [] {
        const auto tags = makeTestTags(0, 10000);

        const std::int32_t n_samples = static_cast<std::int32_t>(tags.size() * 10000 + 100000);
        gr::Graph          testGraph;
        auto              &src = testGraph.emplaceBlock<Source<int32_t>>({ { "n_samples_max", n_samples } });
        src.tags               = tags;
        auto &sink             = testGraph.emplaceBlock<DataSink<int32_t>>({ { "name", "test_sink" } });

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(sink)));

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
        std::array<std::shared_ptr<DataSink<int32_t>::DataSetPoller>, matchers.size()> pollers;

        std::vector<std::future<std::vector<int32_t>>>    results;
        std::array<std::vector<int32_t>, matchers.size()> resultsCb;

        for (std::size_t i = 0; i < resultsCb.size(); ++i) {
            auto callback = [&r = resultsCb[i]](const auto &dataset) {
                r.push_back(dataset.signal_values.front());
                r.push_back(dataset.signal_values.back());
            };
            expect(eq(DataSinkRegistry::instance().registerMultiplexedCallback<int32_t>(DataSinkQuery::sinkName("test_sink"), Matcher(matchers[i]), 100000, std::move(callback)), true));

            pollers[i] = DataSinkRegistry::instance().getMultiplexedPoller<int32_t>(DataSinkQuery::sinkName("test_sink"), Matcher(matchers[i]), 100000, BlockingMode::Blocking);
            expect(neq(pollers[i], nullptr));
        }

        for (std::size_t i = 0; i < pollers.size(); ++i) {
            auto f = std::async([poller = pollers[i]] {
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

        gr::scheduler::Simple sched{ std::move(testGraph) };
        sched.runAndWait();

        sink.stop(); // TODO the scheduler should call this

        for (std::size_t i = 0; i < results.size(); ++i) {
            expect(eq(results[i].get(), expected[i]));
            expect(eq(resultsCb[i], expected[i]));
        }
    };

    "blocking polling trigger mode overlapping"_test = [] {
        constexpr std::int32_t kSamples  = 150000;
        constexpr std::size_t  kTriggers = 300;

        gr::Graph testGraph;
        auto     &src = testGraph.emplaceBlock<Source<float>>({ { "n_samples_max", kSamples } });

        for (std::size_t i = 0; i < kTriggers; ++i) {
            src.tags.push_back(Tag{ static_cast<Tag::signed_index_type>(60000 + i), { { "TYPE", "TRIGGER" } } });
        }

        auto &sink = testGraph.emplaceBlock<DataSink<float>>({ { "name", "test_sink" } });

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(sink)));

        auto isTrigger = [](const Tag &) { return TriggerMatchResult::Matching; };

        auto poller = DataSinkRegistry::instance().getTriggerPoller<float>(DataSinkQuery::sinkName("test_sink"), isTrigger, 3000, 2000, BlockingMode::Blocking);
        expect(neq(poller, nullptr));

        auto polling = std::async([poller] {
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
            return std::make_tuple(receivedData, receivedTags);
        });

        gr::scheduler::Simple sched{ std::move(testGraph) };
        sched.runAndWait();

        sink.stop(); // TODO the scheduler should call this

        const auto &[receivedData, receivedTags] = polling.get();
        auto expectedStart                       = std::vector<float>{ 57000, 61999, 57001, 62000, 57002 };
        expect(eq(poller->drop_count.load(), 0u));
        expect(eq(receivedData.size(), 2 * kTriggers) >> fatal);
        expect(eq(std::vector(receivedData.begin(), receivedData.begin() + 5), expectedStart));
        expect(eq(receivedTags.size(), kTriggers));
    };

    "callback trigger mode overlapping"_test = [] {
        constexpr std::int32_t kSamples  = 150000;
        constexpr std::size_t  kTriggers = 300;

        gr::Graph testGraph;
        auto     &src = testGraph.emplaceBlock<Source<float>>({ { "n_samples_max", kSamples } });

        for (std::size_t i = 0; i < kTriggers; ++i) {
            src.tags.push_back(Tag{ static_cast<Tag::signed_index_type>(60000 + i), { { "TYPE", "TRIGGER" } } });
        }

        auto &sink = testGraph.emplaceBlock<DataSink<float>>({ { "name", "test_sink" } });

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(sink)));

        auto isTrigger = [](const Tag &) { return TriggerMatchResult::Matching; };

        std::mutex         m;
        std::vector<float> receivedData;

        auto callback = [&receivedData, &m](auto &&dataset) {
            std::lock_guard lg{ m };
            expect(eq(dataset.signal_values.size(), 5000u));
            receivedData.push_back(dataset.signal_values.front());
            receivedData.push_back(dataset.signal_values.back());
        };

        DataSinkRegistry::instance().registerTriggerCallback<float>(DataSinkQuery::sinkName("test_sink"), isTrigger, 3000, 2000, callback);

        gr::scheduler::Simple sched{ std::move(testGraph) };
        sched.runAndWait();

        sink.stop(); // TODO the scheduler should call this

        std::lock_guard lg{ m };
        auto            expectedStart = std::vector<float>{ 57000, 61999, 57001, 62000, 57002 };
        expect(eq(receivedData.size(), 2 * kTriggers));
        expect(eq(std::vector(receivedData.begin(), receivedData.begin() + 5), expectedStart));
    };

    "non-blocking polling continuous mode"_test = [] {
        constexpr std::int32_t kSamples = 200000;

        gr::Graph testGraph;
        auto     &src  = testGraph.emplaceBlock<Source<float>>({ { "n_samples_max", kSamples } });
        auto     &sink = testGraph.emplaceBlock<DataSink<float>>({ { "name", "test_sink" } });

        expect(eq(ConnectionResult::SUCCESS, testGraph.connect<"out">(src).to<"in">(sink)));

        auto invalid_type_poller = DataSinkRegistry::instance().getStreamingPoller<double>(DataSinkQuery::sinkName("test_sink"));
        expect(eq(invalid_type_poller, nullptr));

        auto poller = DataSinkRegistry::instance().getStreamingPoller<float>(DataSinkQuery::sinkName("test_sink"));
        expect(neq(poller, nullptr));

        auto polling = std::async([poller] {
            expect(neq(poller, nullptr));
            std::size_t samplesSeen  = 0;
            bool        seenFinished = false;
            while (!seenFinished) {
                using namespace std::chrono_literals;
                std::this_thread::sleep_for(20ms);

                seenFinished = poller->finished.load();
                while (poller->process([&samplesSeen](const auto &data) { samplesSeen += data.size(); })) {
                }
            }

            return samplesSeen;
        });

        gr::scheduler::Simple sched{ std::move(testGraph) };
        sched.runAndWait();

        sink.stop(); // TODO the scheduler should call this

        const auto samplesSeen = polling.get();
        expect(eq(samplesSeen + poller->drop_count, static_cast<std::size_t>(kSamples)));
    };
};

int
main() { /* tests are statically executed */
}
