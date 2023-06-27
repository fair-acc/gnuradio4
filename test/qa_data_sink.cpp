#include <boost/ut.hpp>

#include <buffer.hpp>
#include <data_sink.hpp>
#include <graph.hpp>
#include <node.hpp>
#include <reflection.hpp>
#include <scheduler.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <future>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

template<>
struct fmt::formatter<fair::graph::tag_t> {
    template<typename ParseContext>
    constexpr auto
    parse(ParseContext &ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    constexpr auto
    format(const fair::graph::tag_t &tag, FormatContext &ctx) const {
        return fmt::format_to(ctx.out(), "{}", tag.index);
    }
};

namespace fair::graph::data_sink_test {

static constexpr std::int32_t n_samples = 200000;

template<typename T>
struct Source : public node<Source<T>> {
    OUT<T>             out;
    std::int32_t       n_samples_produced = 0;
    std::int32_t       n_samples_max      = 1024;
    std::size_t        n_tag_offset       = 0;
    float              sample_rate        = 1000.0f;
    T                  next_value         = {};
    std::size_t        next_tag           = 0;
    std::vector<tag_t> tags; // must be sorted by index, only one tag per sample

    void
    init(const property_map &, const property_map &) {
        // optional init function that is called after construction and whenever settings change
        fair::graph::publish_tag(out, { { "n_samples_max", n_samples_max } }, n_tag_offset);
    }

    constexpr std::int64_t
    available_samples(const Source &) noexcept {
        const auto ret = static_cast<std::int64_t>(n_samples_max - n_samples_produced);
        // forcing one sample, at a time, see below
        return ret > 0 ? 1 : -1; // '-1' -> DONE, produced enough samples
    }

    T
    process_one() noexcept {
        if (next_tag < tags.size() && tags[next_tag].index <= static_cast<std::make_signed_t<std::size_t>>(n_samples_produced)) {
            tag_t &out_tag = this->output_tags()[0];
            // TODO when not enforcing single samples in available_samples, one would have to do:
            // const auto base = std::max(out.streamWriter().position() + 1, tag_t::signed_index_type{0});
            // out_tag        = tag_t{ tags[next_tag].index - base, tags[next_tag].map };
            // Still think there could be nicer API to set a tag from process_one()
            out_tag = tag_t{ 0, tags[next_tag].map };
            this->forward_tags();
            next_tag++;
        }

        n_samples_produced++;
        return next_value++;
    }
};

/**
 * Example tag observer (TriggerObserver implementation) for the multiplexed listener case (interleaved data). As a toy example, we use
 * data tagged as Year/Month/Day.
 *
 * For each of year, month, day, the user can specify whether:
 *
 *  - option not set: The field is to be ignored
 *  - -1: Whenever a change between the previous and the current tag is observed, start a new data set (StopAndStart)
 *  - other values >= 0: A new dataset is started when the tag matches, and stopped, when a tag doesn't match
 *
 * (Note that the TriggerObserver is stateful and remembers the last tag seen, other than a stateless TriggerPredicate)
 */
struct Observer {
    std::optional<int>                       year;
    std::optional<int>                       month;
    std::optional<int>                       day;
    std::optional<std::tuple<int, int, int>> last_seen;
    bool                                     last_matched = false;

    explicit Observer(std::optional<int> year_, std::optional<int> month_, std::optional<int> day_) : year(year_), month(month_), day(day_) {}

    static inline bool
    same(int x, std::optional<int> other) {
        return other && x == *other;
    }

    static inline bool
    changed(int x, std::optional<int> other) {
        return !same(x, other);
    }

    trigger_test_result
    operator()(const tag_t &tag) {
        const auto ty = tag.get("YEAR");
        const auto tm = tag.get("MONTH");
        const auto td = tag.get("DAY");
        if (!ty || !tm || !td) {
            return trigger_test_result::Ignore;
        }

        const auto tup                    = std::make_tuple(std::get<int>(ty->get()), std::get<int>(tm->get()), std::get<int>(td->get()));
        const auto &[y, m, d]             = tup;
        const auto          ly            = last_seen ? std::optional<int>(std::get<0>(*last_seen)) : std::nullopt;
        const auto          lm            = last_seen ? std::optional<int>(std::get<1>(*last_seen)) : std::nullopt;
        const auto          ld            = last_seen ? std::optional<int>(std::get<2>(*last_seen)) : std::nullopt;

        const auto          year_restart  = year && *year == -1 && changed(y, ly);
        const auto          year_matches  = !year || *year == -1 || same(y, year);
        const auto          month_restart = month && *month == -1 && changed(m, lm);
        const auto          month_matches = !month || *month == -1 || same(m, month);
        const auto          day_restart   = day && *day == -1 && changed(d, ld);
        const auto          day_matches   = !day || *day == -1 || same(d, day);
        const auto          matches       = year_matches && month_matches && day_matches;
        const auto          restart       = year_restart || month_restart || day_restart;

        trigger_test_result r             = trigger_test_result::Ignore;

        if (!matches) {
            r = trigger_test_result::NotMatching;
        } else if (!last_matched || restart) {
            r = trigger_test_result::Matching;
        }

        last_seen    = tup;
        last_matched = matches;
        return r;
    }
};

static tag_t
make_tag(tag_t::signed_index_type index, int year, int month, int day) {
    return tag_t{ index, { { "YEAR", year }, { "MONTH", month }, { "DAY", day } } };
}

static std::vector<tag_t>
make_test_tags(tag_t::signed_index_type first_index, tag_t::signed_index_type interval) {
    std::vector<tag_t> tags;
    for (int y = 1; y <= 3; ++y) {
        for (int m = 1; m <= 2; ++m) {
            for (int d = 1; d <= 3; ++d) {
                tags.push_back(make_tag(first_index, y, m, d));
                first_index += interval;
            }
        }
    }
    return tags;
}

static std::string
to_ascii_art(std::span<trigger_test_result> states) {
    bool        started = false;
    std::string r;
    for (auto s : states) {
        switch (s) {
        case trigger_test_result::Matching:
            r += started ? "||#" : "|#";
            started = true;
            break;
        case trigger_test_result::NotMatching:
            r += started ? "|_" : "_";
            started = false;
            break;
        case trigger_test_result::Ignore: r += started ? "#" : "_"; break;
        }
    };
    return r;
}

template<TriggerObserver O>
std::string
run_observer_test(std::span<const tag_t> tags, O o) {
    std::vector<trigger_test_result> r;
    r.reserve(tags.size());
    for (const auto &tag : tags) {
        r.push_back(o(tag));
    }
    return to_ascii_art(r);
}

} // namespace fair::graph::data_sink_test

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (fair::graph::data_sink_test::Source<T>), out, n_samples_produced, n_samples_max, n_tag_offset, sample_rate);

template<typename T>
std::string
format_list(const T &l) {
    return fmt::format("[{}]", fmt::join(l, ", "));
}

template<typename T, typename U>
bool
indexes_match(const T &lhs, const U &rhs) {
    auto index_match = [](const auto &l, const auto &r) { return l.index == r.index; };

    return std::equal(std::begin(lhs), std::end(lhs), std::begin(rhs), std::end(rhs), index_match);
}

const boost::ut::suite DataSinkTests = [] {
    using namespace boost::ut;
    using namespace fair::graph;
    using namespace fair::graph::data_sink_test;
    using namespace std::string_literals;

    "callback continuous mode"_test = [] {
        static constexpr std::int32_t n_samples  = 200005;
        static constexpr std::size_t  chunk_size = 1000;

        const auto                    src_tags   = make_test_tags(0, 1000);

        graph                         flow_graph;
        auto                         &src  = flow_graph.make_node<Source<float>>({ { "n_samples_max", n_samples } });
        auto                         &sink = flow_graph.make_node<data_sink<float>>();
        src.tags                           = src_tags;
        sink.set_name("test_sink");

        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink)));

        std::atomic<std::size_t> samples_seen1 = 0;
        std::atomic<std::size_t> chunks_seen1  = 0;
        auto                     callback      = [&samples_seen1, &chunks_seen1](std::span<const float> buffer) {
            for (std::size_t i = 0; i < buffer.size(); ++i) {
                expect(eq(buffer[i], static_cast<float>(samples_seen1 + i)));
            }

            samples_seen1 += buffer.size();
            chunks_seen1++;
            if (chunks_seen1 < 201) {
                expect(eq(buffer.size(), chunk_size));
            } else {
                expect(eq(buffer.size(), 5));
            }
        };

        std::mutex         m2;
        std::size_t        samples_seen2 = 0;
        std::size_t        chunks_seen2  = 0;
        std::vector<tag_t> received_tags;
        auto               callback_with_tags = [&samples_seen2, &chunks_seen2, &m2, &received_tags](std::span<const float> buffer, std::span<const tag_t> tags) {
            for (std::size_t i = 0; i < buffer.size(); ++i) {
                expect(eq(buffer[i], static_cast<float>(samples_seen2 + i)));
            }

            for (const auto &tag : tags) {
                ge(tag.index, static_cast<tag_t::signed_index_type>(samples_seen2));
                lt(tag.index, samples_seen2 + buffer.size());
            }

            auto               lg = std::lock_guard{ m2 };
            std::vector<tag_t> adjusted;
            std::transform(tags.begin(), tags.end(), std::back_inserter(adjusted), [samples_seen2](const auto &tag) {
                return tag_t{ static_cast<tag_t::signed_index_type>(samples_seen2) + tag.index, tag.map };
            });
            received_tags.insert(received_tags.end(), adjusted.begin(), adjusted.end());
            samples_seen2 += buffer.size();
            chunks_seen2++;
            if (chunks_seen2 < 201) {
                expect(eq(buffer.size(), chunk_size));
            } else {
                expect(eq(buffer.size(), 5));
            }
        };

        expect(data_sink_registry::instance().register_streaming_callback<float>(data_sink_query::sink_name("test_sink"), chunk_size, callback));
        expect(data_sink_registry::instance().register_streaming_callback<float>(data_sink_query::sink_name("test_sink"), chunk_size, callback_with_tags));

        fair::graph::scheduler::simple sched{ std::move(flow_graph) };
        sched.work();

        sink.stop(); // TODO the scheduler should call this

        auto lg = std::lock_guard{ m2 };
        expect(eq(chunks_seen1.load(), 201));
        expect(eq(chunks_seen2, 201));
        expect(eq(samples_seen1.load(), n_samples));
        expect(eq(samples_seen2, n_samples));
        expect(eq(indexes_match(received_tags, src_tags), true)) << fmt::format("{} != {}", format_list(received_tags), format_list(src_tags));
    };

    "blocking polling continuous mode"_test = [] {
        constexpr std::int32_t n_samples = 200000;

        graph                  flow_graph;
        const auto             tags = make_test_tags(0, 1000);
        auto                  &src  = flow_graph.make_node<Source<float>>({ { "n_samples_max", n_samples } });
        src.tags                    = tags;
        auto &sink                  = flow_graph.make_node<data_sink<float>>();
        sink.set_name("test_sink");

        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink)));

        std::atomic<std::size_t> samples_seen     = 0;

        auto                     poller_data_only = data_sink_registry::instance().get_streaming_poller<float>(data_sink_query::sink_name("test_sink"), blocking_mode::Blocking);
        expect(neq(poller_data_only, nullptr));

        auto poller_with_tags = data_sink_registry::instance().get_streaming_poller<float>(data_sink_query::sink_name("test_sink"), blocking_mode::Blocking);
        expect(neq(poller_with_tags, nullptr));

        auto                           runner1 = std::async([poller = poller_data_only] {
            std::vector<float> received;
            bool               seen_finished = false;
            while (!seen_finished) {
                // TODO make finished vs. pending data handling actually thread-safe
                seen_finished = poller->finished;
                while (poller->process([&received](const auto &data) { received.insert(received.end(), data.begin(), data.end()); })) {
                }
            }

            return received;
        });

        auto                           runner2 = std::async([poller = poller_with_tags] {
            std::vector<float> received;
            std::vector<tag_t> received_tags;
            bool               seen_finished = false;
            while (!seen_finished) {
                // TODO make finished vs. pending data handling actually thread-safe
                seen_finished = poller->finished;
                while (poller->process([&received, &received_tags](const auto &data, const auto &tags_) {
                    auto tags = std::vector<tag_t>(tags_.begin(), tags_.end());
                    for (auto &t : tags) {
                        t.index += static_cast<int64_t>(received.size());
                    }
                    received_tags.insert(received_tags.end(), tags.begin(), tags.end());
                    received.insert(received.end(), data.begin(), data.end());
                })) {
                }
            }

            return std::make_tuple(received, received_tags);
        });

        fair::graph::scheduler::simple sched{ std::move(flow_graph) };
        sched.work();

        sink.stop(); // TODO the scheduler should call this

        std::vector<float> expected(n_samples);
        std::iota(expected.begin(), expected.end(), 0.0);

        const auto received1                   = runner1.get();
        const auto &[received2, received_tags] = runner2.get();
        expect(eq(received1.size(), expected.size()));
        expect(eq(received1, expected));
        expect(eq(poller_data_only->drop_count.load(), 0));
        expect(eq(received2.size(), expected.size()));
        expect(eq(received2, expected));
        expect(eq(received_tags.size(), tags.size()));
        expect(eq(indexes_match(received_tags, tags), true)) << fmt::format("{} != {}", format_list(received_tags), format_list(tags));
        expect(eq(poller_with_tags->drop_count.load(), 0));
    };

    "blocking polling trigger mode non-overlapping"_test = [] {
        constexpr std::int32_t n_samples = 200000;

        graph                  flow_graph;
        auto                  &src  = flow_graph.make_node<Source<int32_t>>({ { "n_samples_max", n_samples } });
        const auto             tags = std::vector<tag_t>{ { 3000, { { "TYPE", "TRIGGER" } } }, tag_t{ 8000, { { "TYPE", "NO_TRIGGER" } } }, { 180000, { { "TYPE", "TRIGGER" } } } };
        src.tags                    = tags;
        auto &sink                  = flow_graph.make_node<data_sink<int32_t>>();
        sink.set_name("test_sink");

        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink)));

        auto is_trigger = [](const tag_t &tag) {
            const auto v = tag.get("TYPE");
            return v && std::get<std::string>(v->get()) == "TRIGGER" ? trigger_test_result::Matching : trigger_test_result::Ignore;
        };

        auto poller = data_sink_registry::instance().get_trigger_poller<int32_t>(data_sink_query::sink_name("test_sink"), is_trigger, 3, 2, blocking_mode::Blocking);
        expect(neq(poller, nullptr));

        auto                           polling = std::async([poller] {
            std::vector<int32_t> received_data;
            std::vector<tag_t>   received_tags;
            bool                 seen_finished = false;
            while (!seen_finished) {
                seen_finished           = poller->finished;
                [[maybe_unused]] auto r = poller->process_one([&received_data, &received_tags](const auto &dataset) {
                    received_data.insert(received_data.end(), dataset.signal_values.begin(), dataset.signal_values.end());
                    expect(eq(dataset.timing_events.size(), 1u)) >> fatal;
                    expect(eq(dataset.timing_events[0].size(), 1u));
                    expect(eq(dataset.timing_events[0][0].index, 3));
                    received_tags.insert(received_tags.end(), dataset.timing_events[0].begin(), dataset.timing_events[0].end());
                });
            }
            return std::make_tuple(received_data, received_tags);
        });

        fair::graph::scheduler::simple sched{ std::move(flow_graph) };
        sched.work();

        sink.stop(); // TODO the scheduler should call this

        const auto &[received_data, received_tags] = polling.get();
        const auto expected_tags                   = { tags[0], tags[2] }; // triggers-only

        expect(eq(received_data.size(), 10));
        expect(eq(received_data, std::vector<int32_t>{ 2997, 2998, 2999, 3000, 3001, 179997, 179998, 179999, 180000, 180001 }));
        expect(eq(received_tags.size(), expected_tags.size()));

        expect(eq(poller->drop_count.load(), 0));
    };

    "blocking polling snapshot mode"_test = [] {
        constexpr std::int32_t n_samples = 200000;

        graph                  flow_graph;
        auto                  &src = flow_graph.make_node<Source<int32_t>>({ { "n_samples_max", n_samples } });
        src.tags                   = { { 3000, { { "TYPE", "TRIGGER" } } }, tag_t{ 8000, { { "TYPE", "NO_TRIGGER" } } }, { 180000, { { "TYPE", "TRIGGER" } } } };
        auto &sink                 = flow_graph.make_node<data_sink<int32_t>>({ {"sample_rate", 10000.f } } );
        sink.set_name("test_sink");

        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink)));

        auto is_trigger = [](const tag_t &tag) {
            const auto v = tag.get("TYPE");
            return (v && std::get<std::string>(v->get()) == "TRIGGER") ? trigger_test_result::Matching : trigger_test_result::Ignore;
        };

        const auto delay  = std::chrono::milliseconds{ 500 }; // sample rate 10000 -> 5000 samples
        auto       poller = data_sink_registry::instance().get_snapshot_poller<int32_t>(data_sink_query::sink_name("test_sink"), is_trigger, delay, blocking_mode::Blocking);
        expect(neq(poller, nullptr));

        auto                           poller_result = std::async([poller] {
            std::vector<int32_t> received_data;

            bool                 seen_finished = false;
            while (!seen_finished) {
                seen_finished           = poller->finished;
                [[maybe_unused]] auto r = poller->process_one([&received_data](const auto &dataset) {
                    expect(eq(dataset.timing_events.size(), 1u)) >> fatal;
                    expect(eq(dataset.timing_events[0].size(), 1u));
                    expect(eq(dataset.timing_events[0][0].index, -5000));
                    received_data.insert(received_data.end(), dataset.signal_values.begin(), dataset.signal_values.end());
                });
            }

            return received_data;
        });

        fair::graph::scheduler::simple sched{ std::move(flow_graph) };
        sched.work();

        sink.stop(); // TODO the scheduler should call this

        const auto received_data = poller_result.get();

        expect(eq(received_data, std::vector<int32_t>{ 8000, 185000 }));
        expect(eq(poller->drop_count.load(), 0));
    };

    "blocking polling multiplexed mode"_test = [] {
        const auto         tags      = make_test_tags(0, 10000);

        const std::int32_t n_samples = tags.size() * 10000 + 100000;
        graph              flow_graph;
        auto              &src = flow_graph.make_node<Source<int32_t>>({ { "n_samples_max", n_samples } });
        src.tags               = tags;
        auto &sink             = flow_graph.make_node<data_sink<int32_t>>();
        sink.set_name("test_sink");

        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink)));

        {
            const auto t = std::span(tags);

            // Test the test observer
            expect(eq(run_observer_test(t, Observer({}, -1, {})), "|###||###||###||###||###||###"s));
            expect(eq(run_observer_test(t, Observer(-1, {}, {})), "|######||######||######"s));
            expect(eq(run_observer_test(t, Observer(1, {}, {})), "|######|____________"s));
            expect(eq(run_observer_test(t, Observer(1, {}, 2)), "_|#|__|#|_____________"s));
            expect(eq(run_observer_test(t, Observer({}, {}, 1)), "|#|__|#|__|#|__|#|__|#|__|#|__"s));
        }

        const auto observers = std::array{ Observer({}, -1, {}), Observer(-1, {}, {}), Observer(1, {}, {}), Observer(1, {}, 2), Observer({}, {}, 1) };

        // Following the patterns above, where each #/_ is 10000 samples
        const auto expected = std::array<std::vector<int32_t>, observers.size()>{ { { 0, 29999, 30000, 59999, 60000, 89999, 90000, 119999, 120000, 149999, 150000, 249999 },
                                                                                    { 0, 59999, 60000, 119999, 120000, 219999 },
                                                                                    { 0, 59999 },
                                                                                    { 10000, 19999, 40000, 49999 },
                                                                                    { 0, 9999, 30000, 39999, 60000, 69999, 90000, 99999, 120000, 129999, 150000, 159999 } } };
        std::vector<std::shared_ptr<data_sink<int32_t>::dataset_poller>> pollers;

        for (const auto &o : observers) {
            auto poller = data_sink_registry::instance().get_multiplexed_poller<int32_t>(data_sink_query::sink_name("test_sink"), o, 100000, blocking_mode::Blocking);
            expect(neq(poller, nullptr));
            pollers.push_back(poller);
        }

        std::vector<std::future<std::vector<int32_t>>> results;

        for (std::size_t i = 0; i < pollers.size(); ++i) {
            auto f = std::async([poller = pollers[i]] {
                std::vector<int32_t> ranges;
                bool                 seen_finished = false;
                while (!seen_finished) {
                    seen_finished = poller->finished.load();
                    while (poller->process_one([&ranges](const auto &dataset) {
                        ranges.push_back(dataset.signal_values.front());
                        ranges.push_back(dataset.signal_values.back());
                    })) {
                    }
                }
                return ranges;
            });
            results.push_back(std::move(f));
        }

        fair::graph::scheduler::simple sched{ std::move(flow_graph) };
        sched.work();

        sink.stop(); // TODO the scheduler should call this

        for (std::size_t i = 0; i < results.size(); ++i) {
            expect(eq(results[i].get(), expected[i]));
        }
    };

    "blocking polling trigger mode overlapping"_test = [] {
        constexpr std::int32_t n_samples  = 150000;
        constexpr std::size_t  n_triggers = 300;

        graph                  flow_graph;
        auto                  &src = flow_graph.make_node<Source<float>>({ { "n_samples_max", n_samples } });

        for (std::size_t i = 0; i < n_triggers; ++i) {
            src.tags.push_back(tag_t{ static_cast<tag_t::signed_index_type>(60000 + i), { { "TYPE", "TRIGGER" } } });
        }

        auto &sink = flow_graph.make_node<data_sink<float>>();
        sink.set_name("test_sink");

        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink)));

        auto is_trigger = [](const tag_t &tag) { return trigger_test_result::Matching; };

        auto poller     = data_sink_registry::instance().get_trigger_poller<float>(data_sink_query::sink_name("test_sink"), is_trigger, 3000, 2000, blocking_mode::Blocking);
        expect(neq(poller, nullptr));

        auto                           polling = std::async([poller] {
            std::vector<float> received_data;
            std::vector<tag_t> received_tags;
            bool               seen_finished = false;
            while (!seen_finished) {
                // TODO make finished vs. pending data handling actually thread-safe
                seen_finished = poller->finished.load();
                while (poller->process_one([&received_data, &received_tags](const auto &dataset) {
                    expect(eq(dataset.signal_values.size(), 5000u) >> fatal);
                    received_data.push_back(dataset.signal_values.front());
                    received_data.push_back(dataset.signal_values.back());
                    expect(eq(dataset.timing_events.size(), 1u)) >> fatal;
                    expect(eq(dataset.timing_events[0].size(), 1u));
                    expect(eq(dataset.timing_events[0][0].index, 3000));
                    received_tags.insert(received_tags.end(), dataset.timing_events[0].begin(), dataset.timing_events[0].end());
                })) {
                }
            }
            return std::make_tuple(received_data, received_tags);
        });

        fair::graph::scheduler::simple sched{ std::move(flow_graph) };
        sched.work();

        sink.stop(); // TODO the scheduler should call this

        const auto &[received_data, received_tags] = polling.get();
        auto expected_start                        = std::vector<float>{ 57000, 61999, 57001, 62000, 57002 };
        expect(eq(poller->drop_count.load(), 0u));
        expect(eq(received_data.size(), 2 * n_triggers) >> fatal);
        expect(eq(std::vector(received_data.begin(), received_data.begin() + 5), expected_start));
        expect(eq(received_tags.size(), n_triggers));
    };

    "callback trigger mode overlapping"_test = [] {
        constexpr std::int32_t n_samples  = 150000;
        constexpr std::size_t  n_triggers = 300;

        graph                  flow_graph;
        auto                  &src = flow_graph.make_node<Source<float>>({ { "n_samples_max", n_samples } });

        for (std::size_t i = 0; i < n_triggers; ++i) {
            src.tags.push_back(tag_t{ static_cast<tag_t::signed_index_type>(60000 + i), { { "TYPE", "TRIGGER" } } });
        }

        auto &sink = flow_graph.make_node<data_sink<float>>();
        sink.set_name("test_sink");

        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink)));

        auto               is_trigger = [](const tag_t &) { return trigger_test_result::Matching; };

        std::mutex         m;
        std::vector<float> received_data;

        auto               callback = [&received_data, &m](auto &&dataset) {
            std::lock_guard lg{ m };
            expect(eq(dataset.signal_values.size(), 5000u));
            received_data.push_back(dataset.signal_values.front());
            received_data.push_back(dataset.signal_values.back());
        };

        data_sink_registry::instance().register_trigger_callback<float>(data_sink_query::sink_name("test_sink"), is_trigger, 3000, 2000, callback);

        fair::graph::scheduler::simple sched{ std::move(flow_graph) };
        sched.work();

        sink.stop(); // TODO the scheduler should call this

        std::lock_guard lg{ m };
        auto            expected_start = std::vector<float>{ 57000, 61999, 57001, 62000, 57002 };
        expect(eq(received_data.size(), 2 * n_triggers));
        expect(eq(std::vector(received_data.begin(), received_data.begin() + 5), expected_start));
    };

    "non-blocking polling continuous mode"_test = [] {
        graph flow_graph;
        auto &src  = flow_graph.make_node<Source<float>>({ { "n_samples_max", n_samples } });
        auto &sink = flow_graph.make_node<data_sink<float>>();
        sink.set_name("test_sink");

        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink)));

        auto invalid_type_poller = data_sink_registry::instance().get_streaming_poller<double>(data_sink_query::sink_name("test_sink"));
        expect(eq(invalid_type_poller, nullptr));

        auto poller = data_sink_registry::instance().get_streaming_poller<float>(data_sink_query::sink_name("test_sink"));
        expect(neq(poller, nullptr));

        auto                           polling = std::async([poller] {
            expect(neq(poller, nullptr));
            std::size_t samples_seen  = 0;
            bool        seen_finished = false;
            while (!seen_finished) {
                // TODO make finished vs. pending data handling actually thread-safe
                using namespace std::chrono_literals;
                std::this_thread::sleep_for(20ms);

                seen_finished = poller->finished.load();
                while (poller->process([&samples_seen](const auto &data) { samples_seen += data.size(); })) {
                }
            }

            return samples_seen;
        });

        fair::graph::scheduler::simple sched{ std::move(flow_graph) };
        sched.work();

        sink.stop(); // TODO the scheduler should call this

        const auto samples_seen = polling.get();
        expect(eq(samples_seen + poller->drop_count, n_samples));
    };
};

int
main() { /* tests are statically executed */
}
