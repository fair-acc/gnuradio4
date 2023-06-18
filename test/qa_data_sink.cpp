#include <boost/ut.hpp>

#include <buffer.hpp>
#include <data_sink.hpp>
#include <graph.hpp>
#include <node.hpp>
#include <reflection.hpp>
#include <scheduler.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <deque>
#include <future>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

namespace fair::graph::data_sink_test {

static constexpr std::int32_t n_samples = 200000;

template<typename T>
struct Source : public node<Source<T>> {
    OUT<T>       out;
    std::int32_t n_samples_produced = 0;
    std::int32_t n_samples_max      = 1024;
    std::int32_t n_tag_offset       = 0;
    float        sample_rate        = 1000.0f;
    T            next_value         = {};
    std::deque<tag_t> tags; // must be sorted by index

    void
    init(const tag_t::map_type &old_settings, const tag_t::map_type &new_settings) {
        // optional init function that is called after construction and whenever settings change
        fair::graph::publish_tag(out, { { "n_samples_max", n_samples_max } }, n_tag_offset);
    }

    constexpr std::int64_t
    available_samples(const Source &self) noexcept {
        const auto ret = static_cast<std::int64_t>(n_samples_max - n_samples_produced);
        return ret > 0 ? ret : -1; // '-1' -> DONE, produced enough samples
    }

    [[nodiscard]] constexpr T
    process_one() noexcept {
        while (!tags.empty() && tags[0].index == n_samples_produced) {
            // TODO there probably is, or should be, an easier way to do this
            const auto pos = output_port<"out">(this).streamWriter().position();
            publish_tag(out, tags[0].map, n_samples_produced - pos);
            tags.pop_front();
        }

        n_samples_produced++;
        const auto v = next_value;
        next_value++;
        return v;
    }
};

struct Observer {
    std::optional<int> year;
    std::optional<int> month;
    std::optional<int> day;
    std::optional<std::tuple<int, int, int>> last_seen;
    bool last_matched = false;

    explicit Observer(std::optional<int> y, std::optional<int> m, std::optional<int> d) : year(y), month(m), day(d) {}

    static inline bool same(int x, std::optional<int> other) {
        return other && x == *other;
    }
    static inline bool changed(int x, std::optional<int> other) {
        return !same(x, other);
    }

    trigger_observer_state operator()(const tag_t &tag) {
        const auto ty = tag.get("Y");
        const auto tm = tag.get("M");
        const auto td = tag.get("D");
        if (!ty || !tm || !td) {
            return trigger_observer_state::Ignore;
        }

        const auto tup = std::make_tuple(std::get<int>(ty->get()), std::get<int>(tm->get()), std::get<int>(td->get()));
        const auto &[y, m, d] = tup;
        const auto ly = last_seen ? std::optional<int>(std::get<0>(*last_seen)) : std::nullopt;
        const auto lm = last_seen ? std::optional<int>(std::get<1>(*last_seen)) : std::nullopt;
        const auto ld = last_seen ? std::optional<int>(std::get<2>(*last_seen)) : std::nullopt;

        const auto year_restart = year && *year == -1 && changed(y, ly);
        const auto year_matches = !year || *year == -1 || same(y, year);
        const auto month_restart = month && *month == -1 && changed(m, lm);
        const auto month_matches = !month || *month == -1 || same(m, month);
        const auto day_restart = day && *day == -1 && changed(d, ld);
        const auto day_matches = !day || *day == -1 || same(d, day);
        const auto matches = year_matches && month_matches && day_matches;
        const auto restart = year_restart || month_restart || day_restart;

        trigger_observer_state r = trigger_observer_state::Ignore;

        if (last_matched && !matches) {
            r = trigger_observer_state::Stop;
        } else if (!last_matched && matches) {
            r = trigger_observer_state::Start;
        } else if ((!last_seen || last_matched) && matches && restart) {
            r = trigger_observer_state::StopAndStart;
        }

        last_seen = tup;
        last_matched = matches;
        return r;
    }
};

static tag_t make_tag(tag_t::index_type index, int y, int m, int d) {
    tag_t::map_type map;
    return tag_t{index, {{"Y", y}, {"M", m}, {"D", d}}};
}

static std::vector<tag_t> make_test_tags(tag_t::index_type first_index, tag_t::index_type interval) {
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

static std::string to_ascii_art(std::span<trigger_observer_state> states) {
    bool started = false;
    std::string r;
    for (auto s : states) {
        switch (s) {
        case trigger_observer_state::Start:
            r += started ? "E" : "|#";
            started = true;
            break;
        case trigger_observer_state::Stop:
            r += started ? "|_" : "E";
            started = false;
            break;
        case trigger_observer_state::StopAndStart:
            r += started ? "||#" : "|#";
            started = true;
            break;
        case trigger_observer_state::Ignore:
            r += started ? "#" : "_";
            break;
        }
    };
    return r;
}

template<TriggerObserver O>
std::string run_observer_test(std::span<const tag_t> tags, O o) {
   std::vector<trigger_observer_state> r;
    r.reserve(tags.size());
    for (const auto &tag : tags) {
        r.push_back(o(tag));
    }
    return to_ascii_art(r);
}

} // namespace fair::graph::data_sink_test

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (fair::graph::data_sink_test::Source<T>), out, n_samples_produced, n_samples_max, n_tag_offset, sample_rate);

const boost::ut::suite DataSinkTests = [] {
    using namespace boost::ut;
    using namespace fair::graph;
    using namespace fair::graph::data_sink_test;
    using namespace std::string_literals;

    "callback continuous mode"_test = [] {
        graph                  flow_graph;

        static constexpr std::int32_t n_samples = 200005;
        static constexpr std::size_t chunk_size = 1000;

        auto &src = flow_graph.make_node<Source<float>>({ { "n_samples_max", n_samples } });
        auto &sink = flow_graph.make_node<data_sink<float>>();
        sink.set_name("test_sink");

        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink)));

        std::size_t samples_seen = 0;
        std::size_t chunks_seen = 0;
        auto callback = [&samples_seen, &chunks_seen](std::span<const float> buffer) {
            for (std::size_t i = 0; i < buffer.size(); ++i) {
                expect(eq(buffer[i], static_cast<float>(samples_seen + i)));
            }

            samples_seen += buffer.size();
            chunks_seen++;
            if (chunks_seen < 201) {
                expect(eq(buffer.size(), chunk_size));
            } else {
                expect(eq(buffer.size(), 5));
            }
        };

        expect(data_sink_registry::instance().register_streaming_callback<float>("test_sink", chunk_size, callback));

        fair::graph::scheduler::simple sched{std::move(flow_graph)};
        sched.work();

        sink.stop(); // TODO the scheduler should call this

        expect(eq(chunks_seen, 201));
        expect(eq(sink.n_samples_consumed, n_samples));
        expect(eq(samples_seen, n_samples));
    };

    "blocking polling continuous mode"_test = [] {

        constexpr std::int32_t n_samples = 200000;

        graph flow_graph;
        auto &src = flow_graph.make_node<Source<float>>({ { "n_samples_max", n_samples } });
        auto &sink = flow_graph.make_node<data_sink<float>>();
        sink.set_name("test_sink");

        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink)));

        std::atomic<std::size_t> samples_seen = 0;

        auto poller1 = data_sink_registry::instance().get_streaming_poller<float>("test_sink", blocking_mode::Blocking);
        expect(neq(poller1, nullptr));

        auto poller2 = data_sink_registry::instance().get_streaming_poller<float>("test_sink", blocking_mode::Blocking);
        expect(neq(poller2, nullptr));

        auto make_runner = [](auto poller) {
            return std::async([poller] {
                std::vector<float> received;
                bool seen_finished = false;
                while (!seen_finished) {
                    // TODO make finished vs. pending data handling actually thread-safe
                    seen_finished = poller->finished.load();
                    while (poller->process_bulk([&received](const auto &data) {
                        received.insert(received.end(), data.begin(), data.end());
                    })) {}
                }

                std::vector<float> expected(n_samples);
                std::iota(expected.begin(), expected.end(), 0.0);
                expect(eq(received.size(), expected.size()));
                expect(eq(received, expected));
                expect(eq(poller->drop_count.load(), 0));
            });
        };

        auto runner1 = make_runner(poller1);
        auto runner2 = make_runner(poller2);

        fair::graph::scheduler::simple sched{std::move(flow_graph)};
        sched.work();

        sink.stop(); // TODO the scheduler should call this

        runner1.wait();
        runner2.wait();

        expect(eq(sink.n_samples_consumed, n_samples));
    };

    "blocking polling trigger mode non-overlapping"_test = [] {
        constexpr std::int32_t n_samples = 200000;

        graph flow_graph;
        auto &src = flow_graph.make_node<Source<float>>({ { "n_samples_max", n_samples } });
        src.tags = {{3000, {{"TYPE", "TRIGGER"}}}, tag_t{8000, {{"TYPE", "NO_TRIGGER"}}}, {180000, {{"TYPE", "TRIGGER"}}}};
        auto &sink = flow_graph.make_node<data_sink<float>>();
        sink.set_name("test_sink");

        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink)));

        auto is_trigger = [](const tag_t &tag) {
            const auto v = tag.get("TYPE");
            return v && std::get<std::string>(v->get()) == "TRIGGER";
        };

        auto poller = data_sink_registry::instance().get_trigger_poller<float>("test_sink", is_trigger, 3, 2, blocking_mode::Blocking);
        expect(neq(poller, nullptr));

        auto polling = std::async([poller] {
            std::vector<float> received_data;
            bool seen_finished = false;
            while (!seen_finished) {
                seen_finished = poller->finished;
                [[maybe_unused]] auto r = poller->process_one([&received_data](const auto &dataset) {
                    received_data.insert(received_data.end(), dataset.signal_values.begin(), dataset.signal_values.end());
                });
            }
            return received_data;
        });

        fair::graph::scheduler::simple sched{std::move(flow_graph)};
        sched.work();

        sink.stop(); // TODO the scheduler should call this

        const auto received_data = polling.get();

        expect(eq(sink.n_samples_consumed, n_samples));
        expect(eq(received_data.size(), 10));
        expect(eq(received_data, std::vector<float>{2997, 2998, 2999, 3000, 3001, 179997, 179998, 179999, 180000, 180001}));
        expect(eq(poller->drop_count.load(), 0));
    };

    "blocking polling snapshot mode"_test = [] {
        constexpr std::int32_t n_samples = 200000;

        graph flow_graph;
        auto &src = flow_graph.make_node<Source<int32_t>>({ { "n_samples_max", n_samples } });
        src.tags = {{3000, {{"TYPE", "TRIGGER"}}}, tag_t{8000, {{"TYPE", "NO_TRIGGER"}}}, {180000, {{"TYPE", "TRIGGER"}}}};
        auto &sink = flow_graph.make_node<data_sink<int32_t>>();
        sink.set_name("test_sink");

        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink)));

        auto is_trigger = [](const tag_t &tag) {
            const auto v = tag.get("TYPE");
            return v && std::get<std::string>(v->get()) == "TRIGGER";
        };

        const auto delay = std::chrono::milliseconds{500}; // sample rate 10000 -> 5000 samples
        auto poller = data_sink_registry::instance().get_snapshot_poller<int32_t>("test_sink", is_trigger, delay, blocking_mode::Blocking);
        expect(neq(poller, nullptr));

        auto poller_result = std::async([poller] {
            std::vector<int32_t> received_data;

            bool seen_finished = false;
            while (!seen_finished) {
                seen_finished = poller->finished;
                [[maybe_unused]] auto r = poller->process_one([&received_data](const auto &dataset) {
                    received_data.insert(received_data.end(), dataset.signal_values.begin(), dataset.signal_values.end());
                });
            }

            return received_data;
        });

        fair::graph::scheduler::simple sched{std::move(flow_graph)};
        sched.work();

        sink.stop(); // TODO the scheduler should call this

        const auto received_data = poller_result.get();

        expect(eq(sink.n_samples_consumed, n_samples));
        expect(eq(received_data, std::vector<int32_t>{8000, 185000}));
        expect(eq(poller->drop_count.load(), 0));
    };

    "blocking polling multiplexed mode"_test = [] {
        const auto tags = make_test_tags(0, 10000);

        const std::int32_t n_samples = tags.size() * 10000 + 100000;
        graph flow_graph;
        auto &src = flow_graph.make_node<Source<int32_t>>({ { "n_samples_max", n_samples } });
        src.tags = std::deque<tag_t>(tags.begin(), tags.end());
        auto &sink = flow_graph.make_node<data_sink<int32_t>>();
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

        auto observer_factory = [](std::optional<int> y, std::optional<int> m, std::optional<int> d) {
            return [y, m, d]() {
                return Observer(y, m, d);
            };
        };
        const auto factories = std::array{observer_factory({}, -1, {}),
                                          observer_factory(-1, {}, {}),
                                          observer_factory(1, {}, {}),
                                          observer_factory(1, {}, 2),
                                          observer_factory({}, {}, 1)};

        // Following the patterns above, where each #/_ is 10000 samples
        const auto expected = std::array<std::vector<int32_t>, factories.size()>{{
            {0, 29999, 30000, 59999, 60000, 89999, 90000, 119999, 120000, 149999, 150000, 249999},
            {0, 59999, 60000, 119999, 120000, 219999},
            {0, 59999},
            {10000, 19999, 40000, 49999},
            {0, 9999, 30000, 39999, 60000, 69999, 90000, 99999, 120000, 129999, 150000, 159999}
        }};
        std::vector<std::shared_ptr<data_sink<int32_t>::dataset_poller>> pollers;

        for (const auto &f : factories) {
            auto poller = data_sink_registry::instance().get_multiplexed_poller<int32_t>("test_sink", f, 100000, blocking_mode::Blocking);
            expect(neq(poller, nullptr));
            pollers.push_back(poller);
        }

        std::vector<std::future<std::vector<int32_t>>> results;

        for (std::size_t i = 0; i < pollers.size(); ++i) {
            auto f = std::async([poller = pollers[i]] {
                std::vector<int32_t> ranges;
                bool seen_finished = false;
                while (!seen_finished) {
                    seen_finished = poller->finished.load();
                    while (poller->process_one([&ranges](const auto &dataset) {
                        ranges.push_back(dataset.signal_values.front());
                        ranges.push_back(dataset.signal_values.back());
                    })) {}
                }
                return ranges;
            });
            results.push_back(std::move(f));
        }

        fair::graph::scheduler::simple sched{std::move(flow_graph)};
        sched.work();

        sink.stop(); // TODO the scheduler should call this

        for (std::size_t i = 0; i < results.size(); ++i) {
            expect(eq(results[i].get(), expected[i]));
        }
        expect(eq(sink.n_samples_consumed, n_samples));
    };

    "blocking polling trigger mode overlapping"_test = [] {
        constexpr std::int32_t n_samples = 2000000;
        constexpr std::size_t n_triggers = 5000;

        graph flow_graph;
        auto &src = flow_graph.make_node<Source<float>>({ { "n_samples_max", n_samples } });

        for (std::size_t i = 0; i < n_triggers; ++i) {
            src.tags.push_back(tag_t{static_cast<tag_t::index_type>(60000 + i), {{"TYPE", "TRIGGER"}}});
        }

        auto &sink = flow_graph.make_node<data_sink<float>>();
        sink.set_name("test_sink");

        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink)));

        auto is_trigger = [](const tag_t &tag) {
            return true;
        };

        auto poller = data_sink_registry::instance().get_trigger_poller<float>("test_sink", is_trigger, 3000, 2000, blocking_mode::Blocking);
        expect(neq(poller, nullptr));

        auto polling = std::async([poller] {
            std::vector<float> received_data;
            bool seen_finished = false;
            while (!seen_finished) {
                // TODO make finished vs. pending data handling actually thread-safe
                seen_finished = poller->finished.load();
                while (poller->process_one([&received_data](const auto &dataset) {
                    expect(eq(dataset.signal_values.size(), 5000));
                    received_data.push_back(dataset.signal_values.front());
                    received_data.push_back(dataset.signal_values.back());
                })) {}
            }
            return received_data;
        });

        fair::graph::scheduler::simple sched{std::move(flow_graph)};
        sched.work();

        sink.stop(); // TODO the scheduler should call this

        const auto received_data = polling.get();
        auto expected_start = std::vector<float>{57000, 61999, 57001, 62000, 57002};
        expect(eq(sink.n_samples_consumed, n_samples));
        expect(eq(received_data.size(), 2 * n_triggers));
        expect(eq(std::vector(received_data.begin(), received_data.begin() + 5), expected_start));
        expect(eq(poller->drop_count.load(), 0));
    };

    "callback trigger mode overlapping"_test = [] {
        constexpr std::int32_t n_samples = 2000000;
        constexpr std::size_t n_triggers = 5000;

        graph flow_graph;
        auto &src = flow_graph.make_node<Source<float>>({ { "n_samples_max", n_samples } });

        for (std::size_t i = 0; i < n_triggers; ++i) {
            src.tags.push_back(tag_t{static_cast<tag_t::index_type>(60000 + i), {{"TYPE", "TRIGGER"}}});
        }

        auto &sink = flow_graph.make_node<data_sink<float>>();
        sink.set_name("test_sink");

        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink)));

        auto is_trigger = [](const tag_t &tag) {
            return true;
        };

        std::mutex m;
        std::vector<float> received_data;

        auto callback = [&received_data, &m](auto &&dataset) {
            std::lock_guard lg{m};
            expect(eq(dataset.signal_values.size(), 5000));
            received_data.push_back(dataset.signal_values.front());
            received_data.push_back(dataset.signal_values.back());
        };

        data_sink_registry::instance().register_trigger_callback<float>("test_sink", is_trigger, 3000, 2000, callback);

        fair::graph::scheduler::simple sched{std::move(flow_graph)};
        sched.work();

        sink.stop(); // TODO the scheduler should call this

        std::lock_guard lg{m};
        auto expected_start = std::vector<float>{57000, 61999, 57001, 62000, 57002};
        expect(eq(sink.n_samples_consumed, n_samples));
        expect(eq(received_data.size(), 2 * n_triggers));
        expect(eq(std::vector(received_data.begin(), received_data.begin() + 5), expected_start));
    };

    "non-blocking polling continuous mode"_test = [] {
        graph flow_graph;
        auto &src = flow_graph.make_node<Source<float>>({ { "n_samples_max", n_samples } });
        auto &sink = flow_graph.make_node<data_sink<float>>();
        sink.set_name("test_sink");

        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink)));

        auto invalid_type_poller = data_sink_registry::instance().get_streaming_poller<double>("test_sink");
        expect(eq(invalid_type_poller, nullptr));

        auto poller = data_sink_registry::instance().get_streaming_poller<float>("test_sink");
        expect(neq(poller, nullptr));

        auto polling = std::async([poller] {
            expect(neq(poller, nullptr));
            std::size_t samples_seen = 0;
            bool seen_finished = false;
            while (!seen_finished) {
                // TODO make finished vs. pending data handling actually thread-safe
                using namespace std::chrono_literals;
                std::this_thread::sleep_for(20ms);

                seen_finished = poller->finished.load();
                while (poller->process_bulk([&samples_seen](const auto &data) {
                    samples_seen += data.size();
                })) {}
            }

            expect(eq(samples_seen + poller->drop_count, n_samples));
        });

        fair::graph::scheduler::simple sched{std::move(flow_graph)};
        sched.work();

        sink.stop(); // TODO the scheduler should call this

        polling.wait();

        expect(eq(sink.n_samples_consumed, n_samples));
    };
};

int
main() { /* tests are statically executed */
}
