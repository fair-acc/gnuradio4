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

} // namespace fair::graph::data_sink_test

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (fair::graph::data_sink_test::Source<T>), out, n_samples_produced, n_samples_max, n_tag_offset, sample_rate);

const boost::ut::suite DataSinkTests = [] {
    using namespace boost::ut;
    using namespace fair::graph;
    using namespace fair::graph::data_sink_test;

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
                    while (poller->process([&received](const auto &data) {
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

        std::mutex m;
        std::vector<float> received_data;

        auto polling = std::async([poller, &received_data, &m] {
            while (!poller->finished) {
                using namespace std::chrono_literals;
                [[maybe_unused]] auto r = poller->process([&received_data, &m](const auto &dataset) {
                    std::lock_guard lg{m};
                    received_data.insert(received_data.end(), dataset.signal_values.begin(), dataset.signal_values.end());
                });
            }
        });

        fair::graph::scheduler::simple sched{std::move(flow_graph)};
        sched.work();

        sink.stop(); // TODO the scheduler should call this

        polling.wait();

        std::lock_guard lg{m};
        expect(eq(sink.n_samples_consumed, n_samples));
        expect(eq(received_data.size(), 10));
        expect(eq(received_data, std::vector<float>{2997, 2998, 2999, 3000, 3001, 179997, 179998, 179999, 180000, 180001}));
        expect(eq(poller->drop_count.load(), 0));
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

        std::mutex m;
        std::vector<float> received_data;

        auto polling = std::async([poller, &received_data, &m] {
            bool seen_finished = false;
            while (!seen_finished) {
                // TODO make finished vs. pending data handling actually thread-safe
                seen_finished = poller->finished.load();
                while (poller->process([&received_data, &m](const auto &dataset) {
                    std::lock_guard lg{m};
                    expect(eq(dataset.signal_values.size(), 5000));
                    received_data.push_back(dataset.signal_values.front());
                    received_data.push_back(dataset.signal_values.back());
                })) {}
            }
        });

        fair::graph::scheduler::simple sched{std::move(flow_graph)};
        sched.work();

        sink.stop(); // TODO the scheduler should call this

        polling.wait();

        std::lock_guard lg{m};
        auto expected_start = std::vector<float>{57000, 61999, 57001, 62000, 57002};
        expect(eq(sink.n_samples_consumed, n_samples));
        expect(eq(received_data.size(), 2 * n_triggers));
        expect(eq(std::vector(received_data.begin(), received_data.begin() + 5), expected_start));
        expect(eq(poller->drop_count.load(), 0));
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
                while (poller->process([&samples_seen](const auto &data) {
                    samples_seen += data.size();
                })) {}
            }

            expect(eq(samples_seen + poller->drop_count.load(), n_samples));
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
