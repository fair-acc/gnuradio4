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

namespace fair::graph::data_sink_test {

template<typename T>
struct Source : public node<Source<T>> {
    OUT<T>       out;
    std::int32_t n_samples_produced = 0;
    std::int32_t n_samples_max      = 1024;
    std::int32_t n_tag_offset       = 0;
    float        sample_rate        = 1000.0f;
    T            next_value         = {};

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
        n_samples_produced++;
        return next_value++;
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
        constexpr std::int32_t n_samples = gr::util::round_up(1'000'000, 1024);

        auto &src = flow_graph.make_node<Source<float>>({ { "n_samples_max", n_samples } });
        auto &sink = flow_graph.make_node<data_sink<float>>();
        sink.set_name("test_sink");

        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink)));

        std::size_t samples_seen = 0;
        auto callback = [&samples_seen](std::span<const float> buffer) {
            for (std::size_t i = 0; i < buffer.size(); ++i) {
                expect(eq(buffer[i], static_cast<float>(samples_seen + i)));
            }
            samples_seen += buffer.size();
        };

        expect(data_sink_registry::instance().register_streaming_callback<float>("test_sink", callback));

        fair::graph::scheduler::simple sched{std::move(flow_graph)};
        sched.work();

        expect(eq(sink.n_samples_consumed, n_samples));
        expect(eq(samples_seen, n_samples));
    };

    "blocking polling continuous mode"_test = [] {
        constexpr std::int32_t n_samples = gr::util::round_up(1'000'000, 1024);

        graph flow_graph;
        auto &src = flow_graph.make_node<Source<float>>({ { "n_samples_max", n_samples } });
        auto &sink = flow_graph.make_node<data_sink<float>>();
        sink.set_name("test_sink");

        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink)));

        std::atomic<std::size_t> samples_seen = 0;

        auto poller = data_sink_registry::instance().get_streaming_poller<float>("test_sink", blocking_mode::Blocking);
        expect(neq(poller, nullptr));

        auto polling = std::async([poller, &samples_seen] {
            while (!poller->finished) {
                [[maybe_unused]] auto r = poller->process([&samples_seen](const auto &data) {
                    samples_seen += data.size();
                });
            }
        });

        fair::graph::scheduler::simple sched{std::move(flow_graph)};
        sched.work();

        poller->finished = true; // TODO this should be done by the block

        polling.wait();

        expect(eq(sink.n_samples_consumed, n_samples));
        expect(eq(samples_seen.load(), n_samples));
        expect(eq(poller->drop_count.load(), 0));
    };

    "non-blocking polling continuous mode"_test = [] {
        constexpr std::int32_t n_samples = gr::util::round_up(1'000'000, 1024);

        graph flow_graph;
        auto &src = flow_graph.make_node<Source<float>>({ { "n_samples_max", n_samples } });
        auto &sink = flow_graph.make_node<data_sink<float>>();
        sink.set_name("test_sink");

        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(sink)));

        std::atomic<std::size_t> samples_seen = 0;

        auto invalid_type_poller = data_sink_registry::instance().get_streaming_poller<double>("test_sink");
        expect(eq(invalid_type_poller, nullptr));

        auto poller = data_sink_registry::instance().get_streaming_poller<float>("test_sink");
        expect(neq(poller, nullptr));

        auto polling = std::async([poller, &samples_seen] {
            expect(neq(poller, nullptr));
            while (!poller->finished) {
                using namespace std::chrono_literals;
                std::this_thread::sleep_for(20ms);
                [[maybe_unused]] auto r = poller->process([&samples_seen](const auto &data) {
                    samples_seen += data.size();
                });
            }
        });

        fair::graph::scheduler::simple sched{std::move(flow_graph)};
        sched.work();

        poller->finished = true; // TODO this should be done by the block

        polling.wait();

        expect(eq(sink.n_samples_consumed, n_samples));
        expect(lt(samples_seen.load(), n_samples));
        expect(gt(poller->drop_count.load(), 0));
    };
};

int
main() { /* tests are statically executed */
}
