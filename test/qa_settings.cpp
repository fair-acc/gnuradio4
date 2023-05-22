#include <boost/ut.hpp>

#include <node.hpp>
#include <graph.hpp>
#include <reflection.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template <>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

namespace fair::graph::setting_test {

template<typename T>
struct Source : public node<Source<T>> {
    OUT<T>       out;
    std::int32_t n_samples_produced = 0;
    std::int32_t n_samples_max      = 1024;
    std::int32_t n_tag_offset       = 0;
    float        sample_rate        = 1000.0f;

    void
    init(const tag_t::map_type &old_settings, const tag_t::map_type &new_settings) {
        // optional init function that is called after construction and whenever settings change
        fair::graph::publish_tag(out, { { "n_samples_max", n_samples_max } }, n_tag_offset);
    }

    constexpr std::int64_t
    available_samples(const Source &self) noexcept {
        const auto ret = static_cast<std::int64_t>(n_samples_max - n_samples_produced);
        return ret >= 0 ? ret : -1; // '-1' -> DONE, produced enough samples
    }

    [[nodiscard]] constexpr T
    process_one() noexcept {
        n_samples_produced++;
        T x{};
        return x;
    }
};

template<typename T>
struct TestBlock : public node<TestBlock<T>> {
    IN<T>       in;
    OUT<T>      out;
    T           scaling_factor = static_cast<T>(1);
    std::string  context;
    std::int32_t n_samples_max = -1;
    float        sample_rate        = 1000.0f;

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V
    process_one(const V &a) const noexcept {
        return a * scaling_factor;
    }
};

template<typename T>
struct Sink : public node<Sink<T>> {
    IN<T> in;
    std::int32_t n_samples_consumed = 0;
    std::int32_t n_samples_max      = -1;
    int64_t      last_tag_position  = -1;
    float        sample_rate        = -1.0f;

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(V) noexcept {
        // alt: optional user-level tag processing
        /*
        if (this->input_tags_present()) {
            if (this->input_tags_present() && this->input_tags()[0].contains("n_samples_max")) {
                const auto value = this->input_tags()[0].at("n_samples_max");
                assert(std::holds_alternative<std::int32_t>(value));
                n_samples_max = std::get<std::int32_t>(value);
                last_tag_position        = in.streamReader().position();
                this->acknowledge_input_tags(); // clears further tag notifications
            }
        }
        */

        if constexpr (fair::meta::any_simd<V>) {
            n_samples_consumed += V::size();
        } else {
            n_samples_consumed++;
        }
    }
};
} // namespace fair::graph::setting_test

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (fair::graph::setting_test::Source<T>), out, n_samples_produced, n_samples_max, n_tag_offset, sample_rate);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (fair::graph::setting_test::TestBlock<T>), in, out, scaling_factor, context, n_samples_max, sample_rate);
ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (fair::graph::setting_test::Sink<T>), in, n_samples_consumed, n_samples_max, last_tag_position, sample_rate);

const boost::ut::suite SettingsTests = [] {
    using namespace boost::ut;
    using namespace fair::graph;
    using namespace fair::graph::setting_test;

    "basic node settings tag"_test = [] {
        graph                  flow_graph;
        constexpr std::int32_t n_samples = gr::util::round_up(1'000'000, 1024);
        // define basic Sink->TestBlock->Sink flow graph
        auto &src = flow_graph.make_node<Source<float>>({ { "n_samples_max", n_samples } });
        expect(eq(src.n_samples_max, n_samples)) << "check map constructor";
        expect(eq(src.settings().auto_update_parameters().size(), 3UL));
        expect(eq(src.settings().auto_forward_parameters().size(), 1UL)); // sample_rate
        auto &block1 = flow_graph.make_node<TestBlock<float>>();
        auto &block2 = flow_graph.make_node<TestBlock<float>>();
        auto &sink  = flow_graph.make_node<Sink<float>>();
        expect(eq(sink.settings().auto_update_parameters().size(), 4UL));
        expect(eq(sink.settings().auto_forward_parameters().size(), 1UL)); // sample_rate

        block1.context = "Test Context";
        block1.settings().update_active_parameters();
        expect(eq(block1.settings().auto_update_parameters().size(), 4UL));
        expect(eq(block1.settings().auto_forward_parameters().size(), 2UL));

        expect(block1.settings().get("context").has_value());
        expect(block1.settings().get({ "context" }).has_value());
        expect(not block1.settings().get({ "test" }).has_value());

        std::vector<std::string>   keys1{ "key1", "key2", "key3" };
        std::span<std::string>     keys2{ keys1 };
        std::array<std::string, 3> keys3{ "key1", "key2", "key3" };
        expect(block1.settings().get(keys1).empty());
        expect(block1.settings().get(keys2).empty());
        expect(block1.settings().get(keys3).empty());
        expect(eq(block1.settings().get().size(), 4UL));

        // set non-existent setting
        expect(not block1.settings().changed()) << "settings not changed";
        auto ret1 = block1.settings().set({ { "unknown", "random value" } });
        expect(eq(ret1.size(), 1)) << "setting one unknown parameter";

        expect(not block1.settings().changed());
        auto ret2 = block1.settings().set({ { "context", "alt context" } });
        expect(not block1.settings().staged_parameters().empty());
        expect(ret2.empty()) << "setting one known parameter";
        expect(block1.settings().changed()) << "settings changed";
        auto forwarding_parameter = block1.settings().apply_staged_parameters();
        expect(eq(forwarding_parameter.size(), 1)) << "initial forward declarations";
        block1.settings().update_active_parameters();

        // src -> block1 -> block2 -> sink
        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(src).to<"in">(block1)));
        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(block1).to<"in">(block2)));
        expect(eq(connection_result_t::SUCCESS, flow_graph.connect<"out">(block2).to<"in">(sink)));

        auto token = flow_graph.init();
        expect(token);
        expect(src.settings().auto_update_parameters().contains("sample_rate"));
        [[maybe_unused]] auto ret = src.settings().set({ { "sample_rate", 49000.0f } });
        flow_graph.work(token);
        expect(eq(src.n_samples_produced, n_samples)) << "did not produce enough output samples";
        expect(eq(sink.n_samples_consumed, n_samples)) << "did not consume enough input samples";

        expect(eq(src.n_samples_max, n_samples)) << "receive tag announcing max samples";
        expect(eq(block1.n_samples_max, n_samples)) << "receive tag announcing max samples";
        expect(eq(sink.n_samples_max, n_samples)) << "receive tag announcing max samples";

        // expect(eq(src.sample_rate, 49000.0f)) << "src matching sample_rate";
        expect(eq(block1.sample_rate, 49000.0f)) << "block1 matching sample_rate";
        expect(eq(block2.sample_rate, 49000.0f)) << "block2 matching sample_rate";
        expect(eq(sink.sample_rate, 49000.0f)) << "sink matching src sample_rate";

        // check auto-update flags
        expect(!src.settings().auto_update_parameters().contains("sample_rate")) << "src should not retain auto-update flag (was manually set)";
        expect(block1.settings().auto_update_parameters().contains("sample_rate")) << "block1 retained auto-update flag";
        expect(block2.settings().auto_update_parameters().contains("sample_rate")) << "block1 retained auto-update flag";
        expect(sink.settings().auto_update_parameters().contains("sample_rate")) << "sink retained auto-update flag";
    };
};

int
main() { /* tests are statically executed */
}
