#include <boost/ut.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/SyncBlock.hpp>
#include <gnuradio-4.0/basic/clock_source.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#if defined(__clang__) && __clang_major__ >= 15
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

/*
 * This block is implemented for testing of the SyncBlock. It can simulate the following scenarios:
 * 1) Time Shift at Start
 * 2) Clock Drift
 * 3) Time-Out: no data
 * 4) Time-Out: no tag
 */
template<typename T>
struct TestSyncBlock : public gr::Block<TestSyncBlock<T>> {
    gr::PortIn<T>                          in; // The input is assumed to be ClockSource block
    std::vector<gr::PortOut<T, gr::Async>> outputs;

    int processBulkCounter = 0;

    gr::Annotated<gr::Size_t, "n_output_ports", gr::Visible, gr::Doc<"variable number of out ports">, gr::Limits<1U, 32U>> n_output_ports = 0U;

    void settingsChanged(const gr::property_map& old_settings, const gr::property_map& new_settings) {
        if (new_settings.contains("n_output_ports") && old_settings.at("n_output_ports") != new_settings.at("n_output_ports")) {
            if (std::any_of(outputs.begin(), outputs.end(), [](const auto& port) { return port.isConnected(); })) {
                throw std::range_error("Number of output ports cannot be changed after Graph initialization.");
            }
        }
        fmt::print("{}: configuration changed: n_output_ports {} -> {}\n", this->name, old_settings.at("n_output_ports"), new_settings.at("n_output_ports"));
        outputs.resize(n_output_ports);
    }

    template<gr::PublishableSpan TOutput>
    gr::work::Status processBulk(gr::ConsumableSpan auto& inSpan, std::span<TOutput>& outSpans) noexcept {
        fmt::println("TestSyncBlock::processBulk inSpan.size:{}, processBulkCounter:{}", inSpan.size(), processBulkCounter++);
        inSpan.consume(inSpan.size());
        for (std::size_t i = 0; i < outSpans.size(); i++) {
            outSpans[i].publish(inSpan.size());
        }
        return gr::work::Status::OK;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (TestSyncBlock<T>), in, outputs, n_output_ports);

const boost::ut::suite SyncBlockTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::basic;
    using namespace gr::testing;

    "SyncBlock_test"_test = [] {
        gr::Graph graph;

        gr::Size_t nPorts     = 2U;
        gr::Size_t nSamples   = 1000U;
        float      sampleRate = 1000.f;

        auto&                                                            clockSrc      = graph.emplaceBlock<gr::basic::ClockSource<double>>({{"n_samples_max", nSamples}, {"sample_rate", sampleRate}, {"chunk_size", gr::Size_t(1)}});
        auto&                                                            testSyncBlock = graph.emplaceBlock<TestSyncBlock<double>>({{"n_output_ports", nPorts}});
        auto&                                                            syncBlock     = graph.emplaceBlock<gr::basic::SyncBlock<double>>({{"n_ports", nPorts}});
        std::vector<TagSink<double, ProcessFunction::USE_PROCESS_BULK>*> sinks;

        expect(gr::ConnectionResult::SUCCESS == graph.connect<"out">(clockSrc).to<"in">(testSyncBlock));
        for (gr::Size_t i = 0; i < nPorts; i++) {
            expect(gr::ConnectionResult::SUCCESS == graph.connect(testSyncBlock, {"outputs", i}, syncBlock, {"inputs", i}));
        }

        for (gr::Size_t i = 0; i < nPorts; i++) {
            sinks.push_back(std::addressof(graph.emplaceBlock<TagSink<double, ProcessFunction::USE_PROCESS_BULK>>()));
            expect(sinks[i]->settings().applyStagedParameters().forwardParameters.empty());
            expect(gr::ConnectionResult::SUCCESS == graph.connect(syncBlock, {"outputs", i}, *sinks[i], {"in"}));
        }

        gr::scheduler::Simple sched{std::move(graph)};
        sched.runAndWait();

        for (gr::Size_t i = 0; i < nPorts; i++) {
            expect(eq(sinks[i]->n_samples_produced, nSamples));
        }
    };
};

int main() {}