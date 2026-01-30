#include <benchmark.hpp>

#include <functional>

#include <gnuradio-4.0/BlockTraits.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/Selector.hpp>
#include <gnuradio-4.0/basic/SyncBlock.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

inline constexpr std::size_t nRepeats = 1; // must be 1 at the moment
inline constexpr gr::Size_t  nPorts   = 2U;
inline constexpr gr::Size_t  nSamples = 4'000'000'000;

gr::Tag genSyncTag(std::size_t index, std::uint64_t triggerTime, std::string triggerName = "TriggerName") { //
    return {index, {{gr::tag::TRIGGER_NAME.shortKey(), triggerName}, {gr::tag::TRIGGER_TIME.shortKey(), triggerTime}}};
};

template<typename TBlock>
void runTest() {
    using namespace boost::ut;
    using namespace benchmark;
    using namespace gr;
    using namespace gr::basic;
    using namespace gr::testing;

    gr::Graph graph;

    const std::size_t nTags = 1; // we need at least 1 tag to synchronize samples

    property_map perfBlockProperties;
    if constexpr (std::is_same_v<TBlock, gr::basic::SyncBlock<int>>) {
        perfBlockProperties = {{"n_ports", nPorts}};
    } else if constexpr (std::is_same_v<TBlock, gr::basic::Selector<int>>) {
        auto               iotaView = std::views::iota(gr::Size_t(0), nPorts);
        Tensor<gr::Size_t> indMap(iotaView.begin(), iotaView.end());
        perfBlockProperties = {{"n_inputs", nPorts}, {"n_outputs", nPorts}, {"map_in", indMap}, {"map_out", indMap}};
    } else {
        throw std::invalid_argument("incorrect TBlock type");
    }

    auto& perfBlock = graph.emplaceBlock<TBlock>(perfBlockProperties);

    std::vector<TagSource<int, ProcessFunction::USE_PROCESS_BULK>*> sources;
    std::vector<TagSink<int, ProcessFunction::USE_PROCESS_BULK>*>   sinks;

    for (std::size_t i = 0; i < nPorts; i++) {
        property_map srcParams = {{"n_samples_max", nSamples}, {"verbose_console", false}, {"disconnect_on_done", false}};
        sources.push_back(std::addressof(graph.emplaceBlock<TagSource<int, ProcessFunction::USE_PROCESS_BULK>>(srcParams)));
        for (std::size_t iT = 0; iT < nTags; iT++) {
            const std::size_t   index = iT * nSamples / nTags + 1;
            const std::uint64_t time  = iT * 100;
            sources[i]->_tags.push_back(genSyncTag(index, time));
        }
        expect(gr::ConnectionResult::SUCCESS == graph.connect(*sources[i], "out"s, perfBlock, "inputs#"s + std::to_string(i)));
    }

    for (std::size_t i = 0; i < nPorts; i++) {
        property_map sinkParams = {{"log_samples", false}, {"log_tags", false}, {"verbose_console", false}, {"disconnect_on_done", false}};
        sinks.push_back(std::addressof(graph.emplaceBlock<TagSink<int, ProcessFunction::USE_PROCESS_BULK>>(sinkParams)));
        expect(gr::ConnectionResult::SUCCESS == graph.connect(perfBlock, "outputs#"s + std::to_string(i), *sinks[i], "in"s));
    }

    gr::scheduler::Simple sched;
    if (auto ret = sched.exchange(std::move(graph)); !ret) {
        throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
    }
    ::benchmark::benchmark<nRepeats>(std::format("src->{}->sink", gr::meta::type_name<TBlock>()), nSamples) = [&]() {
        sched.runAndWait();
        expect(eq(sinks[0]->_nSamplesProduced, nSamples));
    };
}

void runTestPureCopy() {
    using namespace boost::ut;
    using namespace benchmark;
    using namespace gr;
    using namespace gr::basic;
    using namespace gr::testing;

    gr::Graph graph;

    std::vector<TagSource<int, ProcessFunction::USE_PROCESS_BULK>*> sources;
    std::vector<TagSink<int, ProcessFunction::USE_PROCESS_BULK>*>   sinks;
    std::vector<Copy<int>*>                                         copies;

    for (std::size_t i = 0; i < nPorts; i++) {
        property_map srcParams = {{"n_samples_max", nSamples}, {"verbose_console", false}, {"disconnect_on_done", false}};
        sources.push_back(std::addressof(graph.emplaceBlock<TagSource<int, ProcessFunction::USE_PROCESS_BULK>>(srcParams)));
        copies.push_back(std::addressof(graph.emplaceBlock<Copy<int>>()));
        property_map sinkParams = {{"log_samples", false}, {"log_tags", false}, {"verbose_console", false}, {"disconnect_on_done", false}};
        sinks.push_back(std::addressof(graph.emplaceBlock<TagSink<int, ProcessFunction::USE_PROCESS_BULK>>(sinkParams)));

        expect(gr::ConnectionResult::SUCCESS == graph.connect(*sources[i], "out"s, *copies[i], "in"s));
        expect(gr::ConnectionResult::SUCCESS == graph.connect(*copies[i], "out"s, *sinks[i], "in"s));
    }

    gr::scheduler::Simple sched;
    if (auto ret = sched.exchange(std::move(graph)); !ret) {
        throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
    }
    ::benchmark::benchmark<nRepeats>("src->copy->sink", nSamples) = [&]() {
        sched.runAndWait();
        expect(eq(sinks[0]->_nSamplesProduced, nSamples));
    };
}

inline const boost::ut::suite _constexpr_bm = [] {
    runTest<gr::basic::SyncBlock<int>>();
    runTest<gr::basic::Selector<int>>();
    runTestPureCopy();
};

int main() { /* not needed by the UT framework */ }
