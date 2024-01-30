#include <list>

#include <boost/ut.hpp>

#include <gnuradio-4.0/basic/common_blocks.hpp>
#include <gnuradio-4.0/Graph.hpp>

template<typename T>
std::atomic_size_t multi_adder<T>::_unique_id_counter = 0;

template<typename T>
struct fixed_source : public gr::Block<fixed_source<T>, gr::PortOutNamed<T, "out">> {
    T value = 1;

    gr::work::Result
    work(std::size_t requested_work) {
        auto &port   = gr::outputPort<0, gr::PortType::STREAM>(this);
        auto &writer = port.streamWriter();
        auto  data   = writer.reserve_output_range(1UZ);
        data[0]      = value;
        data.publish(1UZ);

        value += 1;
        return { requested_work, 1UZ, gr::work::Status::OK };
    }
};

static_assert(gr::BlockLike<fixed_source<int>>);
static_assert(gr::traits::block::stream_input_ports<fixed_source<int>>::size() == 0);
static_assert(gr::traits::block::stream_output_ports<fixed_source<int>>::size() == 1);

template<typename T>
struct DebugSink : public gr::Block<DebugSink<T>> {
    T             lastValue = {};
    gr::PortIn<T> in;

    void
    processOne(T value) {
        lastValue = value;
    }
};

static_assert(gr::BlockLike<DebugSink<int>>);

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (DebugSink<T>), lastValue, in);

const boost::ut::suite DynamicBlocktests = [] {
    using namespace boost::ut;
    "Change number of ports dynamically"_test = [] {
        constexpr const int         sources_count = 10;
        constexpr const std::size_t events_count  = 5;

        gr::Graph testGraph;

        // Adder has sources_count inputs in total, but let's create
        // sources_count / 2 inputs on construction, and change the number
        // via settings
        auto &adder = testGraph.addBlock(std::make_unique<multi_adder<double>>(sources_count / 2));
        auto &sink  = testGraph.emplaceBlock<DebugSink<double>>({});

        // Function that adds a new source node to the graph, and connects
        // it to one of adder's ports
        std::ignore = adder.settings().set({ { "input_port_count", sources_count } });
        std::ignore = adder.settings().applyStagedParameters();

        std::vector<fixed_source<double> *> sources;
        for (std::size_t i = 0; i < sources_count; ++i) {
            auto &source = testGraph.emplaceBlock<fixed_source<double>>();
            sources.push_back(&source);
            testGraph.connect(source, 0, adder, sources.size() - 1);
        }

        testGraph.connect(adder, 0, sink, 0);

        for (std::size_t i = 0; i < events_count; ++i) {
            for (auto *source : sources) {
                source->work(1UZ);
            }
            std::ignore     = adder.work(1UZ);
            const auto work = sink.work(1UZ);
            expect(eq(work.performed_work, 1UZ));

            expect(eq(sink.lastValue, static_cast<double>((i + 1) * sources.size())));
        }

        // add yet another sources_count number of ports
        std::ignore = adder.settings().set({ { "input_port_count", 2 * sources_count } });
        std::ignore = adder.settings().applyStagedParameters();

        // if we add even more ports (and connections), the old connections should still stay connected
        for (std::size_t i = 0; i < sources_count; ++i) {
            auto &source = testGraph.emplaceBlock<fixed_source<double>>();
            sources.push_back(&source);
            testGraph.connect(source, 0, adder, sources.size() - 1);
        }

        for (std::size_t i = events_count; i < 2 * events_count; ++i) {
            for (auto *source : sources) {
                const auto source_work = source->work(1UZ);
                expect(eq(source_work.performed_work, 1UZ));
            }
            const auto adder_work = adder.work(1UZ);
            expect(eq(adder_work.performed_work, 1UZ));
            const auto sink_work = sink.work(1UZ);
            expect(eq(sink_work.performed_work, 1UZ));

            expect(eq(sink.lastValue, static_cast<double>((i + 1) * sources_count + (i - events_count + 1) * sources_count)));
        }
    };
};

int
main() { /* tests are statically executed */
}
