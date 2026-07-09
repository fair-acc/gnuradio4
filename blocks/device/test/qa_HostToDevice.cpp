#include <boost/ut.hpp>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/TransferBlocks.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

using namespace boost::ut;

const suite<"basic::HostToDevice"> h2dTests =
    [] {
        "Source -> H2D -> D2H -> Sink transfers all samples"_test = [] {
            constexpr gr::Size_t N = 1024;
            gr::Graph            flow;
            auto&                src  = flow.emplaceBlock<gr::testing::CountingSource<float>>({{"n_samples_max", N}});
            auto&                h2d  = flow.emplaceBlock<gr::basic::HostToDevice<float>>({{"chunk_size", gr::Size_t(256)}});
            auto&                d2h  = flow.emplaceBlock<gr::basic::DeviceToHost<float>>();
            auto&                sink = flow.emplaceBlock<gr::testing::CountingSink<float>>({{"n_samples_max", N}});

            expect(flow.connect<"out", "in">(src, h2d).has_value());
            expect(flow.connect<"out", "in">(h2d, d2h).has_value());
            expect(flow.connect<"out", "in">(d2h, sink).has_value());

            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(flow)).has_value());
            expect(sched.runAndWait().has_value());

            expect(eq(sink.count.value, N));
        };

        "chunk_size settings have correct defaults and limits"_test = [] {
            gr::basic::HostToDevice<float> h2d;
            expect(eq(static_cast<gr::Size_t>(h2d.chunk_size), gr::Size_t(4096)));
            expect(eq(static_cast<gr::Size_t>(h2d.min_chunk_size), gr::Size_t(256)));
            expect(eq(static_cast<gr::Size_t>(h2d.max_chunk_size), gr::Size_t(65536)));
        };

        "Source -> H2D -> D2H -> Sink with small chunk_size"_test = [] {
            constexpr gr::Size_t N = 512;
            gr::Graph            flow;
            auto&                src  = flow.emplaceBlock<gr::testing::CountingSource<float>>({{"n_samples_max", N}});
            auto&                h2d  = flow.emplaceBlock<gr::basic::HostToDevice<float>>({{"chunk_size", gr::Size_t(32)}});
            auto&                d2h  = flow.emplaceBlock<gr::basic::DeviceToHost<float>>();
            auto&                sink = flow.emplaceBlock<gr::testing::CountingSink<float>>({{"n_samples_max", N}});

            expect(flow.connect<"out", "in">(src, h2d).has_value());
            expect(flow.connect<"out", "in">(h2d, d2h).has_value());
            expect(flow.connect<"out", "in">(d2h, sink).has_value());

            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(flow)).has_value());
            expect(sched.runAndWait().has_value());

            expect(eq(sink.count.value, N));
        };
};

const suite<"basic::DeviceToHost"> d2hTests = [] {
    "DeviceToHost passes through all samples"_test = [] {
            constexpr gr::Size_t N = 256;
            gr::Graph            flow;
            auto&                src  = flow.emplaceBlock<gr::testing::CountingSource<float>>({{"n_samples_max", N}});
            auto&                d2h  = flow.emplaceBlock<gr::basic::DeviceToHost<float>>();
            auto&                sink = flow.emplaceBlock<gr::testing::CountingSink<float>>({{"n_samples_max", N}});

            expect(flow.connect<"out", "in">(src, d2h).has_value());
            expect(flow.connect<"out", "in">(d2h, sink).has_value());

            gr::scheduler::Simple<> sched;
            expect(sched.exchange(std::move(flow)).has_value());
            expect(sched.runAndWait().has_value());

            expect(eq(sink.count.value, N));
        };
};

int main() { /* not needed for UT */ }
