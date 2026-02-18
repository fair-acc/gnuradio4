#include "message_utils.hpp"

#include <boost/ut.hpp>

#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

#include <gnuradio-4.0/algorithm/ImGraph.hpp>

#include <chrono>

using TraceVectorType = std::vector<std::string>;

class Tracer {
    std::mutex      _traceMutex;
    TraceVectorType _traceVector;

public:
    void trace(std::string_view id) {
        std::scoped_lock lock{_traceMutex};
        if (_traceVector.empty() || _traceVector.back() != id) {
            _traceVector.emplace_back(id);
        }
    }

    TraceVectorType getVector() {
        std::scoped_lock lock{_traceMutex};
        return {_traceVector};
    }
};

// define some example graph nodes
template<typename T>
struct CountSource : public gr::Block<CountSource<T>> {
    gr::PortOut<T> out;
    gr::Size_t     n_samples_max = 0;

    GR_MAKE_REFLECTABLE(CountSource, out, n_samples_max);

    std::shared_ptr<Tracer> tracer{};
    gr::Size_t              count = 0;

    ~CountSource() {
        if (count != n_samples_max) {
            std::println(stderr, "Error: CountSource did not process expected number of samples: {} vs. {}", count, n_samples_max);
        }
    }

    constexpr T processOne() {
        count++;
        if (count >= n_samples_max) {
            this->requestStop();
        }
        tracer->trace(this->name);
        return static_cast<T>(count);
    }
};

static_assert(gr::BlockLike<CountSource<float>>);

template<typename T>
struct ExpectSink : public gr::Block<ExpectSink<T>> {
    gr::PortIn<T> in;
    gr::Size_t    n_samples_max = 0;

    GR_MAKE_REFLECTABLE(ExpectSink, in, n_samples_max);

    std::shared_ptr<Tracer>                         tracer{};
    gr::Size_t                                      count       = 0;
    gr::Size_t                                      false_count = 0;
    std::function<bool(std::int64_t, std::int64_t)> checker;

    ~ExpectSink() { // TODO: throwing exceptions in destructor is bad -> need to refactor test
        if (count != n_samples_max) {
            std::println(stderr, "Error: ExpectSink did not process expected number of samples: {} vs. {}", count, n_samples_max);
        }
        if (false_count != 0) {
            std::println(stderr, "Error: ExpectSink false count {} is not zero", false_count);
        }
    }

    [[nodiscard]] gr::work::Status processBulk(std::span<const T>& input) noexcept {
        tracer->trace(this->name);
        for (T data : input) {
            count++;
            if (!checker(static_cast<std::int64_t>(count), static_cast<std::int64_t>(data))) {
                false_count++;
            };
        }
        return gr::work::Status::OK;
    }
};

template<typename T>
struct Scale : public gr::Block<Scale<T>> {
    using R = decltype(std::declval<T>() * std::declval<T>());
    gr::PortIn<T>           original;
    gr::PortOut<R>          scaled;
    std::shared_ptr<Tracer> tracer{};
    T                       scale_factor = T(1.);

    GR_MAKE_REFLECTABLE(Scale, original, scaled, scale_factor);

    [[nodiscard]] constexpr auto processOne(T a) noexcept {
        tracer->trace(this->name);
        return a * scale_factor;
    }
};

template<typename T>
struct Adder : public gr::Block<Adder<T>> {
    using R = decltype(std::declval<T>() + std::declval<T>());
    gr::PortIn<T>  addend0;
    gr::PortIn<T>  addend1;
    gr::PortOut<R> sum;

    GR_MAKE_REFLECTABLE(Adder, addend0, addend1, sum);

    std::shared_ptr<Tracer> tracer;

    [[nodiscard]] constexpr auto processOne(T a, T b) noexcept {
        tracer->trace(this->name);
        return a + b;
    }
};

template<typename T>
struct Resampler : gr::Block<Resampler<T>, gr::Resampling<>, gr::Stride<>> {
    gr::PortIn<T>  in{};
    gr::PortOut<T> out{};

    GR_MAKE_REFLECTABLE(Resampler, in, out);

    std::shared_ptr<Tracer> tracer{};

    gr::work::Status processBulk(std::span<const T>& /*input*/, std::span<T>& /*output*/) noexcept {
        tracer->trace(this->name);
        return gr::work::Status::OK;
    }
};

gr::Graph getGraphLinear(std::shared_ptr<Tracer> tracer) {
    using gr::PortDirection::INPUT;
    using gr::PortDirection::OUTPUT;
    using namespace boost::ut;

    gr::Size_t nMaxSamples{100000};

    // Blocks need to be alive for as long as the flow is
    gr::Graph flow;
    // Generators
    auto& source1      = flow.emplaceBlock<CountSource<int>>({{"name", "s1"}, {"n_samples_max", nMaxSamples}});
    source1.tracer     = tracer;
    auto& scaleBlock1  = flow.emplaceBlock<Scale<int>>({{"name", "mult1"}, {"scale_factor", 2}});
    scaleBlock1.tracer = tracer;
    auto& scaleBlock2  = flow.emplaceBlock<Scale<int>>({{"name", "mult2"}, {"scale_factor", 4}});
    scaleBlock2.tracer = tracer;
    auto& sink         = flow.emplaceBlock<ExpectSink<int>>({{"name", "out"}, {"n_samples_max", nMaxSamples}});
    sink.tracer        = tracer;
    sink.checker       = [](std::uint64_t count, std::uint64_t data) -> bool { return data == 8 * count; };

    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(scaleBlock2, scaleBlock2.scaled, sink, sink.in)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(scaleBlock1, scaleBlock1.scaled, scaleBlock2, scaleBlock2.original)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(source1, source1.out, scaleBlock1, scaleBlock1.original)));

    return flow;
}

gr::Graph getGraphParallel(std::shared_ptr<Tracer> tracer) {
    using gr::PortDirection::INPUT;
    using gr::PortDirection::OUTPUT;
    using namespace boost::ut;

    gr::Size_t nMaxSamples{100000};

    // Blocks need to be alive for as long as the flow is
    gr::Graph flow;
    // Generators
    auto& source1       = flow.emplaceBlock<CountSource<int>>({{"name", "s1"}, {"n_samples_max", nMaxSamples}});
    source1.tracer      = tracer;
    auto& scaleBlock1a  = flow.emplaceBlock<Scale<int>>({{"name", "mult1a"}, {"scale_factor", 2}});
    scaleBlock1a.tracer = tracer;
    auto& scaleBlock2a  = flow.emplaceBlock<Scale<int>>({{"name", "mult2a"}, {"scale_factor", 3}});
    scaleBlock2a.tracer = tracer;
    auto& sinkA         = flow.emplaceBlock<ExpectSink<int>>({{"name", "outa"}, {"n_samples_max", nMaxSamples}});
    sinkA.tracer        = tracer;
    sinkA.checker       = [](std::uint64_t count, std::uint64_t data) -> bool { return data == 6 * count; };
    auto& scaleBlock1b  = flow.emplaceBlock<Scale<int>>({{"name", "mult1b"}, {"scale_factor", 3}});
    scaleBlock1b.tracer = tracer;
    auto& scaleBlock2b  = flow.emplaceBlock<Scale<int>>({{"name", "mult2b"}, {"scale_factor", 5}});
    scaleBlock2b.tracer = tracer;
    auto& sinkB         = flow.emplaceBlock<ExpectSink<int>>({{"name", "outb"}, {"n_samples_max", nMaxSamples}});
    sinkB.tracer        = tracer;
    sinkB.checker       = [](std::uint64_t count, std::uint64_t data) -> bool { return data == 15 * count; };

    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(scaleBlock1a, scaleBlock1a.scaled, scaleBlock2a, scaleBlock2a.original)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(scaleBlock1b, scaleBlock1b.scaled, scaleBlock2b, scaleBlock2b.original)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(scaleBlock2b, scaleBlock2b.scaled, sinkB, sinkB.in)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(source1, source1.out, scaleBlock1a, scaleBlock1a.original)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(scaleBlock2a, scaleBlock2a.scaled, sinkA, sinkA.in)));
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(source1, source1.out, scaleBlock1b, scaleBlock1b.original)));

    return flow;
}

/**
 * sets up an example graph
 * ┌───────────┐
 * │           │        ┌───────────┐
 * │ SOURCE    ├───┐    │           │
 * │           │   └────┤   x 2     ├───┐
 * └───────────┘        │           │   │    ┌───────────┐     ┌───────────┐
 *                      └───────────┘   └───►│           │     │           │
 *                                           │  SUM      ├────►│ PRINT     │
 *                           ┌──────────────►│           │     │           │
 * ┌───────────┬             ┤               └───────────┘     └───────────┘
 * │           │             │
 * │  SOURCE   ├─────────────┘
 * │           │
 * └───────────┘
 */
gr::Graph getGraphScaledSum(std::shared_ptr<Tracer> tracer, std::source_location loc = std::source_location()) {
    using gr::PortDirection::INPUT;
    using gr::PortDirection::OUTPUT;
    using namespace boost::ut;

    gr::Size_t nMaxSamples{100000};

    // Blocks need to be alive for as long as the flow is
    gr::Graph flow;

    // Generators
    auto& source1     = flow.emplaceBlock<CountSource<int>>({{"name", "s1"}, {"n_samples_max", nMaxSamples}});
    source1.tracer    = tracer;
    auto& source2     = flow.emplaceBlock<CountSource<int>>({{"name", "s2"}, {"n_samples_max", nMaxSamples}});
    source2.tracer    = tracer;
    auto& scaleBlock  = flow.emplaceBlock<Scale<int>>({{"name", "mult"}, {"scale_factor", 2}});
    scaleBlock.tracer = tracer;
    auto& addBlock    = flow.emplaceBlock<Adder<int>>({{"name", "add"}});
    addBlock.tracer   = tracer;
    auto& sink        = flow.emplaceBlock<ExpectSink<int>>({{"name", "out"}, {"n_samples_max", nMaxSamples}});
    sink.tracer       = tracer;
    sink.checker      = [](std::uint64_t count, std::uint64_t data) -> bool { return data == (2 * count) + count; };

    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(source1, source1.out, scaleBlock, scaleBlock.original)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(scaleBlock, scaleBlock.scaled, addBlock, addBlock.addend0)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(source2, source2.out, addBlock, addBlock.addend1)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(addBlock, addBlock.sum, sink, sink.in)), loc);

    return flow;
}

gr::Graph getBasicFeedBackLoop(std::shared_ptr<Tracer> tracer, std::source_location loc = std::source_location()) {
    using namespace boost::ut;

    gr::Size_t       nMaxSamples{2};
    gr::property_map layout_auto{{"layout_pref", std::string("auto")}};

    gr::Graph flow;
    auto&     source1 = flow.emplaceBlock<CountSource<float>>({{"name", "s1"}, {"n_samples_max", nMaxSamples}});
    source1.tracer    = tracer;
    auto& scale1      = flow.emplaceBlock<Scale<float>>({{"name", "alpha"}, {"scale_factor", 0.9f}});
    scale1.tracer     = tracer;
    auto& scale2      = flow.emplaceBlock<Scale<float>>({{"name", "1-alpha"}, {"scale_factor", 0.1f}, {"ui_constraints", layout_auto}});
    scale2.tracer     = tracer;
    auto& sum         = flow.emplaceBlock<Adder<float>>({{"name", "sum"}, {"ui_constraints", layout_auto}});
    sum.tracer        = tracer;
    auto& sink        = flow.emplaceBlock<ExpectSink<float>>({{"name", "out"}, {"n_samples_max", nMaxSamples}});
    sink.tracer       = tracer;
    sink.checker      = [](std::uint64_t /*count*/, float /*data*/) -> bool { return true; };

    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(source1, source1.out, scale1, scale1.original)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(scale1, scale1.scaled, sum, sum.addend0)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(sum, sum.sum, sink, sink.in)), loc);

    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(sum, sum.sum, scale2, scale2.original)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(scale2, scale2.scaled, sum, sum.addend1)), loc);

    return flow;
}

gr::Graph getResamplingFeedbackLoop(std::shared_ptr<Tracer> tracer, std::source_location loc = std::source_location::current()) {
    using namespace boost::ut;

    gr::Size_t       nMaxSamples{10};
    gr::Size_t       ratio{5};
    gr::property_map layout_auto{{"layout_pref", std::string("auto")}};

    gr::Graph flow;
    auto&     source = flow.emplaceBlock<CountSource<float>>({{"name", "src"}, {"n_samples_max", nMaxSamples}});
    source.tracer    = tracer;
    auto& adder      = flow.emplaceBlock<Adder<float>>({{"name", "sum"}, {"ui_constraints", layout_auto}});
    adder.tracer     = tracer;
    auto& sink       = flow.emplaceBlock<ExpectSink<float>>({{"name", "snk"}, {"n_samples_max", nMaxSamples / ratio}});
    sink.tracer      = tracer;
    sink.checker     = [](std::uint64_t /*count*/, float /*data*/) -> bool { return true; };

    // Decimator: 5 input samples → 1 output sample
    auto& decimator  = flow.emplaceBlock<Resampler<float>>({{"name", "dec"}, {"input_chunk_size", ratio}, {"output_chunk_size", 1}});
    decimator.tracer = tracer;
    // Interpolator: 1 input sample → 5 output samples
    auto& interpolator  = flow.emplaceBlock<Resampler<float>>({{"name", "int"}, {"input_chunk_size", 1}, {"output_chunk_size", ratio}});
    interpolator.tracer = tracer;

    // forward path: source → decimator → sum → sink
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(source, source.out, adder, adder.addend0)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(adder, adder.sum, decimator, decimator.in)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(decimator, decimator.out, sink, sink.in)), loc);

    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(decimator, decimator.out, interpolator, interpolator.in)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(interpolator, interpolator.out, adder, adder.addend1)), loc);

    return flow;
}

gr::Graph getMultipleNestedFeedbackLoops(std::shared_ptr<Tracer> tracer, std::source_location loc = std::source_location::current()) {
    using namespace boost::ut;

    gr::Size_t       nMaxSamples{2};
    gr::property_map layout_auto{{"layout_pref", std::string("auto")}};

    gr::Graph flow;
    auto&     source = flow.emplaceBlock<CountSource<float>>({{"name", "src"}, {"n_samples_max", nMaxSamples}});
    source.tracer    = tracer;

    // feedback loop #1: scale1 ⟷ scale2
    auto& scale1  = flow.emplaceBlock<Scale<float>>({{"name", "s1"}, {"scale_factor", 0.8f}, {"ui_constraints", layout_auto}});
    scale1.tracer = tracer;
    auto& scale2  = flow.emplaceBlock<Scale<float>>({{"name", "s2"}, {"scale_factor", 0.9f}, {"ui_constraints", layout_auto}});
    scale2.tracer = tracer;
    auto& adder1  = flow.emplaceBlock<Adder<float>>({{"name", "sum1"}, {"ui_constraints", layout_auto}});
    adder1.tracer = tracer;

    // feedback loop #2: scale3 ⟷ scale4
    auto& scale3  = flow.emplaceBlock<Scale<float>>({{"name", "s3"}, {"scale_factor", 0.7f}, {"ui_constraints", layout_auto}});
    scale3.tracer = tracer;
    auto& scale4  = flow.emplaceBlock<Scale<float>>({{"name", "s4"}, {"scale_factor", 0.6f}, {"ui_constraints", layout_auto}});
    scale4.tracer = tracer;
    auto& adder2  = flow.emplaceBlock<Adder<float>>({{"name", "sum2"}, {"ui_constraints", layout_auto}});
    adder2.tracer = tracer;

    auto& sink   = flow.emplaceBlock<ExpectSink<float>>({{"name", "snk"}, {"n_samples_max", nMaxSamples}});
    sink.tracer  = tracer;
    sink.checker = [](std::uint64_t /*count*/, float /*data*/) -> bool { return true; };

    // forward path: src → scale1 → sum1 → scale3 → sum2 → snk
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(source, source.out, scale1, scale1.original)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(scale1, scale1.scaled, adder1, adder1.addend0)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(adder1, adder1.sum, scale3, scale3.original)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(scale3, scale3.scaled, adder2, adder2.addend0)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(adder2, adder2.sum, sink, sink.in)), loc);

    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(adder1, adder1.sum, scale2, scale2.original)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(scale2, scale2.scaled, adder1, adder1.addend1)), loc);

    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(adder2, adder2.sum, scale4, scale4.original)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(scale4, scale4.scaled, adder2, adder2.addend1)), loc);

    return flow;
}

gr::Graph getIIRFormII(std::shared_ptr<Tracer> tracer, std::source_location loc = std::source_location::current()) {
    using namespace boost::ut;

    gr::Size_t nMaxSamples{5};

    gr::Graph flow;

    // source and sink
    auto& source  = flow.emplaceBlock<CountSource<float>>({{"name", "src"}, {"n_samples_max", nMaxSamples}});
    source.tracer = tracer;
    auto& sink    = flow.emplaceBlock<ExpectSink<float>>({{"name", "snk"}, {"n_samples_max", nMaxSamples}});
    sink.tracer   = tracer;
    sink.checker  = [](std::uint64_t /*count*/, float /*data*/) -> bool { return true; };

    // delay block (mocks)
    auto& d1  = flow.emplaceBlock<Scale<float>>({{"name", "d1"}, {"scale_factor", 1.0f}}); // z^-1
    d1.tracer = tracer;
    auto& d2  = flow.emplaceBlock<Scale<float>>({{"name", "d2"}, {"scale_factor", 1.0f}}); // z^-1
    d2.tracer = tracer;
    auto& d3  = flow.emplaceBlock<Scale<float>>({{"name", "d3"}, {"scale_factor", 1.0f}}); // z^-1
    d3.tracer = tracer;

    // feed-forward coefficients
    auto& b0  = flow.emplaceBlock<Scale<float>>({{"name", "b0"}, {"scale_factor", 1.0f}});
    b0.tracer = tracer;
    auto& b1  = flow.emplaceBlock<Scale<float>>({{"name", "b1"}, {"scale_factor", 1.0f}});
    b1.tracer = tracer;
    auto& b2  = flow.emplaceBlock<Scale<float>>({{"name", "b2"}, {"scale_factor", 1.0f}});
    b2.tracer = tracer;
    auto& b3  = flow.emplaceBlock<Scale<float>>({{"name", "b3"}, {"scale_factor", 1.0f}});
    b3.tracer = tracer;

    // feedback coefficients
    auto& a1  = flow.emplaceBlock<Scale<float>>({{"name", "a1"}, {"scale_factor", -1.0f}});
    a1.tracer = tracer;
    auto& a2  = flow.emplaceBlock<Scale<float>>({{"name", "a2"}, {"scale_factor", -1.0f}});
    a2.tracer = tracer;
    auto& a3  = flow.emplaceBlock<Scale<float>>({{"name", "a3"}, {"scale_factor", -1.0f}});
    a3.tracer = tracer;

    // adders for cascaded feedback signal summation
    auto& feedbackSum0  = flow.emplaceBlock<Adder<float>>({{"name", "fbSum0"}});
    feedbackSum0.tracer = tracer;
    auto& feedbackSum1  = flow.emplaceBlock<Adder<float>>({{"name", "fbSum1"}}); // combines a2 and a3
    feedbackSum1.tracer = tracer;
    auto& feedbackSum2  = flow.emplaceBlock<Adder<float>>({{"name", "fbSum2"}}); // combines a1 with (a2+a3)
    feedbackSum2.tracer = tracer;

    // adders for cascaded feed-forward signal summation
    auto& outputSum0  = flow.emplaceBlock<Adder<float>>({{"name", "ffSum0"}}); // combines b0 and sum(b1,b2,b3)
    outputSum0.tracer = tracer;
    auto& outputSum1  = flow.emplaceBlock<Adder<float>>({{"name", "ffSum1"}}); // combines b2 and b3
    outputSum1.tracer = tracer;
    auto& outputSum2  = flow.emplaceBlock<Adder<float>>({{"name", "ffSum2"}}); // combines b1 with (b2+b3)
    outputSum2.tracer = tracer;

    // main path src -> sum (feedback branches) -> b0 -> sum (feed-forward branches) -> snk
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(source, source.out, feedbackSum0, feedbackSum0.addend0)), loc); // src -> feedbackSum0
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(feedbackSum0, feedbackSum0.sum, b0, b0.original)), loc);        // b0 * v(n)
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(b0, b0.scaled, outputSum0, outputSum0.addend0)), loc);          // b0 -> outputSum0
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(outputSum0, outputSum0.sum, sink, sink.in)), loc);              // outputSum0 -> snk

    // delay line: v(n) → v(n-1) → v(n-2) → v(n-3)
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(feedbackSum0, feedbackSum0.sum, d1, d1.original)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(d1, d1.scaled, d2, d2.original)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(d2, d2.scaled, d3, d3.original)), loc);

    // feedback path
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(d1, d1.scaled, a1, a1.original)), loc); // -a1 * v(n-1)
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(d2, d2.scaled, a2, a2.original)), loc); // -a2 * v(n-2)
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(d3, d3.scaled, a3, a3.original)), loc); // -a3 * v(n-3)

    // cascaded feedback summation: a3 + a2 -> feedbackSum2, then + a1 -> feedbackSum1
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(a2, a2.scaled, feedbackSum2, feedbackSum2.addend0)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(a3, a3.scaled, feedbackSum2, feedbackSum2.addend1)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(a1, a1.scaled, feedbackSum1, feedbackSum1.addend0)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(feedbackSum2, feedbackSum2.sum, feedbackSum1, feedbackSum1.addend1)), loc);
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(feedbackSum1, feedbackSum1.sum, feedbackSum0, feedbackSum0.addend1)), loc);

    // feed-forward path
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(d1, d1.scaled, b1, b1.original)), loc); // b1 * v(n-1)
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(d2, d2.scaled, b2, b2.original)), loc); // b2 * v(n-2)
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(d3, d3.scaled, b3, b3.original)), loc); // b3 * v(n-3)

    // cascaded feed-forward summation: b3 + b2 -> outputSum1, then + b1 -> outputSum2
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(b2, b2.scaled, outputSum1, outputSum1.addend0)), loc);              // FIXED: b2 -> addend0
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(b3, b3.scaled, outputSum1, outputSum1.addend1)), loc);              // b3 -> addend1
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(b1, b1.scaled, outputSum2, outputSum2.addend0)), loc);              // FIXED: b1 -> addend0
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(outputSum1, outputSum1.sum, outputSum2, outputSum2.addend1)), loc); // outputSum1 -> addend1
    expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(outputSum2, outputSum2.sum, outputSum0, outputSum0.addend1)), loc); // FIXED: complete chain to outputSum0

    return flow;
}

template<typename TBlock>
void checkBlockNames(const std::vector<TBlock>& joblist, std::set<std::string> set, std::source_location loc = std::source_location()) {
    boost::ut::expect(boost::ut::that % joblist.size() == set.size(), loc);
    for (auto& block : joblist) {
        boost::ut::expect(boost::ut::that % set.contains(std::string(block->name())), loc) << std::format("{} not in {{{}}}\n", block->name(), gr::join(set));
    }
}

template<typename T>
struct LifecycleSource : public gr::Block<LifecycleSource<T>> {
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(LifecycleSource, out);

    std::int32_t n_samples_produced = 0;
    std::int32_t n_samples_max      = 10;

    [[nodiscard]] constexpr T processOne() noexcept {
        n_samples_produced++;
        if (n_samples_produced >= n_samples_max) {
            this->requestStop();
            return T(n_samples_produced); // this sample will be the last emitted.
        }
        return T(n_samples_produced);
    }
};

template<typename T>
struct LifecycleBlock : public gr::Block<LifecycleBlock<T>> {
    gr::PortIn<T>                in{};
    gr::PortOut<T, gr::Optional> out{};

    GR_MAKE_REFLECTABLE(LifecycleBlock, in, out);

    int process_one_count{};
    int start_count{};
    int stop_count{};
    int reset_count{};
    int pause_count{};
    int resume_count{};

    [[nodiscard]] constexpr T processOne(T a) noexcept {
        process_one_count++;
        return a;
    }

    void start() { start_count++; }

    void stop() { stop_count++; }

    void reset() { reset_count++; }

    void pause() { pause_count++; }

    void resume() { resume_count++; }
};

template<typename T>
struct BusyLoopBlock : public gr::Block<BusyLoopBlock<T>> {
    using enum gr::work::Status;
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(BusyLoopBlock, in, out);

    gr::Sequence _produceCount{0};
    gr::Sequence _invokeCount{0};

    [[nodiscard]] constexpr gr::work::Status processBulk(gr::InputSpanLike auto& input, gr::OutputSpanLike auto& output) noexcept {
        auto produceCount = _produceCount.value();
        _invokeCount.incrementAndGet();

        if (produceCount == 0) {
            // early return by not explicitly consuming/producing but returning incomplete state
            // normally this should be reserved for "starving" blocks. Here, it's being used to unit-test this alternative behaviour
            return gr::lifecycle::isActive(this->state()) ? INSUFFICIENT_OUTPUT_ITEMS : DONE;
        }

        std::println("##BusyLoopBlock produces data _invokeCount: {}", _invokeCount.value());
        std::ranges::copy(input.begin(), input.end(), output.begin());
        produceCount = _produceCount.subAndGet(1L);
        return OK;
    }
};

const boost::ut::suite<"SchedulerTests"> SchedulerSettingsTests = [] {
    using namespace boost::ut;
    using namespace gr;

    "Scheduler move crash"_test = [] {
        // Scheduler crashed if exchanged graph twice
        gr::scheduler::Simple<> s0;
        gr::Graph               g1;
        auto                    oldGraph = s0.exchange(std::move(g1));
        expect(oldGraph.has_value()) << "oldGraph should have a value";

        auto g1Again = s0.exchange(std::move(oldGraph.value()));
        expect(g1Again.has_value()) << "g1Again should have a value";
    };

    "Direct settings change"_test = [] {
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(getGraphLinear(trace)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }

        auto ret1 = sched.settings().set({{"timeout_ms", gr::Size_t(6)}});
        expect(ret1.empty()) << "setting one known parameter";
        expect(sched.settings().stagedParameters().empty());          // set(...) does not change stagedParameters
        expect(not sched.settings().changed()) << "settings changed"; // set(...) does not change changed()
        std::ignore = sched.settings().activateContext();

        std::println("Staged {}", sched.settings().stagedParameters());
        expect(sched.settings().stagedParameters().contains("timeout_ms"));

        expect(sched.settings().changed()) << "settings changed";
        std::ignore = sched.settings().applyStagedParameters();

        expect(eq(sched.timeout_ms.value, 6U));

        sched.settings().updateActiveParameters();

        auto ret2 = sched.settings().set({{"timeout_ms", gr::Size_t(42)}});
        expect(ret2.empty()) << "setting one known parameter";
        expect(sched.settings().stagedParameters().empty());          // set(...) does not change stagedParameters
        expect(not sched.settings().changed()) << "settings changed"; // set(...) does not change changed()
        std::ignore = sched.settings().activateContext();

        std::println("Staged {}", sched.settings().stagedParameters());
        expect(sched.settings().stagedParameters().contains("timeout_ms"));

        expect(sched.settings().changed()) << "settings changed";
        std::ignore = sched.settings().applyStagedParameters();

        expect(eq(sched.timeout_ms.value, 42U));

        sched.settings().updateActiveParameters();

        expect(sched.runAndWait().has_value());
    };
};

const boost::ut::suite<"SchedulerTests"> SchedulerTests = [] {
    using namespace std::chrono_literals;
    using namespace boost::ut;
    using namespace gr;

    // needs to be exceptionally pinned to [2, 2] min/max thread count of unit-test
    using namespace gr::thread_pool;
    auto cpu = std::make_shared<ThreadPoolWrapper>(std::make_unique<BasicThreadPool>(std::string(kDefaultCpuPoolId), TaskType::CPU_BOUND, 2U, 2U), "CPU");
    gr::thread_pool::Manager::instance().replacePool(std::string(kDefaultCpuPoolId), std::move(cpu));
    const auto minThreads = gr::thread_pool::Manager::defaultCpuPool()->minThreads();
    const auto maxThreads = gr::thread_pool::Manager::defaultCpuPool()->maxThreads();
    std::println("INFO: std::thread::hardware_concurrency() = {} - CPU thread bounds = [{}, {}]", std::thread::hardware_concurrency(), minThreads, maxThreads);

    "SimpleScheduler_linear"_test = [] {
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(getGraphLinear(trace)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() == 8u);
        expect(boost::ut::that % t == TraceVectorType{"s1", "mult1", "mult2", "out", "s1", "mult1", "mult2", "out"});
    };

    "BreadthFirstScheduler_linear"_test = [] {
        std::shared_ptr<Tracer>       trace = std::make_shared<Tracer>();
        gr::scheduler::BreadthFirst<> sched;
        if (auto ret = sched.exchange(getGraphLinear(trace)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() == 8u);
        expect(boost::ut::that % t == TraceVectorType{"s1", "mult1", "mult2", "out", "s1", "mult1", "mult2", "out"});
    };

    "SimpleScheduler_parallel"_test = [] {
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(getGraphParallel(trace)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() == 14u);
        expect(boost::ut::that % t == TraceVectorType{"s1", "mult1a", "mult2a", "outa", "mult1b", "mult2b", "outb", "s1", "mult1a", "mult2a", "outa", "mult1b", "mult2b", "outb"});
    };

    "BreadthFirstScheduler_parallel"_test = [] {
        std::shared_ptr<Tracer>       trace = std::make_shared<Tracer>();
        gr::scheduler::BreadthFirst<> sched;
        if (auto ret = sched.exchange(getGraphParallel(trace)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() == 14u);
        expect(boost::ut::that % t == TraceVectorType{
                                          "s1",
                                          "mult1a",
                                          "mult1b",
                                          "mult2a",
                                          "mult2b",
                                          "outa",
                                          "outb",
                                          "s1",
                                          "mult1a",
                                          "mult1b",
                                          "mult2a",
                                          "mult2b",
                                          "outa",
                                          "outb",
                                      });
    };

    "SimpleScheduler_scaled_sum"_test = [] {
        // construct an example graph and get an adjacency list for it
        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(getGraphScaledSum(trace)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() == 10u);
        expect(boost::ut::that % t == TraceVectorType{"s1", "s2", "mult", "add", "out", "s1", "s2", "mult", "add", "out"});
    };

    "BreadthFirstScheduler_scaled_sum"_test = [] {
        std::shared_ptr<Tracer>       trace = std::make_shared<Tracer>();
        gr::scheduler::BreadthFirst<> sched;
        if (auto ret = sched.exchange(getGraphScaledSum(trace)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() == 10u);
        expect(boost::ut::that % t == TraceVectorType{"s1", "s2", "mult", "add", "out", "s1", "s2", "mult", "add", "out"});
    };

    "SimpleScheduler_linear_multi_threaded"_test = [] {
        std::shared_ptr<Tracer>                                              trace = std::make_shared<Tracer>();
        gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded> sched;
        if (auto ret = sched.exchange(getGraphLinear(trace)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(that % t.size() >= 8u);
    };

    "BreadthFirstScheduler_linear_multi_threaded"_test = [] {
        std::shared_ptr<Tracer>                                                    trace = std::make_shared<Tracer>();
        gr::scheduler::BreadthFirst<gr::scheduler::ExecutionPolicy::multiThreaded> sched;
        if (auto ret = sched.exchange(getGraphLinear(trace)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        expect(sched.changeStateTo(gr::lifecycle::State::INITIALISED).has_value());
        expect(sched.jobs()->size() == 2u);
        checkBlockNames(sched.jobs()->at(0), {"s1", "mult2"});
        checkBlockNames(sched.jobs()->at(1), {"mult1", "out"});
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() >= 8u) << std::format("execution order incomplete: {}", gr::join(t, ", "));
    };

    "SimpleScheduler_parallel_multi_threaded"_test = [] {
        std::shared_ptr<Tracer>                                              trace = std::make_shared<Tracer>();
        gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded> sched;
        if (auto ret = sched.exchange(getGraphParallel(trace)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() >= 14u) << std::format("execution order incomplete: {}", gr::join(t, ", "));
    };

    "BreadthFirstScheduler_parallel_multi_threaded"_test = [] {
        std::shared_ptr<Tracer>                                                    trace = std::make_shared<Tracer>();
        gr::scheduler::BreadthFirst<gr::scheduler::ExecutionPolicy::multiThreaded> sched;
        if (auto ret = sched.exchange(getGraphParallel(trace)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        expect(sched.changeStateTo(gr::lifecycle::State::INITIALISED).has_value());
        expect(sched.jobs()->size() == 2u);
        checkBlockNames(sched.jobs()->at(0), {"s1", "mult1b", "mult2b", "outb"});
        checkBlockNames(sched.jobs()->at(1), {"mult1a", "mult2a", "outa"});
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() >= 14u);
    };

    "SimpleScheduler_scaled_sum_multi_threaded"_test = [] {
        // construct an example graph and get an adjacency list for it
        std::shared_ptr<Tracer>                                              trace = std::make_shared<Tracer>();
        gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded> sched;
        if (auto ret = sched.exchange(getGraphScaledSum(trace)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() >= 10u);
    };

    "BreadthFirstScheduler_scaled_sum_multi_threaded"_test = [] {
        std::shared_ptr<Tracer>                                                    trace = std::make_shared<Tracer>();
        gr::scheduler::BreadthFirst<gr::scheduler::ExecutionPolicy::multiThreaded> sched;
        if (auto ret = sched.exchange(getGraphScaledSum(trace)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        expect(sched.changeStateTo(gr::lifecycle::State::INITIALISED).has_value());
        expect(eq(sched.jobs()->size(), 2u));
        checkBlockNames(sched.jobs()->at(0), {"s1", "mult", "out"});
        checkBlockNames(sched.jobs()->at(1), {"s2", "add"});
        expect(sched.runAndWait().has_value());
        auto t = trace->getVector();
        expect(boost::ut::that % t.size() >= 10u);
    };

    "Basic Feedback Loop"_test = [] {
        std::shared_ptr<Tracer>          trace         = std::make_shared<Tracer>();
        Graph                            graph         = getBasicFeedBackLoop(trace);
        std::vector<graph::FeedbackLoop> feedbackLoops = gr::graph::detectFeedbackLoops(graph);
        expect(eq(feedbackLoops.size(), 1UZ));
        gr::graph::printFeedbackLoop(feedbackLoops.at(0UZ));
        auto priming = gr::graph::calculateLoopPrimingSize(feedbackLoops.at(0UZ));
        expect(priming.has_value()) << [&] { return std::format("couldn't calculate loop priming size: {}\n{}\n", priming.error(), feedbackLoops.at(0UZ).edges); };
        expect(eq(priming.value(), 1UZ));

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }

        expect(sched.runAndWait().has_value()) << "scheduler should complete successfully";
        auto t = trace->getVector();
        expect(eq(t.size(), 8UZ));
        expect(eq(t, TraceVectorType{"s1", "alpha", "sum", "out", "1-alpha", "sum", "out", "1-alpha"}));
    };

    "Resampling Feedback Loop"_test = [] {
        std::shared_ptr<Tracer>          trace         = std::make_shared<Tracer>();
        Graph                            graph         = getResamplingFeedbackLoop(trace);
        std::vector<graph::FeedbackLoop> feedbackLoops = gr::graph::detectFeedbackLoops(graph);
        expect(eq(feedbackLoops.size(), 1UZ));
        gr::graph::printFeedbackLoop(feedbackLoops.at(0UZ));
        auto priming = gr::graph::calculateLoopPrimingSize(feedbackLoops.at(0UZ));
        expect(priming.has_value()) << [&] { return std::format("couldn't calculate loop priming size: {}\n{}\n", priming.error(), feedbackLoops.at(0UZ).edges); };
        expect(eq(priming.value(), 5UZ));

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        expect(sched.runAndWait().has_value()) << "scheduler should complete successfully";

        auto t = trace->getVector();
        expect(eq(t.size(), 9UZ));
        std::println("execution trace: {}", t);
        expect(eq(t, TraceVectorType{"src", "sum", "dec", "int", "sum", "snk", "dec", "int", "snk"}));
    };

    "Multiple Nested Feedback Loops"_test = [] {
        std::shared_ptr<Tracer>          trace         = std::make_shared<Tracer>();
        Graph                            graph         = getMultipleNestedFeedbackLoops(trace);
        std::vector<graph::FeedbackLoop> feedbackLoops = gr::graph::detectFeedbackLoops(graph);
        for (const auto& loop : feedbackLoops) {
            gr::graph::printFeedbackLoop(loop);
        }
        expect(eq(feedbackLoops.size(), 2UZ));

        // test priming for both loops
        for (std::size_t i = 0UZ; i < feedbackLoops.size(); ++i) {
            auto priming = gr::graph::calculateLoopPrimingSize(feedbackLoops.at(0UZ));
            expect(priming.has_value()) << [&] { return std::format("couldn't calculate loop priming size: {}\n{}\n", priming.error(), feedbackLoops.at(0UZ).edges); };
            expect(eq(priming.value(), 1UZ));
        }

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        expect(sched.runAndWait().has_value()) << "scheduler should complete successfully";

        auto t = trace->getVector();
        expect(eq(t.size(), 14UZ));
        std::println("execution trace: {}", t);
        expect(eq(t, TraceVectorType{"src", "s1", "sum1", "s3", "sum2", "snk", "s2", "sum1", "s3", "s4", "sum2", "snk", "s2", "s4"}));
    };

    "IIR Form II Feedback Loops"_test = [] {
        std::shared_ptr<Tracer>          trace         = std::make_shared<Tracer>();
        Graph                            graph         = getIIRFormII(trace);
        std::vector<graph::FeedbackLoop> feedbackLoops = gr::graph::detectFeedbackLoops(graph);
        for (const auto& loop : feedbackLoops) {
            gr::graph::printFeedbackLoop(loop);
        }
        expect(eq(feedbackLoops.size(), 1UZ));

        // test priming for both loops
        for (std::size_t i = 0UZ; i < feedbackLoops.size(); ++i) {
            auto priming = gr::graph::calculateLoopPrimingSize(feedbackLoops.at(0UZ));
            expect(priming.has_value()) << [&] { return std::format("couldn't calculate loop priming size: {}\n{}\n", priming.error(), feedbackLoops.at(0UZ).edges); };
            expect(eq(priming.value(), 1UZ));
        }

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(graph)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        expect(sched.runAndWait().has_value()) << "scheduler should complete successfully";

        auto t = trace->getVector();
        expect(eq(t.size(), 85UZ));
        std::println("execution trace: {}", t);
    };

    "LifecycleBlock"_test = [] {
        gr::Graph flow;

        auto& lifecycleSource = flow.emplaceBlock<LifecycleSource<float>>();
        auto& lifecycleBlock  = flow.emplaceBlock<LifecycleBlock<float>>();
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(lifecycleSource, lifecycleSource.out, lifecycleBlock, lifecycleBlock.in)));

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(flow)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        expect(sched.runAndWait().has_value());
        expect(sched.changeStateTo(gr::lifecycle::State::INITIALISED).has_value());

        expect(eq(lifecycleSource.n_samples_produced, lifecycleSource.n_samples_max)) << "Source n_samples_produced != n_samples_max";
        expect(eq(lifecycleBlock.process_one_count, lifecycleSource.n_samples_max)) << "process_one_count != n_samples_produced";

        expect(eq(lifecycleBlock.process_one_count, lifecycleSource.n_samples_produced));
        expect(eq(lifecycleBlock.start_count, 1));
        expect(eq(lifecycleBlock.stop_count, 1));
        expect(eq(lifecycleBlock.pause_count, 0));
        expect(eq(lifecycleBlock.resume_count, 0));
        expect(eq(lifecycleBlock.reset_count, 1));
    };

    "propagate DONE check-infinite loop"_test = [] {
        using namespace gr::testing;
        gr::Graph flow;

        auto& source  = flow.emplaceBlock<CountingSource<float>>();
        auto& monitor = flow.emplaceBlock<Copy<float>>();
        auto& sink    = flow.emplaceBlock<NullSink<float>>();
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(source, source.out, monitor, monitor.in)));
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(monitor, monitor.out, sink, sink.in)));

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(flow)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        std::atomic_bool shutDownByWatchdog{false};

        auto watchdogThread = gr::test::thread_pool::execute("watchdog", [&sched, &shutDownByWatchdog]() {
            while (sched.state() != gr::lifecycle::State::RUNNING) { // wait until scheduler is running
                std::this_thread::sleep_for(40ms);
            }

            if (sched.state() == gr::lifecycle::State::RUNNING) {
                shutDownByWatchdog.store(true, std::memory_order_relaxed);
                sched.requestStop();
            }
        });

        expect(sched.runAndWait().has_value());
        watchdogThread.wait();

        expect(ge(source.count, 0U));
        expect(shutDownByWatchdog.load(std::memory_order_relaxed));
        expect(sched.state() == gr::lifecycle::State::STOPPED);
        std::println("N.B by-design infinite loop correctly stopped after having emitted {} samples", source.count);
    };

    // create and return a watchdog thread and its control flag
    using TDuration     = std::chrono::duration<std::chrono::steady_clock::rep, std::chrono::steady_clock::period>;
    auto createWatchdog = [](auto& sched, TDuration timeOut = 2s, TDuration pollingPeriod = 40ms) {
        using namespace std::chrono_literals;
        auto externalInterventionNeeded = std::make_shared<std::atomic_bool>(false); // unique_ptr because you cannot move atomics

        // Create the watchdog thread
        auto watchdogThread = gr::test::thread_pool::execute("watchdog", [&sched, &externalInterventionNeeded, timeOut, pollingPeriod]() {
            auto timeout = std::chrono::steady_clock::now() + timeOut;
            while (std::chrono::steady_clock::now() < timeout) {
                if (sched.state() == gr::lifecycle::State::STOPPED) {
                    return;
                }
                std::this_thread::sleep_for(pollingPeriod);
            }
            // time-out reached, need to force termination of scheduler
            std::println("watchdog kicked in");
            externalInterventionNeeded->store(true, std::memory_order_relaxed);
            sched.requestStop();
            std::println("requested scheduler to stop");
        });

        return std::make_pair(std::move(watchdogThread), externalInterventionNeeded);
    };

    "propagate source DONE state: down-stream using EOS tag"_test = [&createWatchdog] {
        using namespace gr::testing;
        gr::Graph flow;

        auto& source  = flow.emplaceBlock<ConstantSource<float>>({{"n_samples_max", 1024U}});
        auto& monitor = flow.emplaceBlock<Copy<float>>();
        auto& sink    = flow.emplaceBlock<CountingSink<float>>();
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(source, source.out, monitor, monitor.in)));
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(monitor, monitor.out, sink, sink.in)));

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(flow)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        auto [watchdogThread, externalInterventionNeeded] = createWatchdog(sched, 2s);
        expect(sched.runAndWait().has_value());

        watchdogThread.wait();
        expect(!externalInterventionNeeded->load(std::memory_order_relaxed));
        expect(eq(source.count, 1024U));
        expect(eq(sink.count, 1024U));

        std::println("N.B. 'propagate source DONE state: down-stream using EOS tag' test finished");
    };

    "propagate monitor DONE status: down-stream using EOS tag, upstream via disconnecting ports"_test = [&createWatchdog] {
        using namespace gr::testing;
        gr::Graph flow;

        auto& source  = flow.emplaceBlock<NullSource<float>>();
        auto& monitor = flow.emplaceBlock<HeadBlock<float>>({{"n_samples_max", 1024U}});
        auto& sink    = flow.emplaceBlock<CountingSink<float>>();
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(source, source.out, monitor, monitor.in)));
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(monitor, monitor.out, sink, sink.in)));

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(flow)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        auto [watchdogThread, externalInterventionNeeded] = createWatchdog(sched, 2s);
        expect(sched.runAndWait().has_value());

        watchdogThread.wait();
        expect(!externalInterventionNeeded->load(std::memory_order_relaxed));
        expect(eq(monitor.count, 1024U));
        expect(eq(sink.count, 1024U));

        std::println("N.B. 'propagate monitor DONE status: down-stream using EOS tag, upstream via disconnecting ports' test finished");
    };

    "propagate sink DONE status: upstream via disconnecting ports"_test = [&createWatchdog] {
        using namespace gr::testing;
        gr::Graph flow;

        auto& source  = flow.emplaceBlock<NullSource<float>>();
        auto& monitor = flow.emplaceBlock<Copy<float>>();
        auto& sink    = flow.emplaceBlock<CountingSink<float>>({{"n_samples_max", 1024U}});
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(source, source.out, monitor, monitor.in)));
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(monitor, monitor.out, sink, sink.in)));

        gr::scheduler::Simple<> sched;
        if (auto ret = sched.exchange(std::move(flow)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        auto [watchdogThread, externalInterventionNeeded] = createWatchdog(sched, 2s);
        expect(sched.runAndWait().has_value());

        watchdogThread.wait();
        expect(!externalInterventionNeeded->load(std::memory_order_relaxed));
        expect(eq(sink.count, 1024U));

        std::println("N.B. 'propagate sink DONE status: upstream via disconnecting ports' test finished");
    };

    "blocking scheduler"_test = [] {
        using namespace gr;
        using namespace gr::testing;

        Graph flow;
        auto& source  = flow.emplaceBlock<NullSource<float>>();
        auto& monitor = flow.emplaceBlock<BusyLoopBlock<float>>();
        auto& sink    = flow.emplaceBlock<NullSink<float>>();
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(source, source.out, monitor, monitor.in)));
        expect(eq(gr::ConnectionResult::SUCCESS, flow.connect(monitor, monitor.out, sink, sink.in)));

        scheduler::Simple<scheduler::ExecutionPolicy::singleThreadedBlocking> scheduler;
        if (auto ret = scheduler.exchange(std::move(flow)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        scheduler.timeout_ms               = 100U; // also dynamically settable via messages/block interface
        scheduler.timeout_inactivity_count = 10U;  // also dynamically settable via messages/block interface

        expect(eq(0UZ, scheduler.graph().progress().value())) << "initial progress definition (0)";

        auto schedulerThreadHandle = gr::test::thread_pool::executeScheduler("qa_Sched", scheduler);

        expect(awaitCondition(2s, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";

        expect(scheduler.state() == lifecycle::State::RUNNING) << "scheduler thread up and running";

        auto oldProgress = scheduler.graph().progress().value();
        expect(awaitCondition(2s, [&scheduler, &oldProgress] { // wait until there is no more progress (i.e. wait until all initial buffers are filled)
            std::this_thread::sleep_for(200ms);                // wait
            auto newProgress = scheduler.graph().progress().value();
            if (oldProgress == newProgress) {
                return true;
            }
            oldProgress = newProgress;
            return false;
        })) << "BusyLoopBlock sleeping";

        const auto progressAfterInit = scheduler.graph().progress().value();
        auto       estInvokeCount    = [&monitor] {
            const auto invokeCountInit = monitor._invokeCount.value();
            std::this_thread::sleep_for(20ms);
            return monitor._invokeCount.value() - invokeCountInit;
        };

        const auto invokeCount0 = estInvokeCount();
        expect(eq(scheduler.graph().progress().value(), progressAfterInit)) << "after thread started definition (0) - mark1";

        std::this_thread::sleep_for(200ms); // wait for time-out
        const auto invokeCount1 = estInvokeCount();

        expect(ge(invokeCount0, invokeCount1)) << std::format("info: invoke counts when active: {} sleeping: {}", invokeCount0, invokeCount1);
        std::println("info: invoke counts when active: {} sleeping: {}", invokeCount0, invokeCount1);
        expect(eq(scheduler.graph().progress().value(), progressAfterInit)) << "after thread started definition (0) - mark2";

        monitor._produceCount.setValue(1L);
        const auto invokeCount2 = estInvokeCount();
        expect(ge(invokeCount2, invokeCount1)) << std::format("info: invoke counts when active: {} sleeping: {}", invokeCount2, invokeCount1);
        std::println("info: invoke counts when active: {} sleeping: {}", invokeCount2, invokeCount1);

        expect(ge(scheduler.graph().progress().value(), progressAfterInit)) << "final progress definition (>0)";
        std::println("final progress {}", scheduler.graph().progress().value());

        expect(scheduler.state() == lifecycle::State::RUNNING) << "is running";
        std::println("request to shut-down");
        scheduler.requestStop();

        auto        schedulerResult = schedulerThreadHandle.get();
        std::string errorMsg        = schedulerResult.has_value() ? "" : std::format("nested scheduler execution failed:\n{:f}\n", schedulerResult.error());
        expect(schedulerResult.has_value()) << errorMsg;
    };

    "AdjacencyList_basic_linear_graph"_test = [] {
        using namespace gr;
        using TBlock = Scale<int>;
        gr::Graph graph;

        TBlock& A = graph.emplaceBlock<TBlock>({{"name", "A"}});
        TBlock& B = graph.emplaceBlock<TBlock>({{"name", "B"}});
        TBlock& C = graph.emplaceBlock<TBlock>({{"name", "C"}});

        expect(eq(graph.connect(A, A.scaled, B, B.original), ConnectionResult::SUCCESS));
        expect(eq(graph.connect(B, B.scaled, C, C.original), ConnectionResult::SUCCESS));

        gr::Graph                                flat       = gr::graph::flatten(graph);
        gr::graph::AdjacencyList                 acencyList = gr::graph::computeAdjacencyList(flat);
        std::vector<std::shared_ptr<BlockModel>> sources    = gr::graph::findSourceBlocks(acencyList);

        expect(eq(sources.size(), 1UZ));
        expect(eq(sources[0UZ]->name(), "A"sv));

        std::shared_ptr<gr::BlockModel> srcBlock = gr::graph::findBlock(graph, A.unique_name).value();
        std::span<const Edge* const>    edges    = gr::graph::outgoingEdges(acencyList, srcBlock, 0UZ /* first port - resolved to number in Edge through connection */);
        expect(eq(edges.size(), 1UZ)) << fatal;
        expect(eq(edges[0UZ]->_destinationBlock->name(), "B"sv));
    };

    "AdjacencyList_forked_graph"_test = [] {
        using namespace gr;
        using TBlock = Scale<int>;
        gr::Graph graph;

        TBlock& A = graph.emplaceBlock<TBlock>({{"name", "A"}});
        TBlock& B = graph.emplaceBlock<TBlock>({{"name", "B"}});
        TBlock& C = graph.emplaceBlock<TBlock>({{"name", "C"}});

        expect(eq(graph.connect(A, A.scaled, B, B.original), ConnectionResult::SUCCESS));
        expect(eq(graph.connect(A, A.scaled, C, C.original), ConnectionResult::SUCCESS));

        gr::Graph                                flat          = gr::graph::flatten(graph);
        gr::graph::AdjacencyList                 adjacencyList = gr::graph::computeAdjacencyList(flat);
        std::vector<std::shared_ptr<BlockModel>> srcs          = gr::graph::findSourceBlocks(adjacencyList);

        expect(eq(srcs.size(), 1UZ));
        expect(eq(srcs[0UZ]->name(), "A"sv));

        std::shared_ptr<gr::BlockModel>  srcBlock = gr::graph::findBlock(graph, A.unique_name).value();
        std::span<const gr::Edge* const> edges    = gr::graph::outgoingEdges(adjacencyList, srcBlock, 0UZ /* first port - resolved to number in Edge through connection */);
        expect(eq(edges.size(), 2UZ)) << fatal;
        std::set<std::string_view> targets{edges[0UZ]->_destinationBlock->name(), edges[1UZ]->_destinationBlock->name()};
        expect(targets.contains("B"sv) && targets.contains("C"sv));
    };

    "Scheduler_batchBlocks_round_robin"_test = [] {
        using namespace gr;
        using TBlock = Scale<int>;
        std::vector<std::shared_ptr<gr::BlockModel>> blocks;
        for (std::size_t i = 0UZ; i < 6UZ; ++i) {
            const std::shared_ptr<BlockModel>& newBlock    = std::make_shared<BlockWrapper<TBlock>>();
            TBlock*                            rawBlockRef = static_cast<TBlock*>(newBlock->raw());
            rawBlockRef->name                              = std::format("B{}", i);
            blocks.push_back(newBlock);
        }

        gr::scheduler::JobLists batches = gr::scheduler::detail::batchBlocks(blocks, 3UZ);
        expect(eq(batches.size(), 3UZ));
        expect(eq(batches[0UZ].size(), 2UZ));
        expect(eq(batches[1UZ].size(), 2UZ));
        expect(eq(batches[2UZ].size(), 2UZ));

        // check round-robin assignment (B0, B3), (B1, B4), (B2, B5)
        expect(eq(batches[0UZ][0UZ]->name(), "B0"sv));
        expect(eq(batches[1UZ][0UZ]->name(), "B1"sv));
        expect(eq(batches[2UZ][0UZ]->name(), "B2"sv));
    };

    "findSourceBlocks_mixed_topology"_test = [] {
        using namespace gr;
        using TBlock = Scale<int>;
        gr::Graph graph;

        TBlock& blockA = graph.emplaceBlock<TBlock>({{"name", "A"}});
        TBlock& blockB = graph.emplaceBlock<TBlock>({{"name", "B"}});
        TBlock& blockC = graph.emplaceBlock<TBlock>({{"name", "C"}});
        TBlock& blockD = graph.emplaceBlock<TBlock>({{"name", "D"}}); // isolated

        expect(eq(graph.connect(blockA, blockA.scaled, blockB, blockB.original), ConnectionResult::SUCCESS));
        expect(eq(graph.connect(blockB, blockB.scaled, blockC, blockC.original), ConnectionResult::SUCCESS));

        gr::Graph                flattened     = gr::graph::flatten(graph);
        gr::graph::AdjacencyList adjacencyList = gr::graph::computeAdjacencyList(flattened);

        std::set<std::string_view> srcNames;
        for (std::shared_ptr<BlockModel> s : gr::graph::findSourceBlocks(adjacencyList)) {
            srcNames.insert(s->uniqueName());
        }
        expect(srcNames.contains(blockA.unique_name)) << "didn't find source block";
        expect(!srcNames.contains(blockB.unique_name)) << "blockB is not a source block";
        expect(!srcNames.contains(blockC.unique_name)) << "blockC is not a source block";
        expect(!srcNames.contains(blockD.unique_name)) << "blockD is not a source block"; // isolated node also not in adjacency list (see below)

        std::set<std::string_view> names;
        for (const auto& fromBlock : adjacencyList | std::views::keys) {
            names.insert(fromBlock->uniqueName());
        }

        expect(names.contains(blockA.unique_name)) << "didn't find blockA";
        expect(names.contains(blockB.unique_name)) << "didn't find blockB";
        expect(!names.contains(blockC.unique_name)) << "found blockC though nothing is connected to it";
        expect(!names.contains(blockD.unique_name)) << "isolated node should not be in adjacency list";
    };

    "print topologies"_test = [] {
        auto runTest = [](std::string name, gr::Graph&& graph) {
            for (auto& loop : gr::graph::detectFeedbackLoops(graph)) {
                gr::graph::colour(loop.edges.back(), gr::utf8::color::palette::Default::Cyan); // colour feedback edges
            }
            std::println("{}:\n{}", name, gr::graph::draw(graph));

            gr::scheduler::Simple<> sched;
            if (auto ret = sched.exchange(std::move(graph)); !ret) {
                expect(false) << std::format("couldn't initialise scheduler {}. error: {}", name, ret.error()) << fatal;
            }
            expect(sched.runAndWait().has_value());
        };

        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        runTest("getGraphLinear():\n", getGraphLinear(trace));
        runTest("getGraphParallel():\n", getGraphParallel(trace));
        runTest("getGraphScaledSum():\n", getGraphScaledSum(trace));
        runTest("getBasicFeedBackLoop():\n", getBasicFeedBackLoop(trace));
        runTest("getResamplingFeedbackLoop():\n", getResamplingFeedbackLoop(trace));
        runTest("getMultipleNestedFeedbackLoops():\n", getMultipleNestedFeedbackLoops(trace));
        runTest("getIIRFormII():\n", getIIRFormII(trace));
    };

    // TODO: add flatten test for nested graph once they are fully integrated by Ivan & Dantti

    std::println("N.B. test-suite finished");
};

int main() { /* tests are statically executed */ }
