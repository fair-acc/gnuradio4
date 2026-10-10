#include "message_utils.hpp"

#include <boost/ut.hpp>

#include <gnuradio-4.0/Message.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <gnuradio-4.0/testing/NullSources.hpp>

#include <gnuradio-4.0/algorithm/ImGraph.hpp>

#include <array>
#include <chrono>
#include <mutex>

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

    expect(flow.connect<"scaled", "in">(scaleBlock2, sink).has_value());
    expect(flow.connect<"scaled", "original">(scaleBlock1, scaleBlock2).has_value());
    expect(flow.connect<"out", "original">(source1, scaleBlock1).has_value());

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

    expect(flow.connect<"scaled", "original">(scaleBlock1a, scaleBlock2a).has_value());
    expect(flow.connect<"scaled", "original">(scaleBlock1b, scaleBlock2b).has_value());
    expect(flow.connect<"scaled", "in">(scaleBlock2b, sinkB).has_value());
    expect(flow.connect<"out", "original">(source1, scaleBlock1a).has_value());
    expect(flow.connect<"scaled", "in">(scaleBlock2a, sinkA).has_value());
    expect(flow.connect<"out", "original">(source1, scaleBlock1b).has_value());

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

    expect(flow.connect<"out", "original">(source1, scaleBlock).has_value(), loc);
    expect(flow.connect<"scaled", "addend0">(scaleBlock, addBlock).has_value(), loc);
    expect(flow.connect<"out", "addend1">(source2, addBlock).has_value(), loc);
    expect(flow.connect<"sum", "in">(addBlock, sink).has_value(), loc);

    return flow;
}

gr::Graph getBasicFeedBackLoop(std::shared_ptr<Tracer> tracer, std::source_location loc = std::source_location()) {
    using namespace boost::ut;

    gr::Size_t       nMaxSamples{2};
    gr::property_map layout_auto{{"layout_pref", "auto"}};

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

    expect(flow.connect<"out", "original">(source1, scale1).has_value(), loc);
    expect(flow.connect<"scaled", "addend0">(scale1, sum).has_value(), loc);
    expect(flow.connect<"sum", "in">(sum, sink).has_value(), loc);

    expect(flow.connect<"sum", "original">(sum, scale2).has_value(), loc);
    expect(flow.connect<"scaled", "addend1">(scale2, sum).has_value(), loc);

    return flow;
}

gr::Graph getResamplingFeedbackLoop(std::shared_ptr<Tracer> tracer, std::source_location loc = std::source_location::current()) {
    using namespace boost::ut;

    gr::Size_t       nMaxSamples{10};
    gr::Size_t       ratio{5};
    gr::property_map layout_auto{{"layout_pref", "auto"}};

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
    expect(flow.connect<"out", "addend0">(source, adder).has_value(), loc);
    expect(flow.connect<"sum", "in">(adder, decimator).has_value(), loc);
    expect(flow.connect<"out", "in">(decimator, sink).has_value(), loc);

    expect(flow.connect<"out", "in">(decimator, interpolator).has_value(), loc);
    expect(flow.connect<"out", "addend1">(interpolator, adder).has_value(), loc);

    return flow;
}

gr::Graph getMultipleNestedFeedbackLoops(std::shared_ptr<Tracer> tracer, std::source_location loc = std::source_location::current()) {
    using namespace boost::ut;

    gr::Size_t       nMaxSamples{2};
    gr::property_map layout_auto{{"layout_pref", "auto"}};

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
    expect(flow.connect<"out", "original">(source, scale1).has_value(), loc);
    expect(flow.connect<"scaled", "addend0">(scale1, adder1).has_value(), loc);
    expect(flow.connect<"sum", "original">(adder1, scale3).has_value(), loc);
    expect(flow.connect<"scaled", "addend0">(scale3, adder2).has_value(), loc);
    expect(flow.connect<"sum", "in">(adder2, sink).has_value(), loc);

    expect(flow.connect<"sum", "original">(adder1, scale2).has_value(), loc);
    expect(flow.connect<"scaled", "addend1">(scale2, adder1).has_value(), loc);

    expect(flow.connect<"sum", "original">(adder2, scale4).has_value(), loc);
    expect(flow.connect<"scaled", "addend1">(scale4, adder2).has_value(), loc);

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
    expect(flow.connect<"out", "addend0">(source, feedbackSum0).has_value(), loc); // src -> feedbackSum0
    expect(flow.connect<"sum", "original">(feedbackSum0, b0).has_value(), loc);    // b0 * v(n)
    expect(flow.connect<"scaled", "addend0">(b0, outputSum0).has_value(), loc);    // b0 -> outputSum0
    expect(flow.connect<"sum", "in">(outputSum0, sink).has_value(), loc);          // outputSum0 -> snk

    // delay line: v(n) → v(n-1) → v(n-2) → v(n-3)
    expect(flow.connect<"sum", "original">(feedbackSum0, d1).has_value(), loc);
    expect(flow.connect<"scaled", "original">(d1, d2).has_value(), loc);
    expect(flow.connect<"scaled", "original">(d2, d3).has_value(), loc);

    // feedback path
    expect(flow.connect<"scaled", "original">(d1, a1).has_value(), loc); // -a1 * v(n-1)
    expect(flow.connect<"scaled", "original">(d2, a2).has_value(), loc); // -a2 * v(n-2)
    expect(flow.connect<"scaled", "original">(d3, a3).has_value(), loc); // -a3 * v(n-3)

    // cascaded feedback summation: a3 + a2 -> feedbackSum2, then + a1 -> feedbackSum1
    expect(flow.connect<"scaled", "addend0">(a2, feedbackSum2).has_value(), loc);
    expect(flow.connect<"scaled", "addend1">(a3, feedbackSum2).has_value(), loc);
    expect(flow.connect<"scaled", "addend0">(a1, feedbackSum1).has_value(), loc);
    expect(flow.connect<"sum", "addend1">(feedbackSum2, feedbackSum1).has_value(), loc);
    expect(flow.connect<"sum", "addend1">(feedbackSum1, feedbackSum0).has_value(), loc);

    // feed-forward path
    expect(flow.connect<"scaled", "original">(d1, b1).has_value(), loc); // b1 * v(n-1)
    expect(flow.connect<"scaled", "original">(d2, b2).has_value(), loc); // b2 * v(n-2)
    expect(flow.connect<"scaled", "original">(d3, b3).has_value(), loc); // b3 * v(n-3)

    // cascaded feed-forward summation: b3 + b2 -> outputSum1, then + b1 -> outputSum2
    expect(flow.connect<"scaled", "addend0">(b2, outputSum1).has_value(), loc);      // FIXED: b2 -> addend0
    expect(flow.connect<"scaled", "addend1">(b3, outputSum1).has_value(), loc);      // b3 -> addend1
    expect(flow.connect<"scaled", "addend0">(b1, outputSum2).has_value(), loc);      // FIXED: b1 -> addend0
    expect(flow.connect<"sum", "addend1">(outputSum1, outputSum2).has_value(), loc); // outputSum1 -> addend1
    expect(flow.connect<"sum", "addend1">(outputSum2, outputSum0).has_value(), loc); // FIXED: complete chain to outputSum0

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

template<typename T>
struct IdleSource : public gr::Block<IdleSource<T>> {
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(IdleSource, out);

    gr::Sequence pollCount{0};

    [[nodiscard]] gr::work::Status processBulk(gr::OutputSpanLike auto& output) noexcept {
        pollCount.incrementAndGet();
        output.publish(0UZ);
        return gr::work::Status::OK;
    }
};

struct NotifyPerSample : gr::Block<NotifyPerSample> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;
    gr::Size_t         nNotified = 0U;

    GR_MAKE_REFLECTABLE(NotifyPerSample, in, out);

    [[nodiscard]] float processOne(float value) {
        gr::sendMessage<gr::message::Command::Notify>(this->msgOut, "", "qa_progress", gr::property_map{});
        gr::atomic_ref(nNotified).fetch_add(1U);
        return value;
    }
};

bool awaitIdleCpuPool() {
    return gr::testing::awaitCondition(std::chrono::seconds(4), [] { return gr::thread_pool::Manager::defaultCpuPool()->numTasksRunning() == 0UZ; });
}

struct ErrorOnFirstSample : gr::Block<ErrorOnFirstSample> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    GR_MAKE_REFLECTABLE(ErrorOnFirstSample, in, out);

    bool _reported = false;

    [[nodiscard]] float processOne(float value) {
        if (!std::exchange(_reported, true)) {
            this->emitErrorMessage("qa", "deliberate child error");
        }
        return value;
    }
};

struct FailingWork : gr::Block<FailingWork> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    GR_MAKE_REFLECTABLE(FailingWork, in, out);

    [[nodiscard]] gr::work::Status processBulk(gr::InputSpanLike auto& input, gr::OutputSpanLike auto& output) {
        std::ignore = input.consume(0UZ);
        output.publish(0UZ);
        return gr::work::Status::ERROR;
    }
};

struct StopWitness {
    bool stopped                 = false;
    bool destroyed               = false;
    bool stoppedAfterDestruction = false;
};

struct StopRecorder : gr::Block<StopRecorder> {
    gr::PortIn<float> in;
    StopWitness*      witness = nullptr;

    GR_MAKE_REFLECTABLE(StopRecorder, in);

    ~StopRecorder() { witness->destroyed = true; }

    void stop() {
        witness->stoppedAfterDestruction = witness->destroyed;
        witness->stopped                 = true;
    }

    constexpr void processOne(float) const noexcept {}
};

struct SampleCounter : gr::Block<SampleCounter> {
    gr::PortIn<float> in;
    std::size_t       nSamples = 0UZ;

    GR_MAKE_REFLECTABLE(SampleCounter, in);

    void processOne(float) { ++nSamples; }
};

struct HoldSecondTask : gr::thread_pool::TaskExecutor { // runs every task on its own thread, the second only once released
    std::atomic<std::size_t>  submitted{0UZ};
    std::atomic<std::size_t>  finished{0UZ};
    std::atomic<bool>         released{false};
    std::mutex                threadsMutex;
    std::vector<std::jthread> threads;

    void execute(gr::thread_pool::detail::move_only_function&& task) override {
        const std::size_t index = submitted.fetch_add(1UZ);
        std::lock_guard   lock(threadsMutex);
        threads.emplace_back([this, index, job = std::move(task)]() mutable {
            if (index == 1UZ) {
                released.wait(false);
            }
            job();
            finished.fetch_add(1UZ);
        });
    }

    void release() {
        released = true;
        released.notify_all();
    }

    [[nodiscard]] gr::thread_pool::TaskType     type() const noexcept override { return gr::thread_pool::TaskType::CPU_BOUND; }
    [[nodiscard]] std::string_view              name() const noexcept override { return "qa_hold_second"; }
    [[nodiscard]] std::string_view              device() const noexcept override { return "CPU"; }
    [[nodiscard]] std::size_t                   numThreads() const override { return 2UZ; }
    [[nodiscard]] std::size_t                   numTasksQueued() const override { return 0UZ; }
    [[nodiscard]] std::size_t                   numTasksRunning() const override { return 0UZ; }
    [[nodiscard]] std::size_t                   numTasksRecycled() const override { return 0UZ; }
    void                                        setThreadBounds(uint32_t, uint32_t) override {}
    [[nodiscard]] std::pair<uint32_t, uint32_t> threadBounds() const override { return {2U, 2U}; }
    [[nodiscard]] uint32_t                      minThreads() const override { return 2U; }
    [[nodiscard]] uint32_t                      maxThreads() const override { return 2U; }
    void                                        requestShutdown() override {}
    [[nodiscard]] bool                          isShutdown() const override { return false; }
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

    "poolName setting swaps to a registered pool"_test = [] {
        using namespace gr::thread_pool;
        static constexpr std::string_view kAltPoolName = "qa_alt_pool";

        auto altPool = std::make_shared<ThreadPoolWrapper>(std::make_unique<BasicThreadPool>(std::string(kAltPoolName), TaskType::CPU_BOUND, 1U, 1U), "CPU");
        Manager::instance().replacePool(std::string(kAltPoolName), std::move(altPool));

        gr::scheduler::Simple<> sched;
        gr::MsgPortIn           fromScheduler;
        expect(sched.msgOut.connect(fromScheduler).has_value());

        std::ignore = sched.settings().set({{"poolName", kAltPoolName}});
        std::ignore = sched.settings().activateContext();
        std::ignore = sched.settings().applyStagedParameters();

        expect(eq(std::string_view(sched.poolName.value), kAltPoolName)) << "poolName reflectable should reflect the new value";

        for (const auto& msg : gr::testing::consumeAllReplyMessages(fromScheduler)) {
            expect(msg.data.has_value()) << std::format("unexpected error message on swap to known pool: endpoint='{}' error='{}'", msg.endpoint, msg.data.has_value() ? "" : msg.data.error().message);
        }
    };

    "poolName setting with unknown pool emits error"_test = [] {
        gr::scheduler::Simple<> sched;
        gr::MsgPortIn           fromScheduler;
        expect(sched.msgOut.connect(fromScheduler).has_value());

        std::ignore = gr::testing::consumeAllReplyMessages(fromScheduler); // discard any setup-time messages

        static constexpr std::string_view kBogusPoolName = "qa_definitely_not_a_pool_xyz";
        std::ignore                                      = sched.settings().set({{"poolName", kBogusPoolName}});
        std::ignore                                      = sched.settings().activateContext();
        std::ignore                                      = sched.settings().applyStagedParameters();

        const auto messages     = gr::testing::consumeAllReplyMessages(fromScheduler);
        bool       sawPoolError = false;
        for (const auto& msg : messages) {
            if (msg.endpoint == "settingsChanged(poolName)" && !msg.data.has_value()) {
                expect(msg.data.error().message.find(kBogusPoolName) != std::string::npos) << "error message should mention the rejected pool name";
                sawPoolError = true;
            }
        }
        expect(sawPoolError) << "expected error notification for unknown pool";
    };
};

const boost::ut::suite<"SchedulerExchange"> SchedulerExchangeTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::testing;
    using namespace std::chrono_literals;

    "exchange() on a running single-threaded scheduler reports instead of blocking"_test = [] {
        Graph flow;
        auto& source = flow.emplaceBlock<NullSource<float>>();
        auto& sink   = flow.emplaceBlock<NullSink<float>>();
        expect(flow.connect<"out", "in">(source, sink).has_value()) << fatal;

        scheduler::Simple<scheduler::ExecutionPolicy::singleThreadedBlocking> scheduler;
        expect(scheduler.exchange(std::move(flow)).has_value()) << fatal;
        scheduler.timeout_ms = 50U;

        auto schedulerThreadHandle = gr::test::thread_pool::executeScheduler("qa_Sched::exchange", scheduler);
        expect(awaitCondition(scheduler, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << fatal << "scheduler up and running";
        scheduler.blockUntilWorking(); // RUNNING is published before start() finishes touching the graph.

        Graph replacement;
        auto& otherSource = replacement.emplaceBlock<NullSource<float>>();
        auto& otherSink   = replacement.emplaceBlock<NullSink<float>>();
        expect(replacement.connect<"out", "in">(otherSource, otherSink).has_value()) << fatal;

        const auto exchanged = scheduler.exchange(std::move(replacement));
        expect(!exchanged.has_value()) << "start() drives the loop on the calling thread, so restoring RUNNING here would never return";

        scheduler.requestStop();
        std::ignore = schedulerThreadHandle.get();
    };

    "a scheduler in ERROR stops its blocks before destroying them, and a further stop request is a no-op"_test = [] {
        StopWitness witness;
        {
            Graph flow;
            auto& source     = flow.emplaceBlock<NullSource<float>>();
            auto& failing    = flow.emplaceBlock<FailingWork>();
            auto& recorder   = flow.emplaceBlock<StopRecorder>();
            recorder.witness = &witness;
            expect(flow.connect<"out", "in">(source, failing).has_value()) << fatal;
            expect(flow.connect<"out", "in">(failing, recorder).has_value()) << fatal;

            scheduler::Simple<scheduler::ExecutionPolicy::singleThreaded> scheduler;
            expect(scheduler.exchange(std::move(flow)).has_value()) << fatal;
            std::ignore = scheduler.runAndWait();
            expect(eq(scheduler.state(), lifecycle::State::ERROR));
            expect(witness.stopped) << "blocks are stopped when the scheduler enters ERROR";
            expect(eq(recorder.state(), lifecycle::State::STOPPED));

            scheduler.requestStop();
            expect(eq(scheduler.state(), lifecycle::State::ERROR));
            expect(scheduler.changeStateTo(lifecycle::State::REQUESTED_STOP).has_value());
        }
        expect(witness.destroyed);
        expect(!witness.stoppedAfterDestruction) << "stop() must never run on a destroyed block";
    };

    "a block with an unconnected output releases the source it shares with another branch"_test = []<typename TPolicy> {
        constexpr gr::Size_t kSamplesBeyondRing = 1U << 20U;
        Graph                flow;
        auto&                source   = flow.emplaceBlock<ConstantSource<float>>({{"n_samples_max", kSamplesBeyondRing}});
        auto&                dangling = flow.emplaceBlock<Copy<float>>();
        auto&                counter  = flow.emplaceBlock<SampleCounter>();
        expect(flow.connect<"out", "in">(source, dangling).has_value()) << fatal;
        expect(flow.connect<"out", "in">(source, counter).has_value()) << fatal;

        scheduler::Simple<TPolicy::value> scheduler;
        expect(scheduler.exchange(std::move(flow)).has_value()) << fatal;
        expect(scheduler.runAndWait().has_value());
        expect(eq(counter.nSamples, static_cast<std::size_t>(kSamplesBeyondRing)));
        expect(!dangling.in.isConnected());
    } | std::tuple<std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::singleThreaded>, std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::multiThreaded>>{};
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

    "an idle graph keeps multiThreaded workers polling and parks multiThreadedBlocking workers"_test = []<typename TPolicy> {
        using namespace gr;
        using namespace gr::testing;
        constexpr std::size_t kInactivityCount = 5UZ;
        constexpr std::size_t kWatchdogBumps   = 3UZ;

        Graph flow;
        auto& source = flow.emplaceBlock<IdleSource<float>>();
        auto& sink   = flow.emplaceBlock<NullSink<float>>();
        expect(flow.connect<"out", "in">(source, sink).has_value());

        scheduler::Simple<TPolicy::value> sched;
        if (auto ret = sched.exchange(std::move(flow)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        sched.timeout_inactivity_count = static_cast<gr::Size_t>(kInactivityCount);
        sched.watchdog_timeout         = 20U;

        const Sequence&   progress        = sched.graph().progress();
        const std::size_t progressAtStart = progress.value();
        auto              schedulerHandle = gr::test::thread_pool::executeScheduler("qa_Sched::idle", sched);
        for (std::size_t bump = 0UZ; bump < kWatchdogBumps; ++bump) { // an idle graph only advances through the watchdog
            progress.wait(progress.value());
        }
        const std::size_t polls   = source.pollCount.value();
        const std::size_t wakeUps = progress.value() - progressAtStart;
        sched.requestStop();
        expect(schedulerHandle.get().has_value());

        const std::size_t blockingPollBound = (kInactivityCount + 2UZ) * (wakeUps + 1UZ);
        if constexpr (TPolicy::value == scheduler::ExecutionPolicy::multiThreadedBlocking) {
            expect(le(polls, blockingPollBound)) << std::format("{} polls for {} progress changes", polls, wakeUps);
        } else {
            expect(gt(polls, blockingPollBound)) << std::format("{} polls for {} progress changes", polls, wakeUps);
        }
    } | std::tuple<std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::multiThreaded>, std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::multiThreadedBlocking>>{};

    "a message and a stop wake the parked workers of a blocking scheduler"_test = []<typename TPolicy> {
        using namespace gr;
        using namespace gr::testing;
        using namespace gr::message;
        using enum lifecycle::State;

        Graph flow;
        auto& source = flow.emplaceBlock<IdleSource<float>>();
        auto& sink   = flow.emplaceBlock<NullSink<float>>();
        expect(flow.connect<"out", "in">(source, sink).has_value());

        MsgPortOut                        toScheduler;
        MsgPortIn                         fromScheduler;
        scheduler::Simple<TPolicy::value> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(toScheduler.connect(sched.msgIn).has_value());
        expect(sched.msgOut.connect(fromScheduler).has_value());
        sched.timeout_inactivity_count = 2U;
        sched.watchdog_timeout         = 10'000U; // ms, far beyond the wake-up bounds below

        auto schedulerDone = gr::test::thread_pool::executeScheduler("qa_Sched::wake", sched);
        expect(awaitCondition(4s, [&sched] { return sched.state() == RUNNING; })) << fatal;
        sched.blockUntilWorking();

        std::size_t lastPolls  = source.pollCount.value();
        std::size_t quietPolls = 0UZ;
        expect(awaitCondition(4s, [&] {
            const std::size_t polls = source.pollCount.value();
            quietPolls              = polls == lastPolls ? quietPolls + 1UZ : 0UZ;
            lastPolls               = polls;
            return quietPolls >= 20UZ;
        })) << fatal;

        sendMessage<Command::Get>(toScheduler, sched.unique_name, block::property::kSetting, {});
        expect(awaitCondition(2s, [&fromScheduler] { return fromScheduler.streamReader().available() > 0UZ; })) << "a Get to a parked scheduler is answered only by the watchdog";

        expect(sched.changeStateTo(REQUESTED_STOP).has_value());
        expect(schedulerDone.wait_for(2s) == std::future_status::ready) << "parked workers leave only with the watchdog";
        expect(schedulerDone.get().has_value());
    } | std::tuple<std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::singleThreadedBlocking>, std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::multiThreadedBlocking>>{};

    "a scheduler restarted from another thread right after STOPPED runs until the next stop"_test =
        []<typename TPolicy> {
            using namespace gr;
            using namespace gr::testing;
            using namespace gr::message;
            using enum lifecycle::State;
            constexpr std::size_t kCycles = 50UZ;

            Graph flow;
            auto& source = flow.emplaceBlock<NullSource<float>>();
            auto& copy1  = flow.emplaceBlock<Copy<float>>();
            auto& copy2  = flow.emplaceBlock<Copy<float>>();
            auto& sink   = flow.emplaceBlock<AtomicCountingSink<float>>();
            expect(flow.connect<"out", "in">(source, copy1).has_value());
            expect(flow.connect<"out", "in">(copy1, copy2).has_value());
            expect(flow.connect<"out", "in">(copy2, sink).has_value());

            MsgPortOut                        toScheduler;
            scheduler::Simple<TPolicy::value> sched;
            expect(sched.exchange(std::move(flow)).has_value());
            expect(toScheduler.connect(sched.msgIn).has_value());
            auto requestStop = [&toScheduler, &sched] { sendMessage<Command::Set>(toScheduler, sched.unique_name, block::property::kLifeCycleState, {{"state", std::string(gr::meta::enumName(REQUESTED_STOP).value_or(""))}}); };

            std::vector<std::future<std::expected<void, Error>>> runs; // a single-threaded run lives in the thread that started it
            runs.push_back(gr::test::thread_pool::executeScheduler("qa_Sched::restart", sched));
            for (std::size_t cycle = 0UZ; cycle < kCycles; ++cycle) {
                expect(awaitCondition(4s, [&sched, &sink] { return sched.state() == RUNNING && sink.loadCount() > 0U; })) << std::format("cycle {}: the restarted run does not process", cycle) << fatal;
                requestStop();
                expect(awaitCondition(4s, [&sched] { return sched.state() == STOPPED; })) << std::format("cycle {}: not STOPPED", cycle) << fatal;

                runs.push_back(gr::test::thread_pool::execute("qa_Sched::restart", [&sched] -> std::expected<void, Error> {
                    if (auto initialised = sched.changeStateTo(INITIALISED); !initialised) {
                        return initialised;
                    }
                    return sched.changeStateTo(RUNNING);
                }));
            }
            expect(awaitCondition(4s, [&sched, &sink] { return sched.state() == RUNNING && sink.loadCount() > 0U; })) << fatal;
            requestStop();
            expect(awaitCondition(4s, [&sched] { return sched.state() == STOPPED; })) << fatal;
            sched.waitDone();
            for (std::size_t run = 0UZ; run < runs.size(); ++run) {
                expect(runs[run].get().has_value()) << std::format("run {}", run);
            }
        } |
        std::tuple<std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::singleThreaded>, std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::singleThreadedBlocking>, //
            std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::multiThreaded>, std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::multiThreadedBlocking>>{};

    "a scheduler whose message client stopped reading still stops"_test = []<typename TPolicy> {
        using namespace gr;
        using namespace gr::testing;
        using enum lifecycle::State;

        Graph flow;
        auto& source   = flow.emplaceBlock<NullSource<float>>();
        auto& notifier = flow.emplaceBlock<NotifyPerSample>();
        auto& sink     = flow.emplaceBlock<NullSink<float>>();
        expect(flow.connect<"out", "in">(source, notifier).has_value());
        expect(flow.connect<"out", "in">(notifier, sink).has_value());

        scheduler::Simple<TPolicy::value> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        MsgPortIn neverRead;
        expect(sched.msgOut.connect(neverRead).has_value());
        const std::size_t kMessageCapacity = sched.msgOut.buffer().streamBuffer.size();

        auto run = gr::test::thread_pool::executeScheduler("qa_Sched::unread", sched);
        expect(awaitCondition(10s, [&notifier, kMessageCapacity] { return gr::atomic_ref(notifier.nNotified).load_acquire() > 4UZ * kMessageCapacity; })) << "the notifying block stalls once the unread message buffers are full";
        expect(sched.changeStateTo(REQUESTED_STOP).has_value());
        expect(run.wait_for(4s) == std::future_status::ready) << "the run does not stop while its message client is not reading" << fatal;
        expect(run.get().has_value());
    } | std::tuple<std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::multiThreaded>, std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::multiThreadedBlocking>>{};

    "concurrent scheduler runs share one watchdog task"_test = [] {
        using namespace gr;
        using namespace gr::testing;
        using enum lifecycle::State;
        using TScheduler = scheduler::Simple<scheduler::ExecutionPolicy::multiThreaded>;

        auto       io            = gr::thread_pool::Manager::defaultIoPool();
        auto       ioTasks       = [&io] { return io->numTasksQueued() + io->numTasksRunning(); };
        const auto ioTasksAtRest = ioTasks();

        std::array<TScheduler, 3UZ> schedulers;
        for (auto& sched : schedulers) {
            Graph flow;
            auto& source = flow.emplaceBlock<NullSource<float>>();
            auto& sink   = flow.emplaceBlock<NullSink<float>>();
            expect(flow.connect<"out", "in">(source, sink).has_value());
            expect(sched.exchange(std::move(flow)).has_value());
            expect(sched.changeStateTo(INITIALISED).has_value());
            expect(sched.changeStateTo(RUNNING).has_value());
        }
        expect(le(ioTasks(), ioTasksAtRest + 1UZ)) << "each run holds its own watchdog task on default_io";

        for (auto& sched : schedulers) {
            expect(sched.changeStateTo(REQUESTED_STOP).has_value());
        }
        for (auto& sched : schedulers) {
            sched.waitDone();
        }
    };

    "an idle multi-threaded blocking run holds a single pool thread"_test = [] {
        using namespace gr;
        using namespace gr::testing;
        using enum lifecycle::State;
        constexpr std::string_view kPoolName = "qa_idle_threads";
        if (!gr::thread_pool::Manager::instance().get(kPoolName)) {
            auto pool = std::make_shared<gr::thread_pool::ThreadPoolWrapper>(std::make_unique<gr::thread_pool::BasicThreadPool>(std::string(kPoolName), gr::thread_pool::TaskType::CPU_BOUND, 8U, 8U), "CPU");
            expect(gr::thread_pool::Manager::instance().registerPool(std::string(kPoolName), std::move(pool)).has_value()) << fatal;
        }
        auto pool = gr::thread_pool::Manager::instance().get(kPoolName).value();

        Graph flow;
        auto& source = flow.emplaceBlock<IdleSource<float>>();
        auto& copy1  = flow.emplaceBlock<Copy<float>>();
        auto& copy2  = flow.emplaceBlock<Copy<float>>();
        auto& sink   = flow.emplaceBlock<NullSink<float>>();
        expect(flow.connect<"out", "in">(source, copy1).has_value());
        expect(flow.connect<"out", "in">(copy1, copy2).has_value());
        expect(flow.connect<"out", "in">(copy2, sink).has_value());

        scheduler::Simple<scheduler::ExecutionPolicy::multiThreadedBlocking> sched;
        sched.poolName                 = std::string(kPoolName);
        sched.timeout_inactivity_count = 2U;
        sched.watchdog_timeout         = 10'000U; // ms, so no watchdog bump resumes the idle workers during the check
        expect(sched.exchange(std::move(flow)).has_value());

        std::atomic<bool> runEnded{false};
        std::jthread      run([&sched, &runEnded] {
            std::ignore = sched.runAndWait();
            runEnded    = true;
        });
        expect(awaitCondition(4s, [&sched] { return sched.state() == RUNNING; })) << fatal;
        expect(awaitCondition(4s, [&pool] { return pool->numTasksRunning() == 1UZ; })) << std::format("an idle run holds {} pool threads", pool->numTasksRunning());

        expect(sched.changeStateTo(REQUESTED_STOP).has_value());
        expect(awaitCondition(4s, [&runEnded] { return runEnded.load(); })) << "the run does not end while its workers are handed back" << fatal;
        expect(!sched.isProcessing());
    };

    "stopping and destroying a scheduler does not wait out its timeout_ms"_test = []<typename TPolicy> {
        using namespace gr;
        using namespace gr::testing;
        using enum lifecycle::State;
        constexpr std::size_t kCycles = 3UZ;

        const auto begin = std::chrono::steady_clock::now();
        for (std::size_t cycle = 0UZ; cycle < kCycles; ++cycle) {
            Graph flow;
            auto& source = flow.emplaceBlock<NullSource<float>>();
            auto& sink   = flow.emplaceBlock<NullSink<float>>();
            expect(flow.connect<"out", "in">(source, sink).has_value());

            scheduler::Simple<TPolicy::value> sched;
            sched.timeout_ms = 5'000U;
            expect(sched.exchange(std::move(flow)).has_value());
            expect(sched.changeStateTo(INITIALISED).has_value());
            expect(sched.changeStateTo(RUNNING).has_value());
            expect(sched.changeStateTo(REQUESTED_STOP).has_value());
        }
        expect(lt(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count(), 5000)) << "stop or destruction slept a full timeout_ms";
    } | std::tuple<std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::multiThreaded>, std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::multiThreadedBlocking>>{};

    "runAndWait() fails with a child error nobody listens to"_test = []<typename TPolicy> {
        using namespace gr;
        using namespace gr::testing;

        Graph flow;
        auto& source   = flow.emplaceBlock<ConstantSource<float>>({{"n_samples_max", gr::Size_t(10'000)}});
        auto& reporter = flow.emplaceBlock<ErrorOnFirstSample>();
        auto& sink     = flow.emplaceBlock<CountingSink<float>>();
        expect(flow.connect<"out", "in">(source, reporter).has_value());
        expect(flow.connect<"out", "in">(reporter, sink).has_value());

        scheduler::Simple<TPolicy::value> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        const std::expected<void, Error> result = sched.runAndWait();
        expect(!result.has_value()) << "the child error does not fail the run";
        expect(!result.has_value() && result.error().message.contains("deliberate child error")) << "the run fails with another error than the child's";
    } | std::tuple<std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::singleThreaded>, std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::multiThreadedBlocking>>{};

    "a started scheduler keeps running after a child error nobody listens to"_test = []<typename TPolicy> {
        using namespace gr;
        using namespace gr::testing;
        using enum lifecycle::State;

        Graph flow;
        auto& source   = flow.emplaceBlock<ConstantSource<float>>({{"n_samples_max", gr::Size_t(10'000)}});
        auto& reporter = flow.emplaceBlock<ErrorOnFirstSample>();
        auto& sink     = flow.emplaceBlock<AtomicCountingSink<float>>();
        expect(flow.connect<"out", "in">(source, reporter).has_value());
        expect(flow.connect<"out", "in">(reporter, sink).has_value());

        scheduler::Simple<TPolicy::value> sched;
        expect(sched.exchange(std::move(flow)).has_value());
        expect(sched.changeStateTo(INITIALISED).has_value());
        expect(sched.changeStateTo(RUNNING).has_value());
        expect(awaitCondition(4s, [&sink] { return sink.loadCount() == gr::Size_t(10'000); })) << "data stops flowing after the child error";
        expect(sched.state() != ERROR);
        expect(sched.changeStateTo(REQUESTED_STOP).has_value());
        expect(sched.waitDone());
    } | std::tuple<std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::multiThreaded>, std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::multiThreadedBlocking>>{};

    "a scheduler held through SchedulerModel reports its jobs and a progress that survives setGraph"_test = [] {
        using namespace gr;
        using namespace gr::testing;
        using enum lifecycle::State;

        auto makeFlow = [] {
            Graph flow;
            auto& source = flow.emplaceBlock<NullSource<float>>();
            auto& sink   = flow.emplaceBlock<NullSink<float>>();
            std::ignore  = flow.connect<"out", "in">(source, sink);
            return flow;
        };

        auto            wrapper = std::make_shared<SchedulerWrapper<scheduler::Simple<scheduler::ExecutionPolicy::multiThreadedBlocking>>>();
        SchedulerModel& model   = *wrapper;
        BlockModel&     block   = *wrapper->asBlockModel();
        model.setGraph(makeFlow());
        const std::shared_ptr<Sequence> firstProgress = model.progressHandle();
        expect(!model.isProcessing());

        expect(block.changeStateTo(INITIALISED).has_value());
        expect(block.changeStateTo(RUNNING).has_value());
        expect(model.isProcessing());
        expect(!model.waitDone(std::chrono::milliseconds(20))) << "returns done while the run is still going";
        firstProgress->incrementAndGet();
        firstProgress->notify_all();

        expect(block.changeStateTo(REQUESTED_STOP).has_value());
        expect(model.waitDone()) << "a stopped run's jobs never leave";
        expect(!model.isProcessing());

        model.setGraph(makeFlow());
        expect(model.progressHandle() != firstProgress) << "the handle does not follow the new graph";
        expect(ge(firstProgress->value(), 1UZ)) << "the old graph's progress is not kept alive by its handle";
    };

    "a worker still queued when its run stops is counted until it has left"_test = [] {
        using namespace gr;
        using namespace gr::testing;
        using enum lifecycle::State;

        auto hold = std::make_shared<HoldSecondTask>();
        expect(gr::thread_pool::Manager::instance().registerPool("qa_hold_second", hold).has_value()) << fatal;

        Graph flow;
        auto& source = flow.emplaceBlock<NullSource<float>>();
        auto& sink   = flow.emplaceBlock<NullSink<float>>();
        expect(flow.connect<"out", "in">(source, sink).has_value());

        scheduler::Simple<scheduler::ExecutionPolicy::multiThreaded> sched;
        sched.poolName = std::string("qa_hold_second");
        expect(sched.exchange(std::move(flow)).has_value()) << fatal;
        expect(sched.changeStateTo(INITIALISED).has_value());
        expect(sched.changeStateTo(RUNNING).has_value());
        expect(awaitCondition(4s, [&hold] { return hold->submitted.load() == 2UZ; })) << "two job lists, two workers" << fatal;

        expect(sched.changeStateTo(REQUESTED_STOP).has_value());
        expect(awaitCondition(4s, [&hold] { return hold->finished.load() == 1UZ; })) << fatal;
        expect(sched.isProcessing()) << "the held-back worker of the stopped run is not counted";

        hold->release();
        sched.waitDone();
        expect(!sched.isProcessing());
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

    // Regression test for issue #813: the multiThreaded schedulers size their job-lists by the pool's
    // maxThreads(). When runAndWait() is itself dispatched onto that same pool (as executeScheduler
    // does), it permanently holds one of the two pinned threads, leaving one thread for two job-lists.
    // One job-list then never gets a worker, its blocks never run, back-pressure stalls the rest, and
    // the graph never finishes. After the fix the scheduler sizes to the threads actually free.
    auto expectNoStarvationWhenPoolThreadBusy = [](auto schedulerTag, std::string_view schedulerName) {
        using TScheduler = typename decltype(schedulerTag)::type;

        std::shared_ptr<Tracer> trace = std::make_shared<Tracer>();
        TScheduler              sched;
        if (auto ret = sched.exchange(getGraphLinear(trace)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }

        auto       future   = gr::test::thread_pool::executeScheduler(std::format("qa_Sched::issue813::{}", schedulerName), sched);
        const bool finished = future.wait_for(20s) == std::future_status::ready;
        if (!finished) {
            sched.requestStop(); // break the starvation so the test process does not hang
        }
        const auto result = future.get();
        expect(finished) << std::format("{}: scheduler stalled, a job-list never started because a pool thread was busy (issue #813)", schedulerName);
        expect(result.has_value()) << [&] { return std::format("{}: runAndWait() did not return success: {}", schedulerName, result.error()); };

        const TraceVectorType       t = trace->getVector();
        const std::set<std::string> ranBlocks(t.begin(), t.end());
        expect(eq(ranBlocks.size(), 4UZ)) << std::format("{}: not all blocks ran; only [{}] did work, expected s1, mult1, mult2, out", schedulerName, gr::join(ranBlocks, ", "));
    };

    "Simple scheduler runs all work when a pool thread is busy elsewhere"_test = [&] { expectNoStarvationWhenPoolThreadBusy(std::type_identity<gr::scheduler::Simple<gr::scheduler::ExecutionPolicy::multiThreaded>>{}, "Simple"); };

    "BreadthFirst scheduler runs all work when a pool thread is busy elsewhere"_test = [&] { expectNoStarvationWhenPoolThreadBusy(std::type_identity<gr::scheduler::BreadthFirst<gr::scheduler::ExecutionPolicy::multiThreaded>>{}, "BreadthFirst"); };

    "DepthFirst scheduler runs all work when a pool thread is busy elsewhere"_test = [&] { expectNoStarvationWhenPoolThreadBusy(std::type_identity<gr::scheduler::DepthFirst<gr::scheduler::ExecutionPolicy::multiThreaded>>{}, "DepthFirst"); };

    "BreadthFirstScheduler_linear_multi_threaded"_test = [] {
        std::shared_ptr<Tracer>                                                    trace = std::make_shared<Tracer>();
        gr::scheduler::BreadthFirst<gr::scheduler::ExecutionPolicy::multiThreaded> sched;
        if (auto ret = sched.exchange(getGraphLinear(trace)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        expect(awaitIdleCpuPool()) << fatal;
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
        expect(awaitIdleCpuPool()) << fatal;
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
        expect(awaitIdleCpuPool()) << fatal;
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
        expect(flow.connect<"out", "in">(lifecycleSource, lifecycleBlock).has_value());

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
        expect(flow.connect<"out", "in">(source, monitor).has_value());
        expect(flow.connect<"out", "in">(monitor, sink).has_value());

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
        expect(flow.connect<"out", "in">(source, monitor).has_value());
        expect(flow.connect<"out", "in">(monitor, sink).has_value());

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
        expect(flow.connect<"out", "in">(source, monitor).has_value());
        expect(flow.connect<"out", "in">(monitor, sink).has_value());

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
        expect(flow.connect<"out", "in">(source, monitor).has_value());
        expect(flow.connect<"out", "in">(monitor, sink).has_value());

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
        expect(flow.connect<"out", "in">(source, monitor).has_value());
        expect(flow.connect<"out", "in">(monitor, sink).has_value());

        scheduler::Simple<scheduler::ExecutionPolicy::singleThreadedBlocking> scheduler;
        if (auto ret = scheduler.exchange(std::move(flow)); !ret) {
            expect(false) << std::format("couldn't initialise scheduler. error: {}", ret.error()) << fatal;
        }
        scheduler.timeout_ms               = 100U; // also dynamically settable via messages/block interface
        scheduler.timeout_inactivity_count = 10U;  // also dynamically settable via messages/block interface

        expect(eq(0UZ, scheduler.graph().progress().value())) << "initial progress definition (0)";

        auto schedulerThreadHandle = gr::test::thread_pool::executeScheduler("qa_Sched", scheduler);

        expect(awaitCondition(scheduler, [&scheduler] { return scheduler.state() == lifecycle::State::RUNNING; })) << "scheduler thread up and running w/ timeout";

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

        expect(graph.connect<"scaled", "original">(A, B).has_value());
        expect(graph.connect<"scaled", "original">(B, C).has_value());

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

        expect(graph.connect<"scaled", "original">(A, B).has_value());
        expect(graph.connect<"scaled", "original">(A, C).has_value());

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

        expect(graph.connect<"scaled", "original">(blockA, blockB).has_value());
        expect(graph.connect<"scaled", "original">(blockB, blockC).has_value());

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
