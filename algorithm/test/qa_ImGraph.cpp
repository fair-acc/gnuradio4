#include <boost/ut.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

#include <gnuradio-4.0/algorithm/ImGraph.hpp>

const boost::ut::suite<"ImCanvas"> _1 = [] { using namespace boost::ut; };

namespace std {
template<typename CharT>
struct formatter<std::byte, CharT> {
    constexpr auto parse(basic_format_parse_context<CharT>& ctx) { return ctx.begin(); }

    template<typename FormatContext>
    constexpr auto format(std::byte b, FormatContext& ctx) const {
        return std::format_to(ctx.out(), "0b{}", std::bitset<8>{std::to_integer<unsigned>(b)}.to_string());
    }
};
} // namespace std

//------------------------------------------------------
// five demos
//------------------------------------------------------

/**
 * use this block to mock different input-output scenarios
 */
template<typename T, std::size_t nInSync, std::size_t nInAsync, std::size_t nMsgIn, std::size_t nOutSync, std::size_t nOutAsync, std::size_t nMsgOut>
requires(std::is_arithmetic_v<T>)
struct GenericBlock : gr::Block<GenericBlock<T, nInSync, nInAsync, nMsgIn, nOutSync, nOutAsync, nMsgOut>> {
    std::array<gr::PortIn<T>, nInSync>             in1{};
    std::array<gr::PortIn<T, gr::Async>, nInAsync> in2{};
    std::array<gr::MsgPortIn, nMsgIn>              msgIns{};

    std::array<gr::PortOut<T>, nOutSync>             out1{};
    std::array<gr::PortOut<T, gr::Async>, nOutAsync> out2{};
    std::array<gr::MsgPortOut, nMsgOut>              msgOuts{};

    GR_MAKE_REFLECTABLE(GenericBlock, in1, in2, msgIns, out1, out2, msgOuts);

    gr::work::Status processBulk(auto /*ins1*/, auto /*ins2*/, auto /*outs1*/, auto /*outs2*/) { return gr::work::Status::OK; }
};

inline static gr::Graph example_base1() {
    using namespace boost::ut;

    gr::Graph graph;

    // Create blocks using emplaceBlock (similar to qa_Scheduler examples)
    auto& src1 = graph.emplaceBlock<GenericBlock<float, 0, 0, 0, 1, 0, 0>>({{"name", "src#1"}});
    auto& src2 = graph.emplaceBlock<GenericBlock<float, 0, 0, 0, 1, 0, 0>>({{"name", "src#2"}});
    auto& add  = graph.emplaceBlock<GenericBlock<float, 2, 0, 0, 1, 0, 0>>({{"name", "add"}});
    auto& snk1 = graph.emplaceBlock<GenericBlock<float, 1, 0, 0, 0, 0, 0>>({{"name", "snk#1"}});

    // Connect using the actual gr::Graph API
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(src1).to<"in1", 0UZ>(add)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(src2).to<"in1", 1UZ>(add)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(add).to<"in1", 0>(snk1)));

    return graph;
}

inline static gr::Graph example_base2() {
    using namespace boost::ut;

    gr::Graph graph;

    auto& src1  = graph.emplaceBlock<GenericBlock<float, 0, 0, 0, 0, 1, 0>>({{"name", "src#1"}});
    auto& split = graph.emplaceBlock<GenericBlock<float, 0, 1, 0, 0, 2, 0>>({{"name", "split"}});
    auto& snk1  = graph.emplaceBlock<GenericBlock<float, 0, 1, 0, 0, 0, 0>>({{"name", "snk#1"}});
    auto& snk2  = graph.emplaceBlock<GenericBlock<float, 0, 1, 0, 0, 0, 0>>({{"name", "snk#2"}});

    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out2", 0>(src1).to<"in2", 0>(split)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out2", 0>(split).to<"in2", 0>(snk1)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out2", 1>(split).to<"in2", 0>(snk2)));

    return graph;
}

inline static gr::Graph example_cyclic_A() {
    using namespace boost::ut;

    gr::Graph graph;

    auto& src1 = graph.emplaceBlock<GenericBlock<float, 0, 0, 0, 1, 0, 0>>({{"name", "src#1"}});
    auto& sum  = graph.emplaceBlock<GenericBlock<float, 1, 1, 0, 0, 1, 0>>({{"name", "Σ"}});
    auto& ds   = graph.emplaceBlock<GenericBlock<float, 1, 0, 0, 1, 0, 0>>({{"name", "D(s)"}});
    auto& gs   = graph.emplaceBlock<GenericBlock<float, 1, 0, 0, 1, 0, 0>>({{"name", "G(s)"}});
    auto& ms   = graph.emplaceBlock<GenericBlock<float, 0, 1, 0, 1, 0, 0>>({{"name", "M(s)"}});
    auto& snk1 = graph.emplaceBlock<GenericBlock<float, 1, 0, 0, 0, 0, 0>>({{"name", "snk#1"}});

    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(src1).to<"in1", 0>(sum)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out2", 0>(sum).to<"in1", 0>(ds)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(ds).to<"in1", 0>(gs)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(gs).to<"in1", 0>(snk1)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(gs).to<"in2", 0>(ms)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(ms).to<"in2", 0>(sum))); // feedback

    return graph;
}

inline static gr::Graph example_cyclic_B() {
    using namespace boost::ut;

    gr::Graph graph;

    auto&            src1 = graph.emplaceBlock<GenericBlock<float, 0, 0, 0, 1, 0, 0>>({{"name", "src#1"}});
    gr::property_map prop{{"layout_pref", "vertical"}};
    auto&            sum = graph.emplaceBlock<GenericBlock<float, 1, 1, 0, 0, 1, 0>>({{"name", "Σ"}, {"ui_constraints", prop}});
    // setLayoutPref(gr::graph::findBlock(graph, sum.unique_name).value(), LayoutPref::VERTICAL);

    auto& ds   = graph.emplaceBlock<GenericBlock<float, 1, 0, 0, 1, 0, 0>>({{"name", "D(s)"}});
    auto& gs   = graph.emplaceBlock<GenericBlock<float, 1, 0, 0, 1, 0, 0>>({{"name", "G(s)"}});
    auto& ms   = graph.emplaceBlock<GenericBlock<float, 0, 1, 0, 1, 0, 0>>({{"name", "M(s)"}});
    auto& snk1 = graph.emplaceBlock<GenericBlock<float, 1, 0, 0, 0, 0, 0>>({{"name", "snk#1"}});

    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(src1).to<"in1", 0>(sum)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out2", 0>(sum).to<"in1", 0>(ds)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(ds).to<"in1", 0>(gs)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(gs).to<"in1", 0>(snk1)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(gs).to<"in2", 0>(ms)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(ms).to<"in2", 0>(sum))); // feedback

    return graph;
}

inline static gr::Graph example_auto() {
    using namespace boost::ut;

    gr::Graph graph;

    gr::property_map prop{{"layout_pref", "auto"}};
    auto&            src1 = graph.emplaceBlock<GenericBlock<float, 0, 0, 0, 1, 0, 0>>({{"name", "src#1"}, {"ui_constraints", prop}});
    auto&            sum  = graph.emplaceBlock<GenericBlock<float, 2, 0, 0, 0, 1, 0>>({{"name", "Σ"}, {"ui_constraints", prop}});
    auto&            ds   = graph.emplaceBlock<GenericBlock<float, 1, 0, 0, 1, 0, 0>>({{"name", "D(s)"}, {"ui_constraints", prop}});
    auto&            gs   = graph.emplaceBlock<GenericBlock<float, 1, 0, 0, 1, 0, 0>>({{"name", "G(s)"}, {"ui_constraints", prop}});
    auto&            snk1 = graph.emplaceBlock<GenericBlock<float, 1, 0, 0, 0, 0, 0>>({{"name", "snk#1"}, {"ui_constraints", prop}});

    auto& ms = graph.emplaceBlock<GenericBlock<float, 0, 1, 0, 1, 0, 0>>({{"name", "M(s)"}});
    // setLayoutPref(std::shared_ptr<gr::BlockModel>(graph.findBlock(ms.unique_name).value()), LayoutPref::HORIZONTAL);

    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(src1).to<"in1", 0>(sum)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out2", 0>(sum).to<"in1", 0>(ds)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(ds).to<"in1", 0>(gs)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(gs).to<"in1", 0>(snk1)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(gs).to<"in2", 0>(ms)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(ms).to<"in1", 1>(sum))); // feedback

    return graph;
}

inline static gr::Graph example_large() {
    using namespace boost::ut;

    gr::Graph        graph;
    gr::property_map prop_auto{{"layout_pref", "auto"}};
    gr::property_map prop_ver{{"layout_pref", "vertical"}};
    gr::property_map prop_hor{{"layout_pref", "horizontal"}};

    // Create all blocks
    auto& src1 = graph.emplaceBlock<GenericBlock<float, 0, 0, 0, 1, 0, 0>>({{"name", "src#1"}});

    auto& sum = graph.emplaceBlock<GenericBlock<float, 1, 1, 0, 0, 1, 0>>({{"name", "Σ"}, {"ui_constraints", prop_auto}});
    // Set bottom side for second input port
    // auto sumBlock = graph.findBlock(sum.unique_name).value();
    // in(sumBlock, 1).preferredSide(Side::Bottom);

    auto& ds   = graph.emplaceBlock<GenericBlock<float, 1, 0, 0, 1, 0, 0>>({{"name", "D(s)"}});
    auto& gs   = graph.emplaceBlock<GenericBlock<float, 1, 0, 0, 1, 0, 0>>({{"name", "G(s)"}});
    auto& snk1 = graph.emplaceBlock<GenericBlock<float, 1, 0, 0, 0, 0, 0>>({{"name", "snk#1"}});

    auto& ms  = graph.emplaceBlock<GenericBlock<float, 0, 2, 0, 1, 1, 0>>({{"name", "M(s)"}});
    auto& m2s = graph.emplaceBlock<GenericBlock<float, 1, 0, 0, 1, 0, 0>>({{"name", "M2(s)"}, {"ui_constraints", prop_hor}});

    auto& src2 = graph.emplaceBlock<GenericBlock<float, 0, 0, 0, 1, 0, 0>>({{"name", "src#2"}});
    auto& snk2 = graph.emplaceBlock<GenericBlock<float, 1, 0, 0, 0, 0, 0>>({{"name", "snk#2"}});
    auto& snk3 = graph.emplaceBlock<GenericBlock<float, 1, 0, 0, 0, 0, 0>>({{"name", "snk#3"}});
    auto& m3s  = graph.emplaceBlock<GenericBlock<float, 0, 1, 0, 1, 0, 0>>({{"name", "M3(s)"}, {"ui_constraints", prop_auto}});

    // Main connections
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(src1).to<"in1", 0>(sum)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out2", 0>(sum).to<"in1", 0>(ds)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(ds).to<"in1", 0>(gs)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(gs).to<"in1", 0>(snk1)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(gs).to<"in2", 0>(ms)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(ms).to<"in2", 0>(sum))); // feedback

    // Micro feedback loop
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out2", 0>(ms).to<"in1", 0>(m2s)));
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(m2s).to<"in2", 1>(ms)));

    // Second sub-graph
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(src2, gr::undefined_size, gr::graph::defaultWeight, "special edge").to<"in2", 0>(m3s))); // a noteworthy edge
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(m3s).to<"in1", 0>(snk2)));

    // Additional connection
    expect(eq(gr::ConnectionResult::SUCCESS, graph.connect<"out1", 0>(ms).to<"in1", 0>(snk3)));

    // test custom block, edge, port colouring
    gr::graph::colour(gr::graph::findBlock(graph, sum.unique_name).value(), gr::utf8::color::palette::Default::BrightGreen);
    gr::graph::colour(gr::graph::findBlock(graph, src2.unique_name).value(), gr::utf8::color::palette::Default::BrightGreen);

    for (auto& loop : gr::graph::detectFeedbackLoops(graph)) {
        gr::graph::colour(loop.edges.back(), gr::utf8::color::palette::Default::Cyan); // colour feedback edges
    }
    constexpr gr::utf8::Style edgeStyle{.fg = gr::utf8::color::palette::Default::BrightYellow, .fgSet = true, .bold = true};
    gr::graph::style(gr::graph::findEdge(graph, "special edge").value(), edgeStyle); // colour arbitrary edge

    gr::utf8::Style portStyle{.fg = gr::utf8::color::palette::Default::Red, .fgSet = true, .bold = true, .blinkSlow = true};
    gr::graph::style<gr::PortDirection::OUTPUT>(gr::graph::findBlock(graph, m3s.unique_name).value(), {0UZ, 0UZ}, portStyle);

    portStyle.blinkSlow = false;
    gr::graph::style<gr::PortDirection::INPUT>(gr::graph::findBlock(graph, snk2.unique_name).value(), {0UZ, 0UZ}, portStyle);

    return graph;
}

int main() {
    std::println("=== Base example 1 ===\n{}", gr::graph::draw(example_base1().contents));
    std::println("=== Base example 2 ===\n{}", gr::graph::draw(example_base2().contents));
    std::println("=== Cyclic example A ===\n{}", gr::graph::draw(example_cyclic_A().contents));
    std::println("=== Cyclic example B (vertical) ===\n{}", gr::graph::draw(example_cyclic_B().contents));
    std::println("=== AUTO layout example ===\n{}", gr::graph::draw(example_auto().contents));
    std::println("=== Larger example ===\n{}", gr::graph::draw(example_large().contents));
    return 0;
}
