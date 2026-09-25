#include <boost/ut.hpp>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/test/EventMarbles.hpp>
#include <gnuradio-4.0/trigger/Marble.hpp>

using namespace gr::blocks::trigger;

const boost::ut::suite<"MarbleScript"> _marbleScore = [] {
    using namespace boost::ut;

    "a bare token is a sample with no tags"_test = [] {
        const auto score = MarbleScript::parse("a b c");
        expect(score.has_value());
        expect(eq(score->firstRow().size(), 3UZ));
        expect(eq(std::string_view{score->firstRow()[0].value}, std::string_view{"a"}));
        expect(eq(score->firstRow()[0].tags.size(), 0UZ));
    };

    "a tag symbol binds to the sample after the colon"_test = [] {
        const auto score = MarbleScript::parse("a T:b c");
        expect(score.has_value());
        expect(eq(score->firstRow().size(), 3UZ));
        expect(eq(std::string_view{score->firstRow()[1].value}, std::string_view{"b"}));
        expect(eq(score->firstRow()[1].tags.size(), 1UZ));
        expect(eq(std::string_view{score->firstRow()[1].tags[0]}, std::string_view{"T"}));
    };

    "several tags at one sample keep the order they were written"_test = [] {
        const auto score = MarbleScript::parse("T+U+V:x");
        expect(score.has_value());
        expect(eq(score->firstRow()[0].tags.size(), 3UZ));
        expect(eq(std::string_view{score->firstRow()[0].tags[0]}, std::string_view{"T"}));
        expect(eq(std::string_view{score->firstRow()[0].tags[2]}, std::string_view{"V"}));
    };

    "consecutive tagged samples stay separate"_test = [] {
        const auto score = MarbleScript::parse("T:a U:b");
        expect(score.has_value());
        expect(eq(score->firstRow().size(), 2UZ));
        expect(eq(std::string_view{score->firstRow()[0].tags[0]}, std::string_view{"T"}));
        expect(eq(std::string_view{score->firstRow()[1].tags[0]}, std::string_view{"U"}));
    };

    "a bar marks end-of-stream"_test = [] {
        const auto score = MarbleScript::parse("a |");
        expect(score.has_value());
        expect(eq(score->firstRow().size(), 2UZ));
        expect(score->firstRow()[1].endOfStream);
    };

    "an empty script yields no marbles"_test = [] {
        const auto score = MarbleScript::parse("   ");
        expect(score.has_value());
        expect(score->firstRow().empty());
    };

    "a token naming tags but no sample is refused"_test = [] {
        const auto score = MarbleScript::parse("T:");
        expect(!score.has_value()) << "a tag must attach to something";
    };

    "an empty tag symbol is refused"_test = [] {
        const auto score = MarbleScript::parse("T++U:a");
        expect(!score.has_value());
    };

    "more than one separator in a token is refused"_test = [] {
        const auto score = MarbleScript::parse("T:a:b");
        expect(!score.has_value());
    };

    "a sample carries as many tags as the script gives it"_test = [] {
        const auto score = MarbleScript::parse("A+B+C+D+E:x");
        expect(score.has_value());
        expect(eq(score->firstRow()[0].tags.size(), 5UZ)) << "a GR4 sample has no tag limit, so neither has the notation";
        expect(eq(std::string_view{score->firstRow()[0].tags[4]}, std::string_view{"E"}));
    };

    "a hash ends the stream in an error, with an optional reason"_test = [] {
        const auto bare = MarbleScript::parse("a #");
        expect(bare.has_value());
        expect(eq(bare->firstRow().size(), 2UZ));
        expect(bare->firstRow()[1].error);
        expect(bare->firstRow()[1].reason.empty());

        const auto explained = MarbleScript::parse("a #:overflow");
        expect(explained.has_value());
        expect(explained->firstRow()[1].error);
        expect(eq(std::string_view{explained->firstRow()[1].reason}, std::string_view{"overflow"}));
        expect(eq(std::string_view{explained->toString()}, std::string_view{"a #:overflow"})) << "and it spells back the way it was written";
    };

    "a hash carrying anything but a reason is refused"_test = [] { expect(!MarbleScript::parse("#oops").has_value()) << "a reason must follow the separator"; };

    "a line per stream, each one nameable"_test = [] {
        const auto score = MarbleScript::parse("@ch0 a b |\n@ch1 T:c #:lost");
        expect(score.has_value());
        expect(eq(score->rows.size(), 2UZ));
        expect(eq(std::string_view{score->rows[0].label}, std::string_view{"ch0"}));
        expect(eq(std::string_view{score->rows[1].label}, std::string_view{"ch1"}));
        expect(eq(score->firstRow().size(), 3UZ)) << "the first row is what a single-stream reader sees";
        expect(score->find("ch1") != nullptr);
        expect(score->find("ch2") == nullptr) << "a row that was never written cannot be found";
        expect(eq(std::string_view{score->toString()}, std::string_view{"@ch0 a b |\n@ch1 T:c #:lost"}));
    };

    "an unnamed row stays unnamed, and a label with no name is refused"_test = [] {
        const auto plain = MarbleScript::parse("a b\nc d");
        expect(plain.has_value());
        expect(eq(plain->rows.size(), 2UZ));
        expect(plain->rows[0].label.empty());
        expect(!MarbleScript::parse("@ a b").has_value());
    };

    "what was parsed spells back the way it was written"_test = [] {
        for (const std::string_view script : {"a b c", "a T:b c", "T+U:x", "a |"}) {
            const auto score = MarbleScript::parse(script);
            expect(score.has_value());
            expect(eq(std::string_view{score->toString()}, std::string_view(script))) << std::format("round trip of '{}'", script);
        }
    };
};

const boost::ut::suite<"MarbleSource and MarbleSink"> _marbleBlocks = [] {
    using namespace boost::ut;

    const gr::property_map values{{"a", 1.0f}, {"b", 2.0f}, {"c", 3.0f}};
    const gr::property_map tags{{"T", gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::string("start")}, {std::string(gr::tag::TRIGGER_TIME.key()), std::uint64_t{1U}}, {std::string(gr::tag::TRIGGER_OFFSET.key()), 0.f}}}};

    "a script reaches the sink as the samples it named"_test = [values, tags] {
        gr::Graph graph;
        auto&     source = graph.emplaceBlock<MarbleSource<float>>({{"script", std::string("a b c |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&     sink   = graph.emplaceBlock<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(graph.connect<"out", "in">(source, sink).has_value());

        gr::scheduler::Simple<> scheduler;
        expect(scheduler.exchange(std::move(graph)).has_value());
        expect(scheduler.runAndWait().has_value());

        expect(eq(sink._samples.size(), 3UZ));
        expect(eq(std::string_view{sink.script()}, std::string_view("a b c |"))) << "the terminator the stream actually ended with";
    };

    "a tag rides the sample the script attached it to"_test = [values, tags] {
        gr::Graph graph;
        auto&     source = graph.emplaceBlock<MarbleSource<float>>({{"script", std::string("a T:b c |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&     sink   = graph.emplaceBlock<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(graph.connect<"out", "in">(source, sink).has_value());

        gr::scheduler::Simple<> scheduler;
        expect(scheduler.exchange(std::move(graph)).has_value());
        expect(scheduler.runAndWait().has_value());

        expect(eq(sink._samples.size(), 3UZ));
        expect(eq(sink._tags.size(), 1UZ)) << "exactly the one tag the script named";
        if (!sink._tags.empty()) {
            expect(eq(sink._tags[0].first, 1UZ)) << "attached to the sample after the colon";
        }
        expect(eq(std::string_view{sink.script()}, std::string_view("a T:b c |")));
    };

    "repeat plays the script again"_test = [values, tags] {
        gr::Graph graph;
        auto&     source = graph.emplaceBlock<MarbleSource<float>>({{"script", std::string("a b")}, {"sample_values", values}, {"sample_tags", tags}, {"repeat", gr::Size_t{3U}}});
        auto&     sink   = graph.emplaceBlock<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(graph.connect<"out", "in">(source, sink).has_value());

        gr::scheduler::Simple<> scheduler;
        expect(scheduler.exchange(std::move(graph)).has_value());
        expect(scheduler.runAndWait().has_value());

        expect(eq(sink._samples.size(), 6UZ)) << "two samples, three times";
    };

    "an empty script produces nothing and still ends"_test = [values, tags] {
        gr::Graph graph;
        auto&     source = graph.emplaceBlock<MarbleSource<float>>({{"script", std::string("")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&     sink   = graph.emplaceBlock<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(graph.connect<"out", "in">(source, sink).has_value());

        gr::scheduler::Simple<> scheduler;
        expect(scheduler.exchange(std::move(graph)).has_value());
        expect(scheduler.runAndWait().has_value());
        expect(sink._samples.empty());
    };

    "a source plays the row it was asked for"_test = [values, tags] {
        const std::string script = "@low a b |\n@high c c |";
        gr::Graph         graph;
        auto&             source = graph.emplaceBlock<MarbleSource<float>>({{"script", script}, {"row", std::string("high")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&             sink   = graph.emplaceBlock<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(graph.connect<"out", "in">(source, sink).has_value());

        gr::scheduler::Simple<> scheduler;
        expect(scheduler.exchange(std::move(graph)).has_value());
        expect(scheduler.runAndWait().has_value());

        expect(eq(std::string_view{sink.script()}, std::string_view("c c |"))) << "the named row, not the first";
    };

    "an error terminator ends the stream and says why"_test = [values, tags] {
        gr::Graph graph;
        auto&     source = graph.emplaceBlock<MarbleSource<float>>({{"script", std::string("a b #:overflow")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&     sink   = graph.emplaceBlock<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(graph.connect<"out", "in">(source, sink).has_value());
        expect(graph.connect(source, gr::PortDefinition{std::string("evtOut")}, sink, gr::PortDefinition{std::string("evtIn")}).has_value());

        gr::scheduler::Simple<> scheduler;
        expect(scheduler.exchange(std::move(graph)).has_value());
        expect(scheduler.runAndWait().has_value());

        expect(eq(sink._samples.size(), 2UZ)) << "the samples before the error still arrive";
        expect(sink._errored) << "and the reason reaches the sink as an event, a shared ring carrying no tags";
        expect(eq(std::string_view{sink._reason}, std::string_view("overflow")));
        expect(eq(std::string_view{sink.script()}, std::string_view("a b #:overflow"))) << "so the round trip includes the terminator";

        gr::testing::MarbleDiagram diagram{"MarbleSource: a script that ends in an error"};
        diagram.unit = "sample";
        diagram.row("out").at(0U, "a").at(1U, "b").fails("overflow");
        diagram.condition("MarbleSource(script = \"a b #:overflow\")");
        diagram.print();
    };

    "a symbol the settings never defined is reported, not emitted"_test = [values, tags] {
        gr::Graph graph;
        auto&     source = graph.emplaceBlock<MarbleSource<float>>({{"script", std::string("a z b |")}, {"sample_values", values}, {"sample_tags", tags}});
        auto&     sink   = graph.emplaceBlock<MarbleSink<float>>({{"sample_values", values}, {"sample_tags", tags}});
        expect(graph.connect<"out", "in">(source, sink).has_value());

        gr::scheduler::Simple<> scheduler;
        expect(scheduler.exchange(std::move(graph)).has_value());
        expect(scheduler.runAndWait().has_value());

        expect(eq(sink._samples.size(), 2UZ)) << "the unknown symbol is skipped, the rest still plays";
        expect(eq(std::string_view{sink.script()}, std::string_view("a b |")));
    };
};

int main() { /* not needed for UT */ }
