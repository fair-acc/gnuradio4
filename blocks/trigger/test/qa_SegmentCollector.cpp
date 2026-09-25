#include <boost/ut.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include <gnuradio-4.0/algorithm/ImChart.hpp>
#include <gnuradio-4.0/trigger/SegmentCollector.hpp>

using namespace gr::blocks::trigger;

const boost::ut::suite<"SegmentCollector"> _segmentCollector = [] {
    using namespace boost::ut;

    "a window reaches back before the decision and forward past it"_test = [] {
        SegmentCollector<float> collector;
        collector.setWindow(2UZ, 3UZ);
        const std::vector<float> samples{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};

        collector.push(std::span{samples}.first(4UZ)); // 0 1 2 3
        expect(collector.open(3UZ)) << "the decision sits on sample 3";
        expect(!collector.ready()) << "the samples after it have not arrived yet";

        collector.push(std::span{samples}.subspan(4UZ)); // 4 5 6 7
        expect(collector.ready());

        const auto segment = collector.take();
        expect(segment.has_value());
        if (segment) {
            const std::vector<float> expected{1.f, 2.f, 3.f, 4.f, 5.f, 6.f}; // two before, the decision, three after
            expect(eq(segment->first.size(), expected.size()));
            expect(std::ranges::equal(segment->first, expected)) << "the window spans the decision, not just what followed it";
        }
    };

    "a decision at the very start reads zeroes where the stream had not begun"_test = [] {
        SegmentCollector<float> collector;
        collector.setWindow(2UZ, 1UZ);
        const std::vector<float> samples{7.f, 8.f, 9.f};
        collector.push(samples);
        expect(collector.open(0UZ));

        const auto segment = collector.take();
        expect(segment.has_value());
        if (segment) {
            expect(eq(segment->first.size(), 4UZ)) << "the window keeps its length even at the edge";
            expect(eq(segment->first[0], 0.f)) << "padded where the stream had not begun";
            expect(eq(segment->first[1], 0.f));
            expect(eq(segment->first[2], 7.f)) << "the decision's own sample";
            expect(eq(segment->first[3], 8.f)) << "and one after it";
        }
    };

    "several windows stay open at once, and come back in order"_test = [] {
        SegmentCollector<float> collector;
        collector.setWindow(1UZ, 1UZ);
        const std::vector<float> samples{0.f, 1.f, 2.f, 3.f, 4.f, 5.f};
        collector.push(std::span{samples}.first(3UZ));
        expect(collector.open(1UZ, 111U));
        expect(collector.open(2UZ, 222U));
        expect(eq(collector.pending(), 2UZ));

        collector.push(std::span{samples}.subspan(3UZ));
        const auto first  = collector.take();
        const auto second = collector.take();
        expect(first.has_value() and second.has_value());
        if (first and second) {
            expect(eq(first->second, std::uint64_t{111U})) << "each window carries the decision it belongs to";
            expect(eq(second->second, std::uint64_t{222U}));
            expect(eq(first->first.front(), 0.f)) << "the first window starts one sample before its decision";
            expect(eq(second->first.front(), 1.f));
        }
    };

    "a window whose samples went past before it was read is counted, not faked"_test = [] {
        SegmentCollector<float> collector;
        collector.setWindow(1UZ, 1UZ, 0UZ); // no margin at all, so a pending window is easily overrun
        const std::vector<float> samples(20UZ, 1.f);
        collector.push(std::span{samples}.first(3UZ));
        expect(collector.open(1UZ));
        collector.push(std::span{samples}.subspan(3UZ)); // 17 more samples arrive before it is read

        expect(!collector.take().has_value()) << "its samples are gone, so there is nothing truthful to return";
        expect(eq(collector.lost, 1U)) << "and the loss is counted";
    };

    "a decision whose pre-history has already gone is refused on the spot"_test = [] {
        SegmentCollector<float> collector;
        collector.setWindow(2UZ, 1UZ, 0UZ);
        const std::vector<float> history(40UZ, 1.f);
        collector.push(history);

        expect(!collector.open(0UZ)) << "sample 0 fell out of the buffer long ago";
        expect(eq(collector.refused, 1U));
        expect(collector.open(38UZ)) << "a recent decision is still servable";
    };

    "a reset forgets the windows and the history"_test = [] {
        SegmentCollector<float> collector;
        collector.setWindow(1UZ, 1UZ);
        const std::vector<float> samples{1.f, 2.f, 3.f};
        collector.push(samples);
        expect(collector.open(1UZ));
        collector.reset();

        expect(eq(collector.pending(), 0UZ));
        expect(eq(collector.streamIndex, 0UZ));
        expect(!collector.take().has_value());
    };

    "what a window cuts out, drawn"_test = [] {
        // a decaying pulse part-way through a quiet stream; the decision sits on its peak
        constexpr std::size_t kLength = 60UZ;
        constexpr std::size_t kPeak   = 30UZ;
        std::vector<float>    stream(kLength, 0.f);
        for (std::size_t i = kPeak; i < kLength && i - kPeak < 15UZ; ++i) {
            stream[i] = 5.f * std::exp(-static_cast<float>(i - kPeak) / 4.f);
        }

        SegmentCollector<float> collector;
        collector.setWindow(10UZ, 20UZ);
        collector.push(stream);
        expect(collector.open(kPeak));
        const auto segment = collector.take();
        expect(segment.has_value());
        if (!segment) {
            return;
        }
        expect(eq(segment->first.size(), 31UZ)) << "ten before the decision, the decision, twenty after";

        std::vector<double> xStream(kLength);
        std::vector<double> yStream(kLength);
        for (std::size_t i = 0UZ; i < kLength; ++i) {
            xStream[i] = static_cast<double>(i);
            yStream[i] = static_cast<double>(stream[i]);
        }
        auto whole = gr::graphs::ImChart<90, 12>({{0., static_cast<double>(kLength - 1UZ)}, {-0.5, 6.}});
        whole.draw(xStream, yStream, "stream");
        std::println("\nthe stream, with the decision at sample {} of {}:", kPeak, kLength);
        whole.draw();

        std::vector<double> xCut(segment->first.size());
        std::vector<double> yCut(segment->first.size());
        for (std::size_t i = 0UZ; i < segment->first.size(); ++i) {
            xCut[i] = static_cast<double>(i) - 10.; // zero at the decision, negatives are the pre-history
            yCut[i] = static_cast<double>(segment->first[i]);
        }
        auto cut = gr::graphs::ImChart<90, 12>({{xCut.front(), xCut.back()}, {-0.5, 6.}});
        cut.draw(xCut, yCut, "segment");
        std::println("\nand the window it cut, zero at the decision -- the ten samples left of it are the pre-history:");
        cut.draw();
    };
};

int main() { /* tests are statically executed */ }
