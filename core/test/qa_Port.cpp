#include <boost/ut.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <gnuradio-4.0/Port.hpp>

template<>
struct fmt::formatter<gr::Tag> {
    template<typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    constexpr auto format(const gr::Tag& tag, FormatContext& ctx) const {
        return fmt::format_to(ctx.out(), "  {}->{{ {} }}\n", tag.index, tag.map);
    }
};

const boost::ut::suite PortTests = [] {
    using namespace boost::ut;
    using namespace gr;

    "CustomSizePort"_test = [] {
        PortOut<int, RequiredSamples<1, 16>, StreamBufferType<CircularBuffer<int, 32>>> out;
        expect(out.resizeBuffer(32) == gr::ConnectionResult::SUCCESS);
        // expect(eq(out.buffer().streamBuffer.size(), static_cast<std::size_t>(getpagesize()))); // 4096 (page-size) is the minimum buffer size // difficult to test across architectures
    };

    "InputPort"_test = [] {
        PortIn<int> in;
        expect(eq(in.buffer().streamBuffer.size(), 65536UZ));

        auto writer    = in.buffer().streamBuffer.new_writer();
        auto tagWriter = in.buffer().tagBuffer.new_writer();
        { // put testdata into buffer
            auto writeSpan = writer.reserve<SpanReleasePolicy::ProcessAll>(5);
            auto tagSpan   = tagWriter.reserve(5);
            tagSpan[0]     = {-1, {{"id", "tag@-1"}, {"id0", true}}};
            tagSpan[1]     = {1, {{"id", "tag@101"}, {"id1", true}}};
            tagSpan[2]     = {2, {{"id", "tag@102"}, {"id2", true}}};
            tagSpan[3]     = {3, {{"id", "tag@103"}, {"id3", true}}};
            tagSpan[4]     = {4, {{"id", "tag@104"}, {"id4", true}}};
            std::iota(writeSpan.begin(), writeSpan.end(), 100);
            tagSpan.publish(5);   // this should not be necessary as the ProcessAll policy should publish automatically
            writeSpan.publish(5); // this should not be necessary as the ProcessAll policy should publish automatically
        }
        { // partial consume
            auto data = in.get<SpanReleasePolicy::ProcessAll>(5);
            // fmt::print("idx: {}: data: {}\ntags: {}\nmergedTag: {}\n", data.streamIndex, std::span(data), in.tags, in.getMergedTag());
            expect(std::ranges::equal(data.tags, std::vector<gr::Tag>{{-1, {{"id", "tag@-1"}, {"id0", true}}}, {1, {{"id", "tag@101"}, {"id1", true}}}, {2, {{"id", "tag@102"}, {"id2", true}}}, {3, {{"id", "tag@103"}, {"id3", true}}}, {4, {{"id", "tag@104"}, {"id4", true}}}}));
            expect(std::ranges::equal(data, std::views::iota(100) | std::views::take(5)));
            expect(in.getMergedTag() == gr::Tag{-1, {{"id", "tag@-1"}, {"id0", true}}});
            expect(data.consume(2));
        }
        { // full consume
            auto data = in.get<SpanReleasePolicy::ProcessAll>(2);
            expect(std::ranges::equal(data.tags, std::vector<gr::Tag>{{1, {{"id", "tag@101"}, {"id1", true}}}, {2, {{"id", "tag@102"}, {"id2", true}}}, {3, {{"id", "tag@103"}, {"id3", true}}}}));
            expect(std::ranges::equal(data, std::views::iota(100) | std::views::drop(2) | std::views::take(2)));
            expect(in.getMergedTag() == gr::Tag{-1, {{"id", "tag@102"}, {"id1", true}, {"id2", true}}});
        }
        { // get empty range
            auto data = in.get<SpanReleasePolicy::ProcessAll>(0);
            expect(std::ranges::equal(data.tags, std::vector<gr::Tag>{{3, {{"id", "tag@103"}, {"id3", true}}}}));
            expect(std::ranges::equal(data, std::vector<int>()));
            expect(in.getMergedTag() == gr::Tag{-1, {{"id", "tag@103"}, {"id3", true}}});
        }
        { // get last sample
            auto data = in.get<SpanReleasePolicy::ProcessAll>(1);
            expect(std::ranges::equal(data.tags, std::vector<gr::Tag>{{3, {{"id", "tag@103"}, {"id3", true}}}, {4, {{"id", "tag@104"}, {"id4", true}}}}));
            expect(std::ranges::equal(data, std::views::iota(100) | std::views::drop(4) | std::views::take(1)));
            expect(in.getMergedTag() == gr::Tag{-1, {{"id", "tag@104"}, {"id3", true}, {"id4", true}}});
        }
    };

    "OutputPort"_test = [] {
        PortOut<int> out;
        auto         reader    = out.buffer().streamBuffer.new_reader();
        auto         tagReader = out.buffer().tagBuffer.new_reader();
        {
            auto data = out.reserve<SpanReleasePolicy::ProcessAll>(5);
            out.publishTag({{"id", "tag@-1"}}, -1);
            out.publishPendingTags();
            out.publishTag({{"id", "tag@101"}}, 1);
            out.publishPendingTags();
            out.publishTag({{"id", "tag@104"}}, 4);
            out.publishPendingTags();
            std::iota(data.begin(), data.end(), 100);
            out.publishPendingTags();
            data.publish(5); // should be automatic
        }
        {
            auto data = reader.get<SpanReleasePolicy::ProcessAll>();
            auto tags = tagReader.get<SpanReleasePolicy::ProcessAll>();
            expect(std::ranges::equal(data, std::views::iota(100) | std::views::take(5)));
            expect(std::ranges::equal(tags, std::vector<gr::Tag>{{-1, {{"id", "tag@-1"}}}, {1, {{"id", "tag@101"}}}, {4, {{"id", "tag@104"}}}}));
        }
        {
            auto data = out.reserve<SpanReleasePolicy::ProcessAll>(5);
            out.publishTag({{"id", "tag@-1"}}, -1);
            out.publishPendingTags();
            out.publishTag({{"id", "tag@106"}}, 1);
            out.publishPendingTags();
            out.publishTag({{"id", "tag@109"}}, 4);
            out.publishPendingTags();
            std::iota(data.begin(), data.end(), 105);
            out.publishPendingTags();
            data.publish(5); // should be automatic
        }
        {
            auto data = reader.get();
            auto tags = tagReader.get();
            expect(std::ranges::equal(data, std::views::iota(105) | std::views::take(5)));
            expect(std::ranges::equal(tags, std::vector<gr::Tag>{{-1, {{"id", "tag@-1"}}}, {6, {{"id", "tag@106"}}}, {9, {{"id", "tag@109"}}}}));
        }
    };
};

int main() { /* tests are statically executed */ }
