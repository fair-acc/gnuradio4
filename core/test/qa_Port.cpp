#include <boost/ut.hpp>

#include <format>

#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>

template<>
struct std::formatter<gr::Tag> {
    template<typename ParseContext>
    constexpr auto parse(ParseContext& ctx) {
        return ctx.begin();
    }

    template<typename FormatContext>
    constexpr auto format(const gr::Tag& tag, FormatContext& ctx) const {
        return std::format_to(ctx.out(), "  {}->{{ {} }}\n", tag.index, tag.map);
    }
};

/**
 * std::ranges::equal does not work correctly in gcc < 14.2 because InputSpan::tags() contains references to the tag property maps, while in the expected vector we have values
 */
bool equalTags(auto tags, auto expected) {
    if (tags.size() != expected.size()) {
        return false;
    }
    for (const auto& [tag, expectedTag] : std::views::zip(tags, expected)) {
        if (tag.first != expectedTag.first) {
            return false;
        }
        if (tag.second != expectedTag.second) {
            return false;
        }
    }
    return true;
}

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
        expect(eq(in.buffer().streamBuffer.size(), 4096UZ));

        auto writer    = in.buffer().streamBuffer.new_writer();
        auto tagWriter = in.buffer().tagBuffer.new_writer();
        { // put testdata into buffer
            auto writeSpan = writer.tryReserve<SpanReleasePolicy::ProcessAll>(8);
            auto tagSpan   = tagWriter.tryReserve(6);
            expect(eq(writeSpan.size(), 8UZ));
            expect(eq(tagSpan.size(), 6UZ));
            tagSpan[0] = {0, {{"id", "tag@100"}, {"id0", true}}};
            tagSpan[1] = {1, {{"id", "tag@101"}, {"id1", true}}};
            tagSpan[2] = {3, {{"id", "tag@103"}, {"id3", true}}};
            tagSpan[3] = {4, {{"id", "tag@104"}, {"id4", true}}};
            tagSpan[4] = {5, {{"id", "tag@105"}, {"id5", true}}};
            tagSpan[5] = {6, {{"id", "tag@106"}, {"id6", true}}};
            std::iota(writeSpan.begin(), writeSpan.end(), 100);
            tagSpan.publish(6);   // this should not be necessary as the ProcessAll policy should publish automatically
            writeSpan.publish(8); // this should not be necessary as the ProcessAll policy should publish automatically
        }
        { // partial consume
            auto data = in.get<SpanReleasePolicy::ProcessAll>(6);
            expect(std::ranges::equal(data.rawTags, std::vector<gr::Tag>{{0, {{"id", "tag@100"}, {"id0", true}}}, {1, {{"id", "tag@101"}, {"id1", true}}}, {3, {{"id", "tag@103"}, {"id3", true}}}, {4, {{"id", "tag@104"}, {"id4", true}}}, {5, {{"id", "tag@105"}, {"id5", true}}}}));
            expect(equalTags(data.tags(), std::vector{std::make_pair(0L, gr::property_map{{"id", "tag@100"}, {"id0", true}}), std::make_pair(1L, gr::property_map{{"id", "tag@101"}, {"id1", true}}), std::make_pair(3L, gr::property_map{{"id", "tag@103"}, {"id3", true}}), std::make_pair(4L, gr::property_map{{"id", "tag@104"}, {"id4", true}}), std::make_pair(5L, gr::property_map{{"id", "tag@105"}, {"id5", true}})}));
            expect(std::ranges::equal(data, std::views::iota(100) | std::views::take(6)));
            expect(data.getMergedTag() == gr::Tag{0UZ, {{"id", "tag@100"}, {"id0", true}}});
            expect(data.consume(3));
        }
        { // full consume
            auto data = in.get<SpanReleasePolicy::ProcessAll>(2);
            expect(std::ranges::equal(data.rawTags, std::vector<gr::Tag>{{3, {{"id", "tag@103"}, {"id3", true}}}, {4, {{"id", "tag@104"}, {"id4", true}}}}));
            expect(equalTags(data.tags(), std::vector{std::make_pair(0L, gr::property_map{{"id", "tag@103"}, {"id3", true}}), std::make_pair(1L, gr::property_map{{"id", "tag@104"}, {"id4", true}})}));
            expect(std::ranges::equal(data, std::views::iota(100) | std::views::drop(3) | std::views::take(2)));
            expect(data.getMergedTag() == gr::Tag{0UZ, {{"id", "tag@103"}, {"id3", true}}});
        }
        { // get empty range
            auto data = in.get<SpanReleasePolicy::ProcessAll>(0);
            expect(eq(data.rawTags.size(), 0UZ));
            expect(eq(data.tags().size(), 0UZ));
            expect(std::ranges::equal(data, std::ranges::empty_view<int>()));
            expect(data.getMergedTag() == gr::Tag{0UZ, {}});
        }
        { // get consume only first tag
            auto data = in.get<SpanReleasePolicy::ProcessAll, true>(2);
            expect(std::ranges::equal(data.rawTags, std::vector<gr::Tag>{{5, {{"id", "tag@105"}, {"id5", true}}}, {6, {{"id", "tag@106"}, {"id6", true}}}}));
            expect(equalTags(data.tags(), std::vector{std::make_pair(0L, property_map{{"id", "tag@105"}, {"id5", true}}), std::make_pair(1L, property_map{{"id", "tag@106"}, {"id6", true}})}));
            expect(std::ranges::equal(data, std::views::iota(100) | std::views::drop(5) | std::views::take(2)));
            expect(data.getMergedTag() == gr::Tag{0UZ, {{"id", "tag@105"}, {"id5", true}}});
        }
        { // get last sample, last tag is still available
            auto data = in.get<SpanReleasePolicy::ProcessAll>(1);
            expect(std::ranges::equal(data.rawTags, std::vector<gr::Tag>{{6, {{"id", "tag@106"}, {"id6", true}}}}));
            expect(equalTags(data.tags(), std::vector{std::make_pair(-1L, property_map{{"id", "tag@106"}, {"id6", true}})}));
            expect(std::ranges::equal(data, std::views::iota(100) | std::views::drop(7) | std::views::take(1)));
            expect(data.getMergedTag() == gr::Tag{0UZ, {{"id", "tag@106"}, {"id6", true}}});
        }
    };

    "OutputPort"_test = [] {
        PortOut<int> out;
        auto         reader    = out.buffer().streamBuffer.new_reader();
        auto         tagReader = out.buffer().tagBuffer.new_reader();
        {
            auto data = out.tryReserve<SpanReleasePolicy::ProcessAll>(5);
            expect(eq(data.size(), 5UZ));
            data.publishTag({{"id", "tag@0"}}, 0);
            data.publishTag({{"id", "tag@101"}}, 1);
            data.publishTag({{"id", "tag@104"}}, 4);
            std::iota(data.begin(), data.end(), 100);
            data.publish(5); // should be automatic
        }
        {
            auto data = reader.get<SpanReleasePolicy::ProcessAll>();
            auto tags = tagReader.get<SpanReleasePolicy::ProcessAll>();
            expect(std::ranges::equal(data, std::views::iota(100) | std::views::take(5)));
            expect(std::ranges::equal(tags, std::vector<gr::Tag>{{0, {{"id", "tag@0"}}}, {1, {{"id", "tag@101"}}}, {4, {{"id", "tag@104"}}}}));
        }
        {
            auto data = out.tryReserve<SpanReleasePolicy::ProcessAll>(5);
            expect(eq(data.size(), 5UZ));
            data.publishTag({{"id", "tag@0"}}, 0);
            data.publishTag({{"id", "tag@106"}}, 1);
            data.publishTag({{"id", "tag@109"}}, 4);
            std::iota(data.begin(), data.end(), 105);
            data.publish(5); // should be automatic
        }
        {
            auto data = reader.get();
            auto tags = tagReader.get();
            expect(std::ranges::equal(data, std::views::iota(105) | std::views::take(5)));
            expect(std::ranges::equal(tags, std::vector<gr::Tag>{{5, {{"id", "tag@0"}}}, {6, {{"id", "tag@106"}}}, {9, {{"id", "tag@109"}}}}));
        }
    };
};

int main() { /* tests are statically executed */ }
