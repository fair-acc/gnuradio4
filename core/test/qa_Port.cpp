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

const boost::ut::suite<"Port"> _portTests = [] {
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

boost::ut::suite<"port::BitMask"> _bitmask = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::port;

    "bitmask encode/decode"_test = [] {
        constexpr BitMask m = encodeMask(PortDirection::INPUT, PortType::STREAM, true, false, false);
        expect(static_cast<bool>(m & BitMask::Input));
        expect(static_cast<bool>(m & BitMask::Stream));
        expect(static_cast<bool>(m & BitMask::Synchronous));
        expect(not static_cast<bool>(m & BitMask::Optional));
        expect(not static_cast<bool>(m & BitMask::Connected));
    };

    "bitmask match - exact sync"_test = [] {
        const BitMask m = encodeMask(PortDirection::INPUT, PortType::STREAM, true, false, false);
        const auto    p = matchBits(PortSync::SYNCHRONOUS);
        expect(p.matches(m)) << "Expected synchronous bit to match";
    };

    "bitmask match - async mismatch"_test = [] {
        const BitMask m = encodeMask(PortDirection::INPUT, PortType::STREAM, true, false, false);
        const auto    p = matchBits(PortSync::ASYNCHRONOUS);
        expect(not p.matches(m)) << "Expected mismatch for ASYNC vs SYNC";
    };

    "bitmask match - any port direction"_test = [] {
        const BitMask m = encodeMask(PortDirection::INPUT, PortType::STREAM, true, false, false);
        const auto    p = matchBits(PortDirection::ANY);
        expect(p.matches(m)) << "ANY direction must not filter";
    };

    "pattern composition with |"_test = [] {
        const BitPattern p1       = matchBits(PortDirection::INPUT);
        const BitPattern p2       = matchBits(PortSync::SYNCHRONOUS);
        const auto       composed = p1 | p2;

        const BitMask m = encodeMask(PortDirection::INPUT, PortType::STREAM, true, false, false);
        expect(composed.matches(m)) << "Composed mask should match";
    };

    "pattern<...> NTTP matcher"_test = [] {
        constexpr auto pat = pattern<PortDirection::OUTPUT, PortSync::ASYNCHRONOUS>();
        const BitMask  m1  = encodeMask(PortDirection::OUTPUT, PortType::MESSAGE, false, false, false);
        const BitMask  m2  = encodeMask(PortDirection::INPUT, PortType::MESSAGE, false, false, false);

        expect(pat.matches(m1));
        expect(!pat.matches(m2));
    };

    "encodeMask OUTPUT/MESSAGE/optional/connected"_test = [] {
        const BitMask m = encodeMask(PortDirection::OUTPUT, PortType::MESSAGE, false, true, true);
        expect(not any(m, BitMask::Input));
        expect(not any(m, BitMask::Stream));
        expect(not any(m, BitMask::Synchronous));
        expect(any(m, BitMask::Optional));
        expect(any(m, BitMask::Connected));
    };

    "predicates & decoders"_test = [] {
        const BitMask m = encodeMask(PortDirection::INPUT, PortType::STREAM, true, true, true);
        expect(isInput(m));
        expect(isStream(m));
        expect(isSynchronous(m));
        expect(isConnected(m));
        expect(decodeDirection(m) == PortDirection::INPUT);
        expect(decodePortType(m) == PortType::STREAM);
    };

    "decode from None"_test = [] {
        constexpr BitMask m = BitMask::None;
        expect(not isInput(m));
        expect(not isStream(m));
        expect(not isSynchronous(m));
        expect(not isConnected(m));
        expect(decodeDirection(m) == PortDirection::OUTPUT); // default when Input-bit not set
        expect(decodePortType(m) == PortType::MESSAGE);      // default when Stream-bit not set
    };

    "enum comparison operators"_test = [] {
        const BitMask m = encodeMask(PortDirection::INPUT, PortType::STREAM, false, false, false);
        expect(m == PortDirection::INPUT);
        expect(m != PortDirection::OUTPUT);
        expect(PortType::STREAM == m);
        expect(PortType::MESSAGE != m);
    };

    "bitwise ops"_test = [] {
        using enum BitMask;
        constexpr BitMask a = Input | Stream;
        constexpr BitMask b = Stream | Synchronous;
        expect(any(a & Input, Input));
        expect(any(b & Stream, Stream));
        expect(not any(a & Synchronous, Synchronous));
    };

    "BitPattern::Any matches everything"_test = [] {
        const auto    anyPat = BitPattern::Any();
        const BitMask m1     = encodeMask(PortDirection::INPUT, PortType::STREAM, true, false, false);
        const BitMask m2     = encodeMask(PortDirection::OUTPUT, PortType::MESSAGE, false, true, true);
        expect(anyPat.matches(m1));
        expect(anyPat.matches(m2));
        expect(anyPat.matches(BitMask::None));
    };

    "matchBits don't-care Optional/Connected"_test = [] {
        const BitPattern p = matchBits(PortDirection::INPUT); // only masks 'Input'
        const BitMask    m = encodeMask(PortDirection::INPUT, PortType::STREAM, true, true, true);
        expect(p.matches(m)) << "Extra bits must not invalidate the match";
    };

    "matchBits OUTPUT/MESSAGE"_test = [] {
        const BitPattern pd = matchBits(PortDirection::OUTPUT);
        const BitPattern pt = matchBits(PortType::MESSAGE);
        const BitMask    m  = encodeMask(PortDirection::OUTPUT, PortType::MESSAGE, false, false, false);
        expect(pd.matches(m));
        expect(pt.matches(m));
    };

    "pattern<> empty pack == Any"_test = [] {
        constexpr auto pat = pattern<>();
        const BitMask  m1  = encodeMask(PortDirection::INPUT, PortType::STREAM, true, false, true);
        expect(pat.matches(m1));
        expect(pat.matches(BitMask::None));
    };

    "pattern mismatch"_test = [] {
        const BitPattern p = matchBits(PortDirection::INPUT) | matchBits(PortSync::ASYNCHRONOUS);
        const BitMask    m = encodeMask(PortDirection::INPUT, PortType::STREAM, true, false, false); // SYNC
        expect(not p.matches(m));
    };
};

int main() { /* tests are statically executed */ }
