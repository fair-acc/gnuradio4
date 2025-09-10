#include <boost/ut.hpp>

#include <format>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <string>

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
    // deliberately not using std::ranges::equal (gcc bug)
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

static inline gr::property_map propMap(std::initializer_list<std::pair<const std::string, pmtv::pmt>> init) { return gr::property_map{init.begin(), init.end()}; }

const boost::ut::suite<"Port"> _portTests = [] { // NOSONAR (N.B. lambda size)
    using namespace boost::ut;
    using namespace gr;

    "CustomSizePort"_test = [] {
        PortOut<int, RequiredSamples<1, 16>, StreamBufferType<CircularBuffer<int, 32UZ>>> out;
        expect(out.resizeBuffer(32UZ) == ConnectionResult::SUCCESS);
#if defined(__linux__) || defined(__gnu_linux__)
        expect(eq(out.buffer().streamBuffer.size(), static_cast<std::size_t>(getpagesize())));
#else
        // 4096 (page-size) is the minimum buffer size
        // may be difficult to test across other architectures
#endif
    };

    "ResizeBuffer is no-op for input"_test = [] {
        PortIn<int> in;
        auto        before = in.buffer().streamBuffer.size();
        expect(ConnectionResult::SUCCESS == in.resizeBuffer(1234UZ));
        expect(eq(in.buffer().streamBuffer.size(), before));
    };

    "defaultValue/setDefaultValue"_test = [] {
        PortOut<int> port;
        expect(eq(std::any_cast<int>(port.defaultValue()), 0));
        expect(port.setDefaultValue(std::any(42)));
        expect(eq(std::any_cast<int>(port.defaultValue()), 42));
        expect(!port.setDefaultValue(std::any(std::string{"oops"})));
    };

    "InputPort"_test = [] { // NOSONAR (N.B. lambda size)
        PortIn<int> in;
        expect(eq(in.buffer().streamBuffer.size(), 4096UZ));

        auto writer    = in.buffer().streamBuffer.new_writer();
        auto tagWriter = in.buffer().tagBuffer.new_writer();
        { // put testdata into buffer
            auto writeSpan = writer.tryReserve<SpanReleasePolicy::ProcessAll>(8UZ);
            auto tagSpan   = tagWriter.tryReserve(6UZ);
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
            auto data = in.get<SpanReleasePolicy::ProcessAll>(6UZ);
            expect(std::ranges::equal(data.rawTags, std::vector<gr::Tag>{{0UZ, {{"id", "tag@100"}, {"id0", true}}}, {1, {{"id", "tag@101"}, {"id1", true}}}, {3, {{"id", "tag@103"}, {"id3", true}}}, {4, {{"id", "tag@104"}, {"id4", true}}}, {5, {{"id", "tag@105"}, {"id5", true}}}}));
            expect(equalTags(data.tags(), std::vector{std::make_pair(0L, gr::property_map{{"id", "tag@100"}, {"id0", true}}), std::make_pair(1L, gr::property_map{{"id", "tag@101"}, {"id1", true}}), std::make_pair(3L, gr::property_map{{"id", "tag@103"}, {"id3", true}}), std::make_pair(4L, gr::property_map{{"id", "tag@104"}, {"id4", true}}), std::make_pair(5L, gr::property_map{{"id", "tag@105"}, {"id5", true}})}));
            expect(std::ranges::equal(data, std::views::iota(100) | std::views::take(6UZ)));
            expect(data.getMergedTag() == gr::Tag{0UZ, {{"id", "tag@100"}, {"id0", true}}});
            expect(data.consume(3));
        }
        { // full consume
            auto data = in.get<SpanReleasePolicy::ProcessAll>(2);
            expect(std::ranges::equal(data.rawTags, std::vector<gr::Tag>{{3UZ, {{"id", "tag@103"}, {"id3", true}}}, {4UZ, {{"id", "tag@104"}, {"id4", true}}}}));
            expect(equalTags(data.tags(), std::vector{std::make_pair(0L, gr::property_map{{"id", "tag@103"}, {"id3", true}}), std::make_pair(1L, gr::property_map{{"id", "tag@104"}, {"id4", true}})}));
            expect(std::ranges::equal(data, std::views::iota(100) | std::views::drop(3UZ) | std::views::take(2UZ)));
            expect(data.getMergedTag() == gr::Tag{0UZ, {{"id", "tag@103"}, {"id3", true}}});
        }
        { // get empty range
            auto data = in.get<SpanReleasePolicy::ProcessAll>(0UZ);
            expect(eq(data.rawTags.size(), 0UZ));
            expect(eq(data.tags().size(), 0UZ));
            expect(std::ranges::equal(data, std::ranges::empty_view<int>()));
            expect(data.getMergedTag() == gr::Tag{0UZ, {}});
            // consuming nothing must not crash
        }
        { // get consume only first tag
            auto data = in.get<SpanReleasePolicy::ProcessAll, true>(2UZ);
            expect(std::ranges::equal(data.rawTags, std::vector<gr::Tag>{{5UZ, {{"id", "tag@105"}, {"id5", true}}}, {6UZ, {{"id", "tag@106"}, {"id6", true}}}}));
            expect(equalTags(data.tags(), std::vector{std::make_pair(0L, property_map{{"id", "tag@105"}, {"id5", true}}), std::make_pair(1L, property_map{{"id", "tag@106"}, {"id6", true}})}));
            expect(std::ranges::equal(data, std::views::iota(100) | std::views::drop(5UZ) | std::views::take(2UZ)));
            expect(data.getMergedTag() == gr::Tag{0UZ, {{"id", "tag@105"}, {"id5", true}}});
        }
        { // get last sample, last tag is still available
            auto data = in.get<SpanReleasePolicy::ProcessAll>(1UZ);
            expect(std::ranges::equal(data.rawTags, std::vector<gr::Tag>{{6UZ, {{"id", "tag@106"}, {"id6", true}}}}));
            expect(equalTags(data.tags(), std::vector{std::make_pair(-1L, property_map{{"id", "tag@106"}, {"id6", true}})}));
            expect(std::ranges::equal(data, std::views::iota(100) | std::views::drop(7UZ) | std::views::take(1UZ)));
            expect(data.getMergedTag() == gr::Tag{0UZ, {{"id", "tag@106"}, {"id6", true}}});
        }
    };

    "InputSpan getMergedTag (multiple merge)"_test = [] { // NOSONAR (N.B. lambda size)
        PortIn<int> in2;
        auto        w  = in2.buffer().streamBuffer.new_writer();
        auto        tw = in2.buffer().tagBuffer.new_writer();
        {
            auto ws = w.tryReserve<SpanReleasePolicy::ProcessAll>(4);
            auto ts = tw.tryReserve(3UZ);
            ws[0UZ] = 1;
            ws[1UZ] = 2;
            ws[2UZ] = 3;
            ws[3UZ] = 4;
            ts[0UZ] = {0UZ, propMap({{"a", 1}})};
            ts[1UZ] = {1UZ, propMap({{"b", 2}})};
            ts[2UZ] = {3UZ, propMap({{"c", 3}})};
            ts.publish(3UZ);
            ws.publish(4UZ);
        }
        auto span = in2.get<SpanReleasePolicy::ProcessAll>(3);
        auto tag  = span.getMergedTag(3UZ); // merge tags at indices 0 and 1
        expect(eq(tag.map.size(), 2UZ));
        expect(eq(std::get<int>(tag.map.at("a")), 1));
        expect(eq(std::get<int>(tag.map.at("b")), 2));
        // ensure consumeTags doesn't throw
        span.consumeTags(2);
        expect(span.consume(3));
    };

    "OutputPort"_test = [] { // NOSONAR (N.B. lambda size)
        PortOut<int> out;
        auto         reader    = out.buffer().streamBuffer.new_reader();
        auto         tagReader = out.buffer().tagBuffer.new_reader();
        {
            auto data = out.tryReserve<SpanReleasePolicy::ProcessAll>(5);
            expect(eq(data.size(), 5UZ));
            data.publishTag({{"id", "tag@0"}}, 0UZ);
            data.publishTag({{"id", "tag@101"}}, 1UZ);
            data.publishTag({{"id", "tag@104"}}, 4UZ);
            std::iota(data.begin(), data.end(), 100);
            data.publish(5); // should be automatic
        }
        {
            auto data = reader.get<SpanReleasePolicy::ProcessAll>();
            auto tags = tagReader.get<SpanReleasePolicy::ProcessAll>();
            expect(std::ranges::equal(data, std::views::iota(100) | std::views::take(5UZ)));
            expect(std::ranges::equal(tags, std::vector<gr::Tag>{{0UZ, {{"id", "tag@0"}}}, {1UZ, {{"id", "tag@101"}}}, {4UZ, {{"id", "tag@104"}}}}));
        }
        {
            auto data = out.tryReserve<SpanReleasePolicy::ProcessAll>(5);
            expect(eq(data.size(), 5UZ));
            data.publishTag({{"id", "tag@0"}}, 0UZ);
            data.publishTag({{"id", "tag@106"}}, 1UZ);
            data.publishTag({{"id", "tag@109"}}, 4UZ);
            std::iota(data.begin(), data.end(), 105);
            data.publish(5); // should be automatic
        }
        {
            auto data = reader.get();
            auto tags = tagReader.get();
            expect(std::ranges::equal(data, std::views::iota(105) | std::views::take(5UZ)));
            expect(std::ranges::equal(tags, std::vector<gr::Tag>{{5UZ, {{"id", "tag@0"}}}, {6UZ, {{"id", "tag@106"}}}, {9UZ, {{"id", "tag@109"}}}}));
        }
    };

    "publishPendingTags merges same-index tags"_test = [] { // NOSONAR (N.B. lambda size)
        PortOut<int> out;
        auto         reader    = out.buffer().streamBuffer.new_reader();
        auto         tagReader = out.buffer().tagBuffer.new_reader();
        {
            auto s = out.tryReserve<SpanReleasePolicy::ProcessAll>(2);
            s.publishTag(propMap({{"k1", 1}}), 0UZ);
            s.publishTag(propMap({{"k2", 2}}), 0UZ); // merged into the same index
            s[0UZ] = 11;
            s[1UZ] = 22;
            s.publish(2UZ);
        }
        {
            auto data = reader.get();
            auto tags = tagReader.get();
            expect(eq(tags.size(), 1UZ));
            expect(eq(std::get<int>(tags[0].map.at("k1")), 1));
            expect(eq(std::get<int>(tags[0].map.at("k2")), 2));
            expect(std::ranges::equal(data, std::vector<int>{11, 22}));
        }
    };

    "Async/Optional attribute flags"_test = [] {
        using OptionalPort = gr::PortIn<int, gr::Optional>;
        using AsyncPort    = gr::PortIn<int, gr::Async>;
        static_assert(OptionalPort::kIsOptional);
        static_assert(!OptionalPort::kIsSynch);
        static_assert(!AsyncPort::kIsSynch);
        static_assert(!AsyncPort::kIsOptional);
    };

    "nSamplesUntilNextTag & samples_to_eos_tag"_test = [] {
        PortIn<int> in;
        auto        w  = in.buffer().streamBuffer.new_writer();
        auto        tw = in.buffer().tagBuffer.new_writer();
        {
            auto ws = w.tryReserve<SpanReleasePolicy::ProcessAll>(10UZ);
            auto ts = tw.tryReserve(2UZ);
            std::iota(ws.begin(), ws.end(), 0);
            ts[0UZ] = {3UZ, propMap({{"id", "t0"}})};
            ts[1UZ] = {8UZ, propMap({{"id", "eos"}, {gr::tag::END_OF_STREAM, true}})};
            ts.publish(2UZ);
            ws.publish(10UZ);
        }
        auto dist1 = gr::nSamplesUntilNextTag(in, 0);
        expect(dist1.has_value());
        expect(eq(dist1.value(), 3UZ));

        auto dist2 = gr::samples_to_eos_tag(in, 0);
        expect(dist2.has_value());
        expect(eq(dist2.value(), 8UZ));
    };
};

const boost::ut::suite<"port::BitMask"> _bitmask = [] { // NOSONAR (N.B. lambda size)
    using namespace boost::ut;
    using namespace gr;
    using enum gr::PortDirection;
    using enum gr::PortType;
    using port::BitMask;
    using port::BitPattern;

    "bitmask encode/decode"_test = [] {
        constexpr BitMask mask = port::encodeMask(INPUT, STREAM, true, false, false);
        using enum BitMask;
        expect(static_cast<bool>(mask & Input));
        expect(static_cast<bool>(mask & Stream));
        expect(static_cast<bool>(mask & Synchronous));
        expect(not static_cast<bool>(mask & Optional));
        expect(not static_cast<bool>(mask & Connected));
    };

    "bitmask match - exact sync"_test = [] {
        constexpr BitMask    mask    = port::encodeMask(INPUT, STREAM, true, false, false);
        constexpr BitPattern pattern = port::matchBits(PortSync::SYNCHRONOUS);
        expect(pattern.matches(mask)) << "Expected synchronous bit to match";
    };

    "bitmask match - async mismatch"_test = [] {
        constexpr BitMask    mask    = port::encodeMask(INPUT, STREAM, true, false, false);
        constexpr BitPattern pattern = port::matchBits(PortSync::ASYNCHRONOUS);
        expect(not pattern.matches(mask)) << "Expected mismatch for ASYNC vs SYNC";
    };

    "bitmask match - any port direction"_test = [] {
        constexpr BitMask    mask    = port::encodeMask(INPUT, STREAM, true, false, false);
        constexpr BitPattern pattern = port::pattern<PortDirection::INPUT, PortDirection::OUTPUT>();
        expect(pattern.matches(mask)) << "ANY direction must not filter";
    };

    "pattern composition with |"_test = [] {
        constexpr BitPattern pattern1 = port::matchBits(INPUT);
        constexpr BitPattern pattern2 = port::matchBits(PortSync::SYNCHRONOUS);
        constexpr BitPattern composed = pattern1 | pattern2;

        constexpr BitMask mask = port::encodeMask(INPUT, STREAM, true, false, false);
        expect(composed.matches(mask)) << "composed mask should match";
    };

    "pattern<...> NTTP matcher"_test = [] {
        constexpr BitPattern pattern = port::pattern<OUTPUT, PortSync::ASYNCHRONOUS>();
        constexpr BitMask    mask1   = port::encodeMask(OUTPUT, MESSAGE, false, false, false);
        constexpr BitMask    mask2   = port::encodeMask(INPUT, MESSAGE, false, false, false);

        expect(pattern.matches(mask1));
        expect(!pattern.matches(mask2));
    };

    "encodeMask OUTPUT/MESSAGE/optional/connected"_test = [] {
        constexpr BitMask mask = port::encodeMask(OUTPUT, MESSAGE, false, true, true);
        using enum BitMask;
        expect(not any(mask, Input));
        expect(not any(mask, Stream));
        expect(not any(mask, Synchronous));
        expect(any(mask, Optional));
        expect(any(mask, Connected));
    };

    "predicates & decoders"_test = [] {
        constexpr BitMask mask = port::encodeMask(INPUT, STREAM, true, true, true);
        expect(isInput(mask));
        expect(isStream(mask));
        expect(isSynchronous(mask));
        expect(isConnected(mask));
        expect(decodeDirection(mask) == INPUT);
        expect(decodePortType(mask) == STREAM);
    };

    "decode from None"_test = [] {
        constexpr BitMask mask = BitMask::None;
        expect(not isInput(mask));
        expect(not isStream(mask));
        expect(not isSynchronous(mask));
        expect(not isConnected(mask));
        expect(decodeDirection(mask) == OUTPUT); // default when Input-bit not set
        expect(decodePortType(mask) == MESSAGE); // default when Stream-bit not set
    };

    "enum comparison operators"_test = [] {
        constexpr BitMask mask = port::encodeMask(INPUT, STREAM, false, false, false);
        expect(mask == INPUT);
        expect(mask != OUTPUT);
        expect(STREAM == mask);
        expect(MESSAGE != mask);
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
        constexpr BitPattern anyPattern = BitPattern::Any();
        constexpr BitMask    mask1      = port::encodeMask(INPUT, STREAM, true, false, false);
        constexpr BitMask    mask2      = port::encodeMask(OUTPUT, MESSAGE, false, true, true);
        expect(anyPattern.matches(mask1));
        expect(anyPattern.matches(mask2));
        expect(anyPattern.matches(BitMask::None));
    };

    "matchBits don't-care Optional/Connected"_test = [] {
        constexpr BitPattern pattern = port::matchBits(INPUT); // only masks 'Input'
        constexpr BitMask    mask    = port::encodeMask(INPUT, STREAM, true, true, true);
        expect(pattern.matches(mask)) << "Extra bits must not invalidate the match";
    };

    "matchBits OUTPUT/MESSAGE"_test = [] {
        constexpr BitPattern patternDirection = port::matchBits(OUTPUT);
        constexpr BitPattern patternType      = port::matchBits(MESSAGE);
        constexpr BitMask    mask             = port::encodeMask(OUTPUT, MESSAGE, false, false, false);
        expect(patternDirection.matches(mask));
        expect(patternType.matches(mask));
    };

    "pattern<> empty pack == Any"_test = [] {
        constexpr BitPattern pat  = port::pattern<>();
        constexpr BitMask    mask = port::encodeMask(INPUT, STREAM, true, false, true);
        expect(pat.matches(mask));
        expect(pat.matches(BitMask::None));
    };

    "pattern mismatch"_test = [] {
        constexpr BitPattern pattern = port::matchBits(INPUT) | port::matchBits(PortSync::ASYNCHRONOUS);
        constexpr BitMask    mask    = port::encodeMask(INPUT, STREAM, true, false, false); // SYNC
        expect(not pattern.matches(mask));
    };
};

const boost::ut::suite<"PortMetaInfo"> _pmi = [] { // NOSONAR (N.B. lambda size)
    using namespace boost::ut;
    using namespace gr;
    using namespace std::string_literals;

    "default ctor"_test = [] {
        PortMetaInfo metaInfo;
        expect(eq(metaInfo.sample_rate.value, 1.f));
        expect(eq(metaInfo.signal_name.value, "<unnamed>"s));
    };

    "datatype-ctor"_test = [] {
        PortMetaInfo metaInfo{"float32"};
        expect(eq(metaInfo.data_type.value, "float32"s));
    };

    "initializer list ctor"_test = [] {
        PortMetaInfo portMetaInfo({{gr::tag::SAMPLE_RATE.shortKey(), 48000.f}, {gr::tag::SIGNAL_NAME.shortKey(), "TestSignal"}, //
            {gr::tag::SIGNAL_QUANTITY.shortKey(), "voltage"}, {gr::tag::SIGNAL_UNIT.shortKey(), "V"},                           //
            {gr::tag::SIGNAL_MIN.shortKey(), -1.f}, {gr::tag::SIGNAL_MAX.shortKey(), 1.f}});

        expect(eq(48000.f, portMetaInfo.sample_rate.value));
        expect(eq("TestSignal"s, portMetaInfo.signal_name.value));
        expect(eq("voltage"s, portMetaInfo.signal_quantity.value));
        expect(eq("V"s, portMetaInfo.signal_unit.value));
        expect(eq(-1.f, portMetaInfo.signal_min.value));
        expect(eq(+1.f, portMetaInfo.signal_max.value));
    };

    "initializer list ctor throw"_test = [] { //
        expect(throws<std::exception>([&] { PortMetaInfo portMetaInfo({{gr::tag::SAMPLE_RATE.shortKey(), "WRONG TYPE STRING"s}}); }));
    };

    "property_map ctor throw"_test = [] {
        property_map props = {{gr::tag::SAMPLE_RATE.shortKey(), "WRONG TYPE STRING"s}};
        expect(throws<std::exception>([&] { PortMetaInfo portMetaInfo(props); }));
    };

    "update & get roundtrip"_test = [] {
        PortMetaInfo metaInfo{"f32"};
        property_map props;
        props[tag::SAMPLE_RATE.shortKey()]     = 48000.f;
        props[tag::SIGNAL_NAME.shortKey()]     = std::string("IF");
        props[tag::SIGNAL_QUANTITY.shortKey()] = std::string("voltage");
        props[tag::SIGNAL_UNIT.shortKey()]     = std::string("[V]");
        props[tag::SIGNAL_MIN.shortKey()]      = -1.f;
        props[tag::SIGNAL_MAX.shortKey()]      = 1.f;
        expect(metaInfo.update(props).has_value());
        expect(eq(metaInfo.sample_rate.value, 48000.f));
        expect(eq(metaInfo.signal_name.value, "IF"s));
        expect(eq(metaInfo.signal_quantity.value, "voltage"s));
        expect(eq(metaInfo.signal_unit.value, "[V]"s));
        expect(eq(metaInfo.signal_min.value, -1.f));
        expect(eq(metaInfo.signal_max.value, +1.f));

        const property_map out = metaInfo.get();
        expect(eq(std::get<float>(out.at(tag::SAMPLE_RATE.shortKey())), 48000.f));
        expect(eq(std::get<std::string>(out.at(tag::SIGNAL_NAME.shortKey())), "IF"s));
        expect(eq(std::get<std::string>(out.at(tag::SIGNAL_QUANTITY.shortKey())), "voltage"s));
        expect(eq(std::get<std::string>(out.at(tag::SIGNAL_UNIT.shortKey())), "[V]"s));
        expect(eq(std::get<float>(out.at(tag::SIGNAL_MIN.shortKey())), -1.f));
        expect(eq(std::get<float>(out.at(tag::SIGNAL_MAX.shortKey())), 1.f));
    };

    "update wrong type"_test = [] {
        PortMetaInfo metaInfo;
        property_map wrong;
        wrong[tag::SAMPLE_RATE.shortKey()] = 123; // int instead of float
        expect(!metaInfo.update(wrong).has_value());
    };

    "update partial changes before throw"_test = [] {
        PortMetaInfo metaInfo;
        property_map p;
        // to be sure in which order settings are applied
        metaInfo.auto_update = {gr::tag::SAMPLE_RATE.shortKey(), gr::tag::SIGNAL_MIN.shortKey(), gr::tag::SIGNAL_MAX.shortKey()};

        p[gr::tag::SAMPLE_RATE.shortKey()] = 42.f;                             // ok
        p[gr::tag::SIGNAL_MIN.shortKey()]  = std::string("wrong_type_string"); // wrong type
        p[gr::tag::SIGNAL_MAX.shortKey()]  = 42.;                              // o, but after throw
        expect(!metaInfo.update(p).has_value());
        expect(eq(metaInfo.sample_rate.value, 42.f));                                // sample_rate was updated
        expect(eq(metaInfo.signal_min.value, std::numeric_limits<float>::lowest())); // default value
        expect(eq(metaInfo.signal_max.value, std::numeric_limits<float>::max()));    // default value, it was not updated after throw
    };

    "reset auto_update"_test = [] {
        PortMetaInfo portMetaInfo;
        property_map p;
        p[tag::SAMPLE_RATE.shortKey()] = 42.f;
        expect(portMetaInfo.update(p).has_value());
        expect(eq(portMetaInfo.sample_rate.value, 42.f));
        portMetaInfo.auto_update.clear();
        p[tag::SAMPLE_RATE.shortKey()] = 99.f;
        expect(portMetaInfo.update(p).has_value()); // shouldn't update
        expect(eq(portMetaInfo.sample_rate.value, 42.f));
        portMetaInfo.reset();
        expect(portMetaInfo.auto_update.contains(gr::tag::SAMPLE_RATE.shortKey()));
        expect(portMetaInfo.auto_update.contains(gr::tag::SIGNAL_NAME.shortKey()));
        expect(portMetaInfo.auto_update.contains(gr::tag::SIGNAL_QUANTITY.shortKey()));
        expect(portMetaInfo.auto_update.contains(gr::tag::SIGNAL_UNIT.shortKey()));
        expect(portMetaInfo.auto_update.contains(gr::tag::SIGNAL_MIN.shortKey()));
        expect(portMetaInfo.auto_update.contains(gr::tag::SIGNAL_MAX.shortKey()));
        expect(eq(portMetaInfo.sample_rate.value, 42.f)); // shouldn't reset sample_rate
        expect(portMetaInfo.update(p).has_value());
        expect(eq(portMetaInfo.sample_rate.value, 99.f));
    };

    // extra tests
    "auto_update subset only updates selected keys"_test = [] {
        PortMetaInfo m;
        m.sample_rate = 1.0f;
        m.signal_name = "orig"s;
        m.auto_update = {gr::tag::SAMPLE_RATE.shortKey()}; // Only SAMPLE_RATE will be updated

        property_map p;
        p[gr::tag::SAMPLE_RATE.shortKey()] = 12345.f;
        p[gr::tag::SIGNAL_NAME.shortKey()] = std::string("new-name");

        expect(m.update(p).has_value());
        expect(eq(m.sample_rate.value, 12345.f));
        expect(eq(m.signal_name.value, "orig"s)); // unchanged
    };

    "get() roundtrip after partial update"_test = [] {
        PortMetaInfo m{"f32"};
        property_map p;
        p[gr::tag::SIGNAL_MIN.shortKey()] = -0.5f;
        p[gr::tag::SIGNAL_MAX.shortKey()] = +0.5f;
        expect(m.update(p).has_value());

        auto out = m.get();
        expect(eq(std::get<float>(out.at(gr::tag::SIGNAL_MIN.shortKey())), -0.5f));
        expect(eq(std::get<float>(out.at(gr::tag::SIGNAL_MAX.shortKey())), +0.5f));
        expect(eq(std::get<std::string>(out.at(gr::tag::SIGNAL_NAME.shortKey())), "<unnamed>"s)) << "untouched defaults still there";
    };
};

const boost::ut::suite<"DynamicPort"> _dyn = [] { // NOSONAR (N.B. lambda size)
    using namespace boost::ut;
    using namespace gr;

    "construct & weakRef"_test = [] {
        PortOut<int>      src;
        const DynamicPort dynSrc(src, DynamicPort::non_owned_reference_tag{});
        const DynamicPort wearReference = dynSrc.weakRef();
        expect(dynSrc == wearReference);
        expect(dynSrc.direction() == PortDirection::OUTPUT);
        expect(dynSrc.type() == PortType::STREAM);
        expect(dynSrc.typeName() == std::string("int32"));
    };

    "connect/disconnect runtime"_test = [] {
        PortOut<int> src;
        PortIn<int>  dst;
        DynamicPort  dynSrc(src, DynamicPort::non_owned_reference_tag{});
        DynamicPort  dynDst(dst, DynamicPort::non_owned_reference_tag{});

        expect(!dynSrc.isConnected());
        expect(dynSrc.connect(dynDst) == ConnectionResult::SUCCESS);
        expect(dynSrc.isConnected());
        expect(dynDst.isConnected());
        expect(eq(dynSrc.nReaders(), 1UZ));
        expect(eq(dynDst.nWriters(), 1UZ));

        expect(dynDst.disconnect() == ConnectionResult::SUCCESS);
        expect(!dynSrc.isConnected());
    };

    "resizeBuffer via DynamicPort (only output)"_test = [] {
        PortIn<int>  in;
        PortOut<int> out;
        DynamicPort  dynIn(in, DynamicPort::non_owned_reference_tag{});
        DynamicPort  dynOut(out, DynamicPort::non_owned_reference_tag{});
        expect(dynIn.resizeBuffer(2048UZ) == ConnectionResult::FAILED);
        const std::size_t before = out.buffer().streamBuffer.size();
        expect(dynOut.resizeBuffer(before * 2UZ) == ConnectionResult::SUCCESS);
        expect(eq(out.buffer().streamBuffer.size(), before * 2UZ));
    };

    "portInfo/mask/meta snapshot"_test = [] {
        PortOut<float>    src;
        const DynamicPort dynSrc(src, DynamicPort::non_owned_reference_tag{});
        const PortInfo    info = dynSrc.portInfo();
        expect(info.portType == PortType::STREAM);
        expect(info.portDirection == PortDirection::OUTPUT);
        expect(info.isValueTypeArithmeticLike);
        port::BitMask mask = dynSrc.portMaskInfo();
        expect(gr::port::decodeDirection(mask) == PortDirection::OUTPUT);
        expect(!gr::port::isConnected(mask));
        PortMetaInfo metaInfo = dynSrc.portMetaInfo();
        expect(eq(metaInfo.data_type.value, std::string("float32")));
    };
};

const boost::ut::suite<"DynamicPort edge/error"> _dyn_edges = [] { // NOSONAR (N.B. lambda size)
    using namespace boost::ut;
    using namespace gr;
    using enum gr::ConnectionResult;

    "direction/type mismatch -> FAILED"_test = [] {
        PortIn<int> inA;
        PortIn<int> inB;
        DynamicPort dynInA(inA, DynamicPort::non_owned_reference_tag{});
        DynamicPort dynInB(inB, DynamicPort::non_owned_reference_tag{});
        // input -> input: should not connect
#if DEBUG // asserts when debugging
        expect(throws<std::runtime_error>([&dynInA, &dynInB] { std::ignore = dynInA.connect(dynInB); }));
#else
        expect(dynInA.connect(dynInB) == FAILED);
#endif

        MsgPortOut  msgOut;
        PortIn<int> streamIn;
        DynamicPort dynMsgOut(msgOut, DynamicPort::non_owned_reference_tag{});
        DynamicPort dynStreamIn(streamIn, DynamicPort::non_owned_reference_tag{});
        // message -> stream (value_type mismatch): should fail
#if DEBUG // asserts when debugging
        expect(throws<std::runtime_error>([&dynMsgOut, &dynStreamIn] { std::ignore = dynMsgOut.connect(dynStreamIn); }));
#else
        expect(dynMsgOut.connect(dynStreamIn) == FAILED);
#endif
    };

    "double-connect idempotence & counts"_test = [] {
        PortOut<int> src;
        PortIn<int>  dst;
        DynamicPort  dynSrc(src, DynamicPort::non_owned_reference_tag{});
        DynamicPort  dynDst(dst, DynamicPort::non_owned_reference_tag{});

        expect(dynSrc.connect(dynDst) == SUCCESS);
        expect(dynSrc.connect(dynDst) == SUCCESS); // second time should be harmless
        expect(eq(dynSrc.nReaders(), 1UZ));
        expect(eq(dynDst.nWriters(), 1UZ));
    };

    "multiple readers/writers count"_test = [] {
        PortOut<int> src;
        PortIn<int>  a;
        PortIn<int>  b;
        DynamicPort  dynSrc(src, DynamicPort::non_owned_reference_tag{});
        DynamicPort  dA(a, DynamicPort::non_owned_reference_tag{});
        DynamicPort  dB(b, DynamicPort::non_owned_reference_tag{});

        expect(dynSrc.connect(dA) == SUCCESS);
        expect(dynSrc.connect(dB) == SUCCESS);
        expect(dynSrc.nReaders() == 2UZ);
        expect(dA.nWriters() == 1UZ);
        expect(dB.nWriters() == 1UZ);

        expect(dA.disconnect() == SUCCESS);
        expect(dynSrc.nReaders() == 1UZ);
        expect(dB.nWriters() == 1UZ);
    };

    "owned_value_tag move semantics"_test = [] {
        PortOut<int> src;
        DynamicPort  dynPort1(std::move(src), DynamicPort::owned_value_tag{});
        std::size_t  id_before = dynPort1.portInfo().bufferSize;

        DynamicPort dynPort2(std::move(dynPort1));
        expect(dynPort2.portInfo().bufferSize == id_before);
        // d1 is moved-from; no neat way to inspect, but at least ensure d2 still works:
        expect(!dynPort2.isConnected());
    };
};

const boost::ut::suite<"Buffer sizing & counts"> _buf = [] { // NOSONAR (N.B. lambda size)
    using namespace boost::ut;
    using namespace gr;
    using enum gr::ConnectionResult;

    "resize output twice reallocates & grows"_test = [] {
        PortOut<int> out;
        std::size_t  oldSize = out.buffer().streamBuffer.size();
        expect(out.resizeBuffer(oldSize * 2) == SUCCESS);
        std::size_t midSize = out.buffer().streamBuffer.size();
        expect(eq(midSize, oldSize * 2UZ));
        expect(out.resizeBuffer(midSize * 2) == SUCCESS);
        expect(eq(out.buffer().streamBuffer.size(), midSize * 2UZ));
    };

    "resize input after connect is still no-op"_test = [] {
        PortOut<int> out;
        PortIn<int>  in;
        DynamicPort  dynOut(out, DynamicPort::non_owned_reference_tag{});
        DynamicPort  dynIn(in, DynamicPort::non_owned_reference_tag{});
        expect(dynOut.connect(dynIn) == SUCCESS);

        std::size_t before = in.buffer().streamBuffer.size();
        expect(in.resizeBuffer(before + 1234UZ) == SUCCESS);
        expect(eq(in.buffer().streamBuffer.size(), before));
    };
};

const boost::ut::suite<"Message ports"> _msg = [] { // NOSONAR (N.B. lambda size)
    using namespace boost::ut;
    using namespace gr;
    using enum gr::ConnectionResult;

    "basic MsgPort roundtrip"_test = [] { // NOSONAR (N.B. lambda size)
        MsgPortOut  out;
        MsgPortIn   in;
        DynamicPort dynOut(out, DynamicPort::non_owned_reference_tag{});
        DynamicPort dynIn(in, DynamicPort::non_owned_reference_tag{});
        expect(dynOut.connect(dynIn) == SUCCESS);

        auto reader1 = in.buffer().streamBuffer.new_reader(); // reader1 needs to exist before writing (can read/add read barriers to the writer then)
        {
            // publish three default-constructed messages
            auto span = out.tryReserve<SpanReleasePolicy::ProcessAll>(3UZ);
            expect(eq(span.size(), 3UZ));
            // Leave default gr::Message{} if that's fine, otherwise assign trivial payloads
            span.publish(3);
        } // N.B. actually published when the span goes out-of-scope

        { // first reader
            auto data = reader1.get<SpanReleasePolicy::ProcessAll>();
            expect(eq(data.size(), 3UZ)) << "reader 1 needs to read samples";
        }

        {                                                         // second reader -> added after samples have been published -> should see only samples created after this
            auto reader2 = in.buffer().streamBuffer.new_reader(); // 2nd reader
            auto data    = reader2.get<SpanReleasePolicy::ProcessAll>();
            expect(eq(data.size(), 0UZ)) << "reader 2 (late-added) needs zero samples";
        }
    };

    "MsgPort resize + connect counts"_test = [] {
        MsgPortOut  src;
        MsgPortIn   a;
        MsgPortIn   b;
        DynamicPort dynSrc(src, DynamicPort::non_owned_reference_tag{});
        DynamicPort dA(a, DynamicPort::non_owned_reference_tag{});
        DynamicPort dB(b, DynamicPort::non_owned_reference_tag{});

        expect(dynSrc.connect(dA) == SUCCESS);
        expect(dynSrc.connect(dB) == SUCCESS);
        expect(eq(dynSrc.nReaders(), 2UZ));

        auto before = src.buffer().streamBuffer.size();
        expect(dynSrc.resizeBuffer(before * 2UZ) == SUCCESS);
        expect(eq(src.buffer().streamBuffer.size(), before * 2UZ));
    };
};

const boost::ut::suite<"tag-distance helpers"> _tagdist = [] { // NOSONAR (N.B. lambda size)
    using namespace boost::ut;
    using namespace gr;

    "no tags -> nullopt"_test = [] {
        PortIn<int> in;
        expect(!nSamplesUntilNextTag(in, 0UZ).has_value());
    };

    "custom predicate"_test = [] {
        PortIn<int> in;
        auto        writer    = in.buffer().streamBuffer.new_writer();
        auto        tagWriter = in.buffer().tagBuffer.new_writer();
        {
            auto span    = writer.tryReserve<SpanReleasePolicy::ProcessAll>(5UZ);
            auto tagSpan = tagWriter.tryReserve(1UZ);
            std::iota(span.begin(), span.end(), 0);
            tagSpan[0UZ] = {2UZ, propMap({{"x", 1}})};
            tagSpan.publish(1UZ);
            span.publish(5UZ);
        }
        auto pred = [](const Tag& t, std::size_t pos) { return t.index >= pos && t.map.contains("x"); };
        auto val  = gr::nSamplesToNextTagConditional(in, pred, 0UZ);
        expect(val.has_value());
        expect(eq(val.value(), 2UZ));
    };
};

int main() { /* tests are statically executed */ }
