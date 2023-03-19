#include <boost/ut.hpp>

#include <tag.hpp>

const boost::ut::suite TagTests = [] {
    using namespace boost::ut;
    using namespace fair::graph;

    "TagReflection"_test = [] {
        static_assert(sizeof(tag_t) % 64 == 0, "needs to meet L1 cache size");
        static_assert(refl::descriptor::type_descriptor<fair::graph::tag_t>::name == "fair::graph::tag_t");
        static_assert(refl::member_list<tag_t>::size == 2, "index and map bein declared");
        static_assert(refl::trait::get_t<0, refl::member_list<tag_t>>::name == "index", "class field index is public API");
        static_assert(refl::trait::get_t<1, refl::member_list<tag_t>>::name == "map", "class field map is public API");
    };

    "DefaultTags"_test = [] {
        tag_t testTag;

        testTag.insert_or_assign(tag::SAMPLE_RATE, pmtv::pmt(3.0f));
        testTag.insert_or_assign(tag::SAMPLE_RATE(4.0f));
        // testTag.insert_or_assign(tag::SAMPLE_RATE(5.0)); // type-mismatch -> won't compile
        expect(testTag.at(tag::SAMPLE_RATE) == 4.0f);
        expect(tag::SAMPLE_RATE.shortKey() == "sample_rate");
        expect(tag::SAMPLE_RATE.key() == std::string{ GR_TAG_PREFIX }.append("sample_rate"));

        expect(testTag.get(tag::SAMPLE_RATE).has_value());
        static_assert(!std::is_const_v<decltype(testTag.get(tag::SAMPLE_RATE).value())>);
        expect(not testTag.get(tag::SIGNAL_NAME).has_value());

        static_assert(std::is_same_v<decltype(tag::SAMPLE_RATE), decltype(tag::SIGNAL_RATE)>);
        // test other tag on key definition only
        static_assert(tag::SIGNAL_UNIT.shortKey() == "signal_unit");
        static_assert(tag::SIGNAL_MIN.shortKey() == "signal_min");
        static_assert(tag::SIGNAL_MAX.shortKey() == "signal_max");
        static_assert(tag::TRIGGER_NAME.shortKey() == "trigger_name");
        static_assert(tag::TRIGGER_TIME.shortKey() == "trigger_time");
        static_assert(tag::TRIGGER_OFFSET.shortKey() == "trigger_offset");
    };
};

int
main() { /* tests are statically executed */
}