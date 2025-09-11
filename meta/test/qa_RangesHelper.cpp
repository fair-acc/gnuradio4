#include <boost/ut.hpp>
#include <format>
#include <forward_list>
#include <gnuradio-4.0/meta/RangesHelper.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <vector>

const boost::ut::suite<"AdjacentDeduplicateView tests"> _AdjacentDeduplicateViewTests = [] {
    using namespace boost::ut;
    using namespace gr;

    "AdjacentDeduplicateView - single element"_test = [&] {
        std::vector<int> v{42};
        auto             outView = v | AdjacentDeduplicateView();
        expect(std::ranges::equal(outView, std::vector<int>{42}));
    };

    "AdjacentDeduplicateView - all equal elements"_test = [&] {
        std::vector<int> v(7, 5);
        auto             outView = v | AdjacentDeduplicateView();
        expect(std::ranges::equal(outView, std::vector<int>{5}));
    };

    "AdjacentDeduplicateView - default ctor"_test = [&] {
        std::vector<int> v{1, 1, 2, 2, 2, 3, 2, 2};
        auto             outView = v | AdjacentDeduplicateView();
        expect(std::ranges::equal(outView, std::vector<int>{1, 2, 3, 2}));
    };

    "AdjacentDeduplicateView - ctor with Eq"_test = [&] {
        std::vector<int> v{5, 6, 6, 6, 6, 6, 7, 7};
        auto             outView = v | AdjacentDeduplicateView(std::ranges::equal_to{});
        expect(std::ranges::equal(outView, std::vector<int>{5, 6, 7}));
    };

    "AdjacentDeduplicateView - forward_list"_test = [&] {
        std::forward_list<int> fl{1, 1, 1, 2, 3, 3};
        auto                   outView = fl | AdjacentDeduplicateView{};
        expect(std::ranges::equal(outView, std::vector<int>({1, 2, 3})));
    };

    "AdjacentDeduplicateView - custom predicate on Tag"_test = [] {
        struct Tag {
            std::size_t                index{};
            std::map<std::string, int> map{};
        };
        auto isSame = [](const Tag& a, const Tag& b) { return a.index == b.index && a.map == b.map; };

        std::vector<Tag> inputVec{
            {1, {{"str1", 1}}}, {1, {{"str1", 1}}}, // dup
            {2, {{"str2", 2}}}, {2, {{"str2", 2}}}, // dup
            {3, {{"str3", 3}}},                     //
            {4, {{"str4", 4}}}, {4, {{"str4", 4}}}, // dup
        };

        auto outView = inputVec | AdjacentDeduplicateView{isSame};
        expect(std::ranges::equal(std::vector<Tag>(outView.begin(), outView.end()), std::vector<Tag>{{1, {{"str1", 1}}}, {2, {{"str2", 2}}}, {3, {{"str3", 3}}}, {4, {{"str4", 4}}}}, isSame));
    };

    "AdjacentDeduplicateView - pipeline composition"_test = [&] {
        std::vector<int> v{0, 0, 1, 1, 2, 2};
        auto             outView = v | std::views::transform([](int x) { return x + 1; }) | AdjacentDeduplicateView{} | std::views::transform([](int x) { return x * 10; });
        expect(std::ranges::equal(outView, std::vector<int>({10, 20, 30})));
    };
};

const boost::ut::suite<"PairDeduplicateView tests"> _PairDeduplicateViewTests = [] {
    using namespace boost::ut;
    using namespace gr;

    struct Tag {
        std::size_t                index{};
        std::map<std::string, int> map{};
    };
    auto isSame1 = [](const Tag& a, const Tag& b) { return a.index == b.index; };
    auto isSame2 = [](const Tag& a, const Tag& b) { return a.index == b.index && a.map == b.map; };

    "PairDeduplicateView - all duplicates in one index"_test = [&] {
        std::vector<Tag> v{{1, {{"a", 1}}}, {1, {{"a", 1}}}, {1, {{"a", 1}}}};
        auto             out = v | PairDeduplicateView{isSame1, isSame2};
        expect(std::ranges::equal(out, std::vector<Tag>{{1, {{"a", 1}}}}, isSame2));
    };

    "PairDeduplicateView - no duplicates at all"_test = [&] {
        std::vector<Tag> v{
            {1, {{"a", 1}}}, {1, {{"b", 1}}}, //
            {2, {{"c", 2}}}, {2, {{"d", 2}}}  //
        };
        auto out = v | PairDeduplicateView{isSame1, isSame2};
        expect(std::ranges::equal(out, v, isSame2));
    };

    "PairDeduplicateView - forward_list input (forward_range)"_test = [&] {
        std::forward_list<Tag> fl{Tag{1, {{"a", 1}}}, Tag{1, {{"a", 1}}}, Tag{1, {{"b", 1}}}, Tag{1, {{"b", 1}}}, Tag{1, {{"b", 1}}}};
        auto                   out = fl | PairDeduplicateView{isSame1, isSame2};
        std::vector<Tag>       expected{{1, {{"a", 1}}}, {1, {{"b", 1}}}};
        expect(std::ranges::equal(out, expected, isSame2));
    };

    "PairDeduplicateView - extra"_test = [&] {
        std::vector<Tag> inputVec{
            {1, {{"a", 1}}}, {1, {{"b", 1}}}, {1, {{"b", 1}}}, {1, {{"a", 1}}}, {1, {{"b", 1}}}, {1, {{"b", 1}}}, //
            {2, {{"b", 2}}}, {2, {{"c", 2}}}, {2, {{"c", 2}}}, {2, {{"b", 2}}},                                   //
            {3, {{"c", 3}}},                                                                                      //
            {4, {{"d", 4}}}, {4, {{"d", 4}}},                                                                     // dup
        };

        auto outView = inputVec | PairDeduplicateView{isSame1, isSame2};
        expect(std::ranges::equal(outView,
            std::vector<Tag>{                     //
                {1, {{"a", 1}}}, {1, {{"b", 1}}}, //
                {2, {{"b", 2}}}, {2, {{"c", 2}}}, //
                {3, {{"c", 3}}},                  //
                {4, {{"d", 4}}}},
            isSame2));
    };
};

int main() {}
