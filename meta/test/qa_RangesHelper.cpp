#include <boost/ut.hpp>
#include <format>
#include <forward_list>
#include <gnuradio-4.0/meta/RangesHelper.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <vector>

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

const boost::ut::suite<"MergeView tests"> _MergeViewTests = [] {
    using namespace boost::ut;
    using namespace gr;

    using Pair       = std::pair<std::ptrdiff_t, char>;
    auto compByIndex = [](const Pair& a, const Pair& b) { return a.first < b.first; };

    "MergeView - Pair basics"_test = [&]<class T> {
        T          a{{1, 'a'}, {3, 'a'}, {3, 'a'}, {5, 'a'}};
        T          b{{2, 'b'}, {3, 'b'}, {4, 'b'}, {6, 'b'}};
        T          c{{2, 'c'}, {3, 'c'}, {4, 'c'}, {6, 'c'}};
        std::array inputs{std::views::all(a), std::views::all(b), std::views::all(c)};

        auto outDynamic = inputs | Merge(compByIndex);
        auto outStatic  = inputs | Merge<3>(compByIndex);

        static_assert(std::ranges::range<decltype(outDynamic)>);
        static_assert(std::ranges::view<decltype(outDynamic)>);
        static_assert(std::ranges::forward_range<decltype(outDynamic)>);

        static_assert(std::ranges::range<decltype(outStatic)>);
        static_assert(std::ranges::view<decltype(outStatic)>);
        static_assert(std::ranges::forward_range<decltype(outStatic)>);

        std::vector<Pair> expected{{1, 'a'}, {2, 'b'}, {2, 'c'}, {3, 'a'}, {3, 'a'}, {3, 'b'}, {3, 'c'}, {4, 'b'}, {4, 'c'}, {5, 'a'}, {6, 'b'}, {6, 'c'}};
        expect(std::ranges::equal(outDynamic, expected));
        expect(std::ranges::equal(outStatic, expected));
    } | std::tuple<std::vector<Pair>, std::forward_list<Pair>>{};

    "MergeView - empty + non-empty inputs"_test = [&] {
        std::vector<Pair> a{};
        std::vector<Pair> b{{1, 'b'}, {3, 'b'}};
        std::vector<Pair> c{};
        std::vector<Pair> d{{2, 'd'}};
        std::array        inputs{std::views::all(a), std::views::all(b), std::views::all(c), std::views::all(d)};

        auto out = inputs | Merge(compByIndex);

        std::vector<Pair> expected{{1, 'b'}, {2, 'd'}, {3, 'b'}};
        expect(std::ranges::equal(out, expected));
    };

    "MergeView - all inputs empty"_test = [&] {
        std::vector<Pair> a{}, b{};
        std::array        inputs{std::views::all(a), std::views::all(b)};
        auto              out = inputs | Merge(compByIndex);

        expect(std::ranges::equal(out, std::vector<Pair>{}));
    };

    "MergeView - single input"_test = [&] {
        std::vector<Pair> a{{1, 'a'}, {2, 'a'}, {5, 'a'}};
        std::array        inputs{std::views::all(a)};
        auto              out = inputs | Merge(compByIndex);

        expect(std::ranges::equal(out, a));
    };

    "MergeView - int + default comparator"_test = [&] {
        std::vector<int> a{1, 1, 4, 7};
        std::vector<int> b{2, 3, 5, 6};
        std::vector<int> c{0, 5, 8};
        std::array       inputs{std::views::all(a), std::views::all(b), std::views::all(c)};

        auto out = inputs | Merge(); // default std::ranges::less

        std::vector<int> expected{0, 1, 1, 2, 3, 4, 5, 5, 6, 7, 8};
        expect(std::ranges::equal(out, expected));
    };

    "MergeView - multipass/equality preserving"_test = [&] {
        std::vector<Pair> a{{1, 'a'}, {3, 'a'}};
        std::vector<Pair> b{{2, 'b'}, {4, 'b'}};

        std::array inputs{std::views::all(a), std::views::all(b)};
        auto       out = inputs | Merge(compByIndex);

        auto i1 = std::ranges::begin(out);
        auto i2 = i1;         // copy
        expect(&*i1 == &*i2); // same value after copy
        ++i1;                 // advance first iter
        expect(&*i1 != &*i2); // not equal
        ++i2;                 // advance second iter
        expect(*i1 == *i2);   // same value again

        auto& r1 = *i1;
        auto& r2 = *i1;
        expect(&r1 == &r2);
    };

    "MergeView - sentinel test"_test = [&] {
        std::vector<Pair> a{{1, 'a'}};
        std::array        inputs{std::views::all(a)};
        auto              out = inputs | Merge(compByIndex);

        auto it = std::ranges::begin(out);
        auto ed = std::default_sentinel;

        expect(it != ed);
        ++it;
        expect(it == ed);
        expect(ed == it);
    };

    "MergeView - double pass invariant"_test = [&] {
        std::vector<Pair> a{{1, 'a'}, {3, 'a'}};
        std::vector<Pair> b{{2, 'b'}, {4, 'b'}};
        std::array        inputs{std::views::all(a), std::views::all(b)};
        auto              out = inputs | Merge(compByIndex);

        std::vector<Pair> pass1;
        std::ranges::copy(out, std::back_inserter(pass1));
        std::vector<Pair> pass2;
        std::ranges::copy(out, std::back_inserter(pass2));
        expect(std::ranges::equal(pass1, pass2));
    };
};

int main() {}
