#include <boost/ut.hpp>
#include <format>
#include <forward_list>
#include <gnuradio-4.0/meta/RangesHelper.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>
#include <vector>

const boost::ut::suite<"AdjacentDeduplicateView tests"> _AdjacentDeduplicateViewTests = [] {
    using namespace boost::ut;
    using namespace gr;

    "basic test - single element"_test = [] {
        std::vector<int> v{42};
        auto             outView = v | AdjacentDeduplicateView();
        expect(std::vector(outView.begin(), outView.end()) == std::vector<int>{42});
    };

    "basic test - all equal elements"_test = [] {
        std::vector<int> v(7, 5);
        auto             outView = v | AdjacentDeduplicateView();
        expect(std::vector(outView.begin(), outView.end()) == std::vector<int>{5});
    };

    "basic test - default ctor"_test = [] {
        std::vector<int> v{1, 1, 2, 2, 2, 3, 2, 2};
        auto             outView = v | AdjacentDeduplicateView();
        expect(std::vector(outView.begin(), outView.end()) == std::vector<int>{1, 2, 3, 2});
    };

    "basic test - ctor with Eq"_test = [] {
        std::vector<int> v{5, 6, 6, 6, 6, 6, 7, 7};
        auto             outView = v | AdjacentDeduplicateView(std::ranges::equal_to{});
        expect(std::vector(outView.begin(), outView.end()) == std::vector<int>{5, 6, 7});
    };

    "basic test - forward_list"_test = [] {
        std::forward_list<int> fl{1, 1, 1, 2, 3, 3};
        auto                   outView = fl | AdjacentDeduplicateView{};
        expect(std::vector(outView.begin(), outView.end()) == std::vector<int>({1, 2, 3}));
    };

    "basic test - custom predicate on Tag"_test = [] {
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

    "basic test - pipeline composition"_test = [] {
        std::vector<int> v{0, 0, 1, 1, 2, 2};
        auto             outView = v | std::views::transform([](int x) { return x + 1; }) | AdjacentDeduplicateView{} | std::views::transform([](int x) { return x * 10; });
        expect(std::vector(outView.begin(), outView.end()) == std::vector<int>({10, 20, 30}));
    };
};

int main() {}
