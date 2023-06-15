#include "benchmark.hpp"

#include <algorithm>
#include <boost/ut.hpp>

#include "bm_test_helper.hpp"
#include "timingctx.hpp"

inline constexpr std::size_t N_ITER    = 10;
inline constexpr std::size_t N_SAMPLES = gr::util::round_up(10'000, 1024);

// a simple parse function that just stores every key value pair from a selector string "key0=val0:key1=val1..."
auto parsePred = [](auto t) {
    auto sel     = t->selector.value;
    auto strView = std::string_view{ sel.data(), sel.data() + sel.length() };
    while (true) {
        const auto posColon              = strView.find(':');
        const auto tag                   = posColon != std::string_view::npos ? strView.substr(0, posColon) : strView;

        const auto posEqual              = tag.find('=');

        const auto key                   = tag.substr(0, posEqual);
        const auto valueString           = tag.substr(posEqual + 1, tag.length() - posEqual - 1);

        int32_t    value                 = -1;
        std::ignore                      = std::from_chars(valueString.begin(), valueString.end(), value);
        t->_identifier[std::string(key)] = value;

        if (posColon == std::string_view::npos) {
            return;
        }

        // advance to after the ":"
        strView.remove_prefix(posColon + 1);
    }
};

// a simple match function that checks if all ids in lhs are the same in rhs (this is not a symmetrical relation)
auto matchPred = [](const auto lhs, const auto rhs) { return std::all_of(lhs._identifier.cbegin(), lhs._identifier.cend(), [&](auto v) { return rhs._identifier[v.first] == v.second; }); };

inline const boost::ut::suite _constexpr_bm = [] {
    using namespace boost::ut;
    using namespace benchmark;
    using namespace fair::graph;
    using namespace std::literals::string_literals;

    static const std::array selectors = {
        ""s, "A=0"s, "A=1"s, "A=0:B=1"s, "A=0:B=0:C=0"s, "A=0:B=0:C=1"s, "A=0:B=0:C=1:D=0"s, "A=1:B=0:C=1:D=0"s,
    };
    {
        "timingctx"_benchmark.repeat<N_ITER>(N_SAMPLES) = []() {
            std::size_t matchCount = 0;
            for (std::size_t dist = 0; dist < selectors.size(); ++dist) {
                for (std::size_t first = 0; first < selectors.size(); ++first) {
                    const auto second = (first + dist) % selectors.size();
                    auto       a      = TimingCtx(parsePred, matchPred, selectors[first]);
                    auto       b      = TimingCtx(parsePred, matchPred, selectors[second]);
                    if (a.matches(b)) {
                        matchCount++;
                    }
                }
            }

            expect(gt(matchCount, static_cast<std::size_t>(0)));
        };
    }
};

int
main() { /* not needed by the UT framework */
}
