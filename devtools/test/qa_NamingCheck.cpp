#include <boost/ut.hpp>

#include "NameSpelling.hpp"

#include <regex>
#include <string>
#include <string_view>
#include <vector>

using namespace boost::ut;
using namespace devtools;

// Ground truth: the naming decisions of 2026-10-10 (CLAUDE.md §3), their examples taken as written there; the
// fixture under test/naming-check checks the same rules end to end, on declarations libclang sees
const suite<"NamingCheck"> namingCheckTests = [] {
    const auto accepts = [](const std::regex& form, std::string_view name) { return std::regex_match(std::string{name}, form); };

    "mathematical notation: a short capitalised symbol, or a one-letter or capitalised stem with a short subscript"_test = [&] {
        for (const std::string_view name : {"N", "Nfft", "X", "H2", "x_0", "S_k", "sumRe_02"}) {
            expect(accepts(kMathematical, name)) << name;
        }
        for (const std::string_view name : {"item_count", "TooLongName", "x_toolong", "sample_rate"}) {
            expect(!accepts(kMathematical, name)) << name;
        }
    };

    "a type is UpperCamel or a std-style trait; `*_type` only as a member alias"_test = [&] {
        for (const std::string_view name : {"GoodType", "index_t", "is_widget", "has_size"}) {
            expect(accepts(kTypeName, name)) << name;
        }
        for (const std::string_view name : {"bad_type", "badType", "value_type"}) {
            expect(!accepts(kTypeName, name)) << name;
        }
        expect(accepts(kMemberTypeName, "value_type") && accepts(kMemberTypeName, "element_type"));
    };

    "a constant is kUpperCamel, a single letter, or a std-style variable trait"_test = [&] {
        for (const std::string_view name : {"kLimit", "kMaxBufferSize", "N", "is_small_v", "is_trivial"}) {
            expect(accepts(kConstexprName, name)) << name;
        }
        for (const std::string_view name : {"maxCount", "count", "MAX_COUNT", "kmax"}) {
            expect(!accepts(kConstexprName, name)) << name;
        }
    };

    "a non-type template parameter is lowerCamel, UpperCamel or UPPER_SNAKE"_test = [&] {
        for (const std::string_view name : {"nPorts", "Count", "MAX_SIZE", "N"}) {
            expect(accepts(kNonTypeParameter, name)) << name;
        }
        expect(!accepts(kNonTypeParameter, "bad_param"));
    };

    "a `_` member is `_` then either case"_test = [&] {
        for (const std::string_view name : {"_hostState", "_device_state", "_cachedFft"}) {
            expect(accepts(kUnderscored, name)) << name;
        }
        for (const std::string_view name : {"hostState", "_"}) {
            expect(!accepts(kUnderscored, name)) << name;
        }
    };

    "a name splits into lowercase words at `_` and where the case changes"_test = [] {
        expect(wordsOf("parseHTTPHeader") == std::vector<std::string>{"parse", "http", "header"});
        expect(wordsOf("sample_rate") == std::vector<std::string>{"sample", "rate"});
        expect(wordsOf("_hostState") == std::vector<std::string>{"host", "state"});
        expect(wordsOf("kMaxBufferSize") == std::vector<std::string>{"k", "max", "buffer", "size"});
    };

    "a violation proposes the name in the form its rule asks for"_test = [] {
        expect(eq(spellingFor("itemCount", "snake"), std::string{"item_count"}));
        expect(eq(spellingFor("item_total", "lowerCamel"), std::string{"itemTotal"}));
        expect(eq(spellingFor("Bad_function", "lowerCamel"), std::string{"badFunction"}));
        expect(eq(spellingFor("bad_type", "UpperCamel"), std::string{"BadType"}));
        expect(eq(spellingFor("maxCount", "kConstant"), std::string{"kMaxCount"}));
        expect(eq(spellingFor("kmaxCount", "kConstant"), std::string{"kKmaxCount"}));
        expect(eq(spellingFor("itemCount", "_lowerCamel"), std::string{"_itemCount"}));
        expect(eq(spellingFor("make_thing", "UPPER_SNAKE"), std::string{"MAKE_THING"}));
        expect(eq(spellingFor("Third", "lowerCamel"), std::string{"third"}));
    };

    "every proposal is in the form whose pattern checks it"_test = [] {
        for (const std::string_view form : {"UpperCamel", "lowerCamel", "_lowerCamel", "snake", "kConstant", "UPPER_SNAKE"}) {
            for (const std::string_view name : {"itemCount", "item_count", "ItemCount", "ITEM_COUNT", "_item_count", "x"}) {
                const std::string proposal = spellingFor(name, form);
                expect(std::regex_match(proposal, patternOf(form))) << name << "as" << form << "gives" << proposal;
            }
        }
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
