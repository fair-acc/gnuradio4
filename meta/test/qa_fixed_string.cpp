#include <boost/ut.hpp>

#include <gnuradio-4.0/meta/utils.hpp>

#include <atomic>

static_assert(gr::meta::fixed_string("") == "");
static_assert(gr::meta::fixed_string("") + "" == "");
static_assert("" + gr::meta::fixed_string("") == "");
static_assert(gr::meta::fixed_string("") + 'a' == "a");
static_assert('a' + gr::meta::fixed_string("") == "a");
static_assert(gr::meta::fixed_string("") + "a" == "a");
static_assert("a" + gr::meta::fixed_string("") == "a");
static_assert(gr::meta::fixed_string("").empty());            // NOSONAR
static_assert(not("a" + gr::meta::fixed_string("")).empty()); // NOSONAR

static_assert(gr::meta::fixed_string("text") == "text");
static_assert(gr::meta::fixed_string("text") <= "text");
static_assert(gr::meta::fixed_string("text") >= "text");
static_assert(gr::meta::fixed_string("text") != "txet");
static_assert(gr::meta::fixed_string("text") < "txet");
static_assert(gr::meta::fixed_string("text") <= "txet");

static_assert(gr::meta::fixed_string("text") != "123");
static_assert(gr::meta::fixed_string("text") != "12345");

static_assert(gr::meta::fixed_string("text") < "texta");
static_assert(gr::meta::fixed_string("text") < "tey");
static_assert(gr::meta::fixed_string("text") <= "texta");
static_assert(gr::meta::fixed_string("text") <= "tey");

static_assert(gr::meta::fixed_string("text") > "teata");
static_assert(gr::meta::fixed_string("text") > "tea");
static_assert(gr::meta::fixed_string("text") >= "teata");
static_assert(gr::meta::fixed_string("text") >= "tea");

static_assert("" == gr::meta::fixed_string(""));
static_assert("text" == gr::meta::fixed_string("text"));
static_assert("text" <= gr::meta::fixed_string("text"));
static_assert("text" >= gr::meta::fixed_string("text"));
static_assert("txet" != gr::meta::fixed_string("text"));
static_assert("txet" > gr::meta::fixed_string("text"));
static_assert("txet" >= gr::meta::fixed_string("text"));

static_assert("123" != gr::meta::fixed_string("text"));
static_assert("12345" != gr::meta::fixed_string("text"));

static_assert("texta" > gr::meta::fixed_string("text"));
static_assert("tey" > gr::meta::fixed_string("text"));
static_assert("texta" >= gr::meta::fixed_string("text"));
static_assert("tey" >= gr::meta::fixed_string("text"));

static_assert("teata" < gr::meta::fixed_string("text"));
static_assert("tea" < gr::meta::fixed_string("text"));
static_assert("teata" <= gr::meta::fixed_string("text"));
static_assert("tea" <= gr::meta::fixed_string("text"));

static_assert(gr::meta::constexpr_string<"">() == "");
static_assert(gr::meta::constexpr_string<"text">() == "text");
static_assert(gr::meta::constexpr_string<"text">() <= "text");
static_assert(gr::meta::constexpr_string<"text">() <= gr::meta::fixed_string("text"));
static_assert(gr::meta::constexpr_string<"text">() <= gr::meta::constexpr_string<"text">());
static_assert(gr::meta::constexpr_string<"text">() >= "text");
static_assert(gr::meta::constexpr_string<"text">() != "txet");
static_assert(gr::meta::constexpr_string<"text">() < "txet");
static_assert(gr::meta::constexpr_string<"text">() <= "txet");

static_assert(gr::meta::constexpr_string<"text">() != "123");
static_assert(gr::meta::constexpr_string<"text">() != "12345");

static_assert(gr::meta::constexpr_string<"text">() < "texta");
static_assert(gr::meta::constexpr_string<"text">() < "tey");
static_assert(gr::meta::constexpr_string<"text">() <= "texta");
static_assert(gr::meta::constexpr_string<"text">() <= "tey");

static_assert(gr::meta::constexpr_string<"text">() > "teata");
static_assert(gr::meta::constexpr_string<"text">() > "tea");
static_assert(gr::meta::constexpr_string<"text">() >= "teata");
static_assert(gr::meta::constexpr_string<"text">() >= "tea");

static_assert("" == gr::meta::constexpr_string<"">());
static_assert("text" == gr::meta::constexpr_string<"text">());
static_assert("text" <= gr::meta::constexpr_string<"text">());
static_assert("text" >= gr::meta::constexpr_string<"text">());
static_assert("txet" != gr::meta::constexpr_string<"text">());
static_assert("txet" > gr::meta::constexpr_string<"text">());
static_assert("txet" >= gr::meta::constexpr_string<"text">());

static_assert("123" != gr::meta::constexpr_string<"text">());
static_assert("12345" != gr::meta::constexpr_string<"text">());

static_assert("texta" > gr::meta::constexpr_string<"text">());
static_assert("tey" > gr::meta::constexpr_string<"text">());
static_assert("texta" >= gr::meta::constexpr_string<"text">());
static_assert("tey" >= gr::meta::constexpr_string<"text">());

static_assert("teata" < gr::meta::constexpr_string<"text">());
static_assert("tea" < gr::meta::constexpr_string<"text">());
static_assert("teata" <= gr::meta::constexpr_string<"text">());
static_assert("tea" <= gr::meta::constexpr_string<"text">());

static_assert(gr::meta::constexpr_string_from_number_v<0> == "0");
static_assert(gr::meta::constexpr_string_from_number_v<7> == "7");
static_assert(gr::meta::constexpr_string_from_number_v<9> == "9");
static_assert(gr::meta::constexpr_string_from_number_v<10> == "10");
static_assert(gr::meta::constexpr_string_from_number_v<-1> == "-1");
static_assert(gr::meta::constexpr_string_from_number_v<123> == "123");
static_assert(gr::meta::constexpr_string_from_number_v<(1u << 31)> == "2147483648");
static_assert(gr::meta::constexpr_string_from_number_v<int(1u << 31)> == "-2147483648");

static_assert("ab" + gr::meta::constexpr_string<"cd">() + "ef" == "abcdef");
static_assert('a' + gr::meta::constexpr_string<"cd">() + 'f' == "acdf");

static_assert(std::constructible_from<std::string, gr::meta::constexpr_string<"foo">>);

constexpr auto f() {
    const gr::meta::constexpr_string<"Hello World!"> as_type;
    constexpr auto                                   and_now = "4× " + as_type + " How do you do?";
    return gr::meta::constexpr_string<and_now>();
}

namespace {
auto check = [](const gr::Ratio& r, std::int64_t numerator, std::int64_t denominator, std::source_location loc = std::source_location::current()) {
    boost::ut::expect(boost::ut::eq(r.num(), numerator), loc) << std::format("numerator mismatch: {} != {}", r.num(), numerator);
    boost::ut::expect(boost::ut::eq(r.den(), denominator), loc) << std::format("denominator mismatch: {} != {}", r.den(), denominator);
};

consteval bool ceq(const gr::Ratio& r, std::int64_t n, std::int64_t d) { return r.num() == n && r.den() == d; }
} // namespace

const boost::ut::suite<"gr::on_scope_exit"> _on_scope_exit = [] { // NOSONAR
    using namespace boost::ut;

    "calls functor at end of scope"_test = [] {
        bool called = false;
        {
            gr::on_scope_exit guard([&] { called = true; });
            expect(!called);
        }
        expect(called);
    };

    "fires during exception unwind"_test = [] {
        bool called = false;
        try {
            gr::on_scope_exit guard([&] { called = true; });
            throw 42;
        } catch (...) { // NOSONAR
        }
        expect(called);
    };

    "called exactly once"_test = [] {
        std::atomic_int cnt{0};
        {
            gr::on_scope_exit g([&] { ++cnt; });
        }
        expect(eq(cnt.load(), 1));
    };
};

const boost::ut::suite<"gr::Ratio"> _ratio = [] { // NOSONAR
    using namespace boost::ut;
    using namespace std::string_view_literals;

    "default constructor"_test = [] {
        constexpr gr::Ratio r{};
        static_assert(ceq(r, 1, 1));
    };

    "normalising constructor"_test = [] {
        constexpr gr::Ratio r1{4, 8, gr::normalise};
        static_assert(ceq(r1, 1, 2));

        constexpr gr::Ratio r2{-6, -9, gr::normalise};
        static_assert(ceq(r2, 2, 3));

        constexpr gr::Ratio r3{6, -9, gr::normalise};
        static_assert(ceq(r3, -2, 3));
    };

    "from std::ratio"_test = [] {
        constexpr auto r = gr::Ratio::from<std::ratio<1, 1000>>();
        static_assert(ceq(r, 1, 1000));
        check(r, 1, 1000);
    };

    "value() / reciprocal()"_test = [] {
        gr::Ratio r{3, 4};
        expect(eq(r.value<double>(), 0.75_d));

        gr::Ratio inv = r.reciprocal();
        check(inv, 4, 3);
    };

    "arithmetic add/sub"_test = [] {
        gr::Ratio a{3, 4};
        gr::Ratio b{2, 3};
        check(a + b, 17, 12);
        check(a - b, 1, 12);

        gr::Ratio c{-1, 6};
        gr::Ratio d{1, 3};
        check(c + d, 1, 6);
        check(c - d, -3, 6);
    };

    "arithmetic mul/div"_test = [] {
        gr::Ratio a{3, 4};
        gr::Ratio b{2, 3};
        check(a * b, 1, 2);
        check(a / b, 9, 8);
    };

    "spaceship operator"_test = [] {
        gr::Ratio a{1, 3};
        gr::Ratio b{2, 5};
        gr::Ratio c{2, 5};
        expect(a < b);
        expect(!(b < a));
        expect(b == c);
        expect((a <=> b) == std::strong_ordering::less);
        expect((b <=> c) == std::strong_ordering::equal);
        expect((b <=> a) == std::strong_ordering::greater);
    };

    "formatting"_test = [] {
        gr::Ratio r1{5, 1};
        gr::Ratio r2{7, 3};
        expect(std::format("{}", r1) == "5"sv);
        expect(std::format("{}", r2) == "7/3"sv);
    };

    "parse valid/invalid"_test = [] {
        auto ok1 = gr::Ratio::parse("42/105");
        expect(ok1.has_value());
        check(*ok1, 42, 105);

        auto ok2 = gr::Ratio::parse("7");
        expect(ok2.has_value());
        check(*ok2, 7, 1);

        auto bad1 = gr::Ratio::parse("foo/bar");
        expect(!bad1.has_value());

        auto bad2 = gr::Ratio::parse("1/0");
        expect(!bad2.has_value());
    };

    "parsing constructor"_test = [] {
        static_assert(ceq(gr::Ratio{"42/105"}, 42, 105));
        static_assert(ceq(gr::Ratio{"42/105", gr::normalise}, 2, 5));

        static_assert(ceq(gr::Ratio{"7"}, 7, 1));
        static_assert(ceq(gr::Ratio{"7", gr::normalise}, 7, 1));

        // runtime-failure
        check(gr::Ratio{"foo/bar"}, 0, 0);
        check(gr::Ratio{"1/0"}, 0, 0);

        // the following should not compile
        // static_assert(ceq(gr::Ratio{"foo/bar"}, 7, 1));
        // static_assert(ceq(gr::Ratio{"1/0"}, 7, 1));
    };

    "overflow-minimising ops (sanity)"_test = [] {
        gr::Ratio a{9'000'000, 12'000'000};
        a.normalise(); // reduces to 3/4
        gr::Ratio b{8'000'000, 9'000'000};
        b.normalise(); // reduces to 8/9
        check(a, 3, 4);
        check(b, 8, 9);
        check(a * b, 2, 3);
        check(a / b, 27, 32);
    };

    "unary minus"_test = [] {
        gr::Ratio r{1, 7};
        auto      neg = -r;
        check(neg, -1, 7);
    };
};

int main() { /* tests are statically executed */ }
