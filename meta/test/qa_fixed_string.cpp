#include <gnuradio-4.0/meta/utils.hpp>

static_assert(gr::meta::fixed_string("") == "");
static_assert(gr::meta::fixed_string("") + "" == "");
static_assert("" + gr::meta::fixed_string("") == "");
static_assert(gr::meta::fixed_string("") + 'a' == "a");
static_assert('a' + gr::meta::fixed_string("") == "a");
static_assert(gr::meta::fixed_string("") + "a" == "a");
static_assert("a" + gr::meta::fixed_string("") == "a");
static_assert(gr::meta::fixed_string("").empty());
static_assert(not ("a" + gr::meta::fixed_string("")).empty());

static_assert(gr::meta::fixed_string("text") == "text");
static_assert(gr::meta::fixed_string("text") <= "text");
static_assert(gr::meta::fixed_string("text") >= "text");
static_assert(gr::meta::fixed_string("text") != "txet");
static_assert(gr::meta::fixed_string("text") <  "txet");
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

static_assert(""     == gr::meta::fixed_string(""));
static_assert("text" == gr::meta::fixed_string("text"));
static_assert("text" <= gr::meta::fixed_string("text"));
static_assert("text" >= gr::meta::fixed_string("text"));
static_assert("txet" != gr::meta::fixed_string("text"));
static_assert("txet" >  gr::meta::fixed_string("text"));
static_assert("txet" >= gr::meta::fixed_string("text"));

static_assert("123"   != gr::meta::fixed_string("text"));
static_assert("12345" != gr::meta::fixed_string("text"));

static_assert("texta" >  gr::meta::fixed_string("text"));
static_assert("tey"   >  gr::meta::fixed_string("text"));
static_assert("texta" >= gr::meta::fixed_string("text"));
static_assert("tey"   >= gr::meta::fixed_string("text"));

static_assert("teata" <  gr::meta::fixed_string("text"));
static_assert("tea"   <  gr::meta::fixed_string("text"));
static_assert("teata" <= gr::meta::fixed_string("text"));
static_assert("tea"   <= gr::meta::fixed_string("text"));

static_assert(gr::meta::constexpr_string<"">() == "");
static_assert(gr::meta::constexpr_string<"text">() == "text");
static_assert(gr::meta::constexpr_string<"text">() <= "text");
static_assert(gr::meta::constexpr_string<"text">() <= gr::meta::fixed_string("text"));
static_assert(gr::meta::constexpr_string<"text">() <= gr::meta::constexpr_string<"text">());
static_assert(gr::meta::constexpr_string<"text">() >= "text");
static_assert(gr::meta::constexpr_string<"text">() != "txet");
static_assert(gr::meta::constexpr_string<"text">() <  "txet");
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
static_assert("txet" >  gr::meta::constexpr_string<"text">());
static_assert("txet" >= gr::meta::constexpr_string<"text">());

static_assert("123"   != gr::meta::constexpr_string<"text">());
static_assert("12345" != gr::meta::constexpr_string<"text">());

static_assert("texta" >  gr::meta::constexpr_string<"text">());
static_assert("tey"   >  gr::meta::constexpr_string<"text">());
static_assert("texta" >= gr::meta::constexpr_string<"text">());
static_assert("tey"   >= gr::meta::constexpr_string<"text">());

static_assert("teata" <  gr::meta::constexpr_string<"text">());
static_assert("tea"   <  gr::meta::constexpr_string<"text">());
static_assert("teata" <= gr::meta::constexpr_string<"text">());
static_assert("tea"   <= gr::meta::constexpr_string<"text">());

static_assert(gr::meta::constexpr_string_from_number_v<0> == "0");
static_assert(gr::meta::constexpr_string_from_number_v<7> == "7");
static_assert(gr::meta::constexpr_string_from_number_v<9> == "9");
static_assert(gr::meta::constexpr_string_from_number_v<10> == "10");
static_assert(gr::meta::constexpr_string_from_number_v<-1> == "-1");
static_assert(gr::meta::constexpr_string_from_number_v<123> == "123");
static_assert(gr::meta::constexpr_string_from_number_v<(1u<<31)> == "2147483648");
static_assert(gr::meta::constexpr_string_from_number_v<int(1u<<31)> == "-2147483648");

static_assert("ab" + gr::meta::constexpr_string<"cd">() + "ef" == "abcdef");
static_assert('a' + gr::meta::constexpr_string<"cd">() + 'f' == "acdf");

static_assert(std::constructible_from<std::string, gr::meta::constexpr_string<"foo">>);

constexpr auto
f()
{
  const gr::meta::constexpr_string<"Hello World!"> as_type;
  //constexpr auto as_arg = as_type + ' ' + as_type;
  constexpr auto and_now = "4× " + as_type + " How do you do?";// + as_arg;
  return gr::meta::constexpr_string<and_now>();
}

int main() { /* tests are statically executed */ }
