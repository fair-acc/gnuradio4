#include <gnuradio-4.0/meta/reflection.hpp>

static_assert(gr::refl::type_name<int> == gr::meta::constexpr_string<"int">());
static_assert(gr::refl::type_name<float> == "float");
static_assert(gr::refl::type_name<std::string> == "std::string");
static_assert(gr::refl::type_name<std::array<int, 4>> == "std::array<int, 4>");
static_assert(gr::refl::type_name<std::vector<int>> == "std::vector<int>");
static_assert(gr::refl::type_name<std::complex<float>> == "std::complex<float>");

namespace ns0
{
  template <typename T, auto X>
    struct NotReflected
    {};

  template <typename>
    class Foo
    {};

  template <typename>
    union Bar
    { int x; float y; };

  enum Enum
  { A, B, C };

  enum class EnumClass
  { Foo, Bar };

  static_assert(gr::refl::class_name<NotReflected<int, 5>> == "ns0::NotReflected");
  static_assert(gr::refl::class_name<Foo<int>> == "ns0::Foo");
  static_assert(gr::refl::class_name<Bar<float>> == "ns0::Bar");
  static_assert(gr::refl::class_name<Enum> == "ns0::Enum");
  static_assert(gr::refl::class_name<EnumClass> == "ns0::EnumClass");
  static_assert(gr::refl::type_name<NotReflected<int, 5>> == "ns0::NotReflected<int, 5>");
  static_assert(gr::refl::type_name<Foo<int>> == "ns0::Foo<int>");
  static_assert(gr::refl::type_name<Bar<float>> == "ns0::Bar<float>");
  static_assert(gr::refl::type_name<Enum> == "ns0::Enum");
  static_assert(gr::refl::type_name<EnumClass> == "ns0::EnumClass");

  static_assert(gr::refl::enum_name<A> == "ns0::A");
  static_assert(gr::refl::enum_name<B> == "ns0::B");
  static_assert(gr::refl::enum_name<C> == "ns0::C");
  static_assert(gr::refl::enum_name<EnumClass::Foo> == "ns0::EnumClass::Foo");

  static_assert(gr::refl::nttp_name<1> == "1" or gr::refl::nttp_name<1> == "0x1");
  static_assert(gr::refl::nttp_name<1u> == "1" or gr::refl::nttp_name<1u> == "1U"
                  or gr::refl::nttp_name<1u> == "0x1");
  static_assert(gr::refl::nttp_name<A> == gr::refl::enum_name<A>);
}

struct Test
{
  int a, b, foo;

  GR_MAKE_REFLECTABLE(Test, a, b, foo);
};

static_assert(std::same_as<gr::refl::base_type<Test>, void>);
static_assert(gr::refl::data_member_count<Test> == 3);
static_assert(gr::refl::data_member_name<Test, 0> == "a");
static_assert(gr::refl::data_member_name<Test, 1> == "b");
static_assert(gr::refl::data_member_name<Test, 2> == "foo");
static_assert(gr::refl::class_name<Test> == "Test");
static_assert(std::same_as<gr::refl::data_member_type<Test, 0>, int>);
static_assert(std::same_as<gr::refl::data_member_type<Test, 1>, int>);
static_assert(std::same_as<gr::refl::data_member_type<Test, 2>, int>);
static_assert(std::same_as<gr::refl::data_member_type<Test, "a">, int>);
static_assert(std::same_as<gr::refl::data_member_type<Test, "b">, int>);
static_assert(std::same_as<gr::refl::data_member_type<Test, "foo">, int>);

static_assert([] {
  Test t {1, 2, 3};
  if (&gr::refl::data_member<0>(t) != &t.a)
    return false;
  if (&gr::refl::data_member<1>(t) != &t.b)
    return false;
  if (&gr::refl::data_member<2>(t) != &t.foo)
    return false;
  if (&gr::refl::data_member<"a">(t) != &t.a)
    return false;
  if (&gr::refl::data_member<"b">(t) != &t.b)
    return false;
  if (&gr::refl::data_member<"foo">(t) != &t.foo)
    return false;
  gr::refl::data_member<"b">(t) = -1;
  if (t.b != -1)
    return false;
  return true;
}());

int
test0(Test& t)
{ return gr::refl::data_member<"foo">(t); }

auto
test1()
{ return gr::refl::data_member_name<Test, 1>; }

struct Derived : Test
{
  float in;
  double out;

  GR_MAKE_REFLECTABLE(Derived, in, out);
};

static_assert(gr::refl::reflectable<Derived>);
static_assert(std::same_as<gr::refl::base_type<Derived>, Test>);
static_assert(gr::refl::data_member_count<Derived> == 5);
static_assert(gr::refl::data_member_name<Derived, 0> == "a");
static_assert(gr::refl::data_member_name<Derived, 1> == "b");
static_assert(gr::refl::data_member_name<Derived, 2> == "foo");
static_assert(gr::refl::data_member_name<Derived, 3> == "in");
static_assert(gr::refl::data_member_name<Derived, 4> == "out");
static_assert(gr::refl::class_name<Derived> == "Derived");
static_assert(gr::refl::class_name<gr::refl::base_type<Derived>> == "Test");
static_assert(std::same_as<gr::refl::data_member_type<Derived, 0>, int>);
static_assert(std::same_as<gr::refl::data_member_type<Derived, 1>, int>);
static_assert(std::same_as<gr::refl::data_member_type<Derived, 2>, int>);
static_assert(std::same_as<gr::refl::data_member_type<Derived, 3>, float>);
static_assert(std::same_as<gr::refl::data_member_type<Derived, 4>, double>);
static_assert(std::same_as<gr::refl::data_member_type<Derived, "a">, int>);
static_assert(std::same_as<gr::refl::data_member_type<Derived, "b">, int>);
static_assert(std::same_as<gr::refl::data_member_type<Derived, "foo">, int>);
static_assert(std::same_as<gr::refl::data_member_type<Derived, "in">, float>);
static_assert(std::same_as<gr::refl::data_member_type<Derived, "out">, double>);

static_assert([] {
  Derived t {{1, 2, 3}, 1.1f, 2.2};
  if (&gr::refl::data_member<0>(t) != &t.a)
    return false;
  if (&gr::refl::data_member<1>(t) != &t.b)
    return false;
  if (&gr::refl::data_member<2>(t) != &t.foo)
    return false;
  if (&gr::refl::data_member<3>(t) != &t.in)
    return false;
  if (&gr::refl::data_member<4>(t) != &t.out)
    return false;
  if (&gr::refl::data_member<"a">(t) != &t.a)
    return false;
  if (&gr::refl::data_member<"b">(t) != &t.b)
    return false;
  if (&gr::refl::data_member<"foo">(t) != &t.foo)
    return false;
  if (&gr::refl::data_member<"in">(t) != &t.in)
    return false;
  if (&gr::refl::data_member<"out">(t) != &t.out)
    return false;
  gr::refl::data_member<"in">(t) *= -1;
  if (t.in != -1.1f)
    return false;

  if (&std::get<0>(gr::refl::all_data_members(t)) != &t.a)
    return false;
  if (&std::get<1>(gr::refl::all_data_members(t)) != &t.b)
    return false;
  if (&std::get<2>(gr::refl::all_data_members(t)) != &t.foo)
    return false;
  if (&std::get<3>(gr::refl::all_data_members(t)) != &t.in)
    return false;
  if (&std::get<4>(gr::refl::all_data_members(t)) != &t.out)
    return false;

  return true;
}());

struct Further : Derived
{
  char c;
  GR_MAKE_REFLECTABLE(Further, c);
};

static_assert(gr::refl::reflectable<Further>);
static_assert(std::same_as<gr::refl::base_type<Further>, Derived>);

struct AndAnother : Further
{
  // static data members are also supported
  inline static int baz = 1;

  GR_MAKE_REFLECTABLE(AndAnother, baz);
};

static_assert(gr::refl::reflectable<AndAnother>);
static_assert(std::same_as<gr::refl::base_type<AndAnother>, Further>);

template <typename T, size_t Idx>
using only_floats = std::is_same<gr::refl::data_member_type<T, Idx>, float>;

template <typename T, size_t Idx>
using only_ints = std::is_same<gr::refl::data_member_type<T, Idx>, int>;

template <typename T, size_t Idx>
using name_is_out = std::bool_constant<gr::refl::data_member_name<T, Idx> == "out">;

static_assert(gr::refl::find_data_members<AndAnother, only_floats> == std::array<size_t, 1>{3});
static_assert(gr::refl::find_data_members<AndAnother, only_ints>
                == std::array<size_t, 4>{0, 1, 2, 6});
static_assert(gr::refl::find_data_members<AndAnother, name_is_out> == std::array<size_t, 1>{4});
static_assert(gr::refl::find_data_members_by_type<AndAnother, std::is_floating_point>
                == std::array<size_t, 2>{3, 4});

static_assert([] {
  AndAnother x;
  auto& value = gr::refl::data_member<"baz">(x);
  return &value == &AndAnother::baz;
}());

#if __clang__ < 18
#define ARRAY std::array
#else
#define ARRAY
#endif
static_assert(std::same_as<gr::refl::data_member_types<AndAnother, ARRAY{0, 1}>,
                           gr::meta::typelist<int, int>>);
static_assert(std::same_as<gr::refl::data_member_types<AndAnother, ARRAY{2, 5, 4}>,
                           gr::meta::typelist<int, char, double>>);
static_assert(std::same_as<gr::refl::data_member_types<AndAnother>,
                           gr::meta::typelist<int, int, int, float, double, char, int>>);
#undef ARRAY

static_assert([] {
  size_t sum = 0;
  gr::refl::for_each_data_member_index<AndAnother>([&sum](auto idx) {
    sum += idx;
  });
  return sum;
}() == 0 + 1 + 2 + 3 + 4 + 5 + 6);

namespace ns
{
  template <typename T>
  struct Type
  {
    int blah;
    GR_MAKE_REFLECTABLE(ns::Type<T>, blah);
  };

  struct Type2 : Type<Type2>
  {
    GR_MAKE_REFLECTABLE(ns::Type2);
  };
}

struct Type3 : ns::Type<Type3>
{
  float x, y, z;
  GR_MAKE_REFLECTABLE(Type3, x, y, z);
};

static_assert(gr::refl::reflectable<ns::Type<int>>);
static_assert(std::same_as<gr::refl::base_type<ns::Type<int>>, void>);
static_assert(gr::refl::data_member_count<ns::Type<int>> == 1);
static_assert(gr::refl::type_name<ns::Type<int>> == "ns::Type<int>");
static_assert(gr::refl::class_name<ns::Type<int>> == "ns::Type");

static_assert(gr::refl::reflectable<ns::Type2>);
static_assert(std::same_as<gr::refl::base_type<ns::Type2>, ns::Type<ns::Type2>>);
static_assert(gr::refl::data_member_count<ns::Type2> == 1);
static_assert(gr::refl::class_name<ns::Type2> == "ns::Type2");

static_assert(gr::refl::reflectable<Type3>);
static_assert(gr::refl::class_name<Type3> == "Type3");
static_assert(std::same_as<gr::refl::base_type<Type3>, ns::Type<Type3>>);
static_assert(gr::refl::data_member_count<Type3> == 4);

const char*
string_test()
{ return gr::refl::class_name<ns::Type<int>>.data(); }

std::string_view
string_test2()
{ return gr::refl::class_name<ns::Type<float>>; }

const char*
member_name_string0()
{ return gr::refl::data_member_name<ns::Type<char>, 0>.data(); }

int main() { /* tests are statically executed */ }
