#include <boost/ut.hpp>

#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::meta {

static_assert(!tuple_like<int>);
static_assert(!tuple_like<std::tuple<>>);
static_assert(tuple_like<std::tuple<int>>);
static_assert(tuple_like<std::tuple<int &>>);
static_assert(tuple_like<std::tuple<const int &>>);
static_assert(tuple_like<std::tuple<const int>>);
static_assert(!tuple_like<std::array<int, 0>>);
static_assert(tuple_like<std::array<int, 2>>);
static_assert(tuple_like<std::pair<int, short>>);

static_assert(vector_type<std::vector<std::size_t>>);
static_assert(!vector_type<std::array<std::size_t, 3>>);
static_assert(array_type<std::array<std::size_t, 3>>);
static_assert(!array_type<std::vector<std::size_t>>);

static_assert(array_or_vector_type<std::vector<std::size_t>>);
static_assert(!array_or_vector_type<std::vector<std::size_t>, int>);
static_assert(array_or_vector_type<std::array<std::size_t, 3>>);
static_assert(!array_or_vector_type<std::array<std::size_t, 3>, int>);

static_assert(string_like<std::string>);
static_assert(string_like<std::string_view>);
constexpr auto stringLiteral = "string literal";
static_assert(string_like<decltype(stringLiteral)>);
static_assert(string_like<decltype(fixed_string("abc"))>);
static_assert(!string_like<int>);

class MyClass {
public:
    void
    nonConstFunc() {}

    void
    constFunc() const {}

    void
    constFunc2(int) const {}
};

void
test() {
    // do nothing
}

static_assert(!is_const_member_function(&MyClass::nonConstFunc));
static_assert(is_const_member_function(&MyClass::constFunc));
static_assert(is_const_member_function(&MyClass::constFunc2));
static_assert(!is_const_member_function(&test));

} // namespace gr::meta

int
main() { /* tests are statically executed */
}