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
} // namespace gr::meta

int
main() { /* tests are statically executed */
}