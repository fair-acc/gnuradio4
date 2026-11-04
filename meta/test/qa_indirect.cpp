#include <boost/ut.hpp>

#include <memory>
#include <type_traits>
#include <utility>

#include <gnuradio-4.0/meta/indirect.hpp>

namespace gr::meta::test {

struct CopyConstructibleButNotAssignable {
    int value;

    explicit CopyConstructibleButNotAssignable(int value_) : value(value_) {}

    CopyConstructibleButNotAssignable(const CopyConstructibleButNotAssignable&)            = default;
    CopyConstructibleButNotAssignable& operator=(const CopyConstructibleButNotAssignable&) = delete;
};

const boost::ut::suite<"gr::meta::indirect"> indirectTests = [] {
    using namespace boost::ut;

    "copy construction from a uses the copy constructor"_test = [] {
        gr::meta::indirect<int> original(42);
        gr::meta::indirect<int> copy(original);

        expect(eq(*copy, 42));

        *original = 7;
        expect(eq(*copy, 42));
    };

    "copy assignment only does not need T to be copy assignable"_test = [] {
        gr::meta::indirect<CopyConstructibleButNotAssignable> lhs(std::in_place, 1);
        gr::meta::indirect<CopyConstructibleButNotAssignable> rhs(std::in_place, 2);

        lhs = rhs;
        expect(eq(lhs->value, 2));

        rhs->value = 3;
        expect(eq(lhs->value, 2));
    };

    "single-argument construction is explicit"_test = [] { expect(!std::is_convertible_v<int, gr::meta::indirect<int>>); };
};

} // namespace gr::meta::test

int main() { /* tests are statically executed */ }
