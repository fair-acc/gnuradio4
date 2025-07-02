#include <boost/ut.hpp>

#include <gnuradio-4.0/meta/utils.hpp>

namespace ns0 {
template<typename T, auto X>
struct Notmetaected {};

template<typename>
class Foo {};

template<typename>
union Bar {
    int   x;
    float y;
};

enum Enum { A, B, C };

enum class EnumClass { Foo, Bar };
} // namespace ns0

const boost::ut::suite<"type name tests"> _type_name = [] {
    using namespace boost::ut;
    using namespace ns0;
    using namespace std::string_literals;

    "gr::meta::type_name<T>() - T fundamental types"_test = [] {
        expect(eq(gr::meta::type_name<std::int8_t>(), "int8"s));
        expect(eq(gr::meta::type_name<std::int16_t>(), "int16"s));
        expect(eq(gr::meta::type_name<std::int32_t>(), "int32"s));
        expect(eq(gr::meta::type_name<std::int64_t>(), "int64"s));
        expect(eq(gr::meta::type_name<std::uint8_t>(), "uint8"s));
        expect(eq(gr::meta::type_name<std::uint16_t>(), "uint16"s));
        expect(eq(gr::meta::type_name<std::uint32_t>(), "uint32"s));
        expect(eq(gr::meta::type_name<std::uint64_t>(), "uint64"s));

        expect(eq(gr::meta::type_name<float>(), "float32"s));
        expect(eq(gr::meta::type_name<double>(), "float64"s));
        // portable fixed-size floats (since C++23)
        expect(eq(gr::meta::type_name<std::float32_t>(), "float32"s));
        expect(eq(gr::meta::type_name<std::float64_t>(), "float64"s));

        expect(eq(gr::meta::type_name<std::complex<float>>(), "complex<float32>"s));
        expect(eq(gr::meta::type_name<std::complex<double>>(), "complex<float64>"s));
        expect(eq(gr::meta::type_name<std::complex<std::float32_t>>(), "complex<float32>"s));
        expect(eq(gr::meta::type_name<std::complex<std::float64_t>>(), "complex<float64>"s));
    };

    "gr::meta::type_name<T>()"_test = [] {
        expect(eq(gr::meta::type_name<Notmetaected<int, 5>>(), "ns0::Notmetaected<int32, 5>"s));
        expect(eq(gr::meta::type_name<Foo<int>>(), "ns0::Foo<int32>"s));
        expect(eq(gr::meta::type_name<Bar<float>>(), "ns0::Bar<float32>"s));
        expect(eq(gr::meta::type_name<Enum>(), "ns0::Enum"s));
        expect(eq(gr::meta::type_name<EnumClass>(), "ns0::EnumClass"s));
        expect(eq(gr::meta::type_name<Notmetaected<int, 5>>(), "ns0::Notmetaected<int32, 5>"s));
        expect(eq(gr::meta::type_name<Foo<int>>(), "ns0::Foo<int32>"s));
        expect(eq(gr::meta::type_name<Bar<float>>(), "ns0::Bar<float32>"s));
        expect(eq(gr::meta::type_name<Enum>(), "ns0::Enum"s));
        expect(eq(gr::meta::type_name<EnumClass>(), "ns0::EnumClass"s));

        // some common STL types
        expect(eq(gr::meta::type_name<std::string>(), "string"s));
    };
};

const boost::ut::suite<"shorten type name tests"> _shorten_type_name = [] {
    using namespace boost::ut;
    using namespace ns0;
    using namespace std::string_view_literals;

    "shorten_type_name(sv) – normal cases"_test = [] {
        expect(eq(gr::meta::shorten_type_name("ns1::sns2::cns3::className"sv), "nsc::className"sv));
        expect(eq(gr::meta::shorten_type_name("ns1::sns2::className"sv), "ns::className"sv));
        expect(eq(gr::meta::shorten_type_name("ns1::className"sv), "n::className"sv));
        expect(eq(gr::meta::shorten_type_name("a::b::c::MyThing"sv), "abc::MyThing"sv));
        expect(eq(gr::meta::shorten_type_name("SingleName"sv), "SingleName"sv));
    };

    "shorten_type_name(sv) – edge cases"_test = [] {
        expect(eq(gr::meta::shorten_type_name(""sv), ""sv));
        expect(eq(gr::meta::shorten_type_name("::OnlyLeading"sv), "::OnlyLeading"sv)); // split yields {"", "OnlyLeading"}
        expect(eq(gr::meta::shorten_type_name("trailing::"sv), "t::"sv));              // {"trailing", ""} → t::
        expect(eq(gr::meta::shorten_type_name("::"sv), "::"sv));                       // {"", ""} → ::
        expect(eq(gr::meta::shorten_type_name("::leading::colon::Type"sv), "::lc::Type"sv));
        expect(eq(gr::meta::shorten_type_name("very::long::nested::type::Name"sv), "vlnt::Name"sv));
    };

    "shorten_type_name(sv) – degenerate identifiers"_test = [] {
        expect(eq(gr::meta::shorten_type_name("a::::b::c::Thing"sv), "abc::Thing"sv)); // handles empty segments
        expect(eq(gr::meta::shorten_type_name("::a::b::c::"sv), "::abc::"sv));
        expect(eq(gr::meta::shorten_type_name("::a::b::"sv), "::ab::"sv));
        expect(eq(gr::meta::shorten_type_name("x::::"sv), "x::"sv));
    };
};

int main() { /* tests are statically executed */ }
