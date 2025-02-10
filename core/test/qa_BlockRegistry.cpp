#include <boost/ut.hpp>

#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

#include <fmt/format.h>

#include <map>

using namespace std::string_literals;
using namespace std::string_view_literals;

// Some helpers from Block.hpp
namespace detail {
template<gr::meta::fixed_string Acc>
struct fixed_string_concat_helper {
    static constexpr auto value = Acc;

    template<gr::meta::fixed_string Append>
    constexpr auto operator%(gr::meta::constexpr_string<Append>) const {
        if constexpr (Acc.empty()) {
            return fixed_string_concat_helper<Append>{};
        } else {
            return fixed_string_concat_helper<Acc + "," + Append>{};
        }
    }
};
template<typename... Types>
constexpr auto encodeListOfTypes() {
    return gr::meta::constexpr_string<(fixed_string_concat_helper<"">{} % ... % gr::refl::type_name<Types>).value>();
}
} // namespace detail

template<typename... Types>
struct BlockParameters : gr::meta::typelist<Types...> {
    static constexpr /*meta::constexpr_string*/ auto toString() { return detail::encodeListOfTypes<Types...>(); }
};
namespace gr::testing {

enum class Color { Red, Green, Blue };

template<typename T>
struct NoBlock {};

template<typename T, int N = 0, bool Flag = true, Color = Color::Red>
struct Something {};

template<typename T>
using Alias = Something<T, 123, false, Color::Blue>;

std::string mapName(const std::string_view name) {
    auto trimmed = [](std::string_view view) {
        while (view.front() == ' ') {
            view.remove_prefix(1);
        }
        while (view.back() == ' ') {
            view.remove_suffix(1);
        }
        return view;
    };

    static const auto mapping = std::map<std::string, std::string, std::less<>>{{gr::meta::type_name<std::int8_t>(), "int8_t"s}, {gr::meta::type_name<std::int16_t>(), "int16_t"s}, {gr::meta::type_name<std::int32_t>(), "int32_t"s}, {gr::meta::type_name<std::int64_t>(), "int64_t"s}, {gr::meta::type_name<std::uint8_t>(), "uint8_t"s}, {gr::meta::type_name<std::uint16_t>(), "uint16_t"s}, {gr::meta::type_name<std::uint32_t>(), "uint32_t"s}, {gr::meta::type_name<std::uint64_t>(), "uint64_t"s}, {gr::meta::type_name<std::string>(), "std::string"s}, {gr::meta::type_name<std::complex<float>>(), "std::complex<float>"s}, {gr::meta::type_name<std::complex<double>>(), "std::complex<double>"s}};
    const auto        it      = mapping.find(name);
    if (it != mapping.end()) {
        return it->second;
    }

    std::string_view view   = name;
    auto             cursor = view.find("<");
    if (cursor == std::string_view::npos) {
        return std::string{name};
    }
    auto base = view.substr(0, cursor);

    view.remove_prefix(cursor + 1);
    if (!view.ends_with(">")) {
        return std::string{name};
    }
    view.remove_suffix(1);
    while (view.back() == ' ') {
        view.remove_suffix(1);
    }

    std::vector<std::string> params;

    std::size_t depth = 0;
    cursor            = 0;

    while (cursor < view.size()) {
        if (view[cursor] == '<') {
            depth++;
        } else if (view[cursor] == '>') {
            depth--;
        } else if (view[cursor] == ',' && depth == 0) {
            auto param = trimmed(view.substr(0, cursor));
            params.push_back(mapName(param));
            view.remove_prefix(cursor + 1);
            cursor = 0;
            continue;
        }
        cursor++;
    }
    params.push_back(mapName(trimmed(view)));
    return fmt::format("{}<{}>", base, fmt::join(params, ", "));
}

struct Registry {
    struct Entry {
        std::string rawName;
        std::string aliasName;
        // factory...
    };

    template<typename T>
    std::string blockTypeName() {
        auto       name = mapName(gr::meta::type_name<T>());
        const auto it   = std::find_if(entries.begin(), entries.end(), [&name](const auto& entry) { return entry.rawName == name; });
        if (it != entries.end() && !it->aliasName.empty()) {
            return it->aliasName;
        }
        return name;
    }

    template<typename TBlock>
    void addBlockType(std::string_view aliasName, std::string_view blockParams) {
        auto rawName = mapName(gr::meta::type_name<TBlock>());
        auto alias   = aliasName.empty() ? "" : mapName(fmt::format("{}<{}>", aliasName, blockParams));
        entries.push_back({rawName, alias});
    }

    std::vector<Entry> entries;
};

template<meta::fixed_string Alias, template<typename...> typename TBlock, typename TBlockParameter0, typename... TBlockParameters>
inline int registerBlockWithAlias(auto& registerInstance) {
    using List0     = std::conditional_t<meta::is_instantiation_of<TBlockParameter0, BlockParameters>, TBlockParameter0, BlockParameters<TBlockParameter0>>;
    using ThisBlock = typename List0::template apply<TBlock>;
    registerInstance.template addBlockType<ThisBlock>(Alias, List0::toString());
    if constexpr (sizeof...(TBlockParameters) != 0) {
        return registerBlock<TBlock, TBlockParameters...>(registerInstance);
    } else {
        return {};
    }
}

// From BlockRegistry.hpp
template<template<typename...> typename TBlock, typename TBlockParameter0, typename... TBlockParameters>
inline int registerBlock(auto& registerInstance) {
    return registerBlockWithAlias<"", TBlock, TBlockParameter0, TBlockParameters...>(registerInstance);
}

const boost::ut::suite<"Type name tests"> typeNameTests = [] {
    using namespace boost::ut;
    using namespace gr;

    "Type names"_test = [] {
        Registry r;
        registerBlock<NoBlock, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, std::complex<float>, std::complex<double>, std::string, Packet<float>, Packet<double>, Tensor<float>, Tensor<double>, DataSet<float>, DataSet<double>>(r);

        // Register Alias for Something, with all but the first templated parameter fixed
        registerBlockWithAlias<"gr::testing::Alias", Alias, uint8_t, float, double>(r);

        // char, int, long etc. are mapped to int8_t, int32_t, int64_t etc.
        // std::string and std::complex don't use implementation-specific type names/namespaces
        expect(eq(r.blockTypeName<NoBlock<uint8_t>>(), "gr::testing::NoBlock<uint8_t>"sv));
        expect(eq(r.blockTypeName<NoBlock<uint16_t>>(), "gr::testing::NoBlock<uint16_t>"sv));
        expect(eq(r.blockTypeName<NoBlock<uint32_t>>(), "gr::testing::NoBlock<uint32_t>"sv));
        expect(eq(r.blockTypeName<NoBlock<uint64_t>>(), "gr::testing::NoBlock<uint64_t>"sv));
        expect(eq(r.blockTypeName<NoBlock<int8_t>>(), "gr::testing::NoBlock<int8_t>"sv));
        expect(eq(r.blockTypeName<NoBlock<int16_t>>(), "gr::testing::NoBlock<int16_t>"sv));
        expect(eq(r.blockTypeName<NoBlock<int32_t>>(), "gr::testing::NoBlock<int32_t>"sv));
        expect(eq(r.blockTypeName<NoBlock<int64_t>>(), "gr::testing::NoBlock<int64_t>"sv));
        expect(eq(r.blockTypeName<NoBlock<float>>(), "gr::testing::NoBlock<float>"sv));
        expect(eq(r.blockTypeName<NoBlock<double>>(), "gr::testing::NoBlock<double>"sv));
        expect(eq(r.blockTypeName<NoBlock<std::complex<float>>>(), "gr::testing::NoBlock<std::complex<float>>"sv));
        expect(eq(r.blockTypeName<NoBlock<std::complex<double>>>(), "gr::testing::NoBlock<std::complex<double>>"sv));
        expect(eq(r.blockTypeName<NoBlock<std::string>>(), "gr::testing::NoBlock<std::string>"sv));
        expect(eq(r.blockTypeName<NoBlock<Packet<float>>>(), "gr::testing::NoBlock<gr::Packet<float>>"sv));
        expect(eq(r.blockTypeName<NoBlock<Packet<double>>>(), "gr::testing::NoBlock<gr::Packet<double>>"sv));
        expect(eq(r.blockTypeName<NoBlock<Tensor<float>>>(), "gr::testing::NoBlock<gr::Tensor<float>>"sv));
        expect(eq(r.blockTypeName<NoBlock<Tensor<double>>>(), "gr::testing::NoBlock<gr::Tensor<double>>"sv));
        expect(eq(r.blockTypeName<NoBlock<DataSet<float>>>(), "gr::testing::NoBlock<gr::DataSet<float>>"sv));
        expect(eq(r.blockTypeName<NoBlock<DataSet<double>>>(), "gr::testing::NoBlock<gr::DataSet<double>>"sv));

        // Something blocks that don't match Alias use the "raw" name
        // Note the use of magic numbers in the enum values (should we register enum types and then map to Color::red etc.?)
        expect(eq(r.blockTypeName<Something<uint8_t>>(), "gr::testing::Something<uint8_t, 0, true, (gr::testing::Color)0>"sv));
        expect(eq(r.blockTypeName<Something<uint8_t, 42>>(), "gr::testing::Something<uint8_t, 42, true, (gr::testing::Color)0>"sv));
        expect(eq(r.blockTypeName<Something<uint8_t, 42, false>>(), "gr::testing::Something<uint8_t, 42, false, (gr::testing::Color)0>"sv));
        expect(eq(r.blockTypeName<Something<uint8_t, 42, false, Color::Blue>>(), "gr::testing::Something<uint8_t, 42, false, (gr::testing::Color)2>"sv));
        // Instances matching Alias use the alias name
        // Note that the mapping between instance types and using-declarations (Alias) should be bijective
        expect(eq(r.blockTypeName<Alias<uint8_t>>(), "gr::testing::Alias<uint8_t>"sv));
    };
};

} // namespace gr::testing

int main() { /* tests are statically executed */ }
