#ifndef GNURADIO_ALGORITHM_EXPRESSIONSYMBOLS_HPP
#define GNURADIO_ALGORITHM_EXPRESSIONSYMBOLS_HPP

#include <cmath>
#include <concepts>
#include <cstdint>
#include <expected>
#include <format>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <exprtk.hpp>

#include <gnuradio-4.0/PmtTypeHelpers.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/annotated.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

namespace gr::algorithm {

template<std::floating_point T>
void bindSettings(exprtk::symbol_table<T>& table, auto& block) {
    using TBlock = std::remove_cvref_t<decltype(block)>;
    refl::for_each_data_member_index<TBlock>([&](auto kIdx) {
        auto& member = refl::data_member<kIdx>(block);
        using Member = std::remove_cvref_t<decltype(member)>;
        if constexpr (AnnotatedType<Member>) {
            if constexpr (std::same_as<typename Member::value_type, T>) {
                const std::string name(refl::data_member_name<TBlock, kIdx>.view());
                table.add_variable(name, member.value);
                if (const std::string alias(member.description()); alias != name && !table.symbol_exists(alias)) {
                    std::ignore = table.add_variable(alias, member.value);
                }
            }
        } else if constexpr (std::same_as<Member, T>) {
            table.add_variable(std::string(refl::data_member_name<TBlock, kIdx>.view()), member);
        }
    });
}

enum class UnknownIdentifiers : std::uint8_t { refuse, readFromTags };

/**
 * One identifier namespace for exprtk expressions: the caller's variables (e.g. the sample `x`) and a block's settings
 * (`bindSettings`, under their snake_case name and, if a valid identifier, their `description()`) are bound into `table` before `compile`.
 * With `UnknownIdentifiers::readFromTags`, any other identifier is a tag field read by its short key (`gr:` stripped) from the tags passed to `assign`, NaN until a tag first carries it; by default it is a compile error.
 *
 *   symbols.table.add_variable("x", sample);
 *   std::ignore = symbols.compile("x > threshold && sample_rate > 1e3", expression, UnknownIdentifiers::readFromTags);
 */
template<std::floating_point T>
struct ExpressionSymbols {
    exprtk::symbol_table<T>                          table;
    std::vector<std::pair<std::string, std::string>> tagFields; /**< identifier, prefixed tag key */

    /** every error of the parse, one line each, with its position; `vectorCheck` (optional) guards the expression's vector accesses at run time */
    [[nodiscard]] std::expected<void, std::string> compile(const std::string& source, exprtk::expression<T>& expression, UnknownIdentifiers unknown = UnknownIdentifiers::refuse, exprtk::vector_access_runtime_check* vectorCheck = nullptr) {
        table.add_constants();
        std::vector<std::string> known;
        table.get_variable_list(known);
        expression.register_symbol_table(table);

        exprtk::parser<T> parser;
        if (unknown == UnknownIdentifiers::readFromTags) {
            parser.enable_unknown_symbol_resolver();
        }
        if (vectorCheck != nullptr) {
            parser.register_vector_access_runtime_check(*vectorCheck);
        }
        if (!parser.compile(source, expression)) {
            std::string errors;
            for (std::size_t i = 0UZ; i < parser.error_count(); ++i) {
                const auto error = parser.get_error(i);
                errors += std::format("ExprTk Parser Error({:2}):  Position: {:2}\nType: [{:14}] Msg: {}; expression:\n{}\n", i, error.token.position, exprtk::parser_error::to_str(error.mode), error.diagnostic, source);
            }
            return std::unexpected(errors);
        }

        std::vector<std::string> all;
        table.get_variable_list(all);
        tagFields.clear();
        for (const std::string& name : all) {
            if (std::ranges::find(known, name) == known.end()) {
                tagFields.emplace_back(name, std::string("gr:") + name);
                table.get_variable(name)->ref() = std::numeric_limits<T>::quiet_NaN();
            }
        }
        return {};
    }

    void assign(const property_map_view& tagMap) {
        for (const auto& [field, prefixedKey] : tagFields) {
            auto found = tagMap.find(std::string_view{prefixedKey});
            if (found == tagMap.end()) {
                found = tagMap.find(std::string_view{field});
            }
            if (found != tagMap.end()) {
                if (const auto converted = pmt::convert_safely<T>((*found).second)) {
                    table.get_variable(field)->ref() = *converted;
                }
            }
        }
    }
};

} // namespace gr::algorithm

#endif // GNURADIO_ALGORITHM_EXPRESSIONSYMBOLS_HPP
