#include <gnuradio-4.0/meta/utils.hpp>

#include <fmt/format.h>

std::string gr::meta::detail::makePortableTypeName(std::string_view name) {
    auto trimmed = [](std::string_view view) {
        while (view.front() == ' ') {
            view.remove_prefix(1);
        }
        while (view.back() == ' ') {
            view.remove_suffix(1);
        }
        return view;
    };

    using namespace std::string_literals;
    using gr::meta::detail::local_type_name;
    static const auto typeMapping = std::array<std::pair<std::string, std::string>, 13>{{
        {local_type_name<std::int8_t>(), "int8"s}, {local_type_name<std::int16_t>(), "int16"s}, {local_type_name<std::int32_t>(), "int32"s}, {local_type_name<std::int64_t>(), "int64"s},         //
        {local_type_name<std::uint8_t>(), "uint8"s}, {local_type_name<std::uint16_t>(), "uint16"s}, {local_type_name<std::uint32_t>(), "uint32"s}, {local_type_name<std::uint64_t>(), "uint64"s}, //
        {local_type_name<float>(), "float32"s}, {local_type_name<double>(), "float64"},                                                                                                           //                                                                                                                                                                                                                                                        //
        {local_type_name<std::string>(), "string"s},                                                                                                                                              //
        {local_type_name<std::complex<float>>(), "complex<float32>"s}, {local_type_name<std::complex<double>>(), "complex<float64>"s}                                                             //
    }};

    const auto it = std::ranges::find_if(typeMapping, [&](const auto& pair) { return pair.first == name; });
    if (it != typeMapping.end()) {
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
            params.push_back(makePortableTypeName(param));
            view.remove_prefix(cursor + 1);
            cursor = 0;
            continue;
        }
        cursor++;
    }
    params.push_back(makePortableTypeName(trimmed(view)));
    return fmt::format("{}<{}>", base, fmt::join(params, ", "));
}
