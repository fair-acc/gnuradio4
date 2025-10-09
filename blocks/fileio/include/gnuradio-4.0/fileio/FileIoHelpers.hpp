#ifndef FILEIOHELPERS_HPP
#define FILEIOHELPERS_HPP

namespace gr::blocks::fileio {

namespace detail {
[[nodiscard]] inline bool startsWithScheme(std::string_view uri) {
    const auto p = uri.find("://");
    if (p == std::string_view::npos) {
        return false;
    }
    for (char c : uri.substr(0, p)) {
        if (!std::isalpha(static_cast<unsigned char>(c))) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] inline std::string_view schemeOf(std::string_view uri) {
    if (!startsWithScheme(uri)) {
        return {};
    }
    return uri.substr(0, uri.find("://"));
}

[[nodiscard]] inline bool isHttpUri(std::string_view uri) {
    const auto sc = schemeOf(uri);
    return sc == "http" || sc == "https";
}

[[nodiscard]] inline bool isFileUri(std::string_view uri) { return schemeOf(uri) == "file"; }

[[nodiscard]] inline std::string stripFileUri(std::string_view uri) {
    constexpr std::string_view k = "file:";
    if (uri.rfind(k, 0) != 0) {
        return std::string(uri);
    }
    std::string_view  rest = uri.substr(k.size());
    const std::size_t i    = rest.find_first_not_of('/');
    rest                   = (i == std::string_view::npos) ? std::string_view{} : rest.substr(i);

    std::string out;
    out.reserve(rest.size() + 1);
    out.push_back('/');
    out.append(rest);
    return out;
}
} // namespace detail

} // namespace gr::blocks::fileio
#endif // FILEIOHELPERS_HPP
