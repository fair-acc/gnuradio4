#ifndef FILEIOHELPERS_HPP
#define FILEIOHELPERS_HPP

#include <expected>
#include <gnuradio-4.0/Message.hpp>

namespace gr::algorithm::fileio {

struct DialogOpenHandle {
    // UI side calls this when it has the file contents in memory.
    std::function<void(std::span<const std::uint8_t>)> completeWithMemory;

    // UI side calls this when the dialog returns a native path.
    std::function<void(std::string)> completeWithFile;

    // UI side calls this on error or cancel.
    std::function<void(std::string_view)> fail;
};

namespace detail {

using DialogOpenCallback = std::function<void(DialogOpenHandle&)>;

inline DialogOpenCallback& dialogOpenCallback() {
    static DialogOpenCallback cb;
    return cb;
}

} // namespace detail

inline void setDialogOpenCallback(detail::DialogOpenCallback cb) { detail::dialogOpenCallback() = std::move(cb); }

namespace detail {
[[nodiscard]] inline bool ciEquals(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) {
        return false;
    }
    return std::ranges::equal(a, b, [](char ac, char bc) { return std::tolower(static_cast<unsigned char>(ac)) == std::tolower(static_cast<unsigned char>(bc)); });
}

[[nodiscard]] inline bool startsWithScheme(std::string_view uri) {
    const auto p = uri.find(":/");
    if (p == std::string_view::npos || p == 0UZ) {
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
    return uri.substr(0, uri.find(":/"));
}

[[nodiscard]] inline bool isHttpUri(std::string_view uri) {
    const auto sc = schemeOf(uri);
    return ciEquals(sc, "http") || ciEquals(sc, "https");
}

[[nodiscard]] inline bool isBrowserDownloadUri(std::string_view uri) { return ciEquals(schemeOf(uri), "download"); }

[[nodiscard]] inline bool isFileUri(std::string_view uri) { return ciEquals(schemeOf(uri), "file"); }

[[nodiscard]] inline bool isDialogUri([[maybe_unused]] std::string_view uri) { return ciEquals(schemeOf(uri), "dialog"); }

[[nodiscard]] inline std::expected<std::string, gr::Error> stripBrowserDownloadUri(std::string_view uri) {
    if (!isBrowserDownloadUri(uri) || !isBrowserDownloadUri(uri)) {
        return std::unexpected(gr::Error{std::format("Not a browser download URI:{}", uri)});
    }
    return std::string(uri.substr(std::string_view("download:/").size()));
}

[[nodiscard]] inline std::expected<std::string, gr::Error> stripFileUri(std::string_view uri) {
    if (!isFileUri(uri)) {
        return std::string(uri);
    }

    const auto       colon = uri.find(':');
    std::string_view rest  = (colon == std::string_view::npos) ? std::string_view{} : uri.substr(colon + 1);
    std::string_view path;
    if (rest.starts_with("//")) {
        rest.remove_prefix(2);
        const auto             slash     = rest.find('/');
        const std::string_view authority = (slash == std::string_view::npos) ? rest : rest.substr(0, slash);
        path                             = (slash == std::string_view::npos) ? std::string_view{} : rest.substr(slash);
        if (!authority.empty() && authority != "localhost") {
            return std::unexpected(gr::Error{std::format("URI (`{}`) with non-local host (`{}`) is not supported.", uri, authority)});
        }
    } else {
        path = rest;
    }

    const auto first = path.find_first_not_of('/');
    path             = (first == std::string_view::npos) ? std::string_view{} : path.substr(first);
    return std::format("/{}", path);
}

} // namespace detail

} // namespace gr::algorithm::fileio
#endif // FILEIOHELPERS_HPP
