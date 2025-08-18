#ifndef GNURADIO_GRAPH_YAML_IMPORTER_H
#define GNURADIO_GRAPH_YAML_IMPORTER_H

#include <expected>

#include <gnuradio-4.0/YamlPmt.hpp>

#include "BlockModel.hpp"
#include "Graph.hpp"
#include "PluginLoader.hpp"

namespace gr {

namespace detail {

template<typename T>
inline std::expected<T, gr::Error> getProperty(const gr::property_map& map, std::string_view propertyName) {
    auto it = map.find(propertyName);
    if (it == map.cend()) {
        return std::unexpected(gr::Error(std::format("Missing field {} in YAML object", propertyName)));
    }

    auto* value = std::get_if<T>(&it->second);
    if (value == nullptr) {
        return std::unexpected(gr::Error(std::format("Field {} in YAML object has an incorrect type index={} instead of {}", propertyName, it->second.index(), gr::meta::type_name<T>())));
    }

    return {*value};
}

template<typename T>
inline std::expected<T, gr::Error> getProperty(const gr::property_map& map, std::string_view propertyName, const auto&... propertySubNames)
requires(sizeof...(propertySubNames) > 0)
{
    static_assert((std::is_convertible_v<decltype(propertySubNames), std::string_view> && ...));
    auto it = map.find(propertyName);
    if (it == map.cend()) {
        return std::unexpected(gr::Error(std::format("Missing field {} in YAML object", propertyName)));
    }

    auto* value = std::get_if<gr::property_map>(&it->second);
    if (value == nullptr) {
        return std::unexpected(gr::Error(std::format("Field {} in YAML object has an incorrect type index={} instead of gr::property_map", propertyName, it->second.index())));
    }

    return getProperty<T>(*value, propertySubNames...);
}

template<typename T>
T getOrThrow(std::expected<T, gr::Error>&& expectedValue, std::source_location location = std::source_location::current()) {
    if (!expectedValue) {
        throw gr::exception(std::format("Got an error {}, caller {}:{}", expectedValue.error().message, location.file_name(), location.line()));
    } else {
        return *expectedValue;
    }
}

} // namespace detail

void loadGraphFromMap(PluginLoader& loader, gr::Graph& resultGraph, gr::property_map yaml, std::source_location location = std::source_location::current());

gr::property_map saveGraphToMap(PluginLoader& loader, const gr::Graph& rootGraph);

gr::Graph loadGrc(PluginLoader& loader, std::string_view yamlSrc, std::source_location location = std::source_location::current());

std::string saveGrc(PluginLoader& loader, const gr::Graph& rootGraph);

} // namespace gr

#endif // include guard
