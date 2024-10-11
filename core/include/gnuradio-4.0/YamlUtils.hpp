#ifndef GNURADIO_YAML_UTILS_H
#define GNURADIO_YAML_UTILS_H

#include <charconv>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wshadow"
#endif
#include <yaml-cpp/yaml.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

namespace YAML {
// YAML custom converter for complex numbers
template<typename T>
requires std::same_as<T, double> || std::same_as<T, float>
struct convert<std::complex<T>> {
    static Node encode(const std::complex<T>& rhs) {
        Node node;
        node.push_back(rhs.real());
        node.push_back(rhs.imag());
        return node;
    }

    static bool decode(const Node& node, std::complex<T>& rhs) {
        if (!node.IsSequence() || node.size() != 2) {
            return false;
        }
        rhs = std::complex<T>(node[0].as<T>(), node[1].as<T>());
        return true;
    }
};
} // namespace YAML

namespace gr {

namespace detail {

template<typename T>
inline auto toYamlString(const T& value) {
    if constexpr (std::is_same_v<std::string, std::remove_cvref_t<T>>) {
        return value;
    } else if constexpr (std::is_same_v<bool, std::remove_cvref_t<T>>) {
        return value ? "true" : "false";
    } else if constexpr (requires { std::to_string(value); }) {
        return std::to_string(value);
    } else {
        return "";
    }
}

struct YamlSeq {
    YAML::Emitter& out;

    YamlSeq(YAML::Emitter& out_) : out(out_) { out << YAML::BeginSeq; }

    ~YamlSeq() { out << YAML::EndSeq; }

    template<typename F>
    requires std::is_invocable_v<F>
    void writeFn(const char* /*key*/, F&& fun) {
        fun();
    }
};

struct YamlMap {
    YAML::Emitter& out;

    YamlMap(YAML::Emitter& out_) : out(out_) { out << YAML::BeginMap; }

    ~YamlMap() { out << YAML::EndMap; }

    template<typename T>
    void write(const std::string_view& key, const std::vector<T>& value) {
        out << YAML::Key << key.data();
        YamlSeq seq(out);
        for (const auto& elem : value) {
            if constexpr (std::same_as<T, std::complex<double>> || std::same_as<T, std::complex<float>>) {
                writeComplexValue(out, elem);
            } else {
                out << YAML::Value << toYamlString(elem);
            }
        }
    }

    template<typename T>
    void write(const std::string_view& key, const T& value) {
        out << YAML::Key << key.data();
        if constexpr (std::same_as<T, std::complex<double>> || std::same_as<T, std::complex<float>>) {
            writeComplexValue(out, value);
        } else {
            out << YAML::Value << toYamlString(value);
        }
    }

    template<typename F>
    void writeFn(const std::string_view& key, F&& fun) {
        out << YAML::Key << key.data();
        out << YAML::Value;
        fun();
    }

private:
    template<typename T>
    requires std::same_as<T, std::complex<double>> || std::same_as<T, std::complex<float>>
    void writeComplexValue(YAML::Emitter& outEmitter, const T& value) {
        YamlSeq seq(outEmitter);
        outEmitter << YAML::Value << toYamlString(value.real());
        outEmitter << YAML::Value << toYamlString(value.imag());
    }
};

inline std::size_t parseIndex(std::string_view str) {
    std::size_t index{};
    auto [_, src_ec] = std::from_chars(str.begin(), str.end(), index);
    if (src_ec != std::errc()) {
        throw fmt::format("Unable to parse the index");
    }
    return index;
}

} // namespace detail
} // namespace gr

#endif // include guard
