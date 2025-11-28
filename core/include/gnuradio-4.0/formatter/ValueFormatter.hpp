#ifndef GNURADIO_VALUEFORMATTER_HPP
#define GNURADIO_VALUEFORMATTER_HPP

#include <gnuradio-4.0/Value.hpp>

#include <format>
#include <ostream>
#include <string>
#include <string_view>

namespace gr::pmt::detail {

static const char* value_type_name(Value::ValueType vt) {
    switch (vt) {
    case Value::ValueType::Monostate: return "monostate";
    case Value::ValueType::Bool: return "bool";
    case Value::ValueType::Int8: return "int8";
    case Value::ValueType::Int16: return "int16";
    case Value::ValueType::Int32: return "int32";
    case Value::ValueType::Int64: return "int64";
    case Value::ValueType::UInt8: return "uint8";
    case Value::ValueType::UInt16: return "uint16";
    case Value::ValueType::UInt32: return "uint32";
    case Value::ValueType::UInt64: return "uint64";
    case Value::ValueType::Float32: return "float32";
    case Value::ValueType::Float64: return "float64";
    case Value::ValueType::ComplexFloat32: return "complex<float32>";
    case Value::ValueType::ComplexFloat64: return "complex<float64>";
    case Value::ValueType::String: return "string";
    case Value::ValueType::Value: return "value";
    default: return "unknown";
    }
}

inline constexpr std::string type_name(const Value& value) {
    if (value.is_tensor()) {
        std::string result;
        result = "Tensor<";
        result += value_type_name(value.value_type());
        result += ">";
        return result;
    } else if (value.is_map()) {
        return "Map<string,Value>";
    }
    return value_type_name(value.value_type()); // scalar
}

inline constexpr std::string value_to_string(const Value& v) {
    constexpr auto append_quoted = [](std::string& out, std::string_view s) {
        out.push_back('"');
        out.append(s);
        out.push_back('"');
    };

    std::string out;
    if (v.is_monostate()) {
        out += "monostate";
    } else if (v.is_map()) {
        out += '{';
        if (auto* map = v.get_if<Value::Map>()) {
            bool first = true;
            for (const auto& [k, val] : *map) {
                if (!first) {
                    out += ", ";
                }
                append_quoted(out, k);
                out += ": ";
                out += value_to_string(val); // recursive
                first = false;
            }
        }
        out += '}';
    } else if (v.is_string()) {
        append_quoted(out, v.value_or(std::string_view{}));
    } else if (v.is_complex()) {
        if (v.value_type() == Value::ValueType::ComplexFloat32) {
            const auto& c = *static_cast<const std::complex<float>*>(v._storage.ptr);
            out += '(';
            out += std::to_string(c.real());
            out += " + ";
            out += std::to_string(c.imag());
            out += "i)";
        } else {
            const auto& c = *static_cast<const std::complex<double>*>(v._storage.ptr);
            out += '(';
            out += std::to_string(c.real());
            out += " + ";
            out += std::to_string(c.imag());
            out += "i)";
        }
    } else if (v.value_type() == Value::ValueType::Bool) {
        out += (v._storage.b ? "true" : "false");
    } else if (v.is_signed_integral()) {
        switch (v.value_type()) {
        case Value::ValueType::Int8: out += std::to_string(static_cast<int>(v._storage.i8)); break;
        case Value::ValueType::Int16: out += std::to_string(v._storage.i16); break;
        case Value::ValueType::Int32: out += std::to_string(v._storage.i32); break;
        case Value::ValueType::Int64: out += std::to_string(v._storage.i64); break;
        default: break;
        }
    } else if (v.is_unsigned_integral()) {
        switch (v.value_type()) {
        case Value::ValueType::UInt8: out += std::to_string(static_cast<unsigned>(v._storage.u8)); break;
        case Value::ValueType::UInt16: out += std::to_string(v._storage.u16); break;
        case Value::ValueType::UInt32: out += std::to_string(v._storage.u32); break;
        case Value::ValueType::UInt64: out += std::to_string(v._storage.u64); break;
        default: break;
        }
    } else if (v.value_type() == Value::ValueType::Float32) {
        out += std::to_string(v._storage.f32);
    } else if (v.value_type() == Value::ValueType::Float64) {
        out += std::to_string(v._storage.f64);
    } else if (v.is_tensor()) {
        out += type_name(v);
        out += "[...]";
    }

    return out;
}

} // namespace gr::pmt::detail

namespace gr::pmt {

// optional public helper if you want it
inline std::string to_string(const Value& v) { return detail::value_to_string(v); }

// ostream support – thin wrapper
inline std::ostream& operator<<(std::ostream& os, const Value& v) { return os << detail::value_to_string(v); }

} // namespace gr::pmt

// std::format integration
namespace std {

template<>
struct formatter<gr::pmt::Value, char> {
    formatter<string_view, char> _impl;

    constexpr auto parse(format_parse_context& ctx) { return _impl.parse(ctx); }

    template<class FormatContext>
    auto format(const gr::pmt::Value& v, FormatContext& ctx) const {
        std::string s = gr::pmt::detail::value_to_string(v);
        return _impl.format(std::string_view{s}, ctx);
    }
};

} // namespace std

#endif // GNURADIO_VALUEFORMATTER_HPP
