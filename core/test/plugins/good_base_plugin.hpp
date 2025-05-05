#ifndef GR_EXAMPLE_GOOD_PLUGIN_HPP
#define GR_EXAMPLE_GOOD_PLUGIN_HPP

#include <print>

#include <gnuradio-4.0/Plugin.hpp>
#include <pmtv/pmt.hpp>

gr::plugin<>& grPluginInstance();

namespace good {

GR_REGISTER_BLOCK(good::cout_sink, [T], [ float, double ])
template<typename T>
struct cout_sink : public gr::Block<cout_sink<T>> {
    gr::PortIn<T> in;

    gr::Size_t total_count = std::numeric_limits<gr::Size_t>::max();

    GR_MAKE_REFLECTABLE(cout_sink, in, total_count);

    void processOne(T value) {
        total_count--;
        if (total_count == 0) {
            std::println(stderr, "last value was: {}", value);
        }
    }
};

GR_REGISTER_BLOCK(good::fixed_source, [T], [ float, double ])
template<typename T>
struct fixed_source : public gr::Block<fixed_source<T>> {
    gr::PortOut<T> out;

    gr::Size_t event_count = std::numeric_limits<gr::Size_t>::max();
    T          value       = 0;

    GR_MAKE_REFLECTABLE(fixed_source, out, event_count);

    [[nodiscard]] constexpr T processOne() noexcept {
        value++;
        if (event_count != std::numeric_limits<gr::Size_t>::max() && static_cast<gr::Size_t>(value) >= event_count) {
            this->requestStop();
        }
        return value;
    }
};
} // namespace good

#endif
