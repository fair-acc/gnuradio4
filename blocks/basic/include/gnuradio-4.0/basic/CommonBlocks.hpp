#ifndef TEST_COMMON_NODES
#define TEST_COMMON_NODES

#include <algorithm>
#include <cstdlib> // std::size_t
#include <list>
#include <ranges>
#include <span>
#include <string>
#include <string_view>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

GR_REGISTER_BLOCK(builtin_multiply, [T], [ double, float ])

template<typename T>
struct builtin_multiply : gr::Block<builtin_multiply<T>> {
    T factor = static_cast<T>(1.0f);

    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(builtin_multiply, in, out, factor);

    builtin_multiply() = delete;

    builtin_multiply(gr::property_map properties) {
        auto it = properties.find("factor");
        if (it != properties.cend()) {
            auto ptr = gr::CAP{it->second.get_if<T>()};
            if (ptr != nullptr) {
                factor = *ptr;
            }
        }
    }

    [[nodiscard]] constexpr auto processOne(T a) const noexcept { return a * factor; }
};

GR_REGISTER_BLOCK(builtin_counter, [T], [ double, float ])

template<typename T>
struct builtin_counter : gr::Block<builtin_counter<T>> {
    static gr::Size_t s_event_count;

    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(builtin_counter, in, out);

    [[nodiscard]] constexpr auto processOne(T a) const noexcept {
        s_event_count++;
        return a;
    }
};

template<typename T>
gr::Size_t builtin_counter<T>::s_event_count = 0;

#endif // include guard
