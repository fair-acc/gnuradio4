#ifndef GNURADIO_EMBEDDED_DEMO_BLOCKS_HPP
#define GNURADIO_EMBEDDED_DEMO_BLOCKS_HPP

#include <tuple>
#include <utility>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Port.hpp>

namespace gr::testing::embedded {

template<typename T>
struct CountSource : public gr::Block<CountSource<T>> {
    gr::PortOut<T> random;

    GR_MAKE_REFLECTABLE(CountSource, random);

    [[nodiscard]] constexpr T processOne() const noexcept { return 42; }
};

template<typename T>
struct ExpectSink : public gr::Block<ExpectSink<T>> {
    gr::PortIn<T> sink;

    T           lastValue{};
    std::size_t count{0};

    GR_MAKE_REFLECTABLE(ExpectSink, sink);

    constexpr void processOne(T value) noexcept {
        lastValue = value;
        ++count;
    }
};

template<typename T, T Scale, typename R = decltype(std::declval<T>() * std::declval<T>())>
struct scale : public gr::Block<scale<T, Scale, R>> {
    gr::PortIn<T>  original;
    gr::PortOut<R> scaled;

    GR_MAKE_REFLECTABLE(scale, original, scaled);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto processOne(V a) const noexcept {
        return a * Scale;
    }
};

template<typename T, typename R = decltype(std::declval<T>() + std::declval<T>())>
struct adder : public gr::Block<adder<T>> {
    gr::PortIn<T>  addend0;
    gr::PortIn<T>  addend1;
    gr::PortOut<R> sum;

    GR_MAKE_REFLECTABLE(adder, addend0, addend1, sum);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto processOne(V a, V b) const noexcept {
        return a + b;
    }
};

template<typename T>
struct duplicate : public gr::Block<duplicate<T>> {
    gr::PortIn<T>                              in;
    std::tuple<gr::PortOut<T>, gr::PortOut<T>> out;

    GR_MAKE_REFLECTABLE(duplicate, in, out);

    [[nodiscard]] constexpr auto processOne(T a) const noexcept {
        return [&a]<std::size_t... Is>(std::index_sequence<Is...>) { return std::tuple{((void)Is, a)...}; }(std::make_index_sequence<std::tuple_size_v<decltype(out)>>());
    }
};

} // namespace gr::testing::embedded

#endif // GNURADIO_EMBEDDED_DEMO_BLOCKS_HPP
