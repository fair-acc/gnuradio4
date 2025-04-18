#ifndef OR_HPP
#define OR_HPP

#include <concepts>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::basic {

GR_REGISTER_BLOCK(gr::basic::Or, [ uint8_t, int16_t, int32_t ])

template<std::integral T>
struct Or : Block<Or<T>> {
    using Description = Doc<"@brief Performs a bitwise OR operation on two inputs, producing one output stream.">;

    PortIn<T>  in1;
    PortIn<T>  in2;
    PortOut<T> out;

    GR_MAKE_REFLECTABLE(Or, in1, in2, out);

    [[nodiscard]] constexpr T processOne(T input1, T input2) const noexcept { return input1 | input2; }
};

} // namespace gr::basic

#endif // OR_HPP