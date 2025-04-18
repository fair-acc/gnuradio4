#ifndef NOT_HPP
#define NOT_HPP

#include <concepts>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::basic {

GR_REGISTER_BLOCK(gr::basic::Not, [ uint8_t, int16_t, int32_t ])

template<std::integral T>
struct Not : Block<Not<T>> {
    using Description = Doc<"@brief Performs a bitwise NOT operation on the input stream, producing an output stream with inverted bits.">;

    PortIn<T>  in;
    PortOut<T> out;

    GR_MAKE_REFLECTABLE(Not, in, out);

    [[nodiscard]] constexpr T processOne(T input) const noexcept { return ~input; }
};

} // namespace gr::basic

#endif // NOT_HPP