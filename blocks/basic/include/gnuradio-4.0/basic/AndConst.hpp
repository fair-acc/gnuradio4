#ifndef ANDCONST_HPP
#define ANDCONST_HPP

#include <concepts>
#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::basic {

GR_REGISTER_BLOCK(gr::basic::AndConst, [ uint8_t, int16_t, int32_t ])

template<std::integral T>
struct AndConst : Block<AndConst<T>> {
    using Description = Doc<"@brief Performs a bitwise AND operation on two inputs, producing one output stream.">;

    PortIn<T>  in;
    PortOut<T> out;

    T constant = 1;

    GR_MAKE_REFLECTABLE(AndConst, in, out, constant);

    [[nodiscard]] constexpr T processOne(T input) const noexcept { return input & constant; }

    // Validate constant value (restrict to 0 or 1)
    void settingsChanged(const property_map& /* old_settings */, const property_map& new_settings) {
        if (new_settings.contains("constant")) {
            if (constant != 0 && constant != 1) {
                throw std::runtime_error("Constant must be 0 or 1.");
            }
        }
    }
};

} // namespace gr::basic

#endif // ANDCONST_HPP