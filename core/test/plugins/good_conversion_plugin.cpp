#include <gnuradio-4.0/plugin.hpp>

#include <charconv>
#include <vector>

GP_PLUGIN("Good Conversion Plugin", "Unknown", "LGPL3", "v1")

namespace good {
namespace grg = gr;

template<typename From, typename To>
class convert : public grg::Block<convert<From, To>> {
public:
    grg::PortIn<From> in;
    grg::PortOut<To>  out;

    [[nodiscard]] constexpr auto
    processOne(From value) const noexcept {
        return static_cast<To>(value);
    }
};

} // namespace good

ENABLE_REFLECTION_FOR_TEMPLATE(good::convert, in, out);

// Another is to use the same macro for both single-parametrised
// and mulciple-parametrised nodes, just to have the parameter
// packs wrapped in some special type like this:
GP_PLUGIN_REGISTER_BLOCK(good::convert, BlockParameters<double, float>, BlockParameters<float, double>);
