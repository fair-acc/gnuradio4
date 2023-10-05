#include <gnuradio-4.0/plugin.hpp>

#include <charconv>
#include <vector>

GP_PLUGIN("Good Base Plugin", "Unknown", "LGPL3", "v1")

namespace good {
namespace fg = fair::graph;

template<typename From, typename To>
class convert : public fg::node<convert<From, To>> {
public:
    fg::PortIn<From> in;
    fg::PortOut<To>  out;

    [[nodiscard]] constexpr auto
    process_one(From value) const noexcept {
        return static_cast<To>(value);
    }
};

} // namespace good

ENABLE_REFLECTION_FOR_TEMPLATE(good::convert, in, out);

// Another is to use the same macro for both single-parametrised
// and mulciple-parametrised nodes, just to have the parameter
// packs wrapped in some special type like this:
GP_PLUGIN_REGISTER_NODE(good::convert, node_parameters<double, float>, node_parameters<float, double>);
