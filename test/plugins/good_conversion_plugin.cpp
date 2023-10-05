#include <plugin.hpp>

#include <charconv>
#include <vector>

GP_PLUGIN("Good Base Plugin", "Unknown", "LGPL3", "v1")

namespace good {
namespace fg = fair::graph;

template<typename From, typename To>
class convert : public fg::block<convert<From, To>> {
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
// and mulciple-parametrised blocks, just to have the parameter
// packs wrapped in some special type like this:
GP_PLUGIN_REGISTER_BLOCK(good::convert, block_parameters<double, float>, block_parameters<float, double>);
