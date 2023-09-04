#include <plugin.hpp>

#include <charconv>
#include <vector>

GP_PLUGIN("Good Base Plugin", "Unknown", "LGPL3", "v1")

namespace good {
namespace fg = fair::graph;

template<typename From, typename To>
class convert : public fg::node<convert<From, To>> {
public:
    fg::IN<From> in;
    fg::OUT<To>  out;

    [[nodiscard]] constexpr auto
    process_one(From value) const noexcept {
        return static_cast<To>(value);
    }
};

} // namespace good

ENABLE_REFLECTION_FOR_TEMPLATE(good::convert, in, out);

// One option is to invoke the macro for each desired parameter combination
// GP_PLUGIN_REGISTER_NODE_MULTI_PARAMETER(good::convert, float, double);
// GP_PLUGIN_REGISTER_NODE_MULTI_PARAMETER(good::convert, double, float);

// Another is to use the same macro for both single-parametrised
// and mulciple-parametrised nodes, just to have the parameter
// packs wrapped in some special type like this:
GP_PLUGIN_REGISTER_NODE_EXPERIMENTAL(good::convert, fair::graph::node_type_parameters<double, float>, fair::graph::node_type_parameters<float, double>);
