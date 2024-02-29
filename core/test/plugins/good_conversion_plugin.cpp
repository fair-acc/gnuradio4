#include <gnuradio-4.0/plugin.hpp>

#include <charconv>
#include <vector>

GR_PLUGIN("Good Conversion Plugin", "Unknown", "LGPL3", "v1")

namespace good {

template<typename From, typename To>
class convert : public gr::Block<convert<From, To>> {
public:
    gr::PortIn<From> in;
    gr::PortOut<To>  out;

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
auto registerConvert = gr::registerBlock<good::convert, gr::BlockParameters<double, float>, gr::BlockParameters<float, double>>(grPluginInstance());
