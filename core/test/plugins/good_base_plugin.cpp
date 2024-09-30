#include <charconv>
#include <vector>

#include <pmtv/pmt.hpp>

#include <gnuradio-4.0/plugin.hpp>

GR_PLUGIN("Good Base Plugin", "Unknown", "LGPL3", "v1")

namespace good {

template<typename T>
struct cout_sink : public gr::Block<cout_sink<T>> {
    gr::PortIn<T> in;

    gr::Size_t total_count = std::numeric_limits<gr::Size_t>::max();

    void processOne(T value) {
        total_count--;
        if (total_count == 0) {
            std::cerr << "last value was: " << value << "\n";
        }
    }
};

template<typename T>
struct fixed_source : public gr::Block<fixed_source<T>> {
    gr::PortOut<T> out;

    gr::Size_t event_count = std::numeric_limits<gr::Size_t>::max();
    T          value       = 0;

    [[nodiscard]] constexpr T processOne() noexcept {
        value++;
        if (event_count != std::numeric_limits<gr::Size_t>::max() && static_cast<gr::Size_t>(value) >= event_count) {
            this->requestStop();
        }
        return value;
    }
};
} // namespace good

namespace bts = gr::traits::block;

ENABLE_REFLECTION_FOR_TEMPLATE(good::cout_sink, in, total_count);
auto registerCoutSink = gr::registerBlock<good::cout_sink, float, double>(grPluginInstance());
static_assert(bts::all_input_ports<good::cout_sink<float>>::size == 1);
static_assert(std::is_same_v<bts::all_input_port_types<good::cout_sink<float>>, gr::meta::typelist<float>>);
static_assert(bts::stream_input_ports<good::cout_sink<float>>::size == 1);
static_assert(std::is_same_v<bts::stream_input_port_types<good::cout_sink<float>>, gr::meta::typelist<float>>);

static_assert(bts::all_output_ports<good::cout_sink<float>>::size == 0);
static_assert(std::is_same_v<bts::all_output_port_types<good::cout_sink<float>>, gr::meta::typelist<>>);
static_assert(bts::stream_output_ports<good::cout_sink<float>>::size == 0);
static_assert(std::is_same_v<bts::stream_output_port_types<good::cout_sink<float>>, gr::meta::typelist<>>);

static_assert(bts::all_output_ports<good::cout_sink<float>>::size == 0);
static_assert(std::is_same_v<bts::all_output_port_types<good::cout_sink<float>>, gr::meta::typelist<>>);
static_assert(bts::stream_output_ports<good::cout_sink<float>>::size == 0);
static_assert(std::is_same_v<bts::stream_output_port_types<good::cout_sink<float>>, gr::meta::typelist<>>);

ENABLE_REFLECTION_FOR_TEMPLATE(good::fixed_source, out, event_count);
auto registerFixedSource = gr::registerBlock<good::fixed_source, float, double>(grPluginInstance());
static_assert(bts::all_input_ports<good::fixed_source<float>>::size == 0);
static_assert(std::is_same_v<bts::all_input_port_types<good::fixed_source<float>>, gr::meta::typelist<>>);
static_assert(bts::stream_input_ports<good::fixed_source<float>>::size == 0);
static_assert(std::is_same_v<bts::stream_input_port_types<good::fixed_source<float>>, gr::meta::typelist<>>);
static_assert(bts::all_output_ports<good::fixed_source<float>>::size == 1);
static_assert(std::is_same_v<bts::all_output_port_types<good::fixed_source<float>>, gr::meta::typelist<float>>);
static_assert(bts::stream_output_ports<good::fixed_source<float>>::size == 1);
static_assert(std::is_same_v<bts::stream_output_port_types<good::fixed_source<float>>, gr::meta::typelist<float>>);
