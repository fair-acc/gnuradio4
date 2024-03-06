#include <charconv>
#include <vector>

#include <pmtv/pmt.hpp>

#include <gnuradio-4.0/plugin.hpp>

GR_PLUGIN("Good Base Plugin", "Unknown", "LGPL3", "v1")

namespace good {

template<typename T>
auto
read_total_count(const gr::property_map &params) {
    T total_count = 1;
    if (auto it = params.find("total_count"s); it != params.end()) {
        auto &variant = it->second;
        auto *ptr     = std::get_if<T>(&variant);
        if (ptr) {
            total_count = *ptr;
        }
    }
    return total_count;
}

template<typename T>
class cout_sink : public gr::Block<cout_sink<T>, gr::PortInNamed<T, "in">> {
public:
    std::size_t total_count = -1UZ;

    cout_sink() {}

    explicit cout_sink(const gr::property_map &params) : total_count(read_total_count<std::size_t>(params)) {}

    void
    processOne(T value) {
        total_count--;
        if (total_count == 0) {
            std::cerr << "last value was: " << value << "\n";
        }
    }
};

template<typename T>
class fixed_source : public gr::Block<fixed_source<T>, gr::PortOutNamed<T, "out">> {
public:
    std::size_t event_count = std::numeric_limits<std::size_t>::max();
    T           value       = 1;

    gr::work::Result
    work(std::size_t requested_work) {
        if (this->state() == gr::lifecycle::State::STOPPED) {
            return { requested_work, 0UZ, gr::work::Status::DONE };
        }
        if (event_count == 0) {
            std::cerr << "fixed_source done\n";
            if (auto ret = this->changeStateTo(gr::lifecycle::State::REQUESTED_STOP); !ret) {
                using namespace gr::message;
                this->emitErrorMessage("work()", ret.error());
            }
            this->publishTag({ { gr::tag::END_OF_STREAM, true } }, 0);
            return { requested_work, 0UZ, gr::work::Status::DONE };
        }

        auto &port   = gr::outputPort<0, gr::PortType::STREAM>(this);
        auto &writer = port.streamWriter();
        auto  data   = writer.reserve(1UZ);
        data[0]      = value;
        data.publish(1UZ);

        value += 1;
        if (event_count == std::numeric_limits<std::size_t>::max()) {
            return { requested_work, 1UZ, gr::work::Status::OK };
        }

        event_count--;
        return { requested_work, 1UZ, gr::work::Status::OK };
    }
};
} // namespace good

namespace bts = gr::traits::block;

ENABLE_REFLECTION_FOR_TEMPLATE(good::cout_sink, total_count);
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

ENABLE_REFLECTION_FOR_TEMPLATE(good::fixed_source, event_count);
auto registerFixedSource = gr::registerBlock<good::fixed_source, float, double>(grPluginInstance());
static_assert(bts::all_input_ports<good::fixed_source<float>>::size == 0);
static_assert(std::is_same_v<bts::all_input_port_types<good::fixed_source<float>>, gr::meta::typelist<>>);
static_assert(bts::stream_input_ports<good::fixed_source<float>>::size == 0);
static_assert(std::is_same_v<bts::stream_input_port_types<good::fixed_source<float>>, gr::meta::typelist<>>);
static_assert(bts::all_output_ports<good::fixed_source<float>>::size == 1);
static_assert(std::is_same_v<bts::all_output_port_types<good::fixed_source<float>>, gr::meta::typelist<float>>);
static_assert(bts::stream_output_ports<good::fixed_source<float>>::size == 1);
static_assert(std::is_same_v<bts::stream_output_port_types<good::fixed_source<float>>, gr::meta::typelist<float>>);
