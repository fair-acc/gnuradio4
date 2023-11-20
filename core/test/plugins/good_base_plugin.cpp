#include <charconv>
#include <vector>

#include <pmtv/pmt.hpp>

#include <gnuradio-4.0/plugin.hpp>

GP_PLUGIN("Good Base Plugin", "Unknown", "LGPL3", "v1")

namespace good {

using namespace gr::literals;
namespace grg = gr;

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
class cout_sink : public grg::Block<cout_sink<T>, grg::PortInNamed<T, "in">> {
public:
    std::size_t total_count = -1_UZ;

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
class fixed_source : public grg::Block<fixed_source<T>, grg::PortOutNamed<T, "out">> {
public:
    std::size_t event_count = std::numeric_limits<std::size_t>::max();
    T           value       = 1;

    grg::work::Result
    work(std::size_t requested_work) {
        if (this->state == gr::LifeCycleState::STOPPED) {
            return { requested_work, 0_UZ, gr::work::Status::DONE };
        }
        if (event_count == 0) {
            std::cerr << "fixed_source done\n";
            this->state = gr::LifeCycleState::STOPPED;
            this->publishEOSTag(std::size_t(0));
            return { requested_work, 0_UZ, grg::work::Status::DONE };
        }

        auto &port   = gr::outputPort<0>(this);
        auto &writer = port.streamWriter();
        auto  data   = writer.reserve_output_range(1_UZ);
        data[0]      = value;
        data.publish(1_UZ);

        value += 1;
        if (event_count == std::numeric_limits<std::size_t>::max()) {
            return { requested_work, 1_UZ, grg::work::Status::OK };
        }

        event_count--;
        return { requested_work, 1_UZ, grg::work::Status::OK };
    }
};
} // namespace good

ENABLE_REFLECTION_FOR_TEMPLATE(good::cout_sink, total_count);
GP_PLUGIN_REGISTER_NODE(good::cout_sink, float, double);

ENABLE_REFLECTION_FOR_TEMPLATE(good::fixed_source, event_count);
GP_PLUGIN_REGISTER_NODE(good::fixed_source, float, double);
