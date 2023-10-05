#include <plugin.hpp>

#include <charconv>
#include <vector>

#include <pmtv/pmt.hpp>

GP_PLUGIN("Good Base Plugin", "Unknown", "LGPL3", "v1")

namespace good {

using namespace fair::literals;
namespace fg = fair::graph;

template<typename T>
auto
read_total_count(const fair::graph::property_map &params) {
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
class cout_sink : public fg::block<cout_sink<T>, fg::PortInNamed<T, "in">> {
public:
    std::size_t total_count = -1_UZ;

    cout_sink() {}

    explicit cout_sink(const fair::graph::property_map &params) : total_count(read_total_count<std::size_t>(params)) {}

    void
    process_one(T value) {
        total_count--;
        if (total_count == 0) {
            std::cerr << "last value was: " << value << "\n";
        }
    }
};

template<typename T>
class fixed_source : public fg::block<fixed_source<T>, fg::PortOutNamed<T, "out">> {
public:
    std::size_t event_count = -1_UZ; // infinite count by default

    fixed_source() {}

    T value = 1;

    fg::work_return_t
    work(std::size_t requested_work) {
        if (event_count == 0) {
            std::cerr << "fixed_source done\n";
            return { requested_work, 0_UZ, fg::work_return_status_t::DONE };
        }

        auto &port   = fair::graph::output_port<0>(this);
        auto &writer = port.streamWriter();
        auto  data   = writer.reserve_output_range(1_UZ);
        data[0]      = value;
        data.publish(1_UZ);

        value += 1;
        if (event_count == -1_UZ) {
            return { requested_work, 1_UZ, fg::work_return_status_t::OK };
        }

        event_count--;
        return { requested_work, 1_UZ, fg::work_return_status_t::OK };
    }
};
} // namespace good

ENABLE_REFLECTION_FOR_TEMPLATE(good::cout_sink, total_count);
GP_PLUGIN_REGISTER_BLOCK(good::cout_sink, float, double);

ENABLE_REFLECTION_FOR_TEMPLATE(good::fixed_source, event_count);
GP_PLUGIN_REGISTER_BLOCK(good::fixed_source, float, double);
