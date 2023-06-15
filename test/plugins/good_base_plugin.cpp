#include <plugin.hpp>

#include <charconv>
#include <vector>

#include <pmtv/pmt.hpp>

GP_PLUGIN("Good Math Plugin", "Unknown", "LGPL3", "v1")

namespace good {
namespace fg = fair::graph;

template<typename T>
struct pt;

template<typename T>
auto
total_count(const fair::graph::property_map &params) {
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
class cout_sink : public fg::node<cout_sink<T>, fg::IN<T, 0, 1024, "in">> {
    std::size_t _remaining = 0;

public:
    cout_sink() {}

    explicit cout_sink(const fair::graph::property_map &params) : _remaining(total_count<std::size_t>(params)) {}

    void
    process_one(T value) {
        _remaining--;
        if (_remaining == 0) {
            std::cerr << "last value was: " << value << "\n";
        }
    }
};

template<typename T>
class fixed_source : public fg::node<fixed_source<T>, fg::OUT<T, 0, 1024, "out">> {
public:
    fixed_source() {}

    T value = 1;

    fg::work_return_t
    work() {
        using namespace fair::literals;
        auto &port   = fair::graph::output_port<0>(this);
        auto &writer = port.streamWriter();
        auto  data   = writer.reserve_output_range(1_UZ);
        data[0]      = value;
        data.publish(1_UZ);

        value += 1;
        return fg::work_return_t::OK;
    }
};
} // namespace good

GP_PLUGIN_REGISTER_NODE(good::cout_sink, float, double);
GP_PLUGIN_REGISTER_NODE(good::fixed_source, float, double);
