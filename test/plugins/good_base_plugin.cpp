#include <plugin.hpp>

#include <charconv>
#include <vector>

GP_PLUGIN("Good Math Plugin", "Unknown", "LGPL3", "v1")

namespace good {
namespace fg = fair::graph;

template<typename T>
auto
total_count(fair::graph::node_construction_params params) {
    T    total_count = 1;
    auto value       = params.value("total_count"sv);
    std::ignore      = std::from_chars(value.begin(), value.end(), total_count);
    return total_count;
}

template<typename T>
class cout_sink : public fg::node<cout_sink<T>, fg::IN<T, 0, 1024, "in">> {
    std::size_t _remaining = 0;

public:
    cout_sink() {}

    explicit cout_sink(fair::graph::node_construction_params params) : _remaining(total_count<std::size_t>(params)) {}

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
