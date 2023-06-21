#include "scheduler.hpp"
#include <graph.hpp>

#include <list>

namespace fg = fair::graph;

template<typename T, typename R = decltype(std::declval<T>() * std::declval<T>())>
class scale : public fg::node<scale<T, R>, fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "original">, fg::OUT<R, 0, std::numeric_limits<std::size_t>::max(), "scaled">> {
public:
    explicit scale(std::string_view name = "") { this->_name = name; }

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(V a) const noexcept {
        return a * 2;
    }
};

template<typename T, typename R = decltype(std::declval<T>() + std::declval<T>())>
class adder : public fg::node<adder<T>, fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "addend0">, fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "addend1">,
                              fg::OUT<R, 0, std::numeric_limits<std::size_t>::max(), "sum">> {
public:
    explicit adder(std::string_view name = "") { this->_name = name; }

    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(V a, V b) const noexcept {
        return a + b;
    }
};

template<typename T>
class hier_node : public fg::node_model {
private:
    static std::atomic_size_t _unique_id_counter;
    const std::size_t         _unique_id   = _unique_id_counter++;
    const std::string         _unique_name = fmt::format("multi_adder#{}", _unique_id);

protected:
    using setting_map                        = std::map<std::string, int, std::less<>>;
    std::string                        _name = "multi_adder";
    std::string                        _type_name = "multi_adder";
    fg::property_map                   _meta_information; /// used to store non-graph-processing information like UI block position etc.
    bool                               _input_tags_present  = false;
    bool                               _output_tags_changed = false;
    std::vector<fg::property_map>      _tags_at_input;
    std::vector<fg::property_map>      _tags_at_output;
    std::unique_ptr<fg::settings_base> _settings = std::make_unique<fg::basic_settings<hier_node<T>>>(*this);

    using in_port_t                              = fg::IN<T>;

    fg::scheduler::simple<> _scheduler;

    fg::graph
    make_graph() {
        fg::graph graph;
        auto     &adder_block       = graph.make_node<adder<double>>("adder");
        auto     &left_scale_block  = graph.make_node<scale<double>>();
        auto     &right_scale_block = graph.make_node<scale<double>>();

        std::ignore = graph.connect<"scaled">(left_scale_block).to<"addend0">(adder_block);
        std::ignore = graph.connect<"scaled">(right_scale_block).to<"addend1">(adder_block);

        _dynamic_input_ports.emplace_back(fg::input_port<0>(&left_scale_block));
        _dynamic_input_ports.emplace_back(fg::input_port<0>(&right_scale_block));
        _dynamic_output_ports.emplace_back(fg::output_port<0>(&adder_block));

        _dynamic_ports_loaded = true;
        return graph;
    }

public:
    hier_node() : _scheduler(make_graph()){};

    ~hier_node() override = default;

    [[nodiscard]] std::string_view
    name() const override {
        return _unique_name;
    }

    std::string_view
    type_name() const override {
        return _type_name;
    }

    fg::work_return_t
    work() override {
        return _scheduler.work();
    }

    void *
    raw() override {
        return this;
    }

    void
    set_name(std::string /*name*/) noexcept override {}

    [[nodiscard]] fg::property_map &
    meta_information() noexcept override {
        return _meta_information;
    }

    [[nodiscard]] const fg::property_map &
    meta_information() const noexcept override {
        return _meta_information;
    }

    [[nodiscard]] fg::settings_base &
    settings() const override {
        return *_settings;
    }

    [[nodiscard]] std::string_view
    unique_name() const override {
        return _unique_name;
    }
};

template<typename T>
std::atomic_size_t hier_node<T>::_unique_id_counter = 0;

template<typename T>
class fixed_source : public fg::node<fixed_source<T>, fg::OUT<T, 0, 1024, "out">> {
private:
    std::size_t _remaining_events_count;

public:
    explicit fixed_source(std::size_t events_count) : _remaining_events_count(events_count) {}

    T value = 1;

    fg::work_return_t
    work() {
        if (_remaining_events_count != 0) {
            using namespace fair::literals;
            auto &port   = fg::output_port<0>(this);
            auto &writer = port.streamWriter();
            auto  data   = writer.reserve_output_range(1_UZ);
            data[0]      = value;
            data.publish(1_UZ);

            _remaining_events_count--;
            if (_remaining_events_count == 0) {
                fmt::print("Last value sent was {}\n", value);
            }

            value += 1;
            return fg::work_return_t::OK;
        } else {
            // TODO: Investigate what schedulers do when there is an event written,
            // but we return DONE
            return fg::work_return_t::DONE;
        }
    }
};

template<typename T>
class cout_sink : public fg::node<cout_sink<T>, fg::IN<T, 0, 1024, "in">> {
    std::size_t _remaining = 0;

public:
    cout_sink() = default;

    explicit cout_sink(std::size_t count) : _remaining(count) {}

    void
    process_one(T value) {
        _remaining--;
        if (_remaining == 0) {
            std::cerr << "last value was: " << value << "\n";
        }
    }
};

fg::graph
make_graph(std::size_t events_count) {
    fg::graph graph;

    auto     &source_left_node  = graph.make_node<fixed_source<double>>(events_count);
    auto     &source_right_node = graph.make_node<fixed_source<double>>(events_count);
    auto     &sink              = graph.make_node<cout_sink<double>>(events_count);

    auto     &hier              = graph.add_node(std::make_unique<hier_node<double>>());

    graph.dynamic_connect(source_left_node, 0, hier, 0);
    graph.dynamic_connect(source_right_node, 0, hier, 1);
    graph.dynamic_connect(hier, 0, sink, 0);

    return graph;
}

int
main() {
    auto thread_pool = std::make_shared<fair::thread_pool::BasicThreadPool>("custom pool", fair::thread_pool::CPU_BOUND, 2,2);

    fg::scheduler::simple scheduler(make_graph(10), thread_pool);

    scheduler.work();
}
