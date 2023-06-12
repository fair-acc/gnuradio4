#include "scheduler.hpp"
#include <graph.hpp>

#include <list>

namespace fg = fair::graph;

template<typename T, typename R = decltype(std::declval<T>() * std::declval<T>())>
struct scale : public fg::node<scale<T, R>, fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "original">, fg::OUT<R, 0, std::numeric_limits<std::size_t>::max(), "scaled">> {
    template<fair::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    process_one(V a) const noexcept {
        return a * 2;
    }
};

template<typename T, typename R = decltype(std::declval<T>() + std::declval<T>())>
struct adder : public fg::node<adder<T>, fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "addend0">, fg::IN<T, 0, std::numeric_limits<std::size_t>::max(), "addend1">,
                               fg::OUT<R, 0, std::numeric_limits<std::size_t>::max(), "sum">> {
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
    using setting_map                             = std::map<std::string, int, std::less<>>;
    std::string                        _name      = "multi_adder";
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
        auto     &adder_block       = graph.make_node<adder<double>>({ { "name", "adder" } });
        auto     &left_scale_block  = graph.make_node<scale<double>>();
        auto     &right_scale_block = graph.make_node<scale<double>>();

        std::ignore                 = graph.connect<"scaled">(left_scale_block).to<"addend0">(adder_block);
        std::ignore                 = graph.connect<"scaled">(right_scale_block).to<"addend1">(adder_block);

        _dynamic_input_ports.emplace_back(fg::input_port<0>(&left_scale_block));
        _dynamic_input_ports.emplace_back(fg::input_port<0>(&right_scale_block));
        _dynamic_output_ports.emplace_back(fg::output_port<0>(&adder_block));

        _dynamic_ports_loaded = true;
        return graph;
    }

public:
    hier_node() : _scheduler(make_graph()){};

    ~hier_node() override = default;

    void
    init(std::shared_ptr<gr::Sequence> /*progress*/, std::shared_ptr<fair::thread_pool::BasicThreadPool> /*ioThreadPool*/) override {}

    [[nodiscard]] std::string_view
    name() const override {
        return _unique_name;
    }

    std::string_view
    type_name() const override {
        return _type_name;
    }

    fg::work_return_t
    work(std::size_t requested_work) override {
        _scheduler.run_and_wait();
        return { requested_work, requested_work, fair::graph::work_return_status_t::DONE };
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
struct fixed_source : public fg::node<fixed_source<T>, fg::OUT<T, 0, 1024, "out">> {
    std::size_t remaining_events_count;

    T           value = 1;

    fg::work_return_t
    work(std::size_t requested_work) {
        if (remaining_events_count != 0) {
            using namespace fair::literals;
            auto &port   = fg::output_port<0>(this);
            auto &writer = port.streamWriter();
            auto  data   = writer.reserve_output_range(1_UZ);
            data[0]      = value;
            data.publish(1_UZ);

            remaining_events_count--;
            if (remaining_events_count == 0) {
                fmt::print("Last value sent was {}\n", value);
            }

            value += 1;
            return { requested_work, 1UL, fg::work_return_status_t::OK };
        } else {
            // TODO: Investigate what schedulers do when there is an event written, but we return DONE
            return { requested_work, 1UL, fg::work_return_status_t::DONE };
        }
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (fixed_source<T>), remaining_events_count);

template<typename T>
struct cout_sink : public fg::node<cout_sink<T>, fg::IN<T, 0, 1024, "in">> {
    std::size_t remaining = 0;

    void
    process_one(T value) {
        remaining--;
        if (remaining == 0) {
            std::cerr << "last value was: " << value << "\n";
        }
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (cout_sink<T>), remaining);

fg::graph
make_graph(std::size_t events_count) {
    fg::graph graph;

    auto     &source_left_node  = graph.make_node<fixed_source<double>>({ { "remaining_events_count", events_count } });
    auto     &source_right_node = graph.make_node<fixed_source<double>>({ { "remaining_events_count", events_count } });
    auto     &sink              = graph.make_node<cout_sink<double>>({ { "remaining", events_count } });

    auto     &hier              = graph.add_node(std::make_unique<hier_node<double>>());

    graph.dynamic_connect(source_left_node, 0, hier, 0);
    graph.dynamic_connect(source_right_node, 0, hier, 1);
    graph.dynamic_connect(hier, 0, sink, 0);

    return graph;
}

int
main() {
    auto thread_pool = std::make_shared<fair::thread_pool::BasicThreadPool>("custom pool", fair::thread_pool::CPU_BOUND, 2, 2); // use custom pool to limit number of threads for emscripten

    fg::scheduler::simple scheduler(make_graph(10), thread_pool);

    scheduler.run_and_wait();
}
