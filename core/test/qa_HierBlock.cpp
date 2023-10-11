#include <list>

#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

namespace grg = gr;

template<typename T, typename R = decltype(std::declval<T>() * std::declval<T>())>
struct scale : public grg::Block<scale<T, R>, grg::PortInNamed<T, "original">, grg::PortOutNamed<R, "scaled">> {
    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    processOne(V a) const noexcept {
        return a * 2;
    }
};

template<typename T, typename R = decltype(std::declval<T>() + std::declval<T>())>
struct adder : public grg::Block<adder<T>, grg::PortInNamed<T, "addend0">, grg::PortInNamed<T, "addend1">, grg::PortOutNamed<R, "sum">> {
    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    processOne(V a, V b) const noexcept {
        return a + b;
    }
};

template<typename T>
class HierBlock : public grg::BlockModel {
private:
    static std::atomic_size_t _unique_id_counter;
    const std::size_t         _unique_id   = _unique_id_counter++;
    const std::string         _unique_name = fmt::format("multi_adder#{}", _unique_id);

protected:
    using setting_map                             = std::map<std::string, int, std::less<>>;
    std::string                        _name      = "multi_adder";
    std::string                        _type_name = "multi_adder";
    grg::property_map                  _meta_information; /// used to store non-graph-processing information like UI block position etc.
    bool                               _input_tags_present  = false;
    bool                               _output_tags_changed = false;
    std::vector<grg::property_map>     _tags_at_input;
    std::vector<grg::property_map>     _tags_at_output;
    std::unique_ptr<grg::SettingsBase> _settings = std::make_unique<grg::BasicSettings<HierBlock<T>>>(*this);

    using in_port_t                              = grg::PortIn<T>;

    grg::scheduler::Simple<> _scheduler;

    grg::Graph
    make_graph() {
        grg::Graph graph;
        auto      &adder_block       = graph.emplaceBlock<adder<double>>({ { "name", "adder" } });
        auto      &left_scale_block  = graph.emplaceBlock<scale<double>>();
        auto      &right_scale_block = graph.emplaceBlock<scale<double>>();

        std::ignore                  = graph.connect<"scaled">(left_scale_block).to<"addend0">(adder_block);
        std::ignore                  = graph.connect<"scaled">(right_scale_block).to<"addend1">(adder_block);

        _dynamic_input_ports.emplace_back(grg::inputPort<0>(&left_scale_block), grg::DynamicPort::non_owned_reference_tag{});
        _dynamic_input_ports.emplace_back(grg::inputPort<0>(&right_scale_block), grg::DynamicPort::non_owned_reference_tag{});
        _dynamic_output_ports.emplace_back(grg::outputPort<0>(&adder_block), grg::DynamicPort::non_owned_reference_tag{});

        _DynamicPorts_loaded = true;
        return graph;
    }

public:
    HierBlock() : _scheduler(make_graph()){};

    ~HierBlock() override = default;

    void
    init(std::shared_ptr<gr::Sequence> /*progress*/, std::shared_ptr<gr::thread_pool::BasicThreadPool> /*ioThreadPool*/) override {}

    [[nodiscard]] std::string_view
    name() const override {
        return _unique_name;
    }

    std::string_view
    typeName() const override {
        return _type_name;
    }

    constexpr bool
    isBlocking() const noexcept override {
        return false;
    }

    [[nodiscard]] constexpr std::size_t
    availableInputSamples(std::vector<std::size_t> &) const noexcept override {
        return 0UL;
    }

    [[nodiscard]] constexpr std::size_t
    availableOutputSamples(std::vector<std::size_t> &) const noexcept override {
        return 0UL;
    }

    grg::work::Result
    work(std::size_t requested_work) override {
        _scheduler.runAndWait();
        return { requested_work, requested_work, gr::work::Status::DONE };
    }

    void *
    raw() override {
        return this;
    }

    void
    setName(std::string /*name*/) noexcept override {}

    [[nodiscard]] grg::property_map &
    metaInformation() noexcept override {
        return _meta_information;
    }

    [[nodiscard]] const grg::property_map &
    metaInformation() const override {
        return _meta_information;
    }

    [[nodiscard]] grg::SettingsBase &
    settings() const override {
        return *_settings;
    }

    [[nodiscard]] std::string_view
    uniqueName() const override {
        return _unique_name;
    }
};

template<typename T>
std::atomic_size_t HierBlock<T>::_unique_id_counter = 0;

template<typename T>
struct fixed_source : public grg::Block<fixed_source<T>> {
    grg::PortOut<T, grg::RequiredSamples<1, 1024>> out;
    std::size_t                                    remaining_events_count;

    T                                              value = 1;

    grg::work::Result
    work(std::size_t requested_work) {
        if (remaining_events_count != 0) {
            using namespace gr::literals;
            auto &writer = out.streamWriter();
            auto  data   = writer.reserve_output_range(1_UZ);
            data[0]      = value;
            data.publish(1_UZ);

            remaining_events_count--;
            if (remaining_events_count == 0) {
                fmt::print("Last value sent was {}\n", value);
            }

            value += 1;
            return { requested_work, 1UL, grg::work::Status::OK };
        } else {
            // TODO: Investigate what schedulers do when there is an event written, but we return DONE
            return { requested_work, 1UL, grg::work::Status::DONE };
        }
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (fixed_source<T>), out, remaining_events_count);

template<typename T>
struct cout_sink : public grg::Block<cout_sink<T>> {
    grg::PortIn<T, grg::RequiredSamples<1, 1024>> in;
    std::size_t                                   remaining = 0;

    void
    processOne(T value) {
        remaining--;
        if (remaining == 0) {
            std::cerr << "last value was: " << value << "\n";
        }
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (cout_sink<T>), in, remaining);

grg::Graph
make_graph(std::size_t events_count) {
    grg::Graph graph;

    auto      &source_leftBlock  = graph.emplaceBlock<fixed_source<double>>({ { "remaining_events_count", events_count } });
    auto      &source_rightBlock = graph.emplaceBlock<fixed_source<double>>({ { "remaining_events_count", events_count } });
    auto      &sink              = graph.emplaceBlock<cout_sink<double>>({ { "remaining", events_count } });

    auto      &hier              = graph.addBlock(std::make_unique<HierBlock<double>>());

    graph.dynamic_connect(source_leftBlock, 0, hier, 0);
    graph.dynamic_connect(source_rightBlock, 0, hier, 1);
    graph.dynamic_connect(hier, 0, sink, 0);

    return graph;
}

int
main() {
    auto thread_pool = std::make_shared<gr::thread_pool::BasicThreadPool>("custom pool", gr::thread_pool::CPU_BOUND, 2, 2); // use custom pool to limit number of threads for emscripten

    grg::scheduler::Simple scheduler(make_graph(10), thread_pool);

    scheduler.runAndWait();
}
