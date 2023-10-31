#include <list>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/Scheduler.hpp>

template<typename T, typename R = decltype(std::declval<T>() * std::declval<T>())>
struct scale : public gr::Block<scale<T, R>, gr::PortInNamed<T, "original">, gr::PortOutNamed<R, "scaled">> {
    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    processOne(V a) const noexcept {
        return a * 2;
    }
};

template<typename T, typename R = decltype(std::declval<T>() + std::declval<T>())>
struct adder : public gr::Block<adder<T>, gr::PortInNamed<T, "addend0">, gr::PortInNamed<T, "addend1">, gr::PortOutNamed<R, "sum">> {
    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr auto
    processOne(V a, V b) const noexcept {
        return a + b;
    }
};

template<typename T>
class HierBlock : public gr::BlockModel {
private:
    static std::atomic_size_t _unique_id_counter;
    const std::size_t         _unique_id   = _unique_id_counter++;
    const std::string         _unique_name = fmt::format("multi_adder#{}", _unique_id);

protected:
    using setting_map                            = std::map<std::string, int, std::less<>>;
    std::string                       _name      = "multi_adder";
    std::string                       _type_name = "multi_adder";
    gr::property_map                  _meta_information; /// used to store non-graph-processing information like UI block position etc.
    bool                              _input_tags_present  = false;
    bool                              _output_tags_changed = false;
    std::vector<gr::property_map>     _tags_at_input;
    std::vector<gr::property_map>     _tags_at_output;
    std::unique_ptr<gr::SettingsBase> _settings = std::make_unique<gr::BasicSettings<HierBlock<T>>>(*this);

    using in_port_t = gr::PortIn<T>;

    gr::scheduler::Simple<> _scheduler;

    gr::LifeCycleState state;

    gr::Graph
    make_graph() {
        gr::Graph graph;
        auto     &adder_block       = graph.emplaceBlock<adder<double>>({ { "name", "adder" } });
        auto     &left_scale_block  = graph.emplaceBlock<scale<double>>();
        auto     &right_scale_block = graph.emplaceBlock<scale<double>>();

        assert(gr::ConnectionResult::SUCCESS == graph.connect<"scaled">(left_scale_block).to<"addend0">(adder_block));
        assert(gr::ConnectionResult::SUCCESS == graph.connect<"scaled">(right_scale_block).to<"addend1">(adder_block));

        _dynamicInputPorts.emplace_back(gr::DynamicPort(gr::inputPort<0>(&left_scale_block), gr::DynamicPort::non_owned_reference_tag{}));
        _dynamicInputPorts.emplace_back(gr::DynamicPort(gr::inputPort<0>(&right_scale_block), gr::DynamicPort::non_owned_reference_tag{}));
        _dynamicOutputPorts.emplace_back(gr::DynamicPort(gr::outputPort<0>(&adder_block), gr::DynamicPort::non_owned_reference_tag{}));

        _dynamicPortsLoaded = true;
        return graph;
    }

public:
    HierBlock() : _scheduler(make_graph()){};

    ~HierBlock() override = default;

    void
    init(std::shared_ptr<gr::Sequence> /*progress*/, std::shared_ptr<gr::thread_pool::BasicThreadPool> /*ioThreadPool*/) override {}

    void
    start() override {}

    void
    stop() override {}

    void
    pause() override {}

    void
    resume() override {}

    void
    reset() override {}

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

    gr::work::Result
    work(std::size_t requested_work) override {
        if (state == gr::LifeCycleState::STOPPED) {
            return { requested_work, 0UL, gr::work::Status::DONE };
        }
        _scheduler.runAndWait();
        state = gr::LifeCycleState::STOPPED;
        return { requested_work, requested_work, gr::work::Status::DONE };
    }

    void *
    raw() override {
        return this;
    }

    void
    setName(std::string /*name*/) noexcept override {}

    [[nodiscard]] gr::property_map &
    metaInformation() noexcept override {
        return _meta_information;
    }

    [[nodiscard]] const gr::property_map &
    metaInformation() const override {
        return _meta_information;
    }

    [[nodiscard]] gr::SettingsBase &
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
struct fixed_source : public gr::Block<fixed_source<T>> {
    gr::PortOut<T, gr::RequiredSamples<1, 1024>> out;
    std::size_t                                  remaining_events_count;

    T value = 1;

    gr::work::Result
    work(std::size_t requested_work) {
        if (this->state == gr::LifeCycleState::STOPPED) {
            return { requested_work, 0UL, gr::work::Status::DONE };
        }

        if (remaining_events_count != 0) {
            auto &writer = out.streamWriter();
            auto  data   = writer.reserve_output_range(1UZ);
            data[0]      = value;
            data.publish(1UZ);

            remaining_events_count--;
            if (remaining_events_count == 0) {
                fmt::print("Last value sent was {}\n", value);
            }

            value += 1;
            return { requested_work, 1UL, gr::work::Status::OK };
        } else {
            // TODO: Investigate what schedulers do when there is an event written, but we return DONE
            this->state = gr::LifeCycleState::STOPPED;
            this->publishEOSTag(0);
            return { requested_work, 1UL, gr::work::Status::DONE };
        }
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (fixed_source<T>), out, remaining_events_count);

template<typename T>
struct cout_sink : public gr::Block<cout_sink<T>> {
    gr::PortIn<T, gr::RequiredSamples<1, 1024>> in;
    std::size_t                                 remaining = 0;

    void
    processOne(T value) {
        remaining--;
        if (remaining == 0) {
            std::cerr << "last value was: " << value << "\n";
        }
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE_FULL((typename T), (cout_sink<T>), in, remaining);

gr::Graph
make_graph(std::size_t events_count) {
    gr::Graph graph;

    auto &source_leftBlock  = graph.emplaceBlock<fixed_source<double>>({ { "remaining_events_count", events_count } });
    auto &source_rightBlock = graph.emplaceBlock<fixed_source<double>>({ { "remaining_events_count", events_count } });
    auto &sink              = graph.emplaceBlock<cout_sink<double>>({ { "remaining", events_count } });

    auto &hier = graph.addBlock(std::make_unique<HierBlock<double>>());

    graph.connect(source_leftBlock, 0, hier, 0);
    graph.connect(source_rightBlock, 0, hier, 1);
    graph.connect(hier, 0, sink, 0);

    return graph;
}

int
main() {
    auto thread_pool = std::make_shared<gr::thread_pool::BasicThreadPool>("custom pool", gr::thread_pool::CPU_BOUND, 2, 2); // use custom pool to limit number of threads for emscripten

    gr::scheduler::Simple scheduler(make_graph(10), thread_pool);

    // TODO: This line is commented because of failing tests
    // TODO: HierBlock as it is implemented now does not support tag handling and can not be used with new DONE mechanism via EOS tag
    // TODO: Review HierBlock implementation
    // scheduler.runAndWait();
}
