#ifndef TEST_COMMON_NODES
#define TEST_COMMON_NODES

#include <cstdlib> // std::size_t
#include <list>
#include <string>
#include <string_view>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/reflection.hpp>

template<typename T>
class builtin_multiply : public gr::Block<builtin_multiply<T>> {
    T _factor = static_cast<T>(1.0f);

public:
    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    builtin_multiply() = delete;

    builtin_multiply(gr::property_map properties) {
        auto it = properties.find("factor");
        if (it != properties.cend()) {
            _factor = std::get<T>(it->second);
        }
    }

    [[nodiscard]] constexpr auto
    processOne(T a) const noexcept {
        return a * _factor;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(builtin_multiply, in, out);

template<typename T>
class builtin_counter : public gr::Block<builtin_counter<T>> {
public:
    static std::size_t s_event_count;

    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    [[nodiscard]] constexpr auto
    processOne(T a) const noexcept {
        s_event_count++;
        return a;
    }
};

template<typename T>
std::size_t builtin_counter<T>::s_event_count = 0;
ENABLE_REFLECTION_FOR_TEMPLATE(builtin_counter, in, out);

// TODO: Unify blocks with static and dynamic ports
//  - Port to gr::block
//  - use Block::set_name instead of returning an empty name
// TODO: Inherit from Block class when create new block.
template<typename T>
class multi_adder : public gr::BlockModel {
    static std::atomic_size_t _unique_id_counter;

public:
    int               input_port_count;
    const std::size_t unique_id    = _unique_id_counter++;
    const std::string unique_name_ = fmt::format("multi_adder#{}", unique_id); // TODO: resolve symbol duplication

protected:
    using TPortIn = gr::PortIn<T>;
    std::vector<TPortIn> _input_ports;
    gr::PortOut<T>     _output_port;

protected:
    using setting_map                            = std::map<std::string, int, std::less<>>;
    std::string                       _name      = "multi_adder";
    std::string                       _type_name = "multi_adder";
    gr::property_map                  _meta_information; /// used to store non-graph-processing information like UI block position etc.
    bool                              _input_tags_present  = false;
    bool                              _output_tags_changed = false;
    std::vector<gr::property_map>     _tags_at_input;
    std::vector<gr::property_map>     _tags_at_output;
    std::unique_ptr<gr::SettingsBase> _settings = std::make_unique<gr::BasicSettings<multi_adder<T>>>(*this);

    void
    applyInputCount() {
        if (_input_ports.size() == static_cast<std::size_t>(input_port_count)) return;

        _input_ports.resize(static_cast<std::size_t>(input_port_count));

        _dynamicInputPorts.clear();
        for (auto &input_port : _input_ports) {
            _dynamicInputPorts.emplace_back(gr::DynamicPort(input_port, gr::DynamicPort::non_owned_reference_tag{}));
        }
        if (_dynamicOutputPorts.empty()) {
            _dynamicOutputPorts.emplace_back(gr::DynamicPort(_output_port, gr::DynamicPort::non_owned_reference_tag{}));
        }
        _dynamicPortsLoaded = true;
    }

public:
    explicit multi_adder(int input_ports_size) : input_port_count(input_ports_size) { applyInputCount(); };

    ~multi_adder() override = default;

    void
    settingsChanged(const gr::property_map & /*old_setting*/, const gr::property_map & /*new_setting*/) noexcept {
        applyInputCount();
    }

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
        return unique_name_;
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
        return 0UZ;
    }

    [[nodiscard]] constexpr std::size_t
    availableOutputSamples(std::vector<std::size_t> &) const noexcept override {
        return 0UZ;
    }

    // TODO: integrate with Block::work
    gr::work::Result
    work(std::size_t requested_work) override {
        // TODO: Rewrite with ranges once we can use them
        std::size_t available_samples = std::numeric_limits<std::size_t>::max();
        for (const auto &input_port : _input_ports) {
            auto available_samples_for_port = input_port.streamReader().available();
            if (available_samples_for_port < available_samples) {
                available_samples = available_samples_for_port;
            }
        }

        if (available_samples == 0) {
            return { requested_work, 0UZ, gr::work::Status::OK };
        }

        std::vector<std::span<const double>> readers;
        for (auto &input_port : _input_ports) {
            readers.push_back(input_port.streamReader().get(available_samples));
        }

        auto &writer = _output_port.streamWriter();
        writer.publish(
                [available_samples, &readers](std::span<T> output) {
                    // const auto input = reader.get(n_to_publish);
                    for (std::size_t i = 0; i < available_samples; ++i) {
                        output[i] = std::accumulate(readers.cbegin(), readers.cend(), 0, [i](T sum, auto span) { return sum + span[i]; });
                    }
                },
                available_samples);

        for (auto &input_port [[maybe_unused]] : _input_ports) {
            auto consumed = input_port.streamReader().consume(available_samples);
            assert(available_samples == consumed);
            std::ignore = consumed;
        }
        return { requested_work, available_samples, gr::work::Status::OK };
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
        return unique_name_;
    }
};

// static_assert(gr::BlockLike<multi_adder<int>>);

ENABLE_REFLECTION_FOR_TEMPLATE(multi_adder, input_port_count);

template<typename Registry>
void
                       registerBuiltinBlocks(Registry *registry) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
    GP_REGISTER_NODE_RUNTIME(registry, builtin_multiply, double, float);
    GP_REGISTER_NODE_RUNTIME(registry, builtin_counter, double, float);
#pragma GCC diagnostic pop
}

#endif // include guard
