#ifndef TEST_COMMON_NODES
#define TEST_COMMON_NODES

#include <algorithm>
#include <cstdlib> // std::size_t
#include <list>
#include <ranges>
#include <span>
#include <string>
#include <string_view>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Graph.hpp>
#include <gnuradio-4.0/reflection.hpp>

template<typename T>
class builtin_multiply : public gr::Block<builtin_multiply<T>> {
public:
    T factor = static_cast<T>(1.0f);

    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(builtin_multiply, in, out, factor);

    builtin_multiply() = delete;

    builtin_multiply(gr::property_map properties) {
        auto it = properties.find("factor");
        if (it != properties.cend()) {
            factor = std::get<T>(it->second);
        }
    }

    [[nodiscard]] constexpr auto processOne(T a) const noexcept { return a * factor; }
};

template<typename T>
class builtin_counter : public gr::Block<builtin_counter<T>> {
public:
    static std::size_t s_event_count;

    gr::PortIn<T>  in;
    gr::PortOut<T> out;

    GR_MAKE_REFLECTABLE(builtin_counter, in, out);

    [[nodiscard]] constexpr auto processOne(T a) const noexcept {
        s_event_count++;
        return a;
    }
};

template<typename T>
std::size_t builtin_counter<T>::s_event_count = 0;

template<typename T>
struct MultiAdder : public gr::Block<MultiAdder<T>> {
    std::vector<gr::PortIn<T>> inputs;
    gr::PortOut<T>             out;

    gr::Annotated<gr::Size_t, "n_inputs", gr::Visible, gr::Doc<"variable number of inputs">, gr::Limits<1U, 32U>> n_inputs{0U};

    GR_MAKE_REFLECTABLE(MultiAdder, inputs, out, n_inputs);

    void settingsChanged(const gr::property_map& old_settings, const gr::property_map& new_settings) {
        if (new_settings.contains("n_inputs") && old_settings.at("n_inputs") != new_settings.at("n_inputs")) {
            // if one of the port is already connected and  n_inputs was changed then throw
            if (std::any_of(inputs.begin(), inputs.end(), [](const auto& port) { return port.isConnected(); })) {
                this->emitErrorMessage("settingsChanged(..)", gr::Error("Number of input ports cannot be changed after Graph initialization."));
            }
            fmt::print("{}: configuration changed: n_inputs {} -> {}\n", this->name, old_settings.at("n_inputs"), new_settings.at("n_inputs"));
            inputs.resize(n_inputs);
        }
    }

    template<gr::InputSpanLike TInput>
    gr::work::Status processBulk(std::span<TInput>& inSpans, gr::OutputSpanLike auto& outSpan) {
        std::size_t minSizeIn = std::ranges::min_element(inSpans, [](const auto& lhs, const auto& rhs) { return lhs.size() < rhs.size(); })->size();
        std::size_t available = std::min(outSpan.size(), minSizeIn);

        if (available == 0) {
            return gr::work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        for (std::size_t i = 0; i < available; i++) {
            outSpan[i] = std::accumulate(inSpans.begin(), inSpans.end(), 0, [i](T sum, auto span) { return sum + span[i]; });
        }
        outSpan.publish(available);
        for (auto& inSpan : inSpans) {
            [[maybe_unused]] bool success = inSpan.consume(available);
            assert(success && "Check that consume was successful");
        }

        return gr::work::Status::OK;
    }
};
static_assert(gr::HasProcessBulkFunction<MultiAdder<float>>);

auto registerMultiply = gr::registerBlock<builtin_multiply, double, float>(gr::globalBlockRegistry());
auto registerCounter  = gr::registerBlock<builtin_counter, double, float>(gr::globalBlockRegistry());

#endif // include guard
