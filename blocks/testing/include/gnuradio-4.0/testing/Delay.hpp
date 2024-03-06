#ifndef GNURADIO_TESTING_DELAY_HPP
#define GNURADIO_TESTING_DELAY_HPP

#include <chrono>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::testing {

template<typename T>
struct Delay : public gr::Block<Delay<T>> {
    PortIn<T>  in;
    PortOut<T> out;
    uint32_t   delay_ms = 0;

private:
    using clock                = std::chrono::steady_clock;
    bool              _waiting = true;
    clock::time_point _start_time;

public:
    gr::work::Status
    processBulk(ConsumableSpan auto &input, PublishableSpan auto &output) {
        if (_waiting) {
            if (clock::now() - _start_time < std::chrono::milliseconds(delay_ms)) {
                std::ignore = input.consume(0);
                output.publish(0);
                return work::Status::OK;
            }
            _waiting = false;
        }

        const auto n = std::min(input.size(), output.size());
        std::ranges::copy(input.first(n), output.begin());
        std::ignore = input.consume(n);
        output.publish(n);
        return work::Status::OK;
    }

    void
    start() noexcept {
        _waiting    = true;
        _start_time = clock::now();
    }
};

} // namespace gr::testing

ENABLE_REFLECTION_FOR_TEMPLATE(gr::testing::Delay, in, out, delay_ms)

auto registerTestingDelay = gr::registerBlock<gr::testing::Delay, float, double>(gr::globalBlockRegistry());

#endif // GNURADIO_TESTING_DELAY_HPP
