#ifndef GNURADIO_TESTING_DELAY_HPP
#define GNURADIO_TESTING_DELAY_HPP

#include <chrono>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

namespace gr::testing {

GR_REGISTER_BLOCK(gr::testing::Delay, [T], [ float, double ])

template<typename T>
struct Delay : Block<Delay<T>> {
    using clock = std::chrono::steady_clock;

    PortIn<T>  in;
    PortOut<T> out;
    uint32_t   delay_ms = 0;

    GR_MAKE_REFLECTABLE(Delay, in, out, delay_ms);

    bool              _waiting = true;
    clock::time_point _start_time;

    gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
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

    void start() noexcept {
        _waiting    = true;
        _start_time = clock::now();
    }
};

} // namespace gr::testing

#endif // GNURADIO_TESTING_DELAY_HPP
