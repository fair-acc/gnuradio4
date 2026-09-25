#ifndef GNURADIO_TRIGGER_DIGITALLINE_HPP
#define GNURADIO_TRIGGER_DIGITALLINE_HPP

#include <gnuradio-4.0/algorithm/SchmittTrigger.hpp>

namespace gr::blocks::trigger {

template<typename T>
struct DigitalLine {
    using Comparator = gr::trigger::SchmittTrigger<T, gr::trigger::InterpolationMethod::BASIC_LINEAR_INTERPOLATION, 32UZ>;

    Comparator comparator{};
    bool       high    = false;
    bool       changed = false;
    bool       settled = false;

    void configure(T hysteresis, T threshold) {
        comparator = Comparator{hysteresis == T{} ? T(1) : hysteresis, threshold};
        high       = false;
        changed    = false;
        settled    = false;
    }

    void observe(T sample) {
        using enum gr::trigger::EdgeDetection;
        const auto edge = comparator.processOne(sample);
        changed         = settled && edge != NONE;
        if (edge == RISING) {
            high = true;
        } else if (edge == FALLING) {
            high = false;
        }
        settled = true;
    }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_DIGITALLINE_HPP
