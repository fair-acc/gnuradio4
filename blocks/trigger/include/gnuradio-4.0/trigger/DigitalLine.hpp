#ifndef GNURADIO_TRIGGER_DIGITALLINE_HPP
#define GNURADIO_TRIGGER_DIGITALLINE_HPP

#include <gnuradio-4.0/algorithm/SchmittTrigger.hpp>

namespace gr::blocks::trigger {

/** an analogue input read as a logic line: low, high, or unknown until a boundary is reached (unknown is not low) */
template<typename T>
struct DigitalLine {
    using Comparator = gr::trigger::SchmittTrigger<T, gr::trigger::InterpolationMethod::BASIC_LINEAR_INTERPOLATION, 32UZ>;
    using Levels     = typename Comparator::Levels;

    Comparator comparator{};
    bool       changed = false; /// the last sample moved the line from one known level to the other

    void configure(Levels levels) {
        comparator = Comparator{levels};
        changed    = false;
    }

    void observe(T sample) { changed = comparator.processOne(sample) != gr::trigger::EdgeDetection::NONE; }

    [[nodiscard]] gr::trigger::LineState state() const noexcept { return comparator.state(); }
    [[nodiscard]] bool                   high() const noexcept { return state() == gr::trigger::LineState::high; }
};

} // namespace gr::blocks::trigger

#endif // GNURADIO_TRIGGER_DIGITALLINE_HPP
