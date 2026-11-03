#ifndef GNURADIO_LOCALISATION_STEERED_POWER_HISTORY_HPP
#define GNURADIO_LOCALISATION_STEERED_POWER_HISTORY_HPP

#include <cassert>
#include <cmath>
#include <cstddef>
#include <deque>
#include <expected>
#include <span>
#include <string_view>
#include <vector>

namespace gr::localisation {

/** bounded equal-width maps at one frame rate; a rate or shape change starts a new history. */
struct SteeredPowerHistory {
    static constexpr std::size_t kMaxFrames = 256UZ;

    std::deque<std::vector<float>> rows;
    float                          frameRate = 0.f;

    [[nodiscard]] std::expected<void, std::string_view> append(std::span<const float> values, float rate) {
        if (values.empty() || !std::isfinite(rate) || rate <= 0.f) {
            return std::unexpected("waterfall needs a non-empty map and a finite positive frame rate");
        }
        if (rate != frameRate || (!rows.empty() && rows.back().size() != values.size())) {
            rows.clear();
            frameRate = rate;
        }
        rows.emplace_back(values.begin(), values.end());
        if (rows.size() > kMaxFrames) {
            rows.pop_front();
        }
        return {};
    }
    /** row must index the retained history; the newest row has age zero. */
    [[nodiscard]] double ageSeconds(std::size_t row) const noexcept {
        assert(row < rows.size());
        return -static_cast<double>(rows.size() - 1UZ - row) / static_cast<double>(frameRate);
    }
};

} // namespace gr::localisation

#endif // GNURADIO_LOCALISATION_STEERED_POWER_HISTORY_HPP
