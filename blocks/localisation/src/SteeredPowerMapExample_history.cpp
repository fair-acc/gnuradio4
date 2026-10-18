#include "SteeredPowerMapExample.hpp"

#include <cassert>
#include <cmath>

namespace gr::examples {

std::expected<void, std::string_view> SteeredPowerHistory::append(std::span<const float> values, float rate) {
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

double SteeredPowerHistory::ageSeconds(std::size_t row) const noexcept {
    assert(row < rows.size());
    return -static_cast<double>(rows.size() - 1UZ - row) / static_cast<double>(frameRate);
}

} // namespace gr::examples
