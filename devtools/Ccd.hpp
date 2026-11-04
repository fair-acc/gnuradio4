#ifndef GNURADIO_DEVTOOLS_CCD_HPP
#define GNURADIO_DEVTOOLS_CCD_HPP

// cumulative component dependency (CCD) normalised by a balanced binary tree's, after J. Lakos, "Large-Scale C++
// Software Design", Addison-Wesley, 1996, ch. 4: below 1 flatter than a balanced tree, above 1 more entangled

#include <cmath>
#include <cstddef>

namespace devtools {

/// the CCD of a balanced binary tree of `components` components: (n + 1) log2(n + 1) - n
[[nodiscard]] inline double balancedTreeCcd(std::size_t components) noexcept {
    const double count = static_cast<double>(components);
    return (count + 1.0) * std::log2(count + 1.0) - count;
}

[[nodiscard]] inline double normalisedCcd(std::size_t ccd, std::size_t components) noexcept { return static_cast<double>(ccd) / balancedTreeCcd(components); }

} // namespace devtools

#endif // GNURADIO_DEVTOOLS_CCD_HPP
