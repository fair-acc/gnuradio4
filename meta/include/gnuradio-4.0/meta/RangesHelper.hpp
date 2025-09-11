#ifndef GNURADIO_RANGESHELPER_HPP
#define GNURADIO_RANGESHELPER_HPP

#include <ranges>

namespace gr {
template<class R>
concept ViewableForwardRange = std::ranges::viewable_range<R> && std::ranges::forward_range<std::views::all_t<R>>;

/**
 * @brief View adaptor that removes adjacent duplicates.

* @example
* std::vector<int> v{1,1,2,2,2,3,2,2};
* auto out = v | AdjacentDeduplicateView{}; // -> [1,2,3,2]
*
* // Custom equality (same index AND same map)
* auto same = [](const Tag& a, const Tag& b){ return a.index==b.index && a.map==b.map; };
* auto dedup = tags | AdjacentDeduplicateView{same};
*/
template<class Eq = std::ranges::equal_to>
struct AdjacentDeduplicateView : std::ranges::range_adaptor_closure<AdjacentDeduplicateView<Eq>> {
    Eq eq{};

    AdjacentDeduplicateView() = default;

    template<class T>
    requires std::constructible_from<Eq, T>
    explicit constexpr AdjacentDeduplicateView(T&& t) : eq(std::forward<T>(t)) {}

    template<ViewableForwardRange Range>
    constexpr auto operator()(Range&& range) const {
        return std::forward<Range>(range) | std::views::chunk_by(eq) | std::views::transform([](auto&& chunk) -> decltype(auto) { return *std::ranges::begin(chunk); });
    }
};

template<class Eq>
AdjacentDeduplicateView(Eq) -> AdjacentDeduplicateView<std::decay_t<Eq>>;
} // namespace gr

#endif // GNURADIO_RANGESHELPER_HPP
