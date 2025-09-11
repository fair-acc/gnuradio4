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

template<class Eq1 = std::ranges::equal_to, class Eq2 = Eq1>
struct PairDeduplicateView : std::ranges::range_adaptor_closure<PairDeduplicateView<Eq1, Eq2>> {
    Eq1 eq1{};
    Eq2 eq2{};

    PairDeduplicateView() = default;

    template<class T1, class T2>
    requires std::constructible_from<Eq1, T1> && std::constructible_from<Eq2, T2>
    explicit constexpr PairDeduplicateView(T1&& e1, T2&& e2) : eq1(std::forward<T1>(e1)), eq2(std::forward<T2>(e2)) {}

    template<ViewableForwardRange Range>
    constexpr auto operator()(Range&& r) const {
        return std::forward<Range>(r) | std::views::chunk_by(eq1) | std::views::transform([&eq2 = this->eq2](auto chunk) {
            const auto chunkBegin = std::ranges::begin(chunk);
            const auto n          = static_cast<std::size_t>(std::ranges::distance(chunk));
            const auto iters      = std::views::iota(std::size_t{0}, n) | std::views::transform([chunkBegin](std::size_t i) { return std::ranges::next(chunkBegin, static_cast<std::ptrdiff_t>(i)); });
            auto       filtered   = iters | std::views::filter([chunkBegin, &eq2](auto it) { return std::ranges::find_if(chunkBegin, it, [&](auto const& y) { return std::invoke(eq2, y, *it); }) == it; });
            return filtered | std::views::transform([](auto it) -> decltype(auto) { return *it; });
        }) | std::views::join;
    }
};
template<class Eq1, class Eq2>
PairDeduplicateView(Eq1, Eq2) -> PairDeduplicateView<std::decay_t<Eq1>, std::decay_t<Eq2>>;
} // namespace gr

#endif // GNURADIO_RANGESHELPER_HPP
