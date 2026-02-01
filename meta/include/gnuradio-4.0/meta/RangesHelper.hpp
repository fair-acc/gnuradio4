#ifndef GNURADIO_RANGESHELPER_HPP
#define GNURADIO_RANGESHELPER_HPP

#include <functional>
#include <ranges>
#include <vector>

namespace gr {
template<class R>
concept ViewableForwardRange = std::ranges::viewable_range<R> && std::ranges::forward_range<std::views::all_t<R>>;

/**
 * @brief Eagerly deduplicates elements within chunks, keeping the first occurrence per group.
 *
 * Groups the input by `eq1` (consecutive equal elements) and, inside each chunk, drops later
 * elements that are equal under `eq2`, preserving the original order of the first occurrences.
 *
 * This is an **eager** implementation that returns a vector, avoiding the compile-time overhead
 * of chained range adaptors (chunk_by | transform | filter | transform | join).
 *
 * @tparam Eq1 Equality relation used to form chunks (groups).
 * @tparam Eq2 Equality relation used to deduplicate inside a chunk.
 *
 * @note Runs in ~O(n*k) where n is total elements and k is max chunk size.
 *
 * @example
 * auto same_index     = [](const auto& a, const auto& b){ return a.first == b.first; };
 * auto same_index_map = [](const auto& a, const auto& b){ return a.first == b.first && a.second == b.second; };
 * auto out = tags | PairDeduplicateView{same_index, same_index_map};
 */
template<class TEq1 = std::ranges::equal_to, class TEq2 = TEq1>
struct PairDeduplicateView : std::ranges::range_adaptor_closure<PairDeduplicateView<TEq1, TEq2>> {
    TEq1 eq1{};
    TEq2 eq2{};

    PairDeduplicateView() = default;

    template<class T1, class T2>
    requires std::constructible_from<TEq1, T1> && std::constructible_from<TEq2, T2>
    explicit constexpr PairDeduplicateView(T1&& e1, T2&& e2) : eq1(std::forward<T1>(e1)), eq2(std::forward<T2>(e2)) {}

    template<ViewableForwardRange Range>
    constexpr auto operator()(Range&& r) const {
        using ValueType = std::ranges::range_value_t<Range>;
        std::vector<ValueType> result;

        auto       it  = std::ranges::begin(r);
        const auto end = std::ranges::end(r);

        while (it != end) {
            // Start of a new chunk - find chunk boundaries using eq1
            auto chunkStart = it;

            // Collect unique elements within this chunk
            while (it != end) {
                const auto& current = *it;

                // Check if still in same chunk (eq1 with chunk start)
                if (it != chunkStart && !std::invoke(eq1, *chunkStart, current)) {
                    break; // New chunk starts
                }

                // Check if this element is a duplicate within current chunk's results
                bool isDuplicate = false;
                // Only check against elements from the same chunk (those after chunkStart position in result)
                for (auto checkIt = result.end(); checkIt != result.begin();) {
                    --checkIt;
                    // Stop if we've gone past elements from current chunk
                    if (!std::invoke(eq1, *chunkStart, *checkIt)) {
                        break;
                    }
                    if (std::invoke(eq2, *checkIt, current)) {
                        isDuplicate = true;
                        break;
                    }
                }

                if (!isDuplicate) {
                    result.push_back(current);
                }
                ++it;
            }
        }

        return result;
    }
};

template<class TEq1, class TEq2>
PairDeduplicateView(TEq1, TEq2) -> PairDeduplicateView<std::decay_t<TEq1>, std::decay_t<TEq2>>;

/**
 * @brief Lazy k-way merge over a range of sorted ranges.
 *
 * Two main components:
 *  - `MergeView<R, Comp>` - the iterable view that performs the merge lazily.
 *  - `Merge<Comp>` - the pipeable adaptor that builds a `MergeView` from an outer range.
 *
 * Given an outer range `R` whose elements are themselves ranges (the “inner ranges”),
 * the merge yields a single, sorted sequence by repeatedly selecting the smallest current
 * element across the inner ranges using `comp`.
 *
 * It is required that each inner range is already sorted using `comp`.

 * Usage
 *  // 1) Pipeable adaptor:
 *  auto merged = inputs | Merge{compByKey};
 *  for (auto&& x : merged) { ... }
 *  // 2) Direct view construction:
 *   // *  MergeView view{std::views::all(inputs), compByKey};
 *  for (auto&& x : view) { ... }
 *
 * Example
 *  std::vector<int> a{1,3,5};
 *  std::vector<int> b{2,4,6};
 *  std::array ranges{ std::views::all(a), std::views::all(b) };
 *  auto out = ranges | Merge{std::ranges::less{}};
 *  // yields: 1,2,3,4,5,6
 */
template<class R, class TComp>
struct MergeView : std::ranges::view_interface<MergeView<R, TComp>> {
    using TOut      = R;
    using TInView   = std::remove_cvref_t<std::ranges::range_reference_t<TOut>>;
    using TIterator = std::ranges::iterator_t<TInView>;

    static_assert(std::ranges::forward_range<TInView>);

    TOut                        _out;
    [[no_unique_address]] TComp _comp{};

    MergeView() = default;

    template<class TRFw, class TCompFw>
    requires std::constructible_from<TOut, TRFw> && std::constructible_from<TComp, TCompFw>
    constexpr MergeView(TRFw&& r, TCompFw&& c) : _out(std::forward<TRFw>(r)), _comp(std::forward<TCompFw>(c)) {}

    struct Iterator {
        using iterator_concept  = std::forward_iterator_tag;
        using iterator_category = std::forward_iterator_tag;
        using difference_type   = std::ptrdiff_t;
        using reference         = std::iter_reference_t<TIterator>;
        using value_type        = std::iter_value_t<TIterator>;

        MergeView*             _view{};
        std::vector<TIterator> _its;
        std::size_t            _chosen{static_cast<std::size_t>(-1)};

        Iterator()                           = default;
        Iterator(const Iterator&)            = default;
        Iterator& operator=(const Iterator&) = default;

        explicit Iterator(MergeView* parent) : _view(parent) {
            if constexpr (std::ranges::sized_range<TOut>) {
                auto n = std::ranges::size(_view->_out);
                _its.reserve(n);
            }
            for (auto&& v : _view->_out) {
                _its.push_back(std::ranges::begin(v));
            }
            next();
        }

        void next() {
            const std::size_t N = _its.size();
            _chosen             = N;
            for (auto&& [idx, it, rng] : std::views::zip(std::views::iota(0UZ), _its, _view->_out)) {
                if (it == std::ranges::end(rng)) {
                    continue;
                }
                if (_chosen == N || std::invoke(_view->_comp, *it, *_its[_chosen])) {
                    _chosen = idx;
                }
            }
        }

        reference operator*() const {
            // assert(_chosen < _its.size());
            return *_its[_chosen];
        }

        Iterator& operator++() {
            ++_its[_chosen];
            next();
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        constexpr bool        operator==(std::default_sentinel_t) const noexcept { return _chosen == _its.size(); }
        friend constexpr bool operator==(std::default_sentinel_t s, const Iterator& it) noexcept { return it == s; }

        friend constexpr bool operator==(const Iterator& a, const Iterator& b) noexcept {
            if (a._view != b._view) {
                return false;
            }
            const bool a_end = (a._chosen == a._its.size());
            const bool b_end = (b._chosen == b._its.size());
            if (a_end && b_end) {
                return true;
            }
            if (a_end != b_end) {
                return false;
            }

            return a._its == b._its;
        }
        friend constexpr bool operator!=(const Iterator& a, const Iterator& b) noexcept { return !(a == b); }
    };

    auto begin() { return Iterator{this}; }
    auto end() { return std::default_sentinel; }
};

template<class Comp = std::ranges::less>
struct Merge : std::ranges::range_adaptor_closure<Merge<Comp>> {
    [[no_unique_address]] Comp _comp{};

    Merge() = default;

    template<class TCompFw>
    requires std::constructible_from<Comp, TCompFw>
    explicit Merge(TCompFw&& c) : _comp(std::forward<TCompFw>(c)) {}

    template<class R>
    requires std::ranges::input_range<R> && std::ranges::input_range<std::remove_cvref_t<std::ranges::range_reference_t<std::views::all_t<R>>>>
    auto operator()(R&& r) const {
        return MergeView<std::views::all_t<R>, Comp>(std::views::all(std::forward<R>(r)), _comp);
    }
};

template<class Comp = std::ranges::less>
Merge(Comp) -> Merge<Comp>;

} // namespace gr

#endif // GNURADIO_RANGESHELPER_HPP
