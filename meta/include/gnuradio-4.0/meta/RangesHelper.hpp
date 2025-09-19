#ifndef GNURADIO_RANGESHELPER_HPP
#define GNURADIO_RANGESHELPER_HPP

#include <functional>
#include <ranges>

namespace gr {
template<class R>
concept ViewableForwardRange = std::ranges::viewable_range<R> && std::ranges::forward_range<std::views::all_t<R>>;

/**
 * @brief View adaptor that removes non-adjacent duplicates *within each group*, keeping the first occurrence per group.
 *
 * Groups the input with `std::views::chunk_by(eq1)` (e.g. by `first`) and, inside each chunk, drops later elements that are equal under `eq2` (e.g. same
 * `{first, second}`), preserving the original order of the first occurrences.
 *
 * This lets you deduplicate interleaved values such as `A, B, A, B` **per group** without reordering the range.
 *
 * @tparam Eq1 Equality relation used to form chunks (groups). Elements that are adjacent and `eq1`-equal belong to the same chunk (e.g. same `first`).
 * @tparam Eq2 Equality relation used to deduplicate inside a chunk (e.g. same `second`, or same `{first, second}`).
 *
 * @note The range is not reordered. For correctness, all elements that are equal under `eq1` should be adjacent in the input (e.g. pre-sorted by the
 *       grouping key). This adaptor is **stateless** and runs in ~O(k²) per chunk (k = elements in the chunk); fine when chunks are small.
 *
 * @example
 * // Example with Tag { index, map }
 * struct Tag { std::size_t index; std::map<std::string,int> map; };
 * auto same_index      = [](const Tag& a, const Tag& b){ return a.index == b.index; };
 * auto same_index_map  = [](const Tag& a, const Tag& b){ return a.index == b.index && a.map == b.map; };
 *
 * std::vector<Tag> tags{
 *   {1,{{"a",1}}}, {1,{{"b",1}}}, {1,{{"a",1}}}, {1,{{"b",1}}},
 *   {2,{{"b",2}}}, {2,{{"c",2}}}, {2,{{"c",2}}}, {2,{{"b",2}}}
 * };
 *
 * // Group by index, dedup by (index,map) -> keep first map per index
 * auto out = tags | PairDeduplicateView{same_index, same_index_map};
 * // Result: (1,{"a"}), (1,{"b"}), (2,{"b"}), (2,{"c"})
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
        return std::forward<Range>(r) | std::views::chunk_by(eq1) | std::views::transform([&eq2 = this->eq2](auto chunk) {
            const auto chunkBegin = std::ranges::begin(chunk);
            const auto n          = static_cast<std::size_t>(std::ranges::distance(chunk));
            const auto iters      = std::views::iota(std::size_t{0}, n) | std::views::transform([chunkBegin](std::size_t i) { return std::ranges::next(chunkBegin, static_cast<std::ptrdiff_t>(i)); });
            auto       filtered   = iters | std::views::filter([chunkBegin, &eq2](auto it) { return std::ranges::find_if(chunkBegin, it, [&](auto const& y) { return std::invoke(eq2, y, *it); }) == it; });
            return filtered | std::views::transform([](auto it) -> decltype(auto) { return *it; });
        }) | std::views::join;
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
