#ifndef GNURADIO_TENSOR_HPP
#define GNURADIO_TENSOR_HPP

#include <gnuradio-4.0/PmtCollections.hpp>

#include <algorithm>
#include <cassert>
#include <complex>
#include <concepts>
#include <map>
#include <memory>
#include <memory_resource>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

#if __has_include(<mdspan>)
#include <mdspan>
#define TENSOR_HAVE_MDSPAN 1
#endif

namespace gr {

struct tensor_extents_tag {};
struct tensor_data_tag {};

inline constexpr tensor_extents_tag extents_from{};
inline constexpr tensor_data_tag    data_from{};

namespace detail {
template<std::size_t... Ex>
inline constexpr bool all_static_v = (sizeof...(Ex) > 0) && ((Ex != std::dynamic_extent) && ...);

#ifndef GR_TENSOR_MAX_RANK
inline constexpr std::size_t kMaxRank = 8UZ;
#else
inline constexpr std::size_t kMaxRank = GR_TENSOR_MAX_RANK;
#endif
} // namespace detail

/**
 * @class Tensor[View]
 * @brief A multi-dimensional array container with a flexible compile-time and runtime multi-dimensional extent data storage support
 *
 * @tparam ElementType The type of elements stored (bool is stored as uint8_t internally)
 * @tparam Ex Variable number of extents (use std::dynamic_extent for runtime dimensions)

 * ##BasicUsage Basic Usage Examples:
 * ### fully run-time dynamic Tensor<T>
 * @code
 * Tensor<double> matrix({3, 4}, data);
 * matrix[2, 3] = 42.0;
 * matrix.reshape({2, 6});  // same data, new shape
 * @endcode
 *
 * ### static rank, dynamic extents Tensor<T>
 * @code
 * Tensor<float, std::dynamic_extent, std::dynamic_extent> img({640, 480});
 * img.resize({1920, 1080});  // can change dimensions
 * @endcode
 *
 * ### fully static Tensor<T>
 * @code
 * // all compile-time, zero overhead
 * constexpr Tensor<int, 2, 3> mat{{1, 2, 3}, {4, 5, 6}};
 * constexpr auto elem = mat[1, 2];  // compile-time access
 * static_assert(mat.size() == 6);
 * @endcode
 *
 * ## std::vector compatibility
 * @code
 * std::vector<int> vec{1, 2, 3, 4, 5};
 * Tensor<int> tensor(vec);           // construct from vector
 * tensor = vec;                       // assign from vector
 * auto v2 = static_cast<std::vector<int>>(tensor);  // convert back
 * @endcode
 *
 * ## tagged constructors for disambiguation
 * @code
 * std::vector<size_t> values{10, 20, 30};
 * Tensor<size_t> t1(extents_from, values);  // shape: 10×20×30
 * Tensor<size_t> t2(data_from, values);     // data: {10,20,30}
 * @endcode
 *
 * ## custom polymorphic memory resources support (PMR=
 * @code
 * std::pmr::monotonic_buffer_resource arena(buffer, size);
 * Tensor<double> tensor({1000, 1000}, &arena);  // use arena allocator
 * @endcode
 *
 * ## type and layout conversion
 * @code
 * Tensor<int, 3, 3> static_tensor{...};
 * Tensor<double> dynamic_tensor(static_tensor);  // int→double, static→dynamic
 * @endcode
 *
 * ## mdspan/mdarray compatibility
 * @code
 * auto view = tensor.to_mdspan();  // get mdspan view
 * view(i, j) = value;               // mdspan-style access
 * tensor.stride(0);                 // get stride for dimension
 * @endcode
 *
 * @note row-major (C/C++-style) ordering is used exclusively
 * @note bool tensors store data as uint8_t to avoid vector<bool> issues
 * @note static tensors have compile-time size guarantees and zero overhead
 * @note inspired by @see https://wg21.link/P1684 (mdarray proposal)
 */
template<typename ElementType, std::size_t... Ex>
struct Tensor;

template<typename ElementType, std::size_t... Ex>
struct TensorView;

template<typename T, bool managed, std::size_t... Ex>
struct TensorBase;

template<typename T, std::size_t... Ex>
inline static constexpr bool is_tensor = false;

template<typename Tensor>
struct tensor_traits {
    static_assert(gr::meta::always_false<Tensor>, "TensorTraits not specialized for this type");
};

template<typename E, bool managed, std::size_t... Ex>
struct tensor_traits<TensorBase<E, managed, Ex...>> {
    using value_type                                     = E;
    static constexpr bool        is_view                 = !managed;
    static constexpr bool        static_rank             = (sizeof...(Ex) > 0UZ);
    static constexpr std::size_t rank                    = sizeof...(Ex);
    static constexpr bool        all_static              = (sizeof...(Ex) > 0UZ) && ((Ex != std::dynamic_extent) && ...);
    static constexpr bool        all_dynamic             = (sizeof...(Ex) == 0UZ);
    static constexpr bool        static_rank_dyn_extents = (sizeof...(Ex) > 0) && ((Ex == std::dynamic_extent) && ...);
};

template<typename T, std::size_t... Ex>
struct tensor_traits<Tensor<T, Ex...>> : tensor_traits<TensorBase<T, true, Ex...>> {};
template<typename T, std::size_t... Ex>
struct tensor_traits<const Tensor<T, Ex...>> : tensor_traits<TensorBase<T, true, Ex...>> {};

template<typename T, std::size_t... Ex>
struct tensor_traits<TensorView<T, Ex...>> : tensor_traits<TensorBase<T, false, Ex...>> {};
template<typename T, std::size_t... Ex>
struct tensor_traits<const TensorView<T, Ex...>> : tensor_traits<TensorBase<T, false, Ex...>> {};

template<typename Tensor>
concept TensorLike = is_tensor<std::remove_cvref_t<Tensor>>;
template<typename Tensor, typename T>
concept TensorOf = TensorLike<Tensor> && std::same_as<typename tensor_traits<std::remove_cvref_t<Tensor>>::value_type, std::remove_const_t<T>>;
template<typename Tensor, typename T>
concept StaticTensorOf = TensorOf<Tensor, T> && tensor_traits<std::remove_cvref_t<Tensor>>::all_static;
template<typename Tensor, typename T>
concept StaticRankTensorOf = TensorOf<Tensor, T> && tensor_traits<std::remove_cvref_t<Tensor>>::static_rank;
template<typename Tensor, typename T>
concept DynamicTensorOf = TensorOf<Tensor, T> && tensor_traits<std::remove_cvref_t<Tensor>>::all_dynamic;
template<typename Tensor, typename T>
concept TensorViewOf = TensorOf<Tensor, T> && tensor_traits<std::remove_cvref_t<Tensor>>::is_view;

template<typename T, bool managed, std::size_t... Ex>
struct TensorBase {
    static_assert(!std::is_same_v<T, std::string>);
    using value_type   = T;
    using element_type = std::remove_const_t<T>;

    struct static_extents_store {
        constexpr static std::size_t                   rank = sizeof...(Ex);
        constexpr static std::array<std::size_t, rank> extents{Ex...};
        constexpr static std::array<std::size_t, rank> strides = [] {
            std::array<std::size_t, rank> strides_{};
            if (rank == 0UZ) {
                return strides_;
            }
            std::size_t stride = 1UZ;
            for (std::size_t i = rank; i-- > 0UZ;) {
                strides_[i] = stride;
                stride *= extents[i];
            }
            return strides_;
        }();
    };

    struct semi_static_extents_store {
        constexpr static std::size_t  rank = sizeof...(Ex);
        std::array<std::size_t, rank> extents{};
        std::array<std::size_t, rank> strides{};
    };

    struct dynamic_extents_store {
        std::size_t                               rank = sizeof...(Ex);
        std::array<std::size_t, detail::kMaxRank> extents{};
        std::array<std::size_t, detail::kMaxRank> strides{};
    };

    using extents_store_t = std::conditional_t<detail::all_static_v<Ex...>, static_extents_store, std::conditional_t<(sizeof...(Ex) > 0UZ), semi_static_extents_store, dynamic_extents_store>>;
    [[no_unique_address]] extents_store_t _metaInfo{}; // this is suppressed for statically defined extents

    using container_type = std::conditional_t<detail::all_static_v<Ex...> && managed, std::array<element_type, (Ex * ... * 1UZ)>, gr::pmr::vector<element_type, managed>>;
    container_type _data{};

    [[nodiscard]] std::pmr::memory_resource* resource() const noexcept
    requires(!detail::all_static_v<Ex...>)
    {
        return _data.resource();
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept {
        if constexpr (detail::all_static_v<Ex...>) {
            return (Ex * ... * 1UZ);
        } else {
            return _data.size();
        }
    }

    [[nodiscard]] static consteval std::size_t static_rank() noexcept
    requires(sizeof...(Ex) > 0)
    {
        return sizeof...(Ex);
    }
    [[nodiscard]] constexpr std::size_t rank() const noexcept {
        if constexpr (sizeof...(Ex) == 0) {
            return _metaInfo.rank;
        } else {
            return sizeof...(Ex);
        }
    }
    [[nodiscard]] constexpr std::span<const std::size_t> extents() const noexcept { return {_metaInfo.extents.data(), rank()}; }
    [[nodiscard]] constexpr std::size_t                  extent(std::size_t d) const noexcept { return extents()[d]; }
    [[nodiscard]] constexpr std::span<const std::size_t> strides() const noexcept { return {_metaInfo.strides.data(), rank()}; }
    [[nodiscard]] constexpr std::size_t                  stride(std::size_t r) const noexcept { return strides()[r]; }
    constexpr void                                       recomputeStrides() noexcept
    requires(!detail::all_static_v<Ex...>)
    {
        if (rank() == 0) {
            return;
        }
        std::span<std::size_t> strides{_metaInfo.strides.data(), rank()};
        strides.back() = 1UZ;
        for (std::size_t i = strides.size(); i-- > 1UZ;) {
            strides[i - 1UZ] = strides[i] * extent(i);
        }
    }

    static constexpr std::size_t product(std::span<const std::size_t> ex) {
        std::size_t n{1UZ};
        for (auto e : ex) {
            if (e != 0UZ && n > (std::numeric_limits<std::size_t>::max)() / e) {
                throw std::length_error("Tensor: extents product overflow");
            }
            n *= e;
        }
        return n;
    }
    static constexpr std::size_t checked_size(std::span<const std::size_t> ex) { return product(ex); }

    constexpr void bounds_check(std::span<const std::size_t> indices) const {
        if (indices.size() == 1) {
            if (indices[0] >= size()) {
                throw std::out_of_range("Tensor::at: linear index out of bounds");
            }
            return;
        }

        // multi-dimensional access requires exact rank match
        if (indices.size() != rank()) {
            throw std::out_of_range("Tensor::at: incorrect number of indices");
        }
        for (std::size_t d = 0; d < rank(); ++d) {
            if (indices[d] >= extents()[d]) {
                throw std::out_of_range("Tensor::at: index out of bounds");
            }
        }
    }

    // row-major fold from a span
    [[nodiscard]] constexpr std::size_t index_of(std::span<const std::size_t> idx) const noexcept {
        if (idx.size() == 1UZ) {
            return idx[0];
        }
        std::size_t lin = 0UZ;
        // use strides for indexing to support custom layouts (e.g., column-major)
        const auto s = strides();
        for (std::size_t d = 0UZ; d < idx.size(); ++d) {
            lin += idx[d] * s[d];
        }
        return lin;
    }

    template<std::integral... Indices>
    [[nodiscard]] constexpr std::size_t index_of(Indices... indices) const noexcept {
        if constexpr (detail::all_static_v<Ex...>) {
            if constexpr (sizeof...(Indices) == 1) {
                std::size_t indices_array[]{static_cast<std::size_t>(indices)...};
                return indices_array[0UZ];
            } else {
                static_assert(sizeof...(Indices) == sizeof...(Ex), "TensorBase::index_of: incorrect number of indices");
                std::size_t idx = 0UZ;
                std::size_t dim = 0UZ;
                ((idx += static_cast<std::size_t>(indices) * stride(dim++)), ...);
                return idx;
            }
        } else {
            std::array<std::size_t, sizeof...(Indices)> a{static_cast<std::size_t>(indices)...};
            return index_of(std::span<const std::size_t>(a));
        }
    }

    // 1D vector compatibility functions
    [[nodiscard]] std::size_t capacity() const noexcept { return _data.capacity(); }
    void                      reserve(std::size_t new_cap) { _data.reserve(new_cap); }
    void                      shrink_to_fit() { _data.shrink_to_fit(); }
    void                      clear() noexcept
    requires(!detail::all_static_v<Ex...>)
    {
        if constexpr (sizeof...(Ex) == 0) {
            _metaInfo.rank = 0UZ;
        }
        std::ranges::fill(_metaInfo.extents, 0UZ);
        _data.clear();
    }

    // Resize specific dimension
    void resize_dim(std::size_t dim, std::size_t new_extent)
    requires(!detail::all_static_v<Ex...>)
    {
        if (dim >= rank()) {
            throw std::out_of_range("resize_dim: dimension out of range");
        }

        if (_metaInfo.extents[dim] == new_extent) {
            return;
        }

        _metaInfo.extents[dim]     = new_extent;
        std::size_t new_total_size = product(extents());
        recomputeStrides();

        _data.resize(new_total_size); // May need more sophisticated copying for interior dimensions
        // this is a simplified version - the full implementation would need element-wise padding/truncating
    }

    void resize(std::initializer_list<std::size_t> new_extents, const T& value = {}) { resize(std::span(new_extents), value); }

    template<std::ranges::range Range>
    requires std::same_as<std::ranges::range_value_t<Range>, std::size_t>
    void resize(const Range& newExtents, const T& value = {}) {
        if (std::ranges::empty(newExtents)) { // clear tensor
            clear();
            return;
        }

        _metaInfo.rank       = std::ranges::size(newExtents);
        std::size_t new_size = product(std::span(newExtents.begin(), newExtents.end()));
        std::ranges::copy_n(std::ranges::begin(newExtents), static_cast<std::ptrdiff_t>(_metaInfo.rank), _metaInfo.extents.begin());
        recomputeStrides();
        _data.assign(new_size, value);
    }

    [[nodiscard]] T& front() {
        if (empty()) {
            throw std::runtime_error("front() on empty tensor");
        }
        return _data.front();
    }

    [[nodiscard]] T const& front() const {
        if (empty()) {
            throw std::runtime_error("front() on empty tensor");
        }
        return _data.front();
    }

    [[nodiscard]] T& back() {
        if (empty()) {
            throw std::runtime_error("back() on empty tensor");
        }
        return _data.back();
    }

    [[nodiscard]] T const& back() const {
        if (empty()) {
            throw std::runtime_error("back() on empty tensor");
        }
        return _data.back();
    }

    void swap(TensorBase& other) noexcept {
        if constexpr (detail::all_static_v<Ex...>) {
            _data.swap(other._data);
        } else {
            std::swap(_metaInfo, other._metaInfo);
            _data.swap(other._data);
        }
    }

    [[nodiscard]] constexpr bool empty() const noexcept { return _data.empty(); }
    [[nodiscard]] constexpr bool is_contiguous() const noexcept {
        const auto s = strides();
        const auto e = extents();

        if (rank() == 0UZ) {
            return true;
        }
        if (s[rank() - 1UZ] != 1UZ) { // last stride must be 1
            return false;
        }

        for (std::size_t i = rank() - 1; i > 0; --i) {
            if (s[i - 1] != s[i] * e[i]) {
                return false; // non-standard stride
            }
        }
        return true;
    }

    // clang-format off
    template<bool IsConst>
    class strided_iterator {
        using TPtr = std::conditional_t<IsConst, const T*, T*>;

    public:
        using iterator_category = std::random_access_iterator_tag;
        using value_type        = std::remove_const_t<T>;
        using difference_type   = std::ptrdiff_t;
        using size_type         = std::size_t;
        using pointer           = TPtr;
        using reference         = std::conditional_t<IsConst, const T&, T&>;

    private:
        TPtr               data_{nullptr};
        const std::size_t* extents_{nullptr};
        const std::size_t* strides_{nullptr};
        std::size_t        rank_{0UZ};
        size_type          linear_idx_{0UZ};

    public:
        constexpr strided_iterator() = default;
        constexpr strided_iterator(TPtr data, const std::size_t* ext, const std::size_t* str, std::size_t rank, size_type idx) noexcept :
                        data_(data), extents_(ext), strides_(str), rank_(rank), linear_idx_(idx) {}

        reference operator*() const {
            std::size_t offset    = 0;
            std::size_t remaining = linear_idx_;

            for (std::size_t d = 0; d < rank_; ++d) {
                std::size_t coord = remaining;
                for (std::size_t k = d + 1; k < rank_; ++k) {
                    coord /= extents_[k];
                }
                coord %= extents_[d];
                offset += coord * strides_[d];
            }

            return data_[offset];
        }

        pointer operator->() const { return &**this; }

        strided_iterator& operator++() { ++linear_idx_; return *this; }
        strided_iterator  operator++(int) { auto tmp = *this; ++*this; return tmp; }
        strided_iterator& operator--() { --linear_idx_; return *this; }
        strided_iterator  operator--(int) { auto tmp = *this; --*this; return tmp; }
        strided_iterator& operator+=(difference_type n) { linear_idx_ += static_cast<size_t>(n); return *this; }
        strided_iterator& operator-=(difference_type n) { linear_idx_ -= static_cast<size_t>(n); return *this; }
        strided_iterator  operator+(difference_type n) const { auto tmp = *this; tmp += n; return tmp; }
        strided_iterator  operator-(difference_type n) const { auto tmp = *this; tmp -= n; return tmp; }
        friend strided_iterator operator+(difference_type n, const strided_iterator& it) { return it + n; }

        reference operator[](difference_type n) const { return *(*this + n); }

        friend bool operator==(const strided_iterator& a, const strided_iterator& b) { return a.linear_idx_ == b.linear_idx_; }
        friend auto operator<=>(const strided_iterator& a, const strided_iterator& b) { return a.linear_idx_ <=> b.linear_idx_; }
        friend difference_type operator-(const strided_iterator& a, const strided_iterator& b) { return static_cast<difference_type>(a.linear_idx_) - static_cast<difference_type>(b.linear_idx_); }
    };
    // clang-format on

    using iterator       = std::conditional_t<detail::all_static_v<Ex...> || managed, T*, strided_iterator<false>>;
    using const_iterator = std::conditional_t<detail::all_static_v<Ex...> || managed, const T*, strided_iterator<true>>;

    // --- iterators / STL compat ---
    [[nodiscard]] T*       data() noexcept { return std::to_address(_data.begin()); }
    [[nodiscard]] const T* data() const noexcept { return std::to_address(_data.cbegin()); }

    [[nodiscard]] iterator begin() noexcept {
        if constexpr (detail::all_static_v<Ex...> || managed) {
            return data();
        } else {
            return iterator(data(), extents().data(), strides().data(), rank(), 0);
        }
    }

    [[nodiscard]] iterator end() noexcept {
        if constexpr (detail::all_static_v<Ex...> || managed) {
            return data() + size();
        } else {
            return iterator(data(), extents().data(), strides().data(), rank(), size());
        }
    }

    [[nodiscard]] const_iterator begin() const noexcept {
        if constexpr (detail::all_static_v<Ex...> || managed) {
            return data();
        } else {
            return const_iterator(data(), extents().data(), strides().data(), rank(), 0);
        }
    }

    [[nodiscard]] const_iterator end() const noexcept {
        if constexpr (detail::all_static_v<Ex...> || managed) {
            return data() + size();
        } else {
            return const_iterator(data(), extents().data(), strides().data(), rank(), size());
        }
    }

    [[nodiscard]] const_iterator cbegin() const noexcept { return begin(); }
    [[nodiscard]] const_iterator cend() const noexcept { return end(); }

    [[nodiscard]] constexpr std::span<T>       data_span() noexcept { return std::span(_data); }
    [[nodiscard]] constexpr std::span<const T> data_span() const noexcept { return std::span(_data); }

    // for vector-like compatibility
    [[nodiscard]] T& operator[](std::size_t idx) noexcept
    requires(!std::is_const_v<T>)
    {
        assert(idx < size());
        return _data[idx];
    }
    constexpr const T& operator[](std::size_t idx) const noexcept {
        assert(idx < size());
        return _data[idx];
    }
    template<std::integral... Indices>
    constexpr T& operator[](Indices... idx) noexcept {
        return _data[index_of(idx...)];
    }
    template<std::integral... Indices>
    constexpr const T& operator[](Indices... idx) const noexcept {
        return _data[index_of(idx...)];
    }

    // checked access (throws std::out_of_range)
    [[nodiscard]] T& at(std::span<const std::size_t> indices) {
        bounds_check(indices);
        if (indices.size() == 1UZ) {
            return _data[indices[0]];
        }
        return _data[index_of(indices)];
    }
    [[nodiscard]] const T& at(std::span<const std::size_t> indices) const {
        bounds_check(indices);
        if (indices.size() == 1UZ) {
            return _data[indices[0]];
        }
        return _data[index_of(indices)];
    }

    template<std::integral... Indices>
    [[nodiscard]] T& at(Indices... indices) {
        std::array<std::size_t, sizeof...(Indices)> a{static_cast<std::size_t>(indices)...};
        return at(std::span<const std::size_t>(a));
    }

    template<std::integral... Indices>
    [[nodiscard]] const T& at(Indices... indices) const {
        std::array<std::size_t, sizeof...(Indices)> a{static_cast<std::size_t>(indices)...};
        return at(std::span<const std::size_t>(a));
    }

    constexpr void fill(const T& value) noexcept {
        if constexpr (detail::all_static_v<Ex...>) {
            for (auto& elem : _data) {
                elem = value;
            }
        } else {
            std::ranges::fill(_data, value); // constexpr from C++26 onwards
        }
    }

#if defined(TENSOR_HAVE_MDSPAN)
    template<std::size_t Rank>
    auto to_mdspan() noexcept {
        using index_t = std::size_t;
        auto e        = extents(); // copy to ensure a contiguous buffer for constructor
        // Use layout_right (row-major)
        return std::mdspan<T, std::dextents<index_t, Rank>>(data(), e);
    }
    template<std::size_t Rank>
    auto to_mdspan() const noexcept {
        using index_t = std::size_t;
        auto e        = extents();
        return std::mdspan<const T, std::dextents<index_t, Rank>>(data(), e);
    }
#else
    template<bool is_const>
    struct View {
        using TPtr = std::conditional_t<is_const, const T*, T*>;

    private:
        TPtr                          ptr_;
        std::span<const std::size_t>  extents_;
        std::pmr::vector<std::size_t> strides_;

    public:
        View(TPtr ptr, std::span<const std::size_t> ex, std::pmr::vector<std::size_t> st) : ptr_(ptr), extents_(ex), strides_(std::move(st)) {}

        TPtr data() const noexcept { return ptr_; }
        auto extents() const noexcept { return extents_; }
        auto strides() const noexcept { return std::span<const std::size_t>(strides_); }
    };

    [[nodiscard]] auto to_mdspan() noexcept { return View<false>{data(), extents(), strides()}; }
    [[nodiscard]] auto to_mdspan() const noexcept { return View<true>{data(), extents(), strides()}; }
#endif

    [[nodiscard]] explicit constexpr operator std::array<T, (Ex * ... * 1UZ)>() const
    requires(detail::all_static_v<Ex...>)
    {
        if (rank() != 1UZ) {
            throw std::runtime_error("Can only convert 1D tensors to std::vector");
        }
        return std::array{_data};
    }

    template<typename Container>
    requires std::constructible_from<Container, const_iterator, const_iterator>
    [[nodiscard]] explicit operator Container() const {
        if (rank() != 1UZ) {
            throw std::runtime_error("can only convert 1D tensors to std containers");
        }
        return Container(begin(), end());
    }

    [[nodiscard]] explicit constexpr operator Tensor<std::remove_const_t<T>>() const {
        Tensor<std::remove_const_t<T>> result(extents_from, std::span{extents().data(), rank()});
        std::size_t                    i = 0;
        for (auto&& v : *this) {
            result.data()[i++] = v; // works both with contiguous a non-contigous source data
        }
        return result;
    }

    [[nodiscard]] explicit operator TensorView<T, Ex...>() noexcept { return TensorView<T, Ex...>(*this); }
    [[nodiscard]] explicit operator TensorView<const T, Ex...>() const noexcept { return TensorView<const T, Ex...>(*this); }

    void reshape(std::initializer_list<std::size_t> newExtents) { reshape(std::span(newExtents)); }

    template<std::ranges::range Range>
    requires(std::same_as<std::ranges::range_value_t<Range>, std::size_t>)
    void reshape(const Range& newExtents)
    requires((sizeof...(Ex) == 0UZ) || ((Ex == std::dynamic_extent) && ...))
    {
        const std::size_t newN = product(std::span(newExtents));
        if (newN != size()) {
            throw std::runtime_error("TensorBase::reshape: size mismatch");
        }
        if constexpr (sizeof...(Ex) == 0UZ) {
            _metaInfo.rank = std::ranges::size(newExtents);
        }
        std::ranges::copy_n(std::ranges::begin(newExtents), static_cast<std::ptrdiff_t>(std::ranges::size(newExtents)), _metaInfo.extents.begin());
        recomputeStrides();
    }

    [[nodiscard]] TensorView<T> slice(std::initializer_list<std::pair<std::size_t, std::size_t>> ranges) const {
        assert(ranges.size() <= rank());

        std::vector<std::size_t> new_extents;
        std::vector<std::size_t> new_strides;
        std::size_t              offset = 0;

        auto range_it = ranges.begin();
        for (std::size_t dim = 0; dim < rank(); ++dim) {
            if (range_it != ranges.end()) {
                auto [start, end] = *range_it;
                assert(start <= end && end <= extents()[dim]);
                new_extents.push_back(end - start);
                new_strides.push_back(strides()[dim]);
                offset += start * strides()[dim];
                ++range_it;
            } else {
                // keep full dimension if not specified
                new_extents.push_back(extents()[dim]);
                new_strides.push_back(strides()[dim]);
            }
        }

        return TensorView(const_cast<T*>(_data.data()) + offset, new_extents, new_strides);
    }

    template<typename Range>
    [[nodiscard]] TensorView<T> slice(const Range& ranges) const {
        static_assert(std::is_same_v<typename Range::value_type, std::pair<std::size_t, std::size_t>>, "slice range must hold pairs of (start, end) indices");

        assert(ranges.size() <= rank());

        std::vector<std::size_t> new_extents;
        std::vector<std::size_t> new_strides;
        std::size_t              offset = 0;

        auto range_it = ranges.begin();
        for (std::size_t dim = 0; dim < rank(); ++dim) {
            if (range_it != ranges.end()) {
                auto [start, end] = *range_it;
                assert(start <= end && end <= extents()[dim]);
                new_extents.push_back(end - start);
                new_strides.push_back(strides()[dim]);
                offset += start * strides()[dim];
                ++range_it;
            } else {
                new_extents.push_back(extents()[dim]);
                new_strides.push_back(strides()[dim]);
            }
        }

        return TensorView<T>(_data.data() + offset, new_extents, new_strides);
    }

    [[nodiscard]] TensorView<T> transpose() const {
        const std::size_t                         r = rank();
        std::array<std::size_t, detail::kMaxRank> new_extents{};
        std::array<std::size_t, detail::kMaxRank> new_strides{};
        for (std::size_t i = 0; i < r; ++i) {
            new_extents[i] = extents()[r - 1 - i];
            new_strides[i] = strides()[r - 1 - i];
        }
        return TensorView<T>(const_cast<T*>(_data.data()), std::span<std::size_t>{new_extents.data(), r}, std::span<std::size_t>{new_strides.data(), r});
    }

    [[nodiscard]] TensorView<T> transpose(std::initializer_list<std::size_t> axes) const {
        const auto r = rank();
        assert(axes.size() == r);
        // (Optional sanity) ensure axes is a permutation of [0..r)
#ifndef NDEBUG
        std::array<bool, detail::kMaxRank> seen{false};
        for (auto a : axes) {
            assert(a < r && !seen[a]);
            seen[a] = true;
        }
#endif

        std::array<std::size_t, detail::kMaxRank> new_extents{};
        std::array<std::size_t, detail::kMaxRank> new_strides{};
        std::size_t                               i = 0UZ;
        for (auto axis : axes) {
            new_extents[i] = extents()[axis];
            new_strides[i] = strides()[axis];
            ++i;
        }
        return TensorView<T>(const_cast<T*>(_data.data()), std::span<std::size_t>{new_extents.data(), r}, std::span<std::size_t>{new_strides.data(), r});
    }

    template<typename U, bool OtherManaged, std::size_t... OtherEx>
    requires std::three_way_comparable_with<T, U>
    [[nodiscard]] constexpr auto operator<=>(const TensorBase<U, OtherManaged, OtherEx...>& other) const noexcept {
        if (auto cmp = rank() <=> other.rank(); cmp != 0) {
            return cmp;
        }

        const auto my_extents    = extents();
        const auto other_extents = other.extents();

        for (std::size_t i = 0UZ; i < rank(); ++i) {
            if (auto cmp = my_extents[i] <=> other_extents[i]; cmp != 0) {
                return cmp;
            }
        }

        return std::lexicographical_compare_three_way(begin(), end(), other.begin(), other.end(), std::compare_three_way{});
    }

    template<typename U, bool OtherManaged, std::size_t... OtherEx>
    requires std::equality_comparable_with<T, U>
    [[nodiscard]] constexpr bool operator==(const TensorBase<U, OtherManaged, OtherEx...>& other) const noexcept {
        if (rank() != other.rank()) {
            return false;
        }

        const auto my_extents    = extents();
        const auto other_extents = other.extents();

        for (std::size_t i = 0UZ; i < rank(); ++i) {
            if (my_extents[i] != other_extents[i]) {
                return false;
            }
        }

        return std::equal(begin(), end(), other.begin(), other.end());
    }

    template<std::ranges::contiguous_range Container>
    requires(std::three_way_comparable_with<T, typename Container::value_type> && !std::derived_from<std::remove_cvref_t<Container>, TensorBase>)
    [[nodiscard]] constexpr auto operator<=>(const Container& container) const noexcept {
        if (rank() != 1UZ) {
            return std::strong_ordering::greater;
        }

        if (auto cmp = size() <=> container.size(); cmp != 0) {
            return cmp;
        }

        return std::lexicographical_compare_three_way(begin(), end(), container.begin(), container.end(), std::compare_three_way{});
    }

    template<std::ranges::contiguous_range Container>
    requires(std::three_way_comparable_with<T, typename Container::value_type> && !std::derived_from<std::remove_cvref_t<Container>, TensorBase>)
    [[nodiscard]] constexpr bool operator==(const Container& container) const noexcept {
        return rank() == 1UZ && size() == container.size() && std::equal(begin(), end(), container.begin(), container.end());
    }

    template<std::ranges::contiguous_range Container>
    requires(std::three_way_comparable_with<T, typename Container::value_type> && !std::derived_from<std::remove_cvref_t<Container>, TensorBase>)
    [[nodiscard]] friend constexpr auto operator<=>(const Container& container, const TensorBase& tensor) noexcept {
        auto cmp = tensor <=> container;
        if (cmp < 0) {
            return decltype(cmp)::greater;
        }
        if (cmp > 0) {
            return decltype(cmp)::less;
        }
        if constexpr (requires { decltype(cmp)::equivalent; }) {
            return decltype(cmp)::equivalent;
        } else {
            return decltype(cmp)::equal;
        }
    }

    template<std::ranges::contiguous_range Container>
    requires(std::three_way_comparable_with<T, typename Container::value_type> && !std::derived_from<std::remove_cvref_t<Container>, TensorBase>)
    [[nodiscard]] friend constexpr bool operator==(const Container& container, const TensorBase& tensor) noexcept {
        return tensor == container;
    }
}; // struct TensorBase { ... }

template<typename T, bool managed>
void swap(TensorBase<T, managed>& lhs, TensorBase<T, managed>& rhs) noexcept {
    lhs.swap(rhs);
}

template<typename T>
void swap(Tensor<T>& lhs, Tensor<T>& rhs) noexcept {
    lhs.swap(rhs);
}

template<typename T, std::size_t... Ex>
requires(detail::all_static_v<Ex...>)
struct Tensor<T, Ex...> : TensorBase<T, true, Ex...> { // fully static
    using base_t = TensorBase<T, true, Ex...>;
    using base_t::base_t;
    using extents_store_t = base_t::extents_store_t;
    using container_type  = base_t::container_type;
    using base_t::extent;

    template<std::size_t idx>
    static consteval std::size_t extent() {
        constexpr std::size_t E[]{Ex...};
        return E[idx];
    }

    Tensor(const Tensor& other)                     = default;
    Tensor(Tensor&& other) noexcept                 = default;
    Tensor& operator=(Tensor&& other) noexcept      = default;
    Tensor& operator=(const Tensor& other) noexcept = default;

    constexpr Tensor(std::initializer_list<T> values) {
        if (values.size() != base_t::size()) {
            throw std::runtime_error("Initializer list size doesn't match tensor size");
        }
        std::copy(values.begin(), values.end(), base_t::_data.begin());
    }

    constexpr Tensor(const T& v) { base_t::fill(v); }

    // 2D nested initializer list
    constexpr Tensor(std::initializer_list<std::initializer_list<T>> values)
    requires(sizeof...(Ex) == 2UZ)
    {
        constexpr std::array<std::size_t, 2UZ> dims{Ex...};
        if (values.size() != dims[0UZ]) {
            throw std::runtime_error("Wrong number of rows");
        }

        std::size_t idx = 0UZ;
        for (auto row : values) {
            if (row.size() != dims[1UZ]) {
                throw std::runtime_error("wrong number of columns");
            }
            for (auto val : row) {
                base_t::_data[idx++] = val;
            }
        }
    }

    // 3D nested initializer list
    constexpr Tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>> values)
    requires(sizeof...(Ex) == 3UZ)
    {
        constexpr std::array<std::size_t, 3> dims{Ex...};
        if (values.size() != dims[0]) {
            throw std::runtime_error("Wrong dimension 0 size");
        }

        std::size_t idx = 0;
        for (auto plane : values) {
            if (plane.size() != dims[1]) {
                throw std::runtime_error("Wrong dimension 1 size");
            }
            for (auto row : plane) {
                if (row.size() != dims[2]) {
                    throw std::runtime_error("Wrong dimension 2 size");
                }
                for (auto val : row) {
                    base_t::_data[idx++] = val;
                }
            }
        }
    }

    template<std::size_t... OtherEx>
    requires(sizeof...(Ex) == 0 || sizeof...(OtherEx) == 0 || sizeof...(Ex) == sizeof...(OtherEx))
    explicit Tensor(const Tensor<T, OtherEx...>& other) {
        constexpr std::size_t dstRank = sizeof...(Ex);
        constexpr std::size_t dstN    = (Ex * ... * 1UZ);

        if constexpr (sizeof...(OtherEx) > 0) { // source also static: ranks must match at compile time
            static_assert(sizeof...(OtherEx) == dstRank, "rank mismatch");
            constexpr std::size_t srcN = (OtherEx * ... * 1UZ);
            if constexpr (srcN != dstN) {
                throw std::runtime_error("Tensor: size mismatch in converting ctor - static to static with different sizes.");
            }
        } else {
            // source dynamic: check at runtime
            if (other.rank() != dstRank) [[unlikely]] {
                throw std::runtime_error("Tensor: rank mismatch in converting ctor.");
            }
            for (std::size_t i = 0; i < dstRank; ++i) {
                if (other.extent(i) != base_t::extent(i)) [[unlikely]] {
                    throw std::runtime_error("Tensor: extent mismatch in converting ctor.");
                }
            }
        }

        std::ranges::copy_n(other.begin(), dstN, this->begin());
    }

    // replace the existing template assignment operator with this simplified version
    template<std::size_t... OtherEx>
    Tensor& operator=(const Tensor<T, OtherEx...>& other) {
        if constexpr (std::is_same_v<Tensor, Tensor<T, OtherEx...>>) {
            if (this == &other) {
                return *this;
            }
        }

        Tensor temp(other);
        base_t::swap(temp);
        return *this;
    }

    template<std::ranges::range Range>
    requires(std::same_as<std::remove_cvref_t<std::ranges::range_value_t<Range>>, T> && !TensorLike<Range>)
    constexpr Tensor& operator=(const Range& vec) {
        const auto n = vec.size();
        if (n != base_t::size()) {
            throw std::runtime_error("static Tensor::operator=(Range): size mismatch with existing shape");
        }
        std::copy(vec.begin(), vec.end(), base_t::_data.begin());

        return *this;
    }

    Tensor& operator=(std::initializer_list<T> initializer_list) {
        const auto n = initializer_list.size();

        // fully dynamic rank & no extents yet → define 1D shape from init-list
        if (base_t::extents().empty()) {
            base_t::_metaInfo.extents[0UZ] = n;
            base_t::_data.assign(initializer_list.begin(), initializer_list.end());
            return *this;
        }

        // shape already defined → size must match
        if (n != base_t::size()) {
            throw std::runtime_error("TensorBase::operator=: initializer_list size mismatch");
        }
        std::copy(initializer_list.begin(), initializer_list.end(), base_t::_data.begin());
        return *this;
    }

    [[nodiscard]] static constexpr Tensor identity() {
        constexpr std::size_t N        = (Ex * ...);
        Tensor<T, N>          identity = T(0);
        for (std::size_t i = 0UZ; i < N; ++i) {
            identity[i, i] = T{1};
        }
        return identity;
    }

    template<std::ranges::range Range>
    requires std::same_as<std::ranges::range_value_t<Range>, T>
    Tensor& assign(const Range& range) {
        const auto n = static_cast<std::size_t>(std::ranges::size(range));

        // dynamic tensors w/o shape -> create 1D shape
        if (base_t::_metaInfo.rank == 0UZ) {
            base_t::_metaInfo.rank       = 1;
            base_t::_metaInfo.extents[0] = n;
            base_t::_data.resize(n);
            std::ranges::copy(range, base_t::_data.begin());
            return *this;
        }

        if (n != base_t::size()) {
            throw std::runtime_error("TensorBase::assign: range size doesn't match tensor size");
        }
        std::ranges::copy(range, base_t::_data.begin());
        return *this;
    }

    void assign(std::size_t count, const T& value) {
        if (count != base_t::size()) {
            throw std::runtime_error("TensorBase::assign(count,value): count mismatch with existing shape");
        }
        std::fill(base_t::_data.begin(), base_t::_data.end(), value);
    }

    constexpr Tensor& operator=(const T& value) {
        base_t::fill(value);
        return *this;
    }
}; // struct Tensor -- fully static

template<typename T, std::size_t... Ex>
requires(not detail::all_static_v<Ex...>)
struct Tensor<T, Ex...> : TensorBase<T, true, Ex...> { // fully or partially dynamic
    using base_t = TensorBase<T, true, Ex...>;
    using base_t::base_t;
    using extents_store_t = base_t::extents_store_t;
    using container_type  = base_t::container_type;
    using base_t::extents;
    using base_t::strides;

    Tensor(Tensor&& other) noexcept = default;
    explicit Tensor(const T& v, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        base_t::_data = make_container(mr);
        base_t::fill(v);
    }

    Tensor(std::pmr::memory_resource* mr = std::pmr::get_default_resource()) noexcept {
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = 0UZ;
            std::ranges::fill(base_t::_metaInfo.extents, 0UZ);
        }
        base_t::recomputeStrides();
        base_t::_data = make_container(mr);
    }

    template<typename OtherT, std::size_t... OtherEx>
    explicit Tensor(Tensor<OtherT, OtherEx...>&& other, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        base_t::_data = make_container(mr);

        // copy shape info
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = other.rank();
        }
        std::ranges::copy_n(other._metaInfo.extents.begin(), static_cast<std::ptrdiff_t>(other.rank()), base_t::_metaInfo.extents.begin());
        std::ranges::copy_n(other._metaInfo.strides.begin(), static_cast<std::ptrdiff_t>(other.rank()), base_t::_metaInfo.strides.begin());

        // move data
        base_t::_data.reserve(other._data.size());
        for (auto&& elem : other._data) {
            base_t::_data.push_back(static_cast<T>(std::move(elem)));
        }
    }
    template<typename OtherT, std::size_t... OtherEx>
    explicit Tensor(const Tensor<OtherT, OtherEx...>& other, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        base_t::_data = make_container(mr);

        // Copy shape info
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = other.rank();
        }
        std::ranges::copy_n(other._metaInfo.extents.begin(), static_cast<std::ptrdiff_t>(other.rank()), base_t::_metaInfo.extents.begin());
        std::ranges::copy_n(other._metaInfo.strides.begin(), static_cast<std::ptrdiff_t>(other.rank()), base_t::_metaInfo.strides.begin());

        // Copy data with conversion
        base_t::_data.reserve(other._data.size());
        for (const auto& elem : other._data) {
            base_t::_data.push_back(static_cast<T>(elem));
        }
    }

    template<std::ranges::range Extents>
    requires(std::same_as<std::ranges::range_value_t<Extents>, std::size_t> && !std::same_as<std::remove_cvref_t<Extents>, std::initializer_list<std::size_t>>)
    explicit Tensor(const Extents& extents_, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        auto ext_size = std::ranges::size(extents_);
        if (ext_size > detail::kMaxRank) {
            throw std::runtime_error("Tensor: rank too large");
        }

        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = ext_size;
        } else if (sizeof...(Ex) != ext_size) {
            throw std::runtime_error("Tensor: rank mismatch");
        }

        std::ranges::copy_n(std::ranges::begin(extents_), static_cast<std::ptrdiff_t>(ext_size), base_t::_metaInfo.extents.begin());
        base_t::recomputeStrides();

        base_t::_data = make_container(mr);
        base_t::_data.resize(base_t::checked_size(base_t::extents()));
    }

    template<typename U> // 1. static rank-1: highest priority (most specific - rank is fixed, must be data)
    requires(sizeof...(Ex) == 1UZ && ((Ex == std::dynamic_extent) && ...) && std::convertible_to<U, T>)
    Tensor(std::initializer_list<U> values, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        base_t::_metaInfo.extents[0UZ] = values.size();
        base_t::recomputeStrides();
        base_t::_data = gr::pmr::vector<T, true>(values.begin(), values.end(), mr);
    }

    // 2. integral literals with non-integral T → extents (type mismatch disambiguates)
    Tensor(std::initializer_list<std::size_t> extents_, std::pmr::memory_resource* mr = std::pmr::get_default_resource())
    requires(!std::integral<T> && !(sizeof...(Ex) == 1UZ && ((Ex == std::dynamic_extent) && ...)))
    {
        *this = Tensor(gr::extents_from, std::span<const std::size_t>(extents_.begin(), extents_.size()), mr);
    }

    template<typename U>
    requires(!std::integral<T> && std::integral<U> && !std::same_as<U, std::size_t> && !(sizeof...(Ex) == 1UZ && ((Ex == std::dynamic_extent) && ...)))
    Tensor(std::initializer_list<U> extents_, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        std::vector<std::size_t> dims(extents_.begin(), extents_.end());
        *this = Tensor(gr::extents_from, std::span<const std::size_t>(dims.data(), dims.size()), mr);
    }

    template<typename U> // 3. size_t literals with integral T (but T != size_t) → extents
    requires(std::integral<T> && std::same_as<U, std::size_t> && !std::same_as<T, std::size_t> && !(sizeof...(Ex) == 1UZ && ((Ex == std::dynamic_extent) && ...)))
    Tensor(std::initializer_list<U> extents_, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) : Tensor(gr::extents_from, std::span<const std::size_t>(extents_.begin(), extents_.end()), mr) {}

    template<typename U>                                                              // 4. Same integral type for T and U (or implicitly convertible) → AMBIGUOUS
    requires(std::integral<T> && std::integral<U> && !std::same_as<U, std::size_t> && // Exclude size_t (handled above)
             !(sizeof...(Ex) == 1UZ && ((Ex == std::dynamic_extent) && ...)))
    Tensor(std::initializer_list<U>, std::pmr::memory_resource* = std::pmr::get_default_resource()) {
        static_assert(gr::meta::always_false<T>, "ambiguous constructor: please use explicit Tensor<T>(gr::extents_from, {...}) or Tensor<T>(gr::data_from, {...})");
    }

    template<typename U> // 5. Non-integral literals → data (rank-1)
    requires(!std::integral<U> && std::convertible_to<U, T> && (sizeof...(Ex) == 0UZ || (sizeof...(Ex) == 1UZ && ((Ex == std::dynamic_extent) && ...))))
    Tensor(std::initializer_list<U> values, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = 1UZ;
        } else if (sizeof...(Ex) != 1UZ) {
            throw std::runtime_error("Tensor: data requires rank-1 tensor.");
        }
        base_t::_metaInfo.extents[0UZ] = values.size();
        base_t::recomputeStrides();
        base_t::_data = gr::pmr::vector<T, true>(values.begin(), values.end(), mr);
    }

    // 6. Explicit extents tag (always available)
    Tensor(tensor_extents_tag, std::initializer_list<std::size_t> extents_, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) : Tensor(gr::extents_from, std::span<const std::size_t>(extents_.begin(), extents_.end()), mr) {}

    template<typename U> // 7. Explicit data tag (always available)
    requires(std::convertible_to<U, T>)
    Tensor(tensor_data_tag, std::initializer_list<U> values, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = 1UZ;
        } else if (sizeof...(Ex) != 1UZ) {
            throw std::runtime_error("Tensor: gr::data_from requires rank-1 tensor.");
        }
        base_t::_metaInfo.extents[0UZ] = values.size();
        base_t::recomputeStrides();
        base_t::_data = gr::pmr::vector<T, true>(values.begin(), values.end(), mr);
    }

    template<std::ranges::range Extents, std::ranges::range Data>
    requires(std::same_as<std::ranges::range_value_t<Extents>, std::size_t> && std::same_as<std::ranges::range_value_t<Data>, T>)
    Tensor(const Extents& extents_, const Data& data, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        if (extents.size() > detail::kMaxRank) {
            throw std::runtime_error("Tensor: rank too large for Tensor.");
        }
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = extents_.size();
        } else if (sizeof...(Ex) != extents_.size()) {
            throw std::runtime_error("Tensor: provided rank incompatible to pre-defined extents.");
        }
        std::ranges::copy_n(extents_, extents_.size(), base_t::_metaInfo.extents.begin());
        base_t::recomputeStrides();

        base_t::_data = container_type(std::ranges::begin(data), std::ranges::end(data), mr);
        if (base_t::_data.size() != base_t::checked_size(base_t::extents())) {
            throw std::runtime_error("TensorBase: data size doesn't match extents product.");
        }
    }

    template<std::ranges::range Range>
    requires std::same_as<std::ranges::range_value_t<Range>, std::size_t>
    Tensor(tensor_extents_tag, const Range& extents_, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        if (extents_.size() > detail::kMaxRank) {
            throw std::runtime_error("Tensor: rank too large for Tensor.");
        }
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = extents_.size();
        } else if (sizeof...(Ex) != extents_.size()) {
            throw std::runtime_error("Tensor: provided rank incompatible to pre-defined extents.");
        }
        std::ranges::copy_n(extents_.begin(), static_cast<std::ptrdiff_t>(std::ranges::size(extents_)), base_t::_metaInfo.extents.begin());
        base_t::recomputeStrides();

        base_t::_data = container_type(mr);

        base_t::_data.resize(base_t::checked_size(base_t::extents()));
    }

    template<std::ranges::range Range>
    requires std::same_as<std::ranges::range_value_t<Range>, T>
    Tensor(tensor_data_tag, const Range& data, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = 1UZ;
        } else if (sizeof...(Ex) != 1UZ) {
            throw std::runtime_error("Tensor: provided rank incompatible to pre-defined extents.");
        }
        base_t::_metaInfo.extents[0] = std::ranges::size(data);
        base_t::recomputeStrides();

        base_t::_data = container_type(std::ranges::begin(data), std::ranges::end(data), mr);
    }

    Tensor(std::size_t count, const T& value, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = 1UZ;
        } else if (sizeof...(Ex) != 1UZ) {
            throw std::runtime_error("Tensor: provided rank incompatible to pre-defined extents.");
        }
        base_t::_metaInfo.extents[0UZ] = count;
        base_t::recomputeStrides();

        base_t::_data = container_type(count, value, mr);
    }

    template<std::input_iterator InputIt>
    Tensor(InputIt first, InputIt last, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = 1UZ;
        } else if (sizeof...(Ex) != 1UZ) {
            throw std::runtime_error("Tensor: provided rank incompatible to pre-defined extents.");
        }
        std::ptrdiff_t dist = std::distance(first, last);
        if (dist < 0) {
            throw std::runtime_error("Tensor: negative first -> last iterator distance.");
        }
        base_t::_metaInfo.extents[0UZ] = static_cast<std::size_t>(dist);
        base_t::recomputeStrides();

        base_t::_data = container_type(first, last, mr);
    }

    template<std::ranges::range Data>
    requires std::same_as<std::ranges::range_value_t<Data>, T>
    Tensor(std::initializer_list<std::size_t> extents_, const Data& data, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        if (extents_.size() > detail::kMaxRank) {
            throw std::runtime_error("Tensor: rank too large for Tensor.");
        }
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = extents_.size();
        } else if (sizeof...(Ex) != extents_.size()) {
            throw std::runtime_error("Tensor: provided rank incompatible to pre-defined extents.");
        }
        std::ranges::copy_n(extents_.begin(), static_cast<std::ptrdiff_t>(extents_.size()), base_t::_metaInfo.extents.begin());
        base_t::recomputeStrides();

        base_t::_data = container_type(std::ranges::begin(data), std::ranges::end(data), mr);
        if (base_t::_data.size() != base_t::checked_size(base_t::extents())) {
            throw std::runtime_error("Tensor: data size doesn't match extents product.");
        }
    }

    // two initializer_lists: extents + data
    template<typename U>
    requires std::convertible_to<U, T>
    Tensor(std::initializer_list<std::size_t> extents_, std::initializer_list<U> data, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        if (extents_.size() > detail::kMaxRank) {
            throw std::runtime_error("Tensor: rank too large for Tensor.");
        }
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = extents_.size();
        } else if (sizeof...(Ex) != extents_.size()) {
            throw std::runtime_error("Tensor: provided rank incompatible to pre-defined extents.");
        }
        std::ranges::copy_n(extents_.begin(), static_cast<std::ptrdiff_t>(extents_.size()), base_t::_metaInfo.extents.begin());
        base_t::recomputeStrides();

        base_t::_data = container_type(data.begin(), data.end(), mr);
        if (base_t::_data.size() != base_t::checked_size(base_t::extents())) {
            throw std::runtime_error("Tensor: data size doesn't match extents product.");
        }
    }

    template<std::ranges::range Range>
    requires(std::same_as<std::remove_cvref_t<std::ranges::range_value_t<Range>>, T> && !TensorLike<Range>)
    explicit Tensor(const Range& values, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = 1UZ;
        } else if constexpr (sizeof...(Ex) != 1UZ) {
            throw std::runtime_error("Vector constructor only for 1D tensors");
        }

        base_t::_metaInfo.extents[0] = values.size();
        base_t::recomputeStrides();

        base_t::_data = make_container(mr);
        base_t::_data.assign(values.begin(), values.end());
    }

    explicit Tensor(std::pmr::vector<T>&& vec, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = 1UZ;
        } else if constexpr (sizeof...(Ex) != 1UZ) {
            throw std::runtime_error("Vector constructor only for 1D tensors");
        }

        base_t::_metaInfo.extents[0] = vec.size();
        base_t::recomputeStrides();
        base_t::_data = container_type(std::move(vec), mr);
    }

    Tensor(const Tensor& other, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        base_t::_metaInfo = other._metaInfo;

        if constexpr (sizeof...(Ex) == 0) {
            base_t::_metaInfo.rank = other._metaInfo.rank;
        }
        base_t::_metaInfo.extents = other._metaInfo.extents;
        base_t::recomputeStrides();

        // create container with correct size before copying
        base_t::_data = container_type(other._data.size(), T{}, mr);
        std::copy(other._data.begin(), other._data.end(), base_t::_data.begin());
    }

    // 2D nested initializer list
    constexpr Tensor(std::initializer_list<std::initializer_list<T>> values)
    requires(sizeof...(Ex) == 2UZ)
    {
        constexpr std::array<std::size_t, 2UZ> dims{Ex...};
        if (values.size() != dims[0UZ]) {
            throw std::runtime_error("Wrong number of rows");
        }

        std::size_t idx = 0UZ;
        for (auto row : values) {
            if (row.size() != dims[1UZ]) {
                throw std::runtime_error("Wrong number of columns");
            }
            for (auto val : row) {
                base_t::_data[idx++] = val;
            }
        }
    }

    // 3D nested initializer list
    constexpr Tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>> values)
    requires(sizeof...(Ex) == 3UZ)
    {
        constexpr std::array<std::size_t, 3> dims{Ex...};
        if (values.size() != dims[0]) {
            throw std::runtime_error("Wrong dimension 0 size");
        }

        std::size_t idx = 0;
        for (auto plane : values) {
            if (plane.size() != dims[1]) {
                throw std::runtime_error("Wrong dimension 1 size");
            }
            for (auto row : plane) {
                if (row.size() != dims[2]) {
                    throw std::runtime_error("Wrong dimension 2 size");
                }
                for (auto val : row) {
                    base_t::_data[idx++] = val;
                }
            }
        }
    }

    template<typename OtherT, std::size_t... OtherEx>
    requires(std::convertible_to<OtherT, T> && (sizeof...(Ex) == 0 || sizeof...(OtherEx) == 0 || sizeof...(Ex) == sizeof...(OtherEx)))
    explicit Tensor(const Tensor<OtherT, OtherEx...>& other, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        if (other.rank() > detail::kMaxRank) {
            throw std::runtime_error("Tensor: rank too large for Tensor.");
        }
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = other.rank();
        } else if (sizeof...(Ex) != other.rank()) {
            throw std::runtime_error("Tensor: provided rank incompatible to pre-defined extents.");
        }
        std::ranges::copy_n(other._metaInfo.extents.begin(), static_cast<std::ptrdiff_t>(other.rank()), base_t::_metaInfo.extents.begin());
        base_t::recomputeStrides();

        base_t::_data = make_container(mr);
        base_t::_data.assign(other.begin(), other.end());
    }

    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            auto* my_resource    = base_t::resource();
            auto* other_resource = other.resource();

            if (my_resource == other_resource) { // same resource - can move directly
                base_t::_metaInfo = std::move(other._metaInfo);
                base_t::_data     = std::move(other._data);
            } else { // different resources - need to copy data
                base_t::_metaInfo = other._metaInfo;

                base_t::_data = container_type(my_resource);
                base_t::_data.assign(other._data.begin(), other._data.end());
            }
        }
        return *this;
    }

    template<typename OtherT, std::size_t... OtherEx>
    Tensor& operator=(const Tensor<OtherT, OtherEx...>& other) {
        // Handle self-assignment for identical types
        if constexpr (std::is_same_v<Tensor, Tensor<OtherT, OtherEx...>>) {
            if (this == &other) {
                return *this;
            }
        }

        Tensor temp(other, base_t::resource());
        base_t::swap(temp);
        return *this;
    }

    template<std::ranges::range Range>
    requires(std::same_as<std::remove_cvref_t<std::ranges::range_value_t<Range>>, T> && !TensorLike<Range>)
    Tensor& operator=(const Range& vec) {
        const auto n = static_cast<std::size_t>(std::ranges::size(vec));

        // fully dynamic rank and shape not set yet → define 1-D shape from the range
        if constexpr (sizeof...(Ex) == 0UZ) {
            if (base_t::rank() == 0UZ) {
                base_t::_metaInfo.rank         = 1UZ;
                base_t::_metaInfo.extents[0UZ] = n;
                base_t::recomputeStrides();

                // vector/pmr::vector-like container_type: assign from iterators
                base_t::_data.assign(std::ranges::begin(vec), std::ranges::end(vec));
                return *this;
            }
        }

        // shape already defined → size must match
        if (n != base_t::size()) {
            throw std::runtime_error("dynamic Tensor::operator=(range): size mismatch with existing shape");
        }

        std::ranges::copy(vec, base_t::_data.begin());
        return *this;
    }

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            base_t::_metaInfo = other._metaInfo;

            base_t::_data = container_type(base_t::resource());
            base_t::_data.assign(other._data.begin(), other._data.end());
        }
        return *this;
    }

    template<std::ranges::range Range>
    requires std::same_as<std::ranges::range_value_t<Range>, T>
    Tensor& assign(const Range& values) {
        const auto n = static_cast<std::size_t>(std::ranges::size(values));

        // fully dynamic rank w/o extents -> treat as 1D initialisation
        if (extents().empty()) {
            base_t::_metaInfo.rank       = 1UZ;
            base_t::_metaInfo.extents[0] = n;
            base_t::_data.assign(std::ranges::begin(values), std::ranges::end(values));
            return *this;
        }

        // shape already defined -> just overwrite data (size must match)
        if (n != base_t::size()) {
            throw std::runtime_error("TensorBase::assign: values size doesn't match tensor size");
        }
        std::ranges::copy_n(values.begin(), static_cast<std::ptrdiff_t>(values.size()), base_t::_data.begin());
        return *this;
    }

    Tensor& operator=(std::initializer_list<T> initializer_list) {
        const auto n = initializer_list.size();

        // fully dynamic rank w/o extents -> define 1D shape from init-list
        if (base_t::_metaInfo.rank == 0UZ) {
            base_t::_metaInfo.rank       = 1UZ;
            base_t::_metaInfo.extents[0] = n;
            base_t::recomputeStrides();
            base_t::_data.assign(initializer_list.begin(), initializer_list.end());
            return *this;
        }

        // shape already defined → size must match
        if (n != base_t::size()) {
            throw std::runtime_error("TensorBase::operator=: initializer_list size mismatch");
        }
        std::copy(initializer_list.begin(), initializer_list.end(), base_t::_data.begin());
        return *this;
    }

    void assign(std::size_t count, const T& value) {
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = 1UZ;
        } else if constexpr (sizeof...(Ex) != 1UZ) {
            throw std::runtime_error("assign(count, value) only for 1D tensors");
        }

        base_t::_metaInfo.extents[0] = count;
        base_t::recomputeStrides();
        base_t::_data.assign(count, value);
    }

    constexpr Tensor& operator=(const T& value) {
        base_t::fill(value);
        return *this;
    }

    void push_back(const T& value) {
        if (base_t::rank() > 1) {
            // for multi-dim tensors, flatten to 1D
            if constexpr (sizeof...(Ex) == 0UZ) {
                base_t::_metaInfo.rank = 1UZ;
            }
            base_t::_metaInfo.extents[0] = base_t::size();
            std::ranges::fill(base_t::_metaInfo.extents.begin() + 1, base_t::_metaInfo.extents.end(), 0UZ);
        } else if (base_t::rank() == 0) {
            // for uninitialized tensors, initialize as 1D
            if constexpr (sizeof...(Ex) == 0UZ) {
                base_t::_metaInfo.rank = 1UZ; // ADD: Set rank!
            }
            base_t::_metaInfo.extents[0] = 0;
        }

        base_t::recomputeStrides();
        base_t::_data.resize(base_t::size() + 1UZ);
        base_t::_data.back() = value;
        ++base_t::_metaInfo.extents[0];
    }

    void push_back(T&& value) {
        if (base_t::rank() > 1) {
            base_t::_metaInfo.rank       = 1UZ;
            base_t::_metaInfo.extents[0] = base_t::size();
        } else if (base_t::rank() == 0) {
            base_t::_metaInfo.rank       = 1UZ;
            base_t::_metaInfo.extents[0] = 0UZ;
        }
        base_t::_data.resize(base_t::size() + 1UZ);
        base_t::_data.back() = std::move(value);
        ++base_t::_metaInfo.extents[0];
    }

    template<typename... Args>
    T& emplace_back(Args&&... args) {
        if (base_t::rank() > 1) {
            base_t::_metaInfo.extents = {base_t::size()};
        } else if (base_t::rank() == 0) {
            base_t::_metaInfo.extents = {0};
        }
        auto& ref = base_t::_data.emplace_back(std::forward<Args>(args)...);
        ++base_t::_metaInfo.extents[0];
        return ref;
    }

    void pop_back() {
        if (base_t::empty()) {
            throw std::runtime_error("pop_back on empty tensor");
        }

        if (base_t::rank() <= 1) {
            base_t::_data.pop_back();
            if (base_t::rank() == 1) {
                --base_t::_metaInfo.extents[0];
                if (base_t::_metaInfo.extents[0] == 0) {
                    std::ranges::fill(base_t::_metaInfo.extents, 0UZ);
                }
            }
        } else {
            base_t::_metaInfo.extents = {base_t::size()};
            base_t::_data.pop_back();
            --base_t::_metaInfo.extents[0];
        }
    }

private:
    static container_type make_container(std::pmr::memory_resource* mr) { return container_type(mr); }
}; // struct Tensor -- partially/fully dynamic

// ---- CTAD guides ----
template<typename T>
Tensor(std::initializer_list<T>) -> Tensor<T>;

template<typename T>
Tensor(std::initializer_list<std::initializer_list<T>>) -> Tensor<T>;

template<typename T, std::size_t N>
Tensor(const std::array<T, N>&) -> Tensor<T>;

template<std::ranges::range Extents, std::ranges::range Data>
Tensor(const Extents&, const Data&, std::pmr::memory_resource* = std::pmr::get_default_resource()) -> Tensor<std::ranges::range_value_t<Data>>;

template<std::ranges::range Range>
Tensor(tensor_data_tag, const Range&, std::pmr::memory_resource* = std::pmr::get_default_resource()) -> Tensor<std::ranges::range_value_t<Range>>;

template<std::ranges::range Range>
Tensor(tensor_extents_tag, const Range&, std::pmr::memory_resource* = std::pmr::get_default_resource()) -> Tensor<std::ranges::range_value_t<Range>>;

template<typename T>
Tensor(std::size_t, const T&, std::pmr::memory_resource* = std::pmr::get_default_resource()) -> Tensor<T>;

template<std::input_iterator InputIt>
Tensor(InputIt, InputIt, std::pmr::memory_resource* = std::pmr::get_default_resource()) -> Tensor<typename std::iterator_traits<InputIt>::value_type>;

template<typename T, class Allocator>
Tensor(const std::vector<T, Allocator>&, std::pmr::memory_resource* = std::pmr::get_default_resource()) -> Tensor<T>;

template<typename T>
Tensor(const std::pmr::vector<T>&, std::pmr::memory_resource* = std::pmr::get_default_resource()) -> Tensor<T>;

template<typename T>
Tensor(std::pmr::vector<T>&&, std::pmr::memory_resource* = std::pmr::get_default_resource()) -> Tensor<T>;

/**
 * @brief Non-owning view of a Tensor with optional striding and slicing
 *
 * TensorView provides a lightweight, non-owning view into tensor data.
 * It supports:
 * - Custom strides (for transposed/sliced views)
 * - Arbitrary data pointers (can view any contiguous memory)
 * - Conversion from Tensor
 * - Most const operations of Tensor
 */
template<typename T, std::size_t... Ex>
struct TensorView : TensorBase<T, false, Ex...> {
    static_assert(!std::is_same_v<T, std::string>);
    using base_t = TensorBase<T, false, Ex...>;
    using base_t::base_t;
    using value_type      = T;
    using pointer         = value_type*;
    using const_pointer   = const value_type*;
    using reference       = value_type&;
    using const_reference = const value_type&;

    TensorView()                                 = default;
    TensorView(const TensorView&)                = default;
    TensorView(TensorView&&) noexcept            = default;
    TensorView& operator=(const TensorView&)     = default;
    TensorView& operator=(TensorView&&) noexcept = default;

    // construct from pointer, extents, and strides
    template<typename ExtentRange, typename StrideRange>
    TensorView(T* data, const ExtentRange& extents, const StrideRange& strides) {
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = static_cast<std::size_t>(extents.size());
        }
        std::copy(extents.begin(), extents.end(), base_t::_metaInfo.extents.begin());
        std::copy(strides.begin(), strides.end(), base_t::_metaInfo.strides.begin());
        base_t::_data._size     = base_t::checked_size(base_t::extents());
        base_t::_data._capacity = base_t::_data._size;
        base_t::_data._data     = data;
    }

    // construct from pointer and extents (compute default strides)
    template<typename ExtentRange>
    TensorView(T* data, const ExtentRange& extents) {
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = extents.size();
        }
        std::copy(extents.begin(), extents.end(), base_t::_metaInfo.extents.begin());
        base_t::recomputeStrides();
        base_t::_data._size     = base_t::checked_size(base_t::extents());
        base_t::_data._capacity = base_t::_data.size();
        base_t::_data._data     = data;
    }

    template<typename U, std::size_t... VEx>
    requires(std::same_as<std::remove_const_t<T>, std::remove_const_t<U>> && !std::is_const_v<U>)
    TensorView(const TensorView<U, VEx...>& other) {
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = other.rank();
        }
        if constexpr (!detail::all_static_v<Ex...>) {
            std::copy_n(other.extents().begin(), other.rank(), base_t::_metaInfo.extents.begin());
            std::copy_n(other.strides().begin(), other.rank(), base_t::_metaInfo.strides.begin());
        }

        base_t::_data._data     = const_cast<base_t::element_type*>(other.data());
        base_t::_data._size     = other.size();
        base_t::_data._capacity = other.size();
        base_t::_data._resource = nullptr;
    }

    template<typename U, std::size_t... TEx>
    requires(std::same_as<std::remove_const_t<T>, std::remove_const_t<U>>)
    explicit TensorView(Tensor<U, TEx...>& tensor) {
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = tensor.rank();
        }

        if constexpr (!detail::all_static_v<Ex...>) {
            std::copy_n(tensor.extents().begin(), tensor.rank(), base_t::_metaInfo.extents.begin());
            std::copy_n(tensor.strides().begin(), tensor.rank(), base_t::_metaInfo.strides.begin());
        }

        base_t::_data._data     = tensor.data(); // alias
        base_t::_data._size     = tensor.size();
        base_t::_data._capacity = tensor.size();
        base_t::_data._resource = nullptr; // non-owning
    }

    template<typename U, std::size_t... TEx>
    requires(std::is_const_v<T> && std::same_as<std::remove_const_t<T>, std::remove_const_t<U>>)
    explicit TensorView(const Tensor<U, TEx...>& tensor) : TensorView(const_cast<Tensor<U, TEx...>&>(tensor)) {}

    template<typename U, std::size_t... VEx>
    requires(std::is_const_v<T> && std::same_as<std::remove_const_t<T>, std::remove_const_t<U>>)
    TensorView& operator=(TensorView<U, VEx...>& other) {
        if constexpr (sizeof...(Ex) == 0UZ) {
            base_t::_metaInfo.rank = other.rank();
        }
        if constexpr (!detail::all_static_v<Ex...>) {
            std::copy_n(other.extents().begin(), other.rank(), base_t::_metaInfo.extents.begin());
            std::copy_n(other.strides().begin(), other.rank(), base_t::_metaInfo.strides.begin());
        }

        base_t::_data._data     = other.data();
        base_t::_data._size     = other.size();
        base_t::_data._capacity = other.size();
        base_t::_data._resource = nullptr;

        return *this;
    }
};

// deduction guides
template<typename T, typename ExtentRange>
TensorView(T*, const ExtentRange&) -> TensorView<T>;
template<typename T, typename ExtentRange, typename StrideRange>
TensorView(T*, const ExtentRange&, const StrideRange&) -> TensorView<T>;
template<typename U, std::size_t... Ex>
TensorView(Tensor<U, Ex...>&) -> TensorView<U, Ex...>;
template<typename U, std::size_t... Ex>
TensorView(const Tensor<U, Ex...>&) -> TensorView<const U, Ex...>;
template<typename U, std::size_t... Ex>
TensorView(const TensorView<U, Ex...>&) -> TensorView<U, Ex...>;

template<typename T, std::size_t... Ex>
inline static constexpr bool is_tensor<Tensor<T, Ex...>> = true;

template<typename T, std::size_t... Ex>
inline static constexpr bool is_tensor<TensorView<T, Ex...>> = true;

} // namespace gr

#endif // GNURADIO_TENSOR_HPP
