#ifndef GNURADIO_PMRCOLLECTIONS_HPP
#define GNURADIO_PMRCOLLECTIONS_HPP

#include <gnuradio-4.0/MemoryAllocators.hpp>

namespace gr::pmr {

template<typename T, bool managed>
struct vector;

#ifdef _LIBCPP_VERSION
//
// uninitialized_move in libc++ tries to move from the iterator even if
// operator* returns a value and not a reference (std::views::transform
// for example).
//
// The __iter_move constructs a value (operator*) and then returns
// a reference to it, which leads to UB.
//
// auto __iter_move = [](auto&& __iter) -> decltype(auto) { return std::move(*__iter); };
//
// auto __result = std::__uninitialized_move<_ValueType>(std::move(__ifirst), std::move(__ilast), std::move(__ofirst), std::__always_false(), __iter_move);
// return std::move(__result.second);
//
// The fix applied here removes the UB while preserving move semantics
// in the case when it is possible.
//
template<class InputIterator, class ForwardIterator>
inline ForwardIterator libcxx_fixed_uninitialized_move(InputIterator ifirst, InputIterator ilast, ForwardIterator ofirst) {
    using ValueType = typename std::iterator_traits<ForwardIterator>::value_type;
    if constexpr (std::is_reference_v<ValueType>) {
        return std::uninitialized_move(ifirst, ilast, ofirst);
    } else {
        // If operator* returns a value, construction will be a move
        // from the returned value, as uninitialized_copy doesn't try
        // to do anyting smart as std::uninitialized_move does in libc++
        return std::uninitialized_copy(ifirst, ilast, ofirst);
    }
}
#endif

template<class T>
struct vector<T, false> { // trivially copyable, no lifetime management
    using iterator       = T*;
    using const_iterator = const T*;

    std::pmr::memory_resource* _resource{std::pmr::get_default_resource()};
    std::size_t                _size{0UZ};
    std::size_t                _capacity{0UZ};
    T*                         _data{nullptr};

    [[nodiscard]] constexpr T*               data() noexcept { return _data; }
    [[nodiscard]] constexpr const T*         data() const noexcept { return _data; }
    [[nodiscard]] constexpr std::size_t      size() const noexcept { return _size; }
    [[nodiscard]] constexpr bool             empty() const noexcept { return _size == 0UZ; }
    [[nodiscard]] constexpr std::size_t      capacity() const noexcept { return _capacity; }
    [[nodiscard]] std::pmr::memory_resource* resource() const noexcept { return _resource; }

    void reserve(std::size_t /*new_cap*/) noexcept { /* non-managed: no-op */ }
    void shrink_to_fit() noexcept { /* non-managed: no-op */ }

    [[nodiscard]] constexpr T&       operator[](std::size_t i) noexcept { return _data[i]; }
    [[nodiscard]] constexpr const T& operator[](std::size_t i) const noexcept { return _data[i]; }

    [[nodiscard]] constexpr T& at(std::size_t i) {
        if (i >= _size) [[unlikely]] {
            throw std::out_of_range("gr::pmr::vector::at");
        }
        return _data[i];
    }
    [[nodiscard]] constexpr const T& at(std::size_t i) const {
        if (i >= _size) [[unlikely]] {
            throw std::out_of_range("gr::pmr::vector::at");
        }
        return _data[i];
    }

    [[nodiscard]] constexpr T*       begin() noexcept { return _data; }
    [[nodiscard]] constexpr T*       end() noexcept { return _data + _size; }
    [[nodiscard]] constexpr const T* begin() const noexcept { return _data; }
    [[nodiscard]] constexpr const T* end() const noexcept { return _data + _size; }
    [[nodiscard]] constexpr const T* cbegin() const noexcept { return _data; }
    [[nodiscard]] constexpr const T* cend() const noexcept { return _data + _size; }

    [[nodiscard]] constexpr std::reverse_iterator<T*>       rbegin() noexcept { return std::reverse_iterator<T*>(end()); }
    [[nodiscard]] constexpr std::reverse_iterator<T*>       rend() noexcept { return std::reverse_iterator<T*>(begin()); }
    [[nodiscard]] constexpr std::reverse_iterator<const T*> rbegin() const noexcept { return std::reverse_iterator<const T*>(end()); }
    [[nodiscard]] constexpr std::reverse_iterator<const T*> rend() const noexcept { return std::reverse_iterator<const T*>(begin()); }
    [[nodiscard]] constexpr std::reverse_iterator<const T*> crbegin() const noexcept { return std::reverse_iterator<const T*>(cend()); }
    [[nodiscard]] constexpr std::reverse_iterator<const T*> crend() const noexcept { return std::reverse_iterator<const T*>(cbegin()); }

    [[nodiscard]] constexpr T&       front() noexcept { return _data[0]; }
    [[nodiscard]] constexpr const T& front() const noexcept { return _data[0]; }
    [[nodiscard]] constexpr T&       back() noexcept { return _data[_size - 1]; }
    [[nodiscard]] constexpr const T& back() const noexcept { return _data[_size - 1]; }

    void attach(T* p, std::size_t n, std::size_t cap = std::numeric_limits<std::size_t>::max(), std::pmr::memory_resource* mr = std::pmr::get_default_resource()) noexcept {
        _data     = p;
        _size     = n;
        _capacity = (cap == std::numeric_limits<std::size_t>::max()) ? n : cap;
        _resource = mr;
    }

    [[nodiscard("Ignoring release() will leak memory")]] constexpr T* release() noexcept {
        T* p      = _data;
        _data     = nullptr;
        _size     = 0UZ;
        _capacity = 0UZ;
        return p;
    }

    template<class... Args>
    T& emplace_back(Args&&... args) {
        if (_size >= _capacity) [[unlikely]] {
            throw std::bad_alloc{};
        }
        T* where = _data + _size;
        std::construct_at(where, std::forward<Args>(args)...);
        ++_size;
        return *where;
    }

    void push_back(const T& v) { std::ignore = emplace_back(v); }
    void push_back(T&& v) { std::ignore = emplace_back(std::move(v)); }

    void pop_back() noexcept {
        if (_size == 0UZ) {
            return;
        }
        if constexpr (!std::is_trivially_destructible_v<T>) {
            std::destroy_at(_data + (_size - 1UZ));
        }
        --_size;
    }

    [[nodiscard]] constexpr bool operator==(const vector& other) const { return _size == other._size && std::equal(begin(), end(), other.begin()); }
    [[nodiscard]] constexpr auto operator<=>(const vector& other) const { return std::lexicographical_compare_three_way(begin(), end(), other.begin(), other.end()); }
};

namespace detail {
template<class T, class It>
inline constexpr bool iter_yields_nonconst_T_ref = std::same_as<std::remove_cvref_t<std::iter_reference_t<It>>, T> && !std::is_const_v<std::remove_reference_t<std::iter_reference_t<It>>>;
}

template<class T>
struct vector<T, true> { // managed vector
    using iterator       = T*;
    using const_iterator = const T*;
    std::pmr::memory_resource* _resource{std::pmr::get_default_resource()};
    std::size_t                _size{0UZ};
    std::size_t                _capacity{0UZ};
    T*                         _data{nullptr};

    vector() = default;

    explicit constexpr vector(std::pmr::memory_resource* mr) noexcept : _resource{mr} {}

    vector(std::size_t n, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) : _resource{mr}, _size{n}, _capacity{n} {
        if (n > 0) {
            _data = static_cast<T*>(_resource->allocate(n * sizeof(T), alignof(T)));
            std::uninitialized_value_construct_n(_data, n);
        }
    }

    constexpr vector(std::size_t n, const T& v, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) : _resource{mr}, _size{n}, _capacity{n} {
        if (n > 0) {
            _data = static_cast<T*>(_resource->allocate(n * sizeof(T), alignof(T)));
            std::uninitialized_fill_n(_data, n, v);
        }
    }

    vector(const vector& other)
    requires(std::is_copy_constructible_v<T>)
        : _resource{other._resource} {
        _assign_from_ptr(other._data, other._size);
    }

    vector(vector&& other) noexcept : _resource{other._resource}, _size{other._size}, _capacity{other._capacity}, _data{other._data} {
        other._data     = nullptr;
        other._size     = 0;
        other._capacity = 0;
    }

    vector& operator=(const vector& other)
    requires(std::is_copy_constructible_v<T>)
    {
        if (this != &other) {
            _assign_from_ptr(other._data, other._size);
        }
        return *this;
    }

    constexpr vector& operator=(vector&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        // Destroy and deallocate our resources
        if (_data) {
            if constexpr (!std::is_trivially_destructible_v<T>) {
                std::destroy_n(_data, _size);
            }
            _resource->deallocate(_data, _capacity * sizeof(T), alignof(T));
        }
        // Take ownership of other's resources
        _resource       = other._resource;
        _size           = other._size;
        _capacity       = other._capacity;
        _data           = other._data;
        other._data     = nullptr;
        other._size     = 0UZ;
        other._capacity = 0UZ;
        return *this;
    }

    // Delete copy ops when not copyable
    vector(const vector&)
    requires(!std::is_copy_constructible_v<T>)
    = delete;
    vector& operator=(const vector&)
    requires(!std::is_copy_constructible_v<T>)
    = delete;

    template<std::input_iterator It, std::sentinel_for<It> End>
    vector(It first, End last, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) : _resource{mr} {
        const auto n = static_cast<std::size_t>(std::ranges::distance(first, last));
        if (n > 0) {
            _data     = static_cast<T*>(_resource->allocate(n * sizeof(T), alignof(T)));
            _capacity = n;
            _size     = n;
            try {
                if constexpr (detail::iter_yields_nonconst_T_ref<T, It>) {
                    if constexpr (std::contiguous_iterator<It> && std::is_trivially_copyable_v<T>) {
                        std::memcpy(_data, std::to_address(first), n * sizeof(T));
                    } else {

#ifdef _LIBCPP_VERSION
                        libcxx_fixed_uninitialized_move(first, last, _data); // only use move for non-integral types where it matters
#else
                        std::uninitialized_move(first, last, _data);
#endif
                    }
                } else {
                    std::uninitialized_copy(first, last, _data);
                }
            } catch (...) {
                _resource->deallocate(_data, n * sizeof(T), alignof(T));
                _data     = nullptr;
                _size     = 0;
                _capacity = 0;
                throw;
            }
        }
    }

    template<std::ranges::input_range Range>
    requires(!std::same_as<std::remove_cvref_t<Range>, vector>)
    vector(Range&& r, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) : vector(std::ranges::begin(r), std::ranges::end(r), mr) {}

    vector(std::initializer_list<T> il, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) : vector(il.begin(), il.end(), mr) {}

    vector(std::span<const T> s, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) : vector(s.begin(), s.end(), mr) {}

    ~vector() {
        if (_data) {
            if constexpr (!std::is_trivially_destructible_v<T>) {
                std::destroy_n(_data, _size);
            }
            _resource->deallocate(_data, _capacity * sizeof(T), alignof(T));
        }
    }

    [[nodiscard]] constexpr T*               data() noexcept { return _data; }
    [[nodiscard]] constexpr const T*         data() const noexcept { return _data; }
    [[nodiscard]] constexpr std::size_t      size() const noexcept { return _size; }
    [[nodiscard]] constexpr bool             empty() const noexcept { return _size == 0UZ; }
    [[nodiscard]] constexpr std::size_t      capacity() const noexcept { return _capacity; }
    [[nodiscard]] std::pmr::memory_resource* resource() const noexcept { return _resource; }

    [[nodiscard]] constexpr T*       begin() noexcept { return _data; }
    [[nodiscard]] constexpr T*       end() noexcept { return _data + _size; }
    [[nodiscard]] constexpr const T* begin() const noexcept { return _data; }
    [[nodiscard]] constexpr const T* end() const noexcept { return _data + _size; }
    [[nodiscard]] constexpr const T* cbegin() const noexcept { return _data; }
    [[nodiscard]] constexpr const T* cend() const noexcept { return _data + _size; }

    [[nodiscard]] constexpr std::reverse_iterator<T*>       rbegin() noexcept { return std::reverse_iterator<T*>(end()); }
    [[nodiscard]] constexpr std::reverse_iterator<T*>       rend() noexcept { return std::reverse_iterator<T*>(begin()); }
    [[nodiscard]] constexpr std::reverse_iterator<const T*> rbegin() const noexcept { return std::reverse_iterator<const T*>(end()); }
    [[nodiscard]] constexpr std::reverse_iterator<const T*> rend() const noexcept { return std::reverse_iterator<const T*>(begin()); }
    [[nodiscard]] constexpr std::reverse_iterator<const T*> crbegin() const noexcept { return std::reverse_iterator<const T*>(cend()); }
    [[nodiscard]] constexpr std::reverse_iterator<const T*> crend() const noexcept { return std::reverse_iterator<const T*>(cbegin()); }

    [[nodiscard]] constexpr T& operator[](std::size_t i) noexcept {
        assert(i < _size);
        return _data[i];
    }
    [[nodiscard]] constexpr const T& operator[](std::size_t i) const noexcept {
        assert(i < _size);
        return _data[i];
    }

    [[nodiscard]] constexpr T& at(std::size_t i) {
        if (i >= _size) [[unlikely]] {
            throw std::out_of_range("gr::pmr::vector::at");
        }
        return _data[i];
    }
    [[nodiscard]] constexpr const T& at(std::size_t i) const {
        if (i >= _size) [[unlikely]] {
            throw std::out_of_range("gr::pmr::vector::at");
        }
        return _data[i];
    }

    [[nodiscard]] constexpr T& front() noexcept {
        assert(_size > 0);
        return _data[0];
    }
    [[nodiscard]] constexpr const T& front() const noexcept {
        assert(_size > 0);
        return _data[0];
    }
    [[nodiscard]] constexpr T& back() noexcept {
        assert(_size > 0);
        return _data[_size - 1];
    }
    [[nodiscard]] constexpr const T& back() const noexcept {
        assert(_size > 0);
        return _data[_size - 1];
    }

    void reserve(std::size_t new_cap) {
        if (new_cap <= _capacity) {
            return;
        }
        _grow_capacity(new_cap);
    }

    void shrink_to_fit() {
        if (_size == _capacity) {
            return;
        }
        if (_size == 0UZ) {
            if (_data) {
                _resource->deallocate(_data, _capacity * sizeof(T), alignof(T));
                _data = nullptr;
            }
            _capacity = 0;
            return;
        }
        _grow_capacity(_size);
    }

    template<class... Args>
    T& emplace_back(Args&&... args) {
        if (_size == _capacity) {
            reserve(_next_cap(_capacity, _size + 1UZ));
        }
        T* where = _data + _size;
        std::construct_at(where, std::forward<Args>(args)...);
        ++_size;
        return *where;
    }
    void push_back(const T& v) { (void)emplace_back(v); }
    void push_back(T&& v) { (void)emplace_back(std::move(v)); }
    void pop_back() noexcept {
        if (_size == 0UZ) {
            return;
        }
        if constexpr (!std::is_trivially_destructible_v<T>) {
            std::destroy_at(_data + (_size - 1UZ));
        }
        --_size;
    }

    constexpr void resize(std::size_t new_size) {
        if (new_size <= _size) { // shrink in place - no reallocation!
            if constexpr (!std::is_trivially_destructible_v<T>) {
                std::destroy_n(_data + new_size, _size - new_size);
            }
            _size = new_size;
        } else if (new_size <= _capacity && _data != nullptr) { // grow within capacity - no reallocation!
            std::uninitialized_value_construct_n(_data + _size, new_size - _size);
            _size = new_size;
        } else { // must reallocate
            _grow_capacity(_next_cap(_capacity, new_size));
            std::uninitialized_value_construct_n(_data + _size, new_size - _size);
            _size = new_size;
        }
    }

    constexpr void resize(std::size_t new_size, const T& value) {
        if (new_size <= _size) { // shrink in place
            if constexpr (!std::is_trivially_destructible_v<T>) {
                std::destroy_n(_data + new_size, _size - new_size);
            }
            _size = new_size;
        } else if (new_size <= _capacity && _data != nullptr) { // grow within capacity
            std::uninitialized_fill_n(_data + _size, new_size - _size, value);
            _size = new_size;
        } else { // must reallocate
            _grow_capacity(_next_cap(_capacity, new_size));
            std::uninitialized_fill_n(_data + _size, new_size - _size, value);
            _size = new_size;
        }
    }

    void assign(std::size_t count, const T& value) {
        if (count <= _capacity && _data != nullptr) { // reuse existing capacity
            if constexpr (!std::is_trivially_destructible_v<T>) {
                std::destroy_n(_data, _size);
            }
            std::uninitialized_fill_n(_data, count, value);
            _size = count;
        } else { // need to reallocate
            if (_data) {
                if constexpr (!std::is_trivially_destructible_v<T>) {
                    std::destroy_n(_data, _size);
                }
                _resource->deallocate(_data, _capacity * sizeof(T), alignof(T));
            }
            _data     = static_cast<T*>(_resource->allocate(count * sizeof(T), alignof(T)));
            _capacity = count;
            std::uninitialized_fill_n(_data, count, value);
            _size = count;
        }
    }

    template<class It, class End>
    requires std::input_iterator<It> && std::sentinel_for<End, It>
    void assign(It first, End last) {
        // For contiguous iterators of T, delegate to optimized path
        if constexpr (std::contiguous_iterator<It> && std::same_as<std::iter_value_t<It>, T>) {
            const auto count = static_cast<std::size_t>(std::ranges::distance(first, last));
            _assign_from_ptr(std::to_address(first), count);
        } else { // generic iterator path
            const auto new_size = static_cast<std::size_t>(std::ranges::distance(first, last));
            if (new_size <= _capacity && _data != nullptr) {
                if constexpr (!std::is_trivially_destructible_v<T>) {
                    std::destroy_n(_data, _size);
                }
                std::uninitialized_copy(first, last, _data);
                _size = new_size;
            } else {
                if (_data) {
                    if constexpr (!std::is_trivially_destructible_v<T>) {
                        std::destroy_n(_data, _size);
                    }
                    _resource->deallocate(_data, _capacity * sizeof(T), alignof(T));
                }
                _data     = static_cast<T*>(_resource->allocate(new_size * sizeof(T), alignof(T)));
                _capacity = new_size;
                try {
                    std::uninitialized_copy(first, last, _data);
                    _size = new_size;
                } catch (...) {
                    _resource->deallocate(_data, new_size * sizeof(T), alignof(T));
                    _data     = nullptr;
                    _capacity = 0;
                    throw;
                }
            }
        }
    }

    void assign(std::initializer_list<T> il) { _assign_from_ptr(il.begin(), il.size()); }
    void assign(std::span<const T> s) { _assign_from_ptr(s.data(), s.size()); }
    void assign(const std::pmr::vector<T>& src) { _assign_from_ptr(src.data(), src.size()); }

    void rebind(std::pmr::memory_resource* new_resource) {
        _data     = gr::allocator::pmr::migrate<T>(*new_resource, *_resource, _data, _capacity);
        _resource = new_resource;
        _capacity = _size;
    }

    constexpr void clear() noexcept {
        if constexpr (!std::is_trivially_destructible_v<T>) {
            std::destroy_n(_data, _size);
        }
        _size = 0;
    }

    constexpr void swap(vector& other) noexcept {
        using std::swap;
        swap(_data, other._data);
        swap(_size, other._size);
        swap(_capacity, other._capacity);
        swap(_resource, other._resource);
    }

    [[nodiscard]] constexpr bool operator==(const vector& other) const { return _size == other._size && std::equal(begin(), end(), other.begin()); }
    [[nodiscard]] constexpr auto operator<=>(const vector& other) const { return std::lexicographical_compare_three_way(begin(), end(), other.begin(), other.end()); }

private:
    static constexpr std::size_t _next_cap(std::size_t cur, std::size_t min_needed) noexcept {
        std::size_t cand = cur ? (cur + (cur >> 1U) + 1U) : 2U;
        return cand < min_needed ? min_needed : cand;
    }

    void _assign_from_ptr(const T* src, std::size_t count) {
        if (count <= _capacity && _data != nullptr) { // reuse existing capacity
            if constexpr (!std::is_trivially_destructible_v<T>) {
                std::destroy_n(_data, _size);
            }
            if (count > 0) {
                if constexpr (std::is_trivially_copyable_v<T>) {
                    std::memcpy(_data, src, count * sizeof(T));
                } else {
                    std::uninitialized_copy_n(src, count, _data);
                }
            }
            _size = count;
        } else { // need to reallocate
            T* new_data = static_cast<T*>(_resource->allocate(count * sizeof(T), alignof(T)));
            try {
                if constexpr (std::is_trivially_copyable_v<T>) {
                    std::memcpy(new_data, src, count * sizeof(T));
                } else {
                    std::uninitialized_copy_n(src, count, new_data);
                }
            } catch (...) {
                _resource->deallocate(new_data, count * sizeof(T), alignof(T));
                throw;
            }
            // success - clean up old data
            if (_data) {
                if constexpr (!std::is_trivially_destructible_v<T>) {
                    std::destroy_n(_data, _size);
                }
                _resource->deallocate(_data, _capacity * sizeof(T), alignof(T));
            }
            _data     = new_data;
            _capacity = count;
            _size     = count;
        }
    }

    void _grow_capacity(std::size_t new_cap) {
        T* q = static_cast<T*>(_resource->allocate(new_cap * sizeof(T), alignof(T)));

        std::size_t constructed = 0UZ;
        try {
            if (_size > 0) {
                if constexpr (std::is_trivially_copyable_v<T>) {
                    std::memcpy(q, _data, _size * sizeof(T));
                    constructed = _size;
                } else {
#if !defined(__clang__) // issue with clang's libc++ std::uninitialized_move as of clang20
                    std::uninitialized_move_n(_data, _size, q);
#else
                    if constexpr (detail::iter_yields_nonconst_T_ref<T, decltype(_data)> && !std::is_integral_v<T>) {
                        std::uninitialized_move_n(_data, _size, q); // only use move for non-integral types where it matters
                    } else {
                        std::uninitialized_copy_n(_data, _size, q);
                    }
#endif
                    constructed = _size;
                }
            }
        } catch (...) {
            if constexpr (!std::is_trivially_destructible_v<T>) {
                std::destroy_n(q, constructed);
            }
            _resource->deallocate(q, new_cap * sizeof(T), alignof(T));
            throw;
        }
        if (_data) {
            if constexpr (!std::is_trivially_destructible_v<T>) {
                std::destroy_n(_data, _size);
            }
            _resource->deallocate(_data, _capacity * sizeof(T), alignof(T));
        }
        _data     = q;
        _capacity = new_cap;
    }
};

template<class T, bool managed>
constexpr void swap(vector<T, managed>& a, vector<T, managed>& b) noexcept {
    a.swap(b);
}

template<typename T>
vector(std::initializer_list<T>) -> vector<T, true>;

template<typename T>
vector(std::span<T>) -> vector<T, true>;

} // namespace gr::pmr

#endif // GNURADIO_PMRCOLLECTIONS_HPP
