#ifndef GNURADIO_PMRCOLLECTIONS_HPP
#define GNURADIO_PMRCOLLECTIONS_HPP

#include <gnuradio-4.0/MemoryAllocators.hpp>

namespace gr::pmr {

template<typename T, bool managed>
struct vector;

template<class T>
struct vector<T, false> { // trivially copyable, no lifetime management
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

    [[nodiscard]] constexpr T*       begin() noexcept { return _data; }
    [[nodiscard]] constexpr T*       end() noexcept { return _data + _size; }
    [[nodiscard]] constexpr const T* begin() const noexcept { return _data; }
    [[nodiscard]] constexpr const T* end() const noexcept { return _data + _size; }
    [[nodiscard]] constexpr const T* cbegin() const noexcept { return _data; }
    [[nodiscard]] constexpr const T* cend() const noexcept { return _data + _size; }

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
        : _resource{other._resource}, _size{other._size}, _capacity{other._size} {
        if (_size > 0) {
            _data = static_cast<T*>(_resource->allocate(_capacity * sizeof(T), alignof(T)));
            if constexpr (std::is_trivially_copyable_v<T>) {
                std::memcpy(_data, other._data, _size * sizeof(T));
            } else {
                std::uninitialized_copy_n(other._data, _size, _data);
            }
        }
    }

    vector(vector&& other) noexcept : _resource{other._resource}, _size{other._size}, _capacity{other._capacity}, _data{other._data} {
        other._data     = nullptr;
        other._size     = 0;
        other._capacity = 0;
    }

    vector& operator=(const vector& o)
    requires(std::is_copy_constructible_v<T>)
    {
        if (this == &o) {
            return *this;
        }
        clear();
        if (o._size > 0) {
            _data     = static_cast<T*>(_resource->allocate(o._size * sizeof(T), alignof(T)));
            _capacity = o._size;
            if constexpr (std::is_trivially_copyable_v<T>) {
                std::memcpy(_data, o._data, o._size * sizeof(T));
            } else {
                std::uninitialized_copy_n(o._data, o._size, _data);
            }
            _size = o._size;
        }
        return *this;
    }

    constexpr vector& operator=(vector&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        clear();
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

#if !defined(__clang__) // issue with clang's libc++ std::uninitialized_move as of clang20
                        std::uninitialized_move(first, last, _data);
#else
                        if constexpr (detail::iter_yields_nonconst_T_ref<T, decltype(first)> && !std::is_integral_v<T>) {
                            std::uninitialized_move(first, last, _data); // only use move for non-integral types where it matters
                        } else {
                            std::uninitialized_copy(first, last, _data);
                        }
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

    vector(std::initializer_list<T> il, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) : vector(il.begin(), il.end(), mr) {}

    vector(std::span<const T> s, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) : vector(s.begin(), s.end(), mr) {}

    // FIXED: Range constructor now properly handles lifetime of temporary ranges
    template<std::ranges::input_range R>
    explicit vector(R&& r, std::pmr::memory_resource* mr = std::pmr::get_default_resource())
    requires(!std::same_as<std::remove_cvref_t<R>, vector>)
        : _resource{mr} {
        const auto n = static_cast<std::size_t>(std::ranges::distance(r));
        if (n > 0) {
            _data     = static_cast<T*>(_resource->allocate(n * sizeof(T), alignof(T)));
            _capacity = n;
            _size     = n;
            try {
                auto first = std::ranges::begin(r);
                auto last  = std::ranges::end(r);

                if constexpr (detail::iter_yields_nonconst_T_ref<T, decltype(first)>) {
                    if constexpr (std::contiguous_iterator<decltype(first)> && std::is_trivially_copyable_v<T>) {
                        std::memcpy(_data, std::to_address(first), n * sizeof(T));
                    } else {
#if !defined(__clang__) // issue with clang's libc++ std::uninitialized_move as of clang20
                        std::uninitialized_move(first, last, _data);
#else
                        if constexpr (detail::iter_yields_nonconst_T_ref<T, decltype(first)> && !std::is_integral_v<T>) {
                            std::uninitialized_move(first, last, _data); // only use move for non-integral types where it matters
                        } else {
                            std::uninitialized_copy(first, last, _data);
                        }
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

    vector& operator=(const std::pmr::vector<T>& other) {
        clear();
        if (other.size() > 0) {
            reserve(other.size());
            std::uninitialized_copy(other.begin(), other.end(), _data);
            _size = other.size();
        }
        return *this;
    }

    vector& operator=(std::pmr::vector<T>&& other) {
        // can't move data since they're different types -> optimised copy
        clear();
        if (other.size() > 0) {
            reserve(other.size());
            if constexpr (std::is_trivially_copyable_v<T>) {
                std::memcpy(_data, other.data(), other.size() * sizeof(T));
            } else {

#if !defined(__clang__) // issue with clang's libc++ std::uninitialized_move as of clang20
                std::uninitialized_move(other.begin(), other.end(), _data);
#else
                if constexpr (detail::iter_yields_nonconst_T_ref<T, decltype(other.begin())> && !std::is_integral_v<T>) {
                    std::uninitialized_move(other.begin(), other.end(), _data); // only use move for non-integral types where it matters
                } else {
                    std::uninitialized_copy(other.begin(), other.end(), _data);
                }
#endif
            }
            _size = other.size();
        }
        other.clear(); // clear the source
        return *this;
    }

    explicit vector(const std::pmr::vector<T>& other, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) : _resource{mr} {
        if (other.size() > 0) {
            _capacity = other.size();
            _size     = other.size();
            _data     = static_cast<T*>(_resource->allocate(_capacity * sizeof(T), alignof(T)));
            if constexpr (std::is_trivially_copyable_v<T>) {
                std::memcpy(_data, other.data(), _size * sizeof(T));
            } else {
                std::uninitialized_copy_n(other.data(), _size, _data);
            }
        }
    }

    explicit vector(std::pmr::vector<T>&& other, std::pmr::memory_resource* mr = std::pmr::get_default_resource()) : _resource{mr} {
        if (other.size() > 0) {
            _capacity = other.size();
            _size     = other.size();
            _data     = static_cast<T*>(_resource->allocate(_capacity * sizeof(T), alignof(T)));
            if constexpr (std::is_trivially_copyable_v<T>) {
                std::memcpy(_data, other.data(), _size * sizeof(T));
            } else {
#if !defined(__clang__) // issue with clang's libc++ std::uninitialized_move as of clang20
                std::uninitialized_move(other.begin(), other.end(), _data);
#else
                if constexpr (detail::iter_yields_nonconst_T_ref<T, decltype(other.begin())> && !std::is_integral_v<T>) {
                    std::uninitialized_move(other.begin(), other.end(), _data); // only use move for non-integral types where it matters
                } else {
                    std::uninitialized_copy(other.begin(), other.end(), _data);
                }
#endif
            }
        }
        other.clear();
    }

    ~vector() { clear(); }

    [[nodiscard]] std::pmr::memory_resource* resource() const noexcept { return _resource; }
    [[nodiscard]] constexpr T*               data() noexcept { return _data; }
    [[nodiscard]] constexpr const T*         data() const noexcept { return _data; }
    [[nodiscard]] constexpr std::size_t      size() const noexcept { return _size; }
    [[nodiscard]] constexpr bool             empty() const noexcept { return _size == 0UZ; }
    [[nodiscard]] constexpr std::size_t      capacity() const noexcept { return _capacity; }

    [[nodiscard]] constexpr T&       operator[](std::size_t i) noexcept { return _data[i]; }
    [[nodiscard]] constexpr const T& operator[](std::size_t i) const noexcept { return _data[i]; }

    [[nodiscard]] constexpr T*       begin() noexcept { return _data; }
    [[nodiscard]] constexpr T*       end() noexcept { return _data + _size; }
    [[nodiscard]] constexpr const T* begin() const noexcept { return _data; }
    [[nodiscard]] constexpr const T* end() const noexcept { return _data + _size; }
    [[nodiscard]] constexpr const T* cbegin() const noexcept { return _data; }
    [[nodiscard]] constexpr const T* cend() const noexcept { return _data + _size; }

    [[nodiscard]] constexpr T& front() noexcept {
        assert(_data != nullptr && _size > 0);
        return _data[0];
    }
    [[nodiscard]] constexpr const T& front() const noexcept {
        assert(_data != nullptr && _size > 0);
        return _data[0];
    }
    [[nodiscard]] constexpr T& back() noexcept {
        assert(_data != nullptr && _size > 0);
        return _data[_size - 1];
    }
    [[nodiscard]] constexpr const T& back() const noexcept {
        assert(_data != nullptr && _size > 0);
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
            clear();
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
            std::destroy_n(_data + new_size, _size - new_size);
            _size = new_size;
        } else if (new_size <= _capacity) { // grow within capacity - no reallocation!
            std::uninitialized_value_construct_n(_data + _size, new_size - _size);
            _size = new_size;
        } else {                            // must reallocate
            _grow_capacity(new_size * 2UZ); // factor 2 growth factor
            std::uninitialized_value_construct_n(_data + _size, new_size - _size);
            _size = new_size;
        }
    }

    void assign(std::size_t count, const T& value) {
        clear();
        if (count > 0) {
            reserve(count);
            std::uninitialized_fill_n(_data, count, value);
            _size = count;
        }
    }

    template<class It, class End>
    requires std::input_iterator<It> && std::sentinel_for<End, It>
    void assign(It first, End last) {
        const auto new_size = static_cast<std::size_t>(std::ranges::distance(first, last));
        clear();
        if (new_size > 0) {
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

    void assign(std::initializer_list<T> il) { assign(il.begin(), il.end()); }
    void assign(std::span<const T> s) { assign(s.begin(), s.end()); }
    void assign(const std::pmr::vector<T>& src) {
        clear();
        if (src.size() > 0) {
            _data     = static_cast<T*>(_resource->allocate(src.size() * sizeof(T), alignof(T)));
            _capacity = src.size();
            if constexpr (std::is_trivially_copyable_v<T>) {
                std::memcpy(_data, src.data(), src.size() * sizeof(T));
            } else {
                try {
                    std::uninitialized_copy_n(src.data(), src.size(), _data);
                } catch (...) {
                    _resource->deallocate(_data, src.size() * sizeof(T), alignof(T));
                    _data     = nullptr;
                    _capacity = 0;
                    throw;
                }
            }
            _size = src.size();
        }
    }

    void rebind(std::pmr::memory_resource* new_resource) {
        _data     = gr::allocator::pmr::migrate<T>(*new_resource, *_resource, _data, _capacity);
        _resource = new_resource;
        _capacity = _size;
    }

    constexpr void clear() {
        if (_data) {
            if constexpr (!std::is_trivially_destructible_v<T>) {
                std::destroy_n(_data, _size);
            }
            void* ptr = const_cast<void*>(static_cast<const void*>(_data));
            _resource->deallocate(ptr, _capacity * sizeof(T), alignof(T));
            _data = nullptr;
        }
        _size     = 0;
        _capacity = 0;
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
            // Cast away const for deallocation
            void* ptr = const_cast<void*>(static_cast<const void*>(_data));
            _resource->deallocate(ptr, _capacity * sizeof(T), alignof(T));
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
