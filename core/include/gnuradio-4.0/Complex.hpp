#ifndef GNURADIO_COMPLEX_HPP
#define GNURADIO_COMPLEX_HPP

#include <cmath>
#include <complex>
#include <cstddef>
#include <tuple>
#include <type_traits>

namespace gr {

/**
 * @brief Minimal trivially-copyable complex type for portable device/SIMD arithmetic.
 *
 * A constexpr {re, im} pair, layout-compatible with std::complex<T> (same size and representation),
 * so the two interconvert at controlled ABI boundaries.
 *
 * The reason for a custom type is device-kernel reliability, not byte transport. Modern std::complex<T>
 * is trivially-copyable and fine for host storage and USM transfer, but its arithmetic is not portably
 * supported inside SYCL device kernels under AdaptiveCpp/hipSYCL-style backends: multiply lowers to the
 * libgcc helper __mulsc3 and std::abs to the libm helper cabsf, both unresolved at the CUDA device JIT
 * (verified on acpp/clang 21 generic-SSCP; -ffast-math inlines __mulsc3 but not cabsf). The AdaptiveCpp
 * maintainer's guidance in issues #340/#341 is that SYCL does not generally support arbitrary std:: code in
 * kernels, and recommends a custom complex type; #341, the request for a sycl::complex, has been open and
 * unimplemented since 2020. (#340 itself is cited for that guidance only -- it reports a silent miscompile
 * and was closed as environment-specific, so it is not a second instance of the failure measured here.)
 * Re-checked 2026-09-01 against AdaptiveCpp develop: the pass that would carry such a mapping,
 * StdBuiltinRemapperPass.cpp, is unchanged since before v25.10.0, lists real-valued libm only, and none of
 * the 76 shipped bitcode libraries define __mulsc3 or cabsf -- upgrading does not fix this. gr::complex uses inline arithmetic and std::sqrt-based magnitude
 * (device builtins) and adds the tuple/structured-binding protocol for vir::simdize plus ADL real/imag/abs/norm/conj.
 *
 * Duplicating a standard library type is absolutely exceptional and is not a precedent for wrapping others.
 * It is justified here only because the defect is in code generation for a type the standard library owns, so
 * no care at the call site can avoid it; because it presents as silent garbage or as a context-poisoning JIT
 * failure that takes down unrelated kernels queued after it, rather than as a diagnosable error; and because
 * the alternative is to have no complex arithmetic on a device at all. Nothing here is about performance.
 *
 * Because it exists, it is kept interchangeable with the standard type rather than parallel to it: the two
 * interconvert implicitly, and the settings and wire layers accept gr::complex<T> wherever they accept
 * std::complex<T>, storing it under the same ComplexFloat32/64 tag. Code that has a choice should prefer
 * std::complex on host-only paths and gr::complex on anything that may reach a kernel.
 *
 * The sizeof/alignof static_asserts below are ABI guards; they do not bless arbitrary reinterpret_cast
 * between the two types — treat that as a deliberately isolated low-level bridge, not a general aliasing model.
 */
template<std::floating_point T>
struct complex {
    using value_type = T;

    T re{};
    T im{};

    constexpr complex() noexcept = default;
    constexpr complex(T r, T i = T{}) noexcept : re(r), im(i) {}

    constexpr complex(const std::complex<T>& c) noexcept : re(c.real()), im(c.imag()) {}
    constexpr operator std::complex<T>() const noexcept { return {re, im}; }

    [[nodiscard]] constexpr T real() const noexcept { return re; }
    [[nodiscard]] constexpr T imag() const noexcept { return im; }

    constexpr complex& operator+=(complex rhs) noexcept {
        re += rhs.re;
        im += rhs.im;
        return *this;
    }
    constexpr complex& operator-=(complex rhs) noexcept {
        re -= rhs.re;
        im -= rhs.im;
        return *this;
    }

    constexpr complex& operator*=(complex rhs) noexcept {
        T r = re * rhs.re - im * rhs.im;
        T i = re * rhs.im + im * rhs.re;
        re  = r;
        im  = i;
        return *this;
    }

    constexpr complex& operator/=(complex rhs) noexcept {
        T d = rhs.re * rhs.re + rhs.im * rhs.im;
        T r = (re * rhs.re + im * rhs.im) / d;
        T i = (im * rhs.re - re * rhs.im) / d;
        re  = r;
        im  = i;
        return *this;
    }

    constexpr complex& operator*=(T s) noexcept {
        re *= s;
        im *= s;
        return *this;
    }
    constexpr complex& operator/=(T s) noexcept {
        re /= s;
        im /= s;
        return *this;
    }

    friend constexpr complex operator+(complex a, complex b) noexcept { return {a.re + b.re, a.im + b.im}; }
    friend constexpr complex operator-(complex a, complex b) noexcept { return {a.re - b.re, a.im - b.im}; }
    friend constexpr complex operator-(complex a) noexcept { return {-a.re, -a.im}; }

    friend constexpr complex operator*(complex a, complex b) noexcept { return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }

    friend constexpr complex operator*(complex a, T s) noexcept { return {a.re * s, a.im * s}; }
    friend constexpr complex operator*(T s, complex a) noexcept { return {s * a.re, s * a.im}; }

    friend constexpr complex operator/(complex a, complex b) noexcept {
        T d = b.re * b.re + b.im * b.im;
        return {(a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d};
    }

    friend constexpr complex operator/(complex a, T s) noexcept { return {a.re / s, a.im / s}; }

    friend constexpr bool operator==(complex a, complex b) noexcept { return a.re == b.re && a.im == b.im; }

    // structured binding support (required for vir::simdize)
    template<std::size_t I>
    [[nodiscard]] constexpr T& get() & noexcept {
        if constexpr (I == 0) {
            return re;
        } else {
            static_assert(I == 1);
            return im;
        }
    }

    template<std::size_t I>
    [[nodiscard]] constexpr const T& get() const& noexcept {
        if constexpr (I == 0) {
            return re;
        } else {
            static_assert(I == 1);
            return im;
        }
    }

    template<std::size_t I>
    [[nodiscard]] constexpr T&& get() && noexcept {
        if constexpr (I == 0) {
            return static_cast<T&&>(re);
        } else {
            static_assert(I == 1);
            return static_cast<T&&>(im);
        }
    }
};

template<std::floating_point T>
[[nodiscard]] constexpr T abs(complex<T> z) noexcept {
    return std::sqrt(z.re * z.re + z.im * z.im);
}

template<std::floating_point T>
[[nodiscard]] constexpr T norm(complex<T> z) noexcept {
    return z.re * z.re + z.im * z.im;
}

template<std::floating_point T>
[[nodiscard]] constexpr T arg(complex<T> z) noexcept {
    return std::atan2(z.im, z.re);
}

template<std::floating_point T>
[[nodiscard]] constexpr complex<T> conj(complex<T> z) noexcept {
    return {z.re, -z.im};
}

template<std::floating_point T>
[[nodiscard]] constexpr complex<T> polar(T r, T theta = T{}) noexcept {
    return {r * std::cos(theta), r * std::sin(theta)};
}

// ADL-compatible free functions for interop with code that calls std::real()/std::imag()
template<std::floating_point T>
[[nodiscard]] constexpr T real(complex<T> z) noexcept {
    return z.re;
}

template<std::floating_point T>
[[nodiscard]] constexpr T imag(complex<T> z) noexcept {
    return z.im;
}

} // namespace gr

// tuple protocol — enables structured bindings and vir::simdize
template<std::floating_point T>
struct std::tuple_size<gr::complex<T>> : std::integral_constant<std::size_t, 2> {};

template<std::size_t I, std::floating_point T>
struct std::tuple_element<I, gr::complex<T>> {
    using type = T;
};

// type traits
static_assert(std::is_trivially_copyable_v<gr::complex<float>>);
static_assert(std::is_trivially_copyable_v<gr::complex<double>>);
static_assert(sizeof(gr::complex<float>) == sizeof(std::complex<float>));
static_assert(sizeof(gr::complex<double>) == sizeof(std::complex<double>));
static_assert(sizeof(gr::complex<float>) == 2 * sizeof(float));

#endif // GNURADIO_COMPLEX_HPP
