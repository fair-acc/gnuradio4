#ifndef PFFFT_H
#define PFFFT_H

#include <bit>
#include <cstddef>
#include <new>
#include <source_location>

#include "gnuradio-4.0/MemoryAllocators.hpp"
#include "gnuradio-4.0/Message.hpp"

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"          // error in vir/simd
#pragma GCC diagnostic ignored "-Wsign-conversion" // error in vir/simd
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#ifndef __cpp_aligned_new
#error
#endif

#include <vir/simd.h>

#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"          // error in vir/simd
#pragma GCC diagnostic ignored "-Wsign-conversion" // error in vir/simd
#endif

#include <vir/simd_execution.h>

#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace stdx = vir::stdx;

template<std::floating_point T, int N = 4> // inspired by future C++26 definition
using vec = stdx::simd<T, stdx::simd_abi::deduce_t<T, static_cast<std::size_t>(N)>>;

template<std::size_t Align, typename T>
[[nodiscard]] constexpr bool isAligned(const T* p) noexcept {
    return std::bit_cast<std::uintptr_t>(std::to_address(p)) % Align == 0UZ;
}

template<typename T>
[[nodiscard]] constexpr bool isAligned(const T* p, std::size_t alignment) noexcept {
    return std::bit_cast<std::uintptr_t>(std::to_address(p)) % alignment == 0UZ;
}

enum class Order {
    Ordered,  /// R2C: [DC, Nyquist, Re(1), Im(1), Re(2), Im(2), …, Re(N/2-1), Im(N/2-1)] (Nyquist at index 1)
              ///      ONLY for N % 32 == 0 (power-of-2 aligned)
              /// C2C: [X[0]=DC, … X[N/2]=Nyquist, X[N/2+1]=-Nyquist, …, X[N-1]=left of DC] (natural DFT order)
              ///      for any N factoring into {2, 3, 4, 5}
    Unordered /// R2C: SIMD-tiled with mixed-radix {2,3,4,5} bit-reversal permutation
              ///      - for any N factoring into {2, 3, 4, 5}
              ///      - organized in 8*L blocks: [Re(k)…Re(k+3), Im(k)…Im(k+3)] (L=4)
              ///      - bins NOT sequential, follow radix order (e.g. N=48: [0-3], [6,11,10,9], [12-15], [18,23,22,21])
              ///      - DC at pos 0, Nyquist at pos L
              /// C2C: [Re0,Re4,Re8,Re12, Im0,Im4,Im8,Im12, Re1,Re5,Re9,Re13, Im1,Im5,Im9,Im13,…] (L=4)
              ///      gor any N factoring into {2, 3, 4, 5}
              /// ~20% faster; REQUIRED for non-power-of-2 R2C; preferred for convolution
};

enum class Direction { Forward, Backward }; /// direction of the FFT transform (R2C vs. C2R). N.B. For forward-backward identity bin values needs to be normalised by 'N'.

/* type of transform */
enum class Transform { Real, Complex };

template<class R, class T>
concept InBuf = std::ranges::borrowed_range<R> && std::ranges::contiguous_range<R> && std::convertible_to<decltype(std::ranges::data(std::declval<R&>())), const T*>;
template<class R, class T>
concept OutBuf = std::ranges::borrowed_range<R> && std::ranges::contiguous_range<R> && std::same_as<decltype(std::ranges::data(std::declval<R&>())), T*>;

struct forward_t {
    static constexpr Direction value = Direction::Forward;
};
struct backward_t {
    static constexpr Direction value = Direction::Backward;
};
struct ordered_t {
    static constexpr Order value = Order::Ordered;
};
struct unordered_t {
    static constexpr Order value = Order::Unordered;
};

inline constexpr forward_t   forward{};
inline constexpr backward_t  backward{};
inline constexpr ordered_t   ordered{};
inline constexpr unordered_t unordered{};

template<std::floating_point T>
void rffti1_ps(std::size_t n, std::span<T> wa, std::span<std::size_t, 15> radixPlan);
template<std::floating_point T>
void cffti1_ps(std::size_t n, std::span<T> wa, std::span<std::size_t, 15> radixPlan);

template<std::floating_point T, Transform fftTransform, std::size_t N = std::dynamic_extent>
struct SimdFFT {
    using value_type   = T;
    using vector_type  = vec<T, 4>;
    using V            = vector_type;
    using IsRealValued = std::conditional_t<fftTransform == Transform::Real, std::true_type, std::false_type>;
    using IsDynamic    = std::conditional_t<N == std::dynamic_extent, std::true_type, std::false_type>;

    static constexpr std::size_t kAlignment = 64UZ;
    static constexpr std::size_t ceil_div(std::size_t x, std::size_t y) { return (x + y - 1) / y; }
    static constexpr std::size_t kTwiddleCount = 2 * (V::size() - 1) * V::size() * ceil_div((IsRealValued::value ? (N / 2) : N) / V::size(), V::size());

    using WorkStorage             = std::conditional_t<IsDynamic::value, std::vector<T, gr::allocator::Aligned<T, kAlignment>>, std::array<T, IsRealValued::value ? N : 2 * N>>;
    using StageTwiddleStorage     = std::conditional_t<IsDynamic::value, std::vector<T, gr::allocator::Aligned<T, kAlignment>>, std::array<T, kTwiddleCount>>;
    using ButterflyTwiddleStorage = std::conditional_t<IsDynamic::value, std::vector<T, gr::allocator::Aligned<T, kAlignment>>, std::array<T, 2 * (V::size() - 1) * ((IsRealValued::value ? (N / 2) : N) / V::size())>>; // 2*(L-1)*SV

    std::size_t                   _N = N;
    std::array<std::size_t, 15UZ> _radixPlan{}; // [0]: unused (FFTPACK legacy), [1]: number of stages nf, [2 .. 1 + nf] stage radices in execution ordering (2,3,4,5,...)

    alignas(64) WorkStorage _scratch{};                       // used as local data storage
    alignas(64) StageTwiddleStorage _stageTwiddles{};         // stage-level twiddles
    alignas(64) ButterflyTwiddleStorage _butterflyTwiddles{}; // butterfly simd-level twiddles

    constexpr SimdFFT()
    requires(!IsDynamic::value)
    {
        static_assert(!canProcessSize(size()), "cannot process this size: min>=16C2C (32: R2C) & radix-2, -3, -5 & 'x min' compatible");
        computeTwiddle();
    }

    explicit SimdFFT(std::size_t n, std::source_location loc = std::source_location::current())
    requires(IsDynamic::value)
        : _N(n) {
        if (!canProcessSize(size())) {
            throw gr::exception(std::format("incompatible sizes for {}2C: N ({}) must be multiple of 2,3,4,5 and >{}", fftTransform == Transform::Real ? "R" : "C", size(), minSize()), loc);
        }
        computeTwiddle();
    }

    [[nodiscard]] constexpr std::size_t size() const noexcept {
        if constexpr (IsDynamic::value) {
            return _N;
        } else {
            return N;
        }
    }
    [[nodiscard]] static constexpr std::size_t simdSize() noexcept { return vector_type::size(); }
    [[nodiscard]] static constexpr std::size_t minSize() {
        constexpr std::size_t L = simdSize();
        if constexpr (fftTransform == Transform::Real) {
            return 2UZ * L * L; // min size is N = 32 (SIMD-limit)
        } else {                // transform == Transform::Complex
            return L * L;       // min size is N = 16
        }
    }
    [[nodiscard]] static constexpr bool canProcessSize(std::size_t n, Order ordering = Order::Unordered) {
        if (n < minSize()) {
            return false;
        }

        constexpr std::size_t L     = simdSize();
        constexpr std::size_t N_min = minSize(); // 16 for complex, 32 for real

        if (ordering == Order::Ordered && fftTransform == Transform::Real) {
            if (n % (2 * L * L) != 0) {
                return false;
            }
        }

        // validation: must reduce exactly to N_min
        std::size_t R = n;
        while (R >= 5 * N_min && (R % 5) == 0) {
            R /= 5;
        }
        while (R >= 3 * N_min && (R % 3) == 0) {
            R /= 3;
        }
        while (R >= 2 * N_min && (R % 2) == 0) {
            R /= 2;
        }

        return (R == N_min); // Must reduce exactly to minimum size!
    }
    [[nodiscard]] constexpr std::size_t simdVectorSize() const noexcept {
        return (IsRealValued::value ? (size() / 2) : size()) / simdSize(); // simdVectorSize = number of complex SIMD vectors (N/4 if complex, N/8 if real for lanes=4)
    }

    [[nodiscard]] std::span<const T> butterflyTwiddles() const noexcept { return _butterflyTwiddles; }
    [[nodiscard]] std::span<const T> stageTwiddles() const noexcept { return _stageTwiddles; }
    [[nodiscard]] std::span<T>       scratch() noexcept { return _scratch; }
    [[nodiscard]] std::span<const T> scratch() const noexcept { return _scratch; }

    void computeTwiddle() {
        // not too performance critical, computed usually only once
        constexpr std::size_t L       = simdSize();
        const std::size_t     nScalar = ceil_div(size(), L);
        if constexpr (IsDynamic::value) {
            constexpr std::size_t kGuard = 2UZ;
            _stageTwiddles.resize(2UZ * nScalar + kGuard);
            _butterflyTwiddles.resize(2UZ * (L - 1UZ) * L * ceil_div(simdVectorSize(), L));
            _scratch.resize(std::max(2UZ * simdVectorSize() * L, 2UZ * size())); // see if this max size can be optimised
        }

        // compute stage twiddles & radix plan
        if constexpr (IsRealValued::value) {
            rffti1_ps<T>(size() / L, _stageTwiddles, _radixPlan);
        } else {
            cffti1_ps<T>(size() / L, _stageTwiddles, _radixPlan);
        }

        // butterfly “rotation” scalars in SoA layout
        const std::size_t SV   = simdVectorSize();
        const T           base = (-T{2} * std::numbers::pi_v<T>) / static_cast<T>(size());
        for (std::size_t k = 0UZ; k < SV; ++k) {
            const T           kf = base * static_cast<T>(k);
            const std::size_t i  = k / L;
            const std::size_t j  = k % L;
            for (std::size_t m = 0; m < L - 1; ++m) {
                const T A                                               = kf * static_cast<T>(m + 1);
                _butterflyTwiddles[(2 * (i * (L - 1) + m) + 0) * L + j] = std::cos(A);
                _butterflyTwiddles[(2 * (i * (L - 1) + m) + 1) * L + j] = std::sin(A);
            }
        }

        // factorization check: product(radices) == size()/L
        std::size_t prod = 1;
        for (std::size_t k = 0; k < _radixPlan[1]; ++k) {
            prod *= _radixPlan[2 + k];
        }
        if (prod != nScalar) {
            throw gr::exception(std::format("{} transform (N={}) factorization mismatch: prod(radices={})={} != size()/SIMD_width={} (N={}, SIMD_width={})", //
                fftTransform, size(), _radixPlan, prod, nScalar, size(), vector_type::size()));
        }
    }

    /**
     * Perform a Fourier transform , The z-domain data is stored in the
   most efficient order for transforming it back, or using it for
   convolution. If you need to have its content sorted in the
   "usual" way, that is as an array of interleaved complex numbers,
   either use pffft_transform_ordered , or call pffft_zreordering after
   the forward fft, and before the backward fft.

   Transforms are not scaled: PFFFT_BACKWARD(PFFFT_FORWARD(x)) = N*x.
   Typically you will want to scale the backward transform by 1/N.

   The 'work' pointer should point to an area of N (2*N for complex
   fft) floats, properly aligned. If 'work' is NULL, then stack will
   be used instead (this is probably the best strategy for small
   FFTs, say for N < 16384). Threads usually have a small stack, that
   there's no sufficient amount of memory, usually leading to a crash!
   Use the heap with pffft_aligned_malloc() in this case.

   For a real forward transform (PFFFT_REAL | PFFFT_FORWARD) with real
   input with input(=transformation) length N, the output array is
   'mostly' complex:
     index k in 1 .. N/2 -1  corresponds to frequency k * Samplerate / N
     index k == 0 is a special case:
       the real() part contains the result for the DC frequency 0,
       the imag() part contains the result for the Nyquist frequency Samplerate/2
   both 0-frequency and half frequency components, which are real,
   are assembled in the first entry as  F(0)+i*F(N/2).
   With the output size N/2 complex values (=N real/imag values), it is
   obvious, that the result for negative frequencies are not output,
   cause of symmetry.

   @param order
    * fft_order_t::Unordered -- better performance (notably for back-and-forth transforms
    * fft_order_t::Ordered makes sure that the output is
   ordered as expected (interleaved complex numbers).  This is
   similar to calling pffft_transform and then pffft_zreordering.

   input and output may alias.
*/
    void transform(forward_t, ordered_t, InBuf<T> auto&& in, OutBuf<T> auto&& out, std::source_location loc = std::source_location::current()) { transform<Direction::Forward, Order::Ordered>(std::forward<decltype(in)>(in), std::forward<decltype(out)>(out), loc); }
    void transform(backward_t, ordered_t, InBuf<T> auto&& in, OutBuf<T> auto&& out, std::source_location loc = std::source_location::current()) { transform<Direction::Backward, Order::Ordered>(std::forward<decltype(in)>(in), std::forward<decltype(out)>(out), loc); }
    void transform(forward_t, unordered_t, InBuf<T> auto&& in, OutBuf<T> auto&& out, std::source_location loc = std::source_location::current()) { transform<Direction::Forward, Order::Unordered>(std::forward<decltype(in)>(in), std::forward<decltype(out)>(out), loc); }
    void transform(backward_t, unordered_t, InBuf<T> auto&& in, OutBuf<T> auto&& out, std::source_location loc = std::source_location::current()) { transform<Direction::Backward, Order::Unordered>(std::forward<decltype(in)>(in), std::forward<decltype(out)>(out), loc); }

    template<Direction direction, Order ordering, InBuf<T> Rin, OutBuf<T> Rout>
    void transform(Rin&& in, Rout&& out, std::source_location loc = std::source_location::current()) {
        const auto inputSpan  = std::span<const T>(std::ranges::data(in), std::ranges::size(in));
        auto       outputSpan = std::span<T>(std::ranges::data(out), std::ranges::size(out));

        const std::size_t need = (fftTransform == Transform::Real) ? size() : 2 * size();
        if (inputSpan.size() < need || outputSpan.size() < need) {
            throw gr::exception(std::format("size mismatch: input({}) output({}) setup({})", inputSpan.size(), outputSpan.size(), need), loc);
        }
        if (!SimdFFT<T, fftTransform, N>::canProcessSize(size(), ordering)) {
            if constexpr (ordering == Order::Ordered && fftTransform == Transform::Real) {
                throw gr::exception(std::format("{} {} FFT not supported for N={} (requires N % 32 == 0)", ordering, fftTransform, size()), loc);
            } else {
                throw gr::exception(std::format("{} {} FFT with N={} not supported (must factor into {{2,3,4,5}} and >={})", ordering, fftTransform, size(), minSize()), loc);
            }
        }

        if (!isAligned<kAlignment>(inputSpan.data())) {
            throw gr::exception(std::format("input is not {}-bytes aligned", kAlignment), loc);
        }
        if (!isAligned<kAlignment>(outputSpan.data())) {
            throw gr::exception(std::format("output is not {}-bytes aligned", kAlignment), loc);
        }

        transformInternal<direction, ordering>(*this, inputSpan, outputSpan, scratch());
    }

private:
    // if any
};

/*
   call pffft_zreordering(.., PFFFT_FORWARD) after pffft_transform(...,
   PFFFT_FORWARD) if you want to have the frequency components in
   the correct "canonical" order, as interleaved complex numbers.

   (for real transforms, both 0-frequency and half frequency
   components, which are real, are assembled in the first entry as
   F(0)+i*F(n/2+1). Note that the original fftpack did place
   F(n/2+1) at the end of the arrays).

   input and output should not alias.
*/
template<Direction direction, std::floating_point T, Transform transform, std::size_t N>
constexpr void simdReordering(SimdFFT<T, transform, N>& setup, std::span<const T> input, std::span<T> output);

// TODO: functions below need to be refactored w.r.t. C++API and performance (if possible).

/*
   Perform a multiplication of the frequency components of dft_a and
   dft_b and accumulate them into dft_ab. The arrays should have
   been obtained with pffft_transform(.., PFFFT_FORWARD) and should
   *not* have been reordered with pffft_zreordering (otherwise just
   perform the operation yourself as the dft coefs are stored as
   interleaved complex numbers).

   the operation performed is: dft_ab += (dft_a * fdt_b)*scaling

   The dft_a, dft_b and dft_ab pointers may alias.
*/
template<std::floating_point T, Transform transform, std::size_t N_>
constexpr void zconvolve_accumulate(SimdFFT<T, transform, N_>& s, const T* a, const T* b, T* ab, T scaling) {
    std::size_t Ncvec = s.simdVectorSize();

    const T ar  = a[0];
    const T ai  = a[4];
    const T br  = b[0];
    const T bi  = b[4];
    const T abr = ab[0];
    const T abi = ab[4];

    /* default routine, works fine for non-arm cpus with current compilers */
    const vec<T>       vscal = scaling;
    std::span<const T> sa(a, Ncvec * 8);
    std::span<const T> sb(b, Ncvec * 8);
    std::span<T>       sab(ab, Ncvec * 8);
    vir::transform(vir::execution::simd.prefer_size<8UZ>().unroll_by<2UZ>(), std::views::zip(sa, sb, sab), sab, [=](const auto& tup) {
        const auto& [va, vb, vab] = tup;
        if constexpr (va.size() == 8UZ) {
            auto [ar_, ai_]   = split<4, 4>(va);
            auto [br_, bi_]   = split<4, 4>(vb);
            auto [abr_, abi_] = split<4, 4>(vab);
            complex_multiply(ar_, ai_, br_, bi_);
            return concat((ar * vscal + abr_), (ai * vscal + abi_));
        } else {
            __builtin_trap(); // this should be impossible
            return vab;       // to get the expected return type
        }
    });

    if constexpr (SimdFFT<T, transform, N_>::IsRealValued::value) {
        ab[0] = abr + ar * br * scaling;
        ab[4] = abi + ai * bi * scaling;
    }
}

/*
   Perform a multiplication of the frequency components of dft_a and
   dft_b and put result in dft_ab. The arrays should have
   been obtained with pffft_transform(.., PFFFT_FORWARD) and should
   *not* have been reordered with pffft_zreordering (otherwise just
   perform the operation yourself as the dft coefs are stored as
   interleaved complex numbers).

   the operation performed is: dft_ab = (dft_a * fdt_b)*scaling

   The dft_a, dft_b and dft_ab pointers may alias.
*/
template<std::floating_point T, Transform transform, std::size_t N_>
void pffft_zconvolve_no_accu(SimdFFT<T, transform, N_>& s, const T* a, const T* b, T* ab, T scaling) {
    const vec<T>      vscal       = scaling;
    const std::size_t NcvecMulTwo = 2 * s.simdVectorSize(); /* std::size_t Ncvec = s.simdVectorSize(); */

    const T sar = a[0];
    const T sai = a[vec<T>::size()];
    const T sbr = b[0];
    const T sbi = b[vec<T>::size()];

    /* default routine, works fine for non-arm cpus with current compilers */
    for (std::size_t k = 0; k < NcvecMulTwo; k += 4) {
        vec<T> var(a + (k + 0) * vec<T>::size(), stdx::vector_aligned);
        vec<T> vai(a + (k + 1) * vec<T>::size(), stdx::vector_aligned);
        vec<T> vbr(b + (k + 0) * vec<T>::size(), stdx::vector_aligned);
        vec<T> vbi(b + (k + 1) * vec<T>::size(), stdx::vector_aligned);
        complex_multiply(var, vai, vbr, vbi);
        store_unchecked(var * vscal, ab + (k + 0) * vec<T>::size(), stdx::vector_aligned);
        store_unchecked(vai * vscal, ab + (k + 1) * vec<T>::size(), stdx::vector_aligned);
        var(a + (k + 2) * vec<T>::size(), stdx::vector_aligned);
        vai(a + (k + 3) * vec<T>::size(), stdx::vector_aligned);
        vbr(b + (k + 2) * vec<T>::size(), stdx::vector_aligned);
        vbi(b + (k + 3) * vec<T>::size(), stdx::vector_aligned);
        complex_multiply(var, vai, vbr, vbi);
        store_unchecked(var * vscal, ab + (k + 2) * vec<T>::size(), stdx::vector_aligned);
        store_unchecked(vai * vscal, ab + (k + 3) * vec<T>::size(), stdx::vector_aligned);
    }

    if constexpr (SimdFFT<T, transform, N_>::IsRealValued::value) {
        ab[0]              = sar * sbr * scaling;
        ab[vec<T>::size()] = sai * sbi * scaling;
    }
}

#endif /* PFFFT_H */
