#ifndef SIMD_FFT_HPP
#define SIMD_FFT_HPP

#include <bit>
#include <cstddef>
#include <new>
#include <source_location>

#include "gnuradio-4.0/MemoryAllocators.hpp"
#include "gnuradio-4.0/Message.hpp"

#if defined(__clang__) || defined(__GNUC__)
#define ALWAYS_INLINE(return_type) inline return_type __attribute__((always_inline))
#define NEVER_INLINE(return_type)  return_type __attribute__((noinline))
#define RESTRICT                   __restrict
#elif defined(COMPILER_MSVC)
#define ALWAYS_INLINE(return_type) __forceinline return_type
#define NEVER_INLINE(return_type)  __declspec(noinline) return_type
#define RESTRICT                   __restrict
#endif

#ifndef __cpp_aligned_new
#error
#endif

#include <vir/simd.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"          // warning/error in vir/simd
#pragma GCC diagnostic ignored "-Wsign-conversion" // warning/error in vir/simd
#pragma GCC diagnostic ignored "-Wconversion"      // warning/error in vir/simd
#endif

#include <vir/simd_execution.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace stdx = vir::stdx;

template<std::floating_point T, int N = 4> // inspired by future C++26 definition
using vec = stdx::simd<T, stdx::simd_abi::deduce_t<T, static_cast<std::size_t>(N)>>;

template<typename T>
constexpr ALWAYS_INLINE(void) store_unchecked(const T& v, typename T::value_type* ptr, auto) {
    v.copy_to(ptr, stdx::vector_aligned);
}

template<typename Vec>
constexpr static ALWAYS_INLINE(auto) blend_lo2_hi1(const Vec& a, const Vec& b) noexcept {
    // takes low from 2nd arg, high from 1st (compact, FFT convention)
    // equivalent for lane 'i': out[i] = (i < 2) ? b[i] : a[i];
    constexpr int size = Vec::size();
    return vir::simd_permute<size>(stdx::concat(b, a), [](int i) { return i < 2 ? i : i + size; });
}

template<typename Vec>
constexpr static ALWAYS_INLINE(auto) interleave(const Vec& in1, const Vec& in2, Vec& out1, Vec& out2) noexcept {
    constexpr int size   = Vec::size();
    std::tie(out1, out2) = stdx::split<Vec>(vir::simd_permute(stdx::concat(in1, in2), [](int i) { return (i >> 1) + size * (i & 1); }));
}

template<typename Vec>
constexpr static ALWAYS_INLINE(auto) uninterleave(const Vec& in1, const Vec& in2, Vec& out1, Vec& out2) noexcept {
    constexpr int size   = Vec::size();
    std::tie(out1, out2) = stdx::split<Vec>(vir::simd_permute(stdx::concat(in1, in2), [](int i) noexcept -> int { return (i % size) * 2 + (i / size); }));
}

template<typename Vec>
constexpr static ALWAYS_INLINE(auto) transpose(Vec& x0, Vec& x1, Vec& x2, Vec& x3) noexcept {
    constexpr int size          = Vec::size();
    const auto [y0, y1, y2, y3] = stdx::split<Vec>(vir::simd_permute(stdx::concat(x0, x1, x2, x3), [](int i) noexcept -> int { return (i % size) * size + (i / size); }));
    x0                          = y0;
    x1                          = y1;
    x2                          = y2;
    x3                          = y3;
}

/* shortcuts for complex multiplications */
template<typename Vec, typename T>
constexpr static ALWAYS_INLINE(void) complex_multiply(Vec& ar, Vec& ai, const T& br, const T& bi) noexcept {
    const Vec tmp = ar * bi;
    ar            = ar * br - ai * bi;
    ai            = ai * br + tmp;
}

template<typename Vec, typename T>
constexpr static ALWAYS_INLINE(void) complex_multiply_conj(Vec& ar, Vec& ai, const T& br, const T& bi) noexcept {
    const Vec tmp = ar * bi;
    ar            = ar * br + ai * bi;
    ai            = ai * br - tmp;
}

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

#include "SimdFFT.cpp"

// template<std::floating_point T>
// void rffti1_ps(std::size_t n, std::span<T> wa, std::span<std::size_t, 15> radixPlan);
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
        computeTwiddles();
    }

    explicit SimdFFT(std::size_t n, std::source_location loc = std::source_location::current())
    requires(IsDynamic::value)
        : _N(n) {
        if (!canProcessSize(size())) {
            throw gr::exception(std::format("incompatible sizes for {}2C: N ({}) must be multiple of 2,3,4,5 and >{}", fftTransform == Transform::Real ? "R" : "C", size(), minSize()), loc);
        }
        computeTwiddles();
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

    void computeTwiddles() {
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
        // radix preference order: Real prefers 4 first, Complex prefers 5
        [[maybe_unused]] constexpr auto radixOrderOriginal = (fftTransform == Transform::Real) ? std::array{4UZ, 2UZ, 3UZ, 5UZ, 0UZ} : std::array{5UZ, 3UZ, 4UZ, 2UZ, 0UZ};
        [[maybe_unused]] constexpr auto radixOrder         = radixOrderOriginal; // N.B. this is potentially a platform-specific tuning parameter

        std::size_t       n         = size() / L;
        const std::size_t numStages = decompose<radixOrder>(n, _radixPlan);
        const T           argh      = (2 * std::numbers::pi_v<T>) / static_cast<T>(n);

        std::size_t       twiddlePos = fftTransform == Transform::Real ? 0 : 1;
        std::size_t       l1         = 1;
        const std::size_t loopEnd    = fftTransform == Transform::Real ? (numStages - 1) : numStages;
        for (std::size_t k1 = 1UZ; k1 <= loopEnd; ++k1) {
            const std::size_t radix  = _radixPlan[k1 + 1];
            std::size_t       ld     = 0;
            const std::size_t l2     = l1 * radix;
            const std::size_t stride = n / l2;
            const std::size_t radixm = radix - 1;

            for (std::size_t j = 1; j <= radixm; ++j) {
                ld += l1;
                const T argld = static_cast<T>(ld) * argh;

                if constexpr (fftTransform == Transform::Real) { // simple stride-based indexing
                    std::size_t twiddleIdx = twiddlePos;
                    for (std::size_t fi = 1, ii = 3; ii <= stride; ii += 2, ++fi) {
                        twiddleIdx += 2UZ;
                        _stageTwiddles[twiddleIdx - 2] = std::cos(T(fi) * argld);
                        _stageTwiddles[twiddleIdx - 1] = std::sin(T(fi) * argld);
                    }
                    twiddlePos += stride;
                } else { // fftTransform == Transform::Complex -- more complex indexing with special cases
                    const std::size_t startPos     = twiddlePos;
                    _stageTwiddles[twiddlePos - 1] = 1;
                    _stageTwiddles[twiddlePos]     = 0;

                    const std::size_t complexStride = stride + stride + 2;
                    for (std::size_t fi = 1, ii = 4; ii <= complexStride; ii += 2, ++fi) {
                        twiddlePos += 2;
                        _stageTwiddles[twiddlePos - 1] = std::cos(T(fi) * argld);
                        _stageTwiddles[twiddlePos]     = std::sin(T(fi) * argld);
                    }

                    // special handling for large radices
                    if (radix > 5) {
                        _stageTwiddles[startPos - 1] = _stageTwiddles[twiddlePos - 1];
                        _stageTwiddles[startPos]     = _stageTwiddles[twiddlePos];
                    }
                }
            }
            l1 = l2;
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

        transformInternal<direction, ordering>(inputSpan, outputSpan, scratch());
    }

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
    template<Direction direction>
    constexpr void simdReordering(std::span<const T> input, std::span<T> output) const {
        constexpr std::size_t L     = vector_type::size();
        const std::size_t     Ncvec = simdVectorSize();

        assert(input.data() != output.data());

        if constexpr (IsRealValued::value) {
            assert(canProcessSize(size(), Order::Ordered)); // non-multiple of 32 (while they can be computed) are very hard to bit-reverse
            if constexpr (direction == Direction::Forward) {
                const V* vin  = reinterpret_cast<const V*>(input.data());
                V*       vout = reinterpret_cast<V*>(output.data());

                const std::size_t dk = size() / 32; // For N=48: dk=1

                for (std::size_t k = 0; k < dk; ++k) {
                    V out0_0, out0_1, out2_0, out2_1;

                    // INTERLEAVE2(vin[k*8+0], vin[k*8+1], vout[2*(0*dk+k)+0], vout[2*(0*dk+k)+1])
                    interleave(vin[k * 8 + 0], vin[k * 8 + 1], out0_0, out0_1);
                    vout[2 * (0 * dk + k) + 0] = out0_0;
                    vout[2 * (0 * dk + k) + 1] = out0_1;

                    // INTERLEAVE2(vin[k*8+4], vin[k*8+5], vout[2*(2*dk+k)+0], vout[2*(2*dk+k)+1])
                    interleave(vin[k * 8 + 4], vin[k * 8 + 5], out2_0, out2_1);
                    vout[2 * (2 * dk + k) + 0] = out2_0;
                    vout[2 * (2 * dk + k) + 1] = out2_1;
                }

                // reversed_copy(dk, vin+2, 8, (v4sf*)(out + N/2))
                reversed_copy(dk, vin + 2, 8, reinterpret_cast<V*>(output.data() + size() / 2));

                // reversed_copy(dk, vin+6, 8, (v4sf*)(out + N))
                reversed_copy(dk, vin + 6, 8, reinterpret_cast<V*>(output.data() + size()));
            } else { // Backward
                const V* vin  = reinterpret_cast<const V*>(input.data());
                V*       vout = reinterpret_cast<V*>(output.data());

                const std::size_t dk = size() / 32;

                for (std::size_t k = 0; k < dk; ++k) {
                    V out0_0, out0_1, out4_0, out4_1;

                    uninterleave(vin[2 * (0 * dk + k) + 0], vin[2 * (0 * dk + k) + 1], out0_0, out0_1);
                    vout[k * 8 + 0] = out0_0;
                    vout[k * 8 + 1] = out0_1;

                    uninterleave(vin[2 * (2 * dk + k) + 0], vin[2 * (2 * dk + k) + 1], out4_0, out4_1);
                    vout[k * 8 + 4] = out4_0;
                    vout[k * 8 + 5] = out4_1;
                }

                unreversed_copy(dk, reinterpret_cast<const V*>(input.data() + size() / 4), reinterpret_cast<V*>(output.data() + size() - 6 * L), -8);
                unreversed_copy(dk, reinterpret_cast<const V*>(input.data() + 3 * size() / 4), reinterpret_cast<V*>(output.data() + size() - 2 * L), -8);
            }
            return;
        }

        // Complex FFT - this part was already correct
        const T* inP  = input.data();
        T*       outP = output.data();

        if constexpr (direction == Direction::Forward) {
            for (std::size_t k = 0; k < Ncvec; ++k) {
                const std::size_t kk = (k / 4) + (k % 4) * (Ncvec / 4);
                V                 lo{}, hi{};
                interleave(V(inP + (2 * k + 0) * L, stdx::vector_aligned), V(inP + (2 * k + 1) * L, stdx::vector_aligned), lo, hi);
                store_unchecked(lo, outP + (2 * kk + 0) * L, stdx::vector_aligned);
                store_unchecked(hi, outP + (2 * kk + 1) * L, stdx::vector_aligned);
            }
        } else {
            for (std::size_t k = 0; k < Ncvec; ++k) {
                const std::size_t kk = (k / 4) + (k % 4) * (Ncvec / 4);
                V                 re{}, im{};
                uninterleave(V(inP + (2 * kk + 0) * L, stdx::vector_aligned), V(inP + (2 * kk + 1) * L, stdx::vector_aligned), re, im);
                store_unchecked(re, outP + (2 * k + 0) * L, stdx::vector_aligned);
                store_unchecked(im, outP + (2 * k + 1) * L, stdx::vector_aligned);
            }
        }
    }

private:
    template<Direction direction, Order ordering>
    constexpr void transformInternal(std::span<const T> inputSpan, std::span<T> outputSpan, std::span<T> scratch) {
        if constexpr (fftTransform == Transform::Real) {
            inputSpan  = std::span{inputSpan.data(), size()};
            outputSpan = std::span{outputSpan.data(), size()};
        }

        std::span<T>  buff[2]       = {outputSpan, scratch};
        constexpr int orderinged    = (ordering == Order::Ordered) ? 1 : 0;
        const int     numStages_odd = _radixPlan[1] & 1;

        std::size_t ib = (numStages_odd ^ orderinged) ? 1 : 0;

        const std::size_t Ncvec = simdVectorSize();
        const std::size_t nVecs = IsRealValued::value ? (Ncvec * 2) : Ncvec;

        // complex-valued FFT inverts the stage direction
        constexpr Direction stagesDir           = (IsRealValued::value == (direction == Direction::Forward)) ? Direction::Forward : Direction::Backward;
        auto                processInterleaving = [&](auto operation, std::span<const T> src, std::span<T> dst) {
            assert(isAligned<64>(src.data()));
            assert(isAligned<64>(dst.data()));
            const T* RESTRICT pSrc = std::assume_aligned<64>(src.data());
            T* RESTRICT       pDst = std::assume_aligned<64>(dst.data());

            for (std::size_t k = 0UZ; k < Ncvec; ++k) {
                const std::size_t k2 = 2UZ * k * V::size();
                V                 out0, out1;
                operation(V(pSrc + k2, stdx::vector_aligned), V(pSrc + k2 + V::size(), stdx::vector_aligned), out0, out1);
                store_unchecked(out0, pDst + k2, stdx::vector_aligned);
                store_unchecked(out1, pDst + k2 + V::size(), stdx::vector_aligned);
            }
        };

        if constexpr (direction == Direction::Forward) {
            ib                             = !ib;
            std::span<const T> stagesInput = inputSpan; // default to input
            if constexpr (!IsRealValued::value) {
                processInterleaving(uninterleave<V>, inputSpan, buff[ib]);
                stagesInput = buff[ib];
            }
            std::span<T> outp = fftStages<stagesDir, fftTransform, T>(nVecs, stagesInput, buff[ib], buff[!ib], stageTwiddles(), _radixPlan);
            ib                = (outp.data() == buff[0].data()) ? 0 : 1;

            // finalise: butterfly twiddle application
            if constexpr (IsRealValued::value) {
                realFinalise<T>(Ncvec, buff[ib], buff[!ib], butterflyTwiddles());
            } else {
                complexFinalise(Ncvec, buff[ib], buff[!ib], butterflyTwiddles());
            }

            if constexpr (ordering == Order::Ordered) {
                simdReordering<Direction::Forward>(std::span<const T>{buff[!ib]}, buff[ib]);
            } else {
                ib = !ib;
            }

        } else { // Direction::Backward
            if (inputSpan.data() == buff[ib].data()) {
                ib = !ib;
            }

            if constexpr (ordering == Order::Ordered) {
                simdReordering<Direction::Backward>(inputSpan, buff[ib]);
                inputSpan = buff[ib];
                ib        = !ib;
            }

            // preprocess: inverse butterfly twiddles
            if constexpr (IsRealValued::value) {
                realPreprocess<T>(Ncvec, inputSpan, buff[ib], butterflyTwiddles());
            } else {
                complexPreprocess(Ncvec, inputSpan, buff[ib], butterflyTwiddles());
            }

            std::span<T> outp = fftStages<stagesDir, fftTransform, T>(nVecs, buff[ib], buff[0], buff[1], stageTwiddles(), _radixPlan);
            ib                = (outp.data() == buff[0].data()) ? 0 : 1;

            if constexpr (!IsRealValued::value) {
                processInterleaving(interleave<V>, buff[ib], buff[ib]);
            }
        }

        if (buff[ib].data() != outputSpan.data()) { // final copy -- only if needed
            std::memcpy(std::assume_aligned<64>(outputSpan.data()), std::assume_aligned<64>(buff[ib].data()), Ncvec * 2 * V::size() * sizeof(T));
            ib = !ib;
        }
        assert(buff[ib].data() == outputSpan.data());
    }
};

#ifdef ALWAYS_INLINE
#undef ALWAYS_INLINE
#endif
#ifdef NEVER_INLINE
#undef NEVER_INLINE
#endif
#ifdef RESTRICT
#undef RESTRICT
#endif

#endif /* SIMD_FFT_HPP */
