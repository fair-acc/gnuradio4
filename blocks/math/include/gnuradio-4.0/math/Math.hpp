#ifndef GNURADIO_MATH_HPP
#define GNURADIO_MATH_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>
#include <algorithm>   // std::max_element, std::distance
#include<cmath>


namespace gr::blocks::math {

namespace detail {
template<typename T>
T defaultValue() noexcept {
    if constexpr (gr::arithmetic_or_complex_like<T> || gr::UncertainValueLike<T>) {
        return static_cast<T>(1);
    } else if constexpr (std::is_arithmetic_v<T>) {
        return static_cast<T>(1);
    } else {
        return T{};
    }
}
} // namespace detail
template<class T = void> struct max;
template<class T = void> struct min;
GR_REGISTER_BLOCK("gr::blocks::math::AddConst", gr::blocks::math::MathOpImpl, ([T], std::plus<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::SubtractConst", gr::blocks::math::MathOpImpl, ([T], std::minus<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::MultiplyConst", gr::blocks::math::MathOpImpl, ([T], std::multiplies<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::DivideConst", gr::blocks::math::MathOpImpl, ([T], std::divides<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])

template<typename T, typename op>
struct MathOpImpl : Block<MathOpImpl<T, op>> {
    PortIn<T>  in{};
    PortOut<T> out{};
    T          value = detail::defaultValue<T>();

    GR_MAKE_REFLECTABLE(MathOpImpl, in, out, value);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr V processOne(const V& a) const noexcept {
        if constexpr (gr::meta::any_simd<V, T>) {
            if constexpr (std::same_as<op, std::multiplies<T>>) {
                return a * value;
            } else if constexpr (std::same_as<op, std::divides<T>>) {
                return a / value;
            } else if constexpr (std::same_as<op, std::plus<T>>) {
                return a + value;
            } else if constexpr (std::same_as<op, std::minus<T>>) {
                return a - value;
            }
             /* ---- NEW SIMD‑aware operators ---------------------------------- */
        else if constexpr (std::same_as<op, std::bit_and<T>>) { return a & value; }
        else if constexpr (std::same_as<op, std::bit_or<T>>)  { return a | value; }
        else if constexpr (std::same_as<op, std::bit_xor<T>>) { return a ^ value; }
        else if constexpr (std::same_as<op, gr::blocks::math::max<T>>)
                                                            { return std::max(a, V(value)); }
        else if constexpr (std::same_as<op, gr::blocks::math::min<T>>)
                                                            { return std::min(a, V(value)); }
            else {
                static_assert(gr::meta::always_false<T>, "unknown op");
                return V{};
            }
        } else { // non-simd branch
            return op()(a, value);
        }
    }
};

template<typename T>
using AddConst = MathOpImpl<T, std::plus<T>>;
template<typename T>
using SubtractConst = MathOpImpl<T, std::minus<T>>;
template<typename T>
using MultiplyConst = MathOpImpl<T, std::multiplies<T>>;
template<typename T>
using DivideConst = MathOpImpl<T, std::divides<T>>;

GR_REGISTER_BLOCK("gr::blocks::math::Add", gr::blocks::math::MathOpImpl, ([T], std::plus<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::Subtract", gr::blocks::math::MathOpImpl, ([T], std::minus<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::Multiply", gr::blocks::math::MathOpImpl, ([T], std::multiplies<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::Divide", gr::blocks::math::MathOpImpl, ([T], std::divides<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double> ])
GR_REGISTER_BLOCK("gr::blocks::math::Max", gr::blocks::math::MathOpMultiPortImpl, ([T], gr::blocks::math::max<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::math::Min", gr::blocks::math::MathOpMultiPortImpl, ([T], gr::blocks::math::min<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, double ])
GR_REGISTER_BLOCK("gr::blocks::math::And", gr::blocks::math::MathOpImpl, ([T], std::bit_and<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t ])
GR_REGISTER_BLOCK("gr::blocks::math::Or", gr::blocks::math::MathOpImpl, ([T], std::bit_or<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t ])
GR_REGISTER_BLOCK("gr::blocks::math::Xor", gr::blocks::math::MathOpImpl, ([T], std::bit_xor<[T]>), [ uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t ])

template<typename T, typename op>
requires(std::is_arithmetic_v<T>)
struct MathOpMultiPortImpl : Block<MathOpMultiPortImpl<T, op>> {
    using Description = Doc<R""(@brief Math block combining multiple inputs into a single output with a given operation

    Depending on the operator op this block computes:
    - Multiply: out = in_1 * in_2 * in_3 * ...
    - Divide: out = in_1 / in_2 / in_3 / ...
    - Add: out = in_1 + in_2 + in_3 + ...
    - Subtract: out = in_1 - in_2 - in_3 - ...
    - Max: out = max(in_1, in_2, in_3, ...)
    - Min: out = min(in_1, in_2, in_3, ...)
    - And: out = in_1 & in_2 & in_3 & ...
    - Or: out = in_1 | in_2 | in_3 | ...
    - Xor: out = in_1 ^ in_2 ^ in_3 ^ ...
    )"">;

    // ports
    std::vector<PortIn<T>> in;
    PortOut<T>             out;

    // settings
    Annotated<gr::Size_t, "n_inputs", Visible, Doc<"Number of inputs">, Limits<1U, 32U>> n_inputs = 0U;

    GR_MAKE_REFLECTABLE(MathOpMultiPortImpl, in, out, n_inputs);

    void settingsChanged(const gr::property_map& old_settings, const gr::property_map& new_settings) {
        if (new_settings.contains("n_inputs") && old_settings.at("n_inputs") != new_settings.at("n_inputs")) {
            in.resize(n_inputs);
        }
    }

    template<gr::InputSpanLike TInSpan>
    gr::work::Status processBulk(const std::span<TInSpan>& ins, gr::OutputSpanLike auto& sout) const {
        std::copy(ins[0].begin(), ins[0].end(), sout.begin());
        for (std::size_t n = 1; n < ins.size(); n++) {
            std::transform(sout.begin(), sout.end(), ins[n].begin(), sout.begin(), op{});
        }
        return gr::work::Status::OK;
    }
};

template<typename T>
using Add = MathOpMultiPortImpl<T, std::plus<T>>;
template<typename T>
using Subtract = MathOpMultiPortImpl<T, std::minus<T>>;
template<typename T>
using Multiply = MathOpMultiPortImpl<T, std::multiplies<T>>;
template<typename T>
using Divide = MathOpMultiPortImpl<T, std::divides<T>>;

template<typename T>
using And = MathOpMultiPortImpl<T, std::bit_and<T>>;
template<typename T>
using Or = MathOpMultiPortImpl<T, std::bit_or<T>>;
template<typename T>
using Xor = MathOpMultiPortImpl<T, std::bit_xor<T>>;

template<class T>
struct max {
    constexpr T operator()(const T& lhs, const T& rhs) const { return std::max(lhs, rhs); }
};

template<class T>
struct min {
    constexpr T operator()(const T& lhs, const T& rhs) const { return std::min(lhs, rhs); }
};

template<typename T>
using Max = MathOpMultiPortImpl<T, max<T>>;

template<typename T>
using Min = MathOpMultiPortImpl<T, min<T>>;

/* ------------------------------------------------------------------ *
 *  Unary element‑wise ops (Negate / Not / Abs)                       *
 * ------------------------------------------------------------------ */

GR_REGISTER_BLOCK("gr::blocks::math::Negate", gr::blocks::math::MathOpSinglePortImpl,
                  ([T], std::negate<[T]>),
                  [ int8_t,  int16_t,  int32_t,  int64_t,
                    float,   double,
                    gr::UncertainValue<float>,  gr::UncertainValue<double> ])

GR_REGISTER_BLOCK("gr::blocks::math::Not",    gr::blocks::math::MathOpSinglePortImpl,
                  ([T], std::bit_not<[T]>),
                  [ uint8_t, uint16_t, uint32_t, uint64_t,
                    int8_t,  int16_t,  int32_t,  int64_t ])

GR_REGISTER_BLOCK("gr::blocks::math::Abs",    gr::blocks::math::MathOpSinglePortImpl,
                  ([T], gr::blocks::math::abs_op<[T]>),
                  [ uint8_t, uint16_t, uint32_t, uint64_t,
                    int8_t,  int16_t,  int32_t,  int64_t,
                    float,   double ])

template<typename T, typename op>
requires(std::is_arithmetic_v<T>)
struct MathOpSinglePortImpl : public gr::Block<MathOpSinglePortImpl<T, op>> {
    using Description = Doc<R""(
    @brief Math block transforming a single input to a single output with a given operation

    Depending on the operator op this block computes:
    - Negate: out = - in
    - Not: out = ~ in
    - Abs: out = abs(in)
    )"">;

    // ports
    PortIn<T>  in;
    PortOut<T> out;

    GR_MAKE_REFLECTABLE(MathOpSinglePortImpl, in, out);

    [[nodiscard]] constexpr T processOne(const auto& a) const noexcept { return op{}(a); }
};

template<typename T>
using Negate = MathOpSinglePortImpl<T, std::negate<T>>;
template<typename T>
using Not = MathOpSinglePortImpl<T, std::bit_not<T>>;

template<class T = void>
struct abs_op {
    constexpr T operator()(const T& v) const {
        if constexpr (std::is_unsigned_v<T>) {
            // |v| == v  ➜ avoid ambiguous std::abs and silence -Wabsolute-value
            return v;
        } else if constexpr (std::is_integral_v<T>) {
            // std::abs promotes to int/long long; cast back to T
            using Promoted = std::conditional_t<(sizeof(long long) > sizeof(int) &&
                                                 sizeof(T) > sizeof(int)),
                                                long long,
                                                int>;
            return static_cast<T>(std::abs(static_cast<Promoted>(v)));
        } else {
            // float, double, long double: return type is already correct
            return std::abs(v);
        }
    }
};

template<typename T>
using Abs = MathOpSinglePortImpl<T, abs_op<T>>;

/* ------------------------------------------------------------------ *
 *  Log10  :  out = n · log10(|in|) + k                               *
 * ------------------------------------------------------------------ */
GR_REGISTER_BLOCK("gr::blocks::math::Log10",
                  gr::blocks::math::Log10,
                  ([T]),
                  [ float, double ])

template<typename T>
requires std::is_floating_point_v<T>
struct Log10 : gr::Block<Log10<T>>
{
    using Description = Doc<R""(
@brief Compute \(n \cdot \log_{10}(|x|) + k\)

This reproduces the behaviour of the historic *nlog10_ff* block:

- **n** – multiplicative scale (defaults to 10).
- **k** – additive offset       (defaults to 0).

Only floating-point types make sense, therefore the block is
registered for **float** and **double** streams.
)"">;

    /* ---------- ports ---------------------------- */
    PortIn<T>  in;
    PortOut<T> out;

    /* ---------- settings ------------------------- */
    Annotated<T, "n", Visible,
              Doc<"scale factor">, Limits<T(0), T(1e6)>>  n = T(10);
    Annotated<T, "k", Visible,
              Doc<"additive offset">>                     k = T(0);

    GR_MAKE_REFLECTABLE(Log10, in, out, n, k);

    /* ---------- element-wise work ---------------- */
    [[nodiscard]] constexpr T processOne(const T& v) const noexcept
    {
        /* |v| and a small floor avoid log10(0) */
        const T mag = std::fabs(v);
        const T safe = std::max(mag, std::numeric_limits<T>::min());
        return n * std::log10(safe) + k;
    }
};

/* convenient alias identical to 3.x name */
template<typename T>
using nlog10 = Log10<T>;

/* ------------------------------------------------------------------ *
 *  Integrate : running sum over N samples, output 1 value, decimate   *
 * ------------------------------------------------------------------ */

 GR_REGISTER_BLOCK("gr::blocks::math::Integrate",
                  gr::blocks::math::Integrate,
                  ([T]),
                  [ uint8_t, uint16_t, uint32_t, uint64_t,
                    int8_t,  int16_t,  int32_t,  int64_t,
                    float, double, std::complex<float>, std::complex<double> ])

template<typename T>
requires std::is_arithmetic_v<T>
struct Integrate : gr::Block<Integrate<T>>
{
    /* ---------- Doc string ---------------------------------------- */
    using Description = Doc<R""(
@brief Integrate successive samples and decimate.

For every *decim* input items the block outputs the sum of those
*decim* items and then resets the accumulator.

Example, decim = 4  
in : 1 2 3 4 5 6 7 8   →  out : 10 26
)"">;

    /* ---------- Ports & settings ---------------------------------- */
    PortIn<T>  in;
    PortOut<T> out;

    Annotated<gr::Size_t,
              "decim",
              Visible,
              Doc<"decimation / integration length">,
              Limits<1UZ, (1UZ << 20)> >    // ← space before last '>'
        decim = 1UZ;

    GR_MAKE_REFLECTABLE(Integrate, in, out, decim);

    /* ---------- State --------------------------------------------- */
    T           _acc  = T(0);
    std::size_t _seen = 0;

    /* ---------- Reset on start / stop ----------------------------- */
    void start()        { _acc = T(0); _seen = 0; }
    /* no explicit stop() needed – Block takes care of it */

    /* ---------- Work (scalar & SIMD transparent) ------------------ */
    template<gr::InputSpanLike TSpanIn, gr::OutputSpanLike TSpanOut>
    gr::work::Status processBulk(const TSpanIn& ins, TSpanOut& outs)
    {
        auto       outIt   = outs.begin();
        const auto dec     = decim.value;

        for (auto v : ins) {
            _acc += v;
            if (++_seen == dec) {
                *outIt++ = _acc;
                _acc  = T(0);
                _seen = 0;
            }
        }
        outs.publish(static_cast<std::size_t>(outIt - outs.begin()));   // how many we produced
        return gr::work::Status::OK;
    }
};

/* helper alias so tests can just write Integrate<T> */
template<typename T>
using integrate = Integrate<T>;

/* ------------------------------------------------------------------ *
 *  Argmax  : find index of the largest value in every vlen-element    *
 *            vector and output that index (0 … vlen-1)                *
 * ------------------------------------------------------------------ */

GR_REGISTER_BLOCK("gr::blocks::math::Argmax",
                  gr::blocks::math::Argmax,
                  ([T]),
                  [ uint8_t,  uint16_t,  uint32_t,  uint64_t,
                    int8_t,   int16_t,   int32_t,   int64_t,
                    float,    double ])

template<typename T>
requires std::is_arithmetic_v<T>
struct Argmax : gr::Block<Argmax<T>>
{
    /* ---------- one-liner shown in GUI / docs --------------------- */
    using Description = Doc<"Argmax – returns the index of the maximum element in every vlen-sample input vector">;

    /* ---------- ports -------------------------------------------- */
    PortIn< T >         in;                 // stream of scalars
    PortOut<gr::Size_t> out;                // one index per vector

    /* ---------- settings ----------------------------------------- */
    Annotated<gr::Size_t,
              "vlen",
              Visible,
              Doc<"Vector length analysed per result">,
              Limits<1UZ, (1UZ << 16)>>
        vlen = 1UZ;

    GR_MAKE_REFLECTABLE(Argmax, in, out, vlen);

    /* ---------- work --------------------------------------------- */
    template<gr::InputSpanLike  TSpanIn,
             gr::OutputSpanLike TSpanOut>
    gr::work::Status processBulk(const TSpanIn& ins, TSpanOut& outs)
    {
        const auto L = static_cast<std::size_t>(vlen.value);

        auto  in_it  = ins.begin();
        auto  out_it = outs.begin();

        while (std::distance(in_it, ins.end()) >= static_cast<std::ptrdiff_t>(L)) {
    auto max_it = std::max_element(in_it, in_it + static_cast<std::ptrdiff_t>(L));
    *out_it++   = static_cast<gr::Size_t>(std::distance(in_it, max_it));
    in_it      += static_cast<std::ptrdiff_t>(L);
}
        outs.publish(static_cast<std::size_t>(out_it - outs.begin()));
        return gr::work::Status::OK;
    }
};

/* helper alias for tests – mirrors the 3.x name */
template<typename T>
using argmax = Argmax<T>;

} // namespace gr::blocks::math

#endif // GNURADIO_MATH_HPP
