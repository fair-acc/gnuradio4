#ifndef GNURADIO_ALGORITHM_FFT_TYPES_HPP
#define GNURADIO_ALGORITHM_FFT_TYPES_HPP

#include <type_traits>
#include <complex>

namespace gr::algorithm {
template<typename T>
concept ComplexType = std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>;

template<typename T, typename U>
using FFTInDataType = std::conditional_t<ComplexType<T>, std::complex<U>, U>;

template<typename U>
using FFTOutDataType = std::complex<U>;

template<typename T>
struct FFTAlgoPrecision {
    using type = T;
};

template<ComplexType T>
struct FFTAlgoPrecision<T> {
    using type = T::value_type;
};

} // namespace gr::algorithm
#endif // GNURADIO_ALGORITHM_FFT_TYPES_HPP
