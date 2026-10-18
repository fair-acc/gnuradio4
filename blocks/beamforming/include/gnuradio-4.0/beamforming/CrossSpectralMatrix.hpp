#ifndef GNURADIO_BEAMFORMING_CROSS_SPECTRAL_MATRIX_HPP
#define GNURADIO_BEAMFORMING_CROSS_SPECTRAL_MATRIX_HPP

#include <algorithm>
#include <array>
#include <complex>
#include <cstddef>
#include <format>
#include <memory_resource>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Complex.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/device/ParallelFor.hpp>
#include <gnuradio-4.0/meta/DeviceAnnotations.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::beamforming {

GR_REGISTER_BLOCK(gr::beamforming::CrossSpectralMatrix, [T], [ float, double, std::complex<float>, std::complex<double> ])

template<typename T>
requires std::floating_point<gr::meta::fundamental_base_value_type_t<T>>
struct CrossSpectralMatrix : Block<CrossSpectralMatrix<T>> {
    using Description = Doc<R""(@brief the spatial covariance per bin of spectra [M,F], averaged over frames: R_f <- (1 - alpha) R_f + alpha x_f x_f^H,
x_f the M elements' bin f; every frame a tensor [F,M,M], R_f[a,b] = E[x_a conj(x_b)].

The first frame, and the first after a change of M or F, sets R_f = x_f x_f^H; `alpha` 1 keeps only the newest frame.
A snapshot [M] counts as one bin: [1,M,M] (narrowband). `T` is the sample type the spectra came from, as for `STFT`.)"">;

    using Real    = gr::meta::fundamental_base_value_type_t<T>;
    using Complex = std::complex<Real>;

    TensorPortIn<Complex>  in;
    TensorPortOut<Complex> out;

    Annotated<float, "alpha", Doc<"weight of the newest frame">, Limits<0.f, 1.f, [](float weight) { return 0.f < weight; }>> alpha = 0.1f;

    /** [F,M,M]; reflected, so a kernel updates it in place */
    std::pmr::vector<Complex> _covariance;

    GR_MAKE_REFLECTABLE(CrossSpectralMatrix, in, out, alpha, _covariance);

    std::size_t _M = 0UZ;
    std::size_t _F = 0UZ;

    void start() { _covariance.clear(); }

    [[nodiscard]] gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) { return average(gr::device::hostBackend(), input, output); }

    [[nodiscard]] gr::work::Status processBulk(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) { return average(ctx, input, output); }

    [[nodiscard]] gr::work::Status average(gr::device::DeviceContext& ctx, InputSpanLike auto& input, OutputSpanLike auto& output) {
        using DeviceComplex        = gr::complex<Real>;
        const std::size_t nOffered = std::min(input.size(), output.size());
        std::size_t       k        = 0UZ;
        for (; k < nOffered; ++k) {
            const TensorView<Complex>& spectra = input[k];
            if (spectra.rank() < 1UZ || spectra.rank() > 2UZ) [[unlikely]] {
                this->emitErrorMessage("processBulk()", std::format("frame {} has shape {}, expects [M,F] or [M]", input.streamIndex + k, spectra.extents()));
                this->requestStop();
                break;
            }
            const std::size_t                M = spectra.extent(0UZ);
            const std::size_t                F = spectra.rank() == 2UZ ? spectra.extent(1UZ) : 1UZ;
            const std::array<std::size_t, 3> extents{F, M, M};
            auto                             covariance = gr::emit(*this, output, k, extents);
            if (!covariance) [[unlikely]] {
                break;
            }
            const bool restart = _covariance.empty() || _M != M || _F != F;
            if (restart) {
                _covariance.assign(F * M * M, Complex{});
                _M = M;
                _F = F;
            }
            const Real        newest = restart ? Real{1} : static_cast<Real>(alpha);
            const auto* const X      = reinterpret_cast<const DeviceComplex*>(spectra.data());
            auto* const       R      = reinterpret_cast<DeviceComplex*>(_covariance.data());
            auto* const       Y      = reinterpret_cast<DeviceComplex*>(covariance->data());
            gr::device::parallelFor(ctx, F * M * M, [X, R, Y, M, F, newest] GR_DEVICE_LAMBDA(std::size_t n) {
                const std::size_t f = n / (M * M);
                const std::size_t a = n / M % M;
                const std::size_t b = n % M;
                R[n]                = R[n] * (Real{1} - newest) + X[a * F + f] * conj(X[b * F + f]) * newest;
                Y[n]                = R[n];
            });
        }
        std::ignore = input.consume(k);
        output.publish(k);
        return gr::work::Status::OK;
    }
};

} // namespace gr::beamforming

#endif // GNURADIO_BEAMFORMING_CROSS_SPECTRAL_MATRIX_HPP
