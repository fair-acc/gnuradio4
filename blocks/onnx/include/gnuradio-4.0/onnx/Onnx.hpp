#ifndef GNURADIO_ONNX_HPP
#define GNURADIO_ONNX_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/meta/UncertainValue.hpp>

namespace gr::blocks::onnx {

template<typename T>
struct Onnx : public gr::Block<Onnx<T>> {
    PortIn<T>  in;
    PortOut<T> out;

    GR_MAKE_REFLECTABLE(Onnx, in, out);

    template<gr::meta::t_or_simd<T> V>
    [[nodiscard]] constexpr work::Status processBulk(InputSpanLike auto& inSpan, OutputSpanLike auto& outSpan) {

        return work::Status::OK;
    }
};

} // namespace gr::blocks::onnx

const inline auto registerConstMath = gr::registerBlock<gr::blocks::onnx::Onnx, float, double /*, gr::UncertainValue<float>, gr::UncertainValue<double>, std::complex<float>, std::complex<double>, std::string, gr::Packet<float>, gr::Packet<double>, gr::Tensor<float>, gr::Tensor<double>, gr::DataSet<float>, gr::DataSet<double> */>(gr::globalBlockRegistry());

#endif // GNURADIO_ONNX_HPP
