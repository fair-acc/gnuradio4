#ifndef GNURADIO_MATH_MULTIPLY_BY_MATRIX_HPP
#define GNURADIO_MATH_MULTIPLY_BY_MATRIX_HPP

#include <algorithm>
#include <format>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/Tensor.hpp>

namespace gr::blocks::math {

GR_REGISTER_BLOCK(gr::blocks::math::MultiplyByMatrix, [T], [ float, double, std::complex<float>, std::complex<double> ])

template<typename T>
requires std::floating_point<T> || meta::complex_like<T>
struct MultiplyByMatrix : Block<MultiplyByMatrix<T>> {
    using Description = Doc<R""(@brief out#r = Σ_c matrix[r, c] · in#c, sample by sample (GR3: Multiply by Matrix).

The rows of `matrix` give the outputs, its columns the inputs; a matrix that is not of rank 2, has no element, or would reshape connected ports is refused and the previous one kept.)"">;

    std::vector<PortIn<T>>  in;
    std::vector<PortOut<T>> out;

    Annotated<Tensor<T>, "matrix", Doc<"rows: outputs, columns: inputs">> matrix{};

    GR_MAKE_REFLECTABLE(MultiplyByMatrix, in, out, matrix);

    template<gr::InputSpanLike TInSpan, gr::OutputSpanLike TOutSpan>
    [[nodiscard]] gr::work::Status processBulk(std::span<TInSpan>& ins, std::span<TOutSpan>& outs) const {
        for (std::size_t row = 0UZ; row < outs.size(); ++row) {
            auto& sum = outs[row];
            std::ranges::transform(ins[0], sum.begin(), [weight = matrix.value[row, 0UZ]](const T& sample) { return weight * sample; });
            for (std::size_t column = 1UZ; column < ins.size(); ++column) {
                std::ranges::transform(sum, ins[column], sum.begin(), [weight = matrix.value[row, column]](const T& partial, const T& sample) { return partial + weight * sample; });
            }
        }
        return gr::work::Status::OK;
    }

    void settingsChanged(const property_map& oldSettings, const property_map& newSettings) {
        if (!newSettings.contains("matrix")) {
            return;
        }
        const Tensor<T>& proposed  = matrix.value;
        const auto       connected = [](const auto& ports) { return std::ranges::any_of(ports, [](const auto& port) { return port.isConnected(); }); };
        const bool       reshapes  = proposed.rank() != 2UZ || proposed.extent(0UZ) != out.size() || proposed.extent(1UZ) != in.size();
        if (proposed.rank() != 2UZ || proposed.empty() || (reshapes && (connected(in) || connected(out)))) [[unlikely]] {
            this->emitErrorMessage("settingsChanged()", std::format("a matrix of rank {} and {} elements is not outputs × inputs, or reshapes connected ports: keeps the previous one", proposed.rank(), proposed.size()));
            matrix = oldSettings.value_or<Tensor<T>>("matrix", Tensor<T>{});
            return;
        }
        in.resize(proposed.extent(1UZ));
        out.resize(proposed.extent(0UZ));
    }
};

} // namespace gr::blocks::math

#endif // GNURADIO_MATH_MULTIPLY_BY_MATRIX_HPP
