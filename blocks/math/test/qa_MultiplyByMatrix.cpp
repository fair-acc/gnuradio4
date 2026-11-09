#include <boost/ut.hpp>

#include <gnuradio-4.0/math/MultiplyByMatrix.hpp>
#include <gnuradio-4.0/testing/StreamProbes.hpp>

namespace {
template<typename T>
std::vector<std::vector<T>> multiplied(const gr::Tensor<T>& matrix, const std::vector<std::vector<T>>& inputs) {
    const auto columns = inputs.size();
    const auto rows    = matrix.extent(0UZ);
    return gr::testing::ProbedPorts<gr::blocks::math::MultiplyByMatrix<T>, T>(inputs, {{"matrix", matrix}}, gr::testing::indexedPorts("in", columns), gr::testing::indexedPorts("out", rows)).run();
}
} // namespace

const auto multiplyByMatrixTests = [] {
    using namespace boost::ut;

    "the identity passes three inputs through"_test = []<typename T>(const T&) {
        const gr::Tensor<T> identity({3UZ, 3UZ}, {T(1), T(0), T(0), T(0), T(1), T(0), T(0), T(0), T(1)});
        expect(std::ranges::equal(multiplied<T>(identity, {{T(1)}, {T(2)}, {T(3)}}), std::vector<std::vector<T>>{{T(1)}, {T(2)}, {T(3)}}));
    } | std::tuple<float, double, std::complex<float>, std::complex<double>>{};

    "a 4 × 3 matrix adds a fourth output that sums the inputs"_test = [] {
        const gr::Tensor<double> matrix({4UZ, 3UZ}, {1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1.});
        expect(std::ranges::equal(multiplied<double>(matrix, {{1., 10.}, {2., 20.}, {3., 30.}}), std::vector<std::vector<double>>{{1., 10.}, {2., 20.}, {3., 30.}, {6., 60.}}));
    };

    "a 2 × 2 rotation by 90 degrees maps (x, y) to (-y, x)"_test = [] {
        const gr::Tensor<float> rotation({2UZ, 2UZ}, {0.f, -1.f, 1.f, 0.f});
        expect(std::ranges::equal(multiplied<float>(rotation, {{1.f, 0.f}, {0.f, 2.f}}), std::vector<std::vector<float>>{{0.f, -2.f}, {1.f, 0.f}}));
    };

    "complex weights rotate the phase of each sample"_test = [] {
        using C = std::complex<float>;
        const gr::Tensor<C> matrix({1UZ, 2UZ}, {C(0.f, 1.f), C(2.f, 0.f)});
        expect(std::ranges::equal(multiplied<C>(matrix, {{C(1.f, 0.f)}, {C(0.f, 1.f)}}), std::vector<std::vector<C>>{{C(0.f, 3.f)}}));
    };

    "a matrix that is not of rank 2, or has no element, is refused and the previous one kept"_test = [] {
        gr::blocks::math::MultiplyByMatrix<float> block;
        gr::MsgPortIn                             errors;
        expect(block.msgOut.connect(errors).has_value());
        block.settings().init();
        expect(block.settings().set({{"matrix", gr::Tensor<float>({1UZ, 2UZ}, {1.f, 2.f})}}).empty());
        std::ignore = block.settings().activateContext();
        std::ignore = block.settings().applyStagedParameters();
        expect(block.settings().set({{"matrix", gr::Tensor<float>(gr::data_from, {1.f, 2.f, 3.f})}}).empty());
        std::ignore = block.settings().activateContext();
        std::ignore = block.settings().applyStagedParameters();
        expect(eq(block.matrix.value.rank(), 2UZ));
        expect(block.settings().set({{"matrix", gr::Tensor<float>({2UZ, 0UZ})}}).empty());
        std::ignore = block.settings().activateContext();
        std::ignore = block.settings().applyStagedParameters();
        expect(eq(block.matrix.value.size(), 2UZ));
        expect(eq(errors.streamReader().available(), 2UZ));
        expect(eq(block.in.size(), 2UZ) and eq(block.out.size(), 1UZ));
    };

    "a connected block refuses a matrix that would reshape its ports"_test = [] {
        gr::blocks::math::MultiplyByMatrix<float> block;
        gr::MsgPortIn                             errors;
        expect(block.msgOut.connect(errors).has_value());
        block.settings().init();
        expect(block.settings().set({{"matrix", gr::Tensor<float>({1UZ, 1UZ}, {2.f})}}).empty());
        std::ignore = block.settings().activateContext();
        std::ignore = block.settings().applyStagedParameters();
        gr::PortIn<float> downstream;
        expect(block.out[0].connect(downstream).has_value());
        expect(block.settings().set({{"matrix", gr::Tensor<float>({2UZ, 2UZ}, {1.f, 0.f, 0.f, 1.f})}}).empty());
        std::ignore = block.settings().activateContext();
        std::ignore = block.settings().applyStagedParameters();
        expect(eq(block.in.size(), 1UZ) and eq(block.out.size(), 1UZ));
        expect(eq(errors.streamReader().available(), 1UZ));
        expect(block.settings().set({{"matrix", gr::Tensor<float>({1UZ, 1UZ}, {3.f})}}).empty());
        std::ignore = block.settings().activateContext();
        std::ignore = block.settings().applyStagedParameters();
        expect(eq(block.matrix.value[0UZ, 0UZ], 3.f)) << "a matrix of the same shape is taken";
    };
};

int main() {
    using namespace boost::ut;
    "MultiplyByMatrix"_test = multiplyByMatrixTests;
    return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0;
}
