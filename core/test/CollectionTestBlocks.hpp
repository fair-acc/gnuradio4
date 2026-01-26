#ifndef COLLECTION_TEST_BLOCKS_HPP
#define COLLECTION_TEST_BLOCKS_HPP

#include <array>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

namespace gr::testing {
// Vector-based test blocks

GR_REGISTER_BLOCK("gr::testing::VectorSource", gr::testing::VectorSource, [T], [ float, double ])

template<typename T>
struct VectorSource : gr::Block<VectorSource<T>> {
    static constexpr std::size_t N_SubPorts = 2UZ;
    std::vector<gr::PortOut<T>>  outA{N_SubPorts};
    std::vector<gr::PortOut<T>>  outB{N_SubPorts};

    GR_MAKE_REFLECTABLE(VectorSource, outA, outB);

    gr::work::Status processBulk(auto&, auto&) { return gr::work::Status::OK; }
};

GR_REGISTER_BLOCK("gr::testing::VectorSinkImpl", gr::testing::VectorSinkImpl, ([T], true, 42), [ float, double ])

template<typename T, bool SomeFlag, int SomeInt>
struct VectorSinkImpl : gr::Block<VectorSinkImpl<T, SomeFlag, SomeInt>> {
    static constexpr std::size_t N_SubPorts = 2UZ;
    std::vector<gr::PortIn<T>>   inA{N_SubPorts};
    std::vector<gr::PortIn<T>>   inB{N_SubPorts};

    gr::Annotated<bool, "bool setting">                                             bool_setting{false};
    gr::Annotated<std::string, "String setting">                                    string_setting;
    gr::Annotated<std::complex<double>, "std::complex settings">                    complex_setting;
    gr::Annotated<std::vector<bool>, "Bool vector setting">                         bool_vector;
    gr::Annotated<std::vector<pmt::Value>, "String vector setting">                 string_vector;
    gr::Annotated<std::vector<double>, "Double vector setting">                     double_vector;
    gr::Annotated<std::vector<int16_t>, "int16_t vector setting">                   int16_vector;
    gr::Annotated<std::vector<std::complex<double>>, "std::complex vector setting"> complex_vector;

    GR_MAKE_REFLECTABLE(VectorSinkImpl, inA, inB, bool_setting, string_setting, complex_setting, bool_vector, string_vector, double_vector, int16_vector, complex_vector);

    gr::work::Status processBulk(auto&, auto&) { return gr::work::Status::OK; }
};

// Extra template arguments to test using-declaration plus alias
GR_REGISTER_BLOCK("gr::testing::VectorSink", gr::testing::VectorSink, [T], [ float, double ])
template<typename T>
using VectorSink = VectorSinkImpl<T, true, 42>;

// Array-based test blocks

GR_REGISTER_BLOCK("gr::testing::ArraySource", gr::testing::ArraySource, [T], [ float, double ])

template<typename T>
struct ArraySource : gr::Block<ArraySource<T>> {
    static constexpr std::size_t           N_SubPorts = 2UZ;
    std::array<gr::PortOut<T>, N_SubPorts> outA{};
    std::array<gr::PortOut<T>, N_SubPorts> outB{};

    GR_MAKE_REFLECTABLE(ArraySource, outA, outB);

    gr::work::Status processBulk(auto&, auto&) { return gr::work::Status::OK; }
};

GR_REGISTER_BLOCK("gr::testing::ArraySinkImpl", gr::testing::ArraySinkImpl, ([T], true, 42), [ float, double ])

template<typename T, bool SomeFlag, int SomeInt>
struct ArraySinkImpl : gr::Block<ArraySinkImpl<T, SomeFlag, SomeInt>> {
    // TODO re-enable this -> pre-requisite revert std::array<T, N> not being handled as collection but as tuple.
    static constexpr std::size_t          N_SubPorts = 2UZ;
    std::array<gr::PortIn<T>, N_SubPorts> inA;
    std::array<gr::PortIn<T>, N_SubPorts> inB;

    gr::Annotated<bool, "bool setting">                                        bool_setting{false};
    gr::Annotated<std::string, "String setting">                               string_setting;
    gr::Annotated<std::complex<double>, "std::complex settings">               complex_setting;
    gr::Annotated<Tensor<bool>, "Bool vector setting">                         bool_vector;
    gr::Annotated<Tensor<pmt::Value>, "String vector setting">                 string_vector;
    gr::Annotated<Tensor<double>, "Double vector setting">                     double_vector;
    gr::Annotated<Tensor<int16_t>, "int16_t vector setting">                   int16_vector;
    gr::Annotated<Tensor<std::complex<double>>, "std::complex vector setting"> complex_vector;

    GR_MAKE_REFLECTABLE(ArraySinkImpl, inA, inB, bool_setting, string_setting, complex_setting, bool_vector, string_vector, double_vector, int16_vector, complex_vector);

    gr::work::Status processBulk(auto&, auto&) { return gr::work::Status::OK; }
};

// Extra template arguments to test using-declaration plus alias
GR_REGISTER_BLOCK("gr::testing::ArraySink", gr::testing::ArraySink, [T], [ float, double ])
template<typename T>
using ArraySink = ArraySinkImpl<T, true, 42>;
} // namespace gr::testing

#endif // COLLECTION_TEST_BLOCKS_HPP
