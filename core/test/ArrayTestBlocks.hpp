#ifndef ARRAY_TEST_BLOCKS_HPP
#define ARRAY_TEST_BLOCKS_HPP

GR_REGISTER_BLOCK("ArraySource", ArraySource, [T], [ float, double ])

template<typename T>
struct ArraySource : gr::Block<ArraySource<T>> {
    // TODO re-enable this -> pre-requisite revert std::array<T, N> not being handled as collection but as tuple.
    static constexpr std::size_t N_SubPorts = 2UZ;
    // std::array<gr::PortOut<T>, N_SubPorts> outA{};
    // std::array<gr::PortOut<T>, N_SubPorts> outB{};
    std::vector<gr::PortOut<T>> outA{N_SubPorts};
    std::vector<gr::PortOut<T>> outB{N_SubPorts};

    GR_MAKE_REFLECTABLE(ArraySource, outA, outB);

    // template<gr::OutputSpanLike TOutputSpan1, gr::OutputSpanLike TOutputSpan2, gr::OutputSpanLike TOutputSpan3, gr::OutputSpanLike TOutputSpan4>
    // gr::work::Status processBulk(TOutputSpan1&, TOutputSpan2&, TOutputSpan3&, TOutputSpan4&) {
    //     return gr::work::Status::OK;
    // }
    gr::work::Status processBulk(auto&, auto&) { return gr::work::Status::OK; }
};

GR_REGISTER_BLOCK("ArraySinkImpl", ArraySinkImpl, ([T], true, 42), [ float, double ])

template<typename T, bool SomeFlag, int SomeInt>
struct ArraySinkImpl : gr::Block<ArraySinkImpl<T, SomeFlag, SomeInt>> {
    // TODO re-enable this -> pre-requisite revert std::array<T, N> not being handled as collection but as tuple.
    static constexpr std::size_t N_SubPorts = 2UZ;
    // std::array<gr::PortIn<T>, 2>                                                    inA;
    // std::array<gr::PortIn<T>, 2>                                                    inB;
    std::vector<gr::PortIn<T>> inA{N_SubPorts};
    std::vector<gr::PortIn<T>> inB{N_SubPorts};

    gr::Annotated<bool, "bool setting">                                             bool_setting{false};
    gr::Annotated<std::string, "String setting">                                    string_setting;
    gr::Annotated<std::complex<double>, "std::complex settings">                    complex_setting;
    gr::Annotated<std::vector<bool>, "Bool vector setting">                         bool_vector;
    gr::Annotated<std::vector<std::string>, "String vector setting">                string_vector;
    gr::Annotated<std::vector<double>, "Double vector setting">                     double_vector;
    gr::Annotated<std::vector<int16_t>, "int16_t vector setting">                   int16_vector;
    gr::Annotated<std::vector<std::complex<double>>, "std::complex vector setting"> complex_vector;

    GR_MAKE_REFLECTABLE(ArraySinkImpl, inA, inB, bool_setting, string_setting, complex_setting, bool_vector, string_vector, double_vector, int16_vector, complex_vector);

    // template<gr::InputSpanLike TInputSpan1, gr::InputSpanLike TInputSpan2, gr::InputSpanLike TInputSpan3, gr::InputSpanLike TInputSpan4>
    // gr::work::Status processBulk(TInputSpan1&, TInputSpan2&, TInputSpan3&, TInputSpan4&) {
    //     return gr::work::Status::OK;
    // }
    gr::work::Status processBulk(auto&, auto&) { return gr::work::Status::OK; }
};

// Extra template arguments to test using-declaration plus alias
GR_REGISTER_BLOCK("ArraySink", ArraySink, [T], [ float, double ])
template<typename T>
using ArraySink = ArraySinkImpl<T, true, 42>;

#endif // ARRAY_TEST_BLOCKS_HPP
