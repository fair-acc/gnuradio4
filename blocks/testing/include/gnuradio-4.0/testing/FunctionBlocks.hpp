#ifndef GNURADIO_TESTING_FUNCTION_BLOCKS_HPP
#define GNURADIO_TESTING_FUNCTION_BLOCKS_HPP

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/reflection.hpp>
#include <gnuradio-4.0/Tag.hpp>

namespace gr::testing {

template<typename T>
struct FunctionSource : public gr::Block<FunctionSource<T>> {
    PortOut<T> out;

    std::function<std::optional<T>(FunctionSource<T> *)>                                                                     generator;
    std::function<void(FunctionSource<T> *, gr::MsgPortInNamed<"__Builtin"> &, std::span<const gr::Message>)> messageProcessor;

    T
    processOne() {
        if (!generator)
            this->requestStop();

        auto value = generator(this);
        if (!value)
            this->requestStop();

        return *value;
    }

    void
    processMessages(gr::MsgPortInNamed<"__Builtin"> &port, std::span<const gr::Message> message) {
        if (messageProcessor) messageProcessor(this, port, message);
        gr::Block<FunctionSource<T>>::processMessages(port, message);
    }
};

template<typename T>
struct FunctionProcess : public gr::Block<FunctionProcess<T>> {
    PortIn<T>  in;
    PortOut<T> out;

    std::function<T(FunctionProcess<T> *, T)>                                                                  processor;
    std::function<void(FunctionProcess<T> *, gr::MsgPortInNamed<"__Builtin"> &, std::span<const gr::Message>)> messageProcessor;

    T
    processOne(T value) {
        return processor ? processor(this, value) : value;
    }

    void
    processMessages(gr::MsgPortInNamed<"__Builtin"> &port, std::span<const gr::Message> message) {
        if (messageProcessor) messageProcessor(this, port, message);
        gr::Block<FunctionProcess<T>>::processMessages(port, message);
    }
};

template<typename T>
struct FunctionSink : public gr::Block<FunctionSink<T>> {
    PortIn<T> in;

    std::function<void(FunctionSink<T> *, T)>                                                               sink;
    std::function<void(FunctionSink<T> *, gr::MsgPortInNamed<"__Builtin"> &, std::span<const gr::Message>)> messageProcessor;

    void
    processOne(T value) {
        if (sink) sink(this, value);
    }

    void
    processMessages(gr::MsgPortInNamed<"__Builtin"> &port, std::span<const gr::Message> message) {
        if (messageProcessor) messageProcessor(this, port, message);
        gr::Block<FunctionSink<T>>::processMessages(port, message);
    }
};

template<typename T>
struct MessageSender : public gr::Block<MessageSender<T>> {
    using super_t = gr::Block<MessageSender<T>>;
    std::function<std::optional<Message>(MessageSender*)> messageGenerator;

    gr::PortOut<T> unused;

    constexpr T
    processOne() {
        assert(messageGenerator);

        auto message = messageGenerator(this);
        if (message) {
            super_t::emitMessage(super_t::msgOut, *message);
        } else {
            this->requestStop();
        }

        return T{};
    }
};


} // namespace gr::testing

ENABLE_REFLECTION_FOR_TEMPLATE(gr::testing::FunctionSource, out);
ENABLE_REFLECTION_FOR_TEMPLATE(gr::testing::FunctionProcess, in, out);
ENABLE_REFLECTION_FOR_TEMPLATE(gr::testing::FunctionSink, in);
ENABLE_REFLECTION_FOR_TEMPLATE(gr::testing::MessageSender, unused)

#endif // GNURADIO_TESTING_FUNCTION_BLOCKS_HPP
