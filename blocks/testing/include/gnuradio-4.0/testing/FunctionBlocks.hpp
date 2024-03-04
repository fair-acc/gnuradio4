#ifndef GNURADIO_TESTING_FUNCTION_BLOCKS_HPP
#define GNURADIO_TESTING_FUNCTION_BLOCKS_HPP

#include "gnuradio-4.0/BlockRegistry.hpp"
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/reflection.hpp>
#include <gnuradio-4.0/Tag.hpp>

namespace gr::testing {

/**
 * FunctionSource is a convenience struct template to allow easy creation
 * of a source block by providing a generator function of values that
 * the source block should emit.
 *
 * - the `generator` member variable should be initialized to a function
 *   that only takes the this pointer as an argument and that returns
 *   optional<T>, where an empty optional means that the source has finished,
 *   and that the block should request the processing to stop.
 *
 * - If you want this block to have custom processing of messages received
 *   on the builtin message port (msgIn), you can define the message handler
 *   by assigning it to the `messageProcessor` member variable.
 *
 *
 * For example, a FunctionSource that sends values from 0 to 99 and
 * then quits can be written as:
 *
 * ```
 *     FunctionSource<int> source;
 *     source.generator = [counter = 0] (auto*) mutable -> optional<int> {
 *         if (counter < 100)
 *             return { counter };
 *         else
 *             return {};
 *     };
 * ```
 *
 * For example, a FunctionSource that prints out the number of messages
 * that have been send to it can be written as:
 *
 * ```
 *     FunctionSource<int> source;
 *     source.generator = ...;
 *     source.messageProcessor = [count = 0UZ] (auto* _this, auto& port, std::span<const gr::Message> messages) mutable {
 *         count += messages.size();
 *         std::print("Received {} messages so far\n", count);
 *     };
 * ```
 *
 * Note: It is only meant to be used for testing purposes, and should not be
 * used for benchmarking as it uses std::function internally.
 */
template<typename T>
struct FunctionSource : gr::Block<FunctionSource<T>> {
    PortOut<T> out;

    /** A function that generates the values sent by this source.
     * It takes a pointer to this FunctionSource and returns optional<T>.
     * If the result is an empty optional, it means that the source
     * has no more values to send
     */
    std::function<std::optional<T>(FunctionSource<T> *)> generator;

    /** A function that processes messages sent to this block on the default
     * message port msgIn.
     *
     * The arguments for the function are the pointer to this FunctionSource,
     * a reference to the port on which the message has arrived, and a span
     * of received messages
     */
    std::function<void(FunctionSource<T> *, gr::MsgPortInNamed<"__Builtin"> &, std::span<const gr::Message>)> messageProcessor;

    T
    processOne() {
        if (!generator) this->requestStop();

        auto value = generator(this);
        if (!value) this->requestStop();

        return *value;
    }

    void
    processMessages(gr::MsgPortInNamed<"__Builtin"> &port, std::span<const gr::Message> message) {
        if (messageProcessor) messageProcessor(this, port, message);
        gr::Block<FunctionSource<T>>::processMessages(port, message);
    }
};

/**
 * FunctionProcess is a convenience struct template to allow easy creation
 * of a processing block that has one input and one output by providing a
 * processing function that takes an input value and calculates the output
 * value.
 *
 * - the `processor` member variable should be initialized to a function
 *   that gets a pointer to this, and a value that it needs to process,
 *   and returns the value that this block should emit on its output port.
 *
 * - If you want this block to have custom processing of messages received
 *   on the builtin message port (msgIn), you can define the message handler
 *   by assigning it to the `messageProcessor` member variable.
 *
 *
 * For example, a FunctionProcess that multiplies values it receives by 2
 * can be written as:
 *
 * ```
 *     FunctionProcess<int> source;
 *     source.processor = [counter = 0] (auto*, int value) mutable {
 *         return 2 * value;
 *     };
 * ```
 *
 * Note: It is only meant to be used for testing purposes, and should not be
 * used for benchmarking as it uses std::function internally.
 */
template<typename T>
struct FunctionProcess : gr::Block<FunctionProcess<T>> {
    PortIn<T>  in;
    PortOut<T> out;

    /** A function that processes the values sent to this source.
     * It takes a pointer to this FunctionSource, a value of type T
     * and returns a new value of type T.
     */
    std::function<T(FunctionProcess<T> *, T)> processor;

    /** A function that processes messages sent to this block on the default
     * message port msgIn.
     *
     * The arguments for the function are the pointer to this FunctionSource,
     * a reference to the port on which the message has arrived, and a span
     * of received messages
     */
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

/**
 * FunctionSink is a convenience struct template to allow easy creation
 * of a sink block by providing a function that processes values that
 * were sent to this block.
 *
 * - the `sink` member variable should be initialized to a function
 *   that gets a pointer to this, and a value that it needs to process.
 *
 * - If you want this block to have custom processing of messages received
 *   on the builtin message port (msgIn), you can define the message handler
 *   by assigning it to the `messageProcessor` member variable.
 *
 *
 * For example, a FunctionSink that prints out the values it receives
 * can be written as:
 *
 * ```
 *     FunctionSink<int> source;
 *     source.sink = [] (auto*, T value) mutable {
 *         fmt::print("Got value {}\n", value);
 *     };
 * ```
 *
 * Note: It is only meant to be used for testing purposes, and should not be
 * used for benchmarking as it uses std::function internally.
 */
template<typename T>
struct FunctionSink : gr::Block<FunctionSink<T>> {
    PortIn<T> in;

    std::function<void(FunctionSink<T> *, T)>                                                               sink;
    std::function<void(FunctionSink<T> *, gr::MsgPortInNamed<"__Builtin"> &, std::span<const gr::Message>)> messageProcessor;

    /** A function that processes the values sent to this source.
     * It takes a pointer to this FunctionSource, and a value of type T
     * that it should process.
     */
    void
    processOne(T value) {
        if (sink) sink(this, value);
    }

    /** A function that processes messages sent to this block on the default
     * message port msgIn.
     *
     * The arguments for the function are the pointer to this FunctionSource,
     * a reference to the port on which the message has arrived, and a span
     * of received messages
     */
    void
    processMessages(gr::MsgPortInNamed<"__Builtin"> &port, std::span<const gr::Message> message) {
        if (messageProcessor) messageProcessor(this, port, message);
        gr::Block<FunctionSink<T>>::processMessages(port, message);
    }
};

/**
 * MessageSender is a convenience struct template to allow easy creation
 * of a block that generates and sends messages on the builtin messaging
 * port.
 *
 * - the `messageGenerator` member variable should be initialized to a function
 *   that gets a pointer to this, and returns optional<gr::Message>.
 *   If the result is an empty optional, it means that this block has
 *   finished and that there are no more messages to be send.
 *
 * For example, a MessageSender that sends 10 multicast messages
 * can be written as:
 *
 * ```
 *      gr::testing::MessageSender<float> messageSender;
 *      messageSender.messageGenerator = [&, count = 10UZ](auto * _this) mutable -> std::optional<gr::Message> {
 *          if (count > 0) {
 *              count--;
 *              gr::Message message;
 *              message[gr::message::key::Kind]   = "custom_kind";
 *              message[gr::message::key::Target] = "";
 *              return message;
 *          } else {
 *              return {};
 *          }
 *      };
 *
 * Note: It is only meant to be used for testing purposes, and should not be
 * used for benchmarking as it uses std::function internally.
 */
template<typename T>
struct MessageSender : public gr::Block<MessageSender<T>> {
    using super_t = gr::Block<MessageSender<T>>;

    /** A function that generates the messages to be sent by this block.
     * It takes a pointer to this FunctionSource and returns optional<gr::Message>.
     * If the result is an empty optional, it means that the block
     * has no more messages to send
     */
    std::function<std::optional<Message>(MessageSender *)> messageGenerator;

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

/**
 * A convenience class to make writing unit tests easier.
 * This sink allows to inspect the input port values as a class member.
 */
template<typename T>
class InspectSink : public gr::Block<InspectSink<T>> {
public:
    gr::PortIn<T> in;
    T             value{};

    constexpr void
    processOne(T val) {
        value = val;
    }
};

} // namespace gr::testing

ENABLE_REFLECTION_FOR_TEMPLATE(gr::testing::FunctionSource, out);
ENABLE_REFLECTION_FOR_TEMPLATE(gr::testing::FunctionProcess, in, out);
ENABLE_REFLECTION_FOR_TEMPLATE(gr::testing::FunctionSink, in);
ENABLE_REFLECTION_FOR_TEMPLATE(gr::testing::MessageSender, unused);
ENABLE_REFLECTION_FOR_TEMPLATE(gr::testing::InspectSink, in, value);

auto registerFunctionSource  = gr::registerBlock<gr::testing::FunctionSource, double, float, int>(gr::globalBlockRegistry());
auto registerFunctionProcess = gr::registerBlock<gr::testing::FunctionProcess, double, float, int>(gr::globalBlockRegistry());
auto registerFunctionSink    = gr::registerBlock<gr::testing::FunctionSink, double, float, int>(gr::globalBlockRegistry());
auto registerMessageSender   = gr::registerBlock<gr::testing::MessageSender, double, float, int>(gr::globalBlockRegistry());
auto registerInspectSink     = gr::registerBlock<gr::testing::InspectSink, double, float, int>(gr::globalBlockRegistry());

#endif // GNURADIO_TESTING_FUNCTION_BLOCKS_HPP
