#include "gnuradio-4.0/basic/common_blocks.hpp"
#include <boost/ut.hpp>

#include <gnuradio-4.0/Scheduler.hpp>
#include <magic_enum.hpp>
#include <magic_enum_utility.hpp>

#include <gnuradio-4.0/testing/FunctionBlocks.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

using namespace std::chrono_literals;
using namespace std::string_literals;

template<typename IsDone>
auto
createOnesGenerator(IsDone &isDone) {
    return [&isDone](auto * /*_this */) -> std::optional<float> {
        if (!isDone()) {
            return 1.0f;
        } else {
            return {};
        }
    };
}

auto
messageProcessorCounter(std::atomic_size_t &countdown, std::atomic_size_t &totalCounter, std::string inMessageKind, gr::Message replyMessage) {
    return [&, inMessageKind, replyMessage](auto *_this, gr::MsgPortInNamed<"__Builtin"> &port, std::span<const gr::Message> messages) {
        if (countdown > 0) {
            for (const auto &message : messages) {
                const auto kind = gr::messageField<std::string>(message, gr::message::key::Kind);
                assert(kind);
                if (kind != "custom_kind") {
                    continue;
                }

                countdown--;
                totalCounter++;

                if (!replyMessage.empty()) _this->emitMessage(_this->msgOut, std::move(replyMessage));
            }
        }
    };
}

const boost::ut::suite MessagesTests = [] {
    using namespace boost::ut;
    using namespace gr;

    // Testing if multicast messages sent from outside of the graph reach
    // the nodes inside, and if messages sent from the node reach the outside
    "MulticastMessaggingWithTheWorld"_test = [] {
        using Scheduler = gr::scheduler::Simple<>;

        std::atomic_size_t    collectedEventCount     = 0;
        std::atomic_size_t    sourceMessagesCountdown = 1;
        std::atomic_size_t    sinkMessagesCountdown   = 1;
        constexpr std::size_t requiredEventCount      = 2;
        std::atomic_bool      receiverGotAMessage     = false;
        auto                  isDone                  = [&] { return requiredEventCount == collectedEventCount && receiverGotAMessage; };

        auto scheduler = [&] {
            gr::Graph flow;
            auto     &source = flow.emplaceBlock<gr::testing::FunctionSource<float>>();

            source.generator        = createOnesGenerator(isDone);
            source.messageProcessor = messageProcessorCounter(sourceMessagesCountdown, collectedEventCount, "custom_kind"s, {});

            auto &sink            = flow.emplaceBlock<gr::testing::FunctionSink<float>>();
            sink.messageProcessor = messageProcessorCounter(sinkMessagesCountdown, collectedEventCount, "custom_kind"s, [] {
                gr::Message outMessage;
                outMessage[gr::message::key::Kind] = "custom_reply_kind";
                return outMessage;
            }());

            auto &process = flow.emplaceBlock<builtin_multiply<float>>(property_map{});

            expect(eq(ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(process)));
            expect(eq(ConnectionResult::SUCCESS, flow.connect<"out">(process).to<"in">(sink)));

            return Scheduler(std::move(flow));
        }();

        auto &flow = scheduler.graph();

        fmt::print("Graph:\n");
        for (const auto &block : flow.blocks()) {
            fmt::print("    {}\n", block->uniqueName());
        }

        gr::testing::MessageSender<float> messageSender;
        messageSender.messageGenerator = [&](auto * /*this*/) -> std::optional<gr::Message> {
            if (!isDone()) {
                gr::Message message;
                message[gr::message::key::Kind]   = "custom_kind";
                message[gr::message::key::Target] = "";
                return message;
            } else {
                return {};
            }
        };
        expect(eq(ConnectionResult::SUCCESS, messageSender.msgOut.connect(flow.msgIn)));

        gr::testing::FunctionSink<float> messageReceiver;
        messageReceiver.messageProcessor = [&](auto *_this, gr::MsgPortInNamed<"__Builtin"> &port, std::span<const gr::Message> messages) {
            for (const auto &message : messages) {
                const auto kind = gr::messageField<std::string>(message, gr::message::key::Kind);
                if (kind == "custom_reply_kind") {
                    receiverGotAMessage = true;
                }
            }
        };
        expect(eq(ConnectionResult::SUCCESS, flow.msgOut.connect(messageReceiver.msgIn)));

        std::thread messenger([&] {
            while (!isDone()) {
                std::this_thread::sleep_for(100ms);
                messageSender.processOne();
                messageReceiver.processScheduledMessages();
            }

            fmt::print("sender and receiver thread done\n");
        });

        scheduler.runAndWait();
        messenger.join();
    };

    // Testing if targetted messages sent from outside of the graph reach
    // the node
    "TargettedMessageForABlock"_test = [] {
        using Scheduler = gr::scheduler::Simple<>;

        constexpr std::size_t requiredEventCount      = 2;
        std::atomic_size_t    collectedEventCount     = 0;
        std::atomic_size_t    sourceMessagesCountdown = 1; // not a target, should not go down
        std::atomic_size_t    sinkMessagesCountdown   = 2;
        auto                  isDone                  = [&] { return requiredEventCount == collectedEventCount; };

        auto scheduler = [&] {
            gr::Graph flow;
            auto     &source = flow.emplaceBlock<gr::testing::FunctionSource<float>>();

            source.generator        = createOnesGenerator(isDone);
            source.messageProcessor = messageProcessorCounter(sourceMessagesCountdown, collectedEventCount, "custom_kind"s, {});

            auto &sink            = flow.emplaceBlock<gr::testing::FunctionSink<float>>();
            sink.messageProcessor = messageProcessorCounter(sinkMessagesCountdown, collectedEventCount, "custom_kind"s, {});

            auto &process = flow.emplaceBlock<builtin_multiply<float>>(property_map{});

            expect(eq(ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(process)));
            expect(eq(ConnectionResult::SUCCESS, flow.connect<"out">(process).to<"in">(sink)));

            return Scheduler(std::move(flow));
        }();

        auto &flow = scheduler.graph();

        std::string target;
        fmt::print("Graph:\n");
        for (const auto &block : flow.blocks()) {
            fmt::print("    {}\n", block->uniqueName());
            if (block->uniqueName().contains("FunctionSink")) {
                target = "*/"s + std::string(block->uniqueName());
                fmt::print("This is going to be the target {}\n", target);
            }
        }

        gr::testing::MessageSender<float> messageSender;
        messageSender.messageGenerator = [&](auto * /*this*/) -> std::optional<gr::Message> {
            if (!isDone()) {
                gr::Message message;
                message[gr::message::key::Kind]   = "custom_kind";
                message[gr::message::key::Target] = target;
                return message;
            } else {
                return {};
            }
        };
        expect(eq(ConnectionResult::SUCCESS, messageSender.msgOut.connect(flow.msgIn)));

        std::thread messenger([&] {
            while (!isDone()) {
                std::this_thread::sleep_for(100ms);
                messageSender.processOne();
            }

            fmt::print("sender and receiver thread done\n");
        });

        scheduler.runAndWait();
        messenger.join();

        expect(sourceMessagesCountdown == 1);
        expect(sinkMessagesCountdown == 0);
    };

    // Testing if settings messages work
    "SettingsManagement"_test = [] {
        using Scheduler = gr::scheduler::Simple<>;

        constexpr std::atomic_size_t requiredMessageCount        = 4;
        std::atomic_size_t           settingsSuccessMessageCount = 0;
        std::atomic_size_t           settingsFailureMessageCount = 0;
        //
        auto isDone = [&] { return settingsSuccessMessageCount >= requiredMessageCount && settingsFailureMessageCount >= requiredMessageCount; };

        auto scheduler = [&] {
            gr::Graph flow;
            auto     &source = flow.emplaceBlock<gr::testing::FunctionSource<float>>();

            source.generator = createOnesGenerator(isDone);

            auto &sink    = flow.emplaceBlock<gr::testing::FunctionSink<float>>();
            auto &process = flow.emplaceBlock<builtin_multiply<float>>(property_map{});

            expect(eq(ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(process)));
            expect(eq(ConnectionResult::SUCCESS, flow.connect<"out">(process).to<"in">(sink)));

            return Scheduler(std::move(flow));
        }();

        auto &flow = scheduler.graph();

        fmt::print("Graph:\n");
        for (const auto &block : flow.blocks()) {
            fmt::print("    {}\n", block->uniqueName());
        }

        gr::testing::MessageSender<float> messageSender;
        messageSender.messageGenerator = [&](auto * /*this*/) -> std::optional<gr::Message> {
            if (!isDone()) {
                gr::Message message;
                message[gr::message::key::Kind]   = gr::message::kind::UpdateSettings;
                message[gr::message::key::Target] = std::string();
                message[gr::message::key::Data]   = gr::property_map{ { "factor", 4.4f } };
                return message;
            } else {
                return {};
            }
        };
        expect(eq(ConnectionResult::SUCCESS, messageSender.msgOut.connect(flow.msgIn)));

        gr::testing::FunctionSink<float> messageReceiver;
        messageReceiver.messageProcessor = [&](auto *_this, gr::MsgPortInNamed<"__Builtin"> &port, std::span<const gr::Message> messages) {
            for (const auto &message : messages) {
                const auto kind = *gr::messageField<std::string>(message, gr::message::key::Kind);
                if (kind == gr::message::kind::SettingsChanged) {
                    const auto sender    = gr::messageField<std::string>(message, gr::message::key::Sender);
                    const auto errorInfo = gr::messageField<property_map>(message, gr::message::key::ErrorInfo);

                    if (errorInfo.has_value()) {
                        settingsSuccessMessageCount++;
                    } else {
                        settingsFailureMessageCount++;
                    }
                }
            }
        };
        expect(eq(ConnectionResult::SUCCESS, flow.msgOut.connect(messageReceiver.msgIn)));

        std::thread messenger([&] {
            while (!isDone()) {
                std::this_thread::sleep_for(100ms);
                messageSender.processOne();
                messageReceiver.processScheduledMessages();
            }

            fmt::print("sender and receiver thread done\n");
        });

        scheduler.runAndWait();
        messenger.join();
    };
};

int
main() { /* tests are statically executed */
}
