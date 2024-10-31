#include <boost/ut.hpp>

#include "gnuradio-4.0/Block.hpp"
#include "gnuradio-4.0/Message.hpp"
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/clock_source.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <magic_enum.hpp>
#include <magic_enum_utility.hpp>

#include <optional>

using namespace std::chrono_literals;
using namespace std::string_literals;

using namespace gr::message;

namespace gr::testing {

template<typename T>
struct TestBlock : public gr::Block<TestBlock<T>> {
    gr::PortIn<T>  in{};
    gr::PortOut<T> out{};
    T              factor = static_cast<T>(1.0f);

    GR_MAKE_REFLECTABLE(TestBlock, in, out, factor);

    void settingsChanged(const property_map& /* oldSettings */, const property_map& newSettings) {
        if (newSettings.contains("factor")) {
            this->notifyListeners("Settings", {{"factor", newSettings.at("factor")}});
            // notifies only subscribed listeners
            // alt: sendMessage<message::Command::Notify>(this->msgOut, this->unique_name /* serviceName */, "Settings", { { "factor", newSettings.at("factor") } }); // notifies all
        }
    }

    std::optional<Message> unmatchedPropertyHandler(std::string_view propertyName, Message msg) {
        fmt::println("called Block's {} unmatchedPropertyHandler({}, {})", this->name, propertyName, msg);
        if (msg.endpoint == "Unknown") {
            // special case for the unit-test to mimic unknown properties
            return std::nullopt;
        }

        if (msg.endpoint == "CustomEndpoint") {
            sendMessage<message::Command::Notify>(this->msgOut, "", "custom_reply_kind", property_map{{"key", "testReplyData"}});
            return std::nullopt;
        }

        throw gr::exception(fmt::format("Blocks {} unsupported property in unmatchedPropertyHandler({}, {})", this->name, propertyName, msg));
    }

    [[nodiscard]] constexpr auto processOne(T a) const noexcept { return a * factor; }
};

} // namespace gr::testing

template<typename T>
struct ProcessMessageStdSpanBlock : gr::Block<ProcessMessageStdSpanBlock<T>> {
    gr::PortIn<T> in;

    T processOne(T) { return {}; }

    void processMessages(gr::MsgPortInBuiltin&, std::span<const gr::Message>) {}
};

static_assert(gr::traits::block::can_processMessagesForPortReaderSpan<ProcessMessageStdSpanBlock<int>, gr::MsgPortInBuiltin>);
static_assert(gr::traits::block::can_processMessagesForPortStdSpan<ProcessMessageStdSpanBlock<int>, gr::MsgPortInBuiltin>);

template<typename T>
struct ProcessMessageReaderSpanBlock : gr::Block<ProcessMessageReaderSpanBlock<T>> {
    gr::PortIn<T> in;

    T processOne(T) { return {}; }

    void processMessages(gr::MsgPortInBuiltin&, gr::ReaderSpanLike auto) {}
};

static_assert(gr::traits::block::can_processMessagesForPortReaderSpan<ProcessMessageReaderSpanBlock<int>, gr::MsgPortInBuiltin>);
static_assert(!gr::traits::block::can_processMessagesForPortStdSpan<ProcessMessageReaderSpanBlock<int>, gr::MsgPortInBuiltin>);

using namespace boost::ut;
using namespace gr;

const boost::ut::suite MessagesTests = [] {
    using namespace std::string_literals;
    using namespace boost::ut;
    using namespace gr;

    static auto returnReplyMsg = [](gr::MsgPortIn& port) {
        ReaderSpanLike auto span = port.streamReader().get<SpanReleasePolicy::ProcessAll>(1UZ);
        Message             msg  = span[0];
        expect(span.consume(span.size()));
        return msg;
    };

    "Block<T>-level message tests"_test = [] {
        using namespace gr::testing;
        using enum gr::message::Command;

        "Block<T>-level heartbeat tests"_test = [] {
            gr::MsgPortOut toBlock;
            TestBlock<int> unitTestBlock(property_map{{"name", "UnitTestBlock"}});
            gr::MsgPortIn  fromBlock;

            expect(eq(ConnectionResult::SUCCESS, toBlock.connect(unitTestBlock.msgIn)));
            expect(eq(ConnectionResult::SUCCESS, unitTestBlock.msgOut.connect(fromBlock)));

            "w/o explicit serviceName"_test = [&] {
                sendMessage<Set>(toBlock, "" /* serviceName */, block::property::kHeartbeat /* endpoint */, {{"myKey", "value"}} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "didn't receive heartbeat reply message";
                const Message reply = returnReplyMsg(fromBlock);
                expect(reply.cmd == Final) << fmt::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
                expect(eq(reply.serviceName, unitTestBlock.unique_name));
                expect(eq(reply.clientRequestID, ""s));
                expect(eq(reply.endpoint, std::string(block::property::kHeartbeat)));
                expect(reply.data.has_value());
                expect(reply.data.value().contains("heartbeat"));
            };

            "w/ explicit serviceName = unique_name"_test = [&] {
                sendMessage<Get>(toBlock, unitTestBlock.unique_name /* serviceName */, block::property::kHeartbeat /* endpoint */, {{"myKey", "value"}} /* data  */, "client#42");
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "didn't receive heartbeat reply message";
                const Message reply = returnReplyMsg(fromBlock);
                expect(reply.cmd == Final) << fmt::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
                expect(eq(reply.serviceName, unitTestBlock.unique_name));
                expect(eq(reply.clientRequestID, "client#42"s));
                expect(eq(reply.endpoint, std::string(block::property::kHeartbeat)));
                expect(reply.data.has_value());
                expect(reply.data.value().contains("heartbeat"));
            };

            "subscription"_test = [&] {
                sendMessage<Subscribe>(toBlock, unitTestBlock.unique_name /* serviceName */, block::property::kHeartbeat /* endpoint */, {} /* data  */, "client#42");
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";
                expect(eq(fromBlock.streamReader().available(), 0UZ)) << "should not receive reply";

                // trigger any message action
                sendMessage<Get>(toBlock, "", "Unknown", {});
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "should receive heartbeat";
                const Message heartbeat1 = returnReplyMsg(fromBlock);
                expect(heartbeat1.cmd == Notify) << fmt::format("mismatch between heartbeat1.cmd = {} and expected {} command", heartbeat1.cmd, Notify);
                expect(eq(heartbeat1.endpoint, std::string(block::property::kHeartbeat)));
                expect(heartbeat1.data.has_value());
                expect(heartbeat1.data.value().contains("heartbeat"));

                // unsubscribe
                sendMessage<Unsubscribe>(toBlock, "", block::property::kHeartbeat, {}, "client#42");
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";
                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "should receive heartbeat";
                const Message heartbeat2 = returnReplyMsg(fromBlock);
                expect(heartbeat2.cmd == Notify) << fmt::format("mismatch between heartbeat2.cmd = {} and expected {} command", heartbeat2.cmd, Notify);
                expect(eq(heartbeat2.endpoint, std::string(block::property::kHeartbeat)));
                expect(heartbeat2.data.has_value());
                expect(heartbeat2.data.value().contains("heartbeat"));

                sendMessage<Get>(toBlock, "", "Unknown", {});
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";
                expect(eq(fromBlock.streamReader().available(), 0UZ)) << "should not receive heartbeat";
            };
        };

        "Block<T>-level echo tests"_test = [] {
            gr::MsgPortOut toBlock;
            TestBlock<int> unitTestBlock(property_map{{"name", "UnitTestBlock"}});
            gr::MsgPortIn  fromBlock;

            expect(eq(ConnectionResult::SUCCESS, toBlock.connect(unitTestBlock.msgIn)));
            expect(eq(ConnectionResult::SUCCESS, unitTestBlock.msgOut.connect(fromBlock)));

            "w/o explicit serviceName"_test = [&] {
                sendMessage<Set>(toBlock, "" /* serviceName */, block::property::kEcho /* endpoint */, {{"myKey", "value"}} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "didn't receive reply message";
                const Message reply = returnReplyMsg(fromBlock);
                expect(reply.cmd == Final) << fmt::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
                expect(eq(reply.serviceName, unitTestBlock.unique_name));
                expect(eq(reply.clientRequestID, ""s));
                expect(eq(reply.endpoint, std::string(block::property::kEcho)));
                expect(reply.data.has_value());
                expect(reply.data.value().contains("myKey"));
            };

            "w/ explicit serviceName = unique_name"_test = [&] {
                sendMessage<Set>(toBlock, unitTestBlock.unique_name /* serviceName */, block::property::kEcho /* endpoint */, {{"myKey", "value"}} /* data  */, "client#42");
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "didn't receive reply message";
                const Message reply = returnReplyMsg(fromBlock);
                expect(reply.cmd == Final) << fmt::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
                expect(eq(reply.serviceName, unitTestBlock.unique_name));
                expect(eq(reply.clientRequestID, "client#42"s));
                expect(eq(reply.endpoint, std::string(block::property::kEcho)));
                expect(reply.data.has_value());
                expect(reply.data.value().contains("myKey"));
            };

            "w/ explicit serviceName = name"_test = [&] {
                sendMessage<Set>(toBlock, unitTestBlock.name /* serviceName */, block::property::kEcho /* endpoint */, {{"myKey", "value"}} /* data  */, "client#42");
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "didn't receive reply message";
                const Message reply = returnReplyMsg(fromBlock);
                expect(reply.cmd == Final) << fmt::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
                expect(eq(reply.serviceName, unitTestBlock.unique_name));
                expect(eq(reply.clientRequestID, "client#42"s));
                expect(eq(reply.endpoint, std::string(block::property::kEcho)));
                expect(reply.data.has_value());
                expect(reply.data.value().contains("myKey"));
            };

            "w/ explicit serviceName = <unknown>"_test = [&] {
                sendMessage<Set>(toBlock, "<unknown>" /* serviceName */, block::property::kEcho /* endpoint */, {} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 0UZ)) << "should not receive reply message for unknown/mismatching service";
            };

            "w/ unknown endpoint"_test = [&] {
                sendMessage<Set>(toBlock, unitTestBlock.name /* serviceName */, "Unknown" /* endpoint */, {} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 0UZ)) << "should not receive reply message for unknown property";
            };

            "w/ unknown command"_test = [&] {
                sendMessage<Get>(toBlock, "" /* serviceName */, block::property::kEcho /* endpoint */, {{"myKey", "value"}} /* data  */, "client#42");
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "didn't receive reply message";
                const Message reply = returnReplyMsg(fromBlock);
                expect(reply.cmd == Final) << fmt::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
                expect(eq(reply.serviceName, unitTestBlock.unique_name));
                expect(eq(reply.clientRequestID, "client#42"s));
                expect(eq(reply.endpoint, std::string(block::property::kEcho)));
                expect(!reply.data.has_value());
            };
        };

        "Block<T>-level lifecycle::State tests"_test = [] {
            gr::MsgPortOut toBlock;
            TestBlock<int> unitTestBlock(property_map{{"name", "UnitTestBlock"}});
            gr::MsgPortIn  fromBlock;

            expect(eq(ConnectionResult::SUCCESS, toBlock.connect(unitTestBlock.msgIn)));
            expect(eq(ConnectionResult::SUCCESS, unitTestBlock.msgOut.connect(fromBlock)));

            "get - state"_test = [&] {
                sendMessage<Get>(toBlock, "" /* serviceName */, block::property::kLifeCycleState /* endpoint */, {} /* data  */, "client#42");
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "didn't receive reply message";
                const Message reply = returnReplyMsg(fromBlock);
                expect(reply.cmd == Final) << fmt::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
                expect(eq(reply.serviceName, unitTestBlock.unique_name));
                expect(eq(reply.clientRequestID, "client#42"s));
                expect(eq(reply.endpoint, std::string(block::property::kLifeCycleState)));
                expect(reply.data.has_value());
            };

            "set - state"_test = [&] {
                sendMessage<Set>(toBlock, "" /* serviceName */, block::property::kLifeCycleState /* endpoint */, {{"state", "INITIALISED"}} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 0UZ)) << "should not receive a reply";
                expect(unitTestBlock.state() == lifecycle::State::INITIALISED);
            };

            "set - state - error cases"_test = [&] {
                sendMessage<Set>(toBlock, "" /* serviceName */, block::property::kLifeCycleState /* endpoint */, {} /* no data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";
                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "should have one error (missing set data)";
                expect(fromBlock.streamReader().get(1UZ).consume(1UZ));
                expect(unitTestBlock.state() == lifecycle::State::INITIALISED);

                sendMessage<Set>(toBlock, "" /* serviceName */, block::property::kLifeCycleState /* endpoint */, {{"MisSpelledStateKey", "INITIALISED"}} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";
                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "should have one error (unknown key)";
                expect(fromBlock.streamReader().get(1UZ).consume(1UZ));
                expect(unitTestBlock.state() == lifecycle::State::INITIALISED);

                sendMessage<Set>(toBlock, "" /* serviceName */, block::property::kLifeCycleState /* endpoint */, {{"state", "UNKNOWN_STATE"}} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";
                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "should have one error";
                expect(fromBlock.streamReader().get(1UZ).consume(1UZ));
                expect(unitTestBlock.state() == lifecycle::State::INITIALISED);

                sendMessage<Set>(toBlock, "" /* serviceName */, block::property::kLifeCycleState /* endpoint */, {{"state", 6}} /* wrong state type  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";
                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "should have one error";
                expect(fromBlock.streamReader().get(1UZ).consume(1UZ));
                expect(unitTestBlock.state() == lifecycle::State::INITIALISED);
            };
        };

        "Block<T>-level (staged) settings tests"_test = [] {
            gr::MsgPortOut toBlock;
            TestBlock<int> unitTestBlock(property_map{{"name", "UnitTestBlock"}});
            std::ignore = unitTestBlock.settings().applyStagedParameters(); // call manually (N.B. normally initialised by Graph/Scheduler)
            gr::MsgPortIn fromBlock;

            expect(eq(ConnectionResult::SUCCESS, toBlock.connect(unitTestBlock.msgIn)));
            expect(eq(ConnectionResult::SUCCESS, unitTestBlock.msgOut.connect(fromBlock)));

            "get - Settings"_test = [&] {
                sendMessage<Get>(toBlock, "" /* serviceName */, block::property::kSetting /* endpoint */, {} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "didn't receive reply message";
                const Message reply = returnReplyMsg(fromBlock);
                expect(reply.cmd == Final) << fmt::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
                expect(eq(reply.serviceName, unitTestBlock.unique_name));
                expect(eq(reply.endpoint, std::string(block::property::kSetting)));
                expect(reply.data.has_value());
                expect(!reply.data.value().empty());
                expect(reply.data.value().contains("factor"));
                expect(eq(1, std::get<int>(reply.data.value().at("factor"))));
            };

            "get - StagedSettings"_test = [&] {
                sendMessage<Get>(toBlock, "" /* serviceName */, block::property::kStagedSetting /* endpoint */, {} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "didn't receive staged setting reply message";
                const Message reply = returnReplyMsg(fromBlock);
                expect(reply.cmd == Final) << fmt::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
                expect(eq(reply.serviceName, unitTestBlock.unique_name));
                expect(eq(reply.endpoint, std::string(block::property::kStagedSetting)));
                expect(reply.data.has_value());
                expect(reply.data.value().empty());
            };

            "set - StagedSettings"_test = [&] {
                sendMessage<Set>(toBlock, "" /* serviceName */, block::property::kStagedSetting /* endpoint */, {{"factor", 42}} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 0UZ)) << "should not receive a reply";
                property_map stagedSettings = unitTestBlock.settings().stagedParameters();
                expect(stagedSettings.contains("factor"));
                expect(eq(42, std::get<int>(stagedSettings.at("factor"))));

                // setting staged setting via staged setting (N.B. non-real-time <-> real-time setting decoupling
                sendMessage<Set>(toBlock, "" /* serviceName */, block::property::kSetting /* endpoint */, {{"factor", 43}} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 0UZ)) << "should not receive a reply";
                stagedSettings = unitTestBlock.settings().stagedParameters();
                expect(stagedSettings.contains("factor"));
                expect(eq(43, std::get<int>(stagedSettings.at("factor"))));
            };
        };
    };

    "Multi-Block<T> message passing tests"_test = [] {
        using namespace gr::testing;
        using enum gr::message::Command;

        gr::MsgPortOut toBlock;
        TestBlock<int> unitTestBlock1(property_map{{"name", "UnitTestBlock1"}});
        TestBlock<int> unitTestBlock2(property_map{{"name", "UnitTestBlock2"}});
        gr::MsgPortIn  fromBlock;

        const auto processMessage = [&]() {
            expect(nothrow([&] { unitTestBlock1.processScheduledMessages(); })) << "manually execute processing of messages";
            expect(nothrow([&] { unitTestBlock2.processScheduledMessages(); })) << "manually execute processing of messages";
        };

        expect(eq(ConnectionResult::SUCCESS, toBlock.connect(unitTestBlock1.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, toBlock.connect(unitTestBlock2.msgIn)));

        // bug/missing connect API, second connect invalidates first unitTestBlock1.msgOut connections
        // expect(eq(ConnectionResult::SUCCESS, unitTestBlock1.msgOut.connect(fromBlock)));
        // expect(eq(ConnectionResult::SUCCESS, unitTestBlock2.msgOut.connect(fromBlock)));
        // workaround
        expect(eq(ConnectionResult::SUCCESS, unitTestBlock1.msgOut.connect(fromBlock)));
        auto buffer = fromBlock.buffer();
        // unitTestBlock1.msgOut.setBuffer(buffer.streamBuffer, buffer.tagBuffer);
        unitTestBlock2.msgOut.setBuffer(buffer.streamBuffer, buffer.tagBuffer);

        expect(eq(toBlock.buffer().streamBuffer.n_readers(), 2UZ)) << "need two consumer";
        expect(eq(unitTestBlock1.msgOut.buffer().streamBuffer.n_readers(), 1UZ)) << "need one consumer";
        expect(eq(unitTestBlock2.msgOut.buffer().streamBuffer.n_readers(), 1UZ)) << "need one consumer";

        sendMessage<Subscribe>(toBlock, unitTestBlock1.unique_name /* serviceName */, block::property::kHeartbeat /* endpoint */, {} /* data  */, "client#42");
        sendMessage<Subscribe>(toBlock, unitTestBlock2.unique_name /* serviceName */, block::property::kHeartbeat /* endpoint */, {} /* data  */, "client#42");
        processMessage();
        expect(eq(fromBlock.streamReader().available(), 0UZ)) << "should not receive reply";

        // trigger any message action
        sendMessage<Get>(toBlock, "", "Unknown", {});
        processMessage();

        expect(eq(fromBlock.streamReader().available(), 2UZ)) << "should receive two heartbeats";
        const Message heartbeat1 = returnReplyMsg(fromBlock);
        expect(heartbeat1.cmd == Notify) << fmt::format("mismatch between heartbeat1.cmd = {} and expected {} command", heartbeat1.cmd, Notify);
        expect(eq(heartbeat1.endpoint, std::string(block::property::kHeartbeat)));
        expect(heartbeat1.data.has_value());
        expect(heartbeat1.data.value().contains("heartbeat"));
    };

    constexpr auto schedulingPolicies = std::tuple<std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::singleThreaded>, std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::singleThreadedBlocking>, std::integral_constant<scheduler::ExecutionPolicy, scheduler::ExecutionPolicy::multiThreaded>>{};

    "Message passing via Scheduler"_test = []<typename SchedulerPolicy> {
        using namespace gr::testing;
        gr::Graph flow;

        auto& source  = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TestSource"}, {"n_samples_max", gr::Size_t(-1)}});
        auto& process = flow.emplaceBlock<TestBlock<float>>({{"name", "UnitTestBlock"}});
        auto& sink    = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TestSink"}, {"log_samples", false}});

        expect(eq(ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(process)));
        expect(eq(ConnectionResult::SUCCESS, flow.connect<"out">(process).to<"in">(sink)));

        gr::MsgPortOut toScheduler;
        gr::MsgPortIn  fromScheduler;
        auto           scheduler = scheduler::Simple<SchedulerPolicy::value>(std::move(flow));

        expect(eq(ConnectionResult::SUCCESS, toScheduler.connect(scheduler.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, scheduler.msgOut.connect(fromScheduler)));

        using namespace block;
        auto sendCommand = [&toScheduler](std::string_view msg, gr::message::Command cmd, std::string_view serviceName, std::string_view endPoint, property_map data) {
            using enum gr::message::Command;

            std::fflush(stderr);
            fmt::print("launch test: {}", msg);
            std::fflush(stdout);
            switch (cmd) {
            case Set: sendMessage<Set>(toScheduler, serviceName, endPoint, std::move(data), "uniqueUserID#42"); return;
            case Get: sendMessage<Get>(toScheduler, serviceName, endPoint, std::move(data), "uniqueUserID#42"); return;
            default: throw gr::exception(fmt::format("unknown/unhandled cmd {}", cmd));
            };
        };

        auto checkReply = [&fromScheduler](std::string testCase, std::size_t nReplyExpected, std::string_view serviceName, std::string_view endPoint, auto data) -> std::optional<bool> {
            std::size_t nAvailable = fromScheduler.streamReader().available();
            if (nReplyExpected == 0UZ) {
                if (nAvailable == 0UZ) {
                    return true;
                } else {
                    return false;
                }
            } else if (nAvailable == 0UZ) {
                return std::nullopt;
            }
            expect(eq(nAvailable, nReplyExpected)) << fmt::format("testCase: '{}' actual {} vs. expected {} reply messages", testCase, nAvailable, nReplyExpected);
            if (nReplyExpected == 0) {
                return false;
            }
            const Message reply = returnReplyMsg(fromScheduler);
            expect(eq(reply.clientRequestID, "uniqueUserID#42"s));
            expect(eq(reply.serviceName, serviceName));
            expect(eq(reply.endpoint, std::string(endPoint)));
            if constexpr (std::is_same_v<decltype(data), property_map>) {
                auto is_contained = [](const property_map& haystack, const property_map& needle) -> bool {
                    for (const auto& [key, val] : needle) {
                        auto it = haystack.find(key);
                        if (it == haystack.end() || it->second != val) {
                            return false; // key not found or value mismatch
                        }
                    }
                    return true;
                };

                if (!data.empty()) {
                    expect(reply.data.has_value());
                    if (!is_contained(reply.data.value(), data)) {
                        fmt::print("testCase: '{}' scheduler return reply data: {}\n contains data {}\n", testCase, reply.data.value(), data);
                        return false;
                    }
                }
            } else if constexpr (std::is_same_v<decltype(data), property_map>) {
                expect(!reply.data.has_value());
            } else {
                static_assert(gr::meta::always_false<decltype(data)>, "data type not supported");
            }
            return true;
        };

        using namespace std::literals;
        struct MessageTestCase {
            using SendCommand                                 = std::function<void()>;
            using ResultCheck                                 = std::function<std::optional<bool>()>;
            SendCommand                              cmd      = [] {};
            ResultCheck                              check    = [] { return true; };
            std::chrono::milliseconds                delay    = 1ms;   // delay after 'cmd' which the reply is being checked
            std::chrono::milliseconds                timeout  = 4s;    // time-out for the 'check' test
            std::optional<std::chrono::milliseconds> retryFor = {};    // if the test fails, should we retry it and for how long?
            bool                                     mayFail  = false; // do not assert that the last retry of the command was successful
        };

        using enum gr::message::Command;
        // clang-format off
        std::vector<MessageTestCase> commands = {
                { .cmd = [&] { fmt::print("executing failing test"); /* simulate work */ }, .check = [&] { return false; /* simulate failure */ }, .mayFail = true },
                { .cmd = [&] { fmt::print("executing passing test"); /* simulate work */ }, .check = [&] { return true; /* simulate success */ }, .delay = 500ms },
                { .cmd = [&] { fmt::print("executing test timeout"); /* simulate work */ }, .check = [&] { return std::nullopt; /* simulate time-out */ }, .timeout=100ms, .mayFail = true },
                { .cmd = [&] { sendCommand("get settings      ", Get, "UnitTestBlock", property::kSetting, { }); }, .check = [&] { return checkReply("get settings", 1UZ, process.unique_name, property::kSetting, property_map{ { "factor", 1.0f } }); }, .delay = 100ms, .retryFor = 9s },
                { .cmd = [&] { sendCommand("set settings      ", Set, "UnitTestBlock", property::kSetting, { { "factor", 42.0f } }); }, .check = [&] { return checkReply("set settings", 0UZ, "", "", property_map{ }); }, .delay = 800ms , .retryFor = 9s},
                { .cmd = [&] { sendCommand("verify settings   ", Get, "UnitTestBlock", property::kSetting, { }); }, .check = [&] { return checkReply("verify settings", 1UZ, process.unique_name, property::kSetting, property_map{ { "factor", 42.0f } }); }, .delay = 100ms, .retryFor = 9s },
                { .cmd = [&] { sendCommand("shutdown scheduler", Set, "", property::kLifeCycleState, { { "state", std::string(magic_enum::enum_name(lifecycle::State::REQUESTED_STOP)) } }); }}
        };
        // clang-format on

        fmt::println("##### starting test for scheduler {}", gr::meta::type_name<decltype(scheduler)>());
        std::fflush(stdout);

        std::thread testWorker([&scheduler, &commands] {
            fmt::println("starting testWorker.");
            std::fflush(stdout);
            while (scheduler.state() != gr::lifecycle::State::RUNNING) { // wait until scheduler is running
                std::this_thread::sleep_for(40ms);
            }
            fmt::println("scheduler is running.");
            std::fflush(stdout);

            for (auto& [command, resultCheck, delay, timeout, retryFor, mayFail] : commands) {
                auto commandStartTime = std::chrono::system_clock::now();
                bool success          = false;

                while (true) {
                    fmt::print("executing command: ");
                    std::fflush(stdout);
                    command();                          // execute the command
                    std::this_thread::sleep_for(delay); // wait for approximate time when command should be expected to be applied

                    // poll for result until timeout
                    std::optional<bool> result    = false;
                    auto                startTime = std::chrono::steady_clock::now();
                    while (std::chrono::steady_clock::now() - startTime < timeout) {
                        result = resultCheck();
                        if (result.has_value()) { // if result check passes
                            if (*result) {
                                fmt::println(" - passed.");
                                std::fflush(stdout);
                            } else {
                                fmt::println(" - failed.");
                                std::fflush(stdout);
                                // optional: throw gr::exception("command execution timed out");
                            }
                            break; // move on to the next command
                        }
                        // sleep a bit before polling again to reduce CPU usage
                        std::this_thread::sleep_for(10ms);
                    }
                    if (!result.has_value()) {
                        fmt::println(" - test timed-out after {}", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime));
                        std::fflush(stdout);
                    }

                    // The test passed successfully, no need to repeat
                    if (result.has_value() && *result) {
                        success = true;
                        break;
                    }

                    // Retry not specified, or the time for retrying has passed
                    if (!retryFor.has_value() || std::chrono::system_clock::now() - commandStartTime > *retryFor) {
                        break;
                    }
                }
                if (!mayFail) {
                    expect(success);
                }
            }
        });

        fmt::println("starting scheduler {}", gr::meta::type_name<decltype(scheduler)>());
        std::fflush(stdout);
        expect(scheduler.runAndWait().has_value());
        fmt::println("stopped scheduler {}", gr::meta::type_name<decltype(scheduler)>());

        if (testWorker.joinable()) {
            testWorker.join();
        }

        fmt::println("##### finished test for scheduler {} - produced {} samples", gr::meta::type_name<decltype(scheduler)>(), sink._nSamplesProduced);
    } | schedulingPolicies;

    "Subscribe to scheduler lifecycle messages"_test = []<typename SchedulerPolicy> {
        using namespace gr::testing;

        gr::Graph flow;

        auto& source  = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TestSource"}, {"n_samples_max", gr::Size_t(100)}});
        auto& process = flow.emplaceBlock<TestBlock<float>>({{"name", "UnitTestBlock"}});
        auto& sink    = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TestSink"}, {"log_samples", false}});

        expect(eq(ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(process)));
        expect(eq(ConnectionResult::SUCCESS, flow.connect<"out">(process).to<"in">(sink)));

        gr::MsgPortIn  fromScheduler;
        gr::MsgPortOut toScheduler;
        auto           scheduler = scheduler::Simple<SchedulerPolicy::value>(std::move(flow));
        expect(eq(ConnectionResult::SUCCESS, scheduler.msgOut.connect(fromScheduler)));
        expect(eq(ConnectionResult::SUCCESS, toScheduler.connect(scheduler.msgIn)));
        sendMessage<Command::Subscribe>(toScheduler, scheduler.unique_name, block::property::kLifeCycleState, {}, "TestClient#42");

        auto schedulerThread = std::thread([&scheduler] { scheduler.runAndWait(); });

        std::vector<std::string> receivedStates;

        bool seenStopped = false;
        auto lastSeen    = std::chrono::steady_clock::now();
        while (!seenStopped && std::chrono::steady_clock::now() - lastSeen < 1s) {
            if (fromScheduler.streamReader().available() == 0) {
                std::this_thread::sleep_for(10ms);
                continue;
            }
            const Message msg = returnReplyMsg(fromScheduler);
            expect(msg.cmd == Command::Notify);
            expect(msg.endpoint == block::property::kLifeCycleState);
            expect(msg.data.has_value());
            expect(msg.data.value().contains("state"));
            const auto state = std::get<std::string>(msg.data.value().at("state"));
            receivedStates.push_back(state);
            lastSeen = std::chrono::steady_clock::now();
            if (state == magic_enum::enum_name(lifecycle::State::STOPPED)) {
                seenStopped = true;
            }
        }

        auto name = [](lifecycle::State s) { return std::string(magic_enum::enum_name(s)); };
        expect(eq(receivedStates, std::vector{name(lifecycle::State::INITIALISED), name(lifecycle::State::RUNNING), name(lifecycle::State::REQUESTED_STOP), name(lifecycle::State::STOPPED)}));

        schedulerThread.join();
    } | schedulingPolicies;

    "Settings handling via scheduler"_test = []<typename SchedulerPolicy> {
        // ensure settings can be modified and setting change updates can be subscribed to when connected via the scheduler
        using namespace gr::basic;
        using namespace gr::testing;

        gr::Graph flow;

        auto& source    = flow.emplaceBlock<ClockSource<float>>({{"n_samples_max", gr::Size_t(0)}});
        auto& testBlock = flow.emplaceBlock<TestBlock<float>>({{"factor", 42.f}});
        auto& sink      = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", false}});

        expect(eq(ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(testBlock)));
        expect(eq(ConnectionResult::SUCCESS, flow.connect<"out">(testBlock).to<"in">(sink)));

        auto scheduler = scheduler::Simple<SchedulerPolicy::value>(std::move(flow));

        gr::MsgPortIn  fromScheduler;
        gr::MsgPortOut toScheduler;
        expect(eq(ConnectionResult::SUCCESS, scheduler.msgOut.connect(fromScheduler)));
        expect(eq(ConnectionResult::SUCCESS, toScheduler.connect(scheduler.msgIn)));
        sendMessage<Command::Subscribe>(toScheduler, "", block::property::kStagedSetting, {}, "TestClient");

        auto client = std::thread([&fromScheduler, &toScheduler, blockName = testBlock.unique_name, schedulerName = scheduler.unique_name] {
            sendMessage<Command::Set>(toScheduler, blockName, block::property::kStagedSetting, {{"factor", 43.0f}});
            bool       seenUpdate = false;
            const auto startTime  = std::chrono::steady_clock::now();
            auto       isExpired  = [&startTime] { return std::chrono::steady_clock::now() - startTime > 3s; };
            bool       expired    = false;
            while (!seenUpdate && !expired) {
                expired = isExpired();
                while (fromScheduler.streamReader().available() == 0 && !expired) {
                    expired = isExpired();
                    std::this_thread::sleep_for(10ms);
                }
                if (!expired) {
                    const auto msg = returnReplyMsg(fromScheduler);
                    if (msg.serviceName == blockName && msg.endpoint == block::property::kStagedSetting) {
                        expect(msg.data.has_value());
                        expect(msg.data.value().contains("factor"));
                        const auto factor = std::get<float>(msg.data.value().at("factor"));
                        expect(eq(factor, 43.0f));
                        seenUpdate = true;
                    }
                }
            }
            expect(seenUpdate);
            sendMessage<Command::Set>(toScheduler, schedulerName, block::property::kLifeCycleState, {{"state", std::string(magic_enum::enum_name(lifecycle::State::REQUESTED_STOP))}});
        });

        auto schedulerThread = std::thread([&scheduler] { scheduler.runAndWait(); });

        client.join();
        while (source.state() != lifecycle::State::STOPPED) {
            std::this_thread::sleep_for(10ms);
        }
        schedulerThread.join();
    } | schedulingPolicies;
};

inline Error generateError(std::string_view msg) { return Error(msg); }

const boost::ut::suite messageFormatter = [] {
    using namespace boost::ut;
    using namespace std::string_literals;
    fmt::println("\n\nmessageFormatter test suite (explicitly verbose):");

    "message::Command-Formatter"_test = [] {
        using enum gr::message::Command;
        magic_enum::enum_for_each<Command>([](Command value) { expect(eq(fmt::format("{}", value), std::string(magic_enum::enum_name(value)))); });

        expect(eq(gr::message::commandName<Set>(), std::string(magic_enum::enum_name<Set>())));
    };

    "Message-Formatter"_test = [] {
        using enum gr::message::Command;
        auto loc = fmt::format("{}", Message{.cmd = Set, .serviceName = "MyCustomBlock", .endpoint = "<propertyName>", .data = property_map{{"key", "value"}}, .rbac = "<rbac token>"});
        fmt::println("Message formatter test: {}", loc);
        expect(ge(loc.size(), 0UZ));
    };

    "Error-Formatter"_test = [] {
        using enum gr::message::Command;
        auto loc1 = fmt::format("{}", generateError("ErrorMsg"));
        fmt::println("Error formatter test: {}", loc1);
        expect(ge(loc1.size(), 0UZ));

        auto loc2 = fmt::format("{:s}", generateError("ErrorMsg"));
        fmt::println("Error formatter test: {}", loc2);
        expect(ge(loc2.size(), 0UZ));

        auto loc3 = fmt::format("{:f}", generateError("ErrorMsg"));
        fmt::println("Error formatter test: {}", loc3);
        expect(ge(loc3.size(), 0UZ));

        auto loc4 = fmt::format("{:t}", generateError("ErrorMsg"));
        fmt::println("Error formatter test: {}", loc4);
        expect(ge(loc4.size(), 0UZ));
    };
};

const boost::ut::suite testExceptions = [] {
    using namespace boost::ut;
    using namespace std::string_literals;

    "test gr::exception basic functionality"_test = [] {
        gr::exception ex("test exception");
        expect(eq(std::string_view(ex.what()).substr(0, 14), "test exception"sv));
    };

    "test Error class basic functionality"_test = [] {
        const std::source_location location = std::source_location::current();
        Error                      error("test error");
        expect(eq(error.message, "test error"sv));
        expect(eq(error.methodName(), location.function_name()));
        expect(ge(error.isoTime().size(), 10UZ));
        expect(ge(error.srcLoc().size(), 10UZ));
    };

    "test Error class with std::exception"_test = [] {
        const std::source_location location = std::source_location::current();
        std::exception             stdEx;
        Error                      error(stdEx);
        expect(eq(error.message, "std::exception"sv)) << "Error message should fall back to 'std::exception'";

        gr::exception grEx("custom exception message");
        Error         errorFromGrEx(grEx);
        expect(eq(errorFromGrEx.message, "custom exception message"sv));
        expect(eq(errorFromGrEx.methodName(), location.function_name()));
        expect(ge(errorFromGrEx.isoTime().size(), 10UZ));
        expect(ge(errorFromGrEx.srcLoc().size(), 10UZ));
    };

    "test Error class with gr::exception"_test = [] {
        const std::source_location location = std::source_location::current();
        gr::exception              grEx("test gr::exception");
        Error                      error(grEx);
        expect(eq(error.message, "test gr::exception"sv));
        expect(eq(error.methodName(), location.function_name()));
        expect(ge(error.isoTime().size(), 10UZ));
        expect(ge(error.srcLoc().size(), 10UZ));
    };
};

int main() { /* tests are statically executed */ }
