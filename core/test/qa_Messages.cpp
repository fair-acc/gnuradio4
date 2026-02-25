#include <boost/ut.hpp>

#include "message_utils.hpp"

#include "gnuradio-4.0/Block.hpp"
#include "gnuradio-4.0/Message.hpp"
#include <gnuradio-4.0/Scheduler.hpp>
#include <gnuradio-4.0/basic/ClockSource.hpp>
#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/testing/TagMonitors.hpp>

#include <magic_enum.hpp>
#include <magic_enum_utility.hpp>

#include <optional>

#include <gnuradio-4.0/meta/UnitTestHelper.hpp>

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
        std::println("called Block's {} unmatchedPropertyHandler({}, {})", this->name, propertyName, msg);
        if (msg.endpoint == "Unknown") {
            // special case for the unit-test to mimic unknown properties
            return std::nullopt;
        }

        if (msg.endpoint == "CustomEndpoint") {
            sendMessage<message::Command::Notify>(this->msgOut, "", "custom_reply_kind", property_map{{"key", "testReplyData"}});
            return std::nullopt;
        }

        throw gr::exception(std::format("Blocks {} unsupported property in unmatchedPropertyHandler({}, {})", this->name, propertyName, msg));
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
    using namespace gr::test;

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
                const Message reply = consumeFirstReply(fromBlock);
                expect(reply.cmd == Final) << std::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
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
                const Message reply = consumeFirstReply(fromBlock);
                expect(reply.cmd == Final) << std::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
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
                const Message heartbeat1 = consumeFirstReply(fromBlock);
                expect(heartbeat1.cmd == Notify) << std::format("mismatch between heartbeat1.cmd = {} and expected {} command", heartbeat1.cmd, Notify);
                expect(eq(heartbeat1.endpoint, std::string(block::property::kHeartbeat)));
                expect(heartbeat1.data.has_value());
                expect(heartbeat1.data.value().contains("heartbeat"));

                // unsubscribe
                sendMessage<Unsubscribe>(toBlock, "", block::property::kHeartbeat, {}, "client#42");
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";
                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "should receive heartbeat";
                const Message heartbeat2 = consumeFirstReply(fromBlock);
                expect(heartbeat2.cmd == Notify) << std::format("mismatch between heartbeat2.cmd = {} and expected {} command", heartbeat2.cmd, Notify);
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
                const Message reply = consumeFirstReply(fromBlock);
                expect(reply.cmd == Final) << std::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
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
                const Message reply = consumeFirstReply(fromBlock);
                expect(reply.cmd == Final) << std::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
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
                const Message reply = consumeFirstReply(fromBlock);
                expect(reply.cmd == Final) << std::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
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
                const Message reply = consumeFirstReply(fromBlock);
                expect(reply.cmd == Final) << std::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
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
                const Message reply = consumeFirstReply(fromBlock);
                expect(reply.cmd == Final) << std::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
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
                const Message reply = consumeFirstReply(fromBlock);
                expect(reply.cmd == Final) << std::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
                expect(eq(reply.serviceName, unitTestBlock.unique_name));
                expect(eq(reply.endpoint, std::string(block::property::kSetting)));
                expect(reply.data.has_value());
                expect(!reply.data.value().empty());
                expect(reply.data.value().contains("factor"));
                expect(eq(1, gr::test::get_value_or_fail<int>(reply.data.value().at("factor"))));
            };

            "get - StagedSettings"_test = [&] {
                sendMessage<Get>(toBlock, "" /* serviceName */, block::property::kStagedSetting /* endpoint */, {} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "didn't receive staged setting reply message";
                const Message reply = consumeFirstReply(fromBlock);
                expect(reply.cmd == Final) << std::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
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
                expect(eq(42, gr::test::get_value_or_fail<int>(stagedSettings.at("factor"))));

                // setting staged setting via staged setting (N.B. non-real-time <-> real-time setting decoupling
                sendMessage<Set>(toBlock, "" /* serviceName */, block::property::kSetting /* endpoint */, {{"factor", 43}} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 0UZ)) << "should not receive a reply";
                stagedSettings = unitTestBlock.settings().stagedParameters();
                expect(stagedSettings.contains("factor"));
                expect(eq(43, gr::test::get_value_or_fail<int>(stagedSettings.at("factor"))));
            };

            "set - StagedSettings, property_map field"_test = [&] {
                auto makeUiConstraints = [](float x, float y) { return gr::property_map{{"x", x}, {"y", y}}; };
                sendMessage<Set>(toBlock, "" /* serviceName */, block::property::kStagedSetting /* endpoint */, {{"ui_constraints", makeUiConstraints(42, 6)}} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 0UZ)) << "should not receive a reply";
                property_map stagedSettings = unitTestBlock.settings().stagedParameters();
                expect(stagedSettings.contains("ui_constraints"));
                expect(eq(42.f, gr::test::get_value_or_fail<float>(gr::test::get_value_or_fail<gr::property_map>(stagedSettings.at("ui_constraints"))["x"])));

                // setting staged setting via staged setting (N.B. non-real-time <-> real-time setting decoupling
                sendMessage<Set>(toBlock, "" /* serviceName */, block::property::kStagedSetting /* endpoint */, {{"ui_constraints", makeUiConstraints(43, 7)}} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 0UZ)) << "should not receive a reply";
                stagedSettings = unitTestBlock.settings().stagedParameters();
                expect(stagedSettings.contains("ui_constraints"));
                expect(eq(43.f, gr::test::get_value_or_fail<float>(gr::test::get_value_or_fail<gr::property_map>(stagedSettings.at("ui_constraints"))["x"])));

                expect(unitTestBlock.settings().applyStagedParameters().forwardParameters.empty());
                expect(eq(43.f, gr::test::get_value_or_fail<float>(unitTestBlock.ui_constraints["x"])));
            };
        };

        "Block<T>-level active context tests"_test = [] {
            gr::MsgPortOut toBlock;
            TestBlock<int> unitTestBlock(property_map{{"name", "UnitTestBlock"}});
            unitTestBlock.init(unitTestBlock.progress);
            gr::MsgPortIn fromBlock;

            expect(eq(ConnectionResult::SUCCESS, toBlock.connect(unitTestBlock.msgIn)));
            expect(eq(ConnectionResult::SUCCESS, unitTestBlock.msgOut.connect(fromBlock)));

            "get all contexts default - w/o explicit serviceName"_test = [&] {
                sendMessage<Get>(toBlock, "" /* serviceName */, block::property::kSettingsContexts /* endpoint */, {} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "didn't receive reply message";
                const std::map<pmt::Value, std::vector<gr::SettingsBase::CtxSettingsPair>, settings::PMTCompare> allStored = unitTestBlock.settings().getStoredAll();
                expect(eq(allStored.size(), 1UZ));
                expect(allStored.contains(""));

                const Message reply = consumeFirstReply(fromBlock);
                expect(reply.cmd == Final) << std::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
                expect(eq(reply.serviceName, unitTestBlock.unique_name));
                expect(eq(reply.clientRequestID, ""s));
                expect(eq(reply.endpoint, std::string(block::property::kSettingsContexts)));
                expect(reply.data.has_value());
                expect(reply.data.value().contains("contexts"));
                auto contexts = reply.data.value().at("contexts").value_or(Tensor<pmt::Value>{});
                expect(eq(contexts.size(), 1UZ));
                expect(eq(contexts[0], ""s));
            };

            "get active context - w/o explicit serviceName"_test = [&] {
                sendMessage<Get>(toBlock, "" /* serviceName */, block::property::kActiveContext /* endpoint */, {} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "didn't receive reply message";
                const Message reply = consumeFirstReply(fromBlock);
                expect(reply.cmd == Final) << std::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
                expect(eq(reply.serviceName, unitTestBlock.unique_name));
                expect(eq(reply.clientRequestID, ""s));
                expect(eq(reply.endpoint, std::string(block::property::kActiveContext)));
                expect(reply.data.has_value());
                expect(reply.data.value().contains(gr::tag::CONTEXT.shortKey()));
                expect(eq(""s, gr::test::get_value_or_fail<std::string>(reply.data.value().at(gr::tag::CONTEXT.shortKey()))));
            };

            "create active test_context - w/o explicit serviceName"_test = [&] {
                sendMessage<Set>(toBlock, "" /* serviceName */, block::property::kSettingsCtx /* endpoint */, {{gr::tag::CONTEXT.shortKey(), "test_context"}, {gr::tag::CONTEXT_TIME.shortKey(), static_cast<gr::Size_t>(1UZ)}} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "didn't receive reply message";
                const auto allStored = unitTestBlock.settings().getStoredAll();
                expect(allStored.contains("test_context"s));

                const Message reply = consumeFirstReply(fromBlock);
                expect(reply.cmd == Final) << std::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
                expect(eq(reply.serviceName, unitTestBlock.unique_name));
                expect(eq(reply.clientRequestID, ""s));
                expect(eq(reply.endpoint, std::string(block::property::kSettingsCtx)));
                expect(reply.data.has_value());
                expect(reply.data.value().contains("failed_to_set"));
                auto failed_to_set = gr::test::get_value_or_fail<gr::property_map>(reply.data.value().at("failed_to_set"));
                expect(failed_to_set.empty());
            };

            "create active new_context - w/o explicit serviceName"_test = [&] {
                sendMessage<Set>(toBlock, "" /* serviceName */, block::property::kSettingsCtx /* endpoint */, {{gr::tag::CONTEXT.shortKey(), "new_context"}, {gr::tag::CONTEXT_TIME.shortKey(), static_cast<gr::Size_t>(2UZ)}} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "didn't receive reply message";
                const auto allStored = unitTestBlock.settings().getStoredAll();
                expect(allStored.contains("new_context"s));

                const Message reply = consumeFirstReply(fromBlock);
                expect(reply.cmd == Final) << std::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
                expect(eq(reply.serviceName, unitTestBlock.unique_name));
                expect(eq(reply.clientRequestID, ""s));
                expect(eq(reply.endpoint, std::string(block::property::kSettingsCtx)));
                expect(reply.data.has_value());
                expect(reply.data.value().contains("failed_to_set"));
                auto failed_to_set = gr::test::get_value_or_fail<gr::property_map>(reply.data.value().at("failed_to_set"));
                expect(failed_to_set.empty());
            };

            "activate new_context - w/o explicit serviceName"_test = [&] {
                sendMessage<Set>(toBlock, "" /* serviceName */, block::property::kActiveContext /* endpoint */, {{gr::tag::CONTEXT.shortKey(), "new_context"}, {gr::tag::CONTEXT_TIME.shortKey(), static_cast<gr::Size_t>(2UZ)}} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "didn't receive reply message";
                std::string activeContext = gr::test::get_value_or_fail<std::string>(unitTestBlock.settings().activeContext().context);
                expect(eq("new_context"s, activeContext));

                const Message reply = consumeFirstReply(fromBlock);
                expect(reply.cmd == Final) << std::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
                expect(eq(reply.serviceName, unitTestBlock.unique_name));
                expect(eq(reply.clientRequestID, ""s));
                expect(eq(reply.endpoint, std::string(block::property::kActiveContext)));
                expect(reply.data.has_value());
                expect(reply.data.value().contains(gr::tag::CONTEXT.shortKey()));
                expect(reply.data.value().contains(gr::tag::CONTEXT_TIME.shortKey()));
                expect(eq("new_context"s, gr::test::get_value_or_fail<std::string>(reply.data.value().at(gr::tag::CONTEXT.shortKey()))));
            };

            "get active new_context - w/o explicit serviceName"_test = [&] {
                sendMessage<Get>(toBlock, "" /* serviceName */, block::property::kActiveContext /* endpoint */, {} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "didn't receive reply message";
                const Message reply = consumeFirstReply(fromBlock);
                expect(reply.cmd == Final) << std::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
                expect(eq(reply.serviceName, unitTestBlock.unique_name));
                expect(eq(reply.clientRequestID, ""s));
                expect(eq(reply.endpoint, std::string(block::property::kActiveContext)));
                expect(reply.data.has_value());
                expect(reply.data.value().contains(gr::tag::CONTEXT.shortKey()));
                expect(reply.data.value().contains(gr::tag::CONTEXT_TIME.shortKey()));
                expect(eq("new_context"s, gr::test::get_value_or_fail<std::string>(reply.data.value().at(gr::tag::CONTEXT.shortKey()))));
            };

            "get all contexts - w/o explicit serviceName"_test = [&] {
                sendMessage<Get>(toBlock, "" /* serviceName */, block::property::kSettingsContexts /* endpoint */, {} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "didn't receive reply message";
                const std::map<pmt::Value, std::vector<gr::SettingsBase::CtxSettingsPair>, settings::PMTCompare> allStored = unitTestBlock.settings().getStoredAll();
                expect(eq(allStored.size(), 3UZ));
                expect(allStored.contains(""s));
                expect(allStored.contains("new_context"s));
                expect(allStored.contains("test_context"s));

                const Message reply = consumeFirstReply(fromBlock);
                expect(reply.cmd == Final) << std::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
                expect(eq(reply.serviceName, unitTestBlock.unique_name));
                expect(eq(reply.clientRequestID, ""s));
                expect(eq(reply.endpoint, std::string(block::property::kSettingsContexts)));
                expect(reply.data.has_value());
                expect(reply.data.value().contains("contexts"));
                expect(reply.data.value().contains("times"));
                auto contexts = reply.data.value().at("contexts").value_or(Tensor<pmt::Value>());
                auto times    = reply.data.value().at("times").value_or(Tensor<std::uint64_t>());
                expect(eq(contexts.size(), 3UZ));
                expect(eq(times.size(), 3UZ));
                expect(eq(contexts, std::vector<pmt::Value>{"", "new_context", "test_context"}));
                // We do not check the default context as it's time is now()
                expect(eq(times[1], allStored.at("new_context")[0].context.time)); // We need internal time since wasm change our time
                expect(eq(times[2], allStored.at("test_context")[0].context.time));
            };

            "remove new_context - w/o explicit serviceName"_test = [&] {
                //  We need internal time since wasm change our time
                const std::map<pmt::Value, std::vector<gr::SettingsBase::CtxSettingsPair>, settings::PMTCompare> allStored = unitTestBlock.settings().getStoredAll();
                expect(eq(allStored.size(), 3UZ));
                const std::uint64_t internalTimeForWasm = allStored.at("new_context")[0].context.time;

                sendMessage<Disconnect>(toBlock, "" /* serviceName */, block::property::kSettingsCtx /* endpoint */, {{gr::tag::CONTEXT.shortKey(), "new_context"}, {gr::tag::CONTEXT_TIME.shortKey(), internalTimeForWasm}} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "didn't receive reply message";
                const Message reply         = consumeFirstReply(fromBlock);
                std::string   activeContext = gr::test::get_value_or_fail<std::string>(unitTestBlock.settings().activeContext().context);
                expect(eq(""s, activeContext));
            };

            "get active back to default context '' - w/o explicit serviceName"_test = [&] {
                sendMessage<Get>(toBlock, "" /* serviceName */, block::property::kActiveContext /* endpoint */, {} /* data  */);
                expect(nothrow([&] { unitTestBlock.processScheduledMessages(); })) << "manually execute processing of messages";

                expect(eq(fromBlock.streamReader().available(), 1UZ)) << "didn't receive reply message";
                const Message reply = consumeFirstReply(fromBlock);
                expect(reply.cmd == Final) << std::format("mismatch between reply.cmd = {} and expected {} command", reply.cmd, Final);
                expect(eq(reply.serviceName, unitTestBlock.unique_name));
                expect(eq(reply.clientRequestID, ""s));
                expect(eq(reply.endpoint, std::string(block::property::kActiveContext)));
                expect(reply.data.has_value());
                expect(reply.data.value().contains(gr::tag::CONTEXT.shortKey()));
                expect(eq(""s, gr::test::get_value_or_fail<std::string>(reply.data.value().at(gr::tag::CONTEXT.shortKey()))));
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
        const Message heartbeat1 = consumeFirstReply(fromBlock);
        expect(heartbeat1.cmd == Notify) << std::format("mismatch between heartbeat1.cmd = {} and expected {} command", heartbeat1.cmd, Notify);
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

        gr::MsgPortOut                                toScheduler;
        gr::MsgPortIn                                 fromScheduler;
        gr::scheduler::Simple<SchedulerPolicy::value> scheduler;
        if (auto ret = scheduler.exchange(std::move(flow)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }

        expect(eq(ConnectionResult::SUCCESS, toScheduler.connect(scheduler.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, scheduler.msgOut.connect(fromScheduler)));

        using namespace block;
        auto sendCommand = [&toScheduler](std::string_view msg, gr::message::Command cmd, std::string_view serviceName, std::string_view endPoint, property_map data) {
            using enum gr::message::Command;

            std::fflush(stderr);
            std::print("launch test: {}", msg);
            std::fflush(stdout);
            switch (cmd) {
            case Set: sendMessage<Set>(toScheduler, serviceName, endPoint, std::move(data), "uniqueUserID#42"); return;
            case Get: sendMessage<Get>(toScheduler, serviceName, endPoint, std::move(data), "uniqueUserID#42"); return;
            default: throw gr::exception(std::format("unknown/unhandled cmd {}", cmd));
            };
        };

        auto checkReply = [](auto& fromSchedulerLoc, std::string testCase, std::size_t nReplyExpected, std::string_view serviceName, std::string_view endPoint, auto data, std::source_location location = std::source_location::current()) -> std::optional<bool> {
            using namespace boost::ut;

            std::size_t nAvailable = fromSchedulerLoc.streamReader().available();
            if (nReplyExpected == 0UZ) {
                if (nAvailable == 0UZ) {
                    return true;
                } else {
                    return false;
                }
            } else if (nAvailable == 0UZ) {
                return std::nullopt;
            }
            expect(ge(nAvailable, nReplyExpected)) << std::format("testCase: '{}' actual {} vs. expected {} reply messages, at {}:{}", testCase, nAvailable, nReplyExpected, location.file_name(), location.line());
            if (nReplyExpected == 0) {
                return false;
            }
            const Message reply = consumeFirstReply(fromSchedulerLoc);
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
                        std::print("testCase: '{}' scheduler return reply data: {}\n contains data {}\n", testCase, reply.data.value(), data);
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
                { .cmd = [&] { std::print("executing failing test"); /* simulate work */ }, .check = [&] { return false; /* simulate failure */ }, .mayFail = true },
                { .cmd = [&] { std::print("executing passing test"); /* simulate work */ }, .check = [&] { return true; /* simulate success */ }, .delay = 500ms },
                { .cmd = [&] { std::print("executing test timeout"); /* simulate work */ }, .check = [&] { return std::nullopt; /* simulate time-out */ }, .timeout=100ms, .mayFail = true },
                { .cmd = [&] { sendCommand("get settings      ", Get, "UnitTestBlock", property::kSetting, { }); }, .check = [&] { return checkReply(fromScheduler, "get settings", 1UZ, process.unique_name, property::kSetting, property_map{ { "factor", 1.0f } }); }, .delay = 100ms, .retryFor = 9s },
                { .cmd = [&] { sendCommand("set settings      ", Set, "UnitTestBlock", property::kSetting, { { "factor", 42.0f} }); }, .check = [&] { return checkReply(fromScheduler, "set settings", 0UZ, "", "", property_map{ }); }, .delay = 800ms , .retryFor = 9s},
                { .cmd = [&] { sendCommand("verify settings   ", Get, "UnitTestBlock", property::kSetting, { }); }, .check = [&] { return checkReply(fromScheduler, "verify settings", 1UZ, process.unique_name, property::kSetting, property_map{ { "factor", 42.0f } }); }, .delay = 100ms, .retryFor = 9s },
                { .cmd = [&] { sendCommand("shutdown scheduler", Set, "", property::kLifeCycleState, { { "state", std::string(magic_enum::enum_name(lifecycle::State::REQUESTED_STOP)) } }); }}
        };
        // clang-format on

        std::println("##### starting test for scheduler {}", gr::meta::type_name<decltype(scheduler)>());
        std::fflush(stdout);

        auto testWorker = gr::test::thread_pool::execute("test worker", [&scheduler, &commands] {
            std::println("starting testWorker.");
            std::fflush(stdout);
            while (scheduler.state() != gr::lifecycle::State::RUNNING) { // wait until scheduler is running
                std::this_thread::sleep_for(40ms);
            }
            std::println("scheduler is running.");
            std::fflush(stdout);

            for (auto& [command, resultCheck, delay, timeout, retryFor, mayFail] : commands) {
                auto commandStartTime = std::chrono::system_clock::now();
                bool success          = false;

                while (true) {
                    std::print("executing command: ");
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
                                std::println(" - passed.");
                                std::fflush(stdout);
                            } else {
                                std::println(" - failed.");
                                std::fflush(stdout);
                                // optional: throw gr::exception("command execution timed out");
                            }
                            break; // move on to the next command
                        }
                        // sleep a bit before polling again to reduce CPU usage
                        std::this_thread::sleep_for(10ms);
                    }
                    if (!result.has_value()) {
                        std::println(" - test timed-out after {}", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime));
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

        std::println("starting scheduler {}", gr::meta::type_name<decltype(scheduler)>());
        std::fflush(stdout);
        expect(scheduler.runAndWait().has_value());
        std::println("stopped scheduler {}", gr::meta::type_name<decltype(scheduler)>());

        testWorker.wait();

        std::println("##### finished test for scheduler {} - produced {} samples", gr::meta::type_name<decltype(scheduler)>(), sink._nSamplesProduced);
    } | schedulingPolicies;

    "Subscribe to scheduler lifecycle messages"_test = []<typename SchedulerPolicy> {
        using namespace gr::testing;

        gr::Graph flow;

        auto& source  = flow.emplaceBlock<TagSource<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TestSource"}, {"n_samples_max", gr::Size_t(100)}});
        auto& process = flow.emplaceBlock<TestBlock<float>>({{"name", "UnitTestBlock"}});
        auto& sink    = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"name", "TestSink"}, {"log_samples", false}});

        expect(eq(ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(process)));
        expect(eq(ConnectionResult::SUCCESS, flow.connect<"out">(process).to<"in">(sink)));

        gr::MsgPortIn                                 fromScheduler;
        gr::MsgPortOut                                toScheduler;
        gr::scheduler::Simple<SchedulerPolicy::value> scheduler;
        if (auto ret = scheduler.exchange(std::move(flow)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }
        expect(eq(ConnectionResult::SUCCESS, scheduler.msgOut.connect(fromScheduler)));
        expect(eq(ConnectionResult::SUCCESS, toScheduler.connect(scheduler.msgIn)));
        sendMessage<Command::Subscribe>(toScheduler, scheduler.unique_name, block::property::kLifeCycleState, {}, "TestClient#42");

        auto threadHandle = gr::test::thread_pool::executeScheduler("qa_Messages::scheduler", scheduler);

        std::vector<std::string> receivedStates;

        bool seenStopped = false;
        auto lastSeen    = std::chrono::steady_clock::now();
        while (!seenStopped && std::chrono::steady_clock::now() - lastSeen < 1s) {
            if (fromScheduler.streamReader().available() == 0) {
                std::this_thread::sleep_for(10ms);
                continue;
            }
            const Message msg = consumeFirstReply(fromScheduler);
            expect(msg.cmd == Command::Notify);
            expect(msg.endpoint == block::property::kLifeCycleState);
            expect(msg.data.has_value());
            expect(msg.data.value().contains("state"));
            const auto state = gr::test::get_value_or_fail<std::string>(msg.data.value().at("state"));
            receivedStates.push_back(state);
            lastSeen = std::chrono::steady_clock::now();
            if (state == magic_enum::enum_name(lifecycle::State::STOPPED)) {
                seenStopped = true;
            }
        }

        auto name = [](lifecycle::State s) { return std::string(magic_enum::enum_name(s)); };
        expect(eq(receivedStates, std::vector{name(lifecycle::State::INITIALISED), name(lifecycle::State::RUNNING), name(lifecycle::State::REQUESTED_STOP), name(lifecycle::State::STOPPED)}));

        threadHandle.wait();
    } | schedulingPolicies;

    "Settings handling via scheduler"_test = []<typename SchedulerPolicy> {
        // ensure settings can be modified and setting change updates can be subscribed to when connected via the scheduler
        using namespace gr::basic;
        using namespace gr::testing;

        gr::Graph flow;

        auto& source    = flow.emplaceBlock<ClockSource<float>>({{"n_samples_max", gr::Size_t(0)}});
        auto& testBlock = flow.emplaceBlock<TestBlock<float>>({{"factor", 42.f}, {"ui_constraints", gr::property_map{{"x", 42.f}, {"y", 6.f}}}});
        auto& sink      = flow.emplaceBlock<TagSink<float, ProcessFunction::USE_PROCESS_ONE>>({{"log_samples", false}});

        expect(eq(ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(testBlock)));
        expect(eq(ConnectionResult::SUCCESS, flow.connect<"out">(testBlock).to<"in">(sink)));

        gr::scheduler::Simple<SchedulerPolicy::value> scheduler;
        if (auto ret = scheduler.exchange(std::move(flow)); !ret) {
            throw std::runtime_error(std::format("failed to initialize scheduler: {}", ret.error()));
        }

        gr::MsgPortIn  fromScheduler;
        gr::MsgPortOut toScheduler;
        expect(eq(ConnectionResult::SUCCESS, scheduler.msgOut.connect(fromScheduler)));
        expect(eq(ConnectionResult::SUCCESS, toScheduler.connect(scheduler.msgIn)));
        sendMessage<Command::Subscribe>(toScheduler, "", block::property::kStagedSetting, {}, "TestClient");

        // dispatch client to IO pool: the client polls with sleep_for (IO-bound),
        // and the CPU pool is occupied by scheduler workers in multiThreaded mode
        std::promise<void> clientPromise;
        auto               client = clientPromise.get_future();
        gr::thread_pool::Manager::defaultIoPool()->execute([&fromScheduler, &toScheduler, &testBlock, blockName = testBlock.unique_name, schedulerName = scheduler.unique_name, clientPromise = std::move(clientPromise)] mutable {
            gr::thread_pool::thread::setThreadName("qa_Mess::Client");
            sendMessage<Command::Set>(toScheduler, blockName, block::property::kStagedSetting,
                {{"factor", 43.0f},           //
                    {"name", "My New Name"s}, //
                    {"ui_constraints", gr::property_map{{"x", 43.f}, {"y", 7.f}}}});
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
                    const auto msg = consumeFirstReply(fromScheduler);
                    if (msg.serviceName == blockName && msg.endpoint == block::property::kStagedSetting) {
                        expect(msg.data.has_value());
                        expect(msg.data.value().contains("factor"));
                        const auto factor = gr::test::get_value_or_fail<float>(msg.data.value().at("factor"));
                        expect(eq(factor, 43.0f));

                        expect(msg.data.value().contains("name"));
                        const auto name = gr::test::get_value_or_fail<std::string>(msg.data.value().at("name"));
                        expect(eq(name, "My New Name"s));

                        expect(msg.data.value().contains("ui_constraints"));
                        const auto uiConstraints = gr::test::get_value_or_fail<gr::property_map>(msg.data.value().at("ui_constraints"));
                        expect(uiConstraints == gr::property_map{{"x", 43.f}, {"y", 7.f}});

                        expect(testBlock.settings().applyStagedParameters().forwardParameters.empty());
                        expect(eq(gr::test::get_value_or_fail<float>(testBlock.settings().get("factor").value()), 43.0f));
                        expect(eq(gr::test::get_value_or_fail<std::string>(testBlock.settings().get("name"s).value()), "My New Name"s));
                        expect(eq(gr::test::get_value_or_fail<float>(gr::test::get_value_or_fail<gr::property_map>(testBlock.settings().get("ui_constraints").value())["x"]), 43.f));
                        expect(eq(gr::test::get_value_or_fail<float>(gr::test::get_value_or_fail<gr::property_map>(testBlock.settings().get("ui_constraints").value())["y"]), 7.f));

                        seenUpdate = true;
                    }
                }
            }
            expect(seenUpdate);
            sendMessage<Command::Set>(toScheduler, schedulerName, block::property::kLifeCycleState, {{"state", std::string(magic_enum::enum_name(lifecycle::State::REQUESTED_STOP))}});
            clientPromise.set_value();
        });

        auto threadHandle = gr::test::thread_pool::executeScheduler("qa_Messages::scheduler", scheduler);

        client.wait();
        while (source.state() != lifecycle::State::STOPPED) {
            std::this_thread::sleep_for(10ms);
        }
        threadHandle.wait();
    } | schedulingPolicies;
};

inline Error generateError(std::string_view msg) { return Error(msg); }

const boost::ut::suite messageFormatter = [] {
    using namespace boost::ut;
    using namespace std::string_literals;
    std::println("\n\nmessageFormatter test suite (explicitly verbose):");

    "message::Command-Formatter"_test = [] {
        using enum gr::message::Command;
        magic_enum::enum_for_each<Command>([](Command value) { expect(eq(std::format("{}", value), std::string(magic_enum::enum_name(value)))); });

        expect(eq(gr::message::commandName<Set>(), std::string(magic_enum::enum_name<Set>())));
    };

    "Message-Formatter"_test = [] {
        using enum gr::message::Command;
        auto loc = std::format("{}", Message{.cmd = Set, .serviceName = "MyCustomBlock", .endpoint = "<propertyName>", .data = property_map{{"key", "value"}}, .rbac = "<rbac token>"});
        std::println("Message formatter test: {}", loc);
        expect(ge(loc.size(), 0UZ));
    };

    "Error-Formatter"_test = [] {
        using enum gr::message::Command;
        auto loc1 = std::format("{}", generateError("ErrorMsg"));
        std::println("Error formatter test: {}", loc1);
        expect(ge(loc1.size(), 0UZ));

        auto loc2 = std::format("{:s}", generateError("ErrorMsg"));
        std::println("Error formatter test: {}", loc2);
        expect(ge(loc2.size(), 0UZ));

        auto loc3 = std::format("{:f}", generateError("ErrorMsg"));
        std::println("Error formatter test: {}", loc3);
        expect(ge(loc3.size(), 0UZ));

        auto loc4 = std::format("{:t}", generateError("ErrorMsg"));
        std::println("Error formatter test: {}", loc4);
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
