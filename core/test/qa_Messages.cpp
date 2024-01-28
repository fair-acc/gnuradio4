#include <boost/ut.hpp>

#include <gnuradio-4.0/Scheduler.hpp>
#include <magic_enum.hpp>
#include <magic_enum_utility.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

using namespace std::chrono_literals;
using namespace std::string_literals;

template<typename T>
struct SimpleSource : gr::Block<SimpleSource<T>> {
    T              value;
    gr::PortOut<T> out;

    T
    processOne() {
        return value;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(SimpleSource, out);

template<typename T>
struct CoutSink : gr::Block<CoutSink<T>> {
    gr::PortIn<T> in;

    void
    processOne(T value) {
        std::cout << value << "  ";
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(CoutSink, in);

// TODO:
//  - needed: sources, basic process, sinks
//            message senders, message checkers/receivers
//  - test: one block sends another a message in the same graph
//  - test: block from outsie sends a message to a block inside the graph
//  - test: block from inside of a graph sends a message to outside
//  - test: one block sends a multicast message to everyone in a graph
//  - test: control messages to start and stop
//  - test: settings update messages
//
//  - tests: custom message ports
//  - processMessages overloads with 'different' message port types
//
//  - test: message processing when scheduler is stopped
//
//  idea: tests can use messages to trigger checks

struct EventChecker {
    struct Event {
        std::string who;
        std::string what;
    };

    std::vector<Event> _recordedEvents;
    std::mutex         _recordedEventsMutex;

    void
    addEvent(const std::string &who, const std::string &what) {
        fmt::print(">> Event added to the record {} -> {}\n", who, what);
        _recordedEvents.emplace_back(who, what);
    }

    bool
    checkEvents(std::string who, const std::vector<std::string> &whats) {
        return std::ranges::equal(_recordedEvents                                                                       //
                                          | std::views::filter([&who](const Event &event) { return who == event.who; }) //
                                          | std::views::transform(&Event::what),
                                  whats);
    }
};

///

template<typename T>
struct CountSource : public gr::Block<CountSource<T>> {
    gr::PortOut<T> out;
    std::size_t    n_samples_max = 0;
    std::size_t    count         = 0;

    EventChecker *_eventChecker;

    CountSource(EventChecker &eventChecker) : _eventChecker(std::addressof(eventChecker)) {}

    ~CountSource() {}

    constexpr T
    processOne() {
        count++;
        if (count % 50'000 == 0) {
            fmt::print("Last sent {}\n", count);
        }
        if (count >= n_samples_max) {
            fmt::print("Requesting stop!\n");
            this->requestStop();
        }
        return static_cast<int>(count);
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(CountSource, out, n_samples_max);
static_assert(gr::BlockLike<CountSource<float>>);

///

template<typename T>
struct Amplify : public gr::Block<Amplify<T>> {
    gr::PortIn<T>  in;
    gr::PortOut<T> out;
    T              factor     = 1.0;
    T              lastFactor = 1.0;

    EventChecker *_eventChecker;

    Amplify(EventChecker &eventChecker) : _eventChecker(std::addressof(eventChecker)) {}

    constexpr T
    processOne(T value) {
        if (lastFactor != factor) {
            _eventChecker->addEvent("Amplify", fmt::format("factor_changed_to={}", factor));
            lastFactor = factor;
        }
        return factor * value;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(Amplify, in, out, factor);
static_assert(gr::BlockLike<Amplify<float>>);

///

template<typename T>
struct Sink : gr::Block<Sink<T>> {
    gr::PortIn<T> in;

    EventChecker *_eventChecker;

    Sink(EventChecker &eventChecker) : _eventChecker(std::addressof(eventChecker)) {}

    constexpr void
    processOne(T value) {
        std::cout << value << std::endl;
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(Sink, in);
static_assert(gr::BlockLike<Sink<float>>);

///

template<typename T>
struct FactorSettingsMessageSender : public gr::Block<FactorSettingsMessageSender<T>> {
    gr::PortOut<T> unused;
    int            countdown = 3;
    T              factor    = 1.0;
    std::string    target;

    EventChecker *_eventChecker;

    FactorSettingsMessageSender(EventChecker &eventChecker, std::string _target = {}) : _eventChecker(std::addressof(eventChecker)), target(std::move(_target)) {}

    bool
    finished() const {
        return countdown == 0;
    }

    constexpr T
    processOne() {
        if (countdown > 0) {
            countdown--;

            gr::Message message;
            message[gr::message::key::Kind]   = gr::message::kind::UpdateSettings;
            message[gr::message::key::Target] = target;
            message[gr::message::key::Data]   = gr::property_map{ { "factor", factor } };

            this->emitMessage(std::move(message));

            if (countdown == 0) {
                fmt::print("{} finished sending messages\n", this->unique_name);
            }
            factor += 1;

            std::this_thread::sleep_for(100ms);
        } else {
            this->requestStop();
        }
        return static_cast<T>(countdown);
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(FactorSettingsMessageSender, unused)

///

template<typename T>
struct SettingsUpdatedMessageReceiver : public gr::Block<SettingsUpdatedMessageReceiver<T>> {
    gr::PortOut<T> unused;
    std::size_t    countdown = 10;
    T              factor    = 1.0;

    EventChecker *_eventChecker;

    SettingsUpdatedMessageReceiver(EventChecker &eventChecker) : _eventChecker(std::addressof(eventChecker)) {}

    bool
    finished() const {
        return countdown == 0;
    }

    constexpr T
    processOne() {}

    void
    processMessages(gr::MsgPortInNamed<"__Builtin"> & /*port*/, std::span<const gr::Message> messages) {
        for (const auto &message : messages) {
            // fmt::print("[SUCCESS] {} got a message from {} of kind {}\n", this->unique_name,     //
            //            gr::messageField<std::string>(message, gr::message::key::Sender).value(), //
            //            gr::messageField<std::string>(message, gr::message::key::Kind).value());
            const auto data   = gr::messageField<gr::property_map>(message, gr::message::key::Data).value();
            const auto sender = gr::messageField<std::string>(message, gr::message::key::Sender).value();
            const auto error  = gr::messageField<gr::Message>(message, gr::message::key::ErrorInfo);
            if (error) {
                const auto  errorData = gr::messageField<gr::property_map>(*error, gr::message::key::Data).value();
                std::string keysNotSet;
                for (const auto &[k, v] : errorData) {
                    keysNotSet += " " + k;
                }
                _eventChecker->addEvent("SettingsUpdatedMessageReceiver"s + " "s + sender, "not_set" + keysNotSet);
            } else {
                _eventChecker->addEvent("SettingsUpdatedMessageReceiver"s + " "s + sender, "all_is_fine");
            }
        }
    }
};

ENABLE_REFLECTION_FOR_TEMPLATE(SettingsUpdatedMessageReceiver, unused)
static_assert(gr::traits::block::can_processMessagesForPort<SettingsUpdatedMessageReceiver<float>, gr::MsgPortInNamed<"__Builtin">>);

///

const boost::ut::suite MessagesTests = [] {
    using namespace boost::ut;
    using namespace gr;

    "SimplePortConnections"_test = [] {
        using Scheduler = gr::scheduler::Simple<>;

        EventChecker eventChecker;

        auto scheduler = [&eventChecker] {
            gr::Graph flow;
            auto     &source     = flow.emplaceBlock<CountSource<float>>(eventChecker);
            source.n_samples_max = 1'000'000;

            auto &sink    = flow.emplaceBlock<Sink<float>>(eventChecker);
            auto &process = flow.emplaceBlock<Amplify<float>>(eventChecker);

            expect(eq(ConnectionResult::SUCCESS, flow.connect<"out">(source).to<"in">(process)));
            expect(eq(ConnectionResult::SUCCESS, flow.connect<"out">(process).to<"in">(sink)));

            return Scheduler(std::move(flow));
        }();

        auto &flow = scheduler.graph();

        fmt::print("Graph:\n");
        for (const auto &block : flow.blocks()) {
            fmt::print("    {}\n", block->uniqueName());
        }

        FactorSettingsMessageSender<float>    messageSender(eventChecker);
        SettingsUpdatedMessageReceiver<float> messageReceiver(eventChecker);
        expect(eq(ConnectionResult::SUCCESS, messageSender.msgOut.connect(flow.msgIn)));
        expect(eq(ConnectionResult::SUCCESS, flow.msgOut.connect(messageReceiver.msgIn)));

        std::thread messenger([&] {
            while (!messageSender.finished()) {
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
