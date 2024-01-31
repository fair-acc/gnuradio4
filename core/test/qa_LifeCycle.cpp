#include <boost/ut.hpp>

#include <fmt/format.h>
#include <map>
#include <stdexcept>
#include <thread>
#include <tuple>

#ifdef __GNUC__
#pragma GCC diagnostic push // ignore warning of external libraries that from this lib-context we do not have any control over
#ifndef __clang__
#pragma GCC diagnostic ignored "-Wuseless-cast"
#endif
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
#include <magic_enum.hpp>
#include <magic_enum_utility.hpp>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <gnuradio-4.0/LifeCycle.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

namespace gr::test {

template<lifecycle::StorageType storageType, bool throwException = false>
struct MockStateMachine : public lifecycle::StateMachine<MockStateMachine<storageType, throwException>, storageType> {
    std::string unique_name = gr::meta::type_name<MockStateMachine<storageType, throwException>>();
    int         startCalled{};
    int         stopCalled{};
    int         pauseCalled{};
    int         resumeCalled{};
    int         resetCalled{};

    void
    start() {
        startCalled++;
        if constexpr (throwException) { // throw voluntary exception
            throw std::domain_error("start() throws specific exception");
        }
    }

    void
    stop() {
        stopCalled++;
        if constexpr (throwException) { // throw voluntary unknown exception
            throw "unknown/unnamed exception";
        }
    }

    void
    pause() {
        pauseCalled++;
    }

    void
    resume() {
        resumeCalled++;
    }

    void
    reset() {
        resetCalled++;
    }
};

} // namespace gr::test

const boost::ut::suite StateMachineTest = [] {
    using namespace std::string_literals;
    using namespace boost::ut;
    using namespace gr::lifecycle;

    auto nominalTest = []<class MockStateMachine>() {
        MockStateMachine machine;

        expect(machine.state() == State::IDLE);

        expect(machine.changeStateTo(State::INITIALISED).has_value());
        expect(machine.state() == State::INITIALISED);

        expect(machine.changeStateTo(State::RUNNING).has_value());
        expect(machine.state() == State::RUNNING);
        expect(eq(machine.startCalled, 1)) << "start() called once";

        expect(machine.changeStateTo(State::REQUESTED_PAUSE).has_value());
        expect(machine.state() == State::REQUESTED_PAUSE);
        expect(eq(machine.pauseCalled, 1)) << "pause() called once";

        expect(machine.changeStateTo(State::PAUSED).has_value());
        expect(machine.state() == State::PAUSED);
        expect(eq(machine.pauseCalled, 1)) << "pause() called once";

        expect(machine.changeStateTo(State::REQUESTED_PAUSE).has_value()); // already in STOPPED
        expect(machine.state() == State::PAUSED);
        expect(eq(machine.pauseCalled, 1)) << "PAUSED() should not be called a second time";

        expect(machine.changeStateTo(State::RUNNING).has_value());
        expect(machine.state() == State::RUNNING);
        expect(eq(machine.resumeCalled, 1)) << "resume() called once";

        expect(machine.changeStateTo(State::REQUESTED_STOP).has_value());
        expect(machine.state() == State::REQUESTED_STOP);
        expect(eq(machine.stopCalled, 1)) << "stop() called once";

        expect(machine.changeStateTo(State::STOPPED).has_value());
        expect(machine.state() == State::STOPPED);
        expect(eq(machine.stopCalled, 1)) << "stop() called once";

        expect(machine.changeStateTo(State::STOPPED).has_value()); // already in STOPPED
        expect(machine.state() == State::STOPPED);
        expect(eq(machine.stopCalled, 1)) << "stop() should not be called a second time";

        expect(machine.changeStateTo(State::REQUESTED_STOP).has_value()); // already in STOPPED
        expect(machine.state() == State::STOPPED);
        expect(eq(machine.stopCalled, 1)) << "stop() should not be called a second time";

        expect(machine.changeStateTo(State::INITIALISED).has_value());
        expect(machine.state() == State::INITIALISED);
        expect(eq(machine.resetCalled, 1)) << "reset() called once";

        expect(machine.changeStateTo(State::ERROR).has_value());
        expect(machine.state() == State::ERROR);
        expect(eq(machine.resetCalled, 1)) << "reset() called once";

        // ensure again that the path have been executed only once
        expect(eq(machine.startCalled, 1)) << "end-of-test: start() called once";
        expect(eq(machine.stopCalled, 1)) << "end-of-test: stop() called once";
        expect(eq(machine.pauseCalled, 1)) << "end-of-test: pause() called once";
        expect(eq(machine.resumeCalled, 1)) << "end-of-test: resume() called once";
        expect(eq(machine.resetCalled, 1)) << "end-of-test: reset() called once";
    };

    "StateMachine nominal State transitions -- non-atomic"_test = [&] { nominalTest.template operator()<gr::test::MockStateMachine<StorageType::NON_ATOMIC>>(); };
    "StateMachine nominal State transitions -- atomic"_test     = [&] { // N.B. this workaround is needed because atomic are not copyable
        nominalTest.template operator()<gr::test::MockStateMachine<StorageType::NON_ATOMIC>>();
    };

    "StateMachine all State transitions"_test = [] {
        std::map<State, std::vector<State>> allowedTransitions = {
            { State::IDLE, { State::INITIALISED, State::REQUESTED_STOP, State::STOPPED } },
            { State::INITIALISED, { State::RUNNING, State::REQUESTED_STOP, State::STOPPED } },
            { State::RUNNING, { State::REQUESTED_PAUSE, State::REQUESTED_STOP } },
            { State::REQUESTED_PAUSE, { State::PAUSED } },
            { State::PAUSED, { State::RUNNING, State::REQUESTED_STOP } },
            { State::REQUESTED_STOP, { State::STOPPED } },
            { State::STOPPED, { State::INITIALISED } },
            { State::ERROR, { State::INITIALISED } },
        };

        magic_enum::enum_for_each<State>([&allowedTransitions](State fromState) {
            magic_enum::enum_for_each<State>([&fromState, &allowedTransitions](State toState) {
                bool isAllowed = std::find(allowedTransitions[fromState].begin(), allowedTransitions[fromState].end(), toState) != allowedTransitions[fromState].end();

                // special case: Any state can transition to ERROR and identity
                if (toState == State::ERROR || fromState == toState) {
                    isAllowed = true;
                }

                bool isValid = isValidTransition(fromState, toState);

                // Assert that the function's validity matches the expected validity
                expect(isValid == isAllowed) << fmt::format("Transition from {} to {} should be {}", magic_enum::enum_name(fromState), magic_enum::enum_name(toState),
                                                            isAllowed ? "allowed" : "disallowed");
            });
        });
    };

    "StateMachine move constructor non-atomic"_test = [] {
        using namespace gr::test;
        using namespace gr::lifecycle;

        MockStateMachine<StorageType::NON_ATOMIC> machine1;
        expect(machine1.state() == State::IDLE);
        expect(machine1.changeStateTo(State::ERROR).has_value());
        expect(machine1.state() == State::ERROR);

        MockStateMachine<StorageType::NON_ATOMIC> machine2(std::move(machine1));
        expect(machine2.state() == State::ERROR);
    };

    "StateMachine move constructor atomic"_test = [] {
        using namespace gr::test;
        using namespace gr::lifecycle;

        MockStateMachine<StorageType::ATOMIC> machine1;
        expect(machine1.state() == State::IDLE);
        expect(machine1.changeStateTo(State::ERROR).has_value());
        expect(machine1.state() == State::ERROR);

        MockStateMachine<StorageType::ATOMIC> machine2(std::move(machine1));
        expect(machine2.state() == State::ERROR);
    };

    "StateMachine misc"_test = [] {
        magic_enum::enum_for_each<State>([&](State state) {
            std::vector<State> allowedState{ RUNNING, REQUESTED_PAUSE, PAUSED };

            if (std::ranges::find(allowedState, state) != allowedState.end()) {
                expect(isActive(state));
            } else {
                expect(!isActive(state));
            }
        });

        magic_enum::enum_for_each<State>([&](State state) {
            std::vector<State> allowedState{ REQUESTED_STOP, STOPPED };

            if (std::ranges::find(allowedState, state) != allowedState.end()) {
                expect(isShuttingDown(state));
            } else {
                expect(!isShuttingDown(state));
            }
        });

        gr::test::MockStateMachine<StorageType::ATOMIC> machine;
        expect(machine.state() == State::IDLE);

        std::thread notifyThread([&machine]() {
            using namespace std::literals;
            std::this_thread::sleep_for(100ms);
            expect(machine.changeStateTo(State::INITIALISED).has_value());
            expect(machine.state() == State::INITIALISED);
        });
        expect(machine.state() == State::IDLE);
        machine.waitOnState(State::IDLE); // blocks here
        expect(machine.state() == State::INITIALISED);
        if (notifyThread.joinable()) {
            notifyThread.join();
        }
        // finished successful
    };

    "StateMachine w/o invoking start"_test = [] {
        struct MockStateMachine : public gr::lifecycle::StateMachine<void> {
            int startCalled{};

            void
            start() {
                startCalled++;
            }
        } machine;

        expect(machine.state() == State::IDLE);

        expect(machine.changeStateTo(State::INITIALISED).has_value());
        expect(machine.state() == State::INITIALISED);

        expect(machine.changeStateTo(State::RUNNING).has_value());
        expect(machine.state() == State::RUNNING);
        expect(eq(machine.startCalled, 0)) << "start() should not be called";
    };

    "StateMachine exception handling"_test = [] {
        gr::test::MockStateMachine<StorageType::ATOMIC, true> machine;
        expect(machine.state() == State::IDLE);

        expect(machine.changeStateTo(State::INITIALISED).has_value());
        expect(machine.state() == State::INITIALISED);

        auto ret1 = machine.changeStateTo(State::RUNNING);
        expect(!ret1.has_value());
        fmt::println("error1.message: {}", ret1.error().message);
        fmt::println("error1.location: {}", ret1.error().sourceLocation);
        expect(machine.state() == State::ERROR);
        expect(eq(machine.startCalled, 1)) << "start() called once";
        expect(machine.changeStateTo(State::INITIALISED).has_value()) << "reset()";
        expect(eq(machine.resetCalled, 1)) << "reset() called once";
        expect(machine.state() == State::INITIALISED);

        auto ret2 = machine.changeStateTo(State::REQUESTED_STOP);
        expect(!ret2.has_value());
        fmt::println("error2.message: {}", ret2.error().message);
        fmt::println("error2.message: {}", ret2.error().sourceLocation);
        expect(machine.state() == State::ERROR);
        expect(eq(machine.stopCalled, 1)) << "stop() called once";
        expect(machine.changeStateTo(State::INITIALISED).has_value()) << "reset()";
        expect(eq(machine.resetCalled, 2)) << "reset() called once";
    };
};

int
main() { /* tests are statically executed */
}
