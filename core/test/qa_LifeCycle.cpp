#include <boost/ut.hpp>

#include <fmt/format.h>
#include <map>
#include <thread>
#include <tuple>

#ifdef __GNUC__
#pragma GCC diagnostic push // ignore warning of external libraries that from this lib-context we do not have any control over
#pragma GCC diagnostic ignored "-Wuseless-cast"
#pragma GCC diagnostic ignored "-Wsign-conversion"
#endif
#include <magic_enum.hpp>
#include <magic_enum_utility.hpp>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

#include <gnuradio-4.0/LifeCycle.hpp>

namespace gr::test {

template<lifecycle::StorageType storageType>
struct MockStateMachine : public lifecycle::StateMachine<MockStateMachine<storageType>, storageType> {
    int startCalled{};
    int stopCalled{};
    int pauseCalled{};
    int resumeCalled{};
    int resetCalled{};

    void
    start() {
        startCalled++;
    }

    void
    stop() {
        stopCalled++;
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

        expect(machine.transitionTo(State::INITIALISED));
        expect(machine.state() == State::INITIALISED);

        expect(machine.transitionTo(State::RUNNING));
        expect(machine.state() == State::RUNNING);
        expect(eq(machine.startCalled, 1)) << "start() called once";

        expect(machine.transitionTo(State::REQUESTED_PAUSE));
        expect(machine.state() == State::REQUESTED_PAUSE);
        expect(eq(machine.pauseCalled, 1)) << "pause() called once";

        expect(machine.transitionTo(State::PAUSED));
        expect(machine.state() == State::PAUSED);
        expect(eq(machine.pauseCalled, 1)) << "pause() called once";

        expect(machine.transitionTo(State::RUNNING));
        expect(machine.state() == State::RUNNING);
        expect(eq(machine.resumeCalled, 1)) << "resume() called once";

        expect(machine.transitionTo(State::REQUESTED_STOP));
        expect(machine.state() == State::REQUESTED_STOP);
        expect(eq(machine.stopCalled, 1)) << "stop() called once";

        expect(machine.transitionTo(State::STOPPED));
        expect(machine.state() == State::STOPPED);
        expect(eq(machine.stopCalled, 1)) << "stop() called once";

        expect(machine.transitionTo(State::INITIALISED));
        expect(machine.state() == State::INITIALISED);
        expect(eq(machine.resetCalled, 1)) << "reset() called once";

        expect(machine.transitionTo(State::ERROR));
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
            { State::IDLE, { State::INITIALISED } },
            { State::INITIALISED, { State::RUNNING } },
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

                // special case: Any state can transition to ERROR
                if (toState == State::ERROR) {
                    isAllowed = true;
                }

                bool isValid = isValidTransition(fromState, toState);

                // Assert that the function's validity matches the expected validity
                expect(isValid == isAllowed) << "Transition from " << static_cast<int>(fromState) << " to " << static_cast<int>(toState) << " should be " << (isAllowed ? "allowed" : "disallowed");
            });
        });
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
            expect(machine.transitionTo(State::INITIALISED));
            expect(machine.state() == State::INITIALISED);
        });
        machine.waitOnState(State::IDLE); // blocks here
        expect(machine.state() == State::INITIALISED);
        if (notifyThread.joinable()) {
            notifyThread.join();
        }
        // finished successful
    };
};

int
main() { /* tests are statically executed */
}
