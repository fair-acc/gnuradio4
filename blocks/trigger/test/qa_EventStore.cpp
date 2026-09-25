#include <boost/ut.hpp>

#include <cstdint>
#include <string>

#include <gnuradio-4.0/trigger/EventStore.hpp>

using namespace gr::blocks::trigger;

namespace {
[[nodiscard]] StoredEvent dated(std::uint64_t at, std::string name) {
    gr::property_map event{{std::string(gr::tag::TRIGGER_NAME.key()), std::move(name)}, {std::string(gr::tag::TRIGGER_TIME.key()), at}};
    return StoredEvent{.event = std::move(event), .arrived = 1U, .at = at};
}

[[nodiscard]] StoredEvent undated(std::string name) { //
    return StoredEvent{.event = gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::move(name)}}, .arrived = 1U, .at = std::nullopt};
}

[[nodiscard]] std::string nameOf(const StoredEvent& stored) { //
    return std::string(gr::property_map_view{stored.event}.get_if<std::string_view>(gr::tag::TRIGGER_NAME.key()).value_or(std::string_view{}));
}
} // namespace

const boost::ut::suite<"EventStore"> _eventStore = [] {
    using namespace boost::ut;

    "events arriving out of order are read back in time order"_test = [] {
        EventStore store;
        store.push(dated(300U, "third"));
        store.push(dated(100U, "first"));
        store.push(dated(200U, "second"));
        store.order();

        expect(eq(store.size(), 3UZ));
        expect(eq(nameOf(store.ordered()[0]), std::string("first")));
        expect(eq(nameOf(store.ordered()[1]), std::string("second")));
        expect(eq(nameOf(store.ordered()[2]), std::string("third")));
    };

    "an undated event keeps its arrival order, after the dated ones"_test = [] {
        EventStore store;
        store.push(undated("no time"));
        store.push(dated(500U, "dated"));
        store.push(undated("also no time"));
        store.order();

        expect(eq(nameOf(store.ordered()[0]), std::string("dated"))) << "a time orders, the absence of one does not";
        expect(eq(nameOf(store.ordered()[1]), std::string("no time")));
        expect(eq(nameOf(store.ordered()[2]), std::string("also no time")));
    };

    "a full store loses its oldest entry and counts it"_test = [] {
        EventStore store;
        store.capacity = 2UZ;
        store.push(dated(100U, "first"));
        store.push(dated(200U, "second"));
        store.push(dated(300U, "third"));

        expect(eq(store.size(), 2UZ));
        expect(eq(store.dropped, 1U)) << "the loss is counted, never silent";
        expect(eq(nameOf(store.ordered()[0]), std::string("second"))) << "the oldest went, not the newest";
    };

    "retiring drops what the block has finished with"_test = [] {
        EventStore store;
        store.push(dated(100U, "first"));
        store.push(dated(200U, "second"));
        store.retire(1UZ);

        expect(eq(store.size(), 1UZ));
        expect(eq(nameOf(store.ordered()[0]), std::string("second")));
        store.retire(99UZ);
        expect(store.empty()) << "retiring more than it holds empties it rather than overrunning";
    };

    "how long the oldest has waited comes from this block's own clock"_test = [] {
        EventStore store;
        expect(!store.waitedNs().has_value()) << "nothing waiting, nothing to measure";

        StoredEvent arrived = dated(100U, "now");
        arrived.arrived     = monotonicNowNs();
        store.push(std::move(arrived));
        const auto waited = store.waitedNs();
        expect(waited.has_value()) << "an arrival stamp makes the wait measurable";

        StoredEvent fromDevice = dated(200U, "unstamped");
        fromDevice.arrived     = kUnknownTime;
        EventStore other;
        other.push(std::move(fromDevice));
        expect(!other.waitedNs().has_value()) << "a clock that could not read says so, rather than reporting an epoch";
    };
};

int main() { /* tests are statically executed */ }
