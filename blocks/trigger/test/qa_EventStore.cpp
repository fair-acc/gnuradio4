#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>

#include <gnuradio-4.0/trigger/EventStore.hpp>

using namespace gr::blocks::trigger;

namespace {
[[nodiscard]] StoredEvent dated(std::uint64_t at, std::string name) {
    gr::property_map event{{std::string(gr::tag::TRIGGER_NAME.key()), std::move(name)}, {std::string(gr::tag::TRIGGER_TIME.key()), at}};
    return StoredEvent{.event = std::move(event), .at = at};
}

[[nodiscard]] StoredEvent simultaneous(std::uint64_t at, float offset, std::string name, std::string context) {
    gr::property_map event{
        {std::string(gr::tag::TRIGGER_NAME.key()), std::move(name)},
        {std::string(gr::tag::TRIGGER_TIME.key()), at},
        {std::string(gr::tag::TRIGGER_OFFSET.key()), offset},
        {std::string(gr::tag::CONTEXT.key()), std::move(context)},
    };
    return StoredEvent{.event = std::move(event), .at = at};
}

[[nodiscard]] StoredEvent undated(std::string name) { //
    return StoredEvent{.event = gr::property_map{{std::string(gr::tag::TRIGGER_NAME.key()), std::move(name)}}, .at = std::nullopt};
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

    "simultaneous events are ordered by offset, name and context whichever arrived first"_test = [] {
        const auto  makeAll      = [] { return std::array{simultaneous(100U, 0.f, "b", "x"), simultaneous(100U, 0.f, "a", "y"), simultaneous(100U, 0.f, "a", "x"), simultaneous(100U, -0.25f, "z", "x")}; };
        const auto  expected     = std::vector<std::string>{"z/x", "a/x", "a/y", "b/x"};
        auto        arrival      = std::array<std::size_t, 4UZ>{0UZ, 1UZ, 2UZ, 3UZ};
        std::size_t permutations = 0UZ;
        do {
            auto       all = makeAll();
            EventStore store;
            for (const std::size_t index : arrival) {
                store.push(std::move(all[index]));
            }
            store.order();
            std::vector<std::string> order;
            for (const StoredEvent& stored : store.ordered()) {
                const gr::property_map_view event{stored.event};
                order.push_back(std::string(*event.get_if<std::string_view>(gr::tag::TRIGGER_NAME.key())) + "/" + std::string(*event.get_if<std::string_view>(gr::tag::CONTEXT.key())));
            }
            expect(order == expected) << "arrival permutation " << permutations;
            ++permutations;
        } while (std::ranges::next_permutation(arrival).found);
        expect(eq(permutations, 24UZ));
    };

    "events of equal time and equal tie-break keep their arrival order"_test = [] {
        EventStore store;
        auto       first  = simultaneous(100U, 0.f, "a", "x");
        auto       second = simultaneous(100U, 0.f, "a", "x");
        first.event.insert_or_assign("n", std::int64_t{1});
        second.event.insert_or_assign("n", std::int64_t{2});
        store.push(std::move(second));
        store.push(std::move(first));
        store.order();
        expect(eq(store.ordered()[0].event.template value_or<std::int64_t>("n", 0), std::int64_t{2})) << "stable: indistinguishable events are not reordered";
    };

    "an event is ready once the watermark has passed it by the lateness (spec §3.3)"_test = [] {
        EventStore store;
        store.lateness = 100U;
        store.push(dated(1'000U, "first"));
        store.order();
        expect(eq(store.ready(), 0UZ)) << "nothing yet shows that no earlier event can still come";
        store.push(dated(1'200U, "heartbeat"));
        store.order();
        expect(eq(store.ready(), 1UZ)) << "a later event moves the watermark: 1000 + 100 <= 1200";
    };

    "the watermark is the latest trigger_time seen, and a later arrival of an earlier time does not move it"_test = [] {
        EventStore store;
        store.lateness = 300U; // how far one producer's events may trail another's
        store.push(dated(1'200U, "edge"));
        store.push(dated(1'000U, "pulse")); // arrives second, dated earlier
        store.order();
        expect(eq(store.watermark.value_or(0U), std::uint64_t{1'200U})) << "the earlier time leaves the watermark at 1200";
        expect(eq(nameOf(store.ordered()[0]), std::string("pulse"))) << "ordered by trigger_time, not arrival";
        expect(eq(store.ready(), 0UZ)) << "1000 + 300 > 1200: an event dated before 1000 could still come";

        store.push(dated(1'500U, "later"));
        store.order();
        expect(eq(store.watermark.value_or(0U), std::uint64_t{1'500U}));
        expect(eq(store.ready(), 2UZ)) << "watermark 1500: pulse (1000 + 300) and edge (1200 + 300) are ready, later (1500 + 300) waits";
    };

    "without a lateness every drained event is ready, as before"_test = [] {
        EventStore store;
        store.push(dated(300U, "late"));
        store.push(dated(100U, "early"));
        store.order();
        expect(eq(store.ready(), 2UZ));
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

    "clearing forgets the entries and the loss count"_test = [] {
        EventStore store;
        store.capacity = 1UZ;
        store.push(dated(100U, "first"));
        store.push(dated(200U, "second"));
        store.clear();

        expect(store.empty());
        expect(eq(store.dropped, 0U));
    };
};

int main() { return boost::ut::cfg<>.run({.report_errors = true}) ? 1 : 0; }
