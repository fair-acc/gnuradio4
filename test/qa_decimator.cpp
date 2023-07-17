#include <boost/ut.hpp>

#include <buffer.hpp>
#include <graph.hpp>
#include <node.hpp>
#include <reflection.hpp>
#include <scheduler.hpp>

#include <fmt/format.h>

namespace fg = fair::graph;

using std::size_t;
using ssize_t = std::make_signed_t<size_t>;

template<typename T>
struct source : public fg::node<source<T>, fg::OUT<T, 0, 1024, "out">> {
    ssize_t produced = 0;
    T       value    = 1;

    constexpr ssize_t
    available_samples(const source&) const noexcept {
        return 64 - produced;
    }

    constexpr fg::work_return_status_t
    process_bulk(std::span<T> out) noexcept {
        if (available_samples(*this) <= 0) {
            return fg::work_return_status_t::ERROR;
        }
        for (T &x : out) {
            x = value++;
        }
        produced += ssize_t(out.size());
        const auto ret = (available_samples(*this) <= 0) ? fg::work_return_status_t::DONE : fg::work_return_status_t::OK;
        fmt::println("source::process_bulk; produced = {}, returning {}", produced, int(ret));
        return ret;
    }
};

template<typename T>
struct drop_odd_samples : public fg::node<drop_odd_samples<T>, fg::IN<T, 0, 15, "in">, fg::OUT<T, 0, 15, "out">> {
    bool even = true;

    constexpr fg::work_return_status_t
    process_bulk(std::span<const T> in, fg::PublishableSpan auto& out) noexcept {
        boost::ut::expect(in.size() <= 15);
        boost::ut::expect(out.size() <= 15);
        const size_t to_publish = (in.size() + even) / 2;
        boost::ut::expect(to_publish <= out.size());
        for (size_t i = 0; i < in.size(); ++i) {
            if (even) {
                out[i / 2] = in[i];
            }
            even = !even;
        }
        out.publish(to_publish);
        fmt::println("drop_odd_samples::process_bulk received {} samples and published {} samples", in.size(), to_publish);
        return fg::work_return_status_t::OK;
    }
};

template<typename T, T increment>
struct sink : public fg::node<sink<T, increment>, fg::IN<T, 0, 1024, "in">> {
    T expect_next = 1;

    constexpr void
    process_one(T x) noexcept {
        using namespace boost::ut;
        expect(eq(x, expect_next));
        expect_next += increment;
    }
};

const boost::ut::suite simple_decimator = [] {
    using namespace boost::ut;

    "drop odd"_test = [] {
        fg::graph flow_graph;

        auto     &n0 = flow_graph.make_node<source<int>>();
        auto     &n1 = flow_graph.make_node<drop_odd_samples<int>>();
        auto     &n2 = flow_graph.make_node<sink<int, 2>>();

        expect(eq(flow_graph.connect<"out">(n0).to<"in">(n1), fg::connection_result_t::SUCCESS));
        expect(eq(flow_graph.connect<"out">(n1).to<"in">(n2), fg::connection_result_t::SUCCESS));

        fg::scheduler::simple sched{ std::move(flow_graph) };
        fmt::println("drop odd starts");
        sched.start();
        fmt::println("drop odd done");
    };
};

int
main() {}
