#include "benchmark.hpp"

#include <barrier>
#include <thread>

const boost::ut::suite global_benchmarks = [] {
    using namespace benchmark;
    "string creation2"_benchmark = [] {
        std::string created_string;
        created_string = "hello";
        force_to_memory(created_string);
    };

    "string creation3"_benchmark = [] {
        std::string created_string = "hello";
        force_to_memory(created_string);
    };

    "string creation4"_benchmark.repeat<10'000>() = [] {
        std::string created_string = "hello";
        force_to_memory(created_string);
    };

    "failing bm"_benchmark = [] {
        std::string created_string;
        created_string = "hello";
        force_to_memory(created_string);
        throw std::invalid_argument("fails here on purpose");
    };
#if not defined(_LIBCPP_VERSION)
    "using marker"_benchmark.repeat<10>() = [](MarkerMap<"source", "sink1", "sink2", "finish">& marker) {
        std::barrier start(3, [&marker] { marker.at<"source">().now(); });  // 1 source + 2 sinks
        std::barrier finish(3, [&marker] { marker.at<"finish">().now(); }); // 1 source + 2 sinks
        std::jthread source([&start, &finish] {
            start.arrive_and_wait();
            finish.arrive_and_wait();
        });
        std::jthread sink1([&start, &marker, &finish] {
            start.arrive_and_wait();
            std::this_thread::sleep_for(std::chrono::microseconds(500));
            marker.at<"sink1">().now();
            finish.arrive_and_wait();
        });
        std::jthread sink2([&start, &marker, &finish] {
            start.arrive_and_wait();
            std::this_thread::sleep_for(std::chrono::microseconds(2));
            marker.at<"sink2">().now();
            finish.arrive_and_wait();
        });
        source.join();
        sink1.join();
        sink2.join();
    };
#endif
};

int main() {
    using namespace boost::ut;
    using namespace benchmark;

    "string creation1"_benchmark = [] {
        std::string created_string{"hello"};
        force_to_memory(created_string);
    };
}
