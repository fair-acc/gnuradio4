#include "benchmark.hpp"

#include <barrier>
#include <boost/ut.hpp>
#include <chrono>
#include <iostream>
#include <thread>

#include <fmt/format.h>

#include <buffer.hpp> // new buffer header interface
#include <buffer_skeleton.hpp>
#include <circular_buffer.hpp>
#include <utils.hpp>

#include <iostream>

using namespace gr;

#if defined __has_include && not __EMSCRIPTEN__
#if __has_include(<pthread.h>) && __has_include(<sched.h>)
#include <errno.h>
#include <pthread.h>
#include <sched.h>

void
setCpuAffinity(const int cpuID) // N.B. pthread is not portable
{
    const auto nCPU = std::thread::hardware_concurrency();
    // fmt::print("set CPU affinity to core {}\n", cpuID % nCPU);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpuID % nCPU, &cpuset);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        constexpr auto fmt = "pthread_setaffinity_np({} of {}): {} - {}\n";
        fmt::print(stderr, fmt, cpuID, nCPU, rc, strerror(rc));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20)); // to force re-scheduling
}
#else
void
setCpuAffinity(const int cpuID) {}
#endif
#else
void
setCpuAffinity(const int cpuID) {}
#endif

enum class WriteApi {
    via_lambda, via_split_request_publish_RAII };


template<WriteApi PublisherAPI = WriteApi::via_lambda, typename T = int>
void testNewAPI(Buffer auto &buffer, const std::size_t vector_length, const std::size_t min_samples, const int nProducer,
           const int nConsumer, const std::string_view name) {
    fair::meta::precondition(nProducer > 0);
    fair::meta::precondition(nConsumer > 0);

    constexpr int n_repeat = 8;

    std::barrier barrier(nProducer + nConsumer + 1);

    // init producers
    std::atomic<int> threadID = 0;
    std::vector<std::thread> producers;
    for (int i = 0; i < nProducer; i++) {
        producers.emplace_back([&]() {
            // set thread affinity
            setCpuAffinity(threadID++);
            barrier.arrive_and_wait();
            for (int rep = 0; rep < n_repeat; ++rep) {
                BufferWriter auto writer = buffer.new_writer();
                std::size_t nSamplesProduced = 0;
                barrier.arrive_and_wait();
                while (nSamplesProduced <= (min_samples + nProducer - 1) / nProducer) {
                    if constexpr (PublisherAPI == WriteApi::via_lambda) {
                        writer.publish([](auto &) {}, vector_length);
                    } else if constexpr (PublisherAPI == WriteApi::via_split_request_publish_RAII) {
                        auto data = writer.reserve_output_range(vector_length);

                        data.publish(vector_length);
                    } else {
                        static_assert(fair::meta::always_false<T>, "unknown PublisherAPI case");
                    }
                    nSamplesProduced += vector_length;
                }
                barrier.arrive_and_wait();
            }
        });
    }

    // init consumers
    std::vector<std::thread> consumers;
    for (int i = 0; i < nConsumer; i++) {
        consumers.emplace_back([&]() {
            setCpuAffinity(threadID++);
            barrier.arrive_and_wait();
            for (int rep = 0; rep < n_repeat; ++rep) {
                BufferReader auto reader = buffer.new_reader();
                std::size_t nSamplesConsumed = 0;
                barrier.arrive_and_wait();
                while (nSamplesConsumed < min_samples) {
                    if (reader.available() < vector_length) {
                        continue;
                    }
                    const auto &input = reader.get(vector_length);
                    nSamplesConsumed += input.size();

                    if (!reader.consume(input.size())) {
                        throw std::runtime_error(fmt::format("could not consume {} samples", input.size()));
                    }
                }
                barrier.arrive_and_wait();
            }
        });
    }

    // all producers and consumer are ready, waiting to give the sign
    barrier.arrive_and_wait();
    ::benchmark::benchmark<n_repeat>(fmt::format("{:>8}: {} producers -<{:^4}>-> {} consumers", name, nProducer,
                                                 vector_length, nConsumer),
                                     min_samples)
            = [&]() {
                  barrier.arrive_and_wait();
                  barrier.arrive_and_wait();
              };

    // clean up
    for (std::thread &thread : producers) thread.join();
    for (std::thread &thread : consumers) thread.join();
}

inline const boost::ut::suite _buffer_tests = [] {
    const uint64_t samples = 10'000'000; // minimum number of samples
    enum class BufferStrategy
    {
      posix,
      portable
    };

    for (WriteApi writerAPI : { WriteApi::via_lambda,  WriteApi::via_split_request_publish_RAII }) {
        for (BufferStrategy strategy : { /*BufferStrategy::posix,*/ BufferStrategy::portable }) {
            for (int veclen : { 1, 1024 }) {
                if (not(strategy == BufferStrategy::posix and veclen == 1)) {
                    benchmark::results::add_separator();
                }
                for (int nP = 1; nP <= 4; nP *= 2) {
                    for (int nR = 1; nR <= 4; nR *= 2) {
                        const std::size_t size      = std::max(4096, veclen) * nR * 10;
                        auto              allocator = std::pmr::polymorphic_allocator<int32_t>();
                        const bool        is_posix  = strategy == BufferStrategy::posix;
                        auto              invoke    = [&](auto buffer) {
                            switch (writerAPI) {
                            case WriteApi::via_split_request_publish_RAII:
                                testNewAPI<WriteApi::via_split_request_publish_RAII>(buffer, veclen, samples, nP, nR, is_posix ? "POSIX - RAII writer" : "portable - RAII writer");
                                break;
                            case WriteApi::via_lambda:
                            default:
                                testNewAPI<WriteApi::via_lambda>(buffer, veclen, samples, nP, nR, is_posix ? "POSIX" : "portable");
                                break;
                            }
                        };
                        if (nP == 1) {
                            using BufferType   = circular_buffer<int32_t, std::dynamic_extent, ProducerType::Single>;
                            Buffer auto buffer = is_posix ? BufferType(size) : BufferType(size, allocator);
                            invoke(buffer);
                        } else {
                            using BufferType   = circular_buffer<int32_t, std::dynamic_extent, ProducerType::Multi>;
                            Buffer auto buffer = is_posix ? BufferType(size) : BufferType(size, allocator);
                            invoke(buffer);
                        }
                    }
                }
            }
        }
    }
};

int
main() { /* not needed by the UT framework */
}
