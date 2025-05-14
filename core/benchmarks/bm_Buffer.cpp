#include <benchmark.hpp>

#include <barrier>
#include <chrono>
#include <cstddef>
#include <thread>

#include <format>

#include <gnuradio-4.0/Buffer.hpp> // new buffer header interface
#include <gnuradio-4.0/BufferSkeleton.hpp>
#include <gnuradio-4.0/CircularBuffer.hpp>

#include <gnuradio-4.0/meta/utils.hpp>

using namespace gr;

#if defined(__has_include) && not defined(__EMSCRIPTEN__) && not defined(__APPLE__)
#if __has_include(<pthread.h>) && __has_include(<sched.h>)
#include <errno.h>
#include <pthread.h>
#include <sched.h>

void setCpuAffinity(const int cpuID) // N.B. pthread is not portable
{
    const std::size_t nCPU = std::thread::hardware_concurrency();
    // std::print("set CPU affinity to core {}\n", cpuID % nCPU);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(static_cast<std::size_t>(cpuID) % nCPU, &cpuset);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (rc != 0) {
        std::print(stderr, "pthread_setaffinity_np({} of {}): {} - {}\n", cpuID, nCPU, rc, strerror(rc));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20)); // to force re-scheduling
}
#else
void setCpuAffinity(const int cpuID) {}
#endif
#else
void setCpuAffinity(const int /*cpuID*/) {}
#endif

template<typename T = int>
void runTest(BufferLike auto& buffer, const std::size_t vectorLength, const std::size_t minSamples, const std::size_t nProducer, const std::size_t nConsumer, const std::string_view name) {
    gr::meta::precondition(nProducer > 0);
    gr::meta::precondition(nConsumer > 0);

    constexpr int nRepeat = 8;

    std::barrier barrier(static_cast<std::ptrdiff_t>(nProducer + nConsumer + 1));

    // init producers
    std::atomic<int>         threadID = 0;
    std::vector<std::thread> producers;
    for (std::size_t i = 0; i < nProducer; i++) {
        producers.emplace_back([&]() {
            // set thread affinity
            setCpuAffinity(threadID++);
            barrier.arrive_and_wait();
            for (int rep = 0; rep < nRepeat; ++rep) {
                BufferWriterLike auto writer           = buffer.new_writer();
                std::size_t           nSamplesProduced = 0;
                barrier.arrive_and_wait();
                while (nSamplesProduced <= (minSamples + nProducer - 1) / nProducer) {
                    WriterSpanLike auto data = writer.reserve(vectorLength);
                    if (!data.empty()) {
                        data.publish(vectorLength);
                        nSamplesProduced += vectorLength;
                    } else {
                        // std::this_thread::yield();
                        // std::this_thread::sleep_for(std::chrono::milliseconds(10));
                    }
                }
                barrier.arrive_and_wait();
            }
        });
    }

    // init consumers
    std::vector<std::thread> consumers;
    for (std::size_t i = 0; i < nConsumer; i++) {
        consumers.emplace_back([&]() {
            setCpuAffinity(threadID++);
            barrier.arrive_and_wait();
            for (int rep = 0; rep < nRepeat; ++rep) {
                BufferReaderLike auto reader           = buffer.new_reader();
                std::size_t           nSamplesConsumed = 0;
                barrier.arrive_and_wait();
                while (nSamplesConsumed < minSamples) {
                    if (reader.available() < vectorLength) {
                        continue;
                    }
                    ReaderSpanLike auto input = reader.get(vectorLength);
                    nSamplesConsumed += input.size();

                    if (!input.consume(input.size())) {
                        throw std::runtime_error(std::format("could not consume {} samples", input.size()));
                    }
                }
                barrier.arrive_and_wait();
            }
        });
    }

    // all producers and consumer are ready, waiting to give the sign
    barrier.arrive_and_wait();
    ::benchmark::benchmark<nRepeat>(std::format("{:>8}: {} producers -<{:^4}>-> {} consumers", name, nProducer, vectorLength, nConsumer), minSamples) = [&]() {
        barrier.arrive_and_wait();
        barrier.arrive_and_wait();
    };

    // clean up
    for (std::thread& thread : producers) {
        thread.join();
    }
    for (std::thread& thread : consumers) {
        thread.join();
    }
}

inline const boost::ut::suite _buffer_tests = [] {
    enum class BufferStrategy { posix, portable };

    // Test parameters
    const std::size_t samples             = 1'000'000; // minimum number of samples
    const std::size_t maxProducers        = 4;         // maximum number of producers to test, 1-2-4-8 ....
    const std::size_t maxConsumers        = 4;         // maximum number of consumers to test, 1-2-4-8 ....
    const std::vector bufferStrategyTests = {
#ifdef HAS_POSIX_MAP_INTERFACE
        BufferStrategy::posix,
#endif
        BufferStrategy::portable};
    const std::vector vecLengthTests = {1UL, 1024UL};

    for (const BufferStrategy strategy : bufferStrategyTests) {
        for (const std::size_t veclen : vecLengthTests) {
            benchmark::results::add_separator();
            for (std::size_t nP = 1; nP <= maxProducers; nP *= 2) {
                for (std::size_t nC = 1; nC <= maxConsumers; nC *= 2) {
                    const std::size_t size      = std::max(4096UL, veclen) * nC * 10UL;
                    const bool        isPosix   = strategy == BufferStrategy::posix;
                    const auto        allocator = (isPosix) ? gr::double_mapped_memory_resource::allocator<int32_t>() : std::pmr::polymorphic_allocator<int32_t>();
                    auto              invoke    = [&](auto buffer) { runTest(buffer, veclen, samples, nP, nC, isPosix ? "POSIX" : "portable"); };
                    if (nP == 1) {
                        using BufferType       = CircularBuffer<int32_t, std::dynamic_extent, ProducerType::Single>;
                        BufferLike auto buffer = BufferType(size, allocator);
                        invoke(buffer);
                    } else {
                        using BufferType       = CircularBuffer<int32_t, std::dynamic_extent, ProducerType::Multi>;
                        BufferLike auto buffer = BufferType(size, allocator);
                        invoke(buffer);
                    }
                }
            }
        }
    }
};

int main() { /* not needed by the UT framework */ }
