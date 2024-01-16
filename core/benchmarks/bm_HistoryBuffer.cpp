#include <benchmark.hpp>

#include <chrono>
#include <iostream>
#include <thread>

#include <fmt/format.h>

#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/meta/utils.hpp>

inline const boost::ut::suite _buffer_tests = [] {
    constexpr std::size_t n_repetitions = 10;
    constexpr std::size_t samples       = 10'000'000; // minimum number of samples
    using namespace benchmark;
    using namespace gr;

    {
        CircularBuffer<int, std::dynamic_extent, ProducerType::Multi> buffer(32);
        auto                                                          writer                    = buffer.new_writer();
        auto                                                          reader                    = buffer.new_reader();

        "circular_buffer<int>(32) - multiple producer"_benchmark.repeat<n_repetitions>(samples) = [&writer, &reader] {
            static int counter = 0;
            for (std::size_t i = 0; i < samples; i++) {
                writer.publish([&](auto &vec) { vec[0] = counter; }, 1LU);
                if (const auto data = reader.get(1)[0] != counter) {
                    throw std::runtime_error(fmt::format("write {} read {} mismatch", counter, data));
                }
                [[maybe_unused]] const auto ret = reader.consume(1);
                counter++;
            }
        };
    }
    {
        CircularBuffer<int> buffer(32);
        auto                writer                                                                       = buffer.new_writer();
        auto                reader                                                                       = buffer.new_reader();

        "circular_buffer<int>(32) - single producer via lambda"_benchmark.repeat<n_repetitions>(samples) = [&writer, &reader] {
            static int counter = 0;
            for (std::size_t i = 0; i < samples; i++) {
                writer.publish([&](auto &vec) { vec[0] = counter; }, 1LU);
                if (const auto data = reader.get(1)[0] != counter) {
                    throw std::runtime_error(fmt::format("write {} read {} mismatch", counter, data));
                }
                [[maybe_unused]] const auto ret = reader.consume(1);
                counter++;
            }
        };
    }
    {
        CircularBuffer<int> buffer(32);
        auto                writer                                                                        = buffer.new_writer();
        auto                reader                                                                        = buffer.new_reader();

        "circular_buffer<int>(32) - single producer via reserve"_benchmark.repeat<n_repetitions>(samples) = [&writer, &reader] {
            static int counter = 0;
            for (std::size_t i = 0; i < samples; i++) {
                auto write_data = writer.reserve_output_range(1LU);
                write_data[0]   = counter;
                write_data.publish(1);
                if (const auto data = reader.get(1)[0] != counter) {
                    throw std::runtime_error(fmt::format("write {} read {} mismatch", counter, data));
                }
                [[maybe_unused]] const auto ret = reader.consume(1);
                counter++;
            }
        };
    }
    /*
     * left intentionally some space to improve the circular_buffer<T> implementation here
     */
    {
        HistoryBuffer<int> buffer(32);

        "history_buffer<int>(32)"_benchmark.repeat<n_repetitions>(samples) = [&buffer] {
            static int counter = 0;
            for (std::size_t i = 0; i < samples; i++) {
                buffer.push_back(counter);
                if (const auto data = buffer[0] != counter) {
                    throw std::runtime_error(fmt::format("write {} read {} mismatch", counter, data));
                }
                counter++;
            }
        };
    }
    {
        HistoryBuffer<int> buffer(32);

        "history_buffer<int, 32>"_benchmark.repeat<n_repetitions>(samples) = [&buffer] {
            static int counter = 0;
            for (std::size_t i = 0; i < samples; i++) {
                buffer.push_back(counter);
                if (const auto data = buffer[0] != counter) {
                    throw std::runtime_error(fmt::format("write {} read {} mismatch", counter, data));
                }
                counter++;
            }
        };
    }

    {
        CircularBuffer<int> buffer(32);
        auto                writer                                                      = buffer.new_writer();
        auto                reader                                                      = buffer.new_reader();

        "circular_buffer<int>(32) - no checks"_benchmark.repeat<n_repetitions>(samples) = [&writer, &reader] {
            static int counter    = 0;
            auto       write_data = writer.reserve_output_range(1LU);
            write_data[0]         = counter;
            for (std::size_t i = 0; i < samples; i++) {
                write_data.publish(1);
                [[maybe_unused]] const auto data = reader.get(1)[0];
                [[maybe_unused]] const auto ret  = reader.consume(1);
                counter++;
            }
        };
    }
    {
        HistoryBuffer<int, 32> buffer;

        "history_buffer<int, 32>  - no checks"_benchmark.repeat<n_repetitions>(samples) = [&buffer] {
            static int counter = 0;
            for (std::size_t i = 0; i < samples; i++) {
                buffer.push_back(counter);
                [[maybe_unused]] const auto data = buffer[0];
                counter++;
            }
        };
    }
    {
        HistoryBuffer<int> buffer(32);

        "history_buffer<int>(32)  - no checks"_benchmark.repeat<n_repetitions>(samples) = [&buffer] {
            static int counter = 0;
            for (std::size_t i = 0; i < samples; i++) {
                buffer.push_back(counter);
                [[maybe_unused]] const auto data = buffer[0];
                counter++;
            }
        };
    }
};

int
main() { /* not needed by the UT framework */
}