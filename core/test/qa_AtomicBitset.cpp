#include <boost/ut.hpp>

#include <algorithm>
#include <array>
#include <complex>
#include <format>
#include <numeric>
#include <ranges>
#include <tuple>

#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/BufferSkeleton.hpp>
#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/Sequence.hpp>
#include <gnuradio-4.0/WaitStrategy.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>

template<typename TBitset>
void runAtomicBitsetTest(TBitset& bitset, std::size_t bitsetSize) {
    using namespace boost::ut;

    expect(eq(bitset.size(), bitsetSize)) << std::format("Bitset size should be {}", bitsetSize);

    // test default values
    for (std::size_t i = 0; i < bitset.size(); i++) {
        expect(!bitset.test(i)) << std::format("Bit {} should be false", i);
    }

    // set true for test positions
    std::vector<std::size_t> testPositions = {0UZ, 1UZ, 5UZ, 31UZ, 32UZ, 47UZ, 63UZ, 64UZ, 100UZ, 127UZ};
    for (const std::size_t pos : testPositions) {
        bitset.set(pos);
    }

    // only test positions should be set
    for (std::size_t i = 0; i < bitset.size(); i++) {
        if (std::ranges::find(testPositions, i) != testPositions.end()) {
            expect(bitset.test(i)) << std::format("Bit {} should be set", i);
        } else {
            expect(!bitset.test(i)) << std::format("Bit {} should be false", i);
        }
    }

    // reset test positions
    for (const std::size_t pos : testPositions) {
        bitset.reset(pos);
    }

    // all positions should be reset
    for (std::size_t i = 0; i < bitset.size(); i++) {
        expect(!bitset.test(i)) << std::format("Bit {} should be false", i);
    }

    // Bulk operations
    std::vector<std::pair<std::size_t, std::size_t>> testPositionsBulk = {{10UZ, 20UZ}, {10UZ, 10UZ}, {50UZ, 70UZ}, {0UZ, 127UZ}, {0UZ, 128UZ}, {63UZ, 64UZ}, {127UZ, 128UZ}, {0UZ, 1UZ}, {128UZ, 128UZ}};
    for (const auto& pos : testPositionsBulk) {
        bitset.set(pos.first, pos.second);

        for (std::size_t i = 0; i < bitset.size(); ++i) {
            if (i >= pos.first && i < pos.second) {
                expect(bitset.test(i)) << std::format("Bulk [{},{}) Bit {} should be true", pos.first, pos.second, i);
            } else {
                expect(!bitset.test(i)) << std::format("Bulk [{},{}) Bit {} should be false", pos.first, pos.second, i);
            }
        }

        // all positions should be reset
        bitset.reset(pos.first, pos.second);
        for (std::size_t i = 0; i < bitset.size(); i++) {
            expect(!bitset.test(i)) << std::format("Bulk [{},{}) Bit {} should be false", pos.first, pos.second, i);
        }
    }

#if not defined(__EMSCRIPTEN__) && not defined(NDEBUG) && not defined(_WIN32)
    expect(aborts([&] { bitset.set(bitsetSize); })) << "Setting bit should throw an assertion.";
    expect(aborts([&] { bitset.reset(bitsetSize); })) << "Resetting bit should throw an assertion.";
    expect(aborts([&] { bitset.test(bitsetSize); })) << "Testing bit should throw an assertion.";
    // bulk operations
    expect(aborts([&] { bitset.set(100UZ, 200UZ); })) << "Setting bulk bits should throw an assertion.";
    expect(aborts([&] { bitset.reset(100UZ, 200UZ); })) << "Resetting bulk bits should throw an assertion.";
    expect(aborts([&] { bitset.set(200UZ, 100UZ); })) << "Setting bulk begin > end should throw an assertion.";
    expect(aborts([&] { bitset.reset(200UZ, 100UZ); })) << "Resetting bulk begin > end should throw an assertion.";
#endif
}

const boost::ut::suite AtomicBitsetTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using namespace gr::detail;

    "basics set/reset/test"_test = []() {
        auto dynamicBitset = AtomicBitset<>(128UZ);
        runAtomicBitsetTest(dynamicBitset, 128UZ);

        auto staticBitset = AtomicBitset<128UZ>();
        runAtomicBitsetTest(staticBitset, 128UZ);
    };

    "multithreads"_test = [] {
        constexpr std::size_t    bitsetSize = 256UZ;
        constexpr std::size_t    nThreads   = 16UZ;
        constexpr std::size_t    nRepeats   = 100UZ;
        AtomicBitset<bitsetSize> bitset;
        std::vector<std::thread> threads;

        for (std::size_t iThread = 0; iThread < nThreads; iThread++) {
            threads.emplace_back([&] {
                for (std::size_t iR = 0; iR < nRepeats; iR++) {
                    for (std::size_t i = 0; i < bitsetSize; i++) {
                        if (i < bitsetSize / 2) {
                            bitset.set(i);
                            bitset.reset(i);
                            std::ignore = bitset.test(i);
                        } else {
                            bitset.reset(i);
                            bitset.set(i);
                            std::ignore = bitset.test(i);
                        }
                    }
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        // Verify final state: first half should be reset, second half should be set
        for (std::size_t i = 0; i < bitsetSize; i++) {
            if (i < bitsetSize / 2) {
                expect(!bitset.test(i)) << std::format("Bit {} should be reset", i);
            } else {
                expect(bitset.test(i)) << std::format("Bit {} should be set", i);
            }
        }
    };

    "multithreads bulk"_test = [] {
        constexpr std::size_t    bitsetSize = 2000UZ;
        constexpr std::size_t    nThreads   = 10UZ;
        constexpr std::size_t    chunkSize  = bitsetSize / nThreads;
        constexpr std::size_t    nRepeats   = 1000UZ;
        AtomicBitset<bitsetSize> bitset;
        std::vector<std::thread> threads;

        for (std::size_t iThread = 0; iThread < nThreads; iThread++) {
            threads.emplace_back([&bitset, iThread] {
                for (std::size_t iR = 0; iR < nRepeats; iR++) {
                    if (iThread % 2 == 0) {
                        bitset.set(iThread * chunkSize, (iThread + 1) * chunkSize);
                        bitset.reset(iThread * chunkSize, (iThread + 1) * chunkSize);
                    } else {
                        bitset.reset(iThread * chunkSize, (iThread + 1) * chunkSize);
                        bitset.set(iThread * chunkSize, (iThread + 1) * chunkSize);
                    }
                }
            });
        }

        for (auto& thread : threads) {
            thread.join();
        }

        // Verify final state
        for (std::size_t i = 0; i < bitsetSize; i++) {
            if ((i / chunkSize) % 2 == 0) {
                expect(!bitset.test(i)) << std::format("Bit {} should be reset", i);
            } else {
                expect(bitset.test(i)) << std::format("Bit {} should be set", i);
            }
        }
    };
};

int main() { /* not needed for UT */ }
