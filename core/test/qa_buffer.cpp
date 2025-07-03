#include <algorithm>
#include <array>
#include <complex>
#include <numeric>
#include <ranges>
#include <tuple>

#include <boost/ut.hpp>

#include <gnuradio-4.0/meta/formatter.hpp>

#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/BufferSkeleton.hpp>
#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/HistoryBuffer.hpp>
#include <gnuradio-4.0/Sequence.hpp>
#include <gnuradio-4.0/WaitStrategy.hpp>
#include <gnuradio-4.0/thread/thread_affinity.hpp>

template<gr::WaitStrategyLike auto wait = gr::NoWaitStrategy()>
struct TestStruct {
    [[nodiscard]] constexpr bool test() const noexcept { return true; }
};

struct AllocatorPortable {};
struct AllocatorPosix {};
using CircularBufferSingle = gr::CircularBuffer<int32_t, std::dynamic_extent, gr::ProducerType::Single>;
using CircularBufferMulti  = gr::CircularBuffer<int32_t, std::dynamic_extent, gr::ProducerType::Multi>;

template<typename TCircularBuffer, typename TAllocator>
struct CircularBufferTestTypes {
    using CircularBuffer = TCircularBuffer;
    using Allocator      = TAllocator;

    constexpr static bool isMulti = std::is_same_v<TCircularBuffer, CircularBufferMulti>;
    constexpr static bool isPosix = std::is_same_v<TAllocator, AllocatorPosix>;
};

using CircularBufferTypesToTest = std::tuple< //
#ifdef HAS_POSIX_MAP_INTERFACE
    CircularBufferTestTypes<CircularBufferSingle, AllocatorPosix>, //
    CircularBufferTestTypes<CircularBufferMulti, AllocatorPosix>,  //
#endif
    CircularBufferTestTypes<CircularBufferSingle, AllocatorPortable>, //
    CircularBufferTestTypes<CircularBufferMulti, AllocatorPortable>>;

const boost::ut::suite BasicConceptsTests = [] {
    using namespace boost::ut;

    "BasicConcepts"_test = []<typename T> {
        using namespace gr;

        const auto      allocator = (std::is_same_v<typename T::Allocator, AllocatorPosix>) ? gr::double_mapped_memory_resource::allocator<int32_t>() : std::pmr::polymorphic_allocator<int32_t>();
        BufferLike auto buffer    = typename T::CircularBuffer(1024, allocator);

        auto typeName = std::format("test({}-{})", reflection::type_name<typename T::CircularBuffer>(), reflection::type_name<typename T::Allocator>());
        std::println("BasicConcepts: {}", typeName);

        // N.B. GE because some buffers need to intrinsically allocate more to meet e.g. page-size requirements
        expect(ge(buffer.size(), 1024UZ)) << "for " << typeName << "\n";

        // compile-time interface tests
        BufferReaderLike auto reader = buffer.new_reader(); // tests matching read concept
        BufferWriterLike auto writer = buffer.new_writer(); // tests matching write concept

        static_assert(std::is_same_v<decltype(reader.buffer().new_reader()), decltype(reader)>);
        static_assert(std::is_same_v<decltype(reader.buffer().new_writer()), decltype(writer)>);
        static_assert(std::is_same_v<decltype(writer.buffer().new_writer()), decltype(writer)>);
        static_assert(std::is_same_v<decltype(writer.buffer().new_reader()), decltype(reader)>);

        // runtime interface tests
        expect(eq(reader.available(), 0UZ));
        expect(eq(reader.position(), kInitialCursorValue));
        ReaderSpanLike auto cSpan = reader.get(0UZ);
        expect(nothrow([&cSpan] { expect(eq(cSpan.size(), 0UZ)); })) << typeName << "throws" << "\n";
        expect(nothrow([&cSpan] { expect(cSpan.consume(0UZ)); }));

        expect(writer.available() >= buffer.size());

        {
            WriterSpanLike auto value = writer.tryReserve(1);
            expect(eq(1LU, value.size())) << "for " << typeName << "\n";
            value.publish(1);
        }

        { // publish remaining slots
            WriterSpanLike auto value = writer.template tryReserve<SpanReleasePolicy::ProcessAll>(1023);
            expect(eq(1023UZ, value.size())) << "for " << typeName << "\n";
        }

        if constexpr (!T::isPosix) { // no more available slots, some buffers intrinsically allocate more to meet e.g. page-size requirements
            WriterSpanLike auto value = writer.template tryReserve<SpanReleasePolicy::ProcessAll>(10);
            expect(eq(0UZ, value.size())) << "for " << typeName << "\n";
        }
    } | CircularBufferTypesToTest();
};

const boost::ut::suite SequenceTests = [] {
    using namespace boost::ut;

    "Sequence"_test = [] {
        using namespace gr;
#if not defined(__APPLE__)
        expect(eq(alignof(Sequence), 64UZ));
#endif
        expect(eq(0UZ, kInitialCursorValue));
        expect(nothrow([] { Sequence(); }));
        expect(nothrow([] { Sequence(2); }));

        auto s1 = Sequence();
        expect(eq(s1.value(), kInitialCursorValue));

        const auto s2 = Sequence(2);
        expect(eq(s2.value(), 2UZ));

        expect(nothrow([&s1] { s1.setValue(3); }));
        expect(eq(s1.value(), 3UZ));

        expect(nothrow([&s1] { expect(s1.compareAndSet(3, 4)); }));
        expect(nothrow([&s1] { expect(eq(s1.value(), 4UZ)); }));
        expect(nothrow([&s1] { expect(!s1.compareAndSet(3, 5)); }));
        expect(eq(s1.value(), 4UZ));

        expect(eq(s1.incrementAndGet(), 5UZ));
        expect(eq(s1.value(), 5UZ));
        expect(eq(s1.addAndGet(2), 7UZ));
        expect(eq(s1.value(), 7UZ));

        std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> sequences{std::make_shared<std::vector<std::shared_ptr<Sequence>>>()};
        expect(eq(gr::detail::getMinimumSequence(*sequences), std::numeric_limits<std::size_t>::max()));
        expect(eq(gr::detail::getMinimumSequence(*sequences, 2), 2UZ));
        sequences->emplace_back(std::make_shared<Sequence>(4));
        expect(eq(gr::detail::getMinimumSequence(*sequences), 4UZ));
        expect(eq(gr::detail::getMinimumSequence(*sequences, 5), 4UZ));
        expect(eq(gr::detail::getMinimumSequence(*sequences, 2), 2UZ));

        auto cursor = std::make_shared<Sequence>(10);
        auto s3     = std::make_shared<Sequence>(1);
        expect(eq(sequences->size(), 1UZ));
        expect(eq(gr::detail::getMinimumSequence(*sequences), 4UZ));
        expect(nothrow([&sequences, &cursor, &s3] { gr::detail::addSequences(sequences, *cursor, {s3}); }));
        expect(eq(sequences->size(), 2UZ));
        // newly added sequences are set automatically to the cursor/write position
        expect(eq(s3->value(), 10UZ));
        expect(eq(gr::detail::getMinimumSequence(*sequences), 4UZ));

        expect(nothrow([&sequences, &cursor] { gr::detail::removeSequence(sequences, cursor); }));
        expect(eq(sequences->size(), 2UZ));
        expect(nothrow([&sequences, &s3] { gr::detail::removeSequence(sequences, s3); }));
        expect(eq(sequences->size(), 1UZ));

        std::stringstream ss;
        expect(eq(ss.str().size(), 0UZ));
        expect(nothrow([&ss, &s3] { ss << std::format("{}", *s3); }));
        expect(not ss.str().empty());
    };
};

#if defined(HAS_POSIX_MAP_INTERFACE) && !defined(_GLIBCXX_DEBUG)
const boost::ut::suite DoubleMappedAllocatorTests = [] {
    using namespace boost::ut;

    "DoubleMappedAllocator"_test = [] {
        using Allocator                                       = std::pmr::polymorphic_allocator<int32_t>;
        std::size_t                     size                  = static_cast<std::size_t>(getpagesize()) / sizeof(int32_t);
        auto                            doubleMappedAllocator = gr::double_mapped_memory_resource::allocator<int32_t>();
        std::vector<int32_t, Allocator> vec(size, doubleMappedAllocator);
        expect(eq(vec.size(), size));
        std::iota(vec.begin(), vec.end(), 1);
        for (std::size_t i = 0UZ; i < vec.size(); ++i) {
            expect(eq(vec[i], static_cast<std::int32_t>(i + 1)));
            // to note: can safely read beyond size for this special vector
            expect(eq(vec[size + i], vec[i])); // identical to mirrored copy
        }
    };
};
#endif

template<typename Writer, std::size_t N>
void writeVaryingChunkSizes(Writer& writer, std::size_t writerID) {
    gr::thread_pool::thread::setThreadName(std::format("writer#{}", writerID));
    std::size_t pos    = 0;
    std::size_t iWrite = 0;
    while (pos < N) {
        constexpr auto          kChunkSizes = std::array{1UZ, 2UZ, 3UZ, 5UZ, 7UZ, 42UZ};
        const auto              chunkSize   = std::min(kChunkSizes[iWrite % kChunkSizes.size()], N - pos);
        gr::WriterSpanLike auto out         = writer.tryReserve(chunkSize);
        if (out.size() != 0) {
            boost::ut::expect(boost::ut::eq(writer.nRequestedSamplesToPublish(), 0UZ));
            for (std::size_t i = 0UZ; i < out.size(); i++) {
                out[i] = {{0, static_cast<int>(pos + i)}};
            }
            out.publish(out.size());

            boost::ut::expect(boost::ut::eq(writer.nRequestedSamplesToPublish(), chunkSize));
            pos += chunkSize;
            ++iWrite;
        }
    }
}

const boost::ut::suite WaitStrategiesTests = [] {
    using namespace boost::ut;

    "WaitStrategies"_test = [] {
        using namespace gr;

        expect(isWaitStrategy<BlockingWaitStrategy>);
        expect(isWaitStrategy<BusySpinWaitStrategy>);
        expect(isWaitStrategy<SleepingWaitStrategy>);
        expect(isWaitStrategy<SleepingWaitStrategy>);
        expect(isWaitStrategy<SpinWaitWaitStrategy>);
        expect(isWaitStrategy<TimeoutBlockingWaitStrategy>);
        expect(isWaitStrategy<YieldingWaitStrategy>);
        expect(not isWaitStrategy<int>);

        expect(WaitStrategyLike<BlockingWaitStrategy>);
        expect(WaitStrategyLike<BusySpinWaitStrategy>);
        expect(WaitStrategyLike<SleepingWaitStrategy>);
        expect(WaitStrategyLike<SleepingWaitStrategy>);
        expect(WaitStrategyLike<SpinWaitWaitStrategy>);
        expect(WaitStrategyLike<TimeoutBlockingWaitStrategy>);
        expect(WaitStrategyLike<YieldingWaitStrategy>);
        expect(not WaitStrategyLike<int>);

        TestStruct a;
        expect(a.test());
    };
};

const boost::ut::suite UserApiExamples = [] {
    using namespace boost::ut;

    "UserApi"_test = [] {
        using namespace gr;
        BufferLike auto buffer = CircularBuffer<int32_t>(1024);

        BufferWriterLike auto writer = buffer.new_writer();
        { // source only write example
            BufferReaderLike auto localReader = buffer.new_reader();
            expect(eq(localReader.available(), 0UZ));

            expect(ge(writer.available(), buffer.size()));
            {
                WriterSpanLike auto pSpan = writer.tryReserve(10);
                expect(eq(pSpan.size(), 10UZ));
                pSpan.publish(10);
            }
            expect(eq(writer.available(), buffer.size() - 10UZ));
            expect(eq(localReader.available(), 10UZ));
            expect(eq(buffer.n_readers(), 1UZ)); // N.B. circular_buffer<..> specific
        }
        expect(eq(buffer.n_readers(), 0UZ)); // reader not in scope release atomic reader index

        BufferReaderLike auto reader = buffer.new_reader();
        // reader does not know about previous submitted data as it joined only after
        // data has been written <-> needed for thread-safe joining of readers while writing
        expect(eq(reader.available(), 0UZ));
        // populate with some more data
        for (std::size_t i = 0; i < 3; i++) {
            {
                WriterSpanLike auto pSpan = writer.tryReserve(5);
                expect(eq(pSpan.size(), 5UZ));
                pSpan.publish(5);
            }
            expect(eq(reader.available(), (i + 1) * 5)) << std::format("iteration: {}", i);
        }

        // N.B. here using a simple read-only (sink) example:
        for (int i = 0; reader.available() != 0; i++) {
            ReaderSpanLike auto fixedLength = reader.get(3); // 'std::span<const int32_t> fixedLength' is not allowed explicitly
            ReaderSpanLike auto available   = reader.get();
            std::print("iteration {} - fixed-size data[{:2}]: [{}]\n", i, fixedLength.size(), gr::join(fixedLength, ", "));
            std::print("iteration {} - full-size  data[{:2}]: [{}]\n", i, available.size(), gr::join(available, ", "));

            // consume data -> allows corresponding buffer to be overwritten by writer
            // if there are no other reader claiming that buffer segment
            if (fixedLength.consume(fixedLength.size())) {
                // for info-only - since available() can change in parallel
                // N.B. lock-free buffer and other writer may add while processing
                std::print("iteration {} - consumed {} elements - still available: {}\n", i, fixedLength.size(), reader.available());
            } else {
                throw std::runtime_error("could not consume data");
            }
        }
    };
};

const boost::ut::suite CircularBufferTests = [] {
    using namespace boost::ut;
    using namespace gr;

    "CircularBuffer"_test = []<typename T>() {
        const auto      allocator = (std::is_same_v<typename T::Allocator, AllocatorPosix>) ? gr::double_mapped_memory_resource::allocator<int32_t>() : std::pmr::polymorphic_allocator<int32_t>();
        BufferLike auto buffer    = typename T::CircularBuffer(1024, allocator);

        std::println("CircularBufferTest: {}", std::format("test({}-{})", gr::meta::type_name<typename T::CircularBuffer>(), gr::meta::type_name<typename T::Allocator>()));

        expect(ge(buffer.size(), 1024u));

        BufferWriterLike auto writer = buffer.new_writer();
        expect(nothrow([&writer] { expect(eq(writer.buffer().n_readers(), 0UZ)); })); // no reader, just writer
        BufferReaderLike auto reader = buffer.new_reader();
        expect(nothrow([&reader] { expect(eq(reader.buffer().n_readers(), 1UZ)); })); // created one reader

        expect(eq(reader.available(), 0UZ));
        expect(eq(reader.get().size(), 0UZ));
#if not defined(__EMSCRIPTEN__) && not defined(NDEBUG) && not defined(_WIN32)
        expect(aborts([&reader] { std::ignore = reader.get(1); }));
#endif
        expect(eq(writer.available(), buffer.size()));
        expect(nothrow([&writer, &buffer] {
            WriterSpanLike auto pSpan = writer.tryReserve(buffer.size());
            pSpan.publish(buffer.size());
        })); // fully fill buffer
        expect(eq(writer.available(), 0UZ));
        expect(eq(reader.available(), buffer.size()));
        expect(eq(reader.get().size(), buffer.size()));
        {
            ReaderSpanLike auto inSpan = reader.get(2);
            expect(eq(inSpan.size(), 2UZ));
            expect(eq(reader.nSamplesConsumed(), 0UZ));
            {
                // Subsequent calls to get(), without calling consume() again, will return maximum of _nSamplesFirstGet (2)
                ReaderSpanLike auto inSpan2 = reader.get(3);
                expect(eq(inSpan2.size(), 2UZ));
                expect(eq(reader.nSamplesConsumed(), 0UZ));
                {
                    ReaderSpanLike auto inSpan3 = reader.get(1);
                    expect(eq(inSpan3.size(), 1UZ));
                    expect(eq(reader.nSamplesConsumed(), 0UZ));
                }
            }
            expect(eq(reader.nSamplesConsumed(), 0UZ));
            expect(!inSpan.isConsumeRequested());

            expect(inSpan.consume(0UZ));
            expect(inSpan.isConsumeRequested());
            expect(eq(reader.nSamplesConsumed(), 0UZ));
        }
        expect(eq(reader.nSamplesConsumed(), 0UZ));
        expect(!reader.isConsumeRequested());
        expect(eq(reader.available(), buffer.size()));

#if not defined(__EMSCRIPTEN__) && not defined(NDEBUG) && not defined(_WIN32)
        expect(aborts([&reader] {
            {
                ReaderSpanLike auto inSpan4 = reader.template get<SpanReleasePolicy::Terminate>(3);
                expect(eq(inSpan4.size(), 3UZ));
                expect(!inSpan4.isConsumeRequested());
            }
        }));
#endif
        {
            ReaderSpanLike auto inSpan5 = reader.template get<SpanReleasePolicy::ProcessNone>(3);
            expect(eq(inSpan5.size(), 3UZ));
            expect(!inSpan5.isConsumeRequested());
        }
        expect(eq(reader.nSamplesConsumed(), 0UZ));
        expect(!reader.isConsumeRequested());
        expect(eq(reader.available(), buffer.size()));
        std::size_t inSpan6Size{0UZ};
        {
            ReaderSpanLike auto inSpan6 = reader.template get<SpanReleasePolicy::ProcessAll>();
            inSpan6Size                 = inSpan6.size();
            expect(eq(inSpan6.size(), reader.available()));
            expect(!inSpan6.isConsumeRequested());
        }
        expect(eq(reader.nSamplesConsumed(), inSpan6Size));
        expect(!reader.isConsumeRequested());
        expect(eq(reader.available(), 0UZ));

        expect(eq(writer.available(), buffer.size()));

        // test buffer wrap around twice
        std::size_t counter = 1;
        for (const int _blockSize : {1, 2, 3, 5, 7, 42}) {
            auto blockSize = static_cast<std::size_t>(_blockSize);
            for (std::size_t i = 0; i < buffer.size(); i++) {
                if (i != 0) {
                    expect(eq(reader.nSamplesConsumed(), blockSize));
                }
                {
                    WriterSpanLike auto pSpan = writer.template tryReserve<SpanReleasePolicy::ProcessAll>(blockSize);
                    expect(eq(pSpan.size(), blockSize));
                    std::iota(pSpan.begin(), pSpan.end(), counter += pSpan.size());
                }
                ReaderSpanLike auto cSpan = reader.get(blockSize);
                expect(eq(cSpan.size(), blockSize));
                expect(eq(cSpan.front(), static_cast<int>(counter)));
                expect(eq(cSpan.back(), static_cast<int>(counter + blockSize - 1)));
                expect(cSpan.consume(blockSize));
                expect(eq(reader.nSamplesConsumed(), 0UZ));
            }
        }
        // basic expert writer api
        for (int k = 0; k < 3; k++) {
            // case 0: write fully reserved data
            {
                WriterSpanLike auto pSpan = writer.tryReserve(4);
                expect(eq(pSpan.size(), 4UZ));
                expect(eq(writer.nRequestedSamplesToPublish(), 0UZ));
                for (std::size_t i = 0; i < pSpan.size(); i++) {
                    pSpan[i] = static_cast<int>(i + 1);
                }
                pSpan.publish(4);
                expect(eq(writer.nRequestedSamplesToPublish(), 4UZ));
            }
            ReaderSpanLike auto cSpan = reader.get();
            expect(eq(cSpan.size(), 4UZ));
            for (std::size_t i = 0; i < cSpan.size(); i++) {
                expect(eq(static_cast<int>(i + 1), cSpan[i])) << "case 0: read index " << i;
            }
            expect(cSpan.consume(4));
        }

        if constexpr (!T::isMulti) { // MultiThreaded does not allow to publish less than reserved
            for (int k = 0; k < 3; k++) {
                // case 1: reserve more than actually written
                const auto cursor_initial = buffer.cursor_sequence().value();
                {
                    WriterSpanLike auto pSpan = writer.tryReserve(4);
                    expect(eq(pSpan.size(), 4UZ));
                    expect(eq(writer.nRequestedSamplesToPublish(), 0UZ));
                    for (std::size_t i = 0; i < pSpan.size(); i++) {
                        pSpan[i] = static_cast<int>(i + 1);
                    }
                    pSpan.publish(2);
                    expect(eq(writer.nRequestedSamplesToPublish(), 2UZ));
                }
                const auto cursor_after = buffer.cursor_sequence().value();
                expect(eq(cursor_initial + 2, cursor_after)) << std::format("cursor sequence moving by two: {} -> {}", cursor_initial, cursor_after);
                ReaderSpanLike auto cSpan = reader.get();
                expect(eq(2UZ, cSpan.size())) << std::format("received {} samples instead of expected 2", cSpan.size());
                for (std::size_t i = 0; i < cSpan.size(); i++) {
                    expect(eq(static_cast<int>(i + 1), cSpan[i])) << "read 1: index " << i;
                }
                expect(cSpan.consume(2));
            }

            for (int k = 0; k < 3; k++) {
                // case 2: reserve using RAII token
                const auto cursor_initial = buffer.cursor_sequence().value();
                {
                    WriterSpanLike auto pSpan = writer.tryReserve(4);
                    expect(eq(pSpan.size(), 4UZ));
                    expect(eq(writer.nRequestedSamplesToPublish(), 0UZ));
                    std::span<int32_t> span = pSpan; // tests conversion operator
                    for (std::size_t i = 0; i < pSpan.size(); i++) {
                        pSpan[i] = static_cast<int>(i + 1);
                        expect(eq(pSpan[i], span[i]));
                    }
                    pSpan.publish(2);
                    expect(eq(writer.nRequestedSamplesToPublish(), 2UZ));
                }
                const auto cursor_after = buffer.cursor_sequence().value();
                expect(eq(cursor_initial + 2, cursor_after)) << std::format("cursor sequence moving by two: {} -> {}", cursor_initial, cursor_after);
                ReaderSpanLike auto cSpan = reader.get();
                expect(eq(2UZ, cSpan.size())) << std::format("received {} samples instead of expected 2", cSpan.size());
                for (std::size_t i = 0; i < cSpan.size(); i++) {
                    expect(eq(static_cast<int>(i + 1), cSpan[i])) << "read 1: index " << i;
                }
                expect(cSpan.consume(2));
            }
        }
    } | CircularBufferTypesToTest();

    "MultiProducerStdMapSingleWriter"_test = [] {
        // Using std::map exposed some race conditions in the multi-producer buffer implementation
        // that did not surface with trivial types. (two readers for good measure, issues occurred also
        // with single reader)
        gr::CircularBuffer<std::map<int, int>, std::dynamic_extent, gr::ProducerType::Multi> buffer(1024);

        gr::BufferWriterLike auto writer  = buffer.new_writer();
        gr::BufferReaderLike auto reader1 = buffer.new_reader();
        gr::BufferReaderLike auto reader2 = buffer.new_reader();

        constexpr auto kWrites      = 200000UZ;
        auto           writerThread = std::thread(&writeVaryingChunkSizes<decltype(writer), kWrites>, std::ref(writer), 0Uz);

        auto readerFnc = [](auto reader, std::size_t readerID) {
            gr::thread_pool::thread::setThreadName(std::format("reader#{}", readerID));
            std::size_t i = 0;
            while (i < kWrites) {
                auto in = reader.get().get();
                for (auto j = 0UZ; j < in.size(); j++) {
                    auto vIt = in[j].find(0);
                    expect(vIt != in[j].end());
                    if (vIt != in[j].end()) {
                        expect(eq(vIt->second, static_cast<int>(i)));
                    }
                    i++;
                }
                expect(in.consume(in.size()));
            }
        };

        auto reader1Thread = std::thread(readerFnc, std::ref(reader1), 0UZ);
        auto reader2Thread = std::thread(readerFnc, std::ref(reader2), 1UZ);
        writerThread.join();
        reader1Thread.join();
        reader2Thread.join();
    };

    "MultiProducerStdMapMultipleWriters"_test = [] {
        // now actually use multiple writers, and ensure we see all expected values, in a valid order.
        constexpr auto kNWriters = 5UZ;
        constexpr auto kWrites   = 20000UZ;

        gr::CircularBuffer<std::map<int, int>, std::dynamic_extent, gr::ProducerType::Multi> buffer(1024);
        using WriterType                  = decltype(buffer.new_writer());
        gr::BufferReaderLike auto reader1 = buffer.new_reader();
        gr::BufferReaderLike auto reader2 = buffer.new_reader();

        std::vector<WriterType> writers;
        for (std::size_t i = 0UZ; i < kNWriters; i++) {
            writers.push_back(buffer.new_writer());
        }

        std::array<std::thread, kNWriters> writerThreads;
        for (std::size_t i = 0UZ; i < kNWriters; i++) {
            writerThreads[i] = std::thread(&writeVaryingChunkSizes<decltype(writers[i]), kWrites>, std::ref(writers[i]), i);
        }

        auto readerFnc = [](auto reader) {
            std::array<int, kNWriters> next;
            std::ranges::fill(next, 0);
            std::size_t read = 0UZ;
            while (read < kWrites * kNWriters) {
                auto in = reader.get().get();
                for (const auto& map : in) {
                    auto vIt = map.find(0);
                    expect(vIt != map.end()) << "map does not contain zero";
                    if (vIt == map.end()) {
                        continue;
                    }
                    const auto value = vIt->second;
                    expect(ge(value, 0)) << "value in map should be greater than zero";
                    expect(le(value, static_cast<int>(kWrites))) << "value in map should be smaller than number of samples to publish";
                    const auto nextIt = std::ranges::find(next, value);
                    expect(nextIt != next.end()) << "No writer thread waiting for that number";
                    if (nextIt == next.end()) {
                        continue;
                    }
                    *nextIt = value + 1;
                }
                read += in.size();
                expect(in.consume(in.size())) << "Failed to consume all";
            }
        };

        auto reader1Thread = std::thread(readerFnc, std::ref(reader1));
        auto reader2Thread = std::thread(readerFnc, std::ref(reader2));
        for (std::size_t i = 0; i < kNWriters; i++) {
            writerThreads[i].join();
        }
        reader1Thread.join();
        reader2Thread.join();
    };
};

const boost::ut::suite UserDefinedTypeCasting = [] {
    using namespace boost::ut;
    "UserDefinedTypeCasting"_test = [] {
        using namespace gr;
        BufferLike auto buffer = CircularBuffer<std::complex<float>>(1024);
        expect(ge(buffer.size(), 1024u));

        BufferWriterLike auto writer = buffer.new_writer();
        BufferReaderLike auto reader = buffer.new_reader();
        {
            WriterSpanLike auto pSpan = writer.tryReserve<SpanReleasePolicy::ProcessAll>(2);
            expect(eq(pSpan.size(), 2UZ));
            pSpan[0] = std::complex(1.0f, -1.0f);
            pSpan[1] = std::complex(2.0f, -2.0f);
        }
        expect(eq(reader.available(), 2UZ));
        {
            ReaderSpanLike auto cSpan = reader.get(reader.available());
            expect(eq(cSpan.size(), 2UZ));

            auto const const_bytes = std::as_bytes(static_cast<std::span<const std::complex<float>>>(cSpan));
            expect(eq(const_bytes.size(), cSpan.size() * sizeof(std::complex<float>)));

            auto convertToFloatSpan = [](std::span<const std::complex<float>>& c) -> std::span<const float> {
                return {reinterpret_cast<const float*>(c.data()), c.size() * 2}; // NOSONAR(cpp:S3630) //NOPMD needed
            };
            auto floatArray = convertToFloatSpan(cSpan);
            expect(eq(floatArray[0], +1.0f));
            expect(eq(floatArray[1], -1.0f));
            expect(eq(floatArray[2], +2.0f));
            expect(eq(floatArray[3], -2.0f));

            expect(cSpan.consume(cSpan.size()));
            expect(eq(reader.available(), cSpan.size()));
        }
        expect(eq(reader.available(), 0UZ)); // needed otherwise buffer write will not be called
    };
};

const boost::ut::suite StreamTagConcept = [] {
    using namespace boost::ut;

    "StreamTagConcept"_test = [] {
        // implements a proof-of-concept how stream-tags could be dealt with
        using namespace gr;
        struct alignas(gr::hardware_destructive_interference_size) buffer_tag {
            // N.B. type need to be favourably sized e.g. 1 or a power of 2
            // -> otherwise the automatic buffer sizes are getting very large
            int64_t     index;
            std::string data;
        };
#if not defined(__APPLE__)
        expect(eq(sizeof(buffer_tag), 64UZ)) << "tag size";
#endif
        BufferLike auto buffer    = CircularBuffer<int32_t>(1024);
        BufferLike auto tagBuffer = CircularBuffer<buffer_tag>(32);
        expect(ge(buffer.size(), 1024u));
        expect(ge(tagBuffer.size(), 32u));

        BufferWriterLike auto writer    = buffer.new_writer();
        BufferReaderLike auto reader    = buffer.new_reader();
        BufferWriterLike auto tagWriter = tagBuffer.new_writer();
        BufferReaderLike auto tagReader = tagBuffer.new_reader();

        for (int i = 0; i < 3; i++) { // write-only worker (source) mock-up
            WriterSpanLike auto pSpan = writer.tryReserve<SpanReleasePolicy::ProcessAll>(10);
            expect(eq(pSpan.size(), 10UZ));
            static std::size_t offset = 1;
            std::iota(pSpan.begin(), pSpan.end(), offset);
            offset += pSpan.size();

            WriterSpanLike auto pSpanTag = tagWriter.tryReserve<SpanReleasePolicy::ProcessAll>(1);
            expect(eq(pSpanTag.size(), 1UZ));
            pSpanTag[0] = {static_cast<int64_t>(offset), std::format("<tag data at index {:3}>", offset)};
        }

        { // read-only worker (sink) mock-up
            std::print("read position: {}\n", reader.position());
            ReaderSpanLike auto cSpanData = reader.get(reader.available());
            ReaderSpanLike auto cSpanTags = tagReader.get(tagReader.available());

            std::print("received {} tags\n", cSpanTags.size());
            for (auto& readTag : cSpanTags) {
                std::print("stream-tag @{:3}: '{}'\n", readTag.index, readTag.data);
            }

            expect(cSpanData.consume(cSpanData.size()));
            expect(cSpanTags.consume(cSpanTags.size())); // N.B. consume tag based on expiry
        }
    };
};

const boost::ut::suite NonPowerTwoTests = [] {
    using namespace boost::ut;
    using namespace gr;

    "std::vector<T>"_test = [] {
        using Type                     = std::vector<int>;
        constexpr std::size_t typeSize = sizeof(std::vector<int>);
        expect(not std::has_single_bit(typeSize)) << "type is non-power-of-two";
        BufferLike auto buffer = CircularBuffer<Type>(1024);
        expect(ge(buffer.size(), 1024u));

        BufferWriterLike auto writer = buffer.new_writer();
        BufferReaderLike auto reader = buffer.new_reader();

        const auto genSamples = [&buffer, &writer] {
            for (std::size_t i = 0UZ; i < buffer.size() - 10UZ; i++) { // write-only worker (source) mock-up
                WriterSpanLike auto pSpan = writer.tryReserve<SpanReleasePolicy::ProcessAll>(1);
                expect(eq(pSpan.size(), 1UZ));
                static int offset = 0;
                for (auto& vector : pSpan) {
                    vector.resize(1);
                    vector[0] = offset++;
                }
            }
        };

        const auto readSamples = [&reader] {
            while (reader.available()) {
                ReaderSpanLike auto cSpan = reader.get(reader.available());
                for (auto& vector : cSpan) {
                    static int offset = -1;
                    expect(eq(vector.size(), 1u)) << "vector size == 1";
                    expect(eq(vector[0] - offset, 1)) << "vector offset == 1";
                    offset = vector[0];
                }
                expect(cSpan.consume(cSpan.size()));
            }
        };

        // write-read twice to test wrap-around
        genSamples();
        readSamples();
        genSamples();
        readSamples();
    };
};

const boost::ut::suite<"Small Buffers"> _smallBufferTests = [] {
    using namespace boost::ut;
    using namespace gr;

    // N.B. small buffers are needed for composite types that themselves can store large amount of data (>1 MB)
    // the default mmap posix buffer size is typically multiple of the memory page size (e.g. 4096) which would
    // make those buffers quite expensive. Thus, if there are composite types (i.e. other than fundamental or
    // std::complex<> types, the buffer implementation should default to the non-posix/STL C++ allocator
    std::println("start small buffer test");
    "std::vector<T>"_test = [] {
        using Type = std::vector<int>;
        static_assert(not std::is_trivially_copyable_v<Type>);
        BufferLike auto buffer = CircularBuffer<Type>(4UZ);
        expect(eq(buffer.size(), 4UZ));
    };
};

const boost::ut::suite HistoryBufferTest = [] {
    using namespace boost::ut;
    using namespace gr;

    "HistoryBuffer<double>"_test = [](const std::size_t& capacity) {
        HistoryBuffer<int> hb(capacity);
        const auto&        const_hb = hb; // tests const access
        expect(eq(hb.capacity(), capacity));
        expect(eq(hb.size(), 0UZ));

        for (std::size_t i = 1; i <= capacity + 1; ++i) {
            hb.push_front(static_cast<int>(i));
        }
        expect(eq(hb.capacity(), capacity));
        expect(eq(hb.size(), capacity));

        expect(eq(hb[0], static_cast<int>(capacity + 1))) << "access the last/actual sample";
        expect(eq(hb[1], static_cast<int>(capacity))) << "access the previous sample";
        expect(eq(const_hb[0], static_cast<int>(capacity + 1))) << "const access the last/actual sample";
        expect(eq(const_hb[1], static_cast<int>(capacity))) << "const access the previous sample";

        expect(eq(hb.at(0), static_cast<int>(capacity + 1))) << "checked access the last/actual sample";
        expect(eq(hb.at(1), static_cast<int>(capacity))) << "checked access the previous sample";
        expect(eq(const_hb.at(0), static_cast<int>(capacity + 1))) << "checked const access the last/actual sample";
        expect(eq(const_hb.at(1), static_cast<int>(capacity))) << "checked const access the previous sample";
    } | std::vector<std::size_t>{5, 3, 10};

    "HistoryBuffer - range tests"_test = [] {
        HistoryBuffer<int> hb(5);
        hb.push_front(std::array{1, 2, 3});
        hb.push_front(std::vector{4, 5, 6});
        expect(eq(hb.capacity(), 5UZ));
        expect(eq(hb.size(), 5UZ));

        auto equal = [](const auto& range1, const auto& range2) { // N.B. TODO replacement until libc++ fully supports ranges
            return std::equal(range1.begin(), range1.end(), range2.begin(), range2.end());
        };

        expect(equal(hb.get_span(0, 3), std::vector{6, 5, 4})) << std::format("failed - got [{}]", gr::join(hb.get_span(0, 3), ", "));
        expect(equal(hb.get_span(1, 3), std::vector{5, 4, 3})) << std::format("failed - got [{}]", gr::join(hb.get_span(1, 3), ", "));

        expect(equal(hb.get_span(0), std::vector{6, 5, 4, 3, 2})) << std::format("failed - got [{}]", gr::join(hb.get_span(0), ", "));
        expect(equal(hb.get_span(1), std::vector{5, 4, 3, 2})) << std::format("failed - got [{}]", gr::join(hb.get_span(1), ", "));

        std::vector<int> forward_bracket;
        for (std::size_t i = 0; i < hb.size(); i++) {
            forward_bracket.push_back(hb[i]);
        }
        expect(equal(forward_bracket, std::vector{6, 5, 4, 3, 2})) << std::format("failed - got [{}]", gr::join(forward_bracket, ", "));

        std::vector<int> forward(hb.begin(), hb.end());
        expect(equal(forward, std::vector{6, 5, 4, 3, 2})) << std::format("failed - got [{}]", gr::join(forward, ", "));

        std::vector<int> reverse(hb.rbegin(), hb.rend());
        expect(equal(reverse, std::vector{2, 3, 4, 5, 6})) << std::format("failed - got [{}]", gr::join(reverse, ", "));

        expect(equal(std::vector(hb.cbegin(), hb.cend()), std::vector(hb.begin(), hb.end()))) << "const non-const iterator equivalency";
        expect(equal(std::vector(hb.crbegin(), hb.crend()), std::vector(hb.rbegin(), hb.rend()))) << "const non-const iterator equivalency";
    };

    "HistoryBuffer<T> constexpr sized"_test = [] {
        HistoryBuffer<int, 5UZ> buffer5;
        HistoryBuffer<int, 8UZ> buffer8;

        for (std::size_t i = 0UZ; i <= buffer8.capacity(); ++i) {
            buffer5.push_front(static_cast<int>(i));
            buffer8.push_front(static_cast<int>(i));
        }

        expect(eq(buffer5[0], 8));
        expect(eq(buffer8[0], 8));
    };

    "HistoryBuffer<T> edge cases"_test = [] {
        std::print("\n\ntesting edge cases:\n");
        expect(throws<std::out_of_range>([] { HistoryBuffer<int>(0); })) << "throws for 0 capacity";

        // Create a history buffer of size 1
        HistoryBuffer<int> hb_one(1);
        const auto&        const_hb_one = hb_one; // tests const access
        expect(eq(hb_one.capacity(), 1UZ));
        expect(eq(hb_one.size(), 0UZ));
        hb_one.push_front(41);
        hb_one.push_front(42);
        expect(eq(hb_one.capacity(), 1UZ));
        expect(eq(hb_one.size(), 1UZ));
        expect(eq(hb_one[0], 42));

        expect(throws<std::out_of_range>([&hb_one] { [[maybe_unused]] auto a = hb_one.at(2); })) << "throws for index > size";
        expect(throws<std::out_of_range>([&const_hb_one] { [[maybe_unused]] auto a = const_hb_one.at(2); })) << "throws for index > size";

        // Push more elements than buffer size
        HistoryBuffer<int> hb_overflow(5);
        auto               in = std::vector{1, 2, 3, 4, 5, 6};
        hb_overflow.push_front(in.begin(), in.end());
        expect(eq(hb_overflow[0], 6));
        hb_overflow.push_front(std::vector{7, 8, 9, 10, 11, 12, 13, 14});
        expect(eq(hb_overflow[0], 14));
        hb_overflow.push_front(std::array{15, 16, 17});
        expect(eq(hb_overflow[0], 17));

        // Test with different types, e.g., double
        HistoryBuffer<double> hb_double(5);
        for (int i = 0; i < 10; ++i) {
            hb_double.push_front(i * 0.1);
        }
        expect(eq(hb_double.capacity(), 5UZ));
        expect(eq(hb_double.size(), 5UZ));

        expect(nothrow([&hb_double] { hb_double.reset(); })) << "reset (default) does not throw";
        expect(eq(hb_double.size(), 0UZ));
        expect(std::all_of(hb_double.begin(), hb_double.end(), [](const auto& elem) { return elem == 0.0; }));
        expect(nothrow([&hb_double] { hb_double.reset(2.0); })) << "reset (2.0) does not throw";
        const auto& const_hb_double = hb_double; // tests const access
        expect(std::all_of(const_hb_double.begin(), const_hb_double.end(), [](const auto& elem) { return elem == 2.0; }));

        for (std::size_t i = 0UZ; i < hb_double.capacity(); ++i) {
            expect(eq(2.0, hb_double.data()[i]));
            expect(eq(2.0, const_hb_double.data()[i]));
        }

        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(hb_double.data())>>, "is non-const");
        const auto& const_buffer = hb_double;
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(const_buffer.data())>>, "is const");
    };

    "HistoryBuffer - forward/reversed usage"_test = [] {
        HistoryBuffer<int> forward(5);  // stores push_front(..) with forward[0] being newest sample
        HistoryBuffer<int> backward(5); // stores push_back(..) with backward[0] being the oldest sample

        // push {1,2,3,4,5,6} individually to both
        for (int i = 1; i <= 6; ++i) {
            forward.push_front(i);
            backward.push_back(i);
        }

        // expected content of forward:  [6,5,4,3,2]
        expect(eq(forward.size(), 5UZ));
        expect(eq(forward[0], 6));
        expect(eq(forward[1], 5));
        expect(eq(forward[4], 2));

        // expected content of backward: [2,3,4,5,6]
        expect(eq(backward.size(), 5UZ));
        expect(eq(backward[0], 2));
        expect(eq(backward[1], 3));
        expect(eq(backward[2], 4));
        expect(eq(backward[3], 5));
        expect(eq(backward[4], 6));

        // Bulk test:
        backward.reset();
        backward.push_back(std::vector<int>{10, 11, 12, 13, 14, 15, 16}); // push more than capacity:
        expect(eq(backward[0], 12));
        expect(eq(backward[4], 16));

        expect(throws<std::out_of_range>([&] { (void)backward.at(5); }));
    };

    "HistoryBuffer - resize test"_test = [] {
        using namespace boost::ut;
        using namespace gr;

        // Only for dynamic-extent
        HistoryBuffer<int> hb(5);
        for (int i = 1; i <= 5; ++i) {
            hb.push_front(i);
        }
        // now: [5,4,3,2,1]
        expect(eq(hb.size(), 5UZ));

        // resize to bigger capacity
        hb.resize(8);
        expect(eq(hb.capacity(), 8UZ));
        expect(eq(hb.size(), 5UZ));
        // index[0] should still be '5'
        expect(eq(hb[0], 5));

        // push more data
        for (int i = 6; i <= 10; ++i) {
            hb.push_front(i); // if we keep pushing
        }
        expect(eq(hb.size(), 8UZ)); // now full at 8
        expect(eq(hb[0], 10));      // newest => 10

        // shrink
        hb.resize(3);
        expect(eq(hb.capacity(), 3UZ));
        expect(eq(hb.size(), 3UZ));
        // we keep the "most recent" 3 => new index[0] should be 10 still, if it fits
        expect(eq(hb[0], 10));
    };

    "HistoryBuffer - front/back test"_test = [] {
        HistoryBuffer<int> hb(5);
        expect(eq(hb.empty(), true));

        for (int i = 1; i <= 6; ++i) {
            hb.push_front(i);
        }
        // final ring => [6,5,4,3,2]
        expect(eq(hb.front(), 6)) << "front == [0] => newest sample in push_back orientation";
        expect(eq(hb.back(), 2)) << "back == [size-1] => oldest sample";

        hb.reset();
        for (int i = 1; i <= 6; ++i) {
            hb.push_back(i);
        }
        // final ring => [2,3,4,5,6] in logical terms
        expect(eq(hb.front(), 2)) << "front == [0] => oldest sample in push_front orientation";
        expect(eq(hb.back(), 6)) << "back == [size-1] => newest sample";
    };

    "HistoryBuffer - pop_front/pop_back"_test = [] {
        using namespace boost::ut;
        using namespace gr;

        HistoryBuffer<int> hb(5);
        for (int i = 1; i <= 5; ++i) { // push_back => newest @ [0]
            hb.push_front(i);          // final ring => [5,4,3,2,1]
        }
        expect(eq(hb.size(), 5UZ));
        expect(eq(hb[0], 5));
        expect(eq(hb[4], 1));

        // pop_front => removes [0] => was '5'
        hb.pop_front();
        expect(eq(hb.size(), 4UZ));
        expect(eq(hb[0], 4)); // now ring => [4,3,2,1]

        // pop_back => removes [size()-1] => was '1'
        hb.pop_back();
        expect(eq(hb.size(), 3UZ));
        expect(eq(hb[0], 4)); // ring => [4,3,2]
        expect(eq(hb[2], 2));

        // test emptiness
        hb.pop_front(); // remove '4' => size=2
        hb.pop_front(); // remove '3' => size=1
        hb.pop_back();  // remove '2' => size=0
        expect(eq(hb.size(), 0UZ));
        expect(throws<std::out_of_range>([&] { hb.pop_back(); }));
        expect(throws<std::out_of_range>([&] { hb.pop_front(); }));

        // test push_front orientation
        HistoryBuffer<int> hb2(5);
        for (int i = 1; i <= 5; ++i) {
            hb2.push_back(i); // final ring => [1,2,3,4,5] logically
        }
        expect(eq(hb2[0], 1));
        hb2.pop_front(); // remove '1'
        expect(eq(hb2[0], 2));
        hb2.pop_back(); // remove '5'
        expect(eq(hb2[hb2.size() - 1], 4));
    };
};

int main() { /* not needed for UT */ }
