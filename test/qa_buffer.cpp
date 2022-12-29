//import boost.ut;
#include <boost/ut.hpp>

#include <algorithm>
#include <complex>
#include <numeric>
#include <ranges>
#include <tuple>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <buffer.hpp>
#include <buffer_skeleton.hpp>
#include <circular_buffer.hpp>
#include <sequence.hpp>
#include <wait_strategy.hpp>

template<gr::WaitStrategy auto wait = gr::NoWaitStrategy()>
struct TestStruct {
    [[nodiscard]] constexpr bool test() const noexcept { return true; }
};

const boost::ut::suite BasicConceptsTests = [] {
    using namespace boost::ut;

    "BasicConcepts"_test = []<typename T> {
        using namespace gr;
        Buffer auto buffer = T(1024);
        auto typeName = reflection::type_name<T>();
        // N.B. GE because some buffers need to intrinsically
        // allocate more to meet e.g. page-size requirements
        expect(eq(buffer.size(), 1024)) << "for" << typeName;

        // compile-time interface tests
        BufferReader auto reader = buffer.new_reader(); // tests matching read concept
        BufferWriter auto writer = buffer.new_writer(); // tests matching write concept

        static_assert(std::is_same_v<decltype(reader.buffer().new_reader()), decltype(reader)>);
        static_assert(std::is_same_v<decltype(reader.buffer().new_writer()), decltype(writer)>);
        static_assert(std::is_same_v<decltype(writer.buffer().new_writer()), decltype(writer)>);
        static_assert(std::is_same_v<decltype(writer.buffer().new_reader()), decltype(reader)>);

        // runtime interface tests
        expect(eq(reader.available(), 0));
        expect(eq(reader.position(), -1));
        expect(nothrow([&reader] { expect(eq(reader.get(0).size(), 0)); })) << typeName << "throws";
        expect(nothrow([&reader] { expect(reader.consume(0)); }));

        expect(writer.available() >= buffer.size());
        expect(nothrow([&writer] { writer.publish([](const std::span<int32_t> &) { /* noop */ }, 0); }));
        expect(nothrow([&writer] { writer.publish([](const std::span<int32_t> &, std::int64_t) { /* noop */ }, 0); }));
        expect(nothrow([&writer] { expect(writer.try_publish([](const std::span<int32_t> &) { /* noop */ }, 0)); }));
        expect(nothrow([&writer] {
            expect(writer.try_publish([](const std::span<int32_t> &, std::int64_t) { /* noop */ }, 0));
        }));
    } | std::tuple<gr::test::buffer_skeleton<int32_t>,
            gr::circular_buffer<int32_t, std::dynamic_extent, gr::ProducerType::Single>,
            gr::circular_buffer<int32_t, std::dynamic_extent, gr::ProducerType::Multi>>{2, 2, 2};
};

const boost::ut::suite SequenceTests = [] {
    using namespace boost::ut;

    "Sequence"_test = [] {
        using namespace gr;
        expect(eq(alignof(Sequence), 64));
        expect(eq(-1L, kInitialCursorValue));
        expect(nothrow([] { Sequence(); }));
        expect(nothrow([] { Sequence(2); }));

        auto s1 = Sequence();
        expect(eq(s1.value(), kInitialCursorValue));

        const auto s2 = Sequence(2);
        expect(eq(s2.value(), 2));

        expect(nothrow([&s1] { s1.setValue(3); }));
        expect(eq(s1.value(), 3));

        expect(nothrow([&s1] { expect(s1.compareAndSet(3, 4)); }));
        expect(nothrow([&s1] { expect(eq(s1.value(), 4)); }));
        expect(nothrow([&s1] { expect(!s1.compareAndSet(3, 5)); }));
        expect(eq(s1.value(), 4));

        expect(eq(s1.incrementAndGet(), 5));
        expect(eq(s1.value(), 5));
        expect(eq(s1.addAndGet(2), 7));
        expect(eq(s1.value(), 7));

        std::shared_ptr<std::vector<std::shared_ptr<Sequence>>> sequences{
                std::make_shared<std::vector<std::shared_ptr<Sequence>>>()
        };
        expect(eq(gr::detail::getMinimumSequence(*sequences), std::numeric_limits<std::int64_t>::max()));
        expect(eq(gr::detail::getMinimumSequence(*sequences, 2), 2));
        sequences->emplace_back(std::make_shared<Sequence>(4));
        expect(eq(gr::detail::getMinimumSequence(*sequences), 4));
        expect(eq(gr::detail::getMinimumSequence(*sequences, 5), 4));
        expect(eq(gr::detail::getMinimumSequence(*sequences, 2), 2));

        auto cursor = std::make_shared<Sequence>(10);
        auto s3 = std::make_shared<Sequence>(1);
        expect(eq(sequences->size(), 1));
        expect(eq(gr::detail::getMinimumSequence(*sequences), 4));
        expect(nothrow([&sequences, &cursor, &s3] { gr::detail::addSequences(sequences, *cursor, {s3}); }));
        expect(eq(sequences->size(), 2));
        // newly added sequences are set automatically to the cursor/write position
        expect(eq(s3->value(), 10));
        expect(eq(gr::detail::getMinimumSequence(*sequences), 4));

        expect(nothrow([&sequences, &cursor] { gr::detail::removeSequence(sequences, cursor); }));
        expect(eq(sequences->size(), 2));
        expect(nothrow([&sequences, &s3] { gr::detail::removeSequence(sequences, s3); }));
        expect(eq(sequences->size(), 1));

        std::stringstream ss;
        expect(eq(ss.str().size(), 0));
        expect(nothrow([&ss, &s3] { ss << fmt::format("{}", *s3); }));
        expect(not ss.str().empty());
    };
};

#ifndef __EMSCRIPTEN__
const boost::ut::suite DoubleMappedAllocatorTests = [] {
    using namespace boost::ut;

    "DoubleMappedAllocator"_test = [] {
        using Allocator = std::pmr::polymorphic_allocator<int32_t>;
        std::size_t size = getpagesize() / sizeof(int32_t);
        auto doubleMappedAllocator = gr::double_mapped_memory_resource::allocator<int32_t>();
        std::vector<int32_t, Allocator> vec(size, doubleMappedAllocator);
        expect(eq(vec.size(), size));
        std::iota(vec.begin(), vec.end(), 1);
        for (auto i = 0U; i < vec.size(); i++) {
            expect(eq(vec[i], i + 1));
            // to note: can safely read beyond size for this special vector
            expect(eq(vec[size + i], vec[i])); // identical to mirrored copy
        }
    };
};
#endif

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

        expect(WaitStrategy<BlockingWaitStrategy>);
        expect(WaitStrategy<BusySpinWaitStrategy>);
        expect(WaitStrategy<SleepingWaitStrategy>);
        expect(WaitStrategy<SleepingWaitStrategy>);
        expect(WaitStrategy<SpinWaitWaitStrategy>);
        expect(WaitStrategy<TimeoutBlockingWaitStrategy>);
        expect(WaitStrategy<YieldingWaitStrategy>);
        expect(not WaitStrategy<int>);

        TestStruct a;
        expect(a.test());
    };
};

const boost::ut::suite UserApiExamples = [] {
    using namespace boost::ut;

    "UserApi"_test = [] {
        using namespace gr;
        Buffer auto buffer = circular_buffer<int32_t>(1024);

        BufferWriter auto writer = buffer.new_writer();
        { // source only write example
            BufferReader auto localReader = buffer.new_reader();
            expect(eq(localReader.available(), 0));

            auto lambda = [](auto w) { // test writer generating consecutive samples
                static int offset = 1;
                std::iota(w.begin(), w.end(), offset);
                offset += w.size();
            };

            expect(ge(writer.available(), buffer.size()));
            writer.publish(lambda, 10);
            expect(eq(writer.available(), buffer.size() - 10));
            expect(eq(localReader.available(), 10));
            expect(eq(buffer.n_readers(), 1)); // N.B. circular_buffer<..> specific
        }
        expect(eq(buffer.n_readers(), 0)); // reader not in scope release atomic reader index

        BufferReader auto reader = buffer.new_reader();
        // reader does not know about previous submitted data as it joined only after
        // data has been written <-> needed for thread-safe joining of readers while writing
        expect(eq(reader.available(), 0));
        // populate with some more data
        for (int i = 0; i < 3; i++) {
            const auto demoWriter = [](auto w) {
                static int offset = 1;
                std::iota(w.begin(), w.end(), offset);
                offset += w.size();
            };
            writer.publish(demoWriter, 5); // writer/publish five samples
            expect(eq(reader.available(), (i + 1) * 5)) << fmt::format("iteration: {}", i);
        }

        // N.B. here using a simple read-only (sink) example:
        for (int i = 0; reader.available() != 0; i++) {
            std::span<const int32_t> fixedLength = reader.get(3); // explicitly typed for illustration
            auto available = reader.get();
            fmt::print("iteration {} - fixed-size data[{:2}]: [{}]\n", i, fixedLength.size(),
                       fmt::join(fixedLength, ", "));
            fmt::print("iteration {} - full-size  data[{:2}]: [{}]\n", i, available.size(), fmt::join(available, ", "));

            // consume data -> allows corresponding buffer to be overwritten by writer
            // if there are no other reader claiming that buffer segment
            if (reader.consume(fixedLength.size())) {
                // for info-only - since available() can change in parallel
                // N.B. lock-free buffer and other writer may add while processing
                fmt::print("iteration {} - consumed {} elements - still available: {}\n", i, fixedLength.size(),
                           reader.available());
            } else {
                throw std::runtime_error(fmt::format("could not consume data"));
            }
        }
    };
};

const boost::ut::suite CircularBufferTests = [] {
    using namespace boost::ut;
    using Allocator = std::pmr::polymorphic_allocator<int32_t>;

    "CircularBuffer"_test = [](const Allocator &allocator) {
        using namespace gr;
        Buffer auto buffer = circular_buffer<int32_t>(1024, allocator);
        expect(ge(buffer.size(), 1024));

        BufferWriter auto writer = buffer.new_writer();
        expect(nothrow([&writer] { expect(eq(writer.buffer().n_readers(), 0)); })); // no reader, just writer
        BufferReader auto reader = buffer.new_reader();
        expect(nothrow([&reader] { expect(eq(reader.buffer().n_readers(), 1)); })); // created one reader

        int offset = 1;
        auto lambda = [&offset](auto w) {
            std::iota(w.begin(), w.end(), offset);
            offset += w.size();
        };

        expect(eq(reader.available(), 0));
        expect(eq(reader.get().size(), 0));
        expect(eq(reader.get(1).size(), 0));
        expect(eq(writer.available(), buffer.size()));
        expect(not reader.consume(1)); // false: no data available yet
        expect(nothrow([&writer, &lambda, &buffer] { writer.publish(lambda, buffer.size()); })); // fully fill buffer

        expect(eq(writer.available(), 0));
        expect(eq(reader.available(), buffer.size()));
        expect(eq(reader.get().size(), buffer.size()));
        expect(eq(reader.get(1).size(), 1));

        // full buffer: fill buffer need to fail/return 'false'
        expect(not writer.try_publish(lambda, buffer.size()));

        expect(reader.consume(buffer.size()));
        expect(eq(reader.available(), 0));
        expect(eq(writer.available(), buffer.size()));

        // test buffer wrap around twice
        int32_t counter = 1;
        for (const std::size_t blockSize: {1, 2, 3, 5, 7, 42}) {
            for (uint32_t i = 0; i < buffer.size(); i++) {
                expect(writer.try_publish([&counter](auto &writable) {
                    std::iota(writable.begin(), writable.end(), counter += writable.size());
                }, blockSize));
                auto readable = reader.get();
                expect(eq(readable.size(), blockSize));
                expect(eq(readable.front(), counter));
                expect(eq(readable.back(), counter + blockSize - 1));
                expect(reader.consume(blockSize));
            }
        }
    } | std::vector{
#ifndef __EMSCRIPTEN__
        Allocator(gr::double_mapped_memory_resource::allocator<int32_t>()),
#endif
        Allocator()};
};

const boost::ut::suite CircularBufferExceptionTests = [] {
    using namespace boost::ut;
    "CircularBufferExceptions"_test = [] {
        using namespace gr;
        Buffer auto buffer = circular_buffer<int32_t>(1024);
        expect(ge(buffer.size(), 1024));

        BufferWriter auto writer = buffer.new_writer();
        BufferReader auto reader = buffer.new_reader();

        expect(throws<std::exception>([&writer] { writer.publish([](auto &) { throw std::exception(); }); }));
        expect(throws<std::exception>([&writer] { writer.publish([](auto &) { throw ""; }); }));
        expect(throws<std::exception>([&writer] { writer.try_publish([](auto &) { throw std::exception(); }); }));
        expect(throws<std::runtime_error>([&writer] { writer.try_publish([](auto &) { throw ""; }); }));

        expect(eq(reader.available(), 0)); // needed otherwise buffer write will not be called
    };
};

const boost::ut::suite UserDefinedTypeCasting = [] {
    using namespace boost::ut;
    "UserDefinedTypeCasting"_test = [] {
        using namespace gr;
        Buffer auto buffer = circular_buffer<std::complex<float>>(1024);
        expect(ge(buffer.size(), 1024));

        BufferWriter auto writer = buffer.new_writer();
        BufferReader auto reader = buffer.new_reader();

        writer.publish([](auto &w) {
            w[0] = std::complex(1.0f, -1.0f);
            w[1] = std::complex(2.0f, -2.0f);
        }, 2);
        expect(eq(reader.available(), 2));
        std::span<const std::complex<float>> data = reader.get();
        expect(eq(data.size(), 2));

        auto const const_bytes = std::as_bytes(data);
        expect(eq(const_bytes.size(), data.size() * sizeof(std::complex<float>)));

        auto convertToFloatSpan = [](std::span<const std::complex<float>> &c) -> std::span<const float> {
            return {reinterpret_cast<const float *>(c.data()), c.size() * 2}; //NOSONAR(cpp:S3630) //NOPMD needed
        };
        auto floatArray = convertToFloatSpan(data);
        expect(eq(floatArray[0], +1.0f));
        expect(eq(floatArray[1], -1.0f));
        expect(eq(floatArray[2], +2.0f));
        expect(eq(floatArray[3], -2.0f));

        expect(reader.consume(data.size()));
        expect(eq(reader.available(), 0)); // needed otherwise buffer write will not be called
    };
};

const boost::ut::suite StreamTagConcept = [] {
    using namespace boost::ut;

    "StreamTagConcept"_test = [] {
        // implements a proof-of-concept how stream-tags could be dealt with
        using namespace gr;
        struct alignas(gr::kCacheLine) buffer_tag {
            // N.B. type need to be favourably sized e.g. 1 or a power of 2
            // -> otherwise the automatic buffer sizes are getting very large
            int64_t index;
            std::string data;
        };

        expect(eq(sizeof(buffer_tag), 64)) << "tag size";
        Buffer auto buffer = circular_buffer<int32_t>(1024);
        Buffer auto tagBuffer = circular_buffer<buffer_tag>(32);
        expect(ge(buffer.size(), 1024));
        expect(ge(tagBuffer.size(), 32));


        BufferWriter auto writer = buffer.new_writer();
        BufferReader auto reader = buffer.new_reader();
        BufferWriter auto tagWriter = tagBuffer.new_writer();
        BufferReader auto tagReader = tagBuffer.new_reader();

        for (int i = 0; i < 3; i++) { // write-only worker (source) mock-up
            auto lambda = [&tagWriter](auto w, std::int64_t writePosition) {
                static int offset = 1;
                std::iota(w.begin(), w.end(), offset);
                offset += w.size();

                // read/generated by some method (e.g. reading another buffer)
                tagWriter.publish([&writePosition](auto &tag) {
                    tag[0] = {writePosition, fmt::format("<tag data at index {:3}>", writePosition)};
                }, 1);
            };

            writer.publish(lambda, 10); // optional return param.
        }

        { // read-only worker (sink) mock-up
            fmt::print("read position: {}\n", reader.position());
            const auto readData = reader.get();
            const auto tags = tagReader.get();

            fmt::print("received {} tags\n", tags.size());
            for (auto &tag: tags) {
                fmt::print("stream-tag @{:3}: '{}'\n", tag.index, tag.data);
            }

            expect(reader.consume(readData.size()));
            expect(tagReader.consume(tags.size())); // N.B. consume tag based on expiry
        }
    };
};

int main() {}
