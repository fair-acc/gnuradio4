#ifndef GNURADIO_BYTERINGBUFFER_HPP
#define GNURADIO_BYTERINGBUFFER_HPP

#include <cstddef>
#include <memory_resource>
#include <utility>

#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/CircularBuffer.hpp>

namespace gr {

inline constexpr std::size_t kDefaultByteChunkSize = 256UZ;

/**
 * @brief Thin drop-in wrapper composing `CircularBuffer<T>` with a per-element byte-chunk-size policy.
 *
 * Reuses `CircularBuffer`'s tested `ClaimStrategy`/`Sequence`/`WaitStrategy`/`double_mapped_memory_resource` machinery and its
 * `Reader`/`WriterSpan`/`ReaderSpan`/reader-registration verbatim — no concurrency code is re-implemented. The only added behaviour:
 * `byteChunkSize` is carried by the buffer, and a bare `reserve(n)`/`tryReserve(n)` from its writer auto-forwards to
 * `CircularBuffer::Writer::reserve(n, byteChunkSize)` — so reserving is RT-safe by default (each slot pre-grown once; steady state
 * allocation-free) for `ReservableElement<T>`, and a transparent no-op otherwise (a behaviour-identical drop-in for non-reservable `T`).
 * `reserve(n, hint)` overrides per call. A distinct type (not a `using` alias) so the future non-owning co-located-blob zero-copy
 * path can be added behind the `Tag` `owning` NTTP without touching `CircularBuffer`.
 *
 * Migration nuance (addressed when wiring `DefaultTagBuffer`): the reader/writer's `buffer()` returns the inner `CircularBuffer<T>`,
 * not `ByteRingBuffer<T>`.
 */
template<typename T, std::size_t SIZE = std::dynamic_extent, ProducerType producerType = ProducerType::Single, WaitStrategyLike TWaitStrategy = SleepingWaitStrategy>
class ByteRingBuffer {
    using Ring        = CircularBuffer<T, SIZE, producerType, TWaitStrategy>;
    using InnerWriter = decltype(std::declval<Ring&>().new_writer());

    std::pmr::memory_resource* _resource{nullptr}; // explicit PMR (nullptr ⇒ CircularBuffer's own default); stored + passed to _ring
    Ring                       _ring;
    std::size_t                _byteChunkSize{kDefaultByteChunkSize};

    // resource == nullptr ⇒ CircularBuffer's own default (double-mapped for trivially-copyable T)
    [[nodiscard]] static Ring makeRing(std::size_t minSize, std::pmr::memory_resource* resource) { return resource != nullptr ? Ring(minSize, std::pmr::polymorphic_allocator<T>(resource)) : Ring(minSize); }

public:
    using value_type = T;

    ByteRingBuffer() = delete;
    explicit ByteRingBuffer(std::size_t minSize, std::size_t byteChunkSize = kDefaultByteChunkSize, std::pmr::memory_resource* resource = nullptr) : _resource(resource), _ring(makeRing(minSize, resource)), _byteChunkSize(byteChunkSize) {}

    // drop-in parity with CircularBuffer's (minSize, polymorphic_allocator) ctor so the generic TagBufferType construction site keeps working when a block
    // overrides the tag buffer back to CircularBuffer<Tag>; unambiguous against the ctor above as polymorphic_allocator<T> and std::size_t are mutually non-convertible
    ByteRingBuffer(std::size_t minSize, std::pmr::polymorphic_allocator<T> allocator, std::size_t byteChunkSize = kDefaultByteChunkSize) : _resource(allocator.resource()), _ring(makeRing(minSize, allocator.resource())), _byteChunkSize(byteChunkSize) {}

    // adopt an existing (shared) inner ring rather than allocating — used by the buffer() round-trip in Port::buffer()/setBuffer
    // so a ByteRingBuffer reconnects to the same underlying CircularBuffer; unambiguous vs the size_t-first ctors (identity beats
    // the size_t→Ring user conversion). byteChunkSize defaults here until per-edge tuning is plumbed (see Edge _minTagByteSize, deferred)
    explicit ByteRingBuffer(Ring ring, std::size_t byteChunkSize = kDefaultByteChunkSize, std::pmr::memory_resource* resource = nullptr) : _resource(resource), _ring(std::move(ring)), _byteChunkSize(byteChunkSize) {}

    ByteRingBuffer(const ByteRingBuffer&)                = default;
    ByteRingBuffer(ByteRingBuffer&&) noexcept            = default;
    ByteRingBuffer& operator=(const ByteRingBuffer&)     = default;
    ByteRingBuffer& operator=(ByteRingBuffer&&) noexcept = default;
    ~ByteRingBuffer()                                    = default;

    // The writer applies the buffer's byteChunkSize on a bare reserve(n)/tryReserve(n); everything else forwards to CircularBuffer's writer unchanged.
    class Writer {
        InnerWriter _w;
        std::size_t _byteChunkSize;

    public:
        Writer(InnerWriter writer, std::size_t byteChunkSize) noexcept : _w(std::move(writer)), _byteChunkSize(byteChunkSize) {}
        Writer(const Writer&)            = delete;
        Writer& operator=(const Writer&) = delete;
        Writer(Writer&&)                 = default;
        Writer& operator=(Writer&&)      = default;
        ~Writer()                        = default;

        template<SpanReleasePolicy policy = SpanReleasePolicy::ProcessNone>
        [[nodiscard]] constexpr auto reserve(std::size_t nSamples) {
            return _w.template reserve<policy>(nSamples, _byteChunkSize);
        }
        template<SpanReleasePolicy policy = SpanReleasePolicy::ProcessNone>
        [[nodiscard]] constexpr auto reserve(std::size_t nSamples, std::size_t elementReserveHint) {
            return _w.template reserve<policy>(nSamples, elementReserveHint);
        }
        template<SpanReleasePolicy policy = SpanReleasePolicy::ProcessNone>
        [[nodiscard]] constexpr auto tryReserve(std::size_t nSamples) {
            return _w.template tryReserve<policy>(nSamples, _byteChunkSize);
        }
        template<SpanReleasePolicy policy = SpanReleasePolicy::ProcessNone>
        [[nodiscard]] constexpr auto tryReserve(std::size_t nSamples, std::size_t elementReserveHint) {
            return _w.template tryReserve<policy>(nSamples, elementReserveHint);
        }

        [[nodiscard]] constexpr std::size_t                position() const noexcept { return _w.position(); }
        [[nodiscard]] constexpr std::size_t                available() const noexcept { return _w.available(); }
        [[nodiscard]] constexpr std::size_t                nRequestedSamplesToPublish() const noexcept { return _w.nRequestedSamplesToPublish(); }
        [[nodiscard]] constexpr std::pmr::memory_resource* resource() const noexcept { return _w.resource(); }
        [[nodiscard]] constexpr auto                       buffer() const noexcept { return _w.buffer(); } // inner CircularBuffer (documented nuance)
    };

    [[nodiscard]] std::size_t                size() const noexcept { return _ring.size(); }
    [[nodiscard]] std::size_t                byteChunkSize() const noexcept { return _byteChunkSize; }
    [[nodiscard]] std::pmr::memory_resource* resource() const noexcept { return _resource; }
    [[nodiscard]] BufferReaderLike auto      new_reader() { return _ring.new_reader(); }
    [[nodiscard]] Writer                     new_writer() { return Writer(_ring.new_writer(), _byteChunkSize); }

    // implementation-specific passthroughs (mirrors CircularBuffer's surface)
    [[nodiscard]] std::size_t n_writers() const { return _ring.n_writers(); }
    [[nodiscard]] std::size_t n_readers() const { return _ring.n_readers(); }
    constexpr void            houseKeeping(HouseKeepDepth depth) noexcept { _ring.houseKeeping(depth); }
};

static_assert(BufferLike<ByteRingBuffer<std::int32_t>>);

} // namespace gr

#endif // GNURADIO_BYTERINGBUFFER_HPP
