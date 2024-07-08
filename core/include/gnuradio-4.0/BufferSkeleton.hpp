#ifndef GNURADIO_BUFFERSKELETON_HPP
#define GNURADIO_BUFFERSKELETON_HPP

#ifndef GNURADIO_BUFFER2_H
#include "buffer.hpp" // TODO: why is this include guard outside of the buffer hpp needed?
#endif

#include <cstdlib> // for malloc
#include <memory>  // for std::shared_ptr
#include <vector>  // as example internal storage container

namespace gr::test {

/**
 * @brief a minimal non-functional buffer implementation to test the buffer 'concept' API
 * and as a starting point for new buffer specialisation
 *
 * @tparam T the internally stored type parameter
 */
template<typename T>
class BufferSkeleton {
    struct BufferImpl {
        const std::size_t _size;
        std::vector<T>    _data;

        BufferImpl() = delete;
        explicit BufferImpl(const std::size_t min_size) : _size(min_size), _data(_size){};
        ~BufferImpl() = default;
    };

    template<typename U>
    class Reader {
        std::shared_ptr<BufferImpl> _buffer;

        Reader() = delete;

        explicit Reader(std::shared_ptr<BufferImpl> buffer) : _buffer(buffer) {}

        friend BufferSkeleton<T>;

    public:
        [[nodiscard]] BufferSkeleton buffer() const noexcept { return BufferSkeleton(_buffer); };

        [[nodiscard]] constexpr std::size_t nSamplesConsumed() const noexcept { return 0UZ; };

        [[nodiscard]] constexpr bool isConsumeRequested() const noexcept { return false; }

        template<bool strict_check = true>
        [[nodiscard]] std::span<const U> get(const std::size_t /* n_requested = 0*/) const noexcept(!strict_check) {
            return {};
        }

        template<bool strict_check = true>
        [[nodiscard]] bool consume(const std::size_t /* n_items = 1 */) const noexcept(!strict_check) {
            return true;
        }

        [[nodiscard]] constexpr std::make_signed_t<std::size_t> position() const noexcept { return -1; }

        [[nodiscard]] constexpr std::size_t available() const noexcept { return 0; }
    };

    template<typename U>
    class Writer {
        std::shared_ptr<BufferImpl> _buffer;

        Writer() = delete;

        explicit Writer(std::shared_ptr<BufferImpl> buffer) : _buffer(buffer) {}

        friend BufferSkeleton<T>;

    public:
        [[nodiscard]] BufferSkeleton buffer() const noexcept { return BufferSkeleton(_buffer); };

        [[nodiscard]] constexpr std::size_t nSamplesPublished() const noexcept { return 0UZ; };

        [[nodiscard]] constexpr auto reserve(std::size_t n) noexcept -> std::span<U> { return {&_buffer->_data[0], n}; }

        [[nodiscard]] constexpr auto tryReserve(std::size_t n) noexcept -> std::span<U> { return {&_buffer->_data[0], n}; }

        [[nodiscard]] constexpr std::size_t available() const noexcept { return _buffer->_data.size(); } // #items that can be written -- dynamic since readers can release in parallel
    };

    // shared pointer is needed to avoid dangling references to reader/writer or generating buffer itself
    std::shared_ptr<BufferImpl> _shared_buffer_ptr;

    explicit BufferSkeleton(std::shared_ptr<BufferImpl> shared_buffer_ptr) : _shared_buffer_ptr(shared_buffer_ptr) {}

public:
    BufferSkeleton() = delete;

    explicit BufferSkeleton(const std::size_t min_size) : _shared_buffer_ptr(std::make_shared<BufferImpl>(min_size)) {}

    ~BufferSkeleton() = default;

    [[nodiscard]] std::size_t size() const { return _shared_buffer_ptr->_data.size(); }

    template<typename WriteDataType = T>
    BufferReaderLike auto new_reader() {
        return Reader<WriteDataType>(_shared_buffer_ptr);
    }

    template<typename ReadDataType = T>
    BufferWriterLike auto new_writer() {
        return Writer<ReadDataType>(_shared_buffer_ptr);
    }
};

static_assert(BufferLike<BufferSkeleton<int32_t>>);

} // namespace gr::test

#endif // GNURADIO_BUFFERSKELETON_HPP
