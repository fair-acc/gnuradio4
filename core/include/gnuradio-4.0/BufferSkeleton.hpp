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
    struct buffer_impl {
        const std::size_t _size;
        std::vector<T>    _data;

        buffer_impl() = delete;
        explicit buffer_impl(const std::size_t min_size) : _size(min_size), _data(_size){};
        ~buffer_impl() = default;
    };

    template<typename U>
    class buffer_reader {
        std::shared_ptr<buffer_impl> _buffer;

        buffer_reader() = delete;

        explicit buffer_reader(std::shared_ptr<buffer_impl> buffer) : _buffer(buffer) {}

        friend BufferSkeleton<T>;

    public:
        [[nodiscard]] BufferSkeleton
        buffer() const noexcept {
            return BufferSkeleton(_buffer);
        };

        template<bool strict_check = true>
        [[nodiscard]] std::span<const U>
        get(const std::size_t /* n_requested = 0*/) const noexcept(!strict_check) {
            return {};
        }

        template<bool strict_check = true>
        [[nodiscard]] bool
        consume(const std::size_t /* n_items = 1 */) const noexcept(!strict_check) {
            return true;
        }

        [[nodiscard]] constexpr std::make_signed_t<std::size_t>
        position() const noexcept {
            return -1;
        }

        [[nodiscard]] constexpr std::size_t
        available() const noexcept {
            return 0;
        }
    };

    template<typename U>
    class buffer_writer {
        std::shared_ptr<buffer_impl> _buffer;

        buffer_writer() = delete;

        explicit buffer_writer(std::shared_ptr<buffer_impl> buffer) : _buffer(buffer) {}

        friend BufferSkeleton<T>;

    public:
        [[nodiscard]] BufferSkeleton
        buffer() const noexcept {
            return BufferSkeleton(_buffer);
        };

        [[nodiscard]] constexpr auto
        reserve_output_range(std::size_t n) noexcept -> std::span<U> {
            return { &_buffer->_data[0], n };
        }

        constexpr void
        publish(std::pair<std::size_t, std::make_signed<std::size_t>>, std::size_t) const { /* empty */
        }

        template<typename... Args, WriterCallback<U, Args...> Translator>
        void
        publish(Translator && /* translator */, std::size_t /* n_slots_to_claim = 1 */, Args &&.../* args */) const noexcept { /* empty */
        }                                                                                                                      // blocks until elements are available

        template<typename... Args, WriterCallback<U, Args...> Translator>
        [[nodiscard]] bool
        try_publish(Translator && /* translator */, std::size_t n_slots_to_claim = 1, Args &&.../* args */) const noexcept {
            return n_slots_to_claim == 0;
        } // returns false if cannot emplace data -> user may need to retry

        [[nodiscard]] constexpr std::size_t
        available() const noexcept {
            return _buffer->_data.size();
        } // #items that can be written -- dynamic since readers can release in parallel
    };

    // shared pointer is needed to avoid dangling references to reader/writer
    // or generating buffer itself
    std::shared_ptr<buffer_impl> _shared_buffer_ptr;

    explicit BufferSkeleton(std::shared_ptr<buffer_impl> shared_buffer_ptr) : _shared_buffer_ptr(shared_buffer_ptr) {}

public:
    BufferSkeleton() = delete;

    explicit BufferSkeleton(const std::size_t min_size) : _shared_buffer_ptr(std::make_shared<buffer_impl>(min_size)) {}

    ~BufferSkeleton() = default;

    [[nodiscard]] std::size_t
    size() const {
        return _shared_buffer_ptr->_data.size();
    }

    template<typename WriteDataType = T>
    BufferReader auto
    new_reader() {
        return buffer_reader<WriteDataType>(_shared_buffer_ptr);
    }

    template<typename ReadDataType = T>
    BufferWriter auto
    new_writer() {
        return buffer_writer<ReadDataType>(_shared_buffer_ptr);
    }
};

static_assert(Buffer<BufferSkeleton<int32_t>>);

} // namespace gr::test

#endif // GNURADIO_BUFFERSKELETON_HPP
