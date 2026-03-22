#ifndef GNURADIO_SCOPED_FD_HPP
#define GNURADIO_SCOPED_FD_HPP

#if !defined(__EMSCRIPTEN__) && !defined(_WIN32)

#include <unistd.h>
#include <utility>

namespace gr::blocks::common {

struct ScopedFd {
    int fd     = -1;
    ScopedFd() = default;
    explicit ScopedFd(int f) : fd(f) {}
    ~ScopedFd() {
        if (fd >= 0) {
            ::close(fd);
        }
    }
    ScopedFd(const ScopedFd&)            = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;
    ScopedFd(ScopedFd&& o) noexcept : fd(std::exchange(o.fd, -1)) {}
    ScopedFd& operator=(ScopedFd&& o) noexcept {
        std::swap(fd, o.fd);
        return *this;
    }
    [[nodiscard]] int release() noexcept { return std::exchange(fd, -1); }
};

} // namespace gr::blocks::common

#endif // POSIX

#endif // GNURADIO_SCOPED_FD_HPP
