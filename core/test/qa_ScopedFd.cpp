#include <boost/ut.hpp>

#include <utility>

#if !defined(__EMSCRIPTEN__) && !defined(_WIN32)
#include <gnuradio-4.0/common/ScopedFd.hpp>

#include <fcntl.h>
#include <unistd.h>
#endif

const boost::ut::suite<"ScopedFd"> scopedFdTests = [] {
    using namespace boost::ut;

#if defined(__EMSCRIPTEN__) || defined(_WIN32)
    "ScopedFd is POSIX-only"_test = [] { expect(true); };
#else
    "default constructed fd is invalid"_test = [] {
        gr::blocks::common::ScopedFd scoped;
        expect(eq(scoped.fd, -1));
        expect(eq(scoped.release(), -1));
    };

    "destructor closes owned fd"_test = [] {
        int fd = ::open("/dev/null", O_RDONLY);
        expect(fd >= 0);
        {
            gr::blocks::common::ScopedFd scoped(fd);
            expect(eq(scoped.fd, fd));
        }
        expect(eq(::close(fd), -1)); // already closed
    };

    "release transfers ownership"_test = [] {
        int fd = ::open("/dev/null", O_RDONLY);
        expect(fd >= 0);
        int released = -1;
        {
            gr::blocks::common::ScopedFd scoped(fd);
            released = scoped.release();
            expect(eq(released, fd));
            expect(eq(scoped.fd, -1));
        }
        expect(eq(::close(released), 0));
    };

    "move constructor and assignment"_test = [] {
        int fd = ::open("/dev/null", O_RDONLY);
        expect(fd >= 0);

        gr::blocks::common::ScopedFd first(fd);
        gr::blocks::common::ScopedFd second(std::move(first));
        expect(eq(first.fd, -1));
        expect(eq(second.fd, fd));

        int otherFd = ::open("/dev/null", O_RDONLY);
        expect(otherFd >= 0);
        gr::blocks::common::ScopedFd third(otherFd);
        third = std::move(second);
        // move-assignment swaps, so the donor keeps the previous destination fd
        expect(eq(second.fd, otherFd));
        expect(eq(third.fd, fd));
    };
#endif
};

int main() { return 0; }
