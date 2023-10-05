#include <boost/ut.hpp>

#include <gnuradio-4.0/reader_writer_lock.hpp>

#if defined(__clang__) && __clang_major__ >= 16
// clang 16 does not like ut's default reporter_junit due to some issues with stream buffers and output redirection
template<>
auto boost::ut::cfg<boost::ut::override> = boost::ut::runner<boost::ut::reporter<>>{};
#endif

namespace fair::graph::reader_writer_lock_test {

const boost::ut::suite basicTests = [] {
    using namespace boost::ut;
    using namespace fair::graph;
    using fair::graph::ReaderWriterLockType::READ;
    using fair::graph::ReaderWriterLockType::WRITE;

    fair::graph::ReaderWriterLock rwlock;

    "basic read/write lock tests"_test = [&] {
        expect(eq(rwlock.lock<READ>(), 1));
        expect(eq(rwlock.lock<READ>(), 2));
        expect(eq(rwlock.unlock<READ>(), 1));
        expect(eq(rwlock.unlock<READ>(), 0));

        expect(eq(rwlock.lock<WRITE>(), -1));
        expect(eq(rwlock.lock<WRITE>(), -2));
        expect(eq(rwlock.unlock<WRITE>(), -1));
        expect(eq(rwlock.unlock<WRITE>(), -0));
    };

    "try write lock when holding read lock"_test = [&] {
        expect(eq(rwlock.lock<READ>(), 1));
        expect(!rwlock.tryLock<WRITE>());
        expect(eq(rwlock.unlock<READ>(), 0));
        expect(rwlock.tryLock<WRITE>());
        expect(eq(rwlock.unlock<WRITE>(), 0));
    };

    "try read lock when holding write lock"_test = [&] {
        expect(eq(rwlock.lock<WRITE>(), -1));
        expect(!rwlock.tryLock<READ>());
        expect(eq(rwlock.unlock<WRITE>(), -0));
        expect(rwlock.tryLock<READ>());
        expect(eq(rwlock.unlock<READ>(), -0));
    };

    "try RAII scoped read lock guard"_test = [&] {
        auto guard = rwlock.scopedGuard<READ>();
        expect(eq(rwlock.value(), 1));
        expect(!rwlock.tryLock<WRITE>());
    };

    "try RAII scoped write lock guard"_test = [&] {
        auto guard = rwlock.scopedGuard<WRITE>();
        expect(eq(rwlock.value(), -1));
        expect(!rwlock.tryLock<READ>());
    };
};
}

int
main() { /* tests are statically executed */
}
