#include <boost/ut.hpp>

#include <gnuradio-4.0/reader_writer_lock.hpp>

namespace gr::reader_writer_lock_test {

const boost::ut::suite basicTests = [] {
    using namespace boost::ut;
    using namespace gr;
    using gr::ReaderWriterLockType::READ;
    using gr::ReaderWriterLockType::WRITE;

    gr::ReaderWriterLock rwlock;

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
} // namespace gr::reader_writer_lock_test

int main() { /* tests are statically executed */ }
