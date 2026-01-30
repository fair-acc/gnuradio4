#include <atomic>
#include <boost/ut.hpp>
#include <numeric>
#include <thread>
#include <vector>

#include <gnuradio-4.0/ComputeDomain.hpp>

using namespace boost::ut;

namespace {

struct CountingMR : std::pmr::memory_resource {
    std::atomic<std::size_t> allocs{0}, frees{0}, bytes{0};
    void*                    do_allocate(std::size_t n, std::size_t) override {
        if (n == 0) {
            n = 1;
        }
        allocs.fetch_add(1, std::memory_order_relaxed);
        bytes.fetch_add(n, std::memory_order_relaxed);
        return ::operator new(n);
    }
    void do_deallocate(void* p, std::size_t, std::size_t) override {
        frees.fetch_add(1, std::memory_order_relaxed);
        ::operator delete(p);
    }
    bool do_is_equal(const std::pmr::memory_resource& o) const noexcept override { return this == &o; }
};

// toy provider: ctx is CountingMR*
std::pmr::memory_resource* toy_provider(const gr::ComputeDomain& d, void* ctx) {
    (void)d; // inspect d if you want to branch on kind/access/tag
    return static_cast<std::pmr::memory_resource*>(ctx);
}
std::pmr::memory_resource* null_provider(const gr::ComputeDomain&, void*) { return nullptr; }

} // namespace

const suite<"ComputeDomain"> _0 = [] {
    "host default binds new_delete_resource"_test = [] {
        auto                  bd    = gr::bind(); // host
        auto                  alloc = bd.allocator<int>();
        std::pmr::vector<int> v(0UZ, alloc);
        v.resize(128UZ);
        expect(eq(v.size(), 128UZ));
    };

    "register + resolve custom backend"_test = [] {
        CountingMR mr;
        gr::ComputeRegistry::instance().register_provider("toy", &toy_provider);

        gr::ComputeDomain dom;
        dom.kind        = "gpu";
        dom.access      = gr::Access::Shared;
        dom.backend     = "toy";
        dom.deviceIndex = 0;

        auto                        bd    = gr::bind(dom, &mr);
        auto                        alloc = bd.allocator<std::byte>();
        std::pmr::vector<std::byte> buf(4096UZ, alloc);

        expect(buf.size() == 4096UZ);
        expect(mr.allocs.load() >= 1UZ);
        expect(mr.bytes.load() >= 4096UZ);
    };

    "heterogenous lookup (string_view key)"_test = [] {
        CountingMR mr;
        // register using std::string, lookup via string_view
        gr::ComputeRegistry::instance().register_provider(std::string{"toy2"}, &toy_provider);

        gr::ComputeDomain dom;
        dom.kind    = "gpu";
        dom.backend = std::string_view{"toy2"};
        dom.access  = gr::Access::DeviceOnly;

        auto                  bd    = gr::bind(dom, &mr);
        auto                  alloc = bd.allocator<int>();
        std::pmr::vector<int> v(256, alloc);
        expect(eq(v.size(), 256UZ));
        expect(mr.allocs.load() >= 1UZ);
    };

    "re-register overrides provider"_test = [] {
        gr::ComputeRegistry::instance().register_provider("toy3", &null_provider);
        // override
        gr::ComputeRegistry::instance().register_provider("toy3", &toy_provider);

        CountingMR        mr;
        gr::ComputeDomain dom;
        dom.kind                  = "gpu";
        dom.backend               = "toy3";
        auto                   bd = gr::bind(dom, &mr); // would throw if null_provider still active
        auto                   a  = bd.allocator<char>();
        std::pmr::vector<char> x(8, a);
        expect(eq(x.size(), 8UZ));
    };

    "unknown backend throws"_test = [] {
        gr::ComputeDomain dom;
        dom.kind    = "gpu";
        dom.backend = "does-not-exist";
        expect(throws<std::runtime_error>([&] { (void)gr::bind(dom); }));
    };

    "provider returned null throws"_test = [] {
        gr::ComputeRegistry::instance().register_provider("null", &null_provider);
        gr::ComputeDomain dom;
        dom.kind    = "gpu";
        dom.backend = "null";
        expect(throws<std::runtime_error>([&] { (void)gr::bind(dom); }));
    };

    "mini example: gpu-shared (toy) alloc"_test = [] {
        CountingMR mr;
        gr::ComputeRegistry::instance().register_provider("toy-shared", &toy_provider);
        auto                    dom = gr::ComputeDomain::gpu_shared("toy-shared", /*idx*/ 0);
        auto                    bd  = gr::bind(dom, &mr);
        std::pmr::vector<float> vf(1024, bd.allocator<float>());
        expect(eq(vf.size(), 1024UZ));
    };

    "basic thread smoke (bind+alloc)"_test = [] {
        CountingMR mr;
        gr::ComputeRegistry::instance().register_provider("toy-thread", &toy_provider);
        gr::ComputeDomain dom;
        dom.kind    = "gpu";
        dom.backend = "toy-thread";

        std::vector<std::thread> th;
        for (std::size_t i = 0UZ; i < 8UZ; ++i) {
            th.emplace_back([&] {
                auto                  bd = gr::bind(dom, &mr);
                auto                  a  = bd.allocator<int>();
                std::pmr::vector<int> v(512, a);
                expect(eq(v.size(), 512UZ));
            });
        }
        for (auto& t : th) {
            t.join();
        }
        expect(mr.allocs.load() >= 8UZ);
    };
};

int main() { /* not needed for UT */ }
