#include <atomic>
#include <boost/ut.hpp>
#include <numeric>
#include <thread>
#include <vector>

#include <algorithm>
#include <gnuradio-4.0/ComputeDomain.hpp>
#include <optional>

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

    "unknown backend returns error"_test = [] {
        gr::ComputeDomain dom;
        dom.kind    = "gpu";
        dom.backend = "does-not-exist";
        auto result = gr::ComputeRegistry::instance().resolve(dom, nullptr);
        expect(!result.has_value());
        // bind() falls back to new_delete_resource
        auto bd = gr::bind(dom);
        expect(bd.mr == std::pmr::new_delete_resource());
    };

    "provider returned null returns error"_test = [] {
        gr::ComputeRegistry::instance().register_provider("null", &null_provider);
        gr::ComputeDomain dom;
        dom.kind    = "gpu";
        dom.backend = "null";
        auto result = gr::ComputeRegistry::instance().resolve(dom, nullptr);
        expect(!result.has_value());
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

const suite<"ComputeDomain::parse"> _parseTests = [] {
    using namespace std::string_view_literals;

    "parse host variants"_test = [] {
        for (auto s : {"host"sv, "default_cpu"sv, "default_io"sv, ""sv}) {
            auto d = gr::ComputeDomain::parse(s);
            expect(eq(d.kind, "host"sv)) << s;
            expect(d.access == gr::Access::HostOnly) << s;
            expect(eq(d.backend, "none"sv)) << s;
            expect(eq(d.deviceIndex, -1)) << s;
        }
    };

    "parse gpu default backend"_test = [] {
        auto d = gr::ComputeDomain::parse("gpu");
        expect(eq(d.kind, "gpu"sv));
        expect(d.access == gr::Access::Shared);
        expect(eq(d.backend, "sycl"sv));
        expect(eq(d.deviceIndex, -1));
    };

    "parse gpu:sycl"_test = [] {
        auto d = gr::ComputeDomain::parse("gpu:sycl");
        expect(eq(d.kind, "gpu"sv));
        expect(eq(d.backend, "sycl"sv));
        expect(eq(d.deviceIndex, -1));
    };

    "parse gpu:sycl:0"_test = [] {
        auto d = gr::ComputeDomain::parse("gpu:sycl:0");
        expect(eq(d.kind, "gpu"sv));
        expect(eq(d.backend, "sycl"sv));
        expect(eq(d.deviceIndex, 0));
    };

    "parse gpu:cuda:3"_test = [] {
        auto d = gr::ComputeDomain::parse("gpu:cuda:3");
        expect(eq(d.kind, "gpu"sv));
        expect(eq(d.backend, "cuda"sv));
        expect(eq(d.deviceIndex, 3));
    };

    "parse gpu:gl"_test = [] {
        auto d = gr::ComputeDomain::parse("gpu:gl");
        expect(eq(d.kind, "gpu"sv));
        expect(eq(d.backend, "gl"sv));
        expect(eq(d.deviceIndex, -1));
    };

    "parse gpu:hip:1"_test = [] {
        auto d = gr::ComputeDomain::parse("gpu:hip:1");
        expect(eq(d.kind, "gpu"sv));
        expect(eq(d.backend, "hip"sv));
        expect(eq(d.deviceIndex, 1));
    };

    "parse fpga"_test = [] {
        auto d = gr::ComputeDomain::parse("fpga");
        expect(eq(d.kind, "fpga"sv));
        expect(d.access == gr::Access::Shared);
        expect(eq(d.backend, "none"sv));
        expect(eq(d.deviceIndex, -1));
    };

    "parse tpu"_test = [] {
        auto d = gr::ComputeDomain::parse("tpu");
        expect(eq(d.kind, "tpu"sv));
        expect(d.access == gr::Access::Shared);
        expect(eq(d.backend, "none"sv));
    };

    "parse unknown falls back to host"_test = [] {
        for (auto s : {"custom_pool"sv, "my_thread"sv, "unknown:stuff"sv}) {
            auto d = gr::ComputeDomain::parse(s);
            expect(eq(d.kind, "host"sv)) << s;
            expect(d.access == gr::Access::HostOnly) << s;
        }
    };

    "parse gpu with SYCL-reported backend passes through"_test = [] {
        auto d = gr::ComputeDomain::parse("gpu:vulkan");
        expect(eq(d.kind, "gpu"sv));
        expect(eq(d.backend, "vulkan"sv)) << "unknown backends are passed through for SYCL device names";
    };

    "parse host:sycl keeps its backend and selects a device"_test = [] {
        auto d = gr::ComputeDomain::parse("host:sycl");
        expect(eq(d.kind, "host"sv));
        expect(eq(d.backend, "sycl"sv)) << "a SYCL CPU device is host memory with SYCL execution";
        expect(d.access == gr::Access::HostOnly);
        expect(d.isDevice()) << "must dispatch through the SYCL runtime";

        auto indexed = gr::ComputeDomain::parse("host:sycl:1");
        expect(eq(indexed.backend, "sycl"sv));
        expect(eq(indexed.deviceIndex, 1));
        expect(indexed.isDevice());
    };

    "plain host domains are not devices"_test = [] {
        for (auto s : {"host"sv, "default_cpu"sv, "default_io"sv, ""sv}) {
            expect(!gr::ComputeDomain::parse(s).isDevice()) << s;
        }
    };

    "an unrecognised kind never becomes a device"_test = [] {
        for (auto s : {"custom_pool"sv, "unknown:stuff"sv, "cpu:sycl"sv, "my_thread:sycl"sv}) { // `cpu` is not a kind
            auto d = gr::ComputeDomain::parse(s);
            expect(eq(d.kind, "host"sv)) << s;
            expect(eq(d.backend, "none"sv)) << s << ": an unknown kind must not smuggle in a backend";
            expect(!d.isDevice()) << s;
        }
    };

    "gpu domains are devices"_test = [] {
        expect(gr::ComputeDomain::parse("gpu").isDevice());
        expect(gr::ComputeDomain::parse("gpu:sycl:0").isDevice());
        expect(gr::ComputeDomain::parse("fpga").isDevice());
    };

    "parse gpu with vendor-specific backend"_test = [] {
        std::string input = "gpu:Intel(R) UHD Graphics:0";
        auto        d     = gr::ComputeDomain::parse(input);
        expect(eq(d.kind, "gpu"sv));
        expect(eq(d.backend, "Intel(R) UHD Graphics"sv));
        expect(eq(d.deviceIndex, 0));
    };
};

const boost::ut::suite<"ComputeDomain resolution"> resolutionTests = [] {
    using namespace boost::ut;
    using namespace std::string_literals;

    const auto servingNothing = [](std::string_view) { return std::optional<std::string>{}; };
    const auto serving        = [](std::initializer_list<std::string_view> served) { //
        return [served](std::string_view rung) { return std::ranges::find(served, rung) != served.end() ? std::optional<std::string>(rung) : std::nullopt; };
    };

    "the canonical spelling names the backend, and the index only when one was given"_test = [] {
        expect(eq(gr::canonicalDomainName(gr::ComputeDomain::parse("host")), "host"s));
        expect(eq(gr::canonicalDomainName(gr::ComputeDomain::parse("gpu")), "gpu:sycl"s)) << "a bare kind still names the backend it parses to";
        expect(eq(gr::canonicalDomainName(gr::ComputeDomain::parse("gpu:sycl:0")), "gpu:sycl:0"s));
        expect(eq(gr::canonicalDomainName(gr::ComputeDomain::parse("host:sycl")), "host:sycl"s));
        expect(eq(gr::canonicalDomainName(gr::ComputeDomain::parse("gpu:cuda:x")), "gpu:cuda"s)) << "an unparsable index is no index at all";
    };

    "a bare kind resolves to the device that serves it"_test = [serving] {
        const gr::DomainResolution resolution = gr::resolveComputeDomain("gpu", serving({"gpu:sycl"}));
        expect(eq(resolution.resolved, "gpu:sycl"s));
        expect(!resolution.downgraded) << "naming the same device a shorter way is not a downgrade";
    };

    "an index nobody serves falls back to the same kind and backend, not to the host"_test = [serving] {
        const gr::DomainResolution resolution = gr::resolveComputeDomain("gpu:sycl:3", serving({"gpu:sycl", "host:sycl"}));
        expect(eq(resolution.resolved, "gpu:sycl"s)) << "the host:sycl rung would hand a device-only ring to a CPU kernel";
        expect(resolution.downgraded);
        expect(eq(resolution.declared, "gpu:sycl:3"s)) << "the warning has to name what was asked for";
    };

    "an unserved backend falls back to the SYCL host device"_test = [serving] {
        const gr::DomainResolution resolution = gr::resolveComputeDomain("gpu:cuda", serving({"host:sycl"}));
        expect(eq(resolution.resolved, "host:sycl"s));
        expect(resolution.downgraded);
    };

    "with nothing served at all the ladder ends at the plain host"_test = [servingNothing] {
        const gr::DomainResolution resolution = gr::resolveComputeDomain("gpu:sycl", servingNothing);
        expect(eq(resolution.resolved, "host"s));
        expect(resolution.downgraded);
    };

    "a host domain resolves to itself without consulting the registry"_test = [] {
        bool                       probed     = false;
        const gr::DomainResolution resolution = gr::resolveComputeDomain("host", [&probed](std::string_view) {
            probed = true;
            return std::optional<std::string>{};
        });
        expect(eq(resolution.resolved, "host"s));
        expect(!resolution.downgraded) << "a graph that asked for nothing must not be told it was downgraded";
        expect(!probed);
    };
};

int main() { /* not needed for UT */ }
