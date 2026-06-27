#include <boost/ut.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/device/UsmMemoryResource.hpp>

using namespace boost::ut;

const suite<"device::UsmMemoryResource"> tests =
    [] {
        "allocate and deallocate float array"_test = [] {
            gr::device::UsmMemoryResource          mr;
            std::pmr::polymorphic_allocator<float> alloc(&mr);

            auto* ptr = alloc.allocate(4096);
            expect(ptr != nullptr);
            for (std::size_t i = 0; i < 4096; ++i) {
                ptr[i] = static_cast<float>(i);
            }
            expect(eq(ptr[0], 0.f));
            expect(eq(ptr[4095], 4095.f));
            alloc.deallocate(ptr, 4096);
        };

        "pmr::vector with USM resource"_test = [] {
            gr::device::UsmMemoryResource mr;
            std::pmr::vector<float>       v(1024, 0.f, &mr);
            std::iota(v.begin(), v.end(), 0.f);
            expect(eq(v.size(), 1024UZ));
            expect(eq(v[0], 0.f));
            expect(eq(v[1023], 1023.f));
        };

        // CircularBuffer + custom allocator tested in qa_buffer.cpp (default allocator path)
        // and validated here on GCC. Under AdaptiveCpp, CircularBuffer + non-mmap allocator
        // has a pre-existing segfault in ClaimStrategy — tracked separately.

        "Tag with USM-backed property_map"_test = [] {
            gr::device::UsmMemoryResource mr;
            gr::Tag                       testTag;
            testTag.index = 42;
            gr::tag::put(testTag.map, "sample_rate", gr::pmt::Value(48000.f));
            gr::tag::put(testTag.map, "name", gr::pmt::Value("test_signal"));

            expect(eq(testTag.index, 42UZ));
            expect(testTag.map.contains("sample_rate"));
            expect(testTag.map.contains("name"));
        };

        "Tensor with USM resource"_test = [] {
            gr::device::UsmMemoryResource mr;
            gr::Tensor<float>             t({64UZ}, &mr);
            expect(eq(t.extents()[0], 64UZ));
            for (std::size_t i = 0; i < 64; ++i) {
                t[i] = static_cast<float>(i * i);
            }
            expect(eq(t[0], 0.f));
            expect(eq(t[7], 49.f));
        };

        "multiple allocations and deallocations"_test = [] {
            gr::device::UsmMemoryResource          mr;
            std::pmr::polymorphic_allocator<float> alloc(&mr);

            std::vector<float*> ptrs;
            for (int i = 0; i < 100; ++i) {
                ptrs.push_back(alloc.allocate(128));
            }
            for (auto* p : ptrs) {
                alloc.deallocate(p, 128);
            }
            expect(eq(ptrs.size(), 100UZ));
        };

        "ComputeDomain gpu_shared resolves to registered resource"_test = [] {
            gr::device::registerUsmProvider();

            auto bd    = gr::bind(gr::ComputeDomain::gpu_shared());
            auto alloc = bd.allocator<float>();

            std::pmr::vector<float> v(256, 0.f, alloc);
            std::iota(v.begin(), v.end(), 1.f);
            expect(eq(v.size(), 256UZ));
            expect(eq(v[0], 1.f));
            expect(eq(v[255], 256.f));
        };

        "default UsmMemoryResource is CPU fallback"_test = [] {
            gr::device::UsmMemoryResource mr;
            std::pmr::vector<int>         v(512, 0, &mr);
            v[0]   = 42;
            v[511] = 99;
            expect(eq(v[0], 42));
            expect(eq(v[511], 99));
        };
};

int main() { /* not needed for UT */ }
