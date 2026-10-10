#include <boost/ut.hpp>

#include <algorithm>
#include <numeric>
#include <print>
#include <ranges>
#include <span>
#include <vector>

#include <gnuradio-4.0/CircularBuffer.hpp>
#include <gnuradio-4.0/DataSet.hpp>
#include <gnuradio-4.0/Tag.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/device/UsmMemoryResource.hpp>
#include <gnuradio-4.0/test/DeviceExpectation.hpp>

using namespace boost::ut;

int main() { // a SYCL kernel in this TU stops AdaptiveCpp registering static suites, so the tests run from main()
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
        gr::property_map              tagPayload;
        gr::tag::put(tagPayload, "sample_rate", gr::pmt::Value(48000.f));
        gr::tag::put(tagPayload, "name", gr::pmt::Value("test_signal"));

        gr::Tag testTag{42UZ, tagPayload};

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

    "a DataSet on shared USM is read and written by a device kernel"_test = [] {
#if GR_DEVICE_HAS_SYCL
        sycl::queue queue = [] {
            try {
                return sycl::queue{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
            } catch (const sycl::exception&) {
                return sycl::queue{sycl::property::queue::in_order{}};
            }
        }();
        std::println("DataSet kernel runs on: {}", queue.get_device().get_info<sycl::info::device::name>());
        if (gr::testing::deviceDomainRequired("gpu:sycl")) {
            expect(queue.get_device().is_gpu()) << "GR4_REQUIRE_DEVICE asks for a GPU" << fatal;
        }
        gr::device::UsmMemoryResource usm{queue, gr::device::UsmKind::shared};
#else
        gr::device::UsmMemoryResource usm;
#endif
        gr::DataSet<float> ds{gr::DataSet<float>::allocator_type{&usm}};
        ds.signal_names.emplace_back("ramp");
        ds.signal_values.resize(4096UZ);
        std::iota(ds.signal_values.begin(), ds.signal_values.end(), 0.f);
        const std::span<float> values = ds.signalValues(0UZ);
        expect(ds.signal_names[0].get_allocator().resource() == &usm);

#if GR_DEVICE_HAS_SYCL
        if (queue.get_device().is_gpu()) {
            expect(sycl::get_pointer_type(values.data(), queue.get_context()) == sycl::usm::alloc::shared) << "the payload is not in shared USM";
        }
        float* const data = values.data();
        queue.parallel_for(sycl::range<1>{values.size()}, [=](sycl::id<1> i) { data[i] *= 2.f; }).wait();
#else
        std::ranges::transform(values, values.begin(), [](float v) { return 2.f * v; });
#endif
        expect(std::ranges::all_of(std::views::iota(0UZ, values.size()), [&values](std::size_t i) { return values[i] == 2.f * static_cast<float>(i); })) << "the doubled payload did not reach the host";
    };
}
