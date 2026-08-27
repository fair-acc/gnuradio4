#include <boost/ut.hpp>

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cmath>
#include <numeric>
#include <print>
#include <vector>

#include <gnuradio-4.0/meta/UnitTestHelper.hpp>
#include <gnuradio-4.0/meta/formatter.hpp>

const boost::ut::suite<"ONNX Runtime Installation"> onnxTests = [] {
    using namespace boost::ut;
    using namespace std::string_literals;

    "prints the ONNX Runtime version, build info and available providers"_test = [] {
        const auto* apiBase = OrtGetApiBase();
        expect(nothrow([&] {
            const auto* api = apiBase->GetApi(ORT_API_VERSION);
            std::println("══════════════════════════════════════════════════════");
            std::println("ONNX Runtime Installation Info");
            std::println("══════════════════════════════════════════════════════");
            std::println("  ORT API Version:    {}", ORT_API_VERSION);
            std::println("  Build Info:         {}", api->GetBuildInfoString());
            std::println("  Available Providers:");
            for (const auto& p : Ort::GetAvailableProviders()) {
                std::println("    - {}", p);
            }
#ifdef __EMSCRIPTEN__
            std::println("  Platform:           WebAssembly (Emscripten)");
#else
            std::println("  Platform:           Native");
#endif
            std::println("══════════════════════════════════════════════════════\n");
        }));
    };

    "creates an ONNX Runtime environment without throwing"_test = [] { expect(nothrow([] { Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "gr-onnx-test"); })); };

    "configures session options without throwing"_test = [] {
        expect(nothrow([] {
            Ort::SessionOptions opts;
            opts.SetIntraOpNumThreads(1);
            opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        }));
    };

    "creates CPU memory info and reports the Cpu allocator name"_test = [] {
        expect(nothrow([] {
            auto info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            expect(eq(info.GetAllocatorName(), "Cpu"s));
        }));
    };

    "allocates and frees memory through the default allocator"_test = [] {
        expect(nothrow([] {
            Ort::Env                         env(ORT_LOGGING_LEVEL_WARNING, "alloc-test");
            Ort::AllocatorWithDefaultOptions alloc;
            void*                            ptr = alloc.Alloc(1024);
            expect(neq(ptr, nullptr));
            alloc.Free(ptr);
        }));
    };

    "creates a 1D float tensor with the expected shape and values"_test = [] {
        expect(nothrow([] {
            auto                             mem   = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            constexpr std::array<int64_t, 1> shape = {4};
            std::vector<float>               data  = {1.f, 2.f, 3.f, 4.f};

            auto tensor = Ort::Value::CreateTensor<float>(mem, data.data(), data.size(), shape.data(), shape.size());
            auto info   = tensor.GetTensorTypeAndShapeInfo();

            expect(tensor.IsTensor());
            expect(eq(info.GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
            expect(eq(info.GetElementCount(), 4UZ));
            expect(eq(info.GetShape(), std::vector<int64_t>{4}));

            const float* out = tensor.GetTensorData<float>();
            for (std::size_t i = 0; i < 4; ++i) {
                expect(lt(std::abs(out[i] - data[i]), 1e-6f));
            }
        }));
    };

    "creates a 2D float tensor with the expected shape and element count"_test = [] {
        expect(nothrow([] {
            auto                             mem   = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            constexpr std::array<int64_t, 2> shape = {2, 3};
            std::vector<float>               data  = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};

            auto tensor = Ort::Value::CreateTensor<float>(mem, data.data(), data.size(), shape.data(), shape.size());
            auto info   = tensor.GetTensorTypeAndShapeInfo();

            expect(eq(info.GetElementCount(), 6UZ));
            expect(eq(info.GetShape(), (std::vector<int64_t>{2, 3})));
        }));
    };

    "creates tensors for every supported numeric type"_test = [] {
        auto                             mem   = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        constexpr std::array<int64_t, 1> shape = {4};

        "creates a float tensor with element type FLOAT"_test = [&] {
            std::vector<float> d = {1.f, 2.f, 3.f, 4.f};
            auto               t = Ort::Value::CreateTensor<float>(mem, d.data(), d.size(), shape.data(), shape.size());
            expect(eq(t.GetTensorTypeAndShapeInfo().GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        };

        "creates a double tensor with element type DOUBLE"_test = [&] {
            std::vector<double> d = {1., 2., 3., 4.};
            auto                t = Ort::Value::CreateTensor<double>(mem, d.data(), d.size(), shape.data(), shape.size());
            expect(eq(t.GetTensorTypeAndShapeInfo().GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE));
        };

        "creates an int32 tensor with element type INT32"_test = [&] {
            std::vector<int32_t> d = {1, 2, 3, 4};
            auto                 t = Ort::Value::CreateTensor<int32_t>(mem, d.data(), d.size(), shape.data(), shape.size());
            expect(eq(t.GetTensorTypeAndShapeInfo().GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32));
        };

        "creates an int64 tensor with element type INT64"_test = [&] {
            std::vector<int64_t> d = {1, 2, 3, 4};
            auto                 t = Ort::Value::CreateTensor<int64_t>(mem, d.data(), d.size(), shape.data(), shape.size());
            expect(eq(t.GetTensorTypeAndShapeInfo().GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64));
        };
    };

    "configures run options without throwing"_test = [] {
        expect(nothrow([] {
            Ort::RunOptions opts;
            opts.SetRunTag("gr-onnx-test");
            opts.SetRunLogVerbosityLevel(2);
        }));
    };

    "the CPU execution provider is always listed as available"_test = [] {
        auto providers = Ort::GetAvailableProviders();
        expect(!providers.empty());
        expect(std::ranges::find(providers, "CPUExecutionProvider") != providers.end());
    };

#ifndef __EMSCRIPTEN__
    "configures intra-op and inter-op thread counts on native builds"_test = [] {
        expect(nothrow([] {
            Ort::Env            env(ORT_LOGGING_LEVEL_WARNING, "thread-test");
            Ort::SessionOptions opts;
            opts.SetIntraOpNumThreads(2);
            opts.SetInterOpNumThreads(1);
        }));
    };
#else
    "configures a single-threaded session on WASM builds"_test = [] {
        expect(nothrow([] {
            Ort::Env            env(ORT_LOGGING_LEVEL_WARNING, "wasm-test");
            Ort::SessionOptions opts;
            opts.SetIntraOpNumThreads(1);
        }));
    };
#endif

    "moving an Ort::Value leaves the target holding a valid tensor"_test = [] {
        expect(nothrow([] {
            auto                             mem   = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            constexpr std::array<int64_t, 1> shape = {4};
            std::vector<float>               data  = {1.f, 2.f, 3.f, 4.f};

            auto t1 = Ort::Value::CreateTensor<float>(mem, data.data(), data.size(), shape.data(), shape.size());
            auto t2 = std::move(t1);
            expect(t2.IsTensor());
        }));
    };

    "creates a batch of independent tensors"_test = [] {
        expect(nothrow([] {
            auto                             mem   = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            constexpr std::array<int64_t, 2> shape = {1, 8};

            std::vector<Ort::Value> tensors;
            for (std::size_t i = 0; i < 4; ++i) {
                std::vector<float> data(8);
                std::iota(data.begin(), data.end(), static_cast<float>(i * 8));
                tensors.push_back(Ort::Value::CreateTensor<float>(mem, data.data(), data.size(), shape.data(), shape.size()));
            }

            expect(eq(tensors.size(), 4UZ));
            for (const auto& t : tensors) {
                expect(t.IsTensor());
            }
        }));
    };

    "creates a tensor with an [N,2] complex IQ layout"_test = [] {
        expect(nothrow([] {
            auto                             mem   = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            constexpr std::size_t            N     = 64;
            constexpr std::array<int64_t, 2> shape = {N, 2};

            std::vector<float> iq(N * 2);
            for (std::size_t i = 0; i < N; ++i) {
                float phase   = 2.f * 3.14159f * static_cast<float>(i) / N;
                iq[i * 2]     = std::cos(phase);
                iq[i * 2 + 1] = std::sin(phase);
            }

            auto tensor = Ort::Value::CreateTensor<float>(mem, iq.data(), iq.size(), shape.data(), shape.size());
            auto info   = tensor.GetTensorTypeAndShapeInfo();

            expect(eq(info.GetElementCount(), N * 2));
            expect(eq(info.GetShape(), (std::vector<int64_t>{N, 2})));
        }));
    };
};

int main() { /* not needed for boost::ut */ }
