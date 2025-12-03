#include <boost/ut.hpp>

#include <gnuradio-4.0/Tensor.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory_resource>
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

template<class T, std::size_t N>
requires(N == 2)
constexpr T det(const gr::Tensor<T, N, N>& A) {
    return A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0];
}

template<class T, std::size_t N>
requires(N == 3)
constexpr T det(const gr::Tensor<T, N, N>& A) {
    return A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1]) - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0]) + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]);
}

const boost::ut::suite<"Tensor<T> Basic Functionality"> _tensorBasic = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using namespace std::string_view_literals;
    using gr::TensorBase;
    using gr::Tensor;
    using gr::TensorView;

    static_assert(std::is_trivially_copyable_v<TensorBase<double, true, 3UZ>>); // fully statically defined ranks
    static_assert(std::is_trivially_copyable_v<TensorBase<double, true, 6UZ, 6UZ>>);
    static_assert(std::is_trivially_copyable_v<TensorBase<double, false, 3UZ>>);
    static_assert(std::is_trivially_copyable_v<TensorBase<double, false, 6UZ, 6UZ>>);
    static_assert(std::is_trivially_copyable_v<TensorBase<double, false>>);                      // non-owning dynamic case
    static_assert(std::is_trivially_copyable_v<TensorBase<double, false, std::dynamic_extent>>); // non-owning, unmanaged case (e.g. TensorView)

    // Tensor w/ fully static rank definition
    static_assert(std::is_trivially_copyable_v<Tensor<float, 3UZ>>);
    static_assert(std::is_trivially_copyable_v<Tensor<float, 6UZ, 6UZ>>);

    // Tensor w/ semi-static/dynamic rank definition
    static_assert(not std::is_trivially_copyable_v<Tensor<float, std::dynamic_extent>>);
    static_assert(std::is_trivially_copyable_v<Tensor<float, 6UZ, 6UZ>>);

    // TensorView
    static_assert(std::is_trivially_copyable_v<TensorView<float, 4>>);
    static_assert(std::is_trivially_copyable_v<TensorView<float, std::dynamic_extent>>);
    static_assert(std::is_trivially_copyable_v<TensorView<float>>);

    "Tensor[***] concepts"_test = [] {
        // TensorLike concept
        static_assert(gr::is_tensor<Tensor<float>>);
        static_assert(gr::TensorLike<Tensor<float>>);
        static_assert(gr::is_tensor<Tensor<float, 2, 3>>);
        static_assert(gr::TensorLike<Tensor<float, 2, 3>>);
        static_assert(gr::is_tensor<Tensor<float, std::dynamic_extent>>);
        static_assert(gr::TensorLike<Tensor<float, std::dynamic_extent>>);
        static_assert(gr::TensorLike<TensorView<float>>);

        // TensorOf<T> concept
        static_assert(gr::TensorOf<Tensor<float>, float>);
        static_assert(!gr::TensorOf<Tensor<float>, double>);
        static_assert(gr::TensorOf<Tensor<float, 2, 3>, float>);
        static_assert(!gr::TensorOf<Tensor<float, 2, 3>, double>);
        static_assert(gr::TensorOf<Tensor<float, std::dynamic_extent>, float>);
        static_assert(!gr::TensorOf<Tensor<float, std::dynamic_extent>, double>);
        static_assert(gr::TensorOf<TensorView<float>, float>);
        static_assert(gr::TensorOf<TensorView<float, 2, 3>, float>);
        static_assert(!gr::TensorOf<TensorView<float>, double>);

        // StaticTensorOf<T> concept
        static_assert(!gr::StaticTensorOf<Tensor<float>, float>);
        static_assert(gr::StaticTensorOf<Tensor<float, 2, 3>, float>);
        static_assert(!gr::StaticTensorOf<Tensor<float, std::dynamic_extent>, float>);
        static_assert(!gr::StaticTensorOf<TensorView<float>, float>);
        static_assert(gr::StaticTensorOf<TensorView<float, 2, 3>, float>);
        static_assert(!gr::StaticTensorOf<TensorView<float, std::dynamic_extent>, float>);

        // StaticRankTensorOf<T> concept
        static_assert(!gr::StaticRankTensorOf<Tensor<float>, float>);
        static_assert(gr::StaticRankTensorOf<Tensor<float, 2, 3>, float>);
        static_assert(gr::StaticRankTensorOf<Tensor<float, std::dynamic_extent>, float>);
        static_assert(!gr::StaticRankTensorOf<TensorView<float>, float>);
        static_assert(gr::StaticRankTensorOf<TensorView<float, 2, 3>, float>);
        static_assert(gr::StaticRankTensorOf<TensorView<float, std::dynamic_extent>, float>);

        // DynamicTensorOf<T> concept
        static_assert(gr::DynamicTensorOf<Tensor<float>, float>);
        static_assert(!gr::DynamicTensorOf<Tensor<float, 2, 3>, float>);
        static_assert(!gr::DynamicTensorOf<Tensor<float, std::dynamic_extent>, float>);
        static_assert(gr::DynamicTensorOf<TensorView<float>, float>);
        static_assert(!gr::DynamicTensorOf<TensorView<float, 2, 3>, float>);
        static_assert(!gr::DynamicTensorOf<TensorView<float, std::dynamic_extent>, float>);

        // TensorViewOf<T> concept
        static_assert(!gr::TensorViewOf<Tensor<float>, float>);
        static_assert(!gr::TensorViewOf<Tensor<float, 2, 3>, float>);
        static_assert(!gr::TensorViewOf<Tensor<float, std::dynamic_extent>, float>);
        static_assert(gr::TensorViewOf<TensorView<float>, float>);
        static_assert(gr::TensorViewOf<TensorView<float, 2, 3>, float>);
        static_assert(gr::TensorViewOf<TensorView<float, std::dynamic_extent>, float>);
    };

    "User API examples"_test = [] {
        // case 1: fully dynamic rank & extents
        std::vector<double> dynData{1, 2, 3, 4, 5, 6};
        Tensor<double>      A({3UZ, 2UZ}, dynData);
        expect(eq(A.rank(), 2UZ));
        expect(eq(A.size(), 6UZ));
        expect(eq(A[2, 1], 6.0));

        // case 2: rank=1 with dynamic extent -> data-only ctor
        Tensor<double, std::dynamic_extent> B({1, 2, 3, 4, 5});
        expect(eq(B.rank(), 1UZ));
        expect(eq(B.size(), 5UZ));
        expect(eq(B[4], 5.0));

        // case 3: fully static 2×3, ultra-compact
        constexpr Tensor<double, 2UZ, 3UZ> C({
            1, 2, 3, // row 0
            4, 5, 7  // row 1
        });
        static_assert(Tensor<double, 2UZ, 3UZ>::static_rank() == 2);
        static_assert(Tensor<double, 2UZ, 3UZ>::extent<1>() == 3);
        static_assert(sizeof(Tensor<double, 2, 3>) == sizeof(std::array<double, 6>));

        expect(eq(C.size(), 6UZ));
        expect(eq(C[1, 2], 7.0));

        // constexpr proof for flat ctor
        constexpr Tensor<double, 2UZ, 2UZ> D({1, 2, 3, 4});
        static_assert(det(D) == -2);
    };

    "Default construction"_test = [] {
        "empty"_test = [] {
            Tensor<int> tensor;
            expect(eq(tensor.rank(), 0UZ));
            expect(eq(tensor.size(), 0UZ));
            expect(tensor.empty());
            expect(eq(tensor.capacity(), 0UZ));
        };
        "default value"_test = [] {
            Tensor<int> tensor(42);
            expect(eq(tensor.rank(), 0UZ));
            expect(eq(tensor.size(), 0UZ));
            expect(tensor.empty());
            expect(eq(tensor.capacity(), 0UZ));
        };
        "default value static tensor"_test = [] {
            Tensor<int, 2UZ, 1UZ> tensor(42);
            expect(eq(tensor.rank(), 2UZ));
            expect(eq(tensor.size(), 2UZ));
            expect(!tensor.empty());
            expect(eq(tensor[0, 0], 42));
            expect(eq(tensor[1, 0], 42));
        };
    };

    "Static constexpr construction"_test = [] {
        constexpr Tensor<int, 2UZ, 2UZ> tensor{1, 2, 3, 4};
        static_assert(tensor.size() == 4);
        static_assert((tensor[0, 0]) == 1);
    };

    "Type sizes"_test = [] {
        static_assert(sizeof(Tensor<double>) == sizeof(std::pmr::vector<double>) + sizeof(Tensor<double, std::dynamic_extent>::dynamic_extents_store));
        static_assert(sizeof(Tensor<double, std::dynamic_extent>) == sizeof(gr::pmr::vector<double, true>) + sizeof(Tensor<double, std::dynamic_extent>::semi_static_extents_store));
        static_assert(sizeof(Tensor<double, 3UZ, 2UZ>) == 3UZ * 2UZ * sizeof(double));
    };

    "Semi-static tensor"_test = [] {
        Tensor<int, std::dynamic_extent, std::dynamic_extent> tensor({3UZ, 4UZ});
        expect(eq(tensor.rank(), 2UZ));
        static_assert(Tensor<int, std::dynamic_extent, std::dynamic_extent>::static_rank() == 2UZ);
        expect(eq(tensor.size(), 12UZ));

        auto extents = tensor.extents();
        expect(eq(extents[0], 3UZ));
        expect(eq(extents[1], 4UZ));
    };

    "Extents construction"_test = [] {
        "single dimension"_test = [] {
            Tensor<int> vec({5UZ});
            expect(eq(vec.rank(), 1UZ));
            expect(eq(vec.size(), 5UZ));
            expect(eq(vec.extent(0UZ), 5UZ));
        };

        "multi-dimensional -- Variant A"_test = [] {
            Tensor<int> matrix(gr::extents_from, {3UZ, 4UZ});
            expect(eq(matrix.rank(), 2UZ));
            expect(eq(matrix.size(), 12UZ));
            expect(eq(matrix.extent(0UZ), 3UZ));
            expect(eq(matrix.extent(1UZ), 4UZ));
        };

        "multi-dimensional -- Variant B"_test = [] {
            Tensor<float> matrix(/*gr::extents_from, -- not needed if T != std::size_t */ {3UZ, 4UZ});
            expect(eq(matrix.rank(), 2UZ));
            expect(eq(matrix.size(), 12UZ));
            expect(eq(matrix.extent(0UZ), 3UZ));
            expect(eq(matrix.extent(1UZ), 4UZ));
        };

        "3D tensor"_test = [] {
            Tensor<double> tensor3d({2UZ, 3UZ, 4UZ});
            expect(eq(tensor3d.rank(), 3UZ));
            expect(eq(tensor3d.size(), 24UZ));
        };
    };

    "Count-value construction"_test = [] {
        Tensor<double> tensor(5UZ, 42.0);
        expect(eq(tensor.rank(), 1UZ));
        expect(eq(tensor.size(), 5UZ));
        expect(std::ranges::all_of(tensor, [](double x) { return x == 42.0; }));
    };

    "Iterator construction"_test = [] {
        std::vector<int> data{10, 20, 30, 40};
        Tensor<int>      tensor(data.begin(), data.end());
        expect(eq(tensor.rank(), 1UZ));
        expect(eq(tensor.size(), 4UZ));
        expect(std::ranges::equal(tensor, data));
    };

    "Extents-data construction"_test = [] {
        std::vector<int> data{1, 2, 3, 4, 5, 6};

        "valid construction"_test = [&] {
            Tensor<int> tensor({2UZ, 3UZ}, data);
            expect(eq(tensor.rank(), 2UZ));
            expect(eq(tensor.size(), 6UZ));
            expect(eq((tensor[0, 0]), 1));
            expect(eq((tensor[1, 2]), 6));
        };

        "size mismatch throws"_test = [&] { expect(throws([&] { Tensor<int>({2UZ, 2UZ}, data); })); };
    };
};

const boost::ut::suite<"Tensor<T> Conversion"> _tensorConversion = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "Cross-type constructors"_test = [] {
        "static to dynamic"_test = [] {
            constexpr Tensor<int, 2UZ, 3UZ> static_tensor{1, 2, 3, 4, 5, 6};
            Tensor<int>                     dynamic_tensor(static_tensor);
            expect(eq(dynamic_tensor.rank(), 2UZ));
            expect(eq(dynamic_tensor.size(), 6UZ));
            expect(std::ranges::equal(dynamic_tensor, static_tensor));

            Tensor<int> dynamic_tensor2; // assignment operator
            dynamic_tensor2 = static_tensor;
            expect(std::ranges::equal(dynamic_tensor2, static_tensor));
        };

        "dynamic to static with size check"_test = [] {
            Tensor<double> source({2UZ, 2UZ});
            std::iota(source.begin(), source.end(), 1.0);
            Tensor<double, 2UZ, 2UZ> target(source);
            expect(std::ranges::equal(target, source));

            Tensor<double> wrong_size({2UZ, 3UZ});
            expect(throws([&] { Tensor<double, 2UZ, 2UZ>{wrong_size}; })) << "wrong size should throw";

            Tensor<double, 2UZ, 2UZ> dest;
            expect(throws([&] { dest = wrong_size; }));
        };
    };

    "static 1D tensor -> std::array/std::vector/std::pmr::vector"_test = [] {
        Tensor<int, 4UZ> tensor{};
        for (std::size_t i = 0; i < tensor.size(); ++i) {
            tensor.data()[i] = static_cast<int>(10 + i);
        }

        auto a = static_cast<std::array<int, 4UZ>>(tensor);
        expect(eq(a.size(), 4UZ));
        for (std::size_t i = 0; i < a.size(); ++i) {
            expect(eq(a[i], tensor.data()[i]));
        }

        auto v = static_cast<std::vector<int>>(tensor);
        expect(eq(v.size(), 4UZ));
        for (std::size_t i = 0; i < v.size(); ++i) {
            expect(eq(v[i], tensor.data()[i]));
        }

        auto pv = static_cast<std::pmr::vector<int>>(tensor);
        expect(eq(pv.size(), 4UZ));
        for (std::size_t i = 0; i < pv.size(); ++i) {
            expect(eq(pv[i], tensor.data()[i]));
        }

        std::deque<int> d = static_cast<std::deque<int>>(tensor);
        expect(eq(d.size(), 4UZ));
        for (std::size_t i = 0; i < d.size(); ++i) {
            expect(eq(d[i], tensor.data()[i]));
        }
    };

    "dynamic 1D tensor -> std::vector/std::pmr::vector"_test = [] {
        Tensor<int> t({5UZ});
        for (std::size_t i = 0; i < t.size(); ++i) {
            t.data()[i] = static_cast<int>(100 + i);
        }

        auto v  = static_cast<std::vector<int>>(t);
        auto pv = static_cast<std::pmr::vector<int>>(t);

        expect(eq(v.size(), 5UZ));
        expect(eq(pv.size(), 5UZ));

        for (std::size_t i = 0; i < t.size(); ++i) {
            expect(eq(v[i], t.data()[i]));
            expect(eq(pv[i], t.data()[i]));
        }
    };

    "conversion throws for non-1D tensors"_test = [] {
        Tensor<int, 2UZ, 2UZ> tensor{};
        static_assert(tensor.rank() == 2UZ);

        expect(throws([&] { [[maybe_unused]] auto v = static_cast<std::vector<int>>(tensor); }));
        expect(throws([&] { [[maybe_unused]] auto pv = static_cast<std::pmr::vector<int>>(tensor); }));
        expect(throws([&] { [[maybe_unused]] auto a = static_cast<std::array<int, 4UZ>>(tensor); }));
    };

    "const Tensor -> Tensor<non-const>"_test = [] {
        using CTensor = Tensor<const int, 4UZ>;

        CTensor ct{};
        for (std::size_t i = 0; i < ct.size(); ++i) {
            const_cast<int*>(ct.data())[i] = static_cast<int>(i * 2); // just for test init
        }

        auto copy = static_cast<Tensor<int>>(ct);
        expect(eq(copy.size(), ct.size()));

        for (std::size_t i = 0; i < copy.size(); ++i) {
            expect(eq(copy.data()[i], ct.data()[i]));
        }
    };

    "Tensor <-> TensorView shares storage"_test = [] {
        Tensor<int, 4UZ> tensor{};
        std::iota(tensor.begin(), tensor.end(), 0);
        auto view = static_cast<gr::TensorView<int, 4UZ>>(tensor);
        expect(eq(std::to_address(tensor.data()), std::to_address(view.data())));
        view[1UZ] = 42;
        expect(eq(tensor[1UZ], 42)) << "mutate via view, observe via tensor";

        tensor[2UZ] = 99;
        expect(eq(view[2UZ], 99)) << "mutate via tensor, observe via view";
    };
};

const boost::ut::suite<"Tensor<T> access"> _tensorAccess = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "single index access"_test = [] {
        Tensor<int> tensor({5UZ});
        std::iota(tensor.begin(), tensor.end(), 10);

        "operator[]"_test = [&] {
            expect(eq(tensor[0], 10));
            expect(eq(tensor[4], 14));
        };

        "at() with bounds checking"_test = [&] { expect(nothrow([&] { std::ignore = tensor.at(0); })); };

        "front/back"_test = [&] {
            expect(eq(tensor.front(), 10));
            expect(eq(tensor.back(), 14));
        };
    };

    "multi-index access"_test = [] {
        Tensor<int> tensor({2UZ, 3UZ});

        for (std::size_t i = 0UZ; i < 2UZ; ++i) {
            for (std::size_t j = 0UZ; j < 3UZ; ++j) {
                tensor[i, j] = static_cast<int>(10UZ * i + j);
            }
        }

        expect(eq((tensor[0UZ, 0UZ]), 0));
        expect(eq((tensor[0UZ, 2UZ]), 2));
        expect(eq((tensor[1UZ, 0UZ]), 10));
        expect(eq((tensor[1UZ, 2UZ]), 12));
    };

    "Variadic at() methods"_test = [] {
        Tensor<int> tensor({3UZ, 4UZ, 2UZ});
        std::iota(tensor.begin(), tensor.end(), 0);

        "bounds-checked access"_test = [&] {
            expect(eq(tensor.at(0, 0, 0), 0));
            expect(eq(tensor.at(1, 2, 1), (tensor[1, 2, 1])));
        };

        "out of bounds"_test = [&] {
            expect(throws([&] { std::ignore = tensor.at(3, 0, 0); }));
            expect(throws([&] { std::ignore = tensor.at(0, 4, 0); }));
        };

        "wrong arity"_test = [&] { expect(throws([&] { std::ignore = tensor.at(0, 0); })); };
    };

    "std::span-based access"_test = [] {
        Tensor<int> tensor({2UZ, 3UZ});
        std::iota(tensor.begin(), tensor.end(), 0);

        std::array<std::size_t, 2UZ> indices{1, 2};
        expect(eq(tensor.at(indices), (tensor[1, 2])));

        std::array<std::size_t, 3> wrong_dim{0UZ};
        expect(throws([wrong_dim, &tensor] { std::ignore = tensor.at(std::span(wrong_dim)); })) << "wrong span dimension";

        std::array<std::size_t, 1> wrong_size{7UZ};
        expect(throws([wrong_size, &tensor] { std::ignore = tensor.at(std::span(wrong_size)); })) << "wrong span size";
    };
};

const boost::ut::suite<"Tensor<T> Shape Operations"> _tensorShape = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "Basic reshape"_test = [] {
        Tensor<int> tensor({2UZ, 3UZ});
        std::iota(tensor.begin(), tensor.end(), 0);

        tensor.reshape({3UZ, 2UZ});
        expect(eq(tensor.rank(), 2UZ));
        expect(eq(tensor.extent(0), 3UZ));
        expect(eq(tensor.extent(1), 2UZ));
        expect(eq(tensor.size(), 6UZ));

        // Verify row-major interpretation
        expect(eq((tensor[0, 0]), 0));
        expect(eq((tensor[0, 1]), 1));
        expect(eq((tensor[2, 1]), 5));
    };

    "Reshape errors"_test = [] {
        Tensor<int> tensor({2UZ, 3UZ});

        expect(throws([&] { tensor.reshape({2UZ, 4UZ}); }));
        expect(throws([&] { tensor.reshape({7UZ}); }));
    };

    "Multi-dimensional resize"_test = [] {
        Tensor<int> tensor;

        "resize to multi-dimensional"_test = [&] {
            tensor.resize({2UZ, 3UZ, 4UZ}, 42);
            expect(eq(tensor.rank(), 3UZ));
            expect(eq(tensor.size(), 24UZ));
            expect(eq((tensor[0, 0, 0]), 42));
        };

        "change shape entirely"_test = [&] {
            tensor.resize({6UZ, 4UZ});
            expect(eq(tensor.rank(), 2UZ));
            expect(eq(tensor.size(), 24UZ));
        };

        "clear with empty resize"_test = [&] {
            tensor.resize({});
            expect(tensor.empty());
            expect(eq(tensor.rank(), 0UZ));
        };
    };

    "Dimension-specific resize"_test = [] {
        Tensor<int> tensor({3UZ, 4UZ});
        std::iota(tensor.begin(), tensor.end(), 0);

        expect(eq(tensor.extent(1UZ), 4UZ));
        tensor.resize_dim(1UZ, 6UZ);
        expect(eq(tensor.extent(0UZ), 3UZ));
        expect(eq(tensor.extent(1UZ), 6UZ));
        expect(eq(tensor.size(), 18UZ));

        // Invalid dimension
        expect(throws([&] { tensor.resize_dim(5UZ, 10UZ); }));
    };

    "Strides"_test = [] {
        Tensor<int> tensor({3UZ, 4UZ, 2UZ});
        auto        strides = tensor.strides();

        expect(eq(strides.size(), 3UZ));
        expect(eq(strides[0], 8UZ));
        expect(eq(strides[1], 2UZ));
        expect(eq(strides[2], 1UZ));
    };
};

const boost::ut::suite<"Tensor<T> STL Compatibility"> _tensorSTL = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;
    using gr::TensorView;

    "Iterator categories"_test = [] {
        "static Tensor"_test = [] {
            Tensor<int, 3, 4> staticTensor{};

            static_assert(std::random_access_iterator<decltype(staticTensor.begin())>);
            static_assert(std::contiguous_iterator<decltype(staticTensor.begin())>);

            auto it = staticTensor.begin();
            expect(eq(it + 5, staticTensor.begin() + 5));
            expect(eq(std::to_address(it + 5), staticTensor.data() + 5));
        };

        "dynamic Tensor"_test = [] {
            Tensor<int> dynamicTensor(gr::extents_from, {3, 4});

            static_assert(std::random_access_iterator<decltype(dynamicTensor.begin())>);
            static_assert(std::contiguous_iterator<decltype(dynamicTensor.begin())>);

            auto it = dynamicTensor.begin();
            expect(eq(it + 5, dynamicTensor.begin() + 5));
            expect(eq(std::to_address(it + 5), dynamicTensor.data() + 5));
        };

        "TensorView"_test = [] {
            Tensor<int>     dynamicTensor(gr::extents_from, {3, 4});
            TensorView<int> tensorView{dynamicTensor};

            static_assert(std::random_access_iterator<decltype(tensorView.begin())>);
            static_assert(not std::contiguous_iterator<decltype(tensorView.begin())>);

            auto it = tensorView.begin();
            expect((it + 5) == (tensorView.begin() + 5));
            // expect(neq(std::to_address(it + 5), tensorView.data() + 5));
        };
    };

    "STL algorithms"_test = [] {
        Tensor<int> tensor({2UZ, 3UZ});
        std::iota(tensor.begin(), tensor.end(), 1);

        int sum = std::accumulate(tensor.begin(), tensor.end(), 0);
        expect(eq(sum, 21));

        expect(std::ranges::all_of(tensor, [](int x) { return x > 0; }));

        // Row-major order verification
        std::vector<int> expected{1, 2, 3, 4, 5, 6};
        expect(std::ranges::equal(tensor, expected));
    };

    "Data span access"_test = [] {
        Tensor<int> tensor(gr::extents_from, {2, 3});
        std::iota(tensor.begin(), tensor.end(), 0);

        "mutable span"_test = [&] {
            auto span = tensor.data_span();
            expect(eq(span.size(), 6UZ));
            expect(eq(span[0], 0));
            expect(eq(span[5], 5));
        };

        "const span"_test = [&] {
            const auto& const_tensor = tensor;
            auto        const_span   = const_tensor.data_span();
            expect(eq(const_span.size(), 6UZ));
        };
    };
};

const boost::ut::suite<"Tensor<T> Vector Compatibility"> _tensorVector = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "Vector construction"_test = [] {
        std::vector<int> vec{1, 2, 3, 4, 5};

        Tensor<int> tensor(vec);
        expect(eq(tensor.rank(), 1UZ));
        expect(eq(tensor.size(), 5UZ));
        expect(std::ranges::equal(tensor, vec));

        // PMR vector
        std::pmr::vector<int> pmr_vec{10, 20, 30};
        Tensor<int>           tensor2(pmr_vec);
        expect(eq(tensor2.size(), 3UZ));
        expect(eq(tensor2[1], 20));
    };

    "Vector assignment"_test = [] {
        Tensor<double> tensor({2, 2}); // type mismatch: int literals with double T → extents

        expect(throws([&tensor] { tensor = std::vector<double>{1.5, 2.5, 3.5}; })) << "wrong RHS size";
        expect(not throws([&tensor] { tensor = std::vector<double>{1.5, 2.5, 3.5, 4.5}; })) << "correct RHS size";
        expect(eq(tensor.rank(), 2UZ));
        expect(eq(tensor.size(), 4UZ));
        expect(std::ranges::equal(tensor, std::vector<double>{1.5, 2.5, 3.5, 4.5}));
    };

    "Vector conversion"_test = [] {
        Tensor<int> tensor({5UZ});
        std::iota(tensor.begin(), tensor.end(), 1);

        auto vec = static_cast<std::vector<int>>(tensor);
        expect(std::ranges::equal(tensor, vec));

        // Multi-dimensional should throw
        Tensor<int> matrix({2UZ, 3UZ});
        expect(throws([&] { std::ignore = static_cast<std::vector<int>>(matrix); }));
    };

    "Cross-type comparisons"_test = [] {
        std::vector<int> vec{1, 2, 3, 4};
        Tensor<int>      tensor(vec);

        expect(tensor == vec);
        expect(vec == tensor);

        std::vector<int> diff_vec{1, 2, 3};
        expect(tensor != diff_vec);

        // Multi-dimensional vs vector
        Tensor<int> matrix({2UZ, 2UZ});
        std::iota(matrix.begin(), matrix.end(), 1);
        expect(matrix != vec);
    };

    "Vector-like operations"_test = [] {
        Tensor<int> tensor;

        "push_back and emplace_back"_test = [&] {
            tensor.push_back(10);
            tensor.push_back(20);
            tensor.emplace_back(30);

            expect(eq(tensor.size(), 3UZ));
            expect(eq(tensor.rank(), 1UZ));
            expect(eq(tensor.front(), 10));
            expect(eq(tensor.back(), 30));
        };

        "pop_back"_test = [&] {
            tensor.pop_back();
            expect(eq(tensor.size(), 2UZ));
            expect(eq(tensor.back(), 20));
        };
    };

    "Multi-dim to vector conversion"_test = [] {
        Tensor<int> matrix({2UZ, 3UZ});
        std::iota(matrix.begin(), matrix.end(), 0);

        matrix.push_back(100);
        expect(eq(matrix.rank(), 1UZ));
        expect(eq(matrix.size(), 7UZ));
        expect(eq(matrix.back(), 100));
    };
};

const boost::ut::suite<"Tensor<T> Assignment"> _tensorAssignment = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "Static to static same type different size"_test = [] {
        Tensor<int, 2, 2> source{1, 2, 3, 4};
        Tensor<int, 3, 3> target;

        expect(throws([&] { target = source; }));
    };

    "Dynamic to static size check"_test = [] {
        Tensor<double>           source({3UZ, 3UZ}, std::vector{1., 2., 3., 4., 5., 6., 7., 8., 9.});
        Tensor<double, 2UZ, 2UZ> target;

        expect(throws([&] { target = source; }));

        Tensor<double> correct_source({2UZ, 2UZ}, std::vector{1., 2., 3., 4.});
        expect(nothrow([&] { target = correct_source; }));
    };

    "Self assignment safety"_test = [] {
        Tensor<int, 2UZ, 2UZ> static_tensor{1, 2, 3, 4};
        Tensor<int>           dynamic_tensor({2UZ, 2UZ}, std::vector{5, 6, 7, 8});

        auto static_copy  = static_tensor;
        auto dynamic_copy = dynamic_tensor;

// test self-assignment
#if defined(__clang__) or defined(__EMSCRIPTEN__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-assign-overloaded"
#endif
        static_tensor  = static_tensor;
        dynamic_tensor = dynamic_tensor;
#if defined(__clang__) or defined(__EMSCRIPTEN__)
#pragma clang diagnostic pop
#endif

        expect(static_tensor == static_copy);
        expect(dynamic_tensor == dynamic_copy);
    };

    "Chained assignment"_test = [] {
        Tensor<int, 2, 2> source{1, 2, 3, 4};
        Tensor<int, 2, 2> target1, target2, target3;

        target3 = target2 = target1 = source;

        expect(target1 == source);
        expect(target2 == source);
        expect(target3 == source);
    };

    "Value assignment"_test = [] {
        Tensor<int> tensor({2UZ, 3UZ});

        tensor = 99;
        expect(std::ranges::all_of(tensor, [](int x) { return x == 99; }));
    };

    "Assign method"_test = [] {
        "from range"_test = [] {
            Tensor<int>      tensor;
            std::vector<int> data{1, 2, 3, 4};
            tensor.assign(data);
            expect(std::ranges::equal(tensor, data));
        };

        "count + value"_test = [] {
            Tensor<int> tensor;
            tensor.assign(3UZ, 99);
            expect(eq(tensor.size(), 3UZ));
            expect(std::ranges::all_of(tensor, [](int x) { return x == 99; }));
        };
    };
};

const boost::ut::suite<"Tensor<T> Comparisons"> _tensorComparisons = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "Equality operator"_test = [] {
        Tensor<int> A({2UZ, 2UZ});
        Tensor<int> B({2UZ, 2UZ});
        std::iota(A.begin(), A.end(), 0);
        std::iota(B.begin(), B.end(), 0);

        expect(A == B);

        B[0, 0] = 100;
        expect(A != B);

        Tensor<int> C({2UZ, 3UZ});
        expect(A != C);
    };

    "Spaceship operator"_test = [] {
        Tensor<int> A({2UZ, 2UZ});
        Tensor<int> B({2UZ, 2UZ});
        std::iota(A.begin(), A.end(), 0);
        std::iota(B.begin(), B.end(), 0);

        expect((A <=> B) == std::strong_ordering::equal);

        Tensor<int> C({3UZ, 2UZ});
        expect((A <=> C) != std::strong_ordering::equal);

        B[0, 0] = 100;
        expect((A <=> B) != std::strong_ordering::equal);
    };
};

const boost::ut::suite<"Tensor<T> Advanced Features"> _tensorAdvanced = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "PMR support"_test = [] {
        std::array<std::byte, 4096UZ>       buffer;
        std::pmr::monotonic_buffer_resource arena(buffer.data(), buffer.size());

        Tensor<int> tensor({4UZ, 4UZ}, &arena);
        std::iota(tensor.begin(), tensor.end(), 0);

        expect(eq(tensor.size(), 16UZ));
        expect(eq((tensor[3, 3]), 15));
        expect(eq(tensor._data.resource(), &arena));
    };

    "PMR resource propagation"_test = [] {
        std::array<std::byte, 1024>         buffer;
        std::pmr::monotonic_buffer_resource resource(buffer.data(), buffer.size());

        Tensor<int> t1(gr::extents_from, {2, 3}, &resource);

        // Copy constructor with resource
        Tensor<int> t2(t1, &resource);
        expect(eq(t2._data.resource(), &resource));

        // Converting constructor with resource
        Tensor<double> t3(t1, &resource);
        expect(eq(t3._data.resource(), &resource));
    };

    "Swap operations"_test = [] {
        Tensor<int> A({2UZ, 2UZ});
        Tensor<int> B({3UZ, 3UZ});
        std::iota(A.begin(), A.end(), 0);
        std::iota(B.begin(), B.end(), 10);

        auto A_copy = A;
        auto B_copy = B;

        A.swap(B);
        expect(A == B_copy);
        expect(B == A_copy);

        swap(A, B);
        expect(A == A_copy);
        expect(B == B_copy);
    };

    "Fill operation"_test = [] {
        Tensor<int> tensor({2UZ, 3UZ});
        tensor.fill(42);
        expect(std::ranges::all_of(tensor, [](int x) { return x == 42; }));
    };
};

const boost::ut::suite<"Tensor<T> Edge Cases"> _tensorEdgeCases = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "Empty tensor operations"_test = [] {
        Tensor<int> tensor;

        expect(throws([&] { std::ignore = tensor.front(); }));
        expect(throws([&] { std::ignore = tensor.back(); }));
        expect(throws([&] { tensor.pop_back(); }));

        expect(nothrow([&] { tensor.reserve(100UZ); }));
        expect(ge(tensor.capacity(), 100UZ));
    };

    "Single element tensor"_test = [] {
        Tensor<int> tensor({1UZ});
        tensor[0] = 42;

        expect(eq(tensor.size(), 1UZ));
        expect(eq(tensor.front(), 42));
        expect(eq(tensor.back(), 42));
        expect(eq(tensor[0], 42));

        tensor.pop_back();
        expect(tensor.empty());
    };

    "Zero dimensions"_test = [] {
        Tensor<int> tensor({3UZ, 0UZ, 4UZ});
        expect(eq(tensor.size(), 0UZ));
        expect(eq(tensor.rank(), 3UZ));
    };

    "Bool tensor behavior"_test = [] {
        Tensor<bool> tensor(5UZ, true);

        static_assert(std::same_as<decltype(tensor)::value_type, bool>);

        expect(tensor[0]);
        tensor[0] = false;
        expect(!tensor[0]);
    };

    "Mixed operations"_test = [] {
        Tensor<int, 2, 3> static_t{1, 2, 3, 4, 5, 6};

        Tensor<int> dynamic_t(static_t);

        dynamic_t.push_back(7);
        expect(eq(dynamic_t.rank(), 1UZ));
        expect(eq(dynamic_t.size(), 7UZ));
    };

    "Static tensor limitations"_test = [] {
        constexpr Tensor<int, 3, 3> static_tensor{1, 2, 3, 4, 5, 6, 7, 8, 9};

        expect(eq(static_tensor.rank(), 2UZ));
        expect(eq(static_tensor.size(), 9UZ));
        expect(eq(static_tensor.extent(0), 3UZ));

        // These would be compile errors if uncommented:
        // static_tensor.clear();
        // static_tensor.reshape({9});
    };
};

const boost::ut::suite<"Tensor<T> Error Handling"> _tensorErrors = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "Overflow detection"_test = [] {
        const std::size_t big = std::numeric_limits<std::size_t>::max() / 2UZ + 1UZ;
        expect(throws([&] { Tensor<float>({big, 3}); }));
    };

    "Bounds checking"_test = [] {
        Tensor<int> tensor({2UZ, 3UZ});

        expect(throws([&] { std::ignore = tensor.at(2, 0); }));
        expect(throws([&] { std::ignore = tensor.at(0, 3); }));
        expect(throws([&] { std::ignore = tensor.at(0, 0, 0); }));

        std::array<std::size_t, 2UZ> bad_idx{2UZ, 0UZ};
        expect(throws([&] { std::ignore = tensor.at(std::span(bad_idx)); }));
    };

    "Conversion errors"_test = [] {
        Tensor<int> dynamic_source(gr::extents_from, {2, 2, 2});
        expect(throws([&] { Tensor<int, std::dynamic_extent, std::dynamic_extent>{dynamic_source}; }));
    };

    "Extents-data mismatch"_test = [] {
        std::vector<int> data{1, 2, 3, 4, 5, 6};

        expect(throws([&] { Tensor<int>({2, 4}, data); }));
        expect(throws([&] { Tensor<int>({3, 3}, data); }));
        expect(throws([&] { Tensor<int>({}, data); }));
    };

    "PMR resource exhaustion"_test = [] {
        auto* null_resource = std::pmr::null_memory_resource();

        expect(throws([&] { Tensor<int> tensor(gr::extents_from, {10}, null_resource); }));
        expect(throws([&] { Tensor<double> tensor(100, 3.14, null_resource); }));
        expect(throws([&] {
            Tensor<int> tensor(null_resource);
            tensor.push_back(42);
        }));
    };

    "Initializer list size errors"_test = [] {
        expect(throws([] { Tensor<int, 2, 2>{1, 2, 3}; }));
        expect(throws([] { Tensor<int, 2, 2>{1, 2, 3, 4, 5}; }));
    };
};

const boost::ut::suite<"Tensor<T> Nested Initializers"> _tensorNestedInit = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "2D nested initializer lists"_test = [] {
        Tensor<int, 2UZ, 3UZ> tensor{{1, 2, 3}, {4, 5, 6}};

        expect(eq((tensor[0, 0]), 1));
        expect(eq((tensor[1, 0]), 4));
    };

    "3D nested initializer lists"_test = [] {
        Tensor<int, 2UZ, 2UZ, 2UZ> tensor{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}};

        expect(eq((tensor[0, 0, 0]), 1));
        expect(eq((tensor[0, 1, 1]), 4));
        expect(eq((tensor[1, 0, 0]), 5));
        expect(eq((tensor[1, 1, 1]), 8));

        expect(throws([] { Tensor<int, 2, 2, 2>{{{1, 2, 3}, {4, 5, 6}}}; }));
    };
};

const boost::ut::suite<"Tensor<T> CTAD"> _tensorCTAD = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "Basic deduction"_test = [] {
        std::vector<double> vec{1.0, 2.0, 3.0};
        Tensor              tensor1(vec);
        static_assert(std::same_as<decltype(tensor1), Tensor<double>>);

        Tensor tensor2(5UZ, 42);
        static_assert(std::same_as<decltype(tensor2), Tensor<int>>);

        Tensor tensor3(vec.begin(), vec.end());
        static_assert(std::same_as<decltype(tensor3), Tensor<double>>);
    };

    "Tagged deduction"_test = [] {
        std::vector<float> data{1.0f, 2.0f, 3.0f};

        Tensor tensor1(gr::data_from, data);
        static_assert(std::same_as<decltype(tensor1), Tensor<float>>);

        std::vector<std::size_t> extents{3UZ, 4UZ};
        Tensor                   tensor2(gr::extents_from, extents);
        static_assert(std::same_as<decltype(tensor2), Tensor<std::size_t>>);
    };
};

const boost::ut::suite<"Tensor<T> Disambiguation"> _tensorDisambiguation = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "Tagged constructors"_test = [] {
        std::vector<std::size_t> vals{10, 20, 30};

        "extents_from tag"_test = [&] {
            Tensor<std::size_t> tensor1(gr::extents_from, vals);
            expect(eq(tensor1.rank(), 3UZ));
            expect(eq(tensor1.extent(0), 10UZ));
            expect(eq(tensor1.size(), 6000UZ));
        };

        "data_from tag"_test = [&] {
            Tensor<std::size_t> tensor2(gr::data_from, vals);
            expect(eq(tensor2.rank(), 1UZ));
            expect(eq(tensor2.size(), 3UZ));
            expect(eq(tensor2[0], 10UZ));
            expect(eq(tensor2[2], 30UZ));
        };
    };

    "Non-size_t types"_test = [] {
        std::vector<int> data{1, 2, 3, 4};

        Tensor<int> tensor1(data);
        expect(eq(tensor1.rank(), 1UZ));

        Tensor<int> tensor2(gr::data_from, data);
        expect(eq(tensor2.rank(), 1UZ));
        expect(std::ranges::equal(tensor2, data));
    };
};

const boost::ut::suite<"Tensor<T> Memory Management"> _tensorMemory = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "Capacity management"_test = [] {
        Tensor<int> tensor;

        tensor.reserve(1000UZ);
        expect(ge(tensor.capacity(), 1000UZ));

        for (int i = 0; i < 500; ++i) {
            tensor.push_back(i);
        }
        expect(eq(tensor.size(), 500UZ));
        expect(ge(tensor.capacity(), 1000UZ));

        tensor.shrink_to_fit();
        // Note: shrink_to_fit is not guaranteed
    };

    "Move semantics"_test = [] {
        Tensor<int> source({100UZ});
        std::iota(source.begin(), source.end(), 0);
        auto original = std::vector<int>(source.begin(), source.end());

        Tensor<int> moved(std::move(source));
        expect(std::ranges::equal(moved, original));

        Tensor<int> target;
        target = std::move(moved);
        expect(std::ranges::equal(target, original));
    };

    "Move constructor for vector"_test = [] {
        std::pmr::vector<int> vec{1, 2, 3, 4, 5};
        Tensor<int>           tensor(std::move(vec));

        expect(eq(tensor.rank(), 1UZ));
        expect(eq(tensor.size(), 5UZ));
        expect(eq(tensor[0], 1));
    };

    "Allocator mismatch in move"_test = [] {
        std::array<std::byte, 1024UZ>       buffer1{};
        std::array<std::byte, 1024UZ>       buffer2{};
        std::pmr::monotonic_buffer_resource resource1(buffer1.data(), buffer1.size());
        std::pmr::monotonic_buffer_resource resource2(buffer2.data(), buffer2.size());

        std::pmr::vector<int> vec({1, 2, 3, 4}, &resource1);

        Tensor<int> tensor(std::move(vec), &resource2);

        expect(eq(tensor.size(), 4UZ));
        expect(eq(tensor._data.resource(), &resource2));
    };
};

const boost::ut::suite<"Tensor<T> Boundary Cases"> _tensorBoundary = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;

    "Maximum dimensions"_test = [] {
        std::vector<std::size_t> many_dims(gr::detail::kMaxRank, 2);
        Tensor<int>              high_rank_tensor(many_dims);

        expect(eq(high_rank_tensor.rank(), gr::detail::kMaxRank));
        expect(eq(high_rank_tensor.size(), 2UZ << (gr::detail::kMaxRank - 1UZ)));

        constexpr static std::array<std::size_t, gr::detail::kMaxRank> zero_indices{0UZ};
        expect(nothrow([&] { high_rank_tensor.at(std::span<const std::size_t>(zero_indices)) = 42; })) << "at(..) not throwing for maximum rank";
        expect(eq(high_rank_tensor.at(std::span<const std::size_t>(zero_indices)), 42));
    };

    "single element tensor operations"_test = [] {
        Tensor<int> single_elem(gr::extents_from, {1, 1, 1, 1});
        single_elem[0, 0, 0, 0] = 99;

        expect(eq(single_elem.size(), 1UZ));
        expect(eq(single_elem.front(), 99));
        expect(eq(single_elem.back(), 99));

        single_elem.reshape({1});
        expect(eq(single_elem[0], 99));
    };

    "extent edge cases"_test = [] {
        Tensor<int> tensor1(gr::extents_from, {1, 5, 1});
        expect(eq(tensor1.size(), 5UZ));

        if constexpr (sizeof(std::size_t) >= 8) {
            const std::size_t large_size = 1UL << 20;
            Tensor<char>      large_tensor(large_size, 'x');
            expect(eq(large_tensor.size(), large_size));
            expect(eq(large_tensor.front(), 'x'));
            expect(eq(large_tensor.back(), 'x'));
        };
    };

    "Indexing edge cases"_test = [] {
        Tensor<int> tensor(gr::extents_from, {3, 4, 5});
        std::iota(tensor.begin(), tensor.end(), 0);

        expect(eq((tensor[0, 0, 0]), 0));
        expect(eq((tensor[2, 3, 4]), int(tensor.size() - 1)));

        expect(eq((tensor[1, 0, 0]), 20));
        expect(eq((tensor[0, 1, 0]), 5));
        expect(eq((tensor[0, 0, 1]), 1));
    };

    "Zero sized dimensions"_test = [] {
        Tensor<int> tensor(gr::extents_from, {3, 0, 2});
        expect(eq(tensor.size(), 0UZ));
        expect(eq(tensor.rank(), 3UZ));
        expect(tensor.empty());

        Tensor<int> all_zero(gr::extents_from, {0, 0, 0});
        expect(eq(all_zero.size(), 0UZ));

        expect(nothrow([&] { tensor.reshape({0}); }));
        expect(throws([&] { std::ignore = tensor.front(); }));
    };
};

const boost::ut::suite<"TensorView"> _tensorView = [] {
    using namespace boost::ut;
    using namespace boost::ut::literals;
    using gr::Tensor;
    using gr::TensorView;

    "Basic view construction"_test = [] {
        "from tensor"_test = [] {
            Tensor<int> tensor(gr::extents_from, {3, 4});
            std::iota(tensor.begin(), tensor.end(), 0);

            TensorView<int> view(tensor);

            expect(eq(view.rank(), tensor.rank()));
            expect(eq(view.size(), tensor.size()));
            expect(eq(view.data(), tensor.data()));

            // Views share data
            view[0, 0] = 100;
            expect(eq((tensor[0, 0]), 100));
        };

        "from raw pointer"_test = [] {
            std::vector<double> data(12);
            std::iota(data.begin(), data.end(), 1.0);

            TensorView<double> view(data.data(), std::vector{3UZ, 4UZ});

            expect(eq(view.rank(), 2UZ));
            expect(eq(view.size(), 12UZ));
            expect(eq((view[0, 0]), 1.0));
            expect(eq((view[2, 3]), 12.0));
        };

        "with custom strides"_test = [] {
            std::vector<int> data(12);
            std::iota(data.begin(), data.end(), 0);

            // Column-major strides for 3x4 matrix
            TensorView<int> col_major(data.data(), std::vector{3UZ, 4UZ}, std::vector{1UZ, 3UZ});

            // Row-major strides for comparison
            TensorView<int> row_major(data.data(), std::vector{3UZ, 4UZ}, std::vector{4UZ, 1UZ});

            // Different element access patterns
            expect(eq((col_major[0, 1]), 3)); // column-major
            expect(eq((row_major[0, 1]), 1)); // row-major
        };
    };

    "View slicing"_test = [] {
        "basic slice"_test = [] {
            Tensor<int> tensor(gr::extents_from, {4, 5});
            std::iota(tensor.begin(), tensor.end(), 0);

            TensorView<int> view(tensor);

            // Slice rows 1-3, columns 2-4
            auto slice = view.slice({{1, 3}, {2, 4}});

            expect(eq(slice.rank(), 2UZ));
            expect(eq(slice.extent(0), 2UZ));
            expect(eq(slice.extent(1), 2UZ));

            // Original: row 1, col 2 = index 7
            expect(eq((slice[0, 0]), 7));
            // Original: row 2, col 3 = index 13
            expect(eq((slice[1, 1]), 13));
        };

        "partial slice"_test = [] {
            Tensor<int> tensor(gr::extents_from, {3, 4, 5});
            std::iota(tensor.begin(), tensor.end(), 0);

            TensorView<int> view(tensor);

            // Slice only first dimension
            auto slice = view.slice({{1, 2}});

            expect(eq(slice.rank(), 3UZ));
            expect(eq(slice.extent(0), 1UZ));
            expect(eq(slice.extent(1), 4UZ));
            expect(eq(slice.extent(2), 5UZ));
        };
    };

    "View transpose"_test = [] {
        "2D transpose"_test = [] {
            Tensor<int> tensor(gr::extents_from, {3, 4});
            std::iota(tensor.begin(), tensor.end(), 0);

            TensorView<int> view(tensor);
            auto            transposed = view.transpose();

            expect(eq(transposed.rank(), 2UZ));
            expect(eq(transposed.extent(0), 4UZ));
            expect(eq(transposed.extent(1), 3UZ));

            // Original [i,j] = transposed [j,i]
            expect(eq((tensor[0, 1]), (transposed[1, 0])));
            expect(eq((tensor[2, 3]), (transposed[3, 2])));
        };

        "custom axis permutation"_test = [] {
            Tensor<int> tensor(gr::extents_from, {2, 3, 4});
            std::iota(tensor.begin(), tensor.end(), 0);

            TensorView<int> view(tensor);

            // Permute axes: (0,1,2) -> (2,0,1)
            auto permuted = view.transpose({2, 0, 1});

            expect(eq(permuted.extent(0), 4UZ));
            expect(eq(permuted.extent(1), 2UZ));
            expect(eq(permuted.extent(2), 3UZ));
        };
    };

    "View properties"_test = [] {
        "contiguity check"_test = [] {
            Tensor<int>     tensor({3UZ, 4UZ});
            TensorView<int> view(tensor);

            expect(tensor.is_contiguous()) << "Tensor is contiguous";
            expect(view.is_contiguous()) << "TensorView is contiguous";

            auto transposed = view.transpose();
            expect(!transposed.is_contiguous()) << "transposed view is not contiguous";

            auto slice = view.slice({{0, 2}, {1, 3}});
            expect(!slice.is_contiguous()) << "sliced view may not be contiguous";
        };

        "const view"_test = [] {
            const Tensor<int> tensor(gr::extents_from, {2, 3});

            TensorView<const int> const_view{tensor};

            expect(eq(const_view.rank(), 2UZ));
            expect(eq(const_view.size(), 6UZ));

            // Can't modify through const view
            // const_view[0, 0] = 42; // Would not compile
        };
    };

    "View to Tensor conversion"_test = [] {
        "contiguous view"_test = [] {
            Tensor<int> original(gr::extents_from, {3, 3});
            std::iota(original.begin(), original.end(), 1);

            TensorView<int> view(original);
            Tensor<int>     copy = static_cast<Tensor<int>>(view);

            expect(copy == original);

            // Modifying copy doesn't affect original
            copy[0, 0] = 100;
            expect(not eq((original[0, 0]), 100));
        };

        "non-contiguous view"_test = [] {
            Tensor<int> original(gr::extents_from, {3, 3});
            std::iota(original.begin(), original.end(), 1);

            TensorView<int> view(original);
            auto            transposed = view.transpose();

            Tensor<int> copy = static_cast<Tensor<int>>(transposed);

            expect(eq(copy.rank(), 2UZ));
            expect(eq(copy.extent(0), 3UZ));
            expect(eq(copy.extent(1), 3UZ));

            // Check transposed values
            expect(eq((copy[0, 0]), (original[0, 0])));
            expect(eq((copy[0, 1]), (original[1, 0])));
            expect(eq((copy[1, 0]), (original[0, 1])));
        };
    };

    "View iteration"_test = [] {
        "contiguous iteration"_test = [] {
            Tensor<int> tensor(gr::extents_from, {2, 3});
            tensor = {1, 2, 3, 4, 5, 6};

            TensorView<int> view(tensor);

            std::vector<int> values;
            for (auto val : view) {
                values.push_back(val);
            }

            expect(std::ranges::equal(values, std::vector{1, 2, 3, 4, 5, 6}));
        };

        "non-contiguous iteration"_test = [] {
            Tensor<int> tensor(gr::extents_from, {2, 3});
            tensor = {1, 2, 3, 4, 5, 6};

            TensorView<int> view(tensor);
            auto            transposed = view.transpose();

            std::vector<int> values;
            for (auto val : transposed) {
                values.push_back(val);
            }

            // Should iterate in transposed order
            expect(std::ranges::equal(values, std::vector{1, 4, 2, 5, 3, 6}));
        };
    };

    "Static TensorView"_test = [] {
        "compile-time extents"_test = [] {
            Tensor<int, 3, 4> static_tensor;
            std::iota(static_tensor.begin(), static_tensor.end(), 0);

            TensorView<int, 3, 4> static_view(static_tensor);

            expect(eq(static_view.rank(), 2UZ));
            expect(eq((static_view[1, 2]), (static_tensor[1, 2])));
        };
    };

    "Advanced view operations"_test = [] {
        "chained operations"_test = [] {
            Tensor<int> tensor(gr::extents_from, {4, 6});
            std::iota(tensor.begin(), tensor.end(), 0);

            TensorView<int> view(tensor);

            // Slice then transpose
            auto result = view.slice({{1, 3}, {2, 5}}).transpose();

            expect(eq(result.extent(0), 3UZ));
            expect(eq(result.extent(1), 2UZ));
        };

        "view of view"_test = [] {
            Tensor<double> tensor({5, 5});
            std::iota(tensor.begin(), tensor.end(), 0.0);

            TensorView<double> view1(tensor);
            auto               slice1 = view1.slice({{1, 4}, {1, 4}});
            auto               slice2 = slice1.slice({{1, 2}, {1, 2}});

            expect(eq(slice2.extent(0), 1UZ));
            expect(eq(slice2.extent(1), 1UZ));
            expect(eq((slice2[0, 0]), (tensor[2, 2])));
        };
    };
};

int main() { /* tests are statically executed */ }
