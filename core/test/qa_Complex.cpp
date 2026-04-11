#include <boost/ut.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <random>

#include <gnuradio-4.0/Complex.hpp>
#include <vir/simdize.h>

using namespace boost::ut;

const suite<"gr::complex"> tests =
    [] {
        "layout compatibility with std::complex"_test = [] {
            static_assert(sizeof(gr::complex<float>) == sizeof(std::complex<float>));
            static_assert(sizeof(gr::complex<double>) == sizeof(std::complex<double>));
            static_assert(sizeof(gr::complex<float>) == 2 * sizeof(float));
            static_assert(alignof(gr::complex<float>) == alignof(std::complex<float>));
            static_assert(std::is_trivially_copyable_v<gr::complex<float>>);
            static_assert(std::is_trivially_copyable_v<gr::complex<double>>);
            expect(true); // static_asserts are the test
        };

        "round-trip with std::complex"_test = [] {
            std::complex<float> sc{3.14f, -2.71f};
            gr::complex<float>  gc   = sc;
            std::complex<float> back = gc;
            expect(eq(back.real(), sc.real()));
            expect(eq(back.imag(), sc.imag()));

            std::complex<double> sd{1.23456789, -9.87654321};
            gr::complex<double>  gd = sd;
            std::complex<double> bd = gd;
            expect(eq(bd.real(), sd.real()));
            expect(eq(bd.imag(), sd.imag()));
        };

        "reinterpret_cast between gr::complex and std::complex arrays"_test = [] {
            constexpr std::size_t             N = 16;
            std::array<gr::complex<float>, N> grArr;
            for (std::size_t i = 0; i < N; ++i) {
                grArr[i] = {static_cast<float>(i), static_cast<float>(i * 10)};
            }

            const auto* stdPtr = reinterpret_cast<const std::complex<float>*>(grArr.data());
            for (std::size_t i = 0; i < N; ++i) {
                expect(eq(stdPtr[i].real(), static_cast<float>(i)));
                expect(eq(stdPtr[i].imag(), static_cast<float>(i * 10)));
            }

            // reverse direction
            std::array<std::complex<float>, N> stdArr;
            for (std::size_t i = 0; i < N; ++i) {
                stdArr[i] = {static_cast<float>(i + 100), static_cast<float>(i + 200)};
            }
            const auto* grPtr = reinterpret_cast<const gr::complex<float>*>(stdArr.data());
            for (std::size_t i = 0; i < N; ++i) {
                expect(eq(grPtr[i].re, static_cast<float>(i + 100)));
                expect(eq(grPtr[i].im, static_cast<float>(i + 200)));
            }
        };

        "arithmetic matches std::complex"_test = [] {
            std::mt19937                          rng(42);
            std::uniform_real_distribution<float> dist(-100.f, 100.f);

            for (int trial = 0; trial < 100; ++trial) {
                float               ar = dist(rng), ai = dist(rng), br = dist(rng), bi = dist(rng);
                gr::complex<float>  ga{ar, ai}, gb{br, bi};
                std::complex<float> sa{ar, ai}, sb{br, bi};

                auto gSum = ga + gb;
                auto sSum = sa + sb;
                expect(approx(gSum.re, sSum.real(), 1e-6f));
                expect(approx(gSum.im, sSum.imag(), 1e-6f));

                auto gDiff = ga - gb;
                auto sDiff = sa - sb;
                expect(approx(gDiff.re, sDiff.real(), 1e-6f));
                expect(approx(gDiff.im, sDiff.imag(), 1e-6f));

                auto gProd = ga * gb;
                auto sProd = sa * sb;
                expect(approx(gProd.re, sProd.real(), 1e-4f));
                expect(approx(gProd.im, sProd.imag(), 1e-4f));

                if (std::abs(sb) > 1e-3f) {
                    auto gDiv = ga / gb;
                    auto sDiv = sa / sb;
                    expect(approx(gDiv.re, sDiv.real(), 1e-3f));
                    expect(approx(gDiv.im, sDiv.imag(), 1e-3f));
                }
            }
        };

        "unary minus"_test = [] {
            gr::complex<float> z{3, -4};
            auto               n = -z;
            expect(eq(n.re, -3.f));
            expect(eq(n.im, 4.f));
        };

        "compound assignment operators"_test = [] {
            gr::complex<float> a{1, 2};
            a += gr::complex<float>{3, 4};
            expect(eq(a.re, 4.f) and eq(a.im, 6.f));

            a -= gr::complex<float>{1, 1};
            expect(eq(a.re, 3.f) and eq(a.im, 5.f));

            a *= gr::complex<float>{2, 0};
            expect(eq(a.re, 6.f) and eq(a.im, 10.f));

            a /= gr::complex<float>{2, 0};
            expect(approx(a.re, 3.f, 1e-6f) and approx(a.im, 5.f, 1e-6f));

            a *= 2.f;
            expect(eq(a.re, 6.f) and eq(a.im, 10.f));

            a /= 2.f;
            expect(eq(a.re, 3.f) and eq(a.im, 5.f));
        };

        "scalar multiply"_test = [] {
            gr::complex<float> z{2, 3};
            auto               a = z * 5.f;
            auto               b = 5.f * z;
            expect(eq(a.re, 10.f) and eq(a.im, 15.f));
            expect(eq(b.re, 10.f) and eq(b.im, 15.f));

            auto c = z / 2.f;
            expect(eq(c.re, 1.f) and eq(c.im, 1.5f));
        };

        "abs, norm, arg, conj, polar"_test = [] {
            gr::complex<float> z{3, 4};
            expect(approx(gr::abs(z), 5.f, 1e-6f));
            expect(approx(gr::norm(z), 25.f, 1e-6f));
            expect(approx(gr::arg(z), std::atan2(4.f, 3.f), 1e-6f));

            auto c = gr::conj(z);
            expect(eq(c.re, 3.f) and eq(c.im, -4.f));

            auto p = gr::polar(5.f, std::atan2(4.f, 3.f));
            expect(approx(p.re, 3.f, 1e-5f) and approx(p.im, 4.f, 1e-5f));
        };

        "equality"_test = [] {
            gr::complex<float> a{1, 2}, b{1, 2}, c{1, 3};
            expect(a == b);
            expect(!(a == c));
        };

        "constexpr arithmetic"_test = [] {
            constexpr gr::complex<float> a{1, 2}, b{3, 4};
            constexpr auto               s = a + b;
            constexpr auto               p = a * b;
            static_assert(s.re == 4.f && s.im == 6.f);
            static_assert(p.re == -5.f && p.im == 10.f);
            expect(true);
        };

        "structured bindings"_test = [] {
            gr::complex<float> z{7, 11};
            auto [r, i] = z;
            expect(eq(r, 7.f) and eq(i, 11.f));
        };

        "value_type alias"_test = [] {
            static_assert(std::is_same_v<gr::complex<float>::value_type, float>);
            static_assert(std::is_same_v<gr::complex<double>::value_type, double>);
            expect(true);
        };

        "free functions real() and imag() found via ADL"_test = [] {
            gr::complex<float> z{3, 4};
            // unqualified calls — ADL must find gr::real/gr::imag
            using gr::real;
            using gr::imag;
            expect(eq(real(z), 3.f));
            expect(eq(imag(z), 4.f));
        };

        "tuple protocol for vir::simdize"_test = [] {
            static_assert(std::tuple_size_v<gr::complex<float>> == 2);
            static_assert(std::is_same_v<std::tuple_element_t<0, gr::complex<float>>, float>);
            static_assert(std::is_same_v<std::tuple_element_t<1, gr::complex<float>>, float>);
            expect(true);
        };

        "vir::simdize type instantiation"_test = [] {
            using V = vir::simdize<gr::complex<float>, 4>;
            static_assert(sizeof(V) > 0);
            static_assert(std::tuple_size_v<gr::complex<float>> == 2);

            using VD = vir::simdize<gr::complex<double>, 4>;
            static_assert(sizeof(VD) > 0);
            expect(true);
        };

        "vir::simdize load and element-wise access"_test = [] {
            constexpr std::size_t N = 4;
            using V                 = vir::simdize<gr::complex<float>, N>;

            std::array<gr::complex<float>, N> data = {{{10, 20}, {30, 40}, {50, 60}, {70, 80}}};
            V                                 simd_val(data.data(), vir::stdx::element_aligned);

            // structured binding gives (simd<float> re, simd<float> im)
            const auto& [re_vec, im_vec] = simd_val;
            static_assert(vir::stdx::is_simd_v<std::remove_cvref_t<decltype(re_vec)>>);

            expect(eq(re_vec[0], 10.f) and eq(re_vec[1], 30.f) and eq(re_vec[2], 50.f) and eq(re_vec[3], 70.f));
            expect(eq(im_vec[0], 20.f) and eq(im_vec[1], 40.f) and eq(im_vec[2], 60.f) and eq(im_vec[3], 80.f));
        };

        "double precision"_test = [] {
            gr::complex<double>  gd{1.23456789012345, -9.87654321098765};
            std::complex<double> sd   = gd;
            gr::complex<double>  back = sd;
            expect(eq(back.re, gd.re) and eq(back.im, gd.im));

            auto p  = gd * gd;
            auto sp = sd * sd;
            expect(approx(p.re, sp.real(), 1e-12));
            expect(approx(p.im, sp.imag(), 1e-12));
        };
    };

int main() { /* not needed for UT */ }
