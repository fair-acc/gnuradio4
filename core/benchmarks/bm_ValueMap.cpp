#include "benchmark.hpp"

#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/Value.hpp>
#include <gnuradio-4.0/ValueMap.hpp>

#include <array>
#include <atomic>
#include <concepts>
#include <cstdlib>
#include <format>
#include <map>
#include <new>
#include <ranges>
#include <string>
#include <string_view>
#include <unordered_map>

// Global allocation counter for catching std::allocator-routed allocations (i.e. non-PMR maps
// like std::map / std::unordered_map). Activated by setting `g_globalAllocCounter` to non-null
// for the duration of a workload; PMR-routed allocations bypass operator new and are counted
// separately via CountingResource.
//
// NOTE: this benchmark binary overrides global ::operator new / ::operator delete. Do not link
// it against any other TU that also overrides those operators (LD will reject duplicate definitions).
namespace {
struct GlobalAllocCount {
    std::size_t allocs    = 0;
    std::size_t deallocs  = 0;
    std::size_t liveBytes = 0;
};
inline thread_local GlobalAllocCount* g_globalAllocCounter = nullptr;
} // namespace

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
void* operator new(std::size_t n) {
    if (g_globalAllocCounter) {
        ++g_globalAllocCounter->allocs;
        g_globalAllocCounter->liveBytes += n;
    }
    void* p = std::malloc(n);
    if (!p) {
        throw std::bad_alloc{};
    }
    return p;
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void  operator delete(void* p) noexcept {
    if (p && g_globalAllocCounter) {
        ++g_globalAllocCounter->deallocs;
    }
    std::free(p);
}
void operator delete[](void* p) noexcept { ::operator delete(p); }
void operator delete(void* p, std::size_t n) noexcept {
    if (p && g_globalAllocCounter) {
        ++g_globalAllocCounter->deallocs;
        g_globalAllocCounter->liveBytes -= n;
    }
    std::free(p);
}
void operator delete[](void* p, std::size_t n) noexcept { ::operator delete(p, n); }
#pragma GCC diagnostic pop

using gr::pmt::Value;
using gr::pmt::ValueMap;

using StdMap  = std::map<std::string, Value>;
using PmrMap  = std::pmr::map<std::pmr::string, Value>;
using StdUMap = std::unordered_map<std::string, Value>;
using PmrUMap = std::pmr::unordered_map<std::pmr::string, Value>;

struct Sample {
    std::string_view key;
    enum class Kind : std::uint8_t { String, Float, Double, Uint64, Uint32, Bool } kind;
    std::string_view sv;
    double           d;
    std::uint64_t    u;
    bool             b;
};

inline constexpr std::array<Sample, 8> kPayload = {{
    {"signal_name", Sample::Kind::String, "demo_signal_42", 0.0, 0U, false},
    {"signal_unit", Sample::Kind::String, "V", 0.0, 0U, false},
    {"context", Sample::Kind::String, "ctx://demo/run-001", 0.0, 0U, false},
    {"sample_rate", Sample::Kind::Float, "", 48000.0, 0U, false},
    {"frequency", Sample::Kind::Double, "", 1.234e6, 0U, false},
    {"trigger_time", Sample::Kind::Uint64, "", 0.0, 1'700'000'000'000ULL, false},
    {"n_dropped_samples", Sample::Kind::Uint32, "", 0.0, 7U, false},
    {"rx_overflow", Sample::Kind::Bool, "", 0.0, 0U, true},
}};

template<typename Map>
constexpr std::pmr::memory_resource* mapResource(const Map& m) {
    if constexpr (std::same_as<Map, ValueMap>) {
        return m.resource();
    } else if constexpr (requires { m.get_allocator().resource(); }) {
        return m.get_allocator().resource();
    } else {
        return std::pmr::get_default_resource();
    }
}

template<typename Map>
Value makeValue(const Sample& s, std::pmr::memory_resource* res) {
    switch (s.kind) {
    case Sample::Kind::String: return Value{s.sv, res};
    case Sample::Kind::Float: return Value{static_cast<float>(s.d), res};
    case Sample::Kind::Double: return Value{s.d, res};
    case Sample::Kind::Uint64: return Value{s.u, res};
    case Sample::Kind::Uint32: return Value{static_cast<std::uint32_t>(s.u), res};
    case Sample::Kind::Bool: return Value{s.b, res};
    }
    return Value{res};
}

template<typename Map>
void insertSample(Map& m, const Sample& s) {
    if constexpr (std::same_as<Map, ValueMap>) {
        switch (s.kind) {
        case Sample::Kind::String: m.insert_or_assign(s.key, s.sv); break;
        case Sample::Kind::Float: m.insert_or_assign(s.key, static_cast<float>(s.d)); break;
        case Sample::Kind::Double: m.insert_or_assign(s.key, s.d); break;
        case Sample::Kind::Uint64: m.insert_or_assign(s.key, s.u); break;
        case Sample::Kind::Uint32: m.insert_or_assign(s.key, static_cast<std::uint32_t>(s.u)); break;
        case Sample::Kind::Bool: m.insert_or_assign(s.key, s.b); break;
        }
    } else {
        auto* res = mapResource(m);
        using K   = typename Map::key_type;
        if constexpr (std::same_as<K, std::pmr::string>) {
            m.insert_or_assign(K{s.key, res}, makeValue<Map>(s, res));
        } else {
            m.insert_or_assign(K{s.key}, makeValue<Map>(s, res));
        }
    }
}

template<typename Map>
void populate(Map& m, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
        insertSample(m, kPayload[i % kPayload.size()]);
    }
}

template<typename Map>
auto findKey(const Map& m, std::string_view key) {
    if constexpr (std::same_as<Map, ValueMap> || requires { m.find(key); }) {
        return m.find(key);
    } else {
        using K = typename Map::key_type;
        if constexpr (std::same_as<K, std::pmr::string>) {
            return m.find(K{key, mapResource(m)});
        } else {
            return m.find(K{key});
        }
    }
}

template<typename Map>
double iterateExtract(const Map& m) {
    double      acc = 0.0;
    std::size_t len = 0UZ;
    for (const auto& [k, value] : m) {
        if (value.is_string()) {
            len += value.value_or(std::string_view{}).size();
        } else if (value.is_floating_point()) {
            acc += value.template value_or<double>(0.0);
        } else if (value.value_type() == Value::ValueType::Float32) {
            acc += static_cast<double>(value.template value_or<float>(0.0f));
        } else if (value.value_type() == Value::ValueType::UInt64) {
            acc += static_cast<double>(value.template value_or<std::uint64_t>(0));
        } else if (value.value_type() == Value::ValueType::UInt32) {
            acc += static_cast<double>(value.template value_or<std::uint32_t>(0));
        }
        std::ignore = k;
    }
    return acc + static_cast<double>(len);
}

template<typename Map>
Map makeEmpty(std::pmr::memory_resource* res) {
    if constexpr (std::same_as<Map, ValueMap> || requires { Map{res}; }) {
        return Map{res};
    } else {
        return Map{};
    }
}

template<typename Map>
Map buildOther(std::pmr::memory_resource* res, std::size_t n) {
    Map m = makeEmpty<Map>(res);
    for (std::size_t i = 0; i < n; ++i) {
        Sample      s = kPayload[i % kPayload.size()];
        std::string suffixed{s.key};
        suffixed += "_b";
        s.key = suffixed;
        insertSample(m, s);
    }
    return m;
}

inline std::string_view stableName(std::string s) {
    static std::vector<std::string> store;
    return store.emplace_back(std::move(s));
}

inline constexpr std::size_t kReps    = 100;
inline constexpr std::size_t kSamples = 10'000;
inline constexpr std::size_t kN       = 20; // entries per map (representative tag-payload size)

// 4 MiB sized so the monotonic_buffer survives the full per-rep workload — find/erase variants
// re-construct pmr::string keys per call (no SSO bypass for pmr::string) and accumulate kSamples
// allocations before the buffer is reset at the end of the rep.
inline constexpr std::size_t                  kStaticBufBytes = 4UZ * 1024UZ * 1024UZ;
inline std::array<std::byte, kStaticBufBytes> g_valueMapStaticBuf{};

inline const boost::ut::suite _value_map_bench = [] {
    using namespace benchmark;
    auto* res = std::pmr::new_delete_resource();

    constexpr std::array kLabels = {
        std::string_view{"std::map"},
        std::string_view{"std::unordered_map"},
        std::string_view{"std::pmr::map"},
        std::string_view{"std::pmr::unordered_map"},
        std::string_view{"gr::pmt::ValueMap"},
    };

    // Static-buffer scenarios for PMR-enabled containers: each bench iteration constructs a
    // monotonic_buffer_resource over a stack-allocated std::array<byte, kStaticBenchBufBytes>
    // with std::pmr::null_memory_resource() as upstream — guarantees zero heap allocation when
    // the workload fits in the arena (worst-case bounded). Comparison vs the new_delete_resource
    // sweep above isolates the cost of arena-vs-heap allocation from the container's own work.
    // Static-arena PMR labels. Each workload's `[static buf]` row uses the appropriate stack-
    // backed PMR resource for the workload's lifetime profile:
    //   - build-per-iter workloads (insert / erase / merge): a fresh `monotonic_buffer_resource`
    //     per iter (resource lifetime matches the map lifetime — no need to free).
    //   - long-lived workloads (ins+er / copy / find / iter): an `unsynchronized_pool_resource`
    //     over a single `monotonic_buffer_resource` (pool reclaims freed allocations into per-
    //     size-class buckets, enabling steady-state mutation in bounded memory).
    // The reader sees a single `[static buf]` column per workload — the implementation chooses
    // the right resource type so the row pairs apples-to-apples with the heap-PMR row above.
    constexpr std::array kStaticLabels = {
        std::string_view{"std::pmr::map [static buf]"},
        std::string_view{"std::pmr::unordered_map [static buf]"},
        std::string_view{"gr::pmt::ValueMap [static buf]"},
    };

    auto regInsert = [res]<typename Map>(std::string_view label) {
        ::benchmark::benchmark<kReps>(stableName(std::format("insert {}", label)), kSamples) = [res] {
            for (std::size_t i = 0; i < kSamples; ++i) {
                Map m = makeEmpty<Map>(res);
                populate(m, kN);
                force_to_memory(m);
            }
        };
    };

    // Steady-state insert/erase: hold a kN-entry map and replace its keys round-robin (one erase
    // + one insert per inner iter). Stresses payload-pool reuse — without an in-place free-list,
    // the pool grows monotonically until the map is destroyed (current behaviour: shrink_to_fit
    // reclaims; eager reclamation is task #13).
    auto regInsertErase = [res]<typename Map>(std::string_view label) {
        ::benchmark::benchmark<kReps>(stableName(std::format("ins+er {}", label)), kSamples) = [res] {
            Map m = makeEmpty<Map>(res);
            populate(m, kN);
            const auto take = std::min(kN, kPayload.size());
            for (std::size_t i = 0; i < kSamples; ++i) {
                const Sample& s = kPayload[i % take];
                if (auto it = findKey(m, s.key); it != m.end()) {
                    m.erase(it);
                }
                insertSample(m, s);
                force_to_memory(m);
            }
        };
    };

    auto regFind = [res]<typename Map>(std::string_view label) {
        ::benchmark::benchmark<kReps>(stableName(std::format("find   {}", label)), kSamples) = [res] {
            Map m = makeEmpty<Map>(res);
            populate(m, kN);
            const auto take = std::min(kN, kPayload.size());
            for (std::size_t i = 0; i < kSamples; ++i) {
                std::size_t hits = 0;
                for (const auto& s : kPayload | std::views::take(take)) {
                    if (findKey(m, s.key) != m.end()) {
                        ++hits;
                    }
                }
                force_store(hits);
            }
        };
    };

    auto regIter = [res]<typename Map>(std::string_view label) {
        ::benchmark::benchmark<kReps>(stableName(std::format("iter   {}", label)), kSamples) = [res] {
            Map m = makeEmpty<Map>(res);
            populate(m, kN);
            for (std::size_t i = 0; i < kSamples; ++i) {
                auto v = iterateExtract(m);
                force_store(v);
            }
        };
    };

    auto regCopy = [res]<typename Map>(std::string_view label) {
        ::benchmark::benchmark<kReps>(stableName(std::format("copy   {}", label)), kSamples) = [res] {
            Map src = makeEmpty<Map>(res);
            populate(src, kN);
            for (std::size_t i = 0; i < kSamples; ++i) {
                Map copy = src;
                force_to_memory(copy);
            }
        };
    };

    auto regMerge = [res]<typename Map>(std::string_view label) {
        ::benchmark::benchmark<kReps>(stableName(std::format("merge  {}", label)), kSamples) = [res] {
            for (std::size_t i = 0; i < kSamples; ++i) {
                Map a = makeEmpty<Map>(res);
                populate(a, kN);
                Map b = buildOther<Map>(res, kN);
                a.merge(b);
                force_to_memory(a);
            }
        };
    };

    auto regErase = [res]<typename Map>(std::string_view label) {
        ::benchmark::benchmark<kReps>(stableName(std::format("erase  {}", label)), kSamples) = [res] {
            const auto take = std::min(kN, kPayload.size());
            for (std::size_t i = 0; i < kSamples; ++i) {
                Map m = makeEmpty<Map>(res);
                populate(m, kN);
                for (const auto& s : kPayload | std::views::take(take)) {
                    if (auto it = findKey(m, s.key); it != m.end()) {
                        m.erase(it);
                    }
                }
                force_to_memory(m);
            }
        };
    };

    // Static-buffer variants: the per-iteration monotonic_buffer_resource is constructed inside
    // the loop so each iteration starts with a fresh arena (otherwise the buffer would fill up
    // across kSamples iterations and overflow into the null upstream → bad_alloc). Each variant
    // reuses the global g_valueMapStaticBuf array; null upstream guarantees zero heap allocation
    // when the workload fits.
    auto regInsertStatic = []<typename Map>(std::string_view label) {
        ::benchmark::benchmark<kReps>(stableName(std::format("insert {}", label)), kSamples) = [] {
            for (std::size_t i = 0; i < kSamples; ++i) {
                std::pmr::monotonic_buffer_resource buf(g_valueMapStaticBuf.data(), g_valueMapStaticBuf.size(), std::pmr::null_memory_resource());
                Map                                 m = makeEmpty<Map>(&buf);
                populate(m, kN);
                force_to_memory(m);
            }
        };
    };

    auto regFindStatic = []<typename Map>(std::string_view label) {
        ::benchmark::benchmark<kReps>(stableName(std::format("find   {}", label)), kSamples) = [] {
            std::pmr::monotonic_buffer_resource buf(g_valueMapStaticBuf.data(), g_valueMapStaticBuf.size(), std::pmr::null_memory_resource());
            Map                                 m = makeEmpty<Map>(&buf);
            populate(m, kN);
            const auto take = std::min(kN, kPayload.size());
            for (std::size_t i = 0; i < kSamples; ++i) {
                std::size_t hits = 0;
                for (const auto& s : kPayload | std::views::take(take)) {
                    if (findKey(m, s.key) != m.end()) {
                        ++hits;
                    }
                }
                force_store(hits);
            }
        };
    };

    auto regIterStatic = []<typename Map>(std::string_view label) {
        ::benchmark::benchmark<kReps>(stableName(std::format("iter   {}", label)), kSamples) = [] {
            std::pmr::monotonic_buffer_resource buf(g_valueMapStaticBuf.data(), g_valueMapStaticBuf.size(), std::pmr::null_memory_resource());
            Map                                 m = makeEmpty<Map>(&buf);
            populate(m, kN);
            for (std::size_t i = 0; i < kSamples; ++i) {
                auto v = iterateExtract(m);
                force_store(v);
            }
        };
    };

    // Steady-state copy: long-lived src in a pool-over-monotonic; each iter constructs a
    // fresh copy (allocator-aware copy ctor → copy lives in the same pool) which dies at
    // end-of-iter and returns its allocations to the pool's free-list. Mirrors the heap
    // regCopy pattern (populate src ONCE, copy per iter) — apples-to-apples comparison.
    auto regCopyStatic = []<typename Map>(std::string_view label) {
        ::benchmark::benchmark<kReps>(stableName(std::format("copy   {}", label)), kSamples) = [] {
            std::pmr::monotonic_buffer_resource    backing(g_valueMapStaticBuf.data(), g_valueMapStaticBuf.size(), std::pmr::null_memory_resource());
            std::pmr::unsynchronized_pool_resource pool(&backing);
            Map                                    src = makeEmpty<Map>(&pool);
            populate(src, kN);
            for (std::size_t i = 0; i < kSamples; ++i) {
                Map copy = src;
                force_to_memory(copy);
            }
        };
    };

    auto regMergeStatic = []<typename Map>(std::string_view label) {
        ::benchmark::benchmark<kReps>(stableName(std::format("merge  {}", label)), kSamples) = [] {
            for (std::size_t i = 0; i < kSamples; ++i) {
                std::pmr::monotonic_buffer_resource buf(g_valueMapStaticBuf.data(), g_valueMapStaticBuf.size(), std::pmr::null_memory_resource());
                Map                                 a = makeEmpty<Map>(&buf);
                populate(a, kN);
                Map b = buildOther<Map>(&buf, kN);
                a.merge(b);
                force_to_memory(a);
            }
        };
    };

    auto regEraseStatic = []<typename Map>(std::string_view label) {
        ::benchmark::benchmark<kReps>(stableName(std::format("erase  {}", label)), kSamples) = [] {
            const auto take = std::min(kN, kPayload.size());
            for (std::size_t i = 0; i < kSamples; ++i) {
                std::pmr::monotonic_buffer_resource buf(g_valueMapStaticBuf.data(), g_valueMapStaticBuf.size(), std::pmr::null_memory_resource());
                Map                                 m = makeEmpty<Map>(&buf);
                populate(m, kN);
                for (const auto& s : kPayload | std::views::take(take)) {
                    if (auto it = findKey(m, s.key); it != m.end()) {
                        m.erase(it);
                    }
                }
                force_to_memory(m);
            }
        };
    };

    // Steady-state ins+er against a long-lived map allocated from a pool-over-monotonic. This
    // mirrors the heap regInsertErase pattern (populate ONCE, then loop mutations) so the row
    // pairs apples-to-apples with `ins+er <Map>`. Pool reclaims the freed erase region into
    // its size-class buckets; the next insert pops a freed slot — bounded memory, zero malloc.
    auto regInsertEraseStatic = []<typename Map>(std::string_view label) {
        ::benchmark::benchmark<kReps>(stableName(std::format("ins+er {}", label)), kSamples) = [] {
            std::pmr::monotonic_buffer_resource    backing(g_valueMapStaticBuf.data(), g_valueMapStaticBuf.size(), std::pmr::null_memory_resource());
            std::pmr::unsynchronized_pool_resource pool(&backing);
            Map                                    m = makeEmpty<Map>(&pool);
            populate(m, kN);
            const auto take = std::min(kN, kPayload.size());
            for (std::size_t i = 0; i < kSamples; ++i) {
                const Sample& s = kPayload[i % take];
                if (auto it = findKey(m, s.key); it != m.end()) {
                    m.erase(it);
                }
                insertSample(m, s);
                force_to_memory(m);
            }
        };
    };

    auto sweep = [&]<typename Reg>(Reg reg) {
        reg.template operator()<StdMap>(kLabels[0]);
        reg.template operator()<StdUMap>(kLabels[1]);
        reg.template operator()<PmrMap>(kLabels[2]);
        reg.template operator()<PmrUMap>(kLabels[3]);
        reg.template operator()<ValueMap>(kLabels[4]);
    };

    auto sweepStatic = [&]<typename Reg>(Reg reg) {
        reg.template operator()<PmrMap>(kStaticLabels[0]);
        reg.template operator()<PmrUMap>(kStaticLabels[1]);
        reg.template operator()<ValueMap>(kStaticLabels[2]);
        ::benchmark::results::add_separator();
    };

    // Order: insert → erase → ins+er → find → iter → copy → merge. The three insert/erase
    // workloads are grouped first so reviewers can read the trio together: pure insert
    // (cold map, monotonic growth), pure erase (warm map, entry-array compaction + payload
    // free-list push), then steady-state ins+er (free-list pop fits the just-erased region).
    // Each workload prints exactly two rows per container: heap-default and `[static buf]`.
    sweep(regInsert);
    sweepStatic(regInsertStatic);
    sweep(regErase);
    sweepStatic(regEraseStatic);
    sweep(regInsertErase);
    sweepStatic(regInsertEraseStatic);
    sweep(regFind);
    sweepStatic(regFindStatic);
    sweep(regIter);
    sweepStatic(regIterStatic);
    sweep(regCopy);
    sweepStatic(regCopyStatic);
    sweep(regMerge);
    sweepStatic(regMergeStatic);
};

// Allocation-count summary — runs after the framework benchmarks. Each workload runs
// `kAllocReps` times. PMR-based maps' allocations come from a per-test CountingResource;
// non-PMR maps (std::map / std::unordered_map) route through std::allocator → operator new
// → caught by the global thread_local counter (g_globalAllocCounter, top of file). Total =
// PMR-routed + std::allocator-routed (one or the other dominates).
inline constexpr std::size_t kAllocReps = 1'000;

template<typename Map, typename Work>
void measureAllocs(std::string_view workloadName, std::string_view containerLabel, Work&& work) {
    gr::allocator::pmr::CountingResource pmrCtr;
    GlobalAllocCount                     globalCtr;
    g_globalAllocCounter = &globalCtr;
    for (std::size_t i = 0; i < kAllocReps; ++i) {
        work(&pmrCtr);
    }
    g_globalAllocCounter            = nullptr;
    const std::size_t totalAllocs   = pmrCtr.allocCount + globalCtr.allocs;
    const std::size_t totalDeallocs = pmrCtr.deallocCount + globalCtr.deallocs;
    const std::size_t liveBytes     = pmrCtr.liveBytes + globalCtr.liveBytes;
    std::println("| {:6s} {:32s} | {:8.2f} | {:8.2f} | {:10} |", workloadName, containerLabel, static_cast<double>(totalAllocs) / static_cast<double>(kAllocReps), static_cast<double>(totalDeallocs) / static_cast<double>(kAllocReps), liveBytes);
}

// Static-buffer variant: same workloads but the per-iteration resource is a
// monotonic_buffer_resource over the static 64 KiB std::array (null upstream → zero heap
// allocation path; demonstrates pure-compute cost when the user provides a bounded arena).
template<typename Work>
void measureAllocsStatic(std::string_view workloadName, std::string_view containerLabel, Work&& work) {
    GlobalAllocCount globalCtr;
    g_globalAllocCounter = &globalCtr;
    for (std::size_t i = 0; i < kAllocReps; ++i) {
        std::pmr::monotonic_buffer_resource buf(g_valueMapStaticBuf.data(), g_valueMapStaticBuf.size(), std::pmr::null_memory_resource());
        work(&buf);
    }
    g_globalAllocCounter = nullptr;
    std::println("| {:6s} {:32s} | {:8.2f} | {:8.2f} | {:>10s} |", workloadName, containerLabel, static_cast<double>(globalCtr.allocs) / static_cast<double>(kAllocReps), static_cast<double>(globalCtr.deallocs) / static_cast<double>(kAllocReps), "0 (arena)");
}

template<typename Map>
void runAllocCount(std::string_view label) {
    measureAllocs<Map>("insert", label, [](auto* res) {
        Map m = makeEmpty<Map>(res);
        populate(m, kN);
    });
    measureAllocs<Map>("iter  ", label, [](auto* res) {
        Map m = makeEmpty<Map>(res);
        populate(m, kN);
        auto v      = iterateExtract(m);
        std::ignore = v;
    });
    measureAllocs<Map>("copy  ", label, [](auto* res) {
        Map src = makeEmpty<Map>(res);
        populate(src, kN);
        Map copy    = src;
        std::ignore = copy;
    });
    measureAllocs<Map>("merge ", label, [](auto* res) {
        Map a = makeEmpty<Map>(res);
        populate(a, kN);
        Map b = buildOther<Map>(res, kN);
        a.merge(b);
    });
}

template<typename Map>
void runAllocCountStatic(std::string_view label) {
    measureAllocsStatic("insert", label, [](auto* res) {
        Map m = makeEmpty<Map>(res);
        populate(m, kN);
    });
    measureAllocsStatic("iter  ", label, [](auto* res) {
        Map m = makeEmpty<Map>(res);
        populate(m, kN);
        auto v      = iterateExtract(m);
        std::ignore = v;
    });
    measureAllocsStatic("copy  ", label, [](auto* res) {
        Map src = makeEmpty<Map>(res);
        populate(src, kN);
        Map copy    = src;
        std::ignore = copy;
    });
    measureAllocsStatic("merge ", label, [](auto* res) {
        Map a = makeEmpty<Map>(res);
        populate(a, kN);
        Map b = buildOther<Map>(res, kN);
        a.merge(b);
    });
}

int main() {
    using namespace gr::pmt;
    std::println("\nAllocation summary (N={}, kReps={}). PMR maps counted via CountingResource;", kN, kAllocReps);
    std::println("non-PMR maps (std::map / std::unordered_map) counted via global operator new override.");
    std::println("|-----------------------------------------|----------|----------|------------|");
    std::println("| workload + container                    | allocs/  | deallocs/| liveBytes  |");
    std::println("|                                         |   iter   |   iter   |  at end    |");
    std::println("|-----------------------------------------|----------|----------|------------|");
    runAllocCount<StdMap>("std::map");
    runAllocCount<StdUMap>("std::unordered_map");
    runAllocCount<PmrMap>("std::pmr::map");
    runAllocCount<PmrUMap>("std::pmr::unordered_map");
    runAllocCount<ValueMap>("gr::pmt::ValueMap");
    runAllocCountStatic<PmrMap>("std::pmr::map [static buf]");
    runAllocCountStatic<PmrUMap>("std::pmr::unordered_map [static buf]");
    runAllocCountStatic<ValueMap>("gr::pmt::ValueMap [static buf]");
    std::println("|-----------------------------------------|----------|----------|------------|");
}
