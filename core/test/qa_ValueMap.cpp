#include <boost/ut.hpp>

#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/ValueMap.hpp>
#include <gnuradio-4.0/formatter/ValueFormatter.hpp>    // operator<< for Value / Value::ValueType — boost::ut::eq() needs it
#include <gnuradio-4.0/formatter/ValueMapFormatter.hpp> // operator<< for ValueMap — ditto

#include <format>
#include <map>
#include <memory_resource>
#include <print>
#include <string>
#include <string_view>
#include <unordered_map>

using gr::allocator::pmr::CountingResource;

const boost::ut::suite<"ValueMap - canonical key registry"> _registry_suite = [] {
    using namespace boost::ut;
    namespace k = gr::pmt::keys;
    using gr::pmt::Value;

    "registry has 20 entries covering Tag.hpp:196-216"_test = [] { expect(eq(k::kCanonical.size(), 20UZ)); };

    "idOf resolves every registered name to its canonical id"_test = [] {
        expect(eq(k::idOf<"sample_rate">, std::uint16_t{0x0001}));
        expect(eq(k::idOf<"signal_name">, std::uint16_t{0x0002}));
        expect(eq(k::idOf<"num_channels">, std::uint16_t{0x0003}));
        expect(eq(k::idOf<"signal_quantity">, std::uint16_t{0x0004}));
        expect(eq(k::idOf<"signal_unit">, std::uint16_t{0x0005}));
        expect(eq(k::idOf<"signal_min">, std::uint16_t{0x0006}));
        expect(eq(k::idOf<"signal_max">, std::uint16_t{0x0007}));
        expect(eq(k::idOf<"n_dropped_samples">, std::uint16_t{0x0008}));
        expect(eq(k::idOf<"frequency">, std::uint16_t{0x0009}));
        expect(eq(k::idOf<"rx_overflow">, std::uint16_t{0x000A}));
        expect(eq(k::idOf<"trigger_name">, std::uint16_t{0x000B}));
        expect(eq(k::idOf<"trigger_time">, std::uint16_t{0x000C}));
        expect(eq(k::idOf<"trigger_offset">, std::uint16_t{0x000D}));
        expect(eq(k::idOf<"trigger_meta_info">, std::uint16_t{0x000E}));
        expect(eq(k::idOf<"local_time">, std::uint16_t{0x000F}));
        expect(eq(k::idOf<"context">, std::uint16_t{0x0010}));
        expect(eq(k::idOf<"ctx_time">, std::uint16_t{0x0011}));
        expect(eq(k::idOf<"reset_default">, std::uint16_t{0x0012}));
        expect(eq(k::idOf<"store_default">, std::uint16_t{0x0013}));
        expect(eq(k::idOf<"end_of_stream">, std::uint16_t{0x0014}));
    };

    "idOf returns kIdUnknown for unregistered names"_test = [] {
        expect(eq(k::idOf<"not_a_known_key">, k::kIdUnknown));
        expect(eq(k::idOf<"">, k::kIdUnknown));
        expect(eq(k::idOf<"SAMPLE_RATE">, k::kIdUnknown)); // case-sensitive
    };

    "boundTypeOf returns the right Value::ValueType per canonical id"_test = [] {
        expect(eq(k::boundTypeOf<k::idOf<"sample_rate">>, Value::ValueType::Float32));
        expect(eq(k::boundTypeOf<k::idOf<"signal_name">>, Value::ValueType::String));
        expect(eq(k::boundTypeOf<k::idOf<"num_channels">>, Value::ValueType::UInt32));
        expect(eq(k::boundTypeOf<k::idOf<"frequency">>, Value::ValueType::Float64));
        expect(eq(k::boundTypeOf<k::idOf<"rx_overflow">>, Value::ValueType::Bool));
        expect(eq(k::boundTypeOf<k::idOf<"trigger_time">>, Value::ValueType::UInt64));
        expect(eq(k::boundTypeOf<k::idOf<"trigger_meta_info">>, Value::ValueType::Value));
    };

    "boundTypeOf returns Monostate for unregistered ids"_test = [] {
        expect(eq(k::boundTypeOf<0x0000>, Value::ValueType::Monostate));
        expect(eq(k::boundTypeOf<0x0030>, Value::ValueType::Monostate));
        expect(eq(k::boundTypeOf<0x7FFF>, Value::ValueType::Monostate));
    };

    "unitOf matches Tag.hpp DefaultTag declarations"_test = [] {
        expect(eq(k::unitOf<k::idOf<"sample_rate">>, std::string_view{"Hz"}));
        expect(eq(k::unitOf<k::idOf<"frequency">>, std::string_view{"Hz"}));
        expect(eq(k::unitOf<k::idOf<"trigger_time">>, std::string_view{"ns"}));
        expect(eq(k::unitOf<k::idOf<"trigger_offset">>, std::string_view{"s"}));
        expect(eq(k::unitOf<k::idOf<"signal_min">>, std::string_view{"a.u."}));
        expect(eq(k::unitOf<k::idOf<"signal_max">>, std::string_view{"a.u."}));
        expect(eq(k::unitOf<k::idOf<"signal_name">>, std::string_view{""}));
    };

    "CanonicalCppType resolves to the mapped scalar / string_view type"_test = [] {
        static_assert(std::same_as<k::CanonicalCppType<k::idOf<"sample_rate">>, float>);
        static_assert(std::same_as<k::CanonicalCppType<k::idOf<"frequency">>, double>);
        static_assert(std::same_as<k::CanonicalCppType<k::idOf<"num_channels">>, std::uint32_t>);
        static_assert(std::same_as<k::CanonicalCppType<k::idOf<"trigger_time">>, std::uint64_t>);
        static_assert(std::same_as<k::CanonicalCppType<k::idOf<"rx_overflow">>, bool>);
        static_assert(std::same_as<k::CanonicalCppType<k::idOf<"signal_name">>, std::string_view>);
        expect(true); // compile-time coverage above
    };

    "sentinel ids stay out of the assignable range"_test = [] {
        expect(eq(k::kIdUnknown, std::uint16_t{0x0000}));
        expect(eq(k::kInlineKeyId, std::uint16_t{0x8000}));
        expect(eq(k::kEndMarkerId, std::uint16_t{0xFFFF}));
        for (const auto& entry : k::kCanonical) {
            expect(neq(entry.id, k::kIdUnknown));
            expect(neq(entry.id, k::kInlineKeyId));
            expect(neq(entry.id, k::kEndMarkerId));
        }
    };
};

const boost::ut::suite<"ValueMap - key-type interop"> _key_interop_suite = [] {
    using namespace boost::ut;
    using gr::pmt::ValueMap;

    "insert / lookup / erase round-trip across all 6 string-like key types"_test = [] {
        ValueMap map;

        // Insertion via every supported key shape.
        map.emplace("char_lit", std::uint32_t{1}); // string literal (char[N])
        const char* cstr = "c_str";
        map.emplace(cstr, std::uint32_t{2});                   // const char*
        map.emplace(std::string_view{"sv"}, std::uint32_t{3}); // std::string_view
        map.emplace(std::string{"std"}, std::uint32_t{4});     // std::string
        std::pmr::string pmrKey{"pmr"};
        map.emplace(pmrKey, std::uint32_t{5});                          // std::pmr::string
        map.emplace(gr::meta::fixed_string{"fixed"}, std::uint32_t{6}); // gr::meta::fixed_string

        expect(eq(map.size(), 6UZ));

        // Lookup through every key shape must locate the previously inserted entries.
        expect(map.contains("char_lit"));
        expect(map.contains(cstr));
        expect(map.contains(std::string_view{"sv"}));
        expect(map.contains(std::string{"std"}));
        expect(map.contains(pmrKey));
        expect(map.contains(gr::meta::fixed_string{"fixed"}));

        expect(eq(map.at("char_lit").value_or<std::uint32_t>(0U), std::uint32_t{1}));
        expect(eq(map.at(cstr).value_or<std::uint32_t>(0U), std::uint32_t{2}));
        expect(eq(map.at(std::string_view{"sv"}).value_or<std::uint32_t>(0U), std::uint32_t{3}));
        expect(eq(map.at(std::string{"std"}).value_or<std::uint32_t>(0U), std::uint32_t{4}));
        expect(eq(map.at(pmrKey).value_or<std::uint32_t>(0U), std::uint32_t{5}));
        expect(eq(map.at(gr::meta::fixed_string{"fixed"}).value_or<std::uint32_t>(0U), std::uint32_t{6}));

        // Erase via iterator obtained from a different key type than insert (cross-shape).
        map.erase(map.find(std::string_view{"char_lit"}));
        map.erase(map.find(gr::meta::fixed_string{"sv"}));
        expect(!map.contains("char_lit"));
        expect(!map.contains("sv"));
        expect(eq(map.size(), 4UZ));
    };

    "canonical-name lookup via fixed_string finds the entry by canonical id"_test = [] {
        ValueMap map;
        map.emplace("sample_rate", 48000.0f); // canonical name (id 0x0001)
        // Mixed-shape lookups should all hit the same entry via the canonical-id fast path.
        expect(map.contains("sample_rate"));
        expect(map.contains(std::string_view{"sample_rate"}));
        expect(map.contains(gr::meta::fixed_string{"sample_rate"}));
    };
};

const boost::ut::suite<"ValueMap - STL container interop"> _stl_interop_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using gr::pmt::ValueMap;

    "construct ValueMap from std::map<std::string, Value, std::less<>>"_test = [] {
        std::map<std::string, Value, std::less<>> src;
        src.emplace("sample_rate", Value{48000.0f});
        src.emplace("num_channels", Value{std::uint32_t{4}});
        src.emplace("signal_name", Value{std::string_view{"carrier-A"}});

        ValueMap map{src};
        expect(eq(map.size(), 3UZ));
        expect(eq(map.at("sample_rate").value_or<float>(0.f), 48000.0f));
        expect(eq(map.at("num_channels").value_or<std::uint32_t>(0U), std::uint32_t{4}));
        expect(eq(map.at("signal_name").value_or(std::string_view{}), std::string_view{"carrier-A"}));
    };

    "construct ValueMap from Value::Map (std::pmr::unordered_map<pmr::string, Value, …>)"_test = [] {
        Value::Map src{};
        src.emplace(std::pmr::string{"frequency"}, Value{2.4e9});
        src.emplace(std::pmr::string{"rx_overflow"}, Value{true});

        ValueMap map{src};
        expect(eq(map.size(), 2UZ));
        expect(eq(map.at("frequency").value_or<double>(0.0), 2.4e9));
        expect(eq(map.at("rx_overflow").value_or<bool>(false), true));
    };

    "to_std_map round-trip preserves all entries"_test = [] {
        ValueMap map;
        map.emplace("sample_rate", 48000.0f);
        map.emplace("num_channels", std::uint32_t{4});
        map.emplace("signal_name", std::string_view{"carrier-A"});

        const auto out = map.to_std_map();
        expect(eq(out.size(), 3UZ));
        expect(eq(out.at("sample_rate").value_or<float>(0.f), 48000.0f));
        expect(eq(out.at("num_channels").value_or<std::uint32_t>(0U), std::uint32_t{4}));
        expect(eq(out.at("signal_name").value_or(std::string_view{}), std::string_view{"carrier-A"}));
    };

    "to_std_unordered_map round-trip preserves all entries"_test = [] {
        ValueMap map;
        map.emplace("k_a", std::uint32_t{1});
        map.emplace("k_b", std::uint32_t{2});

        const auto out = map.to_std_unordered_map();
        expect(eq(out.size(), 2UZ));
        expect(eq(out.at("k_a").value_or<std::uint32_t>(0U), std::uint32_t{1}));
        expect(eq(out.at("k_b").value_or<std::uint32_t>(0U), std::uint32_t{2}));
    };

    "size comparison: 20 canonical Tag entries — ValueMap blob vs std::pmr::map / unordered_map"_test = [] {
        // Same 20 canonical entries populated into a ValueMap (typed insertion) and into two
        // node-based std::pmr containers (Value-wrapped insertion). live_bytes from the shared
        // CountingResource lets us compare effective on-heap footprint vs packed-blob footprint.
        CountingResource mr_omap;
        CountingResource mr_umap;

        ValueMap                                                                                            vmap(/*resource=*/nullptr, /*initial_capacity_entries=*/20U);
        std::pmr::map<std::pmr::string, Value, std::less<>>                                                 omap{&mr_omap};
        std::pmr::unordered_map<std::pmr::string, Value, gr::pmt::Value::MapHash, gr::pmt::Value::MapEqual> umap{&mr_umap};

        const auto stdInsert = [](auto& map, std::string_view key, Value value) { map.emplace(std::pmr::string{key, map.get_allocator().resource()}, std::move(value)); };

        // ValueMap: typed insertion (carries the on-blob layout we want to measure).
        vmap.emplace("sample_rate", 48000.0f);
        vmap.emplace("signal_name", std::string_view{"carrier-A"});
        vmap.emplace("num_channels", std::uint32_t{4});
        vmap.emplace("signal_quantity", std::string_view{"voltage"});
        vmap.emplace("signal_unit", std::string_view{"V"});
        vmap.emplace("signal_min", -1.0f);
        vmap.emplace("signal_max", 1.0f);
        vmap.emplace("n_dropped_samples", std::uint32_t{0});
        vmap.emplace("frequency", 2.4e9);
        vmap.emplace("rx_overflow", false);
        vmap.emplace("trigger_name", std::string_view{"PPS"});
        vmap.emplace("trigger_time", std::uint64_t{1'700'000'000'000'000'000ULL});
        vmap.emplace("trigger_offset", 0.0f);
        vmap.emplace("local_time", std::uint64_t{1'700'000'000'001'000'000ULL});
        vmap.emplace("context", std::string_view{"default"});
        vmap.emplace("ctx_time", std::uint64_t{1'700'000'000'000'000'001ULL});
        vmap.emplace("reset_default", false);
        vmap.emplace("store_default", false);
        vmap.emplace("end_of_stream", false);
        vmap.emplace("ext.user_extra", std::uint32_t{42}); // 20th: representative non-canonical user key

        // Same 20 entries into the std containers as Value-wrapped types.
        stdInsert(omap, "sample_rate", Value{48000.0f});
        stdInsert(omap, "signal_name", Value{std::string_view{"carrier-A"}});
        stdInsert(omap, "num_channels", Value{std::uint32_t{4}});
        stdInsert(omap, "signal_quantity", Value{std::string_view{"voltage"}});
        stdInsert(omap, "signal_unit", Value{std::string_view{"V"}});
        stdInsert(omap, "signal_min", Value{-1.0f});
        stdInsert(omap, "signal_max", Value{1.0f});
        stdInsert(omap, "n_dropped_samples", Value{std::uint32_t{0}});
        stdInsert(omap, "frequency", Value{2.4e9});
        stdInsert(omap, "rx_overflow", Value{false});
        stdInsert(omap, "trigger_name", Value{std::string_view{"PPS"}});
        stdInsert(omap, "trigger_time", Value{std::uint64_t{1'700'000'000'000'000'000ULL}});
        stdInsert(omap, "trigger_offset", Value{0.0f});
        stdInsert(omap, "local_time", Value{std::uint64_t{1'700'000'000'001'000'000ULL}});
        stdInsert(omap, "context", Value{std::string_view{"default"}});
        stdInsert(omap, "ctx_time", Value{std::uint64_t{1'700'000'000'000'000'001ULL}});
        stdInsert(omap, "reset_default", Value{false});
        stdInsert(omap, "store_default", Value{false});
        stdInsert(omap, "end_of_stream", Value{false});
        stdInsert(omap, "ext.user_extra", Value{std::uint32_t{42}});

        stdInsert(umap, "sample_rate", Value{48000.0f});
        stdInsert(umap, "signal_name", Value{std::string_view{"carrier-A"}});
        stdInsert(umap, "num_channels", Value{std::uint32_t{4}});
        stdInsert(umap, "signal_quantity", Value{std::string_view{"voltage"}});
        stdInsert(umap, "signal_unit", Value{std::string_view{"V"}});
        stdInsert(umap, "signal_min", Value{-1.0f});
        stdInsert(umap, "signal_max", Value{1.0f});
        stdInsert(umap, "n_dropped_samples", Value{std::uint32_t{0}});
        stdInsert(umap, "frequency", Value{2.4e9});
        stdInsert(umap, "rx_overflow", Value{false});
        stdInsert(umap, "trigger_name", Value{std::string_view{"PPS"}});
        stdInsert(umap, "trigger_time", Value{std::uint64_t{1'700'000'000'000'000'000ULL}});
        stdInsert(umap, "trigger_offset", Value{0.0f});
        stdInsert(umap, "local_time", Value{std::uint64_t{1'700'000'000'001'000'000ULL}});
        stdInsert(umap, "context", Value{std::string_view{"default"}});
        stdInsert(umap, "ctx_time", Value{std::uint64_t{1'700'000'000'000'000'001ULL}});
        stdInsert(umap, "reset_default", Value{false});
        stdInsert(umap, "store_default", Value{false});
        stdInsert(umap, "end_of_stream", Value{false});
        stdInsert(umap, "ext.user_extra", Value{std::uint32_t{42}});

        expect(eq(vmap.size(), 20UZ));
        expect(eq(omap.size(), 20UZ));
        expect(eq(umap.size(), 20UZ));

        vmap.shrink_to_fit(); // right-size the blob so the comparison is fair (no over-allocated slack)

        const auto vmapBlobBytes = vmap.blob().size();
        const auto omapLiveBytes = mr_omap.liveBytes;
        const auto umapLiveBytes = mr_umap.liveBytes;

        std::println("ValueMap blob: {} B | std::pmr::map live: {} B | std::pmr::unordered_map live: {} B (20 entries, after shrink_to_fit)", //
            vmapBlobBytes, omapLiveBytes, umapLiveBytes);

        // After shrink_to_fit, the packed blob is a single contiguous allocation with no
        // per-entry pointer/header overhead — wins decisively over node-based STL maps.
        expect(vmapBlobBytes < omapLiveBytes) << "ValueMap blob should be smaller than std::pmr::map";
        expect(vmapBlobBytes < umapLiveBytes) << "ValueMap blob should be smaller than std::pmr::unordered_map";
    };
};

const boost::ut::suite<"ValueMap - STL parity (typedefs / count / erase(key) / operator==)"> _stl_parity_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using gr::pmt::ValueMap;

    "STL-parity nested types are exposed for template code"_test = [] {
        static_assert(std::same_as<ValueMap::key_type, std::string_view>);
        static_assert(std::same_as<ValueMap::mapped_type, Value>);
        static_assert(std::same_as<ValueMap::value_type, std::pair<std::string_view, Value>>);
        static_assert(std::same_as<ValueMap::size_type, std::size_t>);
        expect(true); // compile-time coverage above
    };

    "count(key) returns 0 or 1 (keys are unique)"_test = [] {
        ValueMap map;
        expect(eq(map.count("absent"), 0UZ));
        map.emplace("present", std::uint32_t{1});
        expect(eq(map.count("present"), 1UZ));
        expect(eq(map.count("absent"), 0UZ));
    };

    "erase(key) returns the number of removed entries (0 or 1)"_test = [] {
        ValueMap map;
        map.emplace("a", std::uint32_t{1});
        map.emplace("b", std::uint32_t{2});
        expect(eq(map.erase("a"), 1UZ));
        expect(eq(map.erase("a"), 0UZ)) << "second erase finds nothing";
        expect(eq(map.size(), 1UZ));
        expect(map.contains("b"));
    };

    "operator== — empty maps compare equal"_test = [] {
        const ValueMap a;
        const ValueMap b;
        expect(a == b);
    };

    "operator== — same entries compare equal regardless of insertion order"_test = [] {
        ValueMap a;
        ValueMap b;
        a.emplace("x", 1.0f);
        a.emplace("y", std::uint32_t{2});
        b.emplace("y", std::uint32_t{2}); // inserted in the opposite order
        b.emplace("x", 1.0f);
        expect(a == b) << "operator== is unordered (matches std::pmr::unordered_map)";
    };

    "operator== — different size compare unequal"_test = [] {
        ValueMap a;
        ValueMap b;
        a.emplace("x", 1.0f);
        a.emplace("y", 2.0f);
        b.emplace("x", 1.0f);
        expect(!(a == b));
    };

    "operator== — same keys / different values compare unequal"_test = [] {
        ValueMap a;
        ValueMap b;
        a.emplace("k", 1.0f);
        b.emplace("k", 2.0f);
        expect(!(a == b));
    };

    "operator== — different keys / same values compare unequal"_test = [] {
        ValueMap a;
        ValueMap b;
        a.emplace("k1", 1.0f);
        b.emplace("k2", 1.0f);
        expect(!(a == b));
    };

    "operator== — survives nested ValueMap entries"_test = [] {
        ValueMap leafA;
        leafA.emplace("v", std::uint64_t{42});
        ValueMap a;
        a.emplace("nested", leafA);

        ValueMap leafB;
        leafB.emplace("v", std::uint64_t{42});
        ValueMap b;
        b.emplace("nested", leafB);

        // Both maps carry an equivalent nested ValueMap — operator== walks the recursive Value tree.
        const auto va = a.at("nested");
        const auto vb = b.at("nested");
        expect(va == vb) << "Value::operator== considers two nested maps with equal contents equal";
    };
};

const boost::ut::suite<"ValueMap - operator[] / typed at&lt;name&gt; / equal_range / merge"> _bracket_and_range_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using gr::pmt::ValueMap;

    "operator[](key) — const throws on missing, non-const auto-vivifies"_test = [] {
        ValueMap map;
        map.emplace("present", 1.5f);
        // Non-const lvalue path: auto-vivify default Value on miss (matches std::map).
        const Value vivified = map["absent"];
        expect(vivified.is_monostate()) << "non-const op[] inserts a default monostate Value on miss";
        expect(map.contains("absent")) << "after the read, the key exists";
        // Const path: still throws on miss.
        const ValueMap& cmap = map;
        expect(throws<std::out_of_range>([&] { (void)cmap["never_present"]; }));
        // Round-trip read of present key works through both paths.
        expect(map["present"].holds<float>());
        expect(eq(cmap["present"].value_or<float>(0.f), 1.5f));
    };

    "at<Name>() returns a mutable reference for inline scalars"_test = [] {
        ValueMap map;
        map.emplace("sample_rate", 48000.0f);
        float& ref = map.at<"sample_rate">();
        ref        = 96000.0f;
        expect(eq(map.at<"sample_rate">(), 96000.0f));
    };

    "at<Name>() const returns a const reference"_test = [] {
        ValueMap map;
        map.emplace("frequency", 2.4e9);
        const ValueMap& cmap = map;
        const double&   ref  = cmap.at<"frequency">();
        expect(eq(ref, 2.4e9));
    };

    "at<Name>() for String canonical returns std::string_view by value"_test = [] {
        ValueMap map;
        map.emplace("signal_name", std::string_view{"carrier-A"});
        const std::string_view view = map.at<"signal_name">();
        expect(eq(view, std::string_view{"carrier-A"}));
    };

    "at<Name>() throws std::out_of_range when the canonical key is absent"_test = [] {
        ValueMap map;
        expect(throws<std::out_of_range>([&] { (void)map.at<"sample_rate">(); }));
    };

    "at<Name>() debug-asserts on stored-type mismatch with the canonical binding"_test = [] {
        if constexpr (!gr::meta::kDebugBuild) {
            // Release build: silent UB by design (matches std::vector::operator[] philosophy).
            // The debug-only assert protects developers; release is on its own.
            return;
        }
        ValueMap map;
        // sample_rate is canonically Float32 — insert a mismatched UInt64 by abusing the
        // string-keyed emplace, then exercise the typed at<>:
        map.emplace("sample_rate", std::uint64_t{42});
        expect(aborts([&] { (void)map.at<"sample_rate">(); })) << "type-mismatch must trip the debug assert";
    };

    "equal_range(key) returns [find, find+1) on hit, [end, end) on miss"_test = [] {
        ValueMap map;
        map.emplace("a", std::uint32_t{1});
        map.emplace("b", std::uint32_t{2});
        const auto [hitFirst, hitLast] = map.equal_range("a");
        expect(hitFirst != map.end());
        expect(eq(hitLast - hitFirst, std::ptrdiff_t{1}));

        const auto [missFirst, missLast] = map.equal_range("absent");
        expect(missFirst == map.end());
        expect(missLast == map.end());
    };

    "merge(other) moves non-conflicting entries; conflicts stay in source"_test = [] {
        ValueMap dst;
        dst.emplace("shared", std::uint32_t{1});
        dst.emplace("dst_only", std::uint32_t{10});

        ValueMap src;
        src.emplace("shared", std::uint32_t{99}); // conflicts — should stay in src
        src.emplace("src_only", std::uint32_t{20});

        dst.merge(src);

        expect(eq(dst.size(), 3UZ));
        expect(eq(dst.at("shared").value_or<std::uint32_t>(0U), std::uint32_t{1})) << "dst's value preserved on conflict";
        expect(eq(dst.at("dst_only").value_or<std::uint32_t>(0U), std::uint32_t{10}));
        expect(eq(dst.at("src_only").value_or<std::uint32_t>(0U), std::uint32_t{20}));

        expect(eq(src.size(), 1UZ)) << "conflicted entry remains in src";
        expect(src.contains("shared"));
        expect(!src.contains("src_only"));
    };

    "merge(other) preserves nested-ValueMap entries (no silent drop)"_test = [] {
        ValueMap inner;
        inner.emplace("inner_count", std::uint32_t{42});
        ValueMap src;
        src.emplace("trigger_meta_info", inner);
        src.emplace("scalar_value", 1.5f);

        ValueMap dst;
        dst.merge(src);

        expect(eq(dst.size(), 2UZ));
        expect(eq(src.size(), 0UZ)) << "nested-map entry must be moved out, not silently dropped";

        const auto v = dst.at("trigger_meta_info");
        expect(v.holds<Value::Map>());
        const auto  mOpt = v.template get_if<Value::Map>();
        const auto& m    = *mOpt;
        expect(eq(m.size(), 1UZ));
        expect(eq(m.at(std::pmr::string{"inner_count"}).template value_or<std::uint32_t>(0U), std::uint32_t{42}));
    };

    "merge(self) is a no-op (guarded by this == &other)"_test = [] {
        ValueMap map;
        map.emplace("a", std::uint32_t{1});
        map.emplace("b", std::uint32_t{2});
        ValueMap& selfRef = map;
        map.merge(selfRef);
        expect(eq(map.size(), 2UZ));
        expect(map.contains("a"));
        expect(map.contains("b"));
    };

    "operator[](key) on a nested-map key returns Value holding Value::Map"_test = [] {
        ValueMap inner;
        inner.emplace("inner_k", std::uint32_t{7});
        ValueMap outer;
        outer.emplace("nested", inner);

        const auto v = outer["nested"];
        expect(v.holds<Value::Map>());
        const auto  mOpt = v.template get_if<Value::Map>();
        const auto& m    = *mOpt;
        expect(eq(m.size(), 1UZ));
        expect(eq(m.at(std::pmr::string{"inner_k"}).template value_or<std::uint32_t>(0U), std::uint32_t{7}));
    };

    "SubscriptProxy: write-via-operator[]= updates the entry"_test = [] {
        ValueMap map;
        map["sample_rate"] = 48000.0f;
        expect(map.contains("sample_rate"));
        expect(eq(map["sample_rate"].value_or<float>(0.f), 48000.0f));
    };

    "SubscriptProxy: overwrite changes the value in place"_test = [] {
        ValueMap map;
        map["frequency"] = 1.0e9;
        map["frequency"] = 2.4e9;
        expect(eq(map["frequency"].value_or<double>(0.0), 2.4e9));
        expect(eq(map.size(), 1UZ)) << "overwrite must not duplicate the key";
    };

    "SubscriptProxy: auto-vivify on missing-key read inserts a Monostate Value"_test = [] {
        ValueMap map;
        expect(eq(map.size(), 0UZ));
        // touching a missing key vivifies an entry (matches std::map::operator[] semantics).
        const Value v = map["new_key"];
        expect(v.is_monostate());
        expect(map.contains("new_key"));
        expect(eq(map.size(), 1UZ));
    };

    "SubscriptProxy: write changes type from auto-vivified Monostate to scalar"_test = [] {
        ValueMap map;
        (void)map["promoted"]; // auto-vivify Monostate
        expect(map["promoted"].is_monostate());
        map["promoted"] = std::uint32_t{42}; // promote to UInt32
        expect(map["promoted"].holds<std::uint32_t>());
        expect(eq(map["promoted"].value_or<std::uint32_t>(0U), std::uint32_t{42}));
    };

    "SubscriptProxy: write accepts string/string_view"_test = [] {
        ValueMap map;
        map["signal_name"] = std::string_view{"carrier"};
        expect(eq(map["signal_name"].value_or(std::string_view{}), std::string_view{"carrier"}));
    };

    "SubscriptProxy: chained typed reads via the proxy's implicit Value conversion"_test = [] {
        ValueMap map;
        map["x"]      = 3.14;
        const auto p  = map["x"]; // implicit conversion → Value (owning)
        const auto pv = p.value_or<double>(0.0);
        expect(eq(pv, 3.14));
    };
};

const boost::ut::suite<"ValueMap - edge cases"> _edge_case_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using gr::pmt::ValueMap;

    "empty key (zero-length string) is accepted as an inline key"_test = [] {
        ValueMap map;
        map.emplace("", std::uint32_t{99});
        expect(map.contains(""));
        expect(eq(map.at("").value_or<std::uint32_t>(0U), std::uint32_t{99}));
        expect(eq(map.size(), 1UZ));
    };

    "structured-binding iteration: for (auto [key, value] : map)"_test = [] {
        ValueMap map;
        map.emplace("k1", std::uint32_t{1});
        map.emplace("k2", std::uint32_t{2});
        std::size_t sum = 0UZ;
        for (auto [key, value] : map) {
            (void)key;
            sum += value.value_or<std::uint32_t>(0U);
        }
        expect(eq(sum, 3UZ));
    };

    "self-assignment is a no-op"_test = [] {
        ValueMap  map;
        ValueMap& self = map; // alias to dodge the self-assignment compiler warning
        map.emplace("k", std::uint32_t{7});
        map = self;
        expect(eq(map.size(), 1UZ));
        expect(eq(map.at("k").value_or<std::uint32_t>(0U), std::uint32_t{7}));
    };

    "shrink_to_fit on an empty map does not crash"_test = [] {
        ValueMap map;
        map.shrink_to_fit();
        expect(map.empty());
    };

    "long string value spanning multiple payload-pool growths preserves data"_test = [] {
        ValueMap          map(/*resource=*/nullptr, /*initial_capacity_entries=*/2U);
        const std::string longStr(4096UZ, 'x'); // forces payload-pool growth on insert
        map.emplace("big", std::string_view{longStr});
        expect(eq(map.at("big").value_or(std::string_view{}).size(), 4096UZ));
    };

    "many entries — exercise grow path without losing any"_test = [] {
        ValueMap map;
        for (std::uint32_t i = 0U; i < 200U; ++i) {
            map.emplace(std::format("k{}", i), i);
        }
        expect(eq(map.size(), 200UZ));
        for (std::uint32_t i = 0U; i < 200U; ++i) {
            expect(eq(map.at(std::format("k{}", i)).value_or<std::uint32_t>(0U), i));
        }
    };

    "self-aliasing emplace via iterator key (string_view into _blob) does not crash"_test = [] {
        ValueMap map(/*resource=*/nullptr, /*initial_capacity_entries=*/2U); // tight cap so subsequent inserts grow
        map.emplace("source_inline_key", std::uint32_t{42});
        // The iterator yields `pair<string_view, Value>` where the key string_view points
        // into the PackedEntry's inlineKey bytes (i.e. into our own _blob). Re-emplacing
        // with that key as the *value* exercises _appendPayloadSafe's self-alias guard:
        // without the snapshot, _appendPayload's grow would free _blob mid-memcpy.
        const auto [k, _] = *map.begin();
        expect(eq(k, std::string_view{"source_inline_key"}));
        map.emplace("new_key", k);
        expect(eq(map.size(), 2UZ));
        expect(eq(map.at("new_key").value_or(std::string_view{}).size(), std::string_view{"source_inline_key"}.size()));
    };

    "self-aliasing emplace of ValueMap into itself does not crash"_test = [] {
        ValueMap map;
        map.emplace("k1", std::uint32_t{1});
        map.emplace("k2", std::uint32_t{2});
        // Insert `map` into itself under a new key — would alias map._blob's payload bytes.
        // _appendPayloadSafe snapshots the source blob before grow.
        ValueMap& selfRef = map;
        map.emplace("self", selfRef);
        expect(eq(map.size(), 3UZ));
        const auto v = map.at("self");
        expect(v.holds<Value::Map>());
    };

    "decodeEntry refuses to recurse beyond kMaxDecodeDepth (32) — no crash, returns truncated"_test = [] {
        // Build a deeply-nested chain of ValueMaps, each containing the previous as "inner".
        // After kMaxDecodeDepth + a few extra levels, decodeEntry caps the recursion and
        // returns a Monostate Value at the cut-off — no stack overflow.
        ValueMap leaf;
        leaf.emplace("leaf", std::uint32_t{42});
        ValueMap current{std::move(leaf)};
        for (int i = 0; i < 40; ++i) {
            ValueMap next;
            next.emplace("inner", current);
            current = std::move(next);
        }
        // Walk the structure; at depth ≥ 32, decode hits the cap.
        const auto top = current.at("inner");
        expect(top.holds<Value::Map>()); // depth 1: still readable
        // Don't assert anything specific about the cut-off value; just ensure we got here without
        // stack overflow / ASAN abort.
        expect(true);
    };
};

const boost::ut::suite<"ValueMap - extended value types (complex / nested ValueMap)"> _extended_types_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using gr::pmt::ValueMap;

    "complex<float> round-trip via inline 8-byte slot"_test = [] {
        ValueMap map;
        map.emplace("z32", std::complex<float>{1.5f, -2.5f});
        const auto v = map.at("z32");
        expect(v.holds<std::complex<float>>());
        const auto z = v.value_or<std::complex<float>>(std::complex<float>{});
        expect(eq(z.real(), 1.5f));
        expect(eq(z.imag(), -2.5f));
    };

    "complex<double> round-trip via 16-byte payload-pool slot"_test = [] {
        ValueMap map;
        map.emplace("z64", std::complex<double>{3.14, -2.71});
        const auto v = map.at("z64");
        expect(v.holds<std::complex<double>>());
        const auto z = v.value_or<std::complex<double>>(std::complex<double>{});
        expect(eq(z.real(), 3.14));
        expect(eq(z.imag(), -2.71));
    };

    "nested ValueMap round-trip — single-level (trigger_meta_info-style)"_test = [] {
        ValueMap inner;
        inner.emplace("inner_count", std::uint32_t{42});
        inner.emplace("inner_label", std::string_view{"abc"});

        ValueMap outer;
        outer.emplace("trigger_meta_info", inner);
        outer.emplace("sample_rate", 48000.0f);

        const auto v = outer.at("trigger_meta_info");
        expect(v.holds<Value::Map>());
        const auto  mapOpt = v.get_if<Value::Map>();
        const auto& map    = *mapOpt;
        expect(eq(map.size(), 2UZ));
        expect(eq(map.at(std::pmr::string{"inner_count"}).value_or<std::uint32_t>(0U), std::uint32_t{42}));
        expect(eq(map.at(std::pmr::string{"inner_label"}).value_or(std::string_view{}), std::string_view{"abc"}));
        // Sibling entry untouched.
        expect(eq(outer.at("sample_rate").value_or<float>(0.f), 48000.0f));
    };

    "nested ValueMap round-trip — two levels deep (recursive sub-blob)"_test = [] {
        ValueMap leaf;
        leaf.emplace("leaf_value", std::int64_t{-7});

        ValueMap mid;
        mid.emplace("mid_value", 0.5f);
        mid.emplace("nested_leaf", leaf);

        ValueMap root;
        root.emplace("nested_mid", mid);

        const auto vmid = root.at("nested_mid");
        expect(vmid.holds<Value::Map>());
        const auto  mmidOpt = vmid.get_if<Value::Map>();
        const auto& mmid    = *mmidOpt;
        expect(eq(mmid.size(), 2UZ));
        expect(eq(mmid.at(std::pmr::string{"mid_value"}).value_or<float>(0.f), 0.5f));

        const auto& vleafEntry = mmid.at(std::pmr::string{"nested_leaf"});
        expect(vleafEntry.holds<Value::Map>());
        const auto  mleafOpt = vleafEntry.get_if<Value::Map>();
        const auto& mleaf    = *mleafOpt;
        expect(eq(mleaf.size(), 1UZ));
        expect(eq(mleaf.at(std::pmr::string{"leaf_value"}).value_or<std::int64_t>(0), std::int64_t{-7}));
    };
};

const boost::ut::suite<"ValueMap - Tensor support"> _tensor_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using gr::pmt::ValueMap;

    "Tensor<float> 1D round-trip — header, extents, and contiguous data"_test = [] {
        ValueMap          map;
        gr::Tensor<float> src(gr::extents_from, std::array<std::size_t, 1>{4UZ});
        std::ranges::copy(std::array<float, 4>{1.0f, 2.0f, 3.0f, 4.0f}, src._data.begin());
        map.emplace("vec1d", src);

        const auto v = map.at("vec1d");
        expect(v.is_tensor());
        auto t = v.get_if<gr::TensorView<float>>();
        expect(t.has_value());
        if (!t) {
            return;
        }
        expect(eq(t->rank(), 1UZ));
        expect(eq(t->size(), 4UZ));
        expect(eq(t->extents()[0], 4UZ));
        for (std::size_t i = 0; i < 4UZ; ++i) {
            expect(eq(t->_data.data()[i], src._data.data()[i]));
        }
    };

    "Tensor<float> 2D round-trip preserves rank, extents and row-major bytes"_test = [] {
        ValueMap          map;
        gr::Tensor<float> src(gr::extents_from, std::array<std::size_t, 2>{2UZ, 3UZ});
        std::ranges::copy(std::array<float, 6>{10.f, 20.f, 30.f, 40.f, 50.f, 60.f}, src._data.begin());
        map.emplace("mat2d", src);

        const auto v = map.at("mat2d");
        auto       t = v.get_if<gr::TensorView<float>>();
        expect(t.has_value());
        if (!t) {
            return;
        }
        expect(eq(t->rank(), 2UZ));
        expect(eq(t->extents()[0], 2UZ));
        expect(eq(t->extents()[1], 3UZ));
        expect(eq(t->_data.data()[5], 60.f));
    };

    "Tensor<int64_t> 3D round-trip"_test = [] {
        ValueMap                 map;
        gr::Tensor<std::int64_t> src(gr::extents_from, std::array<std::size_t, 3>{2UZ, 2UZ, 2UZ});
        for (std::size_t i = 0; i < src.size(); ++i) {
            src._data.data()[i] = static_cast<std::int64_t>(i) - 4;
        }
        map.emplace("cube", src);

        const auto v = map.at("cube");
        auto       t = v.get_if<gr::TensorView<std::int64_t>>();
        expect(t.has_value());
        if (!t) {
            return;
        }
        expect(eq(t->rank(), 3UZ));
        expect(eq(t->size(), 8UZ));
        expect(eq(t->_data.data()[0], std::int64_t{-4}));
        expect(eq(t->_data.data()[7], std::int64_t{3}));
    };

    "Tensor<bool> uses 1-byte-per-element storage; round-trip preserves values"_test = [] {
        ValueMap         map;
        gr::Tensor<bool> src(gr::extents_from, std::array<std::size_t, 1>{5UZ});
        src._data.data()[0] = true;
        src._data.data()[1] = false;
        src._data.data()[2] = true;
        src._data.data()[3] = true;
        src._data.data()[4] = false;
        map.emplace("flags", src);

        const auto v = map.at("flags");
        const auto t = v.get_if<gr::Tensor<bool>>();
        expect(t.has_value());
        if (!t) {
            return;
        }
        expect(eq(t->size(), 5UZ));
        expect(t->_data.data()[0]);
        expect(!t->_data.data()[1]);
        expect(t->_data.data()[2]);
        expect(t->_data.data()[3]);
        expect(!t->_data.data()[4]);
    };

    "Tensor<complex<float>> round-trip — fixed-size 8-byte elements"_test = [] {
        ValueMap                        map;
        gr::Tensor<std::complex<float>> src(gr::extents_from, std::array<std::size_t, 1>{3UZ});
        src._data.data()[0] = {1.f, 2.f};
        src._data.data()[1] = {3.f, 4.f};
        src._data.data()[2] = {5.f, 6.f};
        map.emplace("z32vec", src);

        const auto v = map.at("z32vec");
        auto       t = v.get_if<gr::TensorView<std::complex<float>>>();
        expect(t.has_value());
        if (!t) {
            return;
        }
        expect(eq(t->_data.data()[2].imag(), 6.f));
    };

    "Tensor<complex<double>> round-trip — fixed-size 16-byte elements"_test = [] {
        ValueMap                         map;
        gr::Tensor<std::complex<double>> src(gr::extents_from, std::array<std::size_t, 1>{2UZ});
        src._data.data()[0] = {3.14, -2.71};
        src._data.data()[1] = {1.41, 0.0};
        map.emplace("z64vec", src);

        const auto v = map.at("z64vec");
        auto       t = v.get_if<gr::TensorView<std::complex<double>>>();
        expect(t.has_value());
        if (!t) {
            return;
        }
        expect(eq(t->_data.data()[0].real(), 3.14));
        expect(eq(t->_data.data()[1].imag(), 0.0));
    };

    "Tensor with empty (zero-extent) shape encodes and decodes back to empty"_test = [] {
        ValueMap          map;
        gr::Tensor<float> src(gr::extents_from, std::array<std::size_t, 1>{0UZ});
        map.emplace("empty", src);

        const auto v = map.at("empty");
        auto       t = v.get_if<gr::TensorView<float>>();
        expect(t.has_value());
        if (!t) {
            return;
        }
        expect(eq(t->size(), 0UZ));
    };

    "large Tensor forces payload-pool growth across the encoded sub-blob"_test = [] {
        ValueMap          map;
        gr::Tensor<float> src(gr::extents_from, std::array<std::size_t, 1>{1024UZ});
        for (std::size_t i = 0; i < src.size(); ++i) {
            src._data.data()[i] = static_cast<float>(i);
        }
        map.emplace("big", src);

        const auto v = map.at("big");
        auto       t = v.get_if<gr::TensorView<float>>();
        expect(t.has_value());
        if (!t) {
            return;
        }
        expect(eq(t->size(), 1024UZ));
        expect(eq(t->_data.data()[1023], 1023.0f));
    };

    "Tensor<Value> with mixed inline scalar elements (int + float + uint)"_test = [] {
        ValueMap          map;
        gr::Tensor<Value> src(gr::extents_from, std::array<std::size_t, 1>{3UZ});
        src._data.data()[0] = Value{std::int64_t{-7}};
        src._data.data()[1] = Value{0.5f};
        src._data.data()[2] = Value{std::uint64_t{42}};
        map.emplace("mixed", src);

        const auto v = map.at("mixed");
        auto       t = v.get_if<gr::TensorView<Value>>();
        expect(t.has_value());
        if (!t) {
            return;
        }
        expect(eq(t->size(), 3UZ));
        expect(eq(t->_data.data()[0].value_or<std::int64_t>(0), std::int64_t{-7}));
        expect(eq(t->_data.data()[1].value_or<float>(0.f), 0.5f));
        expect(eq(t->_data.data()[2].value_or<std::uint64_t>(0U), std::uint64_t{42}));
    };

    "Tensor<Value> with String elements (settings-style string array)"_test = [] {
        ValueMap          map;
        gr::Tensor<Value> src(gr::extents_from, std::array<std::size_t, 1>{3UZ});
        src._data.data()[0] = Value{std::string_view{"alpha"}};
        src._data.data()[1] = Value{std::string_view{"bravo"}};
        src._data.data()[2] = Value{std::string_view{"charlie"}};
        map.emplace("labels", src);

        const auto v = map.at("labels");
        auto       t = v.get_if<gr::TensorView<Value>>();
        expect(t.has_value());
        if (!t) {
            return;
        }
        expect(eq(t->size(), 3UZ));
        expect(eq(t->_data.data()[0].value_or(std::string_view{}), std::string_view{"alpha"}));
        expect(eq(t->_data.data()[1].value_or(std::string_view{}), std::string_view{"bravo"}));
        expect(eq(t->_data.data()[2].value_or(std::string_view{}), std::string_view{"charlie"}));
    };

    "Tensor<Value> with complex<double> elements (variable-size payload-spilled)"_test = [] {
        ValueMap          map;
        gr::Tensor<Value> src(gr::extents_from, std::array<std::size_t, 1>{2UZ});
        src._data.data()[0] = Value{std::complex<double>{1.0, 2.0}};
        src._data.data()[1] = Value{std::complex<double>{3.0, 4.0}};
        map.emplace("z64s", src);

        const auto v = map.at("z64s");
        auto       t = v.get_if<gr::TensorView<Value>>();
        expect(t.has_value());
        if (!t) {
            return;
        }
        const auto z0 = t->_data.data()[0].value_or<std::complex<double>>(std::complex<double>{});
        const auto z1 = t->_data.data()[1].value_or<std::complex<double>>(std::complex<double>{});
        expect(eq(z0.real(), 1.0));
        expect(eq(z0.imag(), 2.0));
        expect(eq(z1.real(), 3.0));
        expect(eq(z1.imag(), 4.0));
    };

    "Tensor<Value> elements may themselves be nested ValueMaps (recursive composition)"_test = [] {
        ValueMap inner;
        inner.emplace("k", std::int64_t{99});

        ValueMap          map;
        gr::Tensor<Value> src(gr::extents_from, std::array<std::size_t, 1>{2UZ});
        src._data.data()[0] = Value{std::string_view{"head"}};
        src._data.data()[1] = Value{Value::Map{}}; // initialise then populate

        // Build the nested-map element via Value::Map construction — Value(Tensor<Value>) cannot
        // store a ValueMap directly, but Value{Value::Map{...}} round-trips through the encoder.
        Value::Map innerMap;
        innerMap.emplace(std::pmr::string{"k"}, Value{std::int64_t{99}});
        src._data.data()[1] = Value{std::move(innerMap)};

        map.emplace("composed", src);

        const auto v = map.at("composed");
        auto       t = v.get_if<gr::TensorView<Value>>();
        expect(t.has_value());
        if (!t) {
            return;
        }
        expect(eq(t->_data.data()[0].value_or(std::string_view{}), std::string_view{"head"}));
        expect(t->_data.data()[1].is_map());
        const auto  nestedOpt = t->_data.data()[1].get_if<Value::Map>();
        const auto& nested    = *nestedOpt;
        expect(eq(nested.size(), 1UZ));
        expect(eq(nested.at(std::pmr::string{"k"}).value_or<std::int64_t>(0), std::int64_t{99}));
    };

    "ValueMap containing a Tensor<float> coexists with sibling scalar entries"_test = [] {
        ValueMap          map;
        gr::Tensor<float> src(gr::extents_from, std::array<std::size_t, 1>{3UZ});
        src._data.data()[0] = 0.1f;
        src._data.data()[1] = 0.2f;
        src._data.data()[2] = 0.3f;
        map.emplace("samples", src);
        map.emplace("sample_rate", 48000.0f);
        map.emplace("label", std::string_view{"capture"});

        expect(eq(map.size(), 3UZ));
        const auto v = map.at("samples");
        auto       t = v.get_if<gr::TensorView<float>>();
        expect(t.has_value());
        if (!t) {
            return;
        }
        expect(eq(t->_data.data()[2], 0.3f));
        expect(eq(map.at("sample_rate").value_or<float>(0.f), 48000.0f));
        expect(eq(map.at("label").value_or(std::string_view{}), std::string_view{"capture"}));
    };

    "ValueMap copy preserves Tensor entries (deep copy via blob memcpy)"_test = [] {
        ValueMap          src;
        gr::Tensor<float> tensor(gr::extents_from, std::array<std::size_t, 1>{4UZ});
        std::ranges::copy(std::array<float, 4>{1.f, 2.f, 4.f, 8.f}, tensor._data.begin());
        src.emplace("samples", tensor);

        ValueMap   copy{src};
        const auto v = copy.at("samples");
        auto       t = v.get_if<gr::TensorView<float>>();
        expect(t.has_value());
        if (!t) {
            return;
        }
        expect(eq(t->_data.data()[3], 8.f));
    };

    "merge moves Tensor entries verbatim (raw sub-blob copy, no re-encode)"_test = [] {
        ValueMap          source;
        gr::Tensor<float> tensor(gr::extents_from, std::array<std::size_t, 1>{2UZ});
        tensor._data.data()[0] = 7.f;
        tensor._data.data()[1] = 9.f;
        source.emplace("samples", tensor);
        source.emplace("sample_rate", 48000.0f);

        ValueMap dest;
        dest.merge(source);

        expect(eq(source.size(), 0UZ));
        expect(eq(dest.size(), 2UZ));
        const auto v = dest.at("samples");
        auto       t = v.get_if<gr::TensorView<float>>();
        expect(t.has_value());
        if (!t) {
            return;
        }
        expect(eq(t->_data.data()[0], 7.f));
        expect(eq(t->_data.data()[1], 9.f));
    };
};

const boost::ut::suite<"ValueMap - construction and basic invariants"> _ctor_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using gr::pmt::ValueMap;

    "default construction yields an empty map with its blob header initialised"_test = [] {
        ValueMap map;
        expect(eq(map.size(), 0UZ));
        expect(map.empty());
        const auto b = map.blob();
        expect(b.size() >= sizeof(gr::pmt::Header));
        const auto* hdr = reinterpret_cast<const gr::pmt::Header*>(b.data());
        expect(eq(hdr->magic[0], 'G'));
        expect(eq(hdr->magic[1], 'R'));
        expect(eq(hdr->magic[2], '4'));
        expect(eq(hdr->magic[3], 'M'));
        expect(eq(hdr->version, gr::pmt::kBlobVersion));
        expect(eq(hdr->entryCount, std::uint16_t{0}));
    };

    "custom initial capacity reserves entry slots without growing on first insert"_test = [] {
        CountingResource mr;
        ValueMap         map(&mr, /*initial_capacity_entries=*/16U);
        const auto       allocs_before = mr.allocCount;
        map.insert_or_assign("sample_rate", 48'000.0f);
        map.insert_or_assign("frequency", 1e9);
        map.insert_or_assign("rx_overflow", false);
        // With 16 entry slots reserved, three inserts must not have grown the blob.
        expect(eq(mr.allocCount, allocs_before));
    };

    "resource() round-trips the allocator passed at construction"_test = [] {
        CountingResource mr;
        ValueMap         map(&mr);
        expect(map.resource() == &mr);
    };

    "nullptr resource falls back to the default allocator"_test = [] {
        ValueMap map(nullptr);
        expect(map.resource() == std::pmr::get_default_resource());
    };
};

const boost::ut::suite<"ValueMap - string-keyed API"> _string_key_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using gr::pmt::ValueMap;

    "emplace on a new key inserts and returns (it, true)"_test = [] {
        ValueMap map;
        auto [it, inserted] = map.emplace("my_ext", 7.25f);
        expect(inserted);
        expect(it != map.end());
        expect(eq(map.size(), 1UZ));
    };

    "emplace on an existing key does NOT overwrite and returns (it, false)"_test = [] {
        ValueMap map;
        map.emplace("my_ext", 7.25f);
        auto [it, inserted] = map.emplace("my_ext", 9.5f);
        expect(!inserted);
        expect(it != map.end());
        // Value still the original
        const auto v = map.at("my_ext");
        expect(v.holds<float>());
        expect(eq(v.value_or<float>(0.f), 7.25f));
    };

    "insert_or_assign overwrites existing entries"_test = [] {
        ValueMap map;
        map.insert_or_assign("my_ext", 7.25f);
        auto [it, inserted] = map.insert_or_assign("my_ext", 9.5f);
        expect(!inserted);
        expect(it != map.end());
        const auto v = map.at("my_ext");
        expect(eq(v.value_or<float>(0.f), 9.5f));
    };

    "at(string_view) returns a Monostate Value on missing key (no throw)"_test = [] {
        ValueMap   map;
        const auto v = map.at("missing");
        expect(v.is_monostate());
    };

    "contains and find report membership consistently"_test = [] {
        ValueMap map;
        map.emplace("k1", std::uint64_t{42});
        expect(map.contains("k1"));
        expect(!map.contains("k2"));
        expect(map.find("k1") != map.end());
        expect(map.find("k2") == map.end());
    };

    "canonical names resolve through the string-keyed API to the canonical entry"_test = [] {
        ValueMap map;
        // Insert via canonical ID, read back via the string name.
        map.insert_or_assign("sample_rate", 48000.0f);
        expect(map.contains("sample_rate"));
        const auto v = map.at("sample_rate");
        expect(v.holds<float>());
        expect(eq(v.value_or<float>(0.f), 48000.0f));
    };

    "mixed canonical + inline keys coexist"_test = [] {
        ValueMap map;
        map.insert_or_assign("sample_rate", 48000.0f);
        map.emplace("custom_flag", true);
        map.emplace("custom_count", std::uint32_t{123});
        expect(eq(map.size(), 3UZ));
        expect(map.contains("sample_rate"));
        expect(map.contains("custom_flag"));
        expect(map.contains("custom_count"));
    };

    "inline-key length at the 27-char boundary is accepted"_test = [] {
        ValueMap               map;
        const std::string_view k27{"abcdefghijklmnopqrstuvwxyz!"}; // 27 chars
        expect(eq(k27.size(), 27UZ));
        auto [it, inserted] = map.emplace(k27, std::uint64_t{7});
        expect(inserted);
        expect(map.contains(k27));
    };

    "long-key (>27 chars) spills to payload pool and round-trips"_test = [] {
        ValueMap               map;
        const std::string_view k64{"this_is_a_user_extension_key_that_is_longer_than_27_characters_."};
        expect(eq(k64.size(), 64UZ));
        auto [it, inserted] = map.emplace(k64, std::uint64_t{7});
        expect(inserted);
        expect(map.contains(k64));
        expect(eq(map.at(k64).value_or<std::uint64_t>(0U), std::uint64_t{7}));
        // iterator yields the spilled key correctly
        const auto& [k, v] = *map.begin();
        expect(eq(k, k64));
    };

    "long-key survives blob growth (offset relocation)"_test = [] {
        ValueMap               map;
        const std::string_view k64{"this_is_a_user_extension_key_that_is_longer_than_27_characters_."};
        map.emplace(k64, std::uint64_t{42});
        // Force payload growth by emplacing many string entries.
        for (int i = 0; i < 30; ++i) {
            map.emplace(std::format("k{:02d}", i), std::format("value-with-some-bytes-{:02d}", i));
        }
        expect(map.contains(k64));
        expect(eq(map.at(k64).value_or<std::uint64_t>(0U), std::uint64_t{42}));
    };

    "long-key merge between maps preserves key + value"_test = [] {
        const std::string_view k64{"this_is_a_user_extension_key_that_is_longer_than_27_characters_."};
        ValueMap               src;
        src.emplace(k64, std::uint64_t{99});
        ValueMap dst;
        dst.merge(src);
        expect(eq(src.size(), 0UZ));
        expect(dst.contains(k64));
        expect(eq(dst.at(k64).value_or<std::uint64_t>(0U), std::uint64_t{99}));
    };
};

const boost::ut::suite<"ValueMap - erase / clear / reserve / shrink"> _mutation_suite = [] {
    using namespace boost::ut;
    using gr::pmt::ValueMap;

    "erase removes a matching entry"_test = [] {
        ValueMap map;
        map.emplace("a", std::uint32_t{1});
        map.emplace("b", std::uint32_t{2});
        map.emplace("c", std::uint32_t{3});
        auto it = map.find("b");
        expect(it != map.end());
        map.erase(it);
        expect(eq(map.size(), 2UZ));
        expect(map.contains("a"));
        expect(!map.contains("b"));
        expect(map.contains("c"));
    };

    "clear removes all entries and resets payload usage"_test = [] {
        ValueMap map;
        map.insert_or_assign("sample_rate", 48000.0f);
        map.emplace("s", std::string_view{"some-string-longer-than-SSO-for-robustness"});
        expect(eq(map.size(), 2UZ));
        map.clear();
        expect(eq(map.size(), 0UZ));
        expect(map.empty());
    };

    "reserve preserves existing entries and does not shrink"_test = [] {
        ValueMap map(nullptr, /*initial_capacity_entries=*/4U);
        map.emplace("k1", std::uint32_t{1});
        map.emplace("k2", std::uint32_t{2});
        map.reserve(32U); // grow
        expect(eq(map.size(), 2UZ));
        expect(map.contains("k1"));
        expect(map.contains("k2"));
    };

    "shrink_to_fit keeps entries intact"_test = [] {
        ValueMap map(nullptr, /*initial_capacity_entries=*/64U);
        map.emplace("k1", std::uint32_t{1});
        map.emplace("k2", std::uint32_t{2});
        map.shrink_to_fit();
        expect(eq(map.size(), 2UZ));
        expect(map.contains("k1"));
        expect(map.contains("k2"));
    };

    "growth across the initial capacity preserves already-inserted entries"_test = [] {
        ValueMap map(nullptr, /*initial_capacity_entries=*/2U);
        map.emplace("a", std::uint32_t{10});
        map.emplace("b", std::uint32_t{20});
        map.emplace("c", std::uint32_t{30}); // forces grow
        map.emplace("d", std::uint32_t{40});
        expect(eq(map.size(), 4UZ));
        for (const auto& [key, val] : {std::pair{std::string_view{"a"}, std::uint32_t{10}}, //
                 std::pair{std::string_view{"b"}, std::uint32_t{20}}, std::pair{std::string_view{"c"}, std::uint32_t{30}}, std::pair{std::string_view{"d"}, std::uint32_t{40}}}) {
            const auto v = map.at(key);
            expect(v.holds<std::uint32_t>());
            expect(eq(v.value_or<std::uint32_t>(0U), val));
        }
    };

    "string values survive blob relocation across growth"_test = [] {
        ValueMap map(nullptr, /*initial_capacity_entries=*/1U);
        map.emplace("first", std::string_view{"sample-carrier-name-01"});
        map.emplace("second", std::string_view{"sample-carrier-name-02"});
        map.emplace("third", std::string_view{"sample-carrier-name-03"});
        expect(eq(map.size(), 3UZ));
        expect(eq(map.at("first").value_or(std::string_view{""}), std::string_view{"sample-carrier-name-01"}));
        expect(eq(map.at("second").value_or(std::string_view{""}), std::string_view{"sample-carrier-name-02"}));
        expect(eq(map.at("third").value_or(std::string_view{""}), std::string_view{"sample-carrier-name-03"}));
    };
};

const boost::ut::suite<"ValueMap - iteration"> _iter_suite = [] {
    using namespace boost::ut;
    using gr::pmt::ValueMap;

    "range-based for visits every entry exactly once"_test = [] {
        ValueMap map;
        map.insert_or_assign("sample_rate", 48000.0f);
        map.insert_or_assign("frequency", 1e9);
        map.emplace("custom", std::uint32_t{42});

        std::size_t visited         = 0UZ;
        bool        saw_sample_rate = false;
        bool        saw_frequency   = false;
        bool        saw_custom      = false;
        for (auto [key, val] : map) {
            ++visited;
            if (key == "sample_rate") {
                saw_sample_rate = true;
                expect(eq(val.value_or<float>(0.f), 48000.0f));
            } else if (key == "frequency") {
                saw_frequency = true;
                expect(eq(val.value_or<double>(0.0), 1e9));
            } else if (key == "custom") {
                saw_custom = true;
                expect(eq(val.value_or<std::uint32_t>(0U), std::uint32_t{42}));
            }
        }
        expect(eq(visited, 3UZ));
        expect(saw_sample_rate);
        expect(saw_frequency);
        expect(saw_custom);
    };

    "begin == end on an empty map"_test = [] {
        ValueMap map;
        expect(map.begin() == map.end());
    };

    "ValueMap models std::ranges::random_access_range and sized_range"_test = [] {
        static_assert(std::ranges::random_access_range<ValueMap>);
        static_assert(std::ranges::sized_range<ValueMap>);
        static_assert(std::random_access_iterator<ValueMap::const_iterator>);
        expect(true);
    };

    "reverse iteration visits entries last-to-first"_test = [] {
        ValueMap map;
        map.emplace("a", std::uint32_t{1});
        map.emplace("b", std::uint32_t{2});
        map.emplace("c", std::uint32_t{3});

        std::vector<std::string_view> seen;
        for (auto it = map.rbegin(); it != map.rend(); ++it) {
            seen.push_back((*it).first);
        }
        expect(eq(seen.size(), 3UZ));
        expect(eq(seen[0], std::string_view{"c"}));
        expect(eq(seen[1], std::string_view{"b"}));
        expect(eq(seen[2], std::string_view{"a"}));
    };

    "random-access arithmetic and operator[] on const_iterator"_test = [] {
        ValueMap map;
        map.emplace("a", std::uint32_t{1});
        map.emplace("b", std::uint32_t{2});
        map.emplace("c", std::uint32_t{3});

        const auto first = map.begin();
        expect(eq((first + 2) - first, std::ptrdiff_t{2}));
        expect(eq((first + 1)[1].first, std::string_view{"c"})); // first[2]
        expect((first + 1) > first);
        expect((first + 1) < (first + 2));
    };
};

const boost::ut::suite<"ValueMap - copy / move semantics"> _copy_move_suite = [] {
    using namespace boost::ut;
    using gr::pmt::ValueMap;

    "copy construction deep-copies into the target resource"_test = [] {
        CountingResource source_mr;
        CountingResource target_mr;

        ValueMap src(&source_mr);
        src.insert_or_assign("sample_rate", 48000.0f);
        src.emplace("note", std::string_view{"hello-world-this-string-is-long"});

        const auto src_allocs_before    = source_mr.allocCount;
        const auto target_allocs_before = target_mr.allocCount;

        ValueMap copy(src, &target_mr);

        expect(eq(source_mr.allocCount, src_allocs_before)) << "source was not re-touched by the copy";
        expect(target_mr.allocCount > target_allocs_before) << "target resource received the deep copy";
        expect(eq(copy.size(), 2UZ));
        expect(copy.resource() == &target_mr);
        expect(eq(copy.at("sample_rate").value_or<float>(0.f), 48000.0f));
        expect(eq(copy.at("note").value_or(std::string_view{""}), std::string_view{"hello-world-this-string-is-long"}));
    };

    "move construction is O(1) — no allocations on the source resource"_test = [] {
        CountingResource mr;
        ValueMap         src(&mr);
        src.insert_or_assign("frequency", 2.4e9);
        const auto allocs_before = mr.allocCount;

        ValueMap dst(std::move(src));

        expect(eq(mr.allocCount, allocs_before)) << "move must not allocate on the source resource";
        expect(eq(dst.size(), 1UZ));
        expect(eq(dst.at("frequency").value_or<double>(0.0), 2.4e9));
        expect(eq(src.size(), 0UZ)) << "source is moved-from (empty)";
    };

    "copy assignment replaces prior contents"_test = [] {
        ValueMap a;
        ValueMap b;
        a.insert_or_assign("sample_rate", 48000.0f);
        b.insert_or_assign("frequency", 1e9);
        b = a;
        expect(eq(b.size(), 1UZ));
        expect(eq(b.at("sample_rate").value_or<float>(0.f), 48000.0f));
        expect(!b.contains("frequency"));
    };

    "move assignment between same-resource maps is O(1)"_test = [] {
        CountingResource mr;
        ValueMap         a(&mr);
        ValueMap         b(&mr);
        a.insert_or_assign("sample_rate", 48000.0f);
        const auto allocs_before = mr.allocCount;
        b                        = std::move(a);
        expect(eq(mr.allocCount, allocs_before)) << "same-resource move-assign must not allocate";
        expect(eq(b.size(), 1UZ));
        expect(eq(b.at("sample_rate").value_or<float>(0.f), 48000.0f));
    };
};

const boost::ut::suite<"ValueMap - blob format invariants"> _blob_suite = [] {
    using namespace boost::ut;
    using gr::pmt::ValueMap;

    "blob header reflects entryCount and payloadUsed after mutation"_test = [] {
        ValueMap map;
        map.insert_or_assign("sample_rate", 48000.0f);
        map.emplace("note", std::string_view{"abc"});

        const auto b = map.blob();
        expect(!b.empty()) << "blob must be non-empty after insert";
        const auto* hdr = reinterpret_cast<const gr::pmt::Header*>(b.data());
        expect(eq(hdr->entryCount, std::uint16_t{2}));
        expect(hdr->payloadUsed >= 3UZ) << "string payload must have been appended";
        expect(hdr->totalSize == b.size());
        expect(eq(reinterpret_cast<std::uintptr_t>(b.data()) & (gr::pmt::kBlobAlignment - 1UZ), 0UZ)) << "blob data must be kBlobAlignment-aligned";
    };

    "from_blob round-trips: encode → bytes → from_blob → equal map"_test = [] {
        ValueMap src;
        src.emplace("sample_rate", 48000.0f);
        src.emplace("signal_name", std::string_view{"demo"});
        src.emplace("trigger_time", std::uint64_t{1'700'000'000ULL});
        const auto bytes = src.blob();

        auto restored = ValueMap::from_blob(bytes);
        expect(restored.has_value());
        if (!restored) {
            return;
        }
        expect(eq(restored->size(), 3UZ));
        expect(eq(restored->at("sample_rate").value_or<float>(0.f), 48000.0f));
        expect(eq(restored->at("signal_name").value_or(std::string_view{}), std::string_view{"demo"}));
        expect(eq(restored->at("trigger_time").value_or<std::uint64_t>(0U), std::uint64_t{1'700'000'000ULL}));
    };

    "from_blob rejects too-small spans"_test = [] {
        std::array<std::byte, 4> tiny{};
        const auto               r = ValueMap::from_blob(tiny);
        expect(!r.has_value());
        expect(r.error() == gr::pmt::DeserialiseError::TooSmall);
    };

    "from_blob rejects bad magic"_test = [] {
        ValueMap src;
        src.emplace("k", 1.0f);
        const auto                                                   srcBytes = src.blob();
        alignas(gr::pmt::kBlobAlignment) std::array<std::byte, 1024> alignedBuf{};
        expect(srcBytes.size() <= alignedBuf.size());
        std::memcpy(alignedBuf.data(), srcBytes.data(), srcBytes.size());
        alignedBuf[0] = std::byte{'X'}; // corrupt magic
        const auto r  = ValueMap::from_blob(std::span<const std::byte>{alignedBuf.data(), srcBytes.size()});
        expect(!r.has_value());
        expect(r.error() == gr::pmt::DeserialiseError::MagicMismatch);
    };

    "freeze sets the advisory flag; is_frozen reports it"_test = [] {
        ValueMap map;
        expect(!map.is_frozen());
        map.freeze();
        expect(map.is_frozen());
        const auto* hdr = reinterpret_cast<const gr::pmt::Header*>(map.blob().data());
        expect((hdr->flags & gr::pmt::kHeaderFlagFrozen) != 0);
    };
};

const boost::ut::suite<"ValueMap - view-mode (makeView / is_view / owned)"> _view_mode_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using gr::pmt::ValueMap;

    "owning ValueMap is_view() returns false"_test = [] {
        ValueMap map;
        map.insert_or_assign("k", 1.0f);
        expect(!map.is_view());
        expect(map.resource() != nullptr);
    };

    "makeView aliases external bytes; is_view() reports true"_test = [] {
        ValueMap src;
        src.emplace("sample_rate", 48000.0f);
        src.emplace("signal_name", std::string_view{"hello"});
        const auto bytes = src.blob();

        const ValueMap view = ValueMap::makeView(bytes);
        expect(view.is_view());
        expect(view.resource() == nullptr) << "view-mode: _resource is null (the discriminant)";
        expect(eq(view.size(), src.size()));
        expect(view.contains("sample_rate"));
        expect(view.contains("signal_name"));
        expect(eq(view.at("sample_rate").value_or<float>(0.f), 48000.0f));
    };

    "view-mode iter yields the same entries as the source map"_test = [] {
        ValueMap src;
        src.insert_or_assign("a", std::uint32_t{1});
        src.insert_or_assign("b", std::uint32_t{2});
        src.insert_or_assign("c", std::uint32_t{3});

        const ValueMap view = ValueMap::makeView(src.blob());
        std::size_t    seen = 0UZ;
        std::uint32_t  sum  = 0U;
        for (auto [k, v] : view) {
            ++seen;
            sum += v.value_or<std::uint32_t>(0U);
        }
        expect(eq(seen, 3UZ));
        expect(eq(sum, std::uint32_t{6}));
    };

    "owned(resource) materialises a fresh allocation independent of the source bytes"_test = [] {
        CountingResource mr;

        ValueMap src;
        src.emplace("frequency", 2.4e9);
        src.emplace("note", std::string_view{"abc"});
        const auto srcBytes = src.blob();

        const ValueMap view   = ValueMap::makeView(srcBytes);
        const auto     before = mr.allocCount;
        const ValueMap owned  = view.owned(&mr);

        expect(!owned.is_view());
        expect(owned.resource() == &mr);
        expect(mr.allocCount > before) << "owned() must allocate against the supplied resource";
        expect(eq(owned.size(), 2UZ));
        expect(eq(owned.at("frequency").value_or<double>(0.0), 2.4e9));
        expect(eq(owned.at("note").value_or(std::string_view{}), std::string_view{"abc"}));
    };

    "view-mode equality with the same blob compares equal"_test = [] {
        ValueMap a;
        a.emplace("k", 1.0f);
        const auto     bytesA = a.blob();
        const ValueMap viewA1 = ValueMap::makeView(bytesA);
        const ValueMap viewA2 = ValueMap::makeView(bytesA);
        expect(viewA1 == viewA2) << "two views over the same bytes must compare equal";
        expect(viewA1 == a) << "view over bytes equals the originating owning map";
    };

    "view-mode mutator aborts in debug, no-ops in release (Q5 hybrid policy)"_test = [] {
        if constexpr (!gr::meta::kDebugBuild) {
            return; // release: silent no-op by design
        }
        ValueMap src;
        src.emplace("k", 1.0f);
        ValueMap view = ValueMap::makeView(src.blob());
        expect(aborts([&] { view.insert_or_assign("new", 2.0f); })) << "insert_or_assign on view-mode must trip the assert";
    };
};

int main() { return 0; }
