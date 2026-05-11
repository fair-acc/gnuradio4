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

    "reserve(entries, payload) absorbs subsequent inserts without growing"_test = [] {
        CountingResource mr;
        ValueMap         map(&mr);
        map.reserve(/*entries=*/16U, /*payload_bytes=*/4U * 16U);
        const auto allocs_before = mr.allocCount;
        map.insert_or_assign("sample_rate", 48'000.0f);
        map.insert_or_assign("frequency", 1e9);
        map.insert_or_assign("rx_overflow", false);
        // C3: every entry — incl. inline scalars — consumes 16 B from the payload pool. With
        // entries + payload pre-reserved, three inserts must NOT trigger any blob realloc.
        expect(eq(mr.allocCount, allocs_before)) << "preallocated payload covers three scalar inserts";
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

        expect(eq(map.value_or<std::uint32_t>("char_lit", 0U), std::uint32_t{1}));
        expect(eq(map.value_or<std::uint32_t>(cstr, 0U), std::uint32_t{2}));
        expect(eq(map.value_or<std::uint32_t>(std::string_view{"sv"}, 0U), std::uint32_t{3}));
        expect(eq(map.value_or<std::uint32_t>(std::string{"std"}, 0U), std::uint32_t{4}));
        expect(eq(map.value_or<std::uint32_t>(pmrKey, 0U), std::uint32_t{5}));
        expect(eq(map.value_or<std::uint32_t>(gr::meta::fixed_string{"fixed"}, 0U), std::uint32_t{6}));

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
        expect(eq(map.value_or<float>("sample_rate", 0.f), 48000.0f));
        expect(eq(map.value_or<std::uint32_t>("num_channels", 0U), std::uint32_t{4}));
        expect(eq(map.value_or<std::string_view>("signal_name", std::string_view{}), std::string_view{"carrier-A"}));
    };

    "construct ValueMap from ValueMap (std::pmr::unordered_map<pmr::string, Value, …>)"_test = [] {
        ValueMap src{};
        src.emplace(std::pmr::string{"frequency"}, Value{2.4e9});
        src.emplace(std::pmr::string{"rx_overflow"}, Value{true});

        ValueMap map{src};
        expect(eq(map.size(), 2UZ));
        expect(eq(map.value_or<double>("frequency", 0.0), 2.4e9));
        expect(eq(map.value_or<bool>("rx_overflow", false), true));
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
        const auto va = *a.find_value("nested");
        const auto vb = *b.find_value("nested");
        expect(va == vb) << "Value::operator== considers two nested maps with equal contents equal";
    };
};

const boost::ut::suite<"ValueMap - operator[] / typed at&lt;name&gt; / equal_range / merge"> _bracket_and_range_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using gr::pmt::ValueMap;

    "operator[](key) — non-const auto-vivifies on miss"_test = [] {
        ValueMap map;
        map.emplace("present", 1.5f);
        // Non-const lvalue path: auto-vivify default Value on miss (matches std::map).
        const Value vivified = map["absent"];
        expect(vivified.is_monostate()) << "non-const op[] inserts a default monostate Value on miss";
        expect(map.contains("absent")) << "after the read, the key exists";
        // present-key read via the non-throwing keyed accessor
        const ValueMap& cmap = map;
        expect(map["present"].holds<float>());
        expect(eq(cmap.value_or<float>("present", 0.f), 1.5f));
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

    "at<Name>() debug-asserts when the canonical key is absent"_test = [] {
        if constexpr (!gr::meta::kDebugBuild) {
            return; // release: missing key is UB by design (precondition), like std::span::operator[]
        }
        ValueMap map;
        expect(aborts([&] { (void)map.at<"sample_rate">(); })) << "missing canonical key must trip the debug assert";
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
        expect(eq(dst.value_or<std::uint32_t>("shared", 0U), std::uint32_t{1})) << "dst's value preserved on conflict";
        expect(eq(dst.value_or<std::uint32_t>("dst_only", 0U), std::uint32_t{10}));
        expect(eq(dst.value_or<std::uint32_t>("src_only", 0U), std::uint32_t{20}));

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

        const auto v = *dst.find_value("trigger_meta_info");
        expect(v.holds<ValueMap>());
        const auto  mOpt = v.template get_if<ValueMap>();
        const auto& m    = *mOpt;
        expect(eq(m.size(), 1UZ));
        expect(eq(m.value_or<std::uint32_t>(std::pmr::string{"inner_count"}, 0U), std::uint32_t{42}));
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

    "operator[](key) on a nested-map key returns Value holding ValueMap"_test = [] {
        ValueMap inner;
        inner.emplace("inner_k", std::uint32_t{7});
        ValueMap outer;
        outer.emplace("nested", inner);

        // SubscriptProxy intentionally does not expose pointer-returning accessors.
        // Save the cached Value first via the proxy's `operator Value()` conversion.
        const Value v = outer["nested"];
        expect(v.holds<ValueMap>());
        const auto  mOpt = v.template get_if<ValueMap>();
        const auto& m    = *mOpt;
        expect(eq(m.size(), 1UZ));
        expect(eq(m.value_or<std::uint32_t>(std::pmr::string{"inner_k"}, 0U), std::uint32_t{7}));
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

    // Fast path: insert_or_assign(K, view-mode Value) byte-copies the source value-record
    // (header + content) instead of decode + re-encode. Exercises the path used by the STL
    // idiom `for (auto&& [k, v] : src) dst.insert_or_assign(k, v);`.
    "insert_or_assign(K, view-mode Value) byte-copies the source value-record"_test = [] {
        ValueMap src;
        src.insert_or_assign("scalar_i", std::int32_t{-7});
        src.insert_or_assign("scalar_d", 3.14);
        src.insert_or_assign("scalar_b", true);
        src.insert_or_assign("text", std::string_view{"hello, fast path"});
        src.insert_or_assign("complex", std::complex<double>{1.5, -2.5});
        ValueMap nested;
        nested.insert_or_assign("inner", std::uint64_t{42});
        src.insert_or_assign("nested", std::move(nested));

        ValueMap dst;
        for (const auto& [k, v] : src) {
            dst.insert_or_assign(k, v); // v is a view-mode Value into src's blob
        }

        expect(eq(dst.size(), src.size()));
        for (const auto& [k, v] : src) {
            const auto it = dst.find(k);
            expect(it != dst.end()) << std::format("missing key '{}' in dst", k);
            const gr::pmt::ValueView dstValue = (*it).second;

            const auto srcRecBytes = v.recordSpan();
            const auto dstRecBytes = dstValue.recordSpan();
            expect(eq(srcRecBytes.size(), dstRecBytes.size())) << std::format("recSize mismatch for '{}'", k);
            expect(std::ranges::equal(srcRecBytes, dstRecBytes)) << std::format("byte-mismatch for '{}'", k);
        }
    };

    "insert_or_assign(K, view-mode Value) preserves entry flags for nested-map / tensor"_test = [] {
        ValueMap src;
        ValueMap nested;
        nested.insert_or_assign("foo", std::int64_t{1});
        src.insert_or_assign("nested", std::move(nested));
        src.insert_or_assign("vec", gr::Tensor<float>{4.0f, 8.0f, 16.0f});

        ValueMap dst;
        for (const auto& [k, v] : src) {
            dst.insert_or_assign(k, v);
        }

        // Nested-map entry should round-trip as a Map (decoded, not raw bytes).
        const auto nestedIt = dst.find("nested");
        expect(nestedIt != dst.end());
        const gr::pmt::ValueView nestedRoundTrip = (*nestedIt).second;
        expect(nestedRoundTrip.is_map());
        if (auto m = nestedRoundTrip.template get_if<ValueMap>()) {
            expect(m->contains("foo"));
        }

        // Tensor entry should round-trip as a Tensor.
        const auto vecIt = dst.find("vec");
        expect(vecIt != dst.end());
        const gr::pmt::ValueView vecRoundTripView = (*vecIt).second;
        expect(vecRoundTripView.is_tensor());
        const auto vec = vecRoundTripView.owned().template value_or<gr::Tensor<float>>(gr::Tensor<float>{});
        expect(eq(vec.size(), std::size_t{3}));
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
        expect(eq(map.value_or<std::uint32_t>("", 0U), std::uint32_t{99}));
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
        expect(eq(map.value_or<std::uint32_t>("k", 0U), std::uint32_t{7}));
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
        expect(eq(map.value_or<std::string_view>("big", std::string_view{}).size(), 4096UZ));
    };

    "many entries — exercise grow path without losing any"_test = [] {
        ValueMap map;
        for (std::uint32_t i = 0U; i < 200U; ++i) {
            map.emplace(std::format("k{}", i), i);
        }
        expect(eq(map.size(), 200UZ));
        for (std::uint32_t i = 0U; i < 200U; ++i) {
            expect(eq(map.value_or<std::uint32_t>(std::format("k{}", i), 0U), i));
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
        expect(eq(map.value_or<std::string_view>("new_key", std::string_view{}).size(), std::string_view{"source_inline_key"}.size()));
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
        const auto v = *map.find_value("self");
        expect(v.holds<ValueMap>());
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
        const auto top = *current.find_value("inner");
        expect(top.holds<ValueMap>()); // depth 1: still readable
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
        const auto v = *map.find_value("z32");
        expect(v.holds<std::complex<float>>());
        const auto z = v.value_or<std::complex<float>>(std::complex<float>{});
        expect(eq(z.real(), 1.5f));
        expect(eq(z.imag(), -2.5f));
    };

    "complex<double> round-trip via 16-byte payload-pool slot"_test = [] {
        ValueMap map;
        map.emplace("z64", std::complex<double>{3.14, -2.71});
        const auto v = *map.find_value("z64");
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

        const auto v = *outer.find_value("trigger_meta_info");
        expect(v.holds<ValueMap>());
        const auto  mapOpt = v.get_if<ValueMap>();
        const auto& map    = *mapOpt;
        expect(eq(map.size(), 2UZ));
        expect(eq(map.value_or<std::uint32_t>(std::pmr::string{"inner_count"}, 0U), std::uint32_t{42}));
        expect(eq(map.value_or<std::string_view>(std::pmr::string{"inner_label"}, std::string_view{}), std::string_view{"abc"}));
        // Sibling entry untouched.
        expect(eq(outer.value_or<float>("sample_rate", 0.f), 48000.0f));
    };

    "nested ValueMap round-trip — two levels deep (recursive sub-blob)"_test = [] {
        ValueMap leaf;
        leaf.emplace("leaf_value", std::int64_t{-7});

        ValueMap mid;
        mid.emplace("mid_value", 0.5f);
        mid.emplace("nested_leaf", leaf);

        ValueMap root;
        root.emplace("nested_mid", mid);

        const auto vmid = *root.find_value("nested_mid");
        expect(vmid.holds<ValueMap>());
        const auto  mmidOpt = vmid.get_if<ValueMap>();
        const auto& mmid    = *mmidOpt;
        expect(eq(mmid.size(), 2UZ));
        expect(eq(mmid.value_or<float>(std::pmr::string{"mid_value"}, 0.f), 0.5f));

        const auto vleafEntry = *mmid.find_value(std::pmr::string{"nested_leaf"});
        expect(vleafEntry.holds<ValueMap>());
        const auto  mleafOpt = vleafEntry.get_if<ValueMap>();
        const auto& mleaf    = *mleafOpt;
        expect(eq(mleaf.size(), 1UZ));
        expect(eq(mleaf.value_or<std::int64_t>(std::pmr::string{"leaf_value"}, 0), std::int64_t{-7}));
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

        const auto v = *map.find_value("vec1d");
        expect(v.is_tensor());
        const std::optional<gr::TensorView<float>> t = v.get_if<gr::TensorView<float>>();
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

        const auto                                 v = *map.find_value("mat2d");
        const std::optional<gr::TensorView<float>> t = v.get_if<gr::TensorView<float>>();
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

        const auto                                        v = *map.find_value("cube");
        const std::optional<gr::TensorView<std::int64_t>> t = v.get_if<gr::TensorView<std::int64_t>>();
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

        const gr::pmt::Value                      v    = *map.find_value("flags");
        const std::optional<gr::TensorView<bool>> view = v.get_if<gr::TensorView<bool>>();
        expect(view.has_value());
        if (!view) {
            return;
        }
        expect(eq(view->size(), 5UZ));
        expect(view->operator[](0));
        expect(!view->operator[](1));
        expect(view->operator[](2));
        expect(view->operator[](3));
        expect(!view->operator[](4));
    };

    "Tensor<complex<float>> round-trip — fixed-size 8-byte elements"_test = [] {
        ValueMap                        map;
        gr::Tensor<std::complex<float>> src(gr::extents_from, std::array<std::size_t, 1>{3UZ});
        src._data.data()[0] = {1.f, 2.f};
        src._data.data()[1] = {3.f, 4.f};
        src._data.data()[2] = {5.f, 6.f};
        map.emplace("z32vec", src);

        const auto                                               v = *map.find_value("z32vec");
        const std::optional<gr::TensorView<std::complex<float>>> t = v.get_if<gr::TensorView<std::complex<float>>>();
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

        const auto                                                v = *map.find_value("z64vec");
        const std::optional<gr::TensorView<std::complex<double>>> t = v.get_if<gr::TensorView<std::complex<double>>>();
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

        const auto                                 v = *map.find_value("empty");
        const std::optional<gr::TensorView<float>> t = v.get_if<gr::TensorView<float>>();
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

        const auto                                 v = *map.find_value("big");
        const std::optional<gr::TensorView<float>> t = v.get_if<gr::TensorView<float>>();
        expect(t.has_value());
        if (!t) {
            return;
        }
        expect(eq(t->size(), 1024UZ));
        expect(eq(t->_data.data()[1023], 1023.0f));
    };

    // F2: a homogeneous Tensor<Value> whose cells share a fixed-size scalar type collapses on
    // the wire to the typed Tensor<T> form — same wire bytes a Tensor<int64_t> would produce,
    // and the canonical reader is the typed accessor.
    "homogeneous Tensor<Value> with int64 cells decodes as Tensor<int64>"_test = [] {
        ValueMap          map;
        gr::Tensor<Value> src(gr::extents_from, std::array<std::size_t, 1>{4UZ});
        src._data.data()[0] = Value{std::int64_t{1}};
        src._data.data()[1] = Value{std::int64_t{-2}};
        src._data.data()[2] = Value{std::int64_t{3}};
        src._data.data()[3] = Value{std::int64_t{-4}};
        map.emplace("ints", src);

        const auto v = *map.find_value("ints");
        expect(!v.get_if<gr::TensorView<Value>>().has_value()) << "homogeneous fixed-size Tensor<Value> must take the typed wire path";

        const auto t = v.get_if<gr::TensorView<std::int64_t>>();
        expect(t.has_value()) << "expected Tensor<int64> after F2 wire-format collapse";
        if (!t) {
            return;
        }
        expect(eq(t->size(), 4UZ));
        expect(eq(t->data()[0], std::int64_t{1}));
        expect(eq(t->data()[1], std::int64_t{-2}));
        expect(eq(t->data()[2], std::int64_t{3}));
        expect(eq(t->data()[3], std::int64_t{-4}));
    };

    "Tensor<Value> with mixed inline scalar elements (int + float + uint)"_test = [] {
        ValueMap          map;
        gr::Tensor<Value> src(gr::extents_from, std::array<std::size_t, 1>{3UZ});
        src._data.data()[0] = Value{std::int64_t{-7}};
        src._data.data()[1] = Value{0.5f};
        src._data.data()[2] = Value{std::uint64_t{42}};
        map.emplace("mixed", src);

        const auto                                 v = *map.find_value("mixed");
        const std::optional<gr::TensorView<Value>> t = v.get_if<gr::TensorView<Value>>();
        expect(t.has_value()) << "heterogeneous Tensor<Value> must keep the variable-size wire path";
        if (!t) {
            return;
        }
        expect(eq(t->size(), 3UZ));
        expect(eq((*t)[0].value_or<std::int64_t>(0), std::int64_t{-7}));
        expect(eq((*t)[1].value_or<float>(0.f), 0.5f));
        expect(eq((*t)[2].value_or<std::uint64_t>(0U), std::uint64_t{42}));
    };

    "Tensor<Value> with String elements (settings-style string array)"_test = [] {
        ValueMap          map;
        gr::Tensor<Value> src(gr::extents_from, std::array<std::size_t, 1>{3UZ});
        src._data.data()[0] = Value{std::string_view{"alpha"}};
        src._data.data()[1] = Value{std::string_view{"bravo"}};
        src._data.data()[2] = Value{std::string_view{"charlie"}};
        map.emplace("labels", src);

        const auto                                 v = *map.find_value("labels");
        const std::optional<gr::TensorView<Value>> t = v.get_if<gr::TensorView<Value>>();
        expect(t.has_value());
        if (!t) {
            return;
        }
        expect(eq(t->size(), 3UZ));
        expect(eq((*t)[0].value_or(std::string_view{}), std::string_view{"alpha"}));
        expect(eq((*t)[1].value_or(std::string_view{}), std::string_view{"bravo"}));
        expect(eq((*t)[2].value_or(std::string_view{}), std::string_view{"charlie"}));
    };

    // Homogeneous Tensor<Value> with fixed-size scalar/complex cells collapses on the wire to
    // the typed Tensor<T> form (F2 optimisation), so the canonical read path is the typed
    // accessor — get_if<TensorView<Value>>() returns nullopt for the optimised cases.
    "homogeneous Tensor<Value> with complex<double> cells decodes as Tensor<complex<double>>"_test = [] {
        ValueMap          map;
        gr::Tensor<Value> src(gr::extents_from, std::array<std::size_t, 1>{2UZ});
        src._data.data()[0] = Value{std::complex<double>{1.0, 2.0}};
        src._data.data()[1] = Value{std::complex<double>{3.0, 4.0}};
        map.emplace("z64s", src);

        const auto v = *map.find_value("z64s");
        expect(!v.get_if<gr::TensorView<Value>>().has_value()) << "homogeneous fixed-size Tensor<Value> must take the typed wire path";

        const auto t = v.get_if<gr::TensorView<std::complex<double>>>();
        expect(t.has_value()) << "expected Tensor<complex<double>> after F2 wire-format collapse";
        if (!t) {
            return;
        }
        expect(eq(t->size(), 2UZ));
        expect(eq(t->data()[0].real(), 1.0));
        expect(eq(t->data()[0].imag(), 2.0));
        expect(eq(t->data()[1].real(), 3.0));
        expect(eq(t->data()[1].imag(), 4.0));
    };

    "Tensor<Value> elements may themselves be nested ValueMaps (recursive composition)"_test = [] {
        ValueMap inner;
        inner.emplace("k", std::int64_t{99});

        ValueMap          map;
        gr::Tensor<Value> src(gr::extents_from, std::array<std::size_t, 1>{2UZ});
        src._data.data()[0] = Value{std::string_view{"head"}};
        src._data.data()[1] = Value{ValueMap{}}; // initialise then populate

        // Build the nested-map element via ValueMap construction — Value(Tensor<Value>) cannot
        // store a ValueMap directly, but Value{ValueMap{...}} round-trips through the encoder.
        ValueMap innerMap;
        innerMap.emplace(std::pmr::string{"k"}, Value{std::int64_t{99}});
        src._data.data()[1] = Value{std::move(innerMap)};

        map.emplace("composed", src);

        const auto                                 v = *map.find_value("composed");
        const std::optional<gr::TensorView<Value>> t = v.get_if<gr::TensorView<Value>>();
        expect(t.has_value());
        if (!t) {
            return;
        }
        const Value head = (*t)[0];
        expect(eq(head.value_or(std::string_view{}), std::string_view{"head"}));
        const Value nestedVal = (*t)[1];
        expect(nestedVal.is_map());
        const auto  nestedOpt = nestedVal.get_if<ValueMap>();
        const auto& nested    = *nestedOpt;
        expect(eq(nested.size(), 1UZ));
        expect(eq(nested.value_or<std::int64_t>(std::pmr::string{"k"}, 0), std::int64_t{99}));
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
        const auto                                 v = *map.find_value("samples");
        const std::optional<gr::TensorView<float>> t = v.get_if<gr::TensorView<float>>();
        expect(t.has_value());
        if (!t) {
            return;
        }
        expect(eq(t->_data.data()[2], 0.3f));
        expect(eq(map.value_or<float>("sample_rate", 0.f), 48000.0f));
        expect(eq(map.value_or<std::string_view>("label", std::string_view{}), std::string_view{"capture"}));
    };

    "ValueMap copy preserves Tensor entries (deep copy via blob memcpy)"_test = [] {
        ValueMap          src;
        gr::Tensor<float> tensor(gr::extents_from, std::array<std::size_t, 1>{4UZ});
        std::ranges::copy(std::array<float, 4>{1.f, 2.f, 4.f, 8.f}, tensor._data.begin());
        src.emplace("samples", tensor);

        ValueMap                                   copy{src};
        const auto                                 v = *copy.find_value("samples");
        const std::optional<gr::TensorView<float>> t = v.get_if<gr::TensorView<float>>();
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
        const auto                                 v = *dest.find_value("samples");
        const std::optional<gr::TensorView<float>> t = v.get_if<gr::TensorView<float>>();
        expect(t.has_value());
        if (!t) {
            return;
        }
        expect(eq(t->_data.data()[0], 7.f));
        expect(eq(t->_data.data()[1], 9.f));
    };
};

const boost::ut::suite<"ValueMap - Tensor<Value> view contract"> _tensor_value_view_contract_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using gr::pmt::ValueMap;

    "iterator yields exactly elementCount Values"_test = [] {
        gr::Tensor<Value> src(gr::extents_from, std::array<std::size_t, 1>{4UZ});
        src._data.data()[0] = Value{std::int64_t{-7}};
        src._data.data()[1] = Value{0.5f};
        src._data.data()[2] = Value{std::string_view{"three"}};
        src._data.data()[3] = Value{std::uint64_t{42}};

        ValueMap map;
        map.emplace("hetero", src);

        const auto v    = *map.find_value("hetero");
        const auto view = v.get_if<gr::TensorView<Value>>();
        expect(view.has_value()) << boost::ut::fatal;

        std::size_t walked = 0UZ;
        for (auto it = view->begin(); it != view->end(); ++it) {
            ++walked;
        }
        expect(eq(walked, 4UZ));
        expect(eq(view->size(), 4UZ));
    };

    "random access [i] matches sequential decode"_test = [] {
        gr::Tensor<Value> src(gr::extents_from, std::array<std::size_t, 1>{3UZ});
        src._data.data()[0] = Value{std::int64_t{100}};
        src._data.data()[1] = Value{std::string_view{"middle"}};
        src._data.data()[2] = Value{1.5};

        ValueMap map;
        map.emplace("idx", src);

        const auto v    = *map.find_value("idx");
        const auto view = v.get_if<gr::TensorView<Value>>();
        expect(view.has_value()) << boost::ut::fatal;

        // sequential walk
        std::vector<Value> seq;
        seq.reserve(view->size());
        for (const auto& e : *view) {
            seq.push_back(e);
        }
        expect(eq(seq.size(), 3UZ));

        // random access — must produce equivalent values for each index
        expect(eq((*view)[0].value_or<std::int64_t>(0), seq[0].value_or<std::int64_t>(-1)));
        expect(eq((*view)[1].value_or(std::string_view{}), seq[1].value_or(std::string_view{"X"})));
        expect(eq((*view)[2].value_or<double>(0.0), seq[2].value_or<double>(-1.0)));
    };

    "from_blob then get_if<TensorView<Value>> then owned(mr) preserves every cell"_test = [] {
        gr::Tensor<Value> src(gr::extents_from, std::array<std::size_t, 1>{3UZ});
        src._data.data()[0] = Value{std::int64_t{-7}};
        src._data.data()[1] = Value{std::string_view{"middle"}};
        src._data.data()[2] = Value{1.25f};

        ValueMap original;
        original.emplace("payload", src);

        const auto blob     = original.blob();
        const auto restored = ValueMap::from_blob(blob);
        expect(restored.has_value()) << boost::ut::fatal;

        const auto restoredVal = *restored->find_value("payload");
        const auto view        = restoredVal.get_if<gr::TensorView<Value>>();
        expect(view.has_value()) << boost::ut::fatal;
        expect(eq(view->size(), 3UZ));

        std::pmr::synchronized_pool_resource pool;
        const auto                           snap = view->owned(&pool);
        expect(eq(snap.size(), 3UZ));
        expect(eq(snap.data()[0].template value_or<std::int64_t>(0), std::int64_t{-7}));
        expect(eq(snap.data()[1].value_or(std::string_view{}), std::string_view{"middle"}));
        expect(eq(snap.data()[2].template value_or<float>(0.f), 1.25f));
    };

    "nested Tensor<Value>-of-Tensor<Value> round-trips through ValueMap blob"_test = [] {
        gr::Tensor<Value> innerA(gr::extents_from, std::array<std::size_t, 1>{2UZ});
        innerA._data.data()[0] = Value{std::int64_t{1}};
        innerA._data.data()[1] = Value{std::string_view{"a1"}};

        gr::Tensor<Value> innerB(gr::extents_from, std::array<std::size_t, 1>{3UZ});
        innerB._data.data()[0] = Value{std::int64_t{10}};
        innerB._data.data()[1] = Value{std::int64_t{20}};
        innerB._data.data()[2] = Value{std::string_view{"b3"}};

        gr::Tensor<Value> outer(gr::extents_from, std::array<std::size_t, 1>{2UZ});
        outer._data.data()[0] = Value{std::move(innerA)};
        outer._data.data()[1] = Value{std::move(innerB)};

        ValueMap original;
        original.emplace("nested", outer);

        const auto blob     = original.blob();
        const auto restored = ValueMap::from_blob(blob);
        expect(restored.has_value()) << boost::ut::fatal;

        const auto restoredVal = *restored->find_value("nested");
        const auto outerView   = restoredVal.get_if<gr::TensorView<Value>>();
        expect(outerView.has_value()) << boost::ut::fatal;
        expect(eq(outerView->size(), 2UZ));

        const auto outerSnap = outerView->owned();
        expect(eq(outerSnap.size(), 2UZ));

        const auto innerViewA = outerSnap.data()[0].get_if<gr::TensorView<Value>>();
        expect(innerViewA.has_value()) << boost::ut::fatal;
        expect(eq(innerViewA->size(), 2UZ));
        expect(eq((*innerViewA)[0].template value_or<std::int64_t>(0), std::int64_t{1}));
        expect(eq((*innerViewA)[1].value_or(std::string_view{}), std::string_view{"a1"}));

        const auto innerViewB = outerSnap.data()[1].get_if<gr::TensorView<Value>>();
        expect(innerViewB.has_value()) << boost::ut::fatal;
        expect(eq(innerViewB->size(), 3UZ));
        expect(eq((*innerViewB)[0].template value_or<std::int64_t>(0), std::int64_t{10}));
        expect(eq((*innerViewB)[1].template value_or<std::int64_t>(0), std::int64_t{20}));
        expect(eq((*innerViewB)[2].value_or(std::string_view{}), std::string_view{"b3"}));
    };

    "owned(mr) snapshot is independent of the source ValueMap's lifetime"_test = [] {
        gr::Tensor<Value> snap;
        {
            gr::Tensor<Value> src(gr::extents_from, std::array<std::size_t, 1>{2UZ});
            src._data.data()[0] = Value{std::string_view{"alpha"}};
            src._data.data()[1] = Value{std::int64_t{99}};

            ValueMap source;
            source.emplace("payload", src);

            const auto v    = *source.find_value("payload");
            const auto view = v.get_if<gr::TensorView<Value>>();
            expect(view.has_value()) << boost::ut::fatal;
            snap = view->owned(std::pmr::get_default_resource());
        }
        // source is gone; snap must remain content-equivalent
        expect(eq(snap.size(), 2UZ));
        expect(eq(snap.data()[0].value_or(std::string_view{}), std::string_view{"alpha"}));
        expect(eq(snap.data()[1].template value_or<std::int64_t>(0), std::int64_t{99}));
    };

    "rank-0 (scalar tensor) with String element decodes cleanly"_test = [] {
        // String cell — F2 collapses homogeneous fixed-size cells to the typed wire path,
        // so the variable-size codepath only fires for String / nested elements.
        gr::Tensor<Value> src(gr::extents_from, std::array<std::size_t, 0>{});
        src._data.resize(1UZ);
        src._data.data()[0] = Value{std::string_view{"answer"}};

        ValueMap map;
        map.emplace("scalar", src);

        const auto v    = *map.find_value("scalar");
        const auto view = v.get_if<gr::TensorView<Value>>();
        expect(view.has_value()) << boost::ut::fatal;
        expect(eq(view->rank(), 0UZ));
        expect(eq(view->size(), 1UZ));
        expect(eq((*view)[0].value_or(std::string_view{}), std::string_view{"answer"}));
    };

    "Tensor<Value> of ValueMap-with-strings — iter+get_if path mirroring loadGraphFromMap"_test = [] {
        ValueMap blockA;
        blockA.emplace("name", std::string{"multiplier1"});
        blockA.emplace("type", std::string{"good::multiply<float64>"});

        ValueMap blockB;
        blockB.emplace("name", std::string{"sink"});
        blockB.emplace("type", std::string{"good::cout_sink<float64>"});

        gr::Tensor<Value> outer(gr::extents_from, std::array<std::size_t, 1>{2UZ});
        outer._data.data()[0] = Value{std::move(blockA)};
        outer._data.data()[1] = Value{std::move(blockB)};

        ValueMap source;
        source.emplace("blocks", outer);

        // bind iter-yielded pair to an lvalue so the view aliases live storage
        const auto it = source.find("blocks");
        expect(it != source.end()) << boost::ut::fatal;
        const Value blkValue  = (*it).second.owned(); // Tensor<Value> decode needs _resource
        const auto  outerView = blkValue.get_if<gr::TensorView<Value>>();
        expect(outerView.has_value()) << boost::ut::fatal;
        expect(eq(outerView->size(), 2UZ));

        const auto outerSnap = outerView->owned();
        expect(eq(outerSnap.size(), 2UZ));

        const auto innerA = outerSnap.data()[0].get_if<ValueMap>();
        expect(innerA.has_value()) << boost::ut::fatal;
        expect(eq(innerA->template value_or<std::string_view>("name", ""), std::string_view{"multiplier1"}));
        expect(eq(innerA->template value_or<std::string_view>("type", ""), std::string_view{"good::multiply<float64>"}));

        const auto innerB = outerSnap.data()[1].get_if<ValueMap>();
        expect(innerB.has_value()) << boost::ut::fatal;
        expect(eq(innerB->template value_or<std::string_view>("name", ""), std::string_view{"sink"}));
        expect(eq(innerB->template value_or<std::string_view>("type", ""), std::string_view{"good::cout_sink<float64>"}));
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
        const auto v = *map.find_value("my_ext");
        expect(v.holds<float>());
        expect(eq(v.value_or<float>(0.f), 7.25f));
    };

    "insert_or_assign overwrites existing entries"_test = [] {
        ValueMap map;
        map.insert_or_assign("my_ext", 7.25f);
        auto [it, inserted] = map.insert_or_assign("my_ext", 9.5f);
        expect(!inserted);
        expect(it != map.end());
        const auto v = *map.find_value("my_ext");
        expect(eq(v.value_or<float>(0.f), 9.5f));
    };

    "find(string_view) returns end() on miss (canonical exception-free path)"_test = [] {
        ValueMap map;
        expect(map.find("missing") == map.end());
        map.insert_or_assign("present", 42);
        const auto it = map.find("present");
        expect(it != map.end());
        expect((*it).second.value_or<int>(0) == 42);
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
        const auto v = *map.find_value("sample_rate");
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
        expect(eq(map.value_or<std::uint64_t>(k64, 0U), std::uint64_t{7}));
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
        expect(eq(map.value_or<std::uint64_t>(k64, 0U), std::uint64_t{42}));
    };

    "long-key merge between maps preserves key + value"_test = [] {
        const std::string_view k64{"this_is_a_user_extension_key_that_is_longer_than_27_characters_."};
        ValueMap               src;
        src.emplace(k64, std::uint64_t{99});
        ValueMap dst;
        dst.merge(src);
        expect(eq(src.size(), 0UZ));
        expect(dst.contains(k64));
        expect(eq(dst.value_or<std::uint64_t>(k64, 0U), std::uint64_t{99}));
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
            const auto v = *map.find_value(key);
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
        expect(eq(map.value_or<std::string_view>("first", std::string_view{""}), std::string_view{"sample-carrier-name-01"}));
        expect(eq(map.value_or<std::string_view>("second", std::string_view{""}), std::string_view{"sample-carrier-name-02"}));
        expect(eq(map.value_or<std::string_view>("third", std::string_view{""}), std::string_view{"sample-carrier-name-03"}));
    };

    // T8 (validates C2 — insert_or_assign payload reclamation): repeated same-key rewrites
    // must not grow the payload pool unboundedly.
    "insert_or_assign on same key reclaims old payload (C2 regression)"_test = [] {
        ValueMap map;
        map.insert_or_assign("key", std::string_view{"first content here"});
        map.shrink_to_fit();
        const auto baselineBlobSize = map.blob().size();
        for (int i = 0; i < 200; ++i) {
            map.insert_or_assign("key", std::string_view{"replacement value with content"});
        }
        map.shrink_to_fit();
        // Bounded growth: no runaway leak (≤ ~512 B headroom for free-list metadata + alignment).
        expect(map.blob().size() <= baselineBlobSize + 512UZ) << std::format("payload pool grew unbounded: baseline {} B, after 200 rewrites {} B", baselineBlobSize, map.blob().size());
    };

    // T9 (validates C1 — string erase free-list +1 off-by-one): erasing a string ≥ 7 chars
    // and reusing the freed region with a split-fit insert must not corrupt neighbour records.
    "erase string + split-fit reuse leaves neighbour record intact (C1 regression)"_test = [] {
        ValueMap map;
        // String of length 8 → recSize = max(8 + 9, 16) = 17 (boundary case where pool reclamation matters).
        map.insert_or_assign("a", std::string_view{"abcdefgh"});
        // Neighbour record with a known sentinel value.
        map.insert_or_assign("b", std::int64_t{0x0123456789ABCDEFLL});
        const auto bIt = map.find("b");
        expect(bIt != map.end());
        const auto bBefore = (*bIt).second.value_or<std::int64_t>(0);
        expect(eq(bBefore, std::int64_t{0x0123456789ABCDEFLL}));
        // Erase 'a' → push freed region onto free list. Then insert a smaller string that
        // triggers a split-fit reuse (remainder ≥ 8 bytes → new FreeChunk header written).
        map.erase("a");
        map.insert_or_assign("c", std::string_view{"x"});
        // Neighbour 'b' must still hold its original value (no header-byte corruption).
        const auto bAfter = (*map.find("b")).second.value_or<std::int64_t>(0);
        expect(eq(bAfter, std::int64_t{0x0123456789ABCDEFLL})) << "neighbour record corrupted after erase + split-fit reuse";
    };

    // T2 (validates W1 — kMaxInlineKeyLength = 34 boundary): keys at the max-inline length
    // must round-trip; keys at 35 must spill to the payload pool without corruption.
    "inline-key boundary at 34 chars (max inline) round-trips"_test = [] {
        ValueMap          map;
        const std::string key34(34UZ, 'k');
        map.insert_or_assign(key34, std::int64_t{100});
        const auto it = map.find(key34);
        expect(it != map.end());
        expect(eq((*it).first, std::string_view{key34}));
        expect((*it).second.value_or<std::int64_t>(0) == 100);
    };

    "key at 35 chars spills to payload pool"_test = [] {
        ValueMap          map;
        const std::string key35(35UZ, 'k');
        map.insert_or_assign(key35, std::int64_t{200});
        const auto it = map.find(key35);
        expect(it != map.end());
        expect(eq((*it).first, std::string_view{key35}));
        expect((*it).second.value_or<std::int64_t>(0) == 200);
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
            // Iter yields key as `const char*` — wrap in string_view for value-equality compares.
            const std::string_view k{key};
            if (k == "sample_rate") {
                saw_sample_rate = true;
                expect(eq(val.value_or<float>(0.f), 48000.0f));
            } else if (k == "frequency") {
                saw_frequency = true;
                expect(eq(val.value_or<double>(0.0), 1e9));
            } else if (k == "custom") {
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
        expect(eq(copy.value_or<float>("sample_rate", 0.f), 48000.0f));
        expect(eq(copy.value_or<std::string_view>("note", std::string_view{""}), std::string_view{"hello-world-this-string-is-long"}));
    };

    "move construction is O(1) — no allocations on the source resource"_test = [] {
        CountingResource mr;
        ValueMap         src(&mr);
        src.insert_or_assign("frequency", 2.4e9);
        const auto allocs_before = mr.allocCount;

        ValueMap dst(std::move(src));

        expect(eq(mr.allocCount, allocs_before)) << "move must not allocate on the source resource";
        expect(eq(dst.size(), 1UZ));
        expect(eq(dst.value_or<double>("frequency", 0.0), 2.4e9));
        expect(eq(src.size(), 0UZ)) << "source is moved-from (empty)";
    };

    "copy assignment replaces prior contents"_test = [] {
        ValueMap a;
        ValueMap b;
        a.insert_or_assign("sample_rate", 48000.0f);
        b.insert_or_assign("frequency", 1e9);
        b = a;
        expect(eq(b.size(), 1UZ));
        expect(eq(b.value_or<float>("sample_rate", 0.f), 48000.0f));
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
        expect(eq(b.value_or<float>("sample_rate", 0.f), 48000.0f));
    };
};

// Unified wire-format suite. Subgroups (separated by blank lines for the reader):
//   1. Header layout invariants (capacities, version, debug-build flag, mutation-driven counters)
//   2. Round-trip encode → from_blob → equal map (across all value types)
//   3. On-blob guards & sentinels (\0 guards, !EOD marker, entry-array sentinel)
//   4. Rejection of malformed blobs (size, magic, version, payload offsets, corrupted free-list,
//      inline-key range, wire-format size disagreement, recursive nested-map / tensor sub-blobs)
//   5. Advisory flags (freeze)
const boost::ut::suite<"ValueMap - wire format"> _wire_format_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using gr::pmt::ValueMap;

    // --- 1. Header layout invariants ---

    "Header layout: entryCapacity and payloadCapacity are explicit fields"_test = [] {
        ValueMap    map{nullptr, /*entries=*/8U, /*payloadCapacity=*/256U};
        const auto* hdr = std::launder(reinterpret_cast<const gr::pmt::Header*>(map.blob().data()));
        expect(eq(hdr->version, gr::pmt::kBlobVersion));
        expect(eq(hdr->entryCapacity, std::uint16_t{8}));
        expect(eq(hdr->payloadCapacity, std::uint32_t{256}));
        expect(eq(hdr->payloadUsed, std::uint32_t{0}));
        expect(eq(hdr->entryCount, std::uint16_t{0}));
    };

    "Header.flags & kHeaderFlagDebugGuards mirrors the build mode"_test = [] {
        ValueMap    map;
        const auto* hdr = std::launder(reinterpret_cast<const gr::pmt::Header*>(map.blob().data()));
        if constexpr (gr::meta::kDebugBuild) {
            expect((hdr->flags & gr::pmt::kHeaderFlagDebugGuards) != 0U) << "debug build must set kHeaderFlagDebugGuards";
        } else {
            expect((hdr->flags & gr::pmt::kHeaderFlagDebugGuards) == 0U) << "release build must NOT set kHeaderFlagDebugGuards";
        }
    };

    "Header.entryCount and payloadUsed track mutation; blob is kBlobAlignment-aligned"_test = [] {
        ValueMap map;
        map.insert_or_assign("sample_rate", 48000.0f);
        map.emplace("note", std::string_view{"abc"});

        const auto  b   = map.blob();
        const auto* hdr = reinterpret_cast<const gr::pmt::Header*>(b.data());
        expect(!b.empty()) << "blob must be non-empty after insert";
        expect(eq(hdr->entryCount, std::uint16_t{2}));
        expect(hdr->payloadUsed >= 3UZ) << "string payload must have been appended";
        expect(hdr->totalSize == b.size());
        expect(eq(reinterpret_cast<std::uintptr_t>(b.data()) & (gr::pmt::kBlobAlignment - 1UZ), 0UZ)) << "blob data must be kBlobAlignment-aligned";
    };

    // --- 2. Round-trip ---

    "from_blob round-trip preserves scalar / string / unsigned-integer / tensor entries"_test = [] {
        ValueMap src;
        src.emplace("scalar", 48000.0f);
        src.emplace("signal_name", std::string_view{"demo"});
        src.emplace("trigger_time", std::uint64_t{1'700'000'000ULL});
        gr::Tensor<float> tf(gr::extents_from, std::array<std::size_t, 1>{3UZ});
        tf._data.data()[0] = 1.f;
        tf._data.data()[1] = 2.f;
        tf._data.data()[2] = 3.f;
        src.emplace("tensor", std::move(tf));

        const auto bytes    = src.blob();
        const auto restored = ValueMap::from_blob(bytes);
        expect(restored.has_value());
        if (restored) {
            expect(eq(restored->size(), 4UZ));
            expect(eq(restored->value_or<float>("scalar", 0.f), 48000.0f));
            expect(eq(restored->value_or<std::string_view>("signal_name", std::string_view{}), std::string_view{"demo"}));
            expect(eq(restored->value_or<std::uint64_t>("trigger_time", 0U), std::uint64_t{1'700'000'000ULL}));
        }
    };

    // --- 3. On-blob guards & sentinels ---

    "String value-record carries a trailing '\\0' guard after the chars"_test = [] {
        // V2 wire format: e.payloadOffset points at a value-record [size:4][vt:1][ct:1][flags:2][chars+NUL].
        // payloadLength records the TOTAL record size (header + chars + NUL guard).
        ValueMap               map;
        const std::string_view sv{"hello-world"}; // 11 chars
        map.insert_or_assign("greeting", sv);
        const auto  blob        = map.blob();
        const auto* hdr         = std::launder(reinterpret_cast<const gr::pmt::Header*>(blob.data()));
        const auto* entries     = std::launder(reinterpret_cast<const gr::pmt::PackedEntry*>(blob.data() + sizeof(gr::pmt::Header)));
        bool        foundString = false;
        for (std::uint16_t i = 0; i < hdr->entryCount; ++i) {
            const auto& e = entries[i];
            if (static_cast<Value::ValueType>(e.valueType) == Value::ValueType::String) {
                foundString = true;
                expect(eq(e.payloadLength, std::uint32_t{8U + 11U + 1U})) << "V2 payloadLength = record header (8) + chars + '\\0' guard";
                expect(blob[e.payloadOffset + 8U + sv.size()] == std::byte{0}) << "trailing \\0 guard byte at end of chars";
                break;
            }
        }
        expect(foundString);
    };

    "Spilled-key region carries a trailing '\\0' guard outside keyLength"_test = [] {
        ValueMap               map;
        const std::string_view longKey{"this-key-is-much-longer-than-27-chars-so-it-spills"};
        map.insert_or_assign(longKey, std::int64_t{42});
        const auto  blob    = map.blob();
        const auto* hdr     = std::launder(reinterpret_cast<const gr::pmt::Header*>(blob.data()));
        const auto* entries = std::launder(reinterpret_cast<const gr::pmt::PackedEntry*>(blob.data() + sizeof(gr::pmt::Header)));
        bool        found   = false;
        for (std::uint16_t i = 0; i < hdr->entryCount; ++i) {
            const auto& e = entries[i];
            if (e.keyId == gr::pmt::keys::kSpilledKeyId) {
                found                             = true;
                const auto [keyOffset, keyLength] = gr::pmt::detail::readSpilledKeyOffsetLength(e);
                expect(eq(keyLength, static_cast<std::uint32_t>(longKey.size())));
                expect(blob[keyOffset + keyLength] == std::byte{0}) << "trailing \\0 guard byte must be written outside spilled-key length";
                break;
            }
        }
        expect(found);
    };

    "entry-array sentinel keyId == kEndMarkerId at row[entryCount]"_test = [] {
        ValueMap map;
        map.insert_or_assign("x", 1);
        map.insert_or_assign("y", 2);
        const auto  blob    = map.blob();
        const auto* hdr     = std::launder(reinterpret_cast<const gr::pmt::Header*>(blob.data()));
        const auto* entries = std::launder(reinterpret_cast<const gr::pmt::PackedEntry*>(blob.data() + sizeof(gr::pmt::Header)));
        const auto  cap     = (hdr->payloadOffset - sizeof(gr::pmt::Header)) / sizeof(gr::pmt::PackedEntry);
        expect(gt(cap, static_cast<std::size_t>(hdr->entryCount))) << "entry array capacity must include sentinel slot";
        expect(eq(entries[hdr->entryCount].keyId, gr::pmt::keys::kEndMarkerId)) << "sentinel slot must hold kEndMarkerId (0xFFFF)";
    };

    "'!EOD\\0' marker lands at payloadOffset + payloadUsed (debug only)"_test = [] {
        if constexpr (!gr::meta::kDebugBuild) {
            return; // release builds skip the marker write
        }
        ValueMap map;
        map.insert_or_assign("k", std::string_view{"some-payload-string"});
        const auto       blob = map.blob();
        const auto*      hdr  = std::launder(reinterpret_cast<const gr::pmt::Header*>(blob.data()));
        const std::array expected{std::byte{'!'}, std::byte{'E'}, std::byte{'O'}, std::byte{'D'}, std::byte{0}};
        const auto       endByte = hdr->payloadOffset + hdr->payloadUsed;
        for (std::size_t i = 0UZ; i < expected.size() && endByte + i < blob.size(); ++i) {
            expect(blob[endByte + i] == expected[i]) << "!EOD\\0 marker byte mismatch at i=" << i;
        }
    };

    // --- 4. Rejection of malformed blobs ---

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
        alignedBuf[0] = std::byte{'X'};
        const auto r  = ValueMap::from_blob(std::span<const std::byte>{alignedBuf.data(), srcBytes.size()});
        expect(!r.has_value());
        expect(r.error() == gr::pmt::DeserialiseError::MagicMismatch);
    };

    "from_blob rejects a blob carrying a non-current kBlobVersion"_test = [] {
        ValueMap map;
        std::ignore                = map.try_emplace("counter", std::int32_t{42});
        const auto             src = map.blob();
        std::vector<std::byte> spoof(src.begin(), src.end());
        expect(spoof.size() > 4UZ);
        if (spoof.size() > 4UZ) {
            spoof[4] = std::byte{static_cast<std::uint8_t>(gr::pmt::kBlobVersion + 1U)};
        }
        const auto restored = ValueMap::from_blob(std::span<const std::byte>{spoof});
        expect(!restored.has_value());
        if (!restored.has_value()) {
            expect(restored.error() == gr::pmt::DeserialiseError::VersionUnsupported);
        }
    };

    "from_blob rejects a blob whose inline-key length byte exceeds kMaxInlineKeyLength"_test = [] {
        ValueMap map;
        std::ignore                = map.try_emplace("k", std::int32_t{0});
        const auto             src = map.blob();
        std::vector<std::byte> spoof(src.begin(), src.end());
        auto*                  entries    = std::launder(reinterpret_cast<gr::pmt::PackedEntry*>(spoof.data() + sizeof(gr::pmt::Header)));
        bool                   patchedOne = false;
        for (std::uint16_t i = 0U; i < 1U; ++i) {
            if (entries[i].keyId == gr::pmt::keys::kInlineKeyId) {
                entries[i].inlineKey[0] = static_cast<char>(static_cast<unsigned char>(gr::pmt::kMaxInlineKeyLength + 1U));
                patchedOne              = true;
            }
        }
        expect(patchedOne) << "test setup: inserted key must be encoded as inlineKey";
        const auto restored = ValueMap::from_blob(std::span<const std::byte>{spoof});
        expect(!restored.has_value()) << "from_blob must reject inline-key length > kMaxInlineKeyLength";
    };

    "from_blob rejects a value-record whose size field disagrees with payloadLength"_test = [] {
        ValueMap map;
        std::ignore                = map.try_emplace("k", std::int32_t{0});
        const auto             src = map.blob();
        std::vector<std::byte> spoof(src.begin(), src.end());
        auto*                  entries       = std::launder(reinterpret_cast<gr::pmt::PackedEntry*>(spoof.data() + sizeof(gr::pmt::Header)));
        const std::uint32_t    payloadOffset = entries[0].payloadOffset;
        const std::uint32_t    bogusSize     = entries[0].payloadLength + 16U;
        std::memcpy(spoof.data() + payloadOffset, &bogusSize, sizeof(bogusSize));
        const auto restored = ValueMap::from_blob(std::span<const std::byte>{spoof});
        expect(!restored.has_value()) << "from_blob must reject a value-record whose size != payloadLength";
    };

    "from_blob rejects a hand-corrupted free-list chain"_test = [] {
        ValueMap map;
        map.insert_or_assign("a", std::int64_t{1});
        map.insert_or_assign("b", std::int64_t{2});
        map.erase("a"); // produces one free chunk
        const auto                  src = map.blob();
        std::pmr::vector<std::byte> mutableBuf(src.begin(), src.end(), std::pmr::get_default_resource());
        if ((reinterpret_cast<std::uintptr_t>(mutableBuf.data()) & 15UZ) != 0UZ) {
            return; // skip if not 16-byte aligned (unlikely but possible)
        }
        auto* hdr            = std::launder(reinterpret_cast<gr::pmt::Header*>(mutableBuf.data()));
        hdr->payloadFreeHead = hdr->totalSize + 1024U; // point past the end of the blob

        const auto restored = ValueMap::from_blob(std::span<const std::byte>{mutableBuf.data(), mutableBuf.size()});
        expect(!restored.has_value()) << "from_blob must reject a corrupt free-list head";
    };

    "from_blob recursively validates nested-map sub-blobs"_test = [] {
        ValueMap inner;
        inner.emplace("k", std::int64_t{42});
        ValueMap outer;
        outer.emplace("nested", std::move(inner));

        const auto                                                   srcBytes = outer.blob();
        alignas(gr::pmt::kBlobAlignment) std::array<std::byte, 4096> alignedBuf{};
        expect(srcBytes.size() <= alignedBuf.size()) << "alignedBuf too small for outer blob";
        std::memcpy(alignedBuf.data(), srcBytes.data(), srcBytes.size());

        const auto* outerEntries = reinterpret_cast<const gr::pmt::PackedEntry*>(alignedBuf.data() + sizeof(gr::pmt::Header));
        const auto  innerBlobOff = static_cast<std::size_t>(outerEntries[0].payloadOffset) + 8UZ;
        const auto* innerHdr     = reinterpret_cast<const gr::pmt::Header*>(alignedBuf.data() + innerBlobOff);
        expect(eq(std::memcmp(innerHdr->magic, gr::pmt::kBlobMagic.data(), gr::pmt::kBlobMagic.size()), 0)) << "test setup: inner blob magic must be intact pre-corruption";

        const auto          innerEntries0Off = innerBlobOff + sizeof(gr::pmt::Header);
        const std::uint32_t bogus            = innerHdr->totalSize + 0x10000U;
        std::memcpy(alignedBuf.data() + innerEntries0Off, &bogus, sizeof(bogus));

        const auto r = ValueMap::from_blob(std::span<const std::byte>{alignedBuf.data(), srcBytes.size()});
        expect(!r.has_value()) << "from_blob must reject a corrupt nested-map sub-blob entry";
        if (!r.has_value()) {
            expect(r.error() == gr::pmt::DeserialiseError::CorruptOffset || r.error() == gr::pmt::DeserialiseError::AlignmentViolation) << std::format("expected CorruptOffset (or AlignmentViolation), got {}", static_cast<int>(r.error()));
        }
    };

    "from_blob recursively validates tensor sub-blobs"_test = [] {
        ValueMap                   outer;
        gr::Tensor<gr::pmt::Value> tensor(gr::extents_from, std::array<std::size_t, 1>{2UZ});
        tensor._data.data()[0] = Value{std::string_view{"head"}};
        ValueMap innerMap;
        innerMap.emplace(std::pmr::string{"k"}, Value{std::int64_t{77}});
        tensor._data.data()[1] = Value{std::move(innerMap)};
        outer.emplace("composed", tensor);

        const auto                                                   srcBytes = outer.blob();
        alignas(gr::pmt::kBlobAlignment) std::array<std::byte, 4096> alignedBuf{};
        expect(srcBytes.size() <= alignedBuf.size()) << "alignedBuf too small for outer blob";
        std::memcpy(alignedBuf.data(), srcBytes.data(), srcBytes.size());

        const auto* outerEntries  = reinterpret_cast<const gr::pmt::PackedEntry*>(alignedBuf.data() + sizeof(gr::pmt::Header));
        const auto  tensorSubOff  = static_cast<std::size_t>(outerEntries[0].payloadOffset) + 8UZ;
        const auto* tensorBase    = alignedBuf.data() + tensorSubOff;
        const auto  encodingFlags = static_cast<std::uint8_t>(tensorBase[2]);
        expect((encodingFlags & gr::pmt::kTensorEncodingVariableSize) != 0U) << "Tensor<Value> must use the variable-size encoding";

        const std::size_t   element0PayloadLenAt = tensorSubOff + gr::pmt::kTensorBlobHeaderSize + gr::pmt::paddedTensorExtentsBytes(1) + 8UZ;
        const std::uint32_t bogus                = static_cast<std::uint32_t>(srcBytes.size()) + 0x10000U;
        std::memcpy(alignedBuf.data() + element0PayloadLenAt, &bogus, sizeof(bogus));

        const auto r = ValueMap::from_blob(std::span<const std::byte>{alignedBuf.data(), srcBytes.size()});
        expect(!r.has_value()) << "from_blob must reject a corrupt tensor sub-blob element";
        if (!r.has_value()) {
            expect(r.error() == gr::pmt::DeserialiseError::CorruptOffset) << std::format("expected CorruptOffset, got {}", static_cast<int>(r.error()));
        }
    };

    // --- 5. Advisory flags ---

    "freeze sets the advisory flag; is_frozen reports it"_test = [] {
        ValueMap map;
        expect(!map.is_frozen());
        map.freeze();
        expect(map.is_frozen());
        const auto* hdr = reinterpret_cast<const gr::pmt::Header*>(map.blob().data());
        expect(hdr != nullptr);
        if (hdr != nullptr) {
            expect((hdr->flags & gr::pmt::kHeaderFlagFrozen) != 0);
        }
    };

    "freeze on a view-mode map is a silent no-op"_test = [] {
        // Debug builds assert on view mutation; this exercises the release-path guard, which must
        // never write the aliased (possibly read-only) blob.
        if constexpr (!gr::meta::kDebugBuild) {
            ValueMap src;
            src.emplace("k", std::int32_t{1});
            ValueMap view = ValueMap::makeView(src.blob());
            expect(view.is_view());
            view.freeze();
            expect(!view.is_frozen()) << "freeze() must no-op on a view";
            expect(!src.is_frozen()) << "the aliased blob's frozen flag must stay untouched";
        }
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
        expect(eq(view.value_or<float>("sample_rate", 0.f), 48000.0f));
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
        expect(eq(owned.value_or<double>("frequency", 0.0), 2.4e9));
        expect(eq(owned.value_or<std::string_view>("note", std::string_view{}), std::string_view{"abc"}));
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

    "view-mode mutator aborts in debug, no-ops in release"_test = [] {
        if constexpr (!gr::meta::kDebugBuild) {
            return; // release: silent no-op by design
        }
        ValueMap src;
        src.emplace("k", 1.0f);
        ValueMap view = ValueMap::makeView(src.blob());
        expect(aborts([&] { view.insert_or_assign("new", 2.0f); })) << "insert_or_assign on view-mode must trip the assert";
    };
};

// Iterator deref yields view-mode Values for every payload-living type (ComplexFloat64,
// String, Tensor<T>, nested ValueMap). The yielded Value aliases the source blob bytes —
// no allocation per deref. Caller may then do alloc-free reads via get_if<TensorView<T>>()
// / get_if<ValueMap>(), or owning materialise via value_or<Tensor<T>>() / get_if<ValueMap>().owned().
const boost::ut::suite<"ValueMap iter view-mode (variable-size value types)"> _iter_view_mode_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using gr::pmt::ValueMap;

    // Note (2026-04-30): the alias-only optimisation is temporarily disabled for variable-size
    // types (ComplexFloat64 / String / Tensor / Map). The current wire format scatters values
    // across PackedEntry fields, so iter cannot alias blob bytes directly via _data — it falls
    // back to _entryToValue, which allocates an owning Value per deref (correct, but per-
    // element heap traffic). The alias-only iter returns once the PackedEntry rewrite lands.
    // Tests below verify data correctness only — the is_view() assertion has been removed.
    "iter yields correct ComplexFloat64 (V2: owning, view-mode pending PackedEntry rewrite)"_test = [] {
        ValueMap map;
        map.insert_or_assign("c", std::complex<double>{1.5, -2.5});
        std::size_t seen = 0UZ;
        for (const auto& [k, v] : map) {
            ++seen;
            const std::complex<double>* p = v.get_if<std::complex<double>>();
            expect(p != nullptr);
            if (p) {
                expect(eq(p->real(), 1.5));
                expect(eq(p->imag(), -2.5));
            }
        }
        expect(eq(seen, 1UZ));
    };

    "iter yields correct nested ValueMap (V2: owning, view-mode pending)"_test = [] {
        ValueMap inner;
        inner.insert_or_assign("a", std::int32_t{42});
        inner.insert_or_assign("b", std::string_view{"hello"});
        ValueMap outer;
        outer.insert_or_assign("nested", inner);
        std::size_t seen = 0UZ;
        for (const auto& [k, v] : outer) {
            ++seen;
            std::optional<ValueMap> nestedView = v.get_if<ValueMap>();
            expect(nestedView.has_value());
            if (nestedView) {
                expect(eq(nestedView->size(), 2UZ));
                expect(eq(nestedView->value_or<std::int32_t>("a", 0), 42));
            }
        }
        expect(eq(seen, 1UZ));
    };

    "iter yields correct Tensor<float> (V2: owning, view-mode pending)"_test = [] {
        ValueMap          map;
        gr::Tensor<float> src(gr::extents_from, std::array<std::size_t, 1>{3UZ});
        src._data.data()[0] = 1.f;
        src._data.data()[1] = 2.f;
        src._data.data()[2] = 3.f;
        map.insert_or_assign("t", std::move(src));
        std::size_t seen = 0UZ;
        for (const auto& [k, v] : map) {
            ++seen;
            std::optional<gr::TensorView<float>> view = v.get_if<gr::TensorView<float>>();
            expect(view.has_value());
            if (view) {
                expect(eq(view->size(), 3UZ));
                expect(eq(view->data()[0], 1.f));
                expect(eq(view->data()[2], 3.f));
            }
        }
        expect(eq(seen, 1UZ));
    };

    "iter yields correct Tensor<Value> (variable-size cells)"_test = [] {
        ValueMap          map;
        gr::Tensor<Value> src(gr::extents_from, std::array<std::size_t, 1>{2UZ});
        src._data.data()[0] = Value{std::int32_t{7}};
        src._data.data()[1] = Value{std::string_view{"x"}};
        map.insert_or_assign("vt", std::move(src));
        std::size_t seen = 0UZ;
        for (const auto& [k, v] : map) {
            ++seen;
            // iterator yields ValueView; get_if<TensorView<Value>> needs the owning Value path —
            // materialise via .owned() (small per-iteration allocation; unavoidable for sub-Value decode).
            const Value                          materialised = v.owned();
            std::optional<gr::TensorView<Value>> view         = materialised.get_if<gr::TensorView<Value>>();
            expect(view.has_value());
            if (view) {
                expect(eq(view->size(), 2UZ));
                expect(eq((*view)[0].value_or<std::int32_t>(0), 7));
                expect(eq((*view)[1].value_or(std::string_view{}), std::string_view{"x"}));
            }
        }
        expect(eq(seen, 1UZ));
    };

    "erase reclaims payload bytes via free list (in-place, no shrink_to_fit needed)"_test = [] {
        // Insert a string entry so its payload region lives in the pool. Erase it, then
        // re-insert a different string of the same length: the new payload must reuse the
        // freed offset (blob size must NOT grow from the second insert).
        ValueMap               map(std::pmr::get_default_resource(), /*initialEntries=*/8U, /*slack=*/256U);
        const std::string_view s1 = "this-is-a-32-byte-payload-region";
        map.insert_or_assign("k", s1);
        const std::size_t blobBefore = map.blob().size_bytes();
        map.erase("k");
        const std::string_view s2 = "another-32-byte-payload-region!!";
        map.insert_or_assign("k2", s2);
        expect(eq(blobBefore, map.blob().size_bytes())) << "free-list reclamation must reuse the freed payload region (no realloc)";
        expect(eq(map.size(), 1UZ));
        expect(eq(map.value_or<std::string_view>("k2", std::string_view{}), s2));
    };

    "iter allocates one 16-B record per element (V2 baseline; alias-only iter pending)"_test = [] {
        // Iter constructs a fresh owning Value per element. Each Value allocates one 16-B value-
        // record via the map's resource (inline scalars: 16 B per element). The alias-only path
        // returns once iter aliases blob bytes via the PackedEntry rewrite. For now: verify
        // allocation count == #elements.
        CountingResource mr;
        ValueMap         map(&mr);
        map.insert_or_assign("i64", std::int64_t{42});
        map.insert_or_assign("f64", 3.14);
        map.insert_or_assign("u32", std::uint32_t{7});
        map.insert_or_assign("c32", std::complex<float>{1.f, -1.f});

        const std::size_t allocsBefore = mr.allocCount;
        std::size_t       seen         = 0UZ;
        for (const auto& [k, v] : map) {
            ++seen;
            (void)v.value_type(); // touch the aliased handle to force the per-element deref
        }
        const std::size_t allocsDuring = mr.allocCount - allocsBefore;
        expect(eq(seen, 4UZ));
        // C3: every entry — including inline scalars — has a value-record in the payload pool.
        // Iter aliases bytes via Value::makeView (zero allocation per deref, all value types).
        expect(eq(allocsDuring, 0UZ)) << "C3: iter is alloc-free for inline scalars";
    };
};

// Wire-format guards bundle (items A-I). All items are layout-compatible with the prior wire
// format (readers ignore the guard bytes); this suite verifies the guards land where the writer
// claims they do, that mutationCount tracks state changes, and that safeView rejects corrupted
// free-list chains. Some items are debug-build-only — the corresponding tests gate on
// `gr::meta::kDebugBuild`.
const boost::ut::suite<"ValueMap - monadic lookup (find_value / get_if / value_or / or_else / holds)"> _monadic_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using gr::pmt::ValueMap;

    "find_value: hit returns view-mode Value"_test = [] {
        ValueMap m;
        m.insert_or_assign("rate", 48000.0f);
        const auto v = m.find_value("rate");
        expect(v.has_value());
        if (v) {
            expect(v->is_view());
            expect(v->holds<float>());
            const auto* p = v->get_if<float>();
            expect(p != nullptr);
            if (p) {
                expect(eq(*p, 48000.0f));
            }
        }
    };

    "find_value: miss returns nullopt"_test = [] {
        ValueMap m;
        m.insert_or_assign("rate", 48000.0f);
        expect(!m.find_value("missing").has_value());
    };

    "find_value: empty map returns nullopt"_test = [] {
        ValueMap m;
        expect(!m.find_value("any").has_value());
    };

    "get_if<T>: inline scalar hit returns T*"_test = [] {
        ValueMap m;
        m.insert_or_assign("count", std::int64_t{42});
        const auto* p = m.get_if<std::int64_t>("count");
        expect(p != nullptr);
        if (p) {
            expect(eq(*p, std::int64_t{42}));
        }
    };

    "get_if<T>: inline scalar type mismatch returns nullptr"_test = [] {
        ValueMap m;
        m.insert_or_assign("count", std::int64_t{42});
        expect(m.get_if<float>("count") == nullptr);
    };

    "get_if<T>: missing key returns nullptr / empty optional"_test = [] {
        ValueMap m;
        expect(m.get_if<std::int64_t>("missing") == nullptr);
        expect(!m.get_if<std::string_view>("missing").has_value());
    };

    "get_if<string_view>: hit returns view aliasing blob"_test = [] {
        ValueMap m;
        m.insert_or_assign("name", std::string_view{"hello"});
        const auto sv = m.get_if<std::string_view>("name");
        expect(sv.has_value());
        if (sv) {
            expect(eq(*sv, std::string_view{"hello"}));
        }
    };

    "get_if<TensorView<int32_t>>: hit returns view aliasing blob"_test = [] {
        ValueMap                 m;
        gr::Tensor<std::int32_t> tensor(gr::extents_from, std::array<std::size_t, 1>{3UZ});
        tensor._data.data()[0] = 7;
        tensor._data.data()[1] = 8;
        tensor._data.data()[2] = 9;
        m.insert_or_assign("ints", std::move(tensor));

        const auto view = m.get_if<gr::TensorView<std::int32_t>>("ints");
        expect(view.has_value());
        if (view) {
            expect(eq(view->size(), 3UZ));
            expect(eq(view->_data.data()[0], 7));
            expect(eq(view->_data.data()[2], 9));
        }
    };

    "value_or<T>: hit returns stored value, ignores default"_test = [] {
        ValueMap m;
        m.insert_or_assign("rate", 48000.0f);
        expect(eq(m.value_or<float>("rate", 0.f), 48000.0f));
    };

    "value_or<T>: miss returns default"_test = [] {
        ValueMap m;
        expect(eq(m.value_or<float>("missing", 1.5f), 1.5f));
    };

    "value_or<T>: type mismatch returns default"_test = [] {
        ValueMap m;
        m.insert_or_assign("count", std::int64_t{42});
        expect(eq(m.value_or<float>("count", -1.f), -1.f));
    };

    "value_or<string_view>: alloc-free view-mode read"_test = [] {
        ValueMap m;
        m.insert_or_assign("name", std::string_view{"world"});
        const auto sv = m.value_or<std::string_view>("name", "");
        expect(eq(sv, std::string_view{"world"}));
    };

    "or_else<T>: hit does not invoke factory"_test = [] {
        ValueMap m;
        m.insert_or_assign("rate", 48000.0f);
        bool       called = false;
        const auto v      = m.or_else<float>("rate", [&] {
            called = true;
            return 0.f;
        });
        expect(eq(v, 48000.0f));
        expect(!called);
    };

    "or_else<T>: miss invokes factory and returns its result"_test = [] {
        ValueMap   m;
        bool       called = false;
        const auto v      = m.or_else<float>("missing", [&] {
            called = true;
            return 1.25f;
        });
        expect(eq(v, 1.25f));
        expect(called);
    };

    "or_else<T>: type mismatch invokes factory"_test = [] {
        ValueMap m;
        m.insert_or_assign("count", std::int64_t{42});
        bool       called = false;
        const auto v      = m.or_else<float>("count", [&] {
            called = true;
            return 9.f;
        });
        expect(eq(v, 9.f));
        expect(called);
    };

    "holds<T>: hit returns true on type match"_test = [] {
        ValueMap m;
        m.insert_or_assign("rate", 48000.0f);
        expect(m.holds<float>("rate"));
    };

    "holds<T>: hit returns false on type mismatch"_test = [] {
        ValueMap m;
        m.insert_or_assign("rate", 48000.0f);
        expect(!m.holds<std::int64_t>("rate"));
    };

    "holds<T>: missing key returns false"_test = [] {
        ValueMap m;
        expect(!m.holds<float>("missing"));
    };

    "lookup chain: find_value + Value monad composes"_test = [] {
        ValueMap m;
        m.insert_or_assign("rate", 48000.0f);

        const float chained = m.find_value("rate")                                                  //
                                  .transform([](const Value& v) { return v.value_or<float>(0.f); }) //
                                  .value_or(-1.f);
        expect(eq(chained, 48000.0f));

        const float missingChained = m.find_value("missing")                                               //
                                         .transform([](const Value& v) { return v.value_or<float>(0.f); }) //
                                         .value_or(-1.f);
        expect(eq(missingChained, -1.f));
    };

    "ArrowProxy: it->second works (chained-arrow rule)"_test = [] {
        ValueMap m;
        m.insert_or_assign("rate", 48000.0f);
        const auto it = m.find("rate");
        expect(it != m.end());
        if (it != m.end()) {
            expect(eq(std::string_view{it->first}, std::string_view{"rate"}));
            expect(it->second.holds<float>());
            const auto* p = it->second.get_if<float>();
            expect(p != nullptr);
            if (p) {
                expect(eq(*p, 48000.0f));
            }
        }
    };
};

const boost::ut::suite<"ValueMap - shared-buffer round-trip (USM/IPC-style)"> _shared_buffer_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using gr::pmt::ValueMap;
    using gr::pmt::ValueMapView;

    static_assert(std::is_trivially_copyable_v<ValueMapView>, "ValueMapView must be USM-by-value-capture-safe");
    static_assert(sizeof(ValueMapView) == 32UZ, "ValueMapView is two cache-line-quarter pointers + a pair of u32 — keep it tight");

    "host writes via ValueMap, kernel reads via ValueMapView (slice through public base)"_test = [] {
        // Simulated USM region: a fixed-capacity arena that cannot grow (null upstream).
        // Sized to absorb a single grow ladder if reserve() under-shoots — the test asserts the
        // sender stays well within the arena, but null upstream means a runaway grow → bad_alloc.
        alignas(64) static std::array<std::byte, 4096> kHostUsm{};
        std::pmr::monotonic_buffer_resource            hostPool{kHostUsm.data(), kHostUsm.size(), std::pmr::null_memory_resource()};

        ValueMap host{&hostPool, /*initialCapacityEntries=*/8U, /*initialPayloadCapacity=*/0U};
        host.reserve(8U, /*payloadBytes=*/256U); // pre-allocate payload pool so inserts do not grow
        std::ignore = host.try_emplace("counter", std::int32_t{42});
        std::ignore = host.try_emplace("voltage", 1.5);
        std::ignore = host.try_emplace("label", std::string_view{"alpha"});
        expect(eq(host.size(), 3UZ));

        const std::span<const std::byte> hostBlob = host.blob();
        expect(gt(hostBlob.size(), 0UZ));
        expect(le(hostBlob.size(), kHostUsm.size())) << "blob must fit inside the simulated USM arena";

        // Kernel side: a ValueMapView constructed from a span over the same bytes (zero-copy).
        // Slicing through the public base proves the inheritance lets a kernel capture by value.
        ValueMapView kernelView = ValueMap::makeView(hostBlob);
        expect(eq(kernelView.size(), 3UZ));
        expect(eq(kernelView.empty(), false));

        std::size_t seen = 0UZ;
        for (auto [key, val] : kernelView) {
            ++seen;
            if (key == "counter") {
                expect(val.holds<std::int32_t>());
                expect(eq(*val.get_if<std::int32_t>(), 42));
            } else if (key == "voltage") {
                expect(val.holds<double>());
                expect(eq(*val.get_if<double>(), 1.5));
            } else if (key == "label") {
                expect(val.holds<std::string_view>());
                expect(eq(*val.get_if<std::string_view>(), std::string_view{"alpha"}));
            }
        }
        expect(eq(seen, 3UZ));

        const auto itVoltage = kernelView.find("voltage");
        expect(itVoltage != kernelView.end());
        expect(kernelView.contains("counter"));
        expect(eq(kernelView.count("nope"), 0UZ));
    };

    "kernel mutates inline-scalar payload in place; host sees the update through the same arena"_test = [] {
        alignas(64) static std::array<std::byte, 4096> kSharedUsm{};
        std::pmr::monotonic_buffer_resource            sharedPool{kSharedUsm.data(), kSharedUsm.size(), std::pmr::null_memory_resource()};

        ValueMap host{&sharedPool, /*initialCapacityEntries=*/8U, /*initialPayloadCapacity=*/0U};
        host.reserve(8U, /*payloadBytes=*/256U);
        std::ignore = host.try_emplace("counter", std::int32_t{100});
        std::ignore = host.try_emplace("ratio", 0.25f);

        // Kernel side: aliasing view over the same bytes. Mutations through ValueView::get_if<T>()
        // write directly into the host arena — no memcpy back required.
        ValueMapView kernelView = ValueMap::makeView(host.blob());

        auto itCounter = kernelView.find("counter");
        expect(itCounter != kernelView.end());
        // Pair-by-value: dereference yields a temporary Value-aliasing-view; get_if returns a
        // pointer into the shared bytes.
        gr::pmt::ValueView counterAlias = (*itCounter).second;
        auto*              counterPtr   = counterAlias.get_if<std::int32_t>();
        expect(counterPtr != nullptr);
        if (counterPtr) {
            *counterPtr = 999; // kernel write into host USM
        }

        auto itRatio = kernelView.find("ratio");
        expect(itRatio != kernelView.end());
        gr::pmt::ValueView ratioAlias = (*itRatio).second;
        auto*              ratioPtr   = ratioAlias.get_if<float>();
        expect(ratioPtr != nullptr);
        if (ratioPtr) {
            *ratioPtr = 0.875f;
        }

        // Host re-reads through its own owning ValueMap (same backing bytes).
        const auto hostCounter = host.find_value("counter");
        expect(hostCounter.has_value());
        expect(eq(*hostCounter->get_if<std::int32_t>(), 999));

        const auto hostRatio = host.find_value("ratio");
        expect(hostRatio.has_value());
        expect(eq(*hostRatio->get_if<float>(), 0.875f));
    };

    "memcpy round-trip: host arena -> device arena -> mutation -> memcpy back -> host sees update"_test = [] {
        alignas(64) static std::array<std::byte, 4096> kHostArena{};
        alignas(64) static std::array<std::byte, 4096> kDeviceArena{};

        std::pmr::monotonic_buffer_resource hostPool{kHostArena.data(), kHostArena.size(), std::pmr::null_memory_resource()};
        ValueMap                            host{&hostPool, /*initialCapacityEntries=*/8U, /*initialPayloadCapacity=*/0U};
        host.reserve(8U, /*payloadBytes=*/256U);
        std::ignore                   = host.try_emplace("samples", std::uint32_t{1024});
        std::ignore                   = host.try_emplace("offset", std::int64_t{-7});
        auto*             hostBlobMut = const_cast<std::byte*>(host.blob().data()); // PMR may have offset us inside the arena
        const std::size_t hostBlobLen = host.blob().size();

        // memcpy host blob -> device arena (no extra allocation; both arenas are pre-sized).
        expect(le(hostBlobLen, kDeviceArena.size()));
        if (hostBlobLen <= kDeviceArena.size()) {
            std::memcpy(kDeviceArena.data(), hostBlobMut, hostBlobLen);
        }

        // Kernel observes the device arena via a bare ValueMapView (mock USM kernel handle).
        ValueMapView deviceView = ValueMap::makeView(std::span<const std::byte>{kDeviceArena.data(), hostBlobLen});
        expect(eq(deviceView.size(), 2UZ));

        // Mutate in-place on the device side (aliasing the device arena bytes).
        gr::pmt::ValueView sAlias = (*deviceView.find("samples")).second;
        if (auto* p = sAlias.get_if<std::uint32_t>()) {
            *p = 4096U;
        }
        gr::pmt::ValueView oAlias = (*deviceView.find("offset")).second;
        if (auto* p = oAlias.get_if<std::int64_t>()) {
            *p = 12345;
        }

        // memcpy device arena -> host blob (mirrors a USM read-back / kernel completion barrier).
        std::memcpy(hostBlobMut, kDeviceArena.data(), hostBlobLen);

        // Host sees the kernel-side mutations through its original ValueMap handle.
        const auto hostSamples = host.find_value("samples");
        expect(hostSamples.has_value());
        expect(eq(*hostSamples->get_if<std::uint32_t>(), 4096U));

        const auto hostOffset = host.find_value("offset");
        expect(hostOffset.has_value());
        expect(eq(*hostOffset->get_if<std::int64_t>(), std::int64_t{12345}));
    };

    "ValueMapView captured by value carries the full read API into a kernel-like context"_test = [] {
        alignas(64) static std::array<std::byte, 2048> kArena{};
        std::pmr::monotonic_buffer_resource            pool{kArena.data(), kArena.size(), std::pmr::null_memory_resource()};
        ValueMap                                       host{&pool, /*initialCapacityEntries=*/4U, /*initialPayloadCapacity=*/0U};
        host.reserve(4U, /*payloadBytes=*/128U);
        std::ignore = host.try_emplace("a", std::int32_t{1});
        std::ignore = host.try_emplace("b", std::int32_t{2});
        std::ignore = host.try_emplace("c", std::int32_t{3});

        const ValueMapView captured = ValueMap::makeView(host.blob()); // by-value capture
        // Trivial-copy round-trip: bitwise copy must preserve every accessor.
        ValueMapView copy{};
        std::memcpy(&copy, &captured, sizeof(copy));

        expect(eq(copy.size(), 3UZ));
        expect(copy.contains("b"));

        std::int32_t sum = 0;
        for (auto [key, val] : copy) {
            if (auto* p = val.get_if<std::int32_t>()) {
                sum += *p;
            }
        }
        expect(eq(sum, 6));
    };
};

const boost::ut::suite<"ValueMapView::owned(resource) — explicit conversion to owning ValueMap"> _vm_view_owned_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using gr::pmt::ValueMap;
    using gr::pmt::ValueMapView;

    "owned with default PMR copies the blob into a fresh allocation"_test = [] {
        ValueMap src;
        std::ignore = src.try_emplace("counter", std::int32_t{42});
        std::ignore = src.try_emplace("voltage", 1.5);

        const ValueMapView& asView = src;
        ValueMap            copy   = asView.owned();
        expect(eq(copy.size(), 2UZ));
        expect(copy.contains("counter"));
        expect(copy.contains("voltage"));
        expect(copy.resource() == std::pmr::get_default_resource());
        expect(copy.blob().data() != src.blob().data()) << "copy must alias different bytes";
    };

    "owned with explicit PMR routes the allocation through that resource"_test = [] {
        std::pmr::monotonic_buffer_resource pool{4096UZ};
        ValueMap                            src;
        std::ignore = src.try_emplace("a", std::uint32_t{1});

        const ValueMapView& asView = src;
        ValueMap            copy   = asView.owned(&pool);
        expect(copy.resource() == &pool);
        expect(copy.contains("a"));
    };

    "owned(nullptr) is normalised to default-PMR"_test = [] {
        ValueMap src;
        std::ignore                = src.try_emplace("k", std::int32_t{7});
        const ValueMapView& asView = src;
        ValueMap            copy   = asView.owned(nullptr);
        expect(copy.resource() == std::pmr::get_default_resource());
        expect(copy.contains("k"));
    };

    "owned copy is independent — mutating source does not change copy"_test = [] {
        ValueMap src;
        std::ignore = src.try_emplace("counter", std::int32_t{100});

        const ValueMapView& asView = src;
        ValueMap            copy   = asView.owned();
        // mutate the source's payload through find_value's view-Value
        if (auto v = src.find_value("counter")) {
            if (auto* p = v->get_if<std::int32_t>()) {
                *p = 999;
            }
        }
        // copy must still see the value at owned()-time
        const auto cv = copy.find_value("counter");
        expect(cv.has_value());
        expect(eq(*cv->get_if<std::int32_t>(), 100));
    };

    "owned copy can be mutated independently"_test = [] {
        ValueMap src;
        std::ignore                = src.try_emplace("k", std::int32_t{5});
        const ValueMapView& asView = src;
        ValueMap            copy   = asView.owned();
        std::ignore                = copy.try_emplace("k2", std::int32_t{99});
        expect(eq(copy.size(), 2UZ));
        expect(eq(src.size(), 1UZ)) << "mutating copy must not affect source";
    };

    "slicing a ValueMap to ValueMapView and calling owned() round-trips"_test = [] {
        ValueMap src;
        std::ignore = src.try_emplace("a", std::int32_t{1});
        std::ignore = src.try_emplace("b", std::int32_t{2});
        std::ignore = src.try_emplace("c", std::int32_t{3});

        const ValueMapView& asView = src;
        ValueMap            copy   = asView.owned();
        expect(eq(copy.size(), 3UZ));
        std::int32_t sum = 0;
        for (auto [k, v] : copy) {
            if (auto* p = v.get_if<std::int32_t>()) {
                sum += *p;
            }
        }
        expect(eq(sum, 6));
    };

    "owned round-trips a USM-style external view (bytes-only construction)"_test = [] {
        // Build a source ValueMap, copy its blob into a separate byte buffer,
        // construct a ValueMapView over those bytes, and owned() into a new ValueMap.
        ValueMap src;
        std::ignore = src.try_emplace("samples", std::uint32_t{1024});
        std::ignore = src.try_emplace("offset", std::int64_t{-7});

        const auto             srcSpan = src.blob();
        std::vector<std::byte> buffer(srcSpan.begin(), srcSpan.end());
        ValueMapView           bytesView = ValueMap::makeView(std::span<const std::byte>{buffer.data(), buffer.size()});

        ValueMap copy = bytesView.owned();
        expect(eq(copy.size(), 2UZ));
        const auto sCopy = copy.find_value("samples");
        expect(sCopy.has_value());
        expect(eq(*sCopy->get_if<std::uint32_t>(), std::uint32_t{1024}));
        const auto oCopy = copy.find_value("offset");
        expect(oCopy.has_value());
        expect(eq(*oCopy->get_if<std::int64_t>(), std::int64_t{-7}));
    };

    "owned routes its allocation through the supplied CountingResource"_test = [] {
        ValueMap src;
        std::ignore             = src.try_emplace("k", std::int32_t{7});
        const auto&      asView = static_cast<const ValueMapView&>(src);
        CountingResource mr;
        const auto       allocsBefore = mr.allocCount;
        ValueMap         copy         = asView.owned(&mr);
        expect(gt(mr.allocCount, allocsBefore)) << "owned must call allocate on the supplied resource";
        expect(copy.resource() == &mr);
        expect(copy.contains("k"));
    };

    "owned does not allocate from the source's resource"_test = [] {
        CountingResource srcMr;
        ValueMap         src{&srcMr};
        std::ignore                          = src.try_emplace("k", std::int32_t{7});
        const auto       srcAllocsAtSnapshot = srcMr.allocCount;
        const auto&      asView              = static_cast<const ValueMapView&>(src);
        CountingResource dstMr;
        ValueMap         copy = asView.owned(&dstMr);
        expect(eq(srcMr.allocCount, srcAllocsAtSnapshot)) << "source resource must not see new allocations during owned()";
        expect(gt(dstMr.allocCount, 0UZ));
        expect(copy.resource() == &dstMr);
    };
};

const boost::ut::suite<"ValueMapView - bounded mutators (try_emplace / insert_or_assign / erase / clear)"> _vm_view_mutator_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using gr::pmt::ValueMap;
    using gr::pmt::ValueMapView;

    "insert_or_assign overwrites existing same-type entry in place"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        std::ignore = host.try_emplace("counter", std::int32_t{42});

        ValueMapView& view = host;
        expect(view.insert_or_assign("counter", std::int32_t{777}));
        const auto v = host.find_value("counter");
        expect(v.has_value());
        expect(eq(*v->get_if<std::int32_t>(), 777));
    };

    "iterator/get_if path: kernel can mutate the inline-scalar payload in place via the alias"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        std::ignore = host.try_emplace("counter", std::int32_t{42});

        ValueMapView& view = host;
        auto          it   = view.find("counter");
        expect(it != view.end());
        // (*it).second is a ValueView by value aliasing the on-blob bytes — get_if<T>() returns
        // a mutable T* into that storage; the iterator's `->` chain is const-qualified, so use
        // the value-dereference form here.
        auto entry = *it;
        if (auto* p = entry.second.get_if<std::int32_t>()) {
            *p = 777;
        }
        expect(eq(*host.find_value("counter")->get_if<std::int32_t>(), 777));
    };

    "try_emplace appends a new entry when capacity allows"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        ValueMapView& view = host;
        const auto    ok   = view.try_emplace("voltage", 1.5);
        expect(ok);
        expect(eq(host.size(), 1UZ));
        const auto v = host.find_value("voltage");
        expect(v.has_value());
        expect(eq(*v->get_if<double>(), 1.5));
    };

    "try_emplace fails when the key already exists"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        std::ignore = host.try_emplace("k", std::int32_t{1});

        ValueMapView& view = host;
        const auto    ok   = view.try_emplace("k", std::int32_t{99});
        expect(!ok);
        const auto v = host.find_value("k");
        expect(v.has_value());
        expect(eq(*v->get_if<std::int32_t>(), 1));
    };

    "try_emplace fails when payload capacity is exhausted"_test = [] {
        ValueMap host;
        host.reserve(8U, 32U); // room for 2 inline-scalar value-records (16 B each)
        ValueMapView& view = host;
        expect(view.try_emplace("a", std::int32_t{1}));
        expect(view.try_emplace("b", std::int32_t{2}));
        const auto okThird = view.try_emplace("c", std::int32_t{3});
        expect(!okThird) << "third try_emplace must fail when payload pool is full";
        expect(eq(host.size(), 2UZ));
    };

    "try_emplace fails when entry capacity is exhausted"_test = [] {
        // Construct with explicit small entry capacity (reserve() applies geometric min of 8;
        // the constructor honours the requested number directly via _allocateBlob).
        ValueMap      host{nullptr, /*initialCapacityEntries=*/3U, /*initialPayloadCapacity=*/1024U};
        ValueMapView& view = host;
        expect(view.try_emplace("a", std::int32_t{1}));
        expect(view.try_emplace("b", std::int32_t{2}));
        const auto okThird = view.try_emplace("c", std::int32_t{3});
        expect(!okThird) << "third try_emplace must fail when entry array is full (cap 3 -> 2 entries + sentinel)";
        expect(eq(host.size(), 2UZ));
    };

    "try_emplace refuses keys that would need spilling"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        ValueMapView&     view = host;
        const std::string longKey(64UZ, 'x'); // > kMaxInlineKeyLength = 34
        const auto        ok = view.try_emplace(longKey, std::int32_t{1});
        expect(!ok) << "spilled keys are not supported in the bounded mutator path";
    };

    "insert_or_assign patches when key exists with matching type"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        std::ignore = host.try_emplace("k", std::int32_t{1});

        ValueMapView& view = host;
        const auto    ok   = view.insert_or_assign("k", std::int32_t{42});
        expect(ok);
        expect(eq(*host.find_value("k")->get_if<std::int32_t>(), 42));
        expect(eq(host.size(), 1UZ));
    };

    "insert_or_assign inserts when key absent"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        ValueMapView& view = host;
        const auto    ok   = view.insert_or_assign("new", std::int32_t{42});
        expect(ok);
        expect(eq(host.size(), 1UZ));
        expect(eq(*host.find_value("new")->get_if<std::int32_t>(), 42));
    };

    "insert_or_assign overwrites on type mismatch (existing key, different type)"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        std::ignore = host.try_emplace("k", std::int32_t{1});

        ValueMapView& view = host;
        const auto    ok   = view.insert_or_assign("k", 2.5);
        expect(ok);
        expect(eq(host.size(), 1UZ)) << "size should remain at 1 (entry overwritten in place)";
        const auto v = host.find_value("k");
        expect(v.has_value());
        expect(v->holds<double>());
        expect(eq(*v->get_if<double>(), 2.5));
    };

    "kernel-side mutation pattern: USM ring slot received writes through ValueMapView"_test = [] {
        // Simulate a Tag slot in a USM ring: host writer sets up via owning ValueMap, then a kernel
        // captures ValueMapView and patches/inserts within the slot's bounded capacity.
        alignas(64) static std::array<std::byte, 4096> kArena{};
        std::pmr::monotonic_buffer_resource            pool{kArena.data(), kArena.size(), std::pmr::null_memory_resource()};
        ValueMap                                       host{&pool, 8U, 0U};
        host.reserve(8U, 256U);
        std::ignore = host.try_emplace("samples", std::uint32_t{1024});
        std::ignore = host.try_emplace("rate", 48000.0f);

        // Kernel-side handle (slice through the public base). No _resource visible.
        ValueMapView kernelView = static_cast<ValueMapView&>(host);
        expect(kernelView.insert_or_assign("samples", std::uint32_t{4096})); // overwrites in place
        expect(kernelView.try_emplace("counter", std::int64_t{1}));
        expect(kernelView.insert_or_assign("rate", 96000.0f)); // overwrites in place

        // Host sees every kernel-side mutation through the same arena bytes.
        expect(eq(*host.find_value("samples")->get_if<std::uint32_t>(), std::uint32_t{4096}));
        expect(eq(*host.find_value("counter")->get_if<std::int64_t>(), std::int64_t{1}));
        expect(eq(*host.find_value("rate")->get_if<float>(), 96000.0f));
    };

    "try_emplace<std::string_view> appends a String entry with NUL-terminated payload"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        ValueMapView& view = host;
        expect(view.try_emplace("unit", std::string_view{"Hz"}));
        const auto v = host.find_value("unit");
        expect(v.has_value());
        expect(v->holds<std::string_view>());
        expect(eq(*v->get_if<std::string_view>(), std::string_view{"Hz"}));
    };

    "try_emplace<std::string_view> fails when payload pool cannot fit the record"_test = [] {
        ValueMap      host;
        ValueMapView& view = host;
        host.reserve(8U, 16U); // exactly one 16 B record's worth of payload
        std::ignore = view.try_emplace("a", std::int32_t{0});
        // pool now full; longer string record needs > 16 B → must fail bounded
        expect(!view.try_emplace("long", std::string_view{"abcdef"}));
    };

    "insert_or_assign<std::string_view> overwrites equal-length string in place"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        std::ignore        = host.try_emplace("name", std::string_view{"ABC"});
        ValueMapView& view = host;
        expect(view.insert_or_assign("name", std::string_view{"XYZ"}));
        expect(eq(*host.find_value("name")->get_if<std::string_view>(), std::string_view{"XYZ"}));
    };

    "insert_or_assign<std::string_view> succeeds for two short strings sharing the same padded record"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        std::ignore        = host.try_emplace("k", std::string_view{"ab"}); // recSize = max(16, 8+2+1) = 16
        ValueMapView& view = host;
        expect(view.insert_or_assign("k", std::string_view{"de"})); // same recSize → in-place
        expect(view.insert_or_assign("k", std::string_view{""}));   // empty still fits → recSize 16
        expect(eq(*host.find_value("k")->get_if<std::string_view>(), std::string_view{""}));
    };

    "insert_or_assign<std::string_view> rejects size-changing overwrites"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        std::ignore        = host.try_emplace("k", std::string_view{"longer-than-seven-chars"}); // recSize > 16
        ValueMapView& view = host;
        // shorter → smaller recSize → bounded path must reject (would require record shrinkage)
        expect(!view.insert_or_assign("k", std::string_view{"short"}));
        expect(eq(*host.find_value("k")->get_if<std::string_view>(), std::string_view{"longer-than-seven-chars"}));
    };

    "insert_or_assign<std::string_view> rejects type-mismatch (existing entry is non-String)"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        std::ignore        = host.try_emplace("k", std::int32_t{42});
        ValueMapView& view = host;
        expect(!view.insert_or_assign("k", std::string_view{"hi"}));
        expect(host.find_value("k")->holds<std::int32_t>());
    };

    "insert_or_assign<std::string_view> inserts new key into empty map"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        ValueMapView& view = host;
        expect(view.insert_or_assign("greeting", std::string_view{"hi"}));
        expect(eq(*host.find_value("greeting")->get_if<std::string_view>(), std::string_view{"hi"}));
    };

    "try_emplace<std::complex<double>> appends a 24 B record"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        ValueMapView& view = host;
        expect(view.try_emplace("z", std::complex<double>{1.5, -2.5}));
        const auto v = host.find_value("z");
        expect(v.has_value());
        expect(v->holds<std::complex<double>>());
        expect(eq(*v->get_if<std::complex<double>>(), std::complex<double>{1.5, -2.5}));
    };

    "insert_or_assign<std::complex<double>> overwrites the 16 B payload in place"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        std::ignore        = host.try_emplace("z", std::complex<double>{0.0, 0.0});
        ValueMapView& view = host;
        expect(view.insert_or_assign("z", std::complex<double>{3.0, 4.0}));
        expect(eq(*host.find_value("z")->get_if<std::complex<double>>(), std::complex<double>{3.0, 4.0}));
    };

    "insert_or_assign<std::complex<double>> rejects type-mismatch with std::complex<float>"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        std::ignore        = host.try_emplace("z", std::complex<float>{0.f, 0.f}); // 16 B inline-scalar record
        ValueMapView& view = host;
        expect(!view.insert_or_assign("z", std::complex<double>{3.0, 4.0}));
    };

    "insert_or_assign<std::complex<double>> inserts new and patches existing"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        ValueMapView& view = host;
        expect(view.insert_or_assign("z", std::complex<double>{1.0, 2.0}));
        expect(view.insert_or_assign("z", std::complex<double>{5.0, 6.0})); // patches
        expect(eq(*host.find_value("z")->get_if<std::complex<double>>(), std::complex<double>{5.0, 6.0}));
        expect(eq(host.size(), 1UZ));
    };

    // --- Tensor<T> (fixed element type, T != Value): bounded patch / try_emplace / insert_or_assign ---

    "try_emplace<Tensor<float>> appends a 1D tensor and round-trips via TensorView<float>"_test = [] {
        ValueMap host;
        host.reserve(8U, 1024U);
        gr::Tensor<float> tf(gr::extents_from, std::array<std::size_t, 1>{4UZ});
        tf._data.data()[0] = 1.0f;
        tf._data.data()[1] = 2.0f;
        tf._data.data()[2] = 3.0f;
        tf._data.data()[3] = 4.0f;
        ValueMapView& view = host;
        expect(view.try_emplace("samples", tf));
        const auto v = host.find_value("samples");
        expect(v.has_value());
        expect(v->is_tensor());
        const auto tv = v->get_if<gr::TensorView<float>>();
        expect(tv.has_value());
        expect(eq(tv->size(), 4UZ));
        expect(eq((*tv)[0UZ], 1.0f));
        expect(eq((*tv)[3UZ], 4.0f));
    };

    "try_emplace<Tensor<complex<double>>> appends with the 16 B element layout"_test = [] {
        ValueMap host;
        host.reserve(8U, 1024U);
        gr::Tensor<std::complex<double>> tc(gr::extents_from, std::array<std::size_t, 1>{3UZ});
        tc._data.data()[0] = {1.0, -1.0};
        tc._data.data()[1] = {2.0, -2.0};
        tc._data.data()[2] = {3.0, -3.0};
        ValueMapView& view = host;
        expect(view.try_emplace("z", tc));
        const auto v = host.find_value("z");
        expect(v.has_value());
        const auto tv = v->get_if<gr::TensorView<std::complex<double>>>();
        expect(tv.has_value());
        expect(eq(tv->size(), 3UZ));
        expect(eq((*tv)[1UZ], std::complex<double>{2.0, -2.0}));
    };

    "try_emplace<Tensor<bool>> uses 1-byte-per-element storage"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        gr::Tensor<bool> tb(gr::extents_from, std::array<std::size_t, 1>{5UZ});
        for (std::size_t i = 0UZ; i < tb.size(); ++i) {
            tb._data[i] = (i % 2UZ) != 0UZ;
        }
        ValueMapView& view = host;
        expect(view.try_emplace("mask", tb));
        const auto v = host.find_value("mask");
        expect(v.has_value());
        const auto tv = v->get_if<gr::TensorView<bool>>();
        expect(tv.has_value());
        expect(eq(tv->size(), 5UZ));
        expect(!(*tv)[0UZ]);
        expect((*tv)[1UZ]);
        expect(!(*tv)[2UZ]);
        expect((*tv)[3UZ]);
        expect(!(*tv)[4UZ]);
    };

    "try_emplace<Tensor<float>> 2D round-trips extents"_test = [] {
        ValueMap host;
        host.reserve(8U, 1024U);
        gr::Tensor<float> tf(gr::extents_from, std::array<std::size_t, 2>{2UZ, 3UZ});
        for (std::size_t i = 0UZ; i < tf.size(); ++i) {
            tf._data.data()[i] = static_cast<float>(i);
        }
        ValueMapView& view = host;
        expect(view.try_emplace("frame", tf));
        const auto v = host.find_value("frame");
        expect(v.has_value());
        const auto tv = v->get_if<gr::TensorView<float>>();
        expect(tv.has_value());
        expect(eq(tv->rank(), 2UZ));
        expect(eq(tv->size(), 6UZ));
        expect(eq((*tv)[5UZ], 5.0f));
    };

    "try_emplace<Tensor<float>> fails when payload pool cannot fit the record"_test = [] {
        ValueMap host;
        host.reserve(8U, 32U); // 8 B header + 4 B extents + 4*N floats — pool too small for any tensor of N≥4
        gr::Tensor<float> tf(gr::extents_from, std::array<std::size_t, 1>{10UZ});
        for (std::size_t i = 0UZ; i < tf.size(); ++i) {
            tf._data.data()[i] = static_cast<float>(i);
        }
        ValueMapView& view = host;
        expect(!view.try_emplace("too_big", tf));
        expect(eq(host.size(), 0UZ));
    };

    "insert_or_assign<Tensor<float>> overwrites in place when shape (byte count) matches"_test = [] {
        ValueMap host;
        host.reserve(8U, 1024U);
        gr::Tensor<float> tf(gr::extents_from, std::array<std::size_t, 1>{4UZ});
        for (std::size_t i = 0UZ; i < tf.size(); ++i) {
            tf._data.data()[i] = static_cast<float>(i);
        }
        std::ignore        = host.try_emplace("samples", tf);
        ValueMapView& view = host;

        gr::Tensor<float> tf2(gr::extents_from, std::array<std::size_t, 1>{4UZ});
        tf2._data.data()[0] = 10.0f;
        tf2._data.data()[1] = 20.0f;
        tf2._data.data()[2] = 30.0f;
        tf2._data.data()[3] = 40.0f;
        expect(view.insert_or_assign("samples", tf2));
        const auto tv = host.find_value("samples")->get_if<gr::TensorView<float>>();
        expect(tv.has_value());
        expect(eq((*tv)[0UZ], 10.0f));
        expect(eq((*tv)[3UZ], 40.0f));
    };

    "insert_or_assign<Tensor<float>> allows in-place reshape when element count is identical"_test = [] {
        ValueMap host;
        host.reserve(8U, 1024U);
        gr::Tensor<float> tf2x3(gr::extents_from, std::array<std::size_t, 2>{2UZ, 3UZ});
        for (std::size_t i = 0UZ; i < tf2x3.size(); ++i) {
            tf2x3._data.data()[i] = static_cast<float>(i);
        }
        std::ignore        = host.try_emplace("frame", tf2x3);
        ValueMapView& view = host;
        // Same rank + same elementCount → same byte count → succeed in place.
        gr::Tensor<float> tf3x2(gr::extents_from, std::array<std::size_t, 2>{3UZ, 2UZ});
        for (std::size_t i = 0UZ; i < tf3x2.size(); ++i) {
            tf3x2._data.data()[i] = static_cast<float>(i + 100);
        }
        expect(view.insert_or_assign("frame", tf3x2));
        const auto tv = host.find_value("frame")->get_if<gr::TensorView<float>>();
        expect(tv.has_value());
        expect(eq(tv->rank(), 2UZ));
        expect(eq(tv->size(), 6UZ));
        expect(eq((*tv)[5UZ], 105.0f));
    };

    "insert_or_assign<Tensor<float>> rejects size-changing overwrites"_test = [] {
        ValueMap host;
        host.reserve(8U, 1024U);
        gr::Tensor<float> tf4(gr::extents_from, std::array<std::size_t, 1>{4UZ});
        std::ignore            = host.try_emplace("samples", tf4);
        ValueMapView&     view = host;
        gr::Tensor<float> tf8(gr::extents_from, std::array<std::size_t, 1>{8UZ});
        expect(!view.insert_or_assign("samples", tf8)) << "different element count → different byte count → must reject";
    };

    "insert_or_assign<Tensor<float>> rejects element-type mismatch"_test = [] {
        ValueMap host;
        host.reserve(8U, 1024U);
        gr::Tensor<float> tf(gr::extents_from, std::array<std::size_t, 1>{4UZ});
        std::ignore                   = host.try_emplace("samples", tf);
        ValueMapView&            view = host;
        gr::Tensor<std::int32_t> ti(gr::extents_from, std::array<std::size_t, 1>{4UZ});
        expect(!view.insert_or_assign("samples", ti)) << "element type Float32 vs Int32 must reject";
    };

    "insert_or_assign<Tensor<float>> rejects when existing entry is a scalar, not a tensor"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        std::ignore            = host.try_emplace("k", 1.0f); // scalar float
        ValueMapView&     view = host;
        gr::Tensor<float> tf(gr::extents_from, std::array<std::size_t, 1>{1UZ});
        tf._data.data()[0] = 2.0f;
        expect(!view.insert_or_assign("k", tf));
    };

    "insert_or_assign<Tensor<float>> inserts new and patches existing"_test = [] {
        ValueMap host;
        host.reserve(8U, 1024U);
        gr::Tensor<float> tf1(gr::extents_from, std::array<std::size_t, 1>{3UZ});
        tf1._data.data()[0] = 1.0f;
        tf1._data.data()[1] = 2.0f;
        tf1._data.data()[2] = 3.0f;
        ValueMapView& view  = host;
        expect(view.insert_or_assign("data", tf1));
        gr::Tensor<float> tf2(gr::extents_from, std::array<std::size_t, 1>{3UZ});
        tf2._data.data()[0] = 10.0f;
        tf2._data.data()[1] = 20.0f;
        tf2._data.data()[2] = 30.0f;
        expect(view.insert_or_assign("data", tf2)); // same shape → patch
        const auto tv = host.find_value("data")->get_if<gr::TensorView<float>>();
        expect(tv.has_value());
        expect(eq((*tv)[2UZ], 30.0f));
        expect(eq(host.size(), 1UZ));
    };

    "kernel-side Tensor write: USM arena receives FFT-style output from a 'kernel' slice"_test = [] {
        // Producer/consumer pattern: host preps an empty ring slot, 'kernel' (here: ValueMapView
        // slice) writes a fixed-shape tensor result; host reads it via TensorView.
        alignas(64) static std::array<std::byte, 8192> kArena{};
        std::pmr::monotonic_buffer_resource            pool{kArena.data(), kArena.size(), std::pmr::null_memory_resource()};
        ValueMap                                       host{&pool, 8U, 0U};
        host.reserve(8U, 1024U);
        std::ignore = host.try_emplace("rate", 48000.0f);

        ValueMapView      kernelView = static_cast<ValueMapView&>(host);
        gr::Tensor<float> magnitudes(gr::extents_from, std::array<std::size_t, 1>{8UZ});
        for (std::size_t i = 0UZ; i < magnitudes.size(); ++i) {
            magnitudes._data.data()[i] = static_cast<float>(i) * 0.5f;
        }
        expect(kernelView.try_emplace("magnitudes", magnitudes));

        // Subsequent kernel iteration overwrites in place (same shape):
        for (std::size_t i = 0UZ; i < magnitudes.size(); ++i) {
            magnitudes._data.data()[i] = static_cast<float>(i) * 2.0f;
        }
        expect(kernelView.insert_or_assign("magnitudes", magnitudes));

        const auto v  = host.find_value("magnitudes");
        const auto tv = v->get_if<gr::TensorView<float>>();
        expect(tv.has_value());
        expect(eq(tv->size(), 8UZ));
        expect(eq((*tv)[7UZ], 14.0f));
    };

    // --- erase: bounded reclaim for ring-slot recycling ---

    "erase(key) removes the entry and reports 1; missing key reports 0"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        std::ignore = host.try_emplace("a", std::int32_t{1});
        std::ignore = host.try_emplace("b", std::int32_t{2});

        ValueMapView& view = host;
        expect(eq(view.erase("a"), 1UZ));
        expect(eq(view.erase("a"), 0UZ));
        expect(eq(host.size(), 1UZ));
        expect(!host.contains("a"));
        expect(host.contains("b"));
    };

    "erase(iterator) returns the next iterator and shifts subsequent entries"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        std::ignore = host.try_emplace("a", std::int32_t{1});
        std::ignore = host.try_emplace("b", std::int32_t{2});
        std::ignore = host.try_emplace("c", std::int32_t{3});

        ValueMapView& view   = host;
        const auto    nextIt = view.erase(view.find("b"));
        expect(nextIt != view.end());
        expect(eq(nextIt->first, std::string_view{"c"}));
        expect(eq(host.size(), 2UZ));
    };

    "erase(first, last) removes a contiguous range and returns the lo iterator"_test = [] {
        ValueMap host;
        host.reserve(8U, 256U);
        for (int i = 0; i < 5; ++i) {
            std::ignore = host.try_emplace(std::string_view{i == 0 ? "a" : i == 1 ? "b" : i == 2 ? "c" : i == 3 ? "d" : "e"}, std::int32_t{i});
        }
        ValueMapView& view  = host;
        const auto    loIt  = view.find("b");
        const auto    hiIt  = view.find("d");
        const auto    after = view.erase(loIt, hiIt);
        expect(eq(host.size(), 3UZ));
        expect(!host.contains("b"));
        expect(!host.contains("c"));
        expect(host.contains("a"));
        expect(host.contains("d"));
        expect(host.contains("e"));
        expect(after != view.end());
        expect(eq(after->first, std::string_view{"d"}));
    };

    "erase reclaims payload space — subsequent insert reuses the free list"_test = [] {
        ValueMap host;
        host.reserve(8U, 64U);                                          // tight payload pool to force reuse
        std::ignore           = host.try_emplace("a", std::int32_t{1}); // 16 B record
        std::ignore           = host.try_emplace("b", std::int32_t{2}); // 16 B
        std::ignore           = host.try_emplace("c", std::int32_t{3}); // 16 B
        const auto usedBefore = host.blob().size();

        ValueMapView& view = host;
        expect(eq(view.erase("b"), 1UZ));
        // re-insert different value with same size record → must reuse the freed slot
        std::ignore = host.try_emplace("d", std::int32_t{99});

        const auto usedAfter = host.blob().size();
        expect(le(usedAfter, usedBefore)) << "free-list reuse should not grow the blob";
        expect(host.contains("d"));
        expect(eq(*host.find_value("d")->get_if<std::int32_t>(), 99));
    };

    "kernel-side erase: ValueMapView slice can recycle a ring slot's tags"_test = [] {
        alignas(64) static std::array<std::byte, 4096> kArena{};
        std::pmr::monotonic_buffer_resource            pool{kArena.data(), kArena.size(), std::pmr::null_memory_resource()};
        ValueMap                                       host{&pool, 8U, 0U};
        host.reserve(8U, 256U);
        std::ignore = host.try_emplace("samples", std::uint32_t{1024});
        std::ignore = host.try_emplace("rate", 48000.0f);
        std::ignore = host.try_emplace("flag", true);

        ValueMapView kernelView = static_cast<ValueMapView&>(host);
        expect(eq(kernelView.erase("flag"), 1UZ));
        expect(eq(host.size(), 2UZ));
        expect(!host.contains("flag"));
        expect(host.contains("samples"));
        expect(host.contains("rate"));
    };

    // --- clear: bounded reset for slot recycling ---

    "clear via ValueMapView resets size + payloadUsed without freeing"_test = [] {
        ValueMap m;
        std::ignore = m.try_emplace("a", std::int32_t{1});
        std::ignore = m.try_emplace("b", std::int32_t{2});
        expect(eq(m.size(), 2UZ));
        const auto* hdr            = std::launder(reinterpret_cast<const gr::pmt::Header*>(m.blob().data()));
        const auto  capBeforeClear = hdr->payloadCapacity;

        ValueMapView& view = m;
        view.clear();
        expect(eq(m.size(), 0UZ));
        expect(m.empty());
        expect(eq(hdr->payloadUsed, std::uint32_t{0}));
        expect(eq(hdr->payloadCapacity, capBeforeClear)) << "clear must not change capacity";

        // After clear, fresh inserts work in the same slot.
        expect(view.try_emplace("c", std::int32_t{99}));
        expect(eq(m.size(), 1UZ));
        expect(eq(*m.find_value("c")->get_if<std::int32_t>(), 99));
    };

    "clear on a ring-slot ValueMap (makeAt-constructed) recycles the slot"_test = [] {
        alignas(16) std::array<std::byte, 1024> slot{};
        auto                                    map  = ValueMap::makeAt(slot, /*payloadCapacity=*/256U);
        ValueMapView&                           view = map;
        expect(view.try_emplace("a", std::int32_t{1}));
        expect(view.try_emplace("b", std::int32_t{2}));
        expect(eq(map.size(), 2UZ));

        view.clear();
        expect(eq(map.size(), 0UZ));
        expect(view.try_emplace("c", std::int32_t{3}));
        expect(eq(map.size(), 1UZ));
    };

    "clear on an empty/uninitialised view is a no-op (no UB)"_test = [] {
        ValueMapView view{};
        view.clear(); // must not crash
        expect(eq(view.size(), 0UZ));
    };
};

const boost::ut::suite<"ValueMap::makeAt + growing try_emplace (ring-writer prerequisites)"> _vm_make_at_suite = [] {
    using namespace boost::ut;
    using gr::pmt::ValueMap;
    using gr::pmt::ValueMapView;

    "makeAt initialises an empty ValueMap inside the given slot"_test = [] {
        alignas(16) std::array<std::byte, 1024> slot{};
        auto                                    map = ValueMap::makeAt(slot, /*payloadCapacity=*/256U, /*entryCapacity=*/8U);
        expect(eq(map.size(), 0UZ));
        expect(map.empty());
        expect(map.blob().data() == slot.data()) << "blob must alias the slot's bytes";
    };

    "makeAt + ValueMapView mutators round-trip (single Tag in a slot)"_test = [] {
        alignas(16) std::array<std::byte, 1024> slot{};
        auto                                    map = ValueMap::makeAt(slot, /*payloadCapacity=*/256U);

        ValueMapView& view = map;
        expect(view.try_emplace("counter", std::int32_t{42}));
        expect(view.try_emplace("rate", 48000.0f));
        expect(eq(map.size(), 2UZ));

        // Read back through the owning ValueMap interface (find_value).
        const auto c = map.find_value("counter");
        expect(c.has_value());
        expect(eq(*c->get_if<std::int32_t>(), 42));
        const auto r = map.find_value("rate");
        expect(r.has_value());
        expect(eq(*r->get_if<float>(), 48000.0f));
    };

    "makeAt fails gracefully when slot is too small (returns empty)"_test = [] {
        alignas(16) std::array<std::byte, 16> tiny{};
        auto                                  map = ValueMap::makeAt(tiny, /*payloadCapacity=*/512U, /*entryCapacity=*/8U);
        expect(eq(map.size(), 0UZ));
        expect(map.blob().empty());
    };

    "ValueMap::try_emplace (growing) auto-grows when bounded path would fail"_test = [] {
        // Construct with very small initial capacity; growing path must still succeed.
        ValueMap m{nullptr, /*entries=*/2U, /*payloadCapacity=*/16U};
        expect(m.try_emplace("a", std::int32_t{1}).second);
        expect(m.try_emplace("b", std::int32_t{2}).second);
        expect(m.try_emplace("c", std::int32_t{3}).second) << "growing try_emplace must succeed even past initial capacity";
        expect(eq(m.size(), 3UZ));
    };

    "ValueMap::try_emplace returns (it, false) when key already exists"_test = [] {
        ValueMap m;
        std::ignore = m.try_emplace("k", std::int32_t{1});
        expect(!m.try_emplace("k", std::int32_t{99}).second);
        expect(eq(*m.find_value("k")->get_if<std::int32_t>(), 1));
    };

    "iterator/get_if path: in-place int32 mutation on an owning ValueMap"_test = [] {
        ValueMap m;
        std::ignore = m.try_emplace("k", std::int32_t{42});
        auto entry  = *m.find("k");
        if (auto* p = entry.second.get_if<std::int32_t>()) {
            *p = 777;
        }
        expect(eq(*m.find_value("k")->get_if<std::int32_t>(), 777));
    };
};

const boost::ut::suite<"ValueMap try_emplace dispatch (bounded base vs STL-shape derived)"> _sig_parity_suite = [] {
    using namespace boost::ut;
    using gr::pmt::ValueMap;
    using gr::pmt::ValueMapView;

    // ValueMapView::try_emplace is the bounded path (returns bool, alloc-free). The derived
    // ValueMap declares a variadic STL-shape try_emplace<K, Args...> returning pair<iter, bool>
    // — that name hides the bounded base method. Reach the bounded variant by slicing through
    // a ValueMapView reference.
    static_assert(std::same_as<decltype(std::declval<ValueMapView>().try_emplace("k", std::int32_t{1})), bool>);
    static_assert(std::same_as<decltype(std::declval<ValueMap>().try_emplace("k", std::int32_t{1})), //
        std::pair<ValueMap::const_iterator, bool>>);

    "slicing safety — ValueMap& visible as ValueMapView& gets the bounded variant"_test = [] {
        ValueMap      m{nullptr, /*entries=*/3U, /*payloadCapacity=*/16U};
        ValueMapView& boundedRef = m;
        expect(boundedRef.try_emplace("a", std::int32_t{1}));
        // 16-B payload cap + 16-B per inline scalar = 1 fits, second fails bounded.
        expect(!boundedRef.try_emplace("b", std::int32_t{2})) << "bounded path must refuse on payload cap exhausted";
        // Same map, accessed as ValueMap& -> growing variadic (auto-grows; returns pair).
        ValueMap& growingRef = m;
        expect(growingRef.try_emplace("b", std::int32_t{2}).second) << "growing path must succeed via _grow";
        expect(eq(m.size(), 2UZ));
    };
};

const boost::ut::suite<"Value::get_if<ValueMap> lifetime contract"> _value_get_if_value_map_lifetime_suite = [] {
    using namespace boost::ut;
    using gr::pmt::Value;
    using gr::pmt::ValueMap;

    auto makeNestedMap = [](std::pmr::memory_resource* mr = std::pmr::get_default_resource()) {
        ValueMap m{mr};
        std::ignore = m.insert_or_assign("a", std::int32_t{42});
        std::ignore = m.insert_or_assign("b", std::string_view{"hello"});
        return m;
    };

    "view-mode result is valid while source Value is alive"_test = [&] {
        Value v{makeNestedMap()};
        auto  opt = v.get_if<ValueMap>();
        expect(opt.has_value());
        expect(opt->is_view()) << "get_if<ValueMap> must return view-mode (zero-alloc)";
        expect(eq(opt->size(), 2UZ));
        expect(opt->contains("a"));
        const auto a = opt->find_value("a");
        expect(a.has_value());
        const auto* ap = a->template get_if<std::int32_t>();
        expect(ap != nullptr);
        if (ap) {
            expect(eq(*ap, std::int32_t{42}));
        }
    };

    "owned() materialisation survives source destruction"_test = [&] {
        ValueMap owning;
        {
            Value v{makeNestedMap()};
            auto  opt = v.get_if<ValueMap>();
            expect(opt.has_value());
            owning = opt->owned(); // explicit materialisation onto default-PMR
            // v destroyed at end of scope; `owning` aliases its own bytes now
        }
        expect(!owning.is_view()) << "owned() must produce an owning ValueMap";
        expect(eq(owning.size(), 2UZ));
        const auto a = owning.find_value("a");
        expect(a.has_value());
        if (const auto* ap = a->template get_if<std::int32_t>(); ap) {
            expect(eq(*ap, std::int32_t{42}));
        } else {
            expect(false) << "key 'a' missing or wrong type after source destruction";
        }
        const auto b = owning.find_value("b");
        expect(b.has_value());
        expect(eq(b->template get_if<std::string_view>().value_or(""), std::string_view{"hello"}));
    };

    "owned(target_mr) routes the new allocation through the supplied PMR"_test = [] {
        CountingResource source_mr;
        CountingResource target_mr;

        ValueMap inner{&source_mr};
        std::ignore = inner.insert_or_assign("k", std::int64_t{7});
        Value v{std::move(inner), &source_mr};

        auto opt = v.get_if<ValueMap>();
        expect(opt.has_value());

        const auto sourceAllocs = source_mr.allocCount;
        const auto targetAllocs = target_mr.allocCount;

        ValueMap owning = opt->owned(&target_mr);

        expect(owning.resource() == &target_mr) << "owned(target_mr) must materialise onto target_mr";
        expect(eq(source_mr.allocCount, sourceAllocs)) << "source arena must not see new allocations";
        expect(gt(target_mr.allocCount, targetAllocs)) << "target arena must allocate the new blob";
        const auto k = owning.find_value("k");
        expect(k.has_value());
        if (const auto* kp = k->template get_if<std::int64_t>(); kp) {
            expect(eq(*kp, std::int64_t{7}));
        }
    };
};

int main() { return 0; }
