#include <boost/ut.hpp>

#include <memory_resource>
#include <span>
#include <string>
#include <vector>

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/Buffer.hpp>
#include <gnuradio-4.0/Tensor.hpp>
#include <gnuradio-4.0/device/DeviceRelocatable.hpp>

namespace gr::relocatable_test {

struct ScalarsOnly : gr::Block<ScalarsOnly> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    gr::Annotated<float, "gain">       gain  = 1.f;
    gr::Annotated<gr::Size_t, "decim"> decim = 1U;
    GR_MAKE_REFLECTABLE(ScalarsOnly, in, out, gain, decim);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return x * gain; }
};

struct PmrTaps : gr::Block<PmrTaps> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    std::pmr::vector<float> taps;
    GR_MAKE_REFLECTABLE(PmrTaps, in, out, taps);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return taps.empty() ? x : x * taps[0]; }
};

struct TensorTaps : gr::Block<TensorTaps> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    gr::Tensor<float> taps;
    GR_MAKE_REFLECTABLE(TensorTaps, in, out, taps);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return x; }
};

struct Resampled : gr::Block<Resampled, gr::Resampling<1UZ, 2UZ, true>> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    float gain = 1.f;
    GR_MAKE_REFLECTABLE(Resampled, in, out, gain);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return x * gain; }
};

struct RawVectorTaps : gr::Block<RawVectorTaps> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    std::vector<float> taps; // host heap: the mirror would carry a host pointer onto the device
    GR_MAKE_REFLECTABLE(RawVectorTaps, in, out, taps);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return x; }
};

struct StringLabel : gr::Block<StringLabel> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    std::pmr::string label; // small-string optimisation stores the data inside the object
    GR_MAKE_REFLECTABLE(StringLabel, in, out, label);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return x; }
};

struct SpanView : gr::Block<SpanView> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    std::span<const float> taps; // trivially copyable, yet the bytes are a host address
    GR_MAKE_REFLECTABLE(SpanView, in, out, taps);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return taps.empty() ? x : x * taps[0]; }
};

struct RawPointer : gr::Block<RawPointer> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    float* scratch = nullptr;
    GR_MAKE_REFLECTABLE(RawPointer, in, out, scratch);

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return x; }
};

struct PlainFunctor { // not a gr::Block: the device test helpers use these
    float gain = 2.f;

    [[nodiscard]] constexpr float processOne(float x) const noexcept { return x * gain; }
};

struct MutableCounter : gr::Block<MutableCounter> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    float               gain   = 1.f;
    mutable std::size_t _calls = 0UZ; // written by a const processOne: the device copy would discard it
    GR_MAKE_REFLECTABLE(MutableCounter, in, out, gain);

    [[nodiscard]] float processOne(float x) const noexcept {
        ++_calls;
        return x * gain;
    }
};

/// the FIR shape: taps in pmr storage next to a mutable scalar. The pmr member used to disqualify the whole block
/// from probing, so the scalar's discarded write went unreported.
struct MutableCounterWithTaps : gr::Block<MutableCounterWithTaps> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    std::pmr::vector<float> taps;
    mutable std::size_t     _calls = 0UZ;
    GR_MAKE_REFLECTABLE(MutableCounterWithTaps, in, out, taps);

    [[nodiscard]] float processOne(float x) const noexcept {
        ++_calls;
        return taps.empty() ? x : x * taps[0];
    }
};

/// writes through a mutable pmr member: shared with the mirror, so nothing is lost -- but the probe must put it back
struct MutableHistory : gr::Block<MutableHistory> {
    gr::PortIn<float>  in;
    gr::PortOut<float> out;

    mutable std::pmr::vector<float> history;
    GR_MAKE_REFLECTABLE(MutableHistory, in, out, history);

    [[nodiscard]] float processOne(float x) const noexcept {
        if (!history.empty()) {
            history[0] = x;
        }
        return x;
    }
};

/// two inputs and a mutable counter: the canary must synthesise one sample per port, not decline the shape
struct MutableTwoInput : gr::Block<MutableTwoInput> {
    gr::PortIn<float>  in0;
    gr::PortIn<float>  in1;
    gr::PortOut<float> out;

    mutable std::size_t _calls = 0UZ;
    GR_MAKE_REFLECTABLE(MutableTwoInput, in0, in1, out);

    [[nodiscard]] float processOne(float a, float b) const noexcept {
        ++_calls;
        return a - 2.f * b;
    }
};

} // namespace gr::relocatable_test

using namespace gr::relocatable_test;

// no real block is trivially copyable — the gr::Block base owns strings, a property_map and the ports' buffers
static_assert(!std::is_trivially_copyable_v<ScalarsOnly>);
static_assert(!std::is_trivially_copyable_v<PmrTaps>);

static_assert(gr::device::DeviceRelocatable<ScalarsOnly>);
static_assert(gr::device::DeviceRelocatable<PmrTaps>);
static_assert(gr::device::DeviceRelocatable<TensorTaps>);
static_assert(gr::device::DeviceRelocatable<Resampled>); // mixin blocks derive from Block<D, Resampling<...>>
static_assert(gr::device::DeviceRelocatable<PlainFunctor>);

// probing is no longer the stricter question: a pmr member is shared with the bit-copy, captured and put back
static_assert(gr::device::DeviceProbeSafe<PmrTaps>);
static_assert(gr::device::DeviceProbeSafe<MutableCounterWithTaps>);

static_assert(!gr::device::DeviceRelocatable<RawVectorTaps>);
static_assert(!gr::device::DeviceRelocatable<StringLabel>);
static_assert(!gr::device::DeviceRelocatable<RawPointer>);
static_assert(!gr::device::DeviceRelocatable<SpanView>); // a non-owning view carries a host address into the kernel

// the kernel-facing spans are pure views: they satisfy the view concepts but not the span concepts
static_assert(gr::InputViewLike<std::span<const float>>);
static_assert(gr::OutputViewLike<std::span<float>>);
static_assert(!gr::ReaderSpanLike<std::span<const float>>);
static_assert(!gr::WriterSpanLike<std::span<float>>);

// the base's own std::string/property_map members must never be what disqualifies a block
static_assert(gr::device::firstNonRelocatableMember<ScalarsOnly>().empty());
static_assert(gr::device::firstNonRelocatableMember<RawVectorTaps>() == "taps");
static_assert(gr::device::firstNonRelocatableMember<StringLabel>() == "label");
static_assert(gr::device::firstNonRelocatableMember<RawPointer>() == "scratch");
static_assert(gr::device::firstNonRelocatableMember<SpanView>() == "taps");

const boost::ut::suite<"device::DeviceRelocatable"> _relocatable = [] {
    using namespace boost::ut;

    "a relocated block reads its scalar settings through the mirrored bytes"_test = [] {
        ScalarsOnly block;
        block.gain = 4.f;

        alignas(ScalarsOnly) std::array<std::byte, sizeof(ScalarsOnly)> storage{};
        auto*                                                           mirror = reinterpret_cast<ScalarsOnly*>(storage.data());
        gr::device::relocateBlockToDevice(mirror, block);

        expect(eq(mirror->processOne(2.f), 8.f)) << "the mirror computes with the settings it was given";
    };

    "a relocated block reads pmr array settings through the mirrored bytes"_test = [] {
        PmrTaps block;
        block.taps = std::pmr::vector<float>{3.f, 1.f};

        alignas(PmrTaps) std::array<std::byte, sizeof(PmrTaps)> storage{};
        auto*                                                   mirror = reinterpret_cast<PmrTaps*>(storage.data());
        gr::device::relocateBlockToDevice(mirror, block);

        expect(eq(mirror->processOne(2.f), 6.f)) << "the mirror indexes the same storage the host owns";
        expect(mirror->taps.data() == block.taps.data()) << "the mirror shares storage; it does not copy it";
    };

    "the offending member is named, so a silent CPU fallback can be explained"_test = [] {
        expect(eq(gr::device::firstNonRelocatableMember<RawVectorTaps>(), std::string_view{"taps"}));
        expect(gr::device::firstNonRelocatableMember<PmrTaps>().empty());
    };
    "a mutable member written by a const processOne is caught"_test = [] {
        MutableCounter mutating;
        expect(gr::device::blockMutatesItsOwnState(mutating, 1.f)) << "the canary sees the mutable write no trait can";

        ScalarsOnly pure;
        expect(!gr::device::blockMutatesItsOwnState(pure, 1.f)) << "and does not accuse a well-behaved block";
    };

    "the canary probes a bit-copy, so the live block keeps its state"_test = [] {
        MutableCounter mutating;
        const auto     callsBefore = mutating._calls;
        expect(gr::device::blockMutatesItsOwnState(mutating, 1.f));
        expect(eq(mutating._calls, callsBefore)) << "the probe runs on a copy: cheap enough to keep in release builds";
    };

    "a mutable scalar is caught even when a pmr member sits beside it"_test = [] {
        MutableCounterWithTaps mixed;
        mixed.taps = std::pmr::vector<float>{2.f, 3.f};
        expect(gr::device::blockMutatesItsOwnState(mixed, 1.f)) << "the pmr sibling must not excuse the block from the canary -- the scalar's write is still discarded";
    };

    "probing a block with pmr storage leaves that storage exactly as it was"_test = [] {
        MutableHistory stateful;
        stateful.history = std::pmr::vector<float>{7.f, 8.f, 9.f};

        expect(!gr::device::blockMutatesItsOwnState(stateful, 1.f)) << "writing through shared pmr storage loses nothing, so it is not a mutation to report";
        expect(eq(stateful.history.size(), 3UZ));
        expect(eq(stateful.history[0], 7.f)) << "the probe put the shared bytes back";
    };

    "a two-input block is probed too, rather than excused for its arity"_test = [] {
        MutableTwoInput twoIn;
        expect(gr::device::blockMutatesItsOwnState(twoIn, 1.f, 2.f)) << "the canary must synthesise a sample per port and still see the mutable write";
        expect(eq(twoIn._calls, 0UZ)) << "and it ran on a bit-copy, so the live block never counted the probe";
    };

    "a pmr member reassigned behind the settings system is reported as a stale mirror"_test = [] {
        PmrTaps block;
        block.taps = std::pmr::vector<float>{3.f, 1.f};

        alignas(PmrTaps) std::array<std::byte, sizeof(PmrTaps)> storage{};
        auto*                                                   mirror = reinterpret_cast<PmrTaps*>(storage.data());
        gr::device::relocateBlockToDevice(mirror, block);
        expect(gr::device::firstStaleMirrorMember(block, *mirror).empty()) << "a freshly built mirror agrees with the block";

        block.taps = std::pmr::vector<float>{2.f, 0.f}; // no settings epoch bump: the mirror keeps the old storage
        expect(eq(gr::device::firstStaleMirrorMember(block, *mirror), std::string_view{"taps"})) << "and the offending member is named";
    };
};

int main() { /* tests are statically executed */ }
