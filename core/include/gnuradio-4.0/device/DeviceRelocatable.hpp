#ifndef GNURADIO_DEVICE_RELOCATABLE_HPP
#define GNURADIO_DEVICE_RELOCATABLE_HPP

#ifdef _GLIBCXX_DEBUG
#error "_GLIBCXX_DEBUG changes container layout (safe-iterator bookkeeping), so a block bit-copied into device memory is structurally invalid there. Build device backends without it."
#endif

#include <array>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <memory_resource>
#include <new>
#include <optional>
#include <span>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <gnuradio-4.0/MemoryAllocators.hpp>
#include <gnuradio-4.0/Port.hpp>
#include <gnuradio-4.0/annotated.hpp>
#include <gnuradio-4.0/meta/reflection.hpp>

namespace gr::device {

/**
 * @brief Which blocks may be bit-copied into device memory and read by a kernel body.
 *
 * Only the block's *own* members are checked: `gr::Block`'s base carries a `std::string`, a `property_map` and
 * the ports, so `std::is_trivially_copyable_v<TBlock>` never holds for a real block. A pmr container qualifies
 * because the framework re-seats its storage onto the device resource first; a `basic_string` never does, since
 * the small-string optimisation keeps short data inside the object.
 */

template<typename M>
concept DeviceSeatableContainer = PmrMigratable<M>                          //
                                  && !requires { typename M::traits_type; } // basic_string: SSO data lives inside the object
                                  && requires(const M& m) {
                                         { m.data() } -> std::convertible_to<const void*>;
                                         { m.size() } -> std::convertible_to<std::size_t>;
                                     } && std::is_trivially_copyable_v<std::remove_cvref_t<decltype(*std::declval<const M&>().data())>>;

/// non-owning views are trivially copyable but their bytes are a host address the kernel cannot follow. A denylist,
/// not a `data()`/`size()` shape test, which would also catch `std::array` — whose storage travels inside the object.
template<typename>
inline constexpr bool kIsNonOwningView = false;
template<typename T, std::size_t kExtent>
inline constexpr bool kIsNonOwningView<std::span<T, kExtent>> = true;
template<typename TChar, typename TTraits>
inline constexpr bool kIsNonOwningView<std::basic_string_view<TChar, TTraits>> = true;

namespace detail {

/// a kernel body never touches ports: their data arrives as `processOne` arguments
template<typename TBlock, std::size_t kIdx>
consteval bool isDeviceRelocatableMember() {
    using Raw = refl::data_member_type<TBlock, kIdx>;
    using M   = unwrap_if_wrapped_t<Raw>;

    if constexpr (PortLike<Raw> || PortLike<M>) {
        return true;
    } else if constexpr (std::is_pointer_v<M> || kIsNonOwningView<M>) {
        return false; // a host address, meaningless on the device
    } else if constexpr (std::is_trivially_copyable_v<M>) {
        return true;
    } else {
        return DeviceSeatableContainer<M>;
    }
}

template<typename TBlock>
consteval std::size_t firstUserMember() {
    return static_cast<std::size_t>(refl::data_member_count<refl::base_type<TBlock>>);
}

template<typename TBlock, std::size_t... kIdx>
consteval bool allUserMembersRelocatable(std::index_sequence<kIdx...>) {
    return ((kIdx < firstUserMember<TBlock>() || isDeviceRelocatableMember<TBlock, kIdx>()) && ...);
}

} // namespace detail

template<typename TBlock>
concept DeviceRelocatable = std::is_trivially_copyable_v<TBlock> // plain functors, e.g. device test helpers
                            || (refl::reflectable<TBlock>        //
                                   && detail::firstUserMember<TBlock>() <= static_cast<std::size_t>(refl::data_member_count<TBlock>) && detail::allUserMembersRelocatable<TBlock>(std::make_index_sequence<static_cast<std::size_t>(refl::data_member_count<TBlock>)>{}));

/**
 * @brief May the mutation canary run this block on a bit-copy?
 *
 * Every relocatable block may. The copy aliases whatever the original points at, so a probed call writes through a
 * pmr member into the live block -- `mutatesItsOwnState` captures those bytes and puts them back, which is what
 * keeps the probe an observation. A member seated on a device resource is the one case it declines, because the
 * host may not read the bytes it would have to restore.
 */
/**
 * @brief A block's own statement that everything the device must see is listed in `GR_MAKE_REFLECTABLE`.
 *
 * The block is bit-copied to the device whole, but only *reflected* members are checked for relocatability and
 * only reflected pmr members have their storage re-seated onto device memory. A member left out of the macro is
 * therefore copied as raw bytes and, if it owns host storage, followed there by the kernel. C++23 cannot
 * enumerate the members the macro omitted, so the block declares the invariant instead and the framework says so
 * when it is missing.
 */
template<typename TBlock>
concept DeclaresDeviceStateReflected = requires { typename TBlock::DeviceStateIsReflected; };

template<typename TBlock>
concept DeviceProbeSafe = DeviceRelocatable<TBlock>;

template<typename TBlock>
[[nodiscard]] consteval std::string_view firstNonRelocatableMember() {
    std::string_view offender{};
    if constexpr (refl::reflectable<TBlock>) {
        [&]<std::size_t... kIdx>(std::index_sequence<kIdx...>) {
            ((offender.empty() && kIdx >= detail::firstUserMember<TBlock>() && !detail::isDeviceRelocatableMember<TBlock, kIdx>() //
                     ? (offender = refl::data_member_name<TBlock, kIdx>.view(), 0)
                     : 0),
                ...);
        }(std::make_index_sequence<static_cast<std::size_t>(refl::data_member_count<TBlock>)>{});
    }
    return offender;
}

/**
 * @brief Bit-copy a block into device memory so a kernel body can read its settings.
 *
 * Blocks are move-only and their base owns host state, so neither copy-construction nor `std::copy` is available,
 * and a deep copy would be wrong: the kernel needs the bytes with pmr storage already re-seated onto the device.
 */
template<typename TBlock>
requires DeviceRelocatable<TBlock>
void relocateBlockToDevice(TBlock* deviceCopy, const TBlock& block) noexcept {
    std::memcpy(static_cast<void*>(deviceCopy), static_cast<const void*>(&block), sizeof(TBlock));
}

/**
 * @brief Copy the block's own trivially-copyable members back from a kernel that ran as a single work item.
 *
 * The forward relocation is deliberately one-way, which is right for the per-element tiers: N work items share one
 * mirror, so a write is a race and discarding it is the honest outcome. The framework *bulk* tier launches exactly
 * one work item, so a body that keeps state (an IIR's memory, a crossing counter) cannot race, and its writes are
 * merely lost. Copying them back costs `sizeof(TBlock)` once per dispatch — not per sample.
 *
 * Only the block's own trivially-copyable members move: a pmr member shares its storage with the mirror, so the
 * host already sees those writes, and the base's host-owned state must never be overwritten from device memory.
 */
template<typename TBlock>
requires DeviceRelocatable<TBlock>
void copyBackUserState(TBlock& block, const TBlock& mirror) noexcept {
    if constexpr (refl::reflectable<TBlock>) {
        refl::for_each_data_member_index<TBlock>([&](auto kIdx) {
            if constexpr (kIdx >= detail::firstUserMember<TBlock>()) {
                using F = std::remove_cvref_t<decltype(refl::data_member<kIdx>(block))>;
                if constexpr (!PortLike<F> && std::is_trivially_copyable_v<F> && !PmrMigratable<unwrap_if_wrapped_t<F>>) {
                    refl::data_member<kIdx>(block) = refl::data_member<kIdx>(mirror);
                }
            }
        });
    }
}

/// name of the first pmr member the mirror no longer agrees with — empty while coherent. Reassigning such a member
/// directly instead of through the settings system frees its device storage without bumping the settings epoch.
/// Reads the mirror, which is host-visible shared USM; a device-only mirror would move this comparison on-device.
template<typename TBlock>
[[nodiscard]] std::string_view firstStaleMirrorMember(const TBlock& block, const TBlock& mirror) {
    std::string_view stale{};
    if constexpr (refl::reflectable<TBlock>) {
        const auto sameStorage = [](const auto& a, const auto& b) { return a.data() == b.data() && a.size() == b.size(); };
        refl::for_each_data_member_index<TBlock>([&](auto kIdx) {
            if constexpr (kIdx >= detail::firstUserMember<TBlock>()) {
                using F         = std::remove_cvref_t<decltype(refl::data_member<kIdx>(block))>;
                using Unwrapped = unwrap_if_wrapped_t<F>;
                if constexpr (DeviceSeatableContainer<F>) {
                    if (stale.empty() && !sameStorage(refl::data_member<kIdx>(block), refl::data_member<kIdx>(mirror))) {
                        stale = refl::data_member_name<TBlock, kIdx>.view();
                    }
                } else if constexpr (is_annotated<F>() && DeviceSeatableContainer<Unwrapped>) {
                    if (stale.empty() && !sameStorage(refl::data_member<kIdx>(block).value, refl::data_member<kIdx>(mirror).value)) {
                        stale = refl::data_member_name<TBlock, kIdx>.view();
                    }
                }
            }
        });
    }
    return stale;
}

/// Does one invocation write the block's own bytes? A `const` processOne/processBulk may still write `mutable`
/// members, which no trait sees before C++26 and which the device would silently discard into the mirror.
/// one entry per pmr member the bit-copy shares with the live block: where those bytes are, and what they were
using SharedStorage = std::vector<std::pair<std::byte*, std::vector<std::byte>>>;

/// `std::nullopt` when a member is seated on a device resource, which the host must not read
template<typename TBlock>
[[nodiscard]] std::optional<SharedStorage> captureSharedStorage(const TBlock& block) {
    SharedStorage shared;
    bool          readable = true;
    if constexpr (refl::reflectable<TBlock>) {
        refl::for_each_data_member_index<TBlock>([&](auto kIdx) {
            if constexpr (kIdx >= detail::firstUserMember<TBlock>()) {
                using F = std::remove_cvref_t<decltype(refl::data_member<kIdx>(block))>;
                using M = unwrap_if_wrapped_t<F>;
                if constexpr (!PortLike<F> && DeviceSeatableContainer<M>) {
                    const M& member = [](const auto& raw) -> const M& {
                        if constexpr (is_annotated<std::remove_cvref_t<decltype(raw)>>()) {
                            return raw.value;
                        } else {
                            return raw;
                        }
                    }(refl::data_member<kIdx>(block));
                    if constexpr (requires { member.get_allocator().resource(); }) {
                        if (gr::isDeviceOnly(member.get_allocator().resource())) {
                            readable = false;
                            return;
                        }
                    } else {
                        readable = false; // no way to ask where it lives, so do not guess that it is host memory
                        return;
                    }
                    const std::size_t bytes = member.size() * sizeof(*member.data());
                    std::byte*        start = const_cast<std::byte*>(reinterpret_cast<const std::byte*>(member.data()));
                    shared.emplace_back(start, std::vector<std::byte>(start, start + bytes));
                }
            }
        });
    }
    if (!readable) {
        return std::nullopt;
    }
    return shared;
}

template<typename TBlock>
void restoreSharedStorage(const TBlock&, const SharedStorage& shared) noexcept {
    for (const auto& [start, bytes] : shared) {
        std::memcpy(start, bytes.data(), bytes.size());
    }
}

/// `invokeOnCopy` runs on a never-destructed bit-copy, so the live block is untouched; `DeviceProbeSafe` gates it.
template<typename TBlock, typename TInvoke>
[[nodiscard]] bool mutatesItsOwnState(const TBlock& block, TInvoke&& invokeOnCopy)
requires DeviceProbeSafe<TBlock>
{
    std::optional<SharedStorage> shared = captureSharedStorage(block);
    if (!shared) {
        return false; // a member is seated on a device resource: the host cannot read the bytes it would restore
    }

    alignas(TBlock) std::array<std::byte, sizeof(TBlock)> probe{};
    alignas(TBlock) std::array<std::byte, sizeof(TBlock)> before{};
    std::memcpy(probe.data(), static_cast<const void*>(&block), sizeof(TBlock));
    before = probe;
    std::forward<TInvoke>(invokeOnCopy)(*std::launder(reinterpret_cast<TBlock*>(probe.data())));
    restoreSharedStorage(block, *shared);
    return std::memcmp(before.data(), probe.data(), sizeof(TBlock)) != 0;
}

template<typename TBlock, typename... TSamples>
[[nodiscard]] bool blockMutatesItsOwnState(const TBlock& block, const TSamples&... samples)
requires DeviceProbeSafe<TBlock> && requires(const TBlock& b) { b.processOne(samples...); }
{
    return mutatesItsOwnState(block, [&samples...](TBlock& copy) { std::ignore = copy.processOne(samples...); });
}

} // namespace gr::device

#endif // GNURADIO_DEVICE_RELOCATABLE_HPP
