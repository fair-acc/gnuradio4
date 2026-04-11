#ifndef GNURADIO_DEVICE_BLOCK_STATE_HPP
#define GNURADIO_DEVICE_BLOCK_STATE_HPP

#include <cassert>
#include <cstring>
#include <type_traits>

#include <gnuradio-4.0/device/DeviceContext.hpp>
#include <gnuradio-4.0/reflection.hpp>

namespace gr::device {

/**
 * @brief Mirrors a block's reflected scalar settings into a flat byte buffer for device transfer.
 *
 * Uses GR4's compile-time reflection (`GR_MAKE_REFLECTABLE`) to extract arithmetic fields
 * from a block and pack them sequentially. Non-scalar fields (ports, strings, containers)
 * are skipped. Device kernels access the mirrored state instead of the host block object.
 *
 * Usage:
 * @code
 * DeviceBlockState<MultiplyConst<float>> state;
 * state.syncFrom(block);
 * ctx.copyHostToDevice(&state.hostCopy, deviceState, 1);
 * @endcode
 */
template<typename TBlock>
struct DeviceBlockState {
    static constexpr bool kReflectable = refl::reflectable<TBlock>;

    struct Data {
        alignas(16) std::byte storage[256]{};
        std::size_t usedBytes = 0;
    };

    Data hostCopy;

    template<typename F>
    static void forEachScalarField(const TBlock& block, F&& f) {
        if constexpr (kReflectable) {
            refl::for_each_data_member_index<TBlock>([&](auto kIdx) {
                using MemberType = std::remove_cvref_t<refl::data_member_type<TBlock, kIdx>>;
                if constexpr (std::is_trivially_copyable_v<MemberType> && std::is_arithmetic_v<MemberType>) {
                    const auto& val = refl::data_member_ref<kIdx>(block);
                    f(kIdx, val);
                }
            });
        }
    }

    void syncFrom(const TBlock& block) {
        hostCopy.usedBytes = 0;
        forEachScalarField(block, [this](auto, const auto& val) {
            using T = std::remove_cvref_t<decltype(val)>;
            assert(hostCopy.usedBytes + sizeof(T) <= sizeof(hostCopy.storage) && "DeviceBlockState: block has more scalar settings than the 256-byte buffer can hold");
            if (hostCopy.usedBytes + sizeof(T) <= sizeof(hostCopy.storage)) {
                std::memcpy(hostCopy.storage + hostCopy.usedBytes, &val, sizeof(T));
                hostCopy.usedBytes += sizeof(T);
            }
        });
    }
};

} // namespace gr::device

#endif // GNURADIO_DEVICE_BLOCK_STATE_HPP
