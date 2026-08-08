#ifndef GNURADIO_TRANSFER_BLOCKS_HPP
#define GNURADIO_TRANSFER_BLOCKS_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>
#include <gnuradio-4.0/device/DeviceContextSycl.hpp>

namespace gr::basic {

GR_REGISTER_BLOCK("gr::basic::HostToDevice", gr::basic::HostToDevice, [T], [ float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct HostToDevice : gr::Block<HostToDevice<T>> {
    using Description = Doc<R""(transfers samples from host to device memory.

Place between CPU and device blocks to control the batch/DMA transfer size.
Larger chunk sizes improve throughput (DMA amortisation); smaller sizes reduce latency.
Always paired with a downstream DeviceToHost block.)"">;

    PortIn<T>  in;
    PortOut<T> out;

    Annotated<gr::Size_t, "chunk size", Limits<8UZ, 4'294'967'296UZ>>     chunk_size     = 4096UZ;

    GR_MAKE_REFLECTABLE(HostToDevice, in, out, chunk_size);

    // constrained to host spans: once the block offers a device hatch the framework also probes `processBulk` as a
    // candidate kernel body, and this host copy is neither valid nor wanted there
    template<typename TIn, typename TOut>
    requires requires(TIn& hostSpan) { hostSpan.first(0UZ); }
    gr::work::Status processBulk(TIn& input, TOut& output) {
        const auto available = std::min(input.size(), output.size());
        const auto n         = std::min(available, static_cast<std::size_t>(chunk_size));
        if (n == 0) {
            std::ignore = input.consume(0);
            output.publish(0);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        std::ranges::copy(input.first(n), output.begin());
        std::ignore = input.consume(n);
        output.publish(n);
        return work::Status::OK;
    }

    /// on a device domain this is a real bulk transfer rather than a host element-wise copy: one queue copy moves the
    /// whole chunk, which is also the only way the destination may be memory the host cannot address at all
    gr::work::Status processBulk_sycl(gr::device::SyclQueue& queue, InputSpanLike auto& input, OutputSpanLike auto& output) {
        const auto available = std::min(input.size(), output.size());
        const auto n         = std::min(available, static_cast<std::size_t>(chunk_size));
        if (n == 0) {
            std::ignore = input.consume(0);
            output.publish(0);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }
        queue.memcpy(output.data(), input.data(), n * sizeof(T)).wait(); // the consumer must see a complete chunk
        std::ignore = input.consume(n);
        output.publish(n);
        return work::Status::OK;
    }
};

GR_REGISTER_BLOCK("gr::basic::DeviceToHost", gr::basic::DeviceToHost, [T], [ float, double, std::complex<float>, std::complex<double> ])

template<typename T>
struct DeviceToHost : gr::Block<DeviceToHost<T>> {
    using Description = Doc<R""(transfers samples from device memory back to host.

Place between device and CPU blocks to mark the device-to-host boundary.
Copies all available samples; the upstream HostToDevice controls the batch size.
Always paired with an upstream HostToDevice block.)"">;

    PortIn<T>  in;
    PortOut<T> out;

    GR_MAKE_REFLECTABLE(DeviceToHost, in, out);

    // constrained to host spans: once the block offers a device hatch the framework also probes `processBulk` as a
    // candidate kernel body, and this host copy is neither valid nor wanted there
    template<typename TIn, typename TOut>
    requires requires(TIn& hostSpan) { hostSpan.first(0UZ); }
    gr::work::Status processBulk(TIn& input, TOut& output) {
        const auto n = std::min(input.size(), output.size());
        if (n == 0) {
            std::ignore = input.consume(0);
            output.publish(0);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }

        std::ranges::copy(input.first(n), output.begin());
        std::ignore = input.consume(n);
        output.publish(n);
        return work::Status::OK;
    }

    /// the device-side half of the pair: one bulk queue copy back, so the source may be memory the host cannot address
    gr::work::Status processBulk_sycl(gr::device::SyclQueue& queue, InputSpanLike auto& input, OutputSpanLike auto& output) {
        const auto n = std::min(input.size(), output.size());
        if (n == 0) {
            std::ignore = input.consume(0);
            output.publish(0);
            return work::Status::INSUFFICIENT_INPUT_ITEMS;
        }
        queue.memcpy(output.data(), input.data(), n * sizeof(T)).wait(); // the consumer must see a complete chunk
        std::ignore = input.consume(n);
        output.publish(n);
        return work::Status::OK;
    }
};

} // namespace gr::basic

#endif // GNURADIO_TRANSFER_BLOCKS_HPP
