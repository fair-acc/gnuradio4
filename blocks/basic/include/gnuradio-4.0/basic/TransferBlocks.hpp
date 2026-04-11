#ifndef GNURADIO_TRANSFER_BLOCKS_HPP
#define GNURADIO_TRANSFER_BLOCKS_HPP

#include <gnuradio-4.0/Block.hpp>
#include <gnuradio-4.0/BlockRegistry.hpp>

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

    Annotated<gr::Size_t, "min chunk size", Limits<8UZ, 4'294'967'296UZ>> min_chunk_size = 256UZ;
    Annotated<gr::Size_t, "max chunk size", Limits<8UZ, 4'294'967'296UZ>> max_chunk_size = 65536UZ;
    Annotated<gr::Size_t, "chunk size", Limits<8UZ, 4'294'967'296UZ>>     chunk_size     = 4096UZ;

    GR_MAKE_REFLECTABLE(HostToDevice, in, out, min_chunk_size, max_chunk_size, chunk_size);

    gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
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

    gr::work::Status processBulk(InputSpanLike auto& input, OutputSpanLike auto& output) {
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
};

} // namespace gr::basic

#endif // GNURADIO_TRANSFER_BLOCKS_HPP
