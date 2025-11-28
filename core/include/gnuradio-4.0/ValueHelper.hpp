#ifndef GNURADIO_VALUEHELPER_HPP
#define GNURADIO_VALUEHELPER_HPP

namespace gr::pmt {

inline constexpr std::size_t memory_usage(const Value& value) noexcept {
    std::size_t size = sizeof(Value);

    switch (value.container_type()) {
    case Value::ContainerType::Complex: size += (value.value_type() == Value::ValueType::ComplexFloat32) ? sizeof(std::complex<float>) : sizeof(std::complex<double>); break;

    case Value::ContainerType::String: size += sizeof(std::pmr::string) + static_cast<const std::pmr::string*>(value._storage.ptr)->capacity(); break;

    case Value::ContainerType::Map: {
        const auto& map = *static_cast<const Value::Map*>(value._storage.ptr);
        size += sizeof(Value::Map) + map.bucket_count() * sizeof(void*);
        for (const auto& [k, v] : map) {
            size += k.capacity() + memory_usage(v);
        }
        break;
    }

    case Value::ContainerType::Tensor:
        // TODO: Dispatch to Tensor::memory_usage() if available
        size += sizeof(void*); // Just count the pointer for now
        break;

    default: break;
    }

    return size;
}
} // namespace gr::pmt

#endif // GNURADIO_VALUEHELPER_HPP
