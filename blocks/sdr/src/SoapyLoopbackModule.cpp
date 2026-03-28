#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-W#warnings"
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcpp"
#endif
#include <SoapySDR/Registry.hpp>
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#include <gnuradio-4.0/sdr/LoopbackDevice.hpp>

namespace {

using gr::blocks::sdr::loopback::DeviceRegistry;
using gr::blocks::sdr::loopback::LoopbackDevice;

SoapySDR::KwargsList findLoopback(const SoapySDR::Kwargs& args) {
    if (!args.empty() && !DeviceRegistry::isLoopbackDriver(args)) {
        return {};
    }

    SoapySDR::KwargsList results;

    auto  lock      = std::lock_guard(DeviceRegistry::mutex());
    auto& instances = DeviceRegistry::instances();
    for (auto it = instances.begin(); it != instances.end();) {
        if (it->second.expired()) {
            it = instances.erase(it);
            continue;
        }
        SoapySDR::Kwargs info;
        info["driver"]      = "loopback#" + std::to_string(it->first);
        info["label"]       = "GR4 Loopback Device #" + std::to_string(it->first);
        info["product"]     = "loopback";
        info["instance_id"] = std::to_string(it->first);
        results.push_back(info);
        ++it;
    }

    if (results.empty()) {
        SoapySDR::Kwargs info;
        info["driver"]  = "loopback";
        info["label"]   = "GR4 Loopback Device";
        info["product"] = "loopback";
        results.push_back(info);
    }

    return results;
}

// SoapySDR owns the returned pointer (calls delete on unmake).
// Each make() creates an independent device that supports both TX and RX
// streams on the same handle — this is the standard SoapySDR pattern.
//
// For shared TX/RX across separate objects (e.g. SoapySource + SoapySink
// as independent GR4 blocks), use DeviceRegistry::findOrCreate() which
// returns a shared_ptr to the same device instance.
SoapySDR::Device* makeLoopback(const SoapySDR::Kwargs& args) { return new LoopbackDevice(args); }

static SoapySDR::Registry registerLoopback("loopback", &findLoopback, &makeLoopback, SOAPY_SDR_ABI_VERSION);

} // namespace
