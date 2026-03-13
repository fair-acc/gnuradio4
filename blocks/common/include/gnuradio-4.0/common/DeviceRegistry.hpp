#ifndef GNURADIO_DEVICE_REGISTRY_HPP
#define GNURADIO_DEVICE_REGISTRY_HPP

#include <expected>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace gr::blocks::common {

/**
 * @brief Abstract interface for browser-mediated device access (WASM only).
 *
 * Derived types encapsulate one device class (e.g. WebSerial, WebUSB) and manage
 * the permission → connect → disconnect lifecycle required by browser security policies.
 * Native builds bypass this entirely — blocks call OS APIs directly.
 */
struct DeviceBase {
    virtual ~DeviceBase() = default;

    [[nodiscard]] virtual std::string_view id() const noexcept          = 0;
    [[nodiscard]] virtual std::string_view displayName() const noexcept = 0;

    virtual void init() = 0;

    [[nodiscard]] virtual bool isApiAvailable() const noexcept = 0;
    [[nodiscard]] virtual int  grantedCount() const noexcept   = 0;
    [[nodiscard]] bool         isGranted() const noexcept { return grantedCount() > 0; }

    virtual void requestPermission() = 0;

    [[nodiscard]] virtual std::expected<int, std::string> connect(int portIndex = 0, int param = 0) = 0;
    virtual void                                          disconnect(int handle)                    = 0;

    [[nodiscard]] virtual std::string lastError() const = 0;
};

/**
 * @brief Singleton registry for browser-mediated devices.
 *
 * Device implementations self-register via AutoRegister at static-init time.
 * Blocks look up devices by string id and dynamic_cast to the concrete type.
 */
struct DeviceRegistry {
    static DeviceRegistry& instance() {
        static DeviceRegistry registry;
        return registry;
    }

    void add(std::shared_ptr<DeviceBase> device) {
        for (const auto& d : _devices) {
            if (d->id() == device->id()) {
                return;
            }
        }
        _devices.push_back(std::move(device));
    }

    [[nodiscard]] DeviceBase* find(std::string_view id) {
        for (auto& d : _devices) {
            if (d->id() == id) {
                return d.get();
            }
        }
        return nullptr;
    }

    template<typename T>
    [[nodiscard]] T* findAs(std::string_view id) {
        return dynamic_cast<T*>(find(id));
    }

    void init() {
        for (auto& d : _devices) {
            d->init();
        }
    }

    void requestAllPermissions() {
        for (auto& d : _devices) {
            if (d->isApiAvailable() && !d->isGranted()) {
                d->requestPermission();
            }
        }
    }

    [[nodiscard]] bool isGranted(std::string_view id) {
        auto* d = find(id);
        return d && d->isGranted();
    }

    [[nodiscard]] int grantedCount(std::string_view id) {
        auto* d = find(id);
        return d ? d->grantedCount() : 0;
    }

    [[nodiscard]] std::size_t deviceCount() const { return _devices.size(); }

private:
    std::vector<std::shared_ptr<DeviceBase>> _devices;
};

struct AutoRegister {
    explicit AutoRegister(std::shared_ptr<DeviceBase> device) { DeviceRegistry::instance().add(std::move(device)); }
};

} // namespace gr::blocks::common

#endif // GNURADIO_DEVICE_REGISTRY_HPP
